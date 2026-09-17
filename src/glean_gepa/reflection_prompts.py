"""Reflection-LLM prompt scaffolding shared by every objective.

Each objective owns the module-responsibility text telling the reflector what it
may edit; the rules and templates here wrap that text with evidence in
``propose_new_texts``.
"""

from __future__ import annotations

import re

DEFAULT_MODULE_RESPONSIBILITY = "Focus only on this module's responsibilities."

# The renderer treats <<<[[name]] ... >>> as a live conditional, so reflection fences
# must not use that syntax — otherwise children echo it and the module is dropped.
MODULE_TEXT_BEGIN = "----- BEGIN CURRENT MODULE TEXT -----"
MODULE_TEXT_END = "----- END CURRENT MODULE TEXT -----"

CONDITIONAL_PRESERVE_RULE = (
    "Never remove conditionals such as <<<[[hitl_approval_instructions]]>>>; keep every "
    "<<<[[name]] ... >>> wrapper. You may modify the enclosed text as needed but be aware of the conditional."
)

CONDITIONAL_ABSENT_RULE = (
    "This module is plain text with no template markup. Do not add conditional blocks or "
    "bracketed placeholder tokens, and do not wrap your answer in delimiters of any kind: "
    "return the module text by itself."
)

_PLACEHOLDER = re.compile(r"\[\[(\w+)\]\]")
_RENDER_SLOT = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")
_VARIANT_HEADING = re.compile(r"(?i)variant\s*\d*\s*(\([^)]*\))?\s*[:.]?")


def render_slots(text: str) -> set[str]:
    """Compile-time splice points such as ``{RULES_EXT}`` that live inside module text."""
    return set(_RENDER_SLOT.findall(text))


def drops_render_slot(proposed: str, *, current: str) -> bool:
    """Whether ``proposed`` lost a slot ``current`` had, which would orphan that module."""
    return bool(render_slots(current) - render_slots(proposed))


def markup_rule_for(current: str) -> str:
    """State which template markup in ``current`` a rewrite has to carry over."""
    rules = []
    if "<<<" in current:
        rules.append(CONDITIONAL_PRESERVE_RULE)
    slots = sorted(render_slots(current))
    if slots:
        names = ", ".join(f"{{{slot}}}" for slot in slots)
        rules.append(
            f"Reproduce {names} verbatim, on its own line and in the same position. Another module is "
            "spliced there at compile time, so a rewrite that drops the slot silently discards it."
        )
    return " ".join(rules) if rules else CONDITIONAL_ABSENT_RULE


def sanitize_proposed_module(proposed: str, *, current: str, module_name: str = "") -> str:
    """Strip echoed fences, variant labels, and template markup the reflector invented."""
    text = proposed.strip()
    if text.startswith(MODULE_TEXT_BEGIN):
        text = text[len(MODULE_TEXT_BEGIN) :].strip()
    if text.endswith(MODULE_TEXT_END):
        text = text[: -len(MODULE_TEXT_END)].strip()
    lines = text.splitlines()
    while lines:
        head = lines[0].strip().strip("*#` ")
        if (module_name and head.casefold() == module_name.casefold()) or _VARIANT_HEADING.fullmatch(head):
            lines = lines[1:]
            continue
        break
    text = "\n".join(lines).strip()

    if "<<<" in current or ">>>" in current:
        base = current.count("<<<")
        while text.count("<<<") > base and re.match(r"^<<<(?!\[\[)", text) and text.endswith(">>>"):
            text = text[3:-3].strip()
        return text

    text = text.replace("<<<", " ").replace(">>>", " ")
    for name in set(_PLACEHOLDER.findall(text)):
        if f"[[{name}]]" not in current:
            text = text.replace(f"[[{name}]]", " ")
    return re.sub(r"[ \t]{2,}", " ", text).strip()


TEACHER_IS_OFFLINE_RULE = (
    "The teacher is an offline scoring reference, not something the student can see at runtime. "
    "Never tell the student to consult, mirror, match, copy, or ask the teacher, and never mention "
    "the teacher in the prompt text at all. Instead, work out WHICH behavior produced the teacher's "
    "result and state that as a standalone rule the student can follow using only the user's request "
    "and its own tool results."
)

NO_EXAMPLE_SPECIFICS_RULE = (
    "Do not name specific customers, documents, fields, headings, counts, or literal values "
    "drawn from the examples. State each rule in terms of the kind of request and the behavior "
    "it should map to, so it applies to requests that are not in the evidence."
)

FAKE_FLOW_RESPONSIBILITY = "Improve the fake coding instructions using the failed examples."


def module_char_budget(current: str, token_budget: int | None = None) -> int | None:
    """Largest rewrite size in characters, or None if unbounded. ~4 chars/token."""
    hard_cap = token_budget * 4 if token_budget else None
    if not current.strip():
        return hard_cap
    growth = max(int(len(current) * 1.1), 200)
    return min(growth, hard_cap) if hard_cap else growth


def length_rule_for(current: str, token_budget: int | None = None) -> str:
    budget = module_char_budget(current, token_budget)
    if not current.strip():
        rule = "The current module is empty; write a short new module rather than copying other sections."
        return f"{rule} It must be at most {budget} characters." if budget else rule
    return (
        f"Keep the revised module succinct: it is currently {len(current)} characters and each candidate "
        f"must be at most {budget} characters. To add guidance, cut or tighten existing text rather than "
        f"appending to it."
    )


EMPTY_DIAGNOSIS_FALLBACK = (
    "No module-specific diagnosis was returned. Propose a conservative rewrite that addresses "
    "the supplied failure evidence while preserving the current instructions."
)


def core_tool_reflection_prompt(module_name: str) -> str:
    """Reflection instructions for editing one core-tool ``schema.description``."""
    return (
        f"You are editing only the prompt-visible schema.description for the core tool `{module_name}`. "
        "The override replaces description text only — not the tool signature, parameters, or Returns. "
        "Rewrite the description so the student uses this tool as the first action when the teacher does, "
        "and does not use it first when the teacher chooses a different tool. Keep the text operational and concise. "
        f"{TEACHER_IS_OFFLINE_RULE}"
    )


def diagnosis_prompt(
    *,
    module_name: str,
    responsibility: str,
    current: str,
    failure_label: str,
    example_blocks: str,
    length_rule: str,
) -> str:
    """First-pass reflection prompt: diagnose failures and propose small patches."""
    return (
        f"You are optimizing ONLY the module {module_name}.\n"
        f"MODULE RESPONSIBILITY:\n{responsibility}\n\n"
        f"CURRENT MODULE TEXT:\n{MODULE_TEXT_BEGIN}\n{current}\n{MODULE_TEXT_END}\n\n"
        f"{failure_label}:\n{example_blocks}\n\n"
        f"Task:\n"
        f"1) Identify recurring failure modes that are plausibly caused by {module_name}.\n"
        f"2) Propose 1-2 SMALL patches (delta edits), each with:\n"
        f"   - BEFORE: quoted snippet from current module\n"
        f"   - AFTER: revised snippet\n"
        f"   - WHY: one sentence\n"
        f"3) Every supplied example is relevant evidence for {module_name}; use it to propose a variant.\n"
        f"4) Make only generalizable changes; do not overfit to individual examples. "
        f"{NO_EXAMPLE_SPECIFICS_RULE}\n"
        f"5) {markup_rule_for(current)}\n"
        f"6) {length_rule}\n"
    )


def consolidate_prompt(
    *,
    module_name: str,
    max_variants: int,
    consolidate_length: str,
    current: str,
    example_blocks: str,
    suggestions: str,
) -> str:
    """Second-pass reflection prompt: turn patch suggestions into full module rewrites."""
    return (
        f"Consolidate the following patch suggestions into up to {max_variants} candidate rewrites "
        f"of the module {module_name}. Preserve good behavior, incorporate consistent changes only, "
        f"and make only generalizable changes. {NO_EXAMPLE_SPECIFICS_RULE} "
        f"{markup_rule_for(current)} {consolidate_length}"
        f"Output each variant separated by '\n===VARIANT===\n'.\n\n"
        f"CURRENT:\n{MODULE_TEXT_BEGIN}\n{current}\n{MODULE_TEXT_END}\n\n"
        f"EVIDENCE (every example is relevant):\n{example_blocks}\n\n"
        f"SUGGESTIONS:\n{suggestions}\n"
    )
