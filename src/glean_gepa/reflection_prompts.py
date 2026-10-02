"""Reflection-LLM prompt scaffolding shared by every objective.

Each objective owns the module-responsibility text telling the reflector what it
may edit; the rules and templates here wrap that text with evidence in
``propose_new_texts``.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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
_CONDITIONAL_OPEN = re.compile(r"<<<\[\[(\w+)\]\]")
_RENDER_SLOT = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")
_VARIANT_HEADING = re.compile(r"(?i)variant\s*\d*\s*(\([^)]*\))?\s*[:.]?")


def render_slots(text: str) -> set[str]:
    """Compile-time splice points such as ``{RULES_EXT}`` that live inside module text."""
    return set(_RENDER_SLOT.findall(text))


def drops_render_slot(proposed: str, *, current: str) -> bool:
    """Whether ``proposed`` lost a slot ``current`` had, which would orphan that module."""
    return bool(render_slots(current) - render_slots(proposed))


def conditional_counts(text: str) -> dict[str, int]:
    """Count ``<<<[[name]] ... >>>`` conditional openers by name."""
    counts: dict[str, int] = {}
    for name in _CONDITIONAL_OPEN.findall(text):
        counts[name] = counts.get(name, 0) + 1
    return counts


def drops_conditional(proposed: str, *, current: str) -> bool:
    """Whether ``proposed`` lost a ``<<<[[name]]`` conditional ``current`` had, or unbalanced the fences.

    Scio picks a branch per request (for example ``has_search_tools`` vs ``no_search_tools``),
    so a rewrite that drops one name silently removes that branch's guidance. Count
    per name: a block may be split or merged, but every conditional name must survive
    and each ``<<<`` needs a matching ``>>>``.
    """
    before = conditional_counts(current)
    if not before:
        return False
    after = conditional_counts(proposed)
    if any(after.get(name, 0) == 0 for name in before):
        return True
    return proposed.count("<<<") != proposed.count(">>>")


def markup_rule_for(current: str) -> str:
    """State which template markup in ``current`` a rewrite has to carry over."""
    rules = []
    if "<<<" in current:
        rules.append(CONDITIONAL_PRESERVE_RULE)
        names = sorted(conditional_counts(current))
        if names:
            listed = ", ".join(f"<<<[[{name}]] ... >>>" for name in names)
            rules.append(
                f"This module has these conditionals: {listed}. Each name must still appear at least once, "
                "and every <<< must close with >>>. A variant that loses a conditional is discarded."
            )
    slots = sorted(render_slots(current))
    if slots:
        names = ", ".join(f"{{{slot}}}" for slot in slots)
        rules.append(
            f"Reproduce {names} verbatim, on its own line and in the same position. Another module is "
            "spliced there at compile time, so a rewrite that drops the slot silently discards it."
        )
    return " ".join(rules) if rules else CONDITIONAL_ABSENT_RULE


# The diagnosis pass asks for patches, not a rewrite, so it must not carry the "return the
# module text by itself" instruction: reflectors obey that over the patch format and skip the
# diagnosis entirely, which leaves the consolidation pass without a mechanism tally.
DIAGNOSIS_MARKUP_RULE = (
    "This module is plain text with no template markup: AFTER snippets must not add conditional "
    "blocks or bracketed placeholder tokens."
)


def diagnosis_markup_rule_for(current: str) -> str:
    """Markup constraints for the patch-proposal pass."""
    rule = markup_rule_for(current)
    return DIAGNOSIS_MARKUP_RULE if rule == CONDITIONAL_ABSENT_RULE else rule


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

GENERALITY_RULES = f"{NO_EXAMPLE_SPECIFICS_RULE} {TEACHER_IS_OFFLINE_RULE}"

# --- Module frames -----------------------------------------------------------
# The fixed editing contract for each module: what text the reflector is touching and
# the shape it must keep. Objectives add the objective-specific body, an optional
# analysis guide, and closing rules through ``compose_responsibility`` so the contract
# is stated once and every objective's responsibility reads the same way.

RULES_EXT_FRAME = (
    "You are writing at most two markdown bullets that will be appended after the existing "
    "**Rules:** list in Writing Code. Each line must start with '- '. Do not repeat those "
    "existing Rules, do not add a heading, and do not exceed two bullets."
)

EXECUTION_DISCIPLINE_FRAME = (
    "You are rewriting the bullets under '### Execution Discipline', which set how much work the "
    "assistant does before it responds. Each line must start with '- '. Do not add a heading."
)

WRITING_CODE_FRAME = (
    "You are rewriting the ## Writing Code body: SDK call patterns, ToolResult handling, "
    "the **Rules:** list, and sandbox privacy. Do not add a heading."
)


def core_tool_frame(tool_name: str) -> str:
    """Editing contract for one core tool's prompt-visible ``schema.description``."""
    return (
        f"You are editing only the prompt-visible schema.description for the core tool `{tool_name}`. "
        "The override replaces description text only — not the tool signature, parameters, or Returns."
    )


def compose_responsibility(frame: str, body: str, *, guide: str = "", closing: str = GENERALITY_RULES) -> str:
    """``frame`` + ``body`` as one paragraph, then ``guide`` and ``closing`` as separate paragraphs."""
    parts = [f"{frame} {body}".strip(), guide.strip(), closing.strip()]
    return "\n\n".join(part for part in parts if part)


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


# Headers are "AFTER:" style; the colon keeps a rule that starts with the word "After" from
# being read as the next block.
_AFTER_BLOCK = re.compile(
    r"^[^\w\n]*AFTER\b\s*:[ \t]*(.*?)(?=^[^\w\n]*(?:BEFORE|AFTER|WHY)\b\s*:|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
# An empty module has no current text to anchor a rewrite's size, so the patches anchor it:
# the rewrite may be this much longer than the AFTER snippets combined, and no shorter floor
# than this so a single terse patch can still be phrased as a full rule.
NEW_MODULE_GROWTH = 1.5
NEW_MODULE_MIN_CHARS = 200


def patch_after_snippets(patches: str) -> list[str]:
    """The AFTER text of every BEFORE / AFTER / WHY block in a diagnosis reply."""
    return [snippet.strip() for snippet in _AFTER_BLOCK.findall(patches) if snippet.strip()]


def new_module_length_rule(patches: str, hard_cap: int | None) -> tuple[str, int | None]:
    """Length rule and character budget for consolidating patches into a module that is empty.

    Without this the only bound is the module's token budget (2,000+ characters for RULES_EXT),
    and a two-patch diagnosis came back as a 23-line, seven-topic rulebook.
    """
    afters = patch_after_snippets(patches)
    if not afters:
        rule = "The current module is empty; write a short new module rather than copying other sections."
        return (f"{rule} It must be at most {hard_cap} characters." if hard_cap else rule), hard_cap
    anchor = max(NEW_MODULE_MIN_CHARS, int(NEW_MODULE_GROWTH * sum(len(after) for after in afters)))
    budget = min(anchor, hard_cap) if hard_cap else anchor
    count = len(afters)
    rule = (
        f"The current module is empty, so the PATCHES define its entire scope: write {count} rule"
        f"{'s' if count != 1 else ''}, one per AFTER snippet, with no headings, groupings or extra topics. "
        f"Each candidate must be at most {budget} characters."
    )
    return rule, budget


EMPTY_DIAGNOSIS_FALLBACK = (
    "No module-specific diagnosis was returned. Propose a conservative rewrite that addresses "
    "the supplied failure evidence while preserving the current instructions."
)


_TOOL_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tool_description_choice_prompt(
    *,
    tools: Mapping[str, str],
    example_blocks: str,
    limit: int,
    usage: Mapping[str, tuple[int, int]] | None = None,
    example_count: int | None = None,
) -> str:
    """Ask the reflector which core-tool descriptions are worth rewriting.

    ``usage`` maps each eligible tool to ``(student_examples, teacher_examples)``: how many
    supplied examples show that side invoking the tool.
    """
    catalog = "\n".join(f"- {name}" for name in tools)
    descriptions = "".join(f"### {name}\n{text}\n" for name, text in tools.items())
    usage_section = ""
    if usage:
        total = f" of {example_count}" if example_count is not None else ""
        lines = "\n".join(
            f"- {name}: student invoked it in {student}{total} examples, teacher in {teacher}{total}"
            for name, (student, teacher) in usage.items()
        )
        usage_section = (
            f"TOOL INVOCATIONS IN THE EXAMPLES:\n{lines}\n\n"
            "A description only changes behavior at the moment the student decides whether to call "
            "that tool. A tool the student rarely invokes cannot be steered by its description; a "
            "behavior that happens in the student's final message (asking in prose, offering a next "
            "step, hedging) is not governed by any tool description. Prefer tools the teacher invokes "
            "and the student skips, or that the student invokes and misuses.\n\n"
        )
    return (
        "You are choosing which core-tool schema.description overrides to rewrite.\n"
        "Do not prefer a tool just because it showed up often; the question is whether rewriting its "
        "description would change what the student did in these examples.\n\n"
        f"Choose at most {limit} tools, most useful first. If none would help, respond with NONE.\n"
        "Reply with one tool key per line and no other text.\n\n"
        f"ELIGIBLE TOOLS:\n{catalog}\n\n"
        f"{usage_section}"
        f"CURRENT DESCRIPTIONS:\n{descriptions}\n"
        f"HIGH-SIGNAL FAILURES:\n{example_blocks}"
    )


def parse_chosen_tool_keys(raw: str, eligible: Sequence[str]) -> list[str]:
    """Tool keys named in the reflector's reply, in the order it named them."""
    text = raw.strip()
    if not text or text.upper() == "NONE":
        return []
    eligible_set = set(eligible)
    chosen: list[str] = []
    for token in _TOOL_KEY.findall(text):
        if token in eligible_set and token not in chosen:
            chosen.append(token)
    return chosen


def core_tool_reflection_prompt(module_name: str) -> str:
    """Reflection instructions for editing one core-tool ``schema.description``."""
    return compose_responsibility(
        core_tool_frame(module_name),
        "Rewrite the description so the student uses this tool as the first action when the teacher does, "
        "and does not use it first when the teacher chooses a different tool. Keep the text operational and concise.",
        closing=TEACHER_IS_OFFLINE_RULE,
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
        f"5) {diagnosis_markup_rule_for(current)}\n"
        f"6) {length_rule}\n\n"
        f"Output format (do NOT return a full rewrite of the module in this step):\n"
        f"DIAGNOSIS:\n"
        f"<failure modes, with a count of how many supplied examples show each one, and which "
        f"examples show the student already getting it right>\n"
        f"PATCHES:\n"
        f"<the BEFORE / AFTER / WHY blocks>\n"
    )


_DIAGNOSIS_HEADER = re.compile(r"^[^\w\n]*DIAGNOSIS\b[^\w\n]*", re.IGNORECASE | re.MULTILINE)
_PATCHES_HEADER = re.compile(r"^[^\w\n]*PATCHES\b[^\w\n]*$", re.IGNORECASE | re.MULTILINE)
_PATCH_MARKERS = re.compile(r"\bBEFORE\b.*\bAFTER\b", re.IGNORECASE | re.DOTALL)
# Below this the reply is too short to be a module rewrite whatever it resembles.
_MIN_REWRITE_CHARS = 400
# A reply this similar to the current module is the module back, not commentary on it.
_REWRITE_SIMILARITY = 0.5


@dataclass(frozen=True)
class DiagnosisReply:
    """The first reflection pass, split into the tally and the patch list."""

    diagnosis: str
    patches: str
    is_module_rewrite: bool

    @property
    def suggestions(self) -> str:
        return "\n\n".join(part for part in (self.diagnosis, self.patches) if part)


def parse_diagnosis_reply(raw: str, *, current: str) -> DiagnosisReply:
    """Split a diagnosis-pass reply into DIAGNOSIS and PATCHES, flagging bare module rewrites.

    Reflectors sometimes answer the diagnosis prompt with the rewritten module and nothing
    else. That reply carries no failure-mode tally, so the consolidation pass has nothing to
    hold its edits to; the caller should re-ask rather than consolidate it.
    """
    text = raw.strip()
    if not text:
        return DiagnosisReply("", "", False)
    if _DIAGNOSIS_HEADER.search(text):
        head, patches = _split_patches(text)
        diagnosis = _DIAGNOSIS_HEADER.sub("", head, count=1).strip()
        return DiagnosisReply(diagnosis, patches.strip(), False)
    if _PATCH_MARKERS.search(text):
        return DiagnosisReply("", text, False)
    if len(text) >= _MIN_REWRITE_CHARS and current.strip():
        ratio = difflib.SequenceMatcher(None, text, current, autojunk=False).ratio()
        if ratio >= _REWRITE_SIMILARITY:
            return DiagnosisReply("", "", True)
    return DiagnosisReply(text, "", False)


def _split_patches(text: str) -> tuple[str, str]:
    match = _PATCHES_HEADER.search(text)
    if match is None:
        return text, ""
    return text[: match.start()], text[match.end() :]


def diagnosis_retry_preface(module_name: str) -> str:
    """Prepended to the diagnosis prompt when the first reply was a module rewrite."""
    return (
        f"IMPORTANT: a previous attempt at this task replied with a full rewrite of {module_name} "
        "instead of the requested sections. Do not return module text. Your reply must start with "
        "the line 'DIAGNOSIS:' followed by the failure-mode tally, then a line 'PATCHES:' followed "
        "by BEFORE / AFTER / WHY blocks. Nothing else.\n\n"
    )


def consolidate_prompt(
    *,
    module_name: str,
    max_variants: int,
    consolidate_length: str,
    current: str,
    example_blocks: str,
    suggestions: str,
    diagnosis: str = "",
) -> str:
    """Second-pass reflection prompt: turn patch suggestions into full module rewrites.

    ``diagnosis`` is the failure-mode tally from the first pass. When present, every edit in a
    variant has to trace back to a tallied failure mode, which is what makes the first pass's
    evidence thresholds (such as "at least three LOSS examples") bind on the rewrite.
    """
    diagnosis_section = ""
    if diagnosis.strip():
        diagnosis_section = (
            f"DIAGNOSIS (failure-mode tally from the first pass):\n{diagnosis.strip()}\n\n"
            "Every rule a variant adds, removes or changes must address a failure mode listed in the "
            "DIAGNOSIS, and must respect the example counts given there: leave out any change whose "
            "failure mode does not meet the evidence threshold stated in the module responsibility, "
            "and do not add rules about behavior the DIAGNOSIS says the student already gets right.\n\n"
        )
    return (
        f"Consolidate the following patch suggestions into up to {max_variants} candidate rewrites "
        f"of the module {module_name}. Preserve good behavior, incorporate consistent changes only, "
        f"and make only generalizable changes. {NO_EXAMPLE_SPECIFICS_RULE} "
        f"{markup_rule_for(current)} {consolidate_length}\n"
        f"{PATCHES_ARE_THE_WHITELIST_RULE}\n"
        f"Output each variant separated by '\n===VARIANT===\n'.\n\n"
        f"CURRENT:\n{MODULE_TEXT_BEGIN}\n{current}\n{MODULE_TEXT_END}\n\n"
        f"EVIDENCE (every example is relevant):\n{example_blocks}\n\n"
        f"{diagnosis_section}"
        f"SUGGESTIONS:\n{suggestions}\n"
    )


PATCHES_ARE_THE_WHITELIST_RULE = (
    "The SUGGESTIONS are the complete list of allowed changes: each variant applies those patches and "
    "nothing else. Do not add rules, headings, sections or topics that no patch introduces, even if an "
    "individual example in the EVIDENCE seems to call for one; a behavior seen in one or two examples is "
    "not a rule. Variants differ in how the same patches are worded, tightened and placed, not in scope."
)
