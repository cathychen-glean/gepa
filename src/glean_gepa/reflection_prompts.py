"""Reflection-LLM prompts for Glean GEPA.

Module-responsibility strings tell the reflector what to edit. Diagnosis and
consolidate templates wrap those strings with evidence in ``propose_new_texts``.
"""

from __future__ import annotations

from glean_gepa.prompt_constants import CORE_TOOL_KEYS, FULL_PROMPT_KEY, RULES_EXT_KEY, WRITING_CODE_KEY

DEFAULT_MODULE_RESPONSIBILITY = "Focus only on this module's responsibilities."

FULL_PROMPT_TEACHER_STUDENT_RESPONSIBILITY = (
    "You are editing the ENTIRE student system prompt as a single string. Any section "
    "may change — routing, execution discipline, coding instructions, tool surface, "
    "or response guidelines — if it improves first-tool alignment with the teacher. "
    "Preserve [[placeholder]] tokens. Propose a complete updated prompt with minimal deltas."
)

RULES_EXT_RESPONSIBILITY = (
    "You are writing at most two markdown bullets that will be appended after the existing "
    "**Rules:** list in Writing Code. Each line must start with '- '. Do not repeat those "
    "existing Rules, do not add a heading, and do not exceed two bullets. Target first-tool "
    "mismatches whose tools are not core tools (for example Write vs (none)). Keep each "
    "bullet operational and concise."
)

WRITING_CODE_SINGLE_MODEL_RESPONSIBILITY = (
    "Focus ONLY on coding instructions that affect shell tool reliability: SDK call patterns, "
    "ToolResult handling, parallelism via asyncio.gather, sandbox rules, and when to print vs extract. "
    "Use shell error examples as evidence. Propose minimal deltas."
)

FAKE_FLOW_RESPONSIBILITY = "Improve the fake coding instructions using the failed examples."

LENGTH_RULE_NONEMPTY = (
    "Keep the revised module succinct: each candidate must be strictly less than 1.1 times "
    "the current module's character length."
)
LENGTH_RULE_EMPTY = "The current module is empty; write a short new module rather than copying other sections."
CONSOLIDATE_LENGTH_NONEMPTY = (
    "Each variant must be succinct and strictly less than 1.1 times the current module's character length. "
)
CONSOLIDATE_LENGTH_EMPTY = "The current module is empty; write a short new module. "

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
        "and does not use it first when the teacher chooses a different tool. Keep the text operational and concise."
    )


def teacher_student_reflection_prompt(module_name: str) -> str:
    """Module responsibility text for teacher-student first-tool reflection."""
    if module_name == FULL_PROMPT_KEY:
        return FULL_PROMPT_TEACHER_STUDENT_RESPONSIBILITY
    if module_name == RULES_EXT_KEY:
        return RULES_EXT_RESPONSIBILITY
    if module_name in CORE_TOOL_KEYS:
        return core_tool_reflection_prompt(module_name)
    return DEFAULT_MODULE_RESPONSIBILITY


def single_model_reflection_prompt(module_name: str) -> str:
    """Module responsibility text for single-model shell-error reflection."""
    if module_name == WRITING_CODE_KEY:
        return WRITING_CODE_SINGLE_MODEL_RESPONSIBILITY
    if module_name in CORE_TOOL_KEYS:
        return core_tool_reflection_prompt(module_name)
    return DEFAULT_MODULE_RESPONSIBILITY


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
        f"CURRENT MODULE TEXT:\n<<<\n{current}\n>>>\n\n"
        f"{failure_label}:\n{example_blocks}\n\n"
        f"Task:\n"
        f"1) Identify recurring failure modes that are plausibly caused by {module_name}.\n"
        f"2) Propose 1-2 SMALL patches (delta edits), each with:\n"
        f"   - BEFORE: quoted snippet from current module\n"
        f"   - AFTER: revised snippet\n"
        f"   - WHY: one sentence\n"
        f"3) Every supplied example is relevant evidence for {module_name}; use it to propose a variant.\n"
        f"4) Make only generalizable changes; do not overfit to individual examples.\n"
        f"5) {length_rule}\n"
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
        f"and make only generalizable changes. {consolidate_length}"
        f"Output each variant separated by '\n===VARIANT===\n'.\n\n"
        f"CURRENT:\n<<<\n{current}\n>>>\n\n"
        f"EVIDENCE (every example is relevant):\n{example_blocks}\n\n"
        f"SUGGESTIONS:\n{suggestions}\n"
    )
