from unittest.mock import MagicMock

import pytest

from glean_gepa.objectives.citation_match import CitationMatchObjective
from glean_gepa.objectives.shell import ShellSuccessObjective
from glean_gepa.objectives.tool_match import FirstToolMatchObjective
from glean_gepa.prompt_constants import (
    CORE_TOOL_DESCRIPTIONS,
    DEFAULT_WRITING_CODE,
    FULL_PROMPT_KEY,
    RULES_EXT_KEY,
    RULES_EXT_TOKEN_BUDGET,
    WRITING_CODE_KEY,
)
from glean_gepa.reflection_prompts import (
    CONDITIONAL_ABSENT_RULE,
    CONDITIONAL_PRESERVE_RULE,
    DEFAULT_MODULE_RESPONSIBILITY,
    DIAGNOSIS_MARKUP_RULE,
    PATCHES_ARE_THE_WHITELIST_RULE,
    consolidate_prompt,
    diagnosis_prompt,
    drops_render_slot,
    length_rule_for,
    markup_rule_for,
    module_char_budget,
    new_module_length_rule,
    parse_diagnosis_reply,
    patch_after_snippets,
    sanitize_proposed_module,
    tool_description_choice_prompt,
)


def test_reflection_prompts_route_by_module():
    tool_match = FirstToolMatchObjective().reflection_prompt
    assert tool_match(RULES_EXT_KEY) == FirstToolMatchObjective.module_responsibilities[RULES_EXT_KEY]
    assert "glean_search" in tool_match("glean_search")
    assert tool_match(WRITING_CODE_KEY) == DEFAULT_MODULE_RESPONSIBILITY

    shell = ShellSuccessObjective(bigquery_client=MagicMock()).reflection_prompt
    assert shell(WRITING_CODE_KEY) == ShellSuccessObjective.module_responsibilities[WRITING_CODE_KEY]
    assert "<<<[[hitl_approval_instructions]]>>>" in shell(WRITING_CODE_KEY)
    assert "glean_search" in shell("glean_search")

    citation_match = CitationMatchObjective().reflection_prompt
    assert citation_match(RULES_EXT_KEY) == CitationMatchObjective.module_responsibilities[RULES_EXT_KEY]
    # The citations run edits Writing Code, so it needs citation-specific guidance there.
    assert citation_match(WRITING_CODE_KEY) == CitationMatchObjective.module_responsibilities[WRITING_CODE_KEY]
    assert "citationId" in citation_match(WRITING_CODE_KEY)
    # Core-tool keys always get the tool-description essay; experiments that should not
    # rewrite those descriptions omit CORE_TOOLS from editable_modules.
    assert "glean_search" in citation_match("glean_search")

    # FULL_PROMPT is the render template, so no objective may claim it as a module.
    for objective in (FirstToolMatchObjective, CitationMatchObjective, ShellSuccessObjective):
        assert FULL_PROMPT_KEY not in objective.module_responsibilities


@pytest.mark.parametrize(
    "prompt",
    [
        FirstToolMatchObjective().reflection_prompt(RULES_EXT_KEY),
        FirstToolMatchObjective().reflection_prompt("glean_search"),
        CitationMatchObjective().reflection_prompt(RULES_EXT_KEY),
    ],
)
def test_teacher_student_prompts_forbid_teacher_referential_edits(prompt: str):
    """The student cannot see the teacher, so reflection must not write it into the prompt.

    Reflection is shown TEACHER_* evidence rows and, unguarded, turns them into inert
    instructions like "cite exactly the teacher's citationIds for this turn".
    """
    assert "offline scoring reference" in prompt
    assert "never mention" in prompt


def test_module_text_fences_do_not_use_conditional_syntax():
    prompt = diagnosis_prompt(
        module_name="glean_search",
        responsibility="r",
        current=CORE_TOOL_DESCRIPTIONS["glean_search"],
        failure_label="FAILURES",
        example_blocks="e",
        length_rule="len",
    )
    assert "----- BEGIN CURRENT MODULE TEXT -----" in prompt
    assert "<<<" not in prompt
    assert DIAGNOSIS_MARKUP_RULE in prompt


def test_diagnosis_prompt_asks_for_a_diagnosis_not_a_rewrite():
    """The patch pass must not carry the 'return the module text by itself' rule.

    GPT-5 obeyed that rule over the BEFORE/AFTER format and returned a finished
    rewrite, so the consolidation pass received no mechanism tally and the
    'fewer than three LOSS examples' constraint never bound.
    """
    prompt = diagnosis_prompt(
        module_name="ask_user_questions",
        responsibility="r",
        current="plain description",
        failure_label="FAILURES",
        example_blocks="e",
        length_rule="len",
    )
    assert CONDITIONAL_ABSENT_RULE not in prompt
    assert "return the module text by itself" not in prompt
    assert "DIAGNOSIS:" in prompt and "PATCHES:" in prompt
    assert "do NOT return a full rewrite" in prompt

    # Modules with template markup keep the preservation rules in the patch pass.
    with_markup = diagnosis_prompt(
        module_name="WRITING_CODE",
        responsibility="r",
        current=DEFAULT_WRITING_CODE,
        failure_label="FAILURES",
        example_blocks="e",
        length_rule="len",
    )
    assert CONDITIONAL_PRESERVE_RULE in with_markup
    assert "{RULES_EXT}" in with_markup


_ASK_DESCRIPTION = (
    "Use ask_user_questions when a missing detail would change the deliverable and it cannot be "
    "inferred from context or tools. Batch the questions, offer sensible options, and default to "
    "the most common interpretation when only one plausible reading exists. Do not ask when the "
    "request is answerable, when the ambiguity does not change the outcome, or after tools have "
    "already produced the needed information; deliver instead and state the assumption you made. "
    "Never ask about audience, tone, depth or format when a reasonable default exists."
)


def test_parse_diagnosis_reply_splits_the_tally_from_the_patches():
    reply = parse_diagnosis_reply(
        "DIAGNOSIS:\n- Asks in prose instead of delivering: 5 of 8 LOSS examples.\n"
        "PATCHES:\nBEFORE: Use ask_user_questions when\nAFTER: Use ask_user_questions only when\nWHY: fewer asks",
        current=_ASK_DESCRIPTION,
    )
    assert not reply.is_module_rewrite
    assert reply.diagnosis == "- Asks in prose instead of delivering: 5 of 8 LOSS examples."
    assert reply.patches.startswith("BEFORE:") and reply.patches.endswith("WHY: fewer asks")


def test_parse_diagnosis_reply_flags_a_bare_module_rewrite():
    """GPT-5 answered the diagnosis prompt with the rewritten description; that is not a diagnosis."""
    rewrite = _ASK_DESCRIPTION.replace("Batch the questions", "Ask at most one batch of questions") + (
        " Prefer delivering a best-effort answer over asking."
    )
    reply = parse_diagnosis_reply(rewrite, current=_ASK_DESCRIPTION)
    assert reply.is_module_rewrite
    assert reply.suggestions == ""

    # Short free-form diagnoses and BEFORE/AFTER patches without a header are still accepted.
    assert not parse_diagnosis_reply("diagnosis", current=_ASK_DESCRIPTION).is_module_rewrite
    unheaded = parse_diagnosis_reply("BEFORE: x\nAFTER: y\nWHY: z", current=_ASK_DESCRIPTION)
    assert not unheaded.is_module_rewrite and unheaded.patches.startswith("BEFORE")


_TWO_PATCHES = (
    "BEFORE:\n[empty]\nAFTER:\n- After writing or editing a file, re-open it and quote only the changed region.\n"
    "WHY: unverified edit claims\n\n"
    "BEFORE:\n[empty]\nAFTER:\n- Update the existing artifact instead of creating a new file.\nWHY: file drift"
)


def test_patch_after_snippets_extracts_each_after_block():
    assert patch_after_snippets(_TWO_PATCHES) == [
        "- After writing or editing a file, re-open it and quote only the changed region.",
        "- Update the existing artifact instead of creating a new file.",
    ]
    assert patch_after_snippets("no patches here") == []


def test_new_module_length_rule_anchors_an_empty_module_to_its_patches():
    """RULES_EXT's 2,048-char token budget let a two-patch diagnosis become a 23-line rulebook."""
    rule, budget = new_module_length_rule(_TWO_PATCHES, 2048)
    afters = patch_after_snippets(_TWO_PATCHES)
    assert budget == int(1.5 * sum(len(a) for a in afters))
    assert budget < 2048
    assert "write 2 rules, one per AFTER snippet" in rule
    assert f"at most {budget} characters" in rule

    _, floored = new_module_length_rule("AFTER:\n- short.", 2048)
    assert floored == 200
    _, capped = new_module_length_rule(_TWO_PATCHES, 100)
    assert capped == 100
    no_patches_rule, no_patches_budget = new_module_length_rule("just prose", 2048)
    assert no_patches_budget == 2048 and "empty" in no_patches_rule


def test_consolidate_prompt_treats_patches_as_a_whitelist():
    prompt = consolidate_prompt(
        module_name="RULES_EXT",
        max_variants=3,
        consolidate_length="len",
        current="",
        example_blocks="e",
        suggestions=_TWO_PATCHES,
    )
    assert PATCHES_ARE_THE_WHITELIST_RULE in prompt
    assert "nothing else" in PATCHES_ARE_THE_WHITELIST_RULE
    assert "not in scope" in PATCHES_ARE_THE_WHITELIST_RULE


def test_consolidate_prompt_binds_variants_to_the_diagnosis_tally():
    prompt = consolidate_prompt(
        module_name="EXECUTION_DISCIPLINE",
        max_variants=3,
        consolidate_length="len",
        current="plain",
        example_blocks="e",
        suggestions="BEFORE: a\nAFTER: b\nWHY: c",
        diagnosis="- hedging: 4 of 10 LOSS examples",
    )
    assert "DIAGNOSIS (failure-mode tally from the first pass):\n- hedging: 4 of 10 LOSS examples" in prompt
    assert "must address a failure mode listed in the DIAGNOSIS" in prompt
    assert prompt.index("DIAGNOSIS (failure-mode") < prompt.index("SUGGESTIONS:")
    without = consolidate_prompt(
        module_name="EXECUTION_DISCIPLINE",
        max_variants=3,
        consolidate_length="len",
        current="plain",
        example_blocks="e",
        suggestions="s",
    )
    assert "DIAGNOSIS (failure-mode" not in without


def test_tool_choice_prompt_lists_invocation_counts():
    prompt = tool_description_choice_prompt(
        tools={"glean_search": "s", "glean_document_reader": "r"},
        example_blocks="e",
        limit=3,
        usage={"glean_search": (2, 7), "glean_document_reader": (0, 5)},
        example_count=9,
    )
    assert "glean_search: student invoked it in 2 of 9 examples, teacher in 7 of 9" in prompt
    assert "glean_document_reader: student invoked it in 0 of 9 examples, teacher in 5 of 9" in prompt
    assert "rarely invokes cannot be steered by its description" in prompt
    assert "including a tool that does not appear in the examples" not in prompt


def test_markup_rule_demands_the_render_slots_back():
    """Writing Code carries the {RULES_EXT} slot, so a rewrite that drops it orphans that module."""
    rule = markup_rule_for(DEFAULT_WRITING_CODE)
    assert CONDITIONAL_PRESERVE_RULE in rule
    assert "{RULES_EXT}" in rule
    assert markup_rule_for("plain text") == CONDITIONAL_ABSENT_RULE

    assert drops_render_slot("- rule\n### Sandbox", current=DEFAULT_WRITING_CODE)
    assert not drops_render_slot("- rule\n{RULES_EXT}\n### Sandbox", current=DEFAULT_WRITING_CODE)
    assert not drops_render_slot("- rule", current="plain text")


def test_sanitize_strips_reflector_artifacts():
    search = CORE_TOOL_DESCRIPTIONS["glean_search"]
    invented = (
        "<<<\nSearch across company documents. Prefer this before opening a doc. "
        "<<<[[hitl_approval_instructions]]>>> Glean search is connected to all datasources. >>>"
    )
    cleaned = sanitize_proposed_module(invented, current=search)
    assert "<<<" not in cleaned
    assert "[[hitl_approval_instructions]]" not in cleaned
    assert "Glean search is connected" in cleaned

    labeled = "Variant 1 (351 chars)\nRULES_EXT\n- Use Edit on the named file."
    assert sanitize_proposed_module(labeled, current="", module_name="RULES_EXT") == "- Use Edit on the named file."

    wrapped = f"<<<\n{DEFAULT_WRITING_CODE}\n>>>"
    assert sanitize_proposed_module(wrapped, current=DEFAULT_WRITING_CODE).count("<<<") == DEFAULT_WRITING_CODE.count(
        "<<<"
    )


def test_length_rule_states_a_character_count():
    tool_text = CORE_TOOL_DESCRIPTIONS["glean_search"]
    budget = module_char_budget(tool_text)
    rule = length_rule_for(tool_text)
    assert str(budget) in rule
    assert str(len(tool_text)) in rule
    assert "1.2" not in rule


def test_module_char_budget_allows_twenty_percent_growth():
    """Waldo's 8102-char module needs room for a ~10-15% consolidated overshoot."""
    current = "x" * 8102
    assert module_char_budget(current) == int(8102 * 1.2) == 9722
    # Token cap still wins when it is lower.
    assert module_char_budget(current, token_budget=2000) == 8000
    # Small modules keep the 200-char floor.
    assert module_char_budget("short") == 200


def test_empty_rules_ext_char_budget_fits_consolidated_bullets():
    """128 tokens capped empty RULES_EXT at 512 chars and dropped the 869-char rewrite."""
    dropped_chars = 869
    assert module_char_budget("", 128) < dropped_chars
    assert module_char_budget("", RULES_EXT_TOKEN_BUDGET) >= dropped_chars
