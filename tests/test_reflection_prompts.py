from unittest.mock import MagicMock

import pytest

from glean_gepa.objectives.citation_match import CitationMatchObjective
from glean_gepa.objectives.loop import LoopEfficiencyObjective
from glean_gepa.objectives.shell import ShellSuccessObjective
from glean_gepa.objectives.tool_match import FirstToolMatchObjective
from glean_gepa.prompt_constants import (
    CORE_TOOL_DESCRIPTIONS,
    DEFAULT_WRITING_CODE,
    FULL_PROMPT_KEY,
    RULES_EXT_KEY,
    WRITING_CODE_KEY,
)
from glean_gepa.reflection_prompts import (
    CONDITIONAL_ABSENT_RULE,
    CONDITIONAL_PRESERVE_RULE,
    DEFAULT_MODULE_RESPONSIBILITY,
    diagnosis_prompt,
    drops_render_slot,
    length_rule_for,
    markup_rule_for,
    module_char_budget,
    sanitize_proposed_module,
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
    # Core-tool keys always get the tool-description essay; packs that should not
    # rewrite those descriptions omit CORE_TOOLS from editable_modules.
    assert "glean_search" in citation_match("glean_search")
    assert "glean_search" in LoopEfficiencyObjective(bigquery_client=MagicMock()).reflection_prompt("glean_search")

    loop = LoopEfficiencyObjective(bigquery_client=MagicMock()).reflection_prompt
    assert loop(WRITING_CODE_KEY) == LoopEfficiencyObjective.module_responsibilities[WRITING_CODE_KEY]

    # FULL_PROMPT is the render template, so no objective may claim it as a module.
    for objective in (FirstToolMatchObjective, CitationMatchObjective, ShellSuccessObjective, LoopEfficiencyObjective):
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
    assert CONDITIONAL_ABSENT_RULE in prompt


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
    assert "1.1" not in rule
