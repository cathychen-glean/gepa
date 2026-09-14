import pytest

from glean_gepa.prompt_constants import FULL_PROMPT_KEY, RULES_EXT_KEY, WRITING_CODE_KEY
from glean_gepa.reflection_prompts import (
    RULES_EXT_RESPONSIBILITY,
    WRITING_CODE_SINGLE_MODEL_RESPONSIBILITY,
    single_model_reflection_prompt,
    teacher_student_citation_reflection_prompt,
    teacher_student_reflection_prompt,
)


def test_reflection_prompts_route_by_module():
    assert "ENTIRE student system prompt" in teacher_student_reflection_prompt(FULL_PROMPT_KEY)
    assert "<<<[[hitl_approval_instructions]]>>>" in teacher_student_reflection_prompt(FULL_PROMPT_KEY)
    assert teacher_student_reflection_prompt(RULES_EXT_KEY) == RULES_EXT_RESPONSIBILITY
    assert "glean_search" in teacher_student_reflection_prompt("glean_search")
    assert teacher_student_reflection_prompt(WRITING_CODE_KEY) == "Focus only on this module's responsibilities."

    assert single_model_reflection_prompt(WRITING_CODE_KEY) == WRITING_CODE_SINGLE_MODEL_RESPONSIBILITY
    assert "<<<[[hitl_approval_instructions]]>>>" in single_model_reflection_prompt(WRITING_CODE_KEY)
    assert "glean_search" in single_model_reflection_prompt("glean_search")
    assert single_model_reflection_prompt(FULL_PROMPT_KEY) == "Focus only on this module's responsibilities."


@pytest.mark.parametrize(
    "prompt",
    [
        teacher_student_reflection_prompt(FULL_PROMPT_KEY),
        teacher_student_reflection_prompt(RULES_EXT_KEY),
        teacher_student_reflection_prompt("glean_search"),
        teacher_student_citation_reflection_prompt(FULL_PROMPT_KEY),
        teacher_student_citation_reflection_prompt(RULES_EXT_KEY),
    ],
)
def test_teacher_student_prompts_forbid_teacher_referential_edits(prompt: str):
    """The student cannot see the teacher, so reflection must not write it into the prompt.

    Reflection is shown TEACHER_* evidence rows and, unguarded, turns them into inert
    instructions like "cite exactly the teacher's citationIds for this turn".
    """
    assert "offline scoring reference" in prompt
    assert "never mention" in prompt
