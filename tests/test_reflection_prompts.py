from glean_gepa.prompt_constants import FULL_PROMPT_KEY, RULES_EXT_KEY, WRITING_CODE_KEY
from glean_gepa.reflection_prompts import (
    RULES_EXT_RESPONSIBILITY,
    WRITING_CODE_SINGLE_MODEL_RESPONSIBILITY,
    single_model_reflection_prompt,
    teacher_student_reflection_prompt,
)


def test_reflection_prompts_route_by_module():
    assert "ENTIRE student system prompt" in teacher_student_reflection_prompt(FULL_PROMPT_KEY)
    assert teacher_student_reflection_prompt(RULES_EXT_KEY) == RULES_EXT_RESPONSIBILITY
    assert "glean_search" in teacher_student_reflection_prompt("glean_search")
    assert teacher_student_reflection_prompt(WRITING_CODE_KEY) == "Focus only on this module's responsibilities."

    assert single_model_reflection_prompt(WRITING_CODE_KEY) == WRITING_CODE_SINGLE_MODEL_RESPONSIBILITY
    assert "glean_search" in single_model_reflection_prompt("glean_search")
    assert single_model_reflection_prompt(FULL_PROMPT_KEY) == "Focus only on this module's responsibilities."
