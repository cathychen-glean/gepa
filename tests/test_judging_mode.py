from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from glean_gepa.al_adapter import ALRunner, Thresholds
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.evolutionary_proposer import pick_modules_to_edit
from glean_gepa.prompt import FULL_PROMPT_KEY, WRITING_CODE_KEY
from glean_gepa.shell_tool_error_util import SHELL_SUCCESS_OBJECTIVE
from glean_gepa.single_model_adapter import SingleModelAdapter
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter


def test_concrete_adapters_own_screening_configuration():
    runner = ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli"))
    thresholds = Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000)
    single_adapter = SingleModelAdapter(
        runner=runner,
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=thresholds,
    )
    single_full_prompt = SingleModelAdapter(
        runner=runner,
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=thresholds,
        editable_modules=[FULL_PROMPT_KEY],
    )
    teacher_adapter = TeacherStudentAdapter(
        runner=runner,
        teacher_model="gpt",
        student_model="fast",
        thresholds=thresholds,
    )
    teacher_full_prompt = TeacherStudentAdapter(
        runner=runner,
        teacher_model="gpt",
        student_model="fast",
        thresholds=thresholds,
        editable_modules=[FULL_PROMPT_KEY],
    )
    shell_eval = GleanEvaluationBatch(
        outputs=[],
        scores=[0.8],
        summary={SHELL_SUCCESS_OBJECTIVE: 0.8, "correctness": 0.5},
    )
    tool_match_eval = GleanEvaluationBatch(
        outputs=[],
        scores=[0.85],
        summary={"tool_alignment": 0.5, "completeness": 1.0, "grounding": 1.0},
    )

    assert single_adapter.primary_objective == SHELL_SUCCESS_OBJECTIVE
    assert single_adapter.default_frontier_type == "objective"
    assert single_adapter.editable_modules == [WRITING_CODE_KEY]
    assert single_full_prompt.editable_modules == [FULL_PROMPT_KEY]
    assert single_adapter.get_screening_score(shell_eval) == 0.8
    assert teacher_adapter.primary_objective == "tool_alignment"
    assert teacher_adapter.default_frontier_type == "hybrid"
    assert teacher_adapter.editable_modules == [WRITING_CODE_KEY]
    assert teacher_full_prompt.editable_modules == [FULL_PROMPT_KEY]
    assert pick_modules_to_edit(single_adapter) == [WRITING_CODE_KEY]
    assert pick_modules_to_edit(single_full_prompt) == [FULL_PROMPT_KEY]
    assert pick_modules_to_edit(teacher_full_prompt) == [FULL_PROMPT_KEY]
    assert teacher_adapter.get_screening_score(tool_match_eval) == 0.5
    assert not hasattr(single_adapter, "judging_mode")
    assert not hasattr(teacher_adapter, "judging_mode")
    assert not hasattr(teacher_adapter, "judge")


def test_single_model_adapter_requires_bigquery_client():
    runner = ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli"))
    with pytest.raises(ValueError, match="bigquery_client is required"):
        SingleModelAdapter(
            runner=runner,
            student_model="fast",
            thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        )
