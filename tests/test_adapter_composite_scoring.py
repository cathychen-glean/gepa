"""The composite-scoring contract both Glean adapters share via GleanAdapterBase."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from glean_gepa.adapter_types import PointwiseJudge
from glean_gepa.al_adapter import ALRunner, Thresholds
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evolutionary_proposer import pick_modules_to_edit
from glean_gepa.objectives.utils.shell_tool_error_util import SHELL_SUCCESS_OBJECTIVE
from glean_gepa.objectives.utils.tool_match_util import TOOL_ALIGNMENT_OBJECTIVE
from glean_gepa.prompt_constants import RULES_EXT_KEY
from glean_gepa.single_model_adapter import SingleModelAdapter
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter

THRESHOLDS = Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000)


def _teacher_student(**kwargs) -> TeacherStudentAdapter:
    return TeacherStudentAdapter(
        runner=ALRunner(evalcli=MagicMock()),
        teacher_model="gpt",
        student_model="fast",
        thresholds=THRESHOLDS,
        **kwargs,
    )


def _single_model(**kwargs) -> SingleModelAdapter:
    return SingleModelAdapter(
        runner=ALRunner(evalcli=MagicMock()),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=THRESHOLDS,
        **kwargs,
    )


BOTH_ADAPTERS = [
    pytest.param(_teacher_student, TOOL_ALIGNMENT_OBJECTIVE, id="teacher_student"),
    pytest.param(_single_model, SHELL_SUCCESS_OBJECTIVE, id="single_model"),
]


@pytest.mark.parametrize(("build", "telemetry_dimension"), BOTH_ADAPTERS)
def test_both_adapters_default_to_scoring_their_own_telemetry_dimension(build, telemetry_dimension):
    adapter = build()

    assert adapter.telemetry_dimensions == (telemetry_dimension,)
    assert telemetry_dimension in adapter.scorable_dimensions()
    assert telemetry_dimension in adapter.composite_weights


@pytest.mark.parametrize(("build", "telemetry_dimension"), BOTH_ADAPTERS)
def test_both_adapters_weight_constants_through_the_composite(build, telemetry_dimension):
    adapter = build(
        composite_weights={telemetry_dimension: 0.6, "fixed_signal": 0.4},
        constant_scores={"fixed_signal": 0.5},
    )

    assert {telemetry_dimension, "fixed_signal"} <= adapter.scorable_dimensions()
    assert adapter.composite_score({telemetry_dimension: 1.0, "fixed_signal": 0.5}) == pytest.approx(0.8)


@pytest.mark.parametrize(("build", "telemetry_dimension"), BOTH_ADAPTERS)
def test_both_adapters_reject_a_composite_they_cannot_score(build, telemetry_dimension):
    with pytest.raises(ValueError, match="cannot score: made_up_signal"):
        build(composite_weights={telemetry_dimension: 0.5, "made_up_signal": 0.5})


def test_only_teacher_student_can_score_a_judge_dimension():
    """The one intentional asymmetry: single_model has no judge plumbing."""
    judges = (PointwiseJudge("answer_quality", "CORRECTNESS", "{}"),)
    teacher_student = _teacher_student(
        pointwise_judges=judges,
        composite_weights={"answer_quality": 1.0},
        constant_scores={},
    )

    assert "answer_quality" in teacher_student.scorable_dimensions()
    with pytest.raises(ValueError, match="cannot score: answer_quality"):
        _single_model(composite_weights={"answer_quality": 1.0}, constant_scores={})


def test_concrete_adapters_own_screening_configuration():
    single_adapter = _single_model()
    single_rules_ext = _single_model(editable_modules=[RULES_EXT_KEY])
    teacher_adapter = _teacher_student()
    shell_eval = GleanEvaluationBatch(
        outputs=[],
        scores=[0.8],
        summary={SHELL_SUCCESS_OBJECTIVE: 0.8, "correctness": 0.5},
    )
    tool_match_eval = GleanEvaluationBatch(
        outputs=[],
        scores=[0.85],
        summary={TOOL_ALIGNMENT_OBJECTIVE: 0.5, "correctness": 1.0},
    )

    correctness_adapter = _teacher_student(primary_objective="correctness")

    assert single_adapter.get_screening_score(shell_eval) == 0.8
    assert teacher_adapter.get_screening_score(tool_match_eval) == 0.5
    assert correctness_adapter.get_screening_score(tool_match_eval) == 1.0
    assert (
        correctness_adapter.high_signal_fix_rate(
            GleanEvaluationBatch(outputs=[], scores=[0.0], trajectories=[{"score": 0.0}]),
            tool_match_eval,
        )
        == 1.0
    )
    assert pick_modules_to_edit(single_rules_ext) == [RULES_EXT_KEY]
    assert not hasattr(single_adapter, "judging_mode")
    assert not hasattr(teacher_adapter, "judge")
