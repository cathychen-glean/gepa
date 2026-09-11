"""The composite-scoring contract both Glean adapters share via GleanAdapterBase."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from glean_gepa.adapter_types import PointwiseJudge
from glean_gepa.al_adapter import ALRunner, Thresholds
from glean_gepa.objectives.utils.shell_tool_error_util import SHELL_SUCCESS_OBJECTIVE
from glean_gepa.single_model_adapter import SingleModelAdapter
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter
from glean_gepa.objectives.utils.tool_match_util import TOOL_ALIGNMENT_OBJECTIVE

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
