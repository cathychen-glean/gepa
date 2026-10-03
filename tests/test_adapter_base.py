"""The composite-scoring contract both Glean adapters share via GleanAdapterBase."""

from __future__ import annotations

import threading

import pytest
from helpers import single_model_adapter, teacher_student_adapter

from glean_gepa.adapter_types import PointwiseJudge
from glean_gepa.al_adapter import GleanAdapterBase
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evolutionary_proposer import _select_screened_children, pick_modules_to_edit
from glean_gepa.objectives.shell import SHELL_SUCCESS_OBJECTIVE
from glean_gepa.objectives.tool_match import TOOL_ALIGNMENT_OBJECTIVE
from glean_gepa.prompt_constants import RULES_EXT_KEY

BOTH_ADAPTERS = [
    pytest.param(teacher_student_adapter, TOOL_ALIGNMENT_OBJECTIVE, id="teacher_student"),
    pytest.param(single_model_adapter, SHELL_SUCCESS_OBJECTIVE, id="single_model"),
]


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
    teacher_student = teacher_student_adapter(
        pointwise_judges=judges,
        composite_weights={"answer_quality": 1.0},
        constant_scores={},
    )

    assert "answer_quality" in teacher_student.scorable_dimensions()
    with pytest.raises(ValueError, match="cannot score: answer_quality"):
        single_model_adapter(composite_weights={"answer_quality": 1.0}, constant_scores={})


def test_screening_score_follows_the_primary_or_the_weighted_blend():
    """Screen on the primary by default; with screening.weights the child must clear the blend,
    and an empty child eval can never pass."""
    tool_match_eval = GleanEvaluationBatch(outputs=[], scores=[0.85], summary={TOOL_ALIGNMENT_OBJECTIVE: 0.5, "correctness": 1.0})
    assert teacher_student_adapter().get_screening_score(tool_match_eval) == 0.5
    assert teacher_student_adapter(primary_objective="correctness").get_screening_score(tool_match_eval) == 1.0

    weighted = teacher_student_adapter(
        primary_objective=TOOL_ALIGNMENT_OBJECTIVE,
        screening_weights={TOOL_ALIGNMENT_OBJECTIVE: 0.5, "agentic_preference_rate": 0.5},
    )
    blended = GleanEvaluationBatch(
        outputs=[], scores=[0.4], summary={TOOL_ALIGNMENT_OBJECTIVE: 0.8, "agentic_preference_rate": 0.4}
    )
    parent = GleanEvaluationBatch(outputs=[], scores=[0.2], trajectories=[{"score": 0.2}])
    assert weighted.get_screening_score(blended) == 0.8
    assert weighted.child_screen_score(parent, blended) == pytest.approx(0.6)
    assert weighted.child_screen_score(tool_match_eval, GleanEvaluationBatch(outputs=[], scores=[])) == float("-inf")


def _batch(scores: list[float]) -> GleanEvaluationBatch:
    trajectories = [
        {
            "data": {
                "eval_set_name": "set",
                "eval_set_version": "v1",
                "deployment_ids": ["prod"],
                "status": "active",
            },
            "output": {"entry_id": f"entry-{index}"},
            "score": score,
            "objective_scores": {},
        }
        for index, score in enumerate(scores)
    ]
    return GleanEvaluationBatch(
        outputs=[],
        scores=scores,
        trajectories=trajectories,
        objective_scores=[{} for _ in scores],
        summary={"objective": sum(scores) / len(scores) if scores else 0.0},
    )


def test_high_signal_batch_contains_only_parent_failures():
    adapter = GleanAdapterBase.__new__(GleanAdapterBase)

    focused = adapter.high_signal_batch(_batch([0.0, 0.5, 1.0]))

    assert len(focused) == 1
    assert focused[0]["eval_entry_ids"] == ["entry-0", "entry-1"]


def test_high_signal_batch_skips_validation_only_eval_sets():
    """Validation runs on customer deployments whose entries are PII-gated, so
    reflection and focused screening must stay on the training deployments."""
    adapter = GleanAdapterBase.__new__(GleanAdapterBase)
    batch = _batch([0.0, 0.0])
    trajectories = batch.trajectories or []
    trajectories[1]["data"] = {
        **trajectories[1]["data"],
        "deployment_ids": ["bill"],
        "eval_set_version": "20260906",
        "validation_only": True,
    }

    focused = adapter.high_signal_batch(batch)

    assert len(focused) == 1
    assert focused[0]["deployment_ids"] == ["prod"]
    assert focused[0]["eval_entry_ids"] == ["entry-0"]


def test_high_signal_fix_rate():
    adapter = GleanAdapterBase.__new__(GleanAdapterBase)
    parent = _batch([0.0, 0.0, 0.0, 1.0])

    assert adapter.high_signal_fix_rate(parent, _batch([1.0, 1.0, 0.0])) == 2 / 3
    assert adapter.high_signal_fix_rate(_batch([1.0]), _batch([1.0])) == 0.0


def test_high_signal_screen_threshold():
    adapter = GleanAdapterBase.__new__(GleanAdapterBase)
    parent = _batch([0.0, 0.0, 0.0])
    keep, reject, exact = object(), object(), object()

    kept = _select_screened_children(
        adapter,
        parent,
        [keep, reject, exact],  # type: ignore[arg-type]
        [_batch([1.0, 1.0, 0.0]), _batch([0.0, 0.0, 0.0]), _batch([1.0, 0.0, 0.0])],
        use_high_signal_gate=True,
    )
    assert [(child, score) for child, _evaluation, score in kept] == [(keep, 2 / 3)]

    below_custom = _select_screened_children(
        adapter,
        parent,
        [keep],  # type: ignore[arg-type]
        [_batch([1.0, 1.0, 0.0])],
        use_high_signal_gate=True,
        high_signal_screen_threshold=0.8,
    )
    assert below_custom == []


def test_high_signal_batch_evaluation_dispatches_children_concurrently():
    adapter = GleanAdapterBase.__new__(GleanAdapterBase)
    barrier = threading.Barrier(2)

    def evaluate_fn(_batch_data, _candidate, _capture_traces):
        barrier.wait(timeout=1)
        return _batch([1.0])

    adapter._evaluate_fn = evaluate_fn
    items = [
        (
            {"WRITING_CODE": f"child-{index}"},
            [
                {
                    "eval_set_name": "set",
                    "eval_set_version": "v1",
                    "deployment_ids": ["prod"],
                    "status": "active",
                    "eval_entry_ids": ["entry"],
                }
            ],
        )
        for index in range(2)
    ]

    results = adapter.batch_evaluate(items)

    assert len(results) == 2
