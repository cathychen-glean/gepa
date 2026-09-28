"""Every registered objective satisfies the shared contract in ``objectives/protocol``."""

from __future__ import annotations

import pytest

from glean_gepa.objectives.base import (
    TELEMETRY_SOURCES,
    ScoredRow,
    SingleModelObjective,
    TeacherStudentObjective,
    _ensure_builtin_objectives_registered,
)
from glean_gepa.objectives.protocol import (
    REQUIRED_ATTRIBUTES,
    REQUIRED_METHODS,
    ObjectiveProtocol,
    check_objective_contract,
    require_objective_contract,
    scored_rows_are_normalized,
)


def _builtin_objective_classes() -> list[tuple[str, type]]:
    _ensure_builtin_objectives_registered()
    return [(f"{mode}/{source}", cls) for (mode, source), cls in sorted(TELEMETRY_SOURCES.items())]


@pytest.mark.parametrize(("label", "cls"), _builtin_objective_classes())
def test_builtin_objectives_satisfy_contract(label: str, cls: type) -> None:
    assert check_objective_contract(cls) == [], label
    # Single-model objectives require a client; any object satisfies the constructor check.
    instance = cls(bigquery_client=object(), lookback_days=1)
    assert isinstance(instance, ObjectiveProtocol), label


def test_contract_lists_every_missing_member_for_bare_class() -> None:
    class Bare:
        pass

    missing = check_objective_contract(Bare)
    assert missing == [f"attribute {name!r}" for name in REQUIRED_ATTRIBUTES] + [
        f"method {name}()" for name in REQUIRED_METHODS
    ]


def test_contract_treats_unimplemented_abstract_methods_as_missing() -> None:
    class Partial(TeacherStudentObjective):
        name = "partial"
        telemetry_dimensions = ("partial",)
        focused_bucket_type = "QUERY_CANONICAL"

    missing = check_objective_contract(Partial)
    assert "method analyze()" in missing
    assert "method entry_row()" in missing
    assert "method aggregate_row()" in missing
    assert "attribute 'failure_label'" not in missing  # inherited default counts


def test_require_contract_error_names_class_source_and_members() -> None:
    class Incomplete(SingleModelObjective):
        name = "incomplete"
        telemetry_dimensions = ("incomplete",)
        focused_bucket_type = "SESSION"

    with pytest.raises(TypeError) as excinfo:
        require_objective_contract(Incomplete, source="downvote_feedback")
    message = str(excinfo.value)
    assert "Incomplete" in message
    assert "source='downvote_feedback'" in message
    assert "method analyze()" in message


def test_reflective_example_helper_builds_the_shared_frame() -> None:
    """Objectives supply four slots; the helper owns Inputs, Metrics, defaults, and evidence caps."""
    from unittest.mock import MagicMock

    from glean_gepa.objectives.base import REFLECTION_EVIDENCE_LIMIT
    from glean_gepa.objectives.loop_efficiency import LoopEfficiencyObjective

    objective = LoopEfficiencyObjective(bigquery_client=MagicMock())
    trajectory = {
        "data": {"eval_set_name": "set", "eval_run_id": "run-1", "eval_trace_id": "trace-1"},
        "output": {"entry_id": "e1", "deployment_id": "dep", "query": "q"},
        "score": 0.5,
        "objective_scores": {objective.name: 0.5},
    }
    example = objective.reflective_example(
        trajectory,
        feedback="fix it",
        generated={"student_answer": "ans"},
        action_inputs=[f"cmd{i}" for i in range(REFLECTION_EVIDENCE_LIMIT + 3)],
        execution_errors=["boom"],
    )
    assert example["Inputs"] == {
        "eval_set": "set",
        "entry_id": "e1",
        "deployment_id": "dep",
        "query": "q",
        "eval_run_id": "run-1",
        "eval_trace_id": "trace-1",
    }
    # Partial ``generated`` is merged over empty defaults.
    assert example["Generated Outputs"] == {
        "student_answer": "ans",
        "teacher_answer": "",
        "student_tools": [],
        "teacher_tools": [],
    }
    assert len(example["Action Inputs"]) == REFLECTION_EVIDENCE_LIMIT
    assert example["Execution Errors"] == ["boom"]
    assert example["Feedback"] == "fix it"
    assert example["Metrics"]["score"] == 0.5

    # Optional ids are omitted, not set to None.
    bare = objective.reflective_example({**trajectory, "data": {"eval_set_name": "set"}}, feedback="f")
    assert "eval_run_id" not in bare["Inputs"]
    assert bare["Action Inputs"] == [] and bare["Execution Errors"] == []


def test_scored_rows_are_normalized_rejects_out_of_range_and_bool() -> None:
    ok = [ScoredRow(entry_id="a", dimension_scores={"m": 0.0}, output={}), ScoredRow("b", {"m": 1.0}, {})]
    assert scored_rows_are_normalized(ok)
    assert not scored_rows_are_normalized([ScoredRow("c", {"m": 1.5}, {})])
    assert not scored_rows_are_normalized([ScoredRow("d", {"m": True}, {})])
