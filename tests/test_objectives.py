"""The objective catalog is the single registration point and rejects bad entries."""

from __future__ import annotations

import pytest

from glean_gepa.experiment_config import ExperimentConfigError, load_experiment_config
from glean_gepa.objectives import register_telemetry_source, registry, unregister_telemetry_source
from glean_gepa.objectives.base import (
    MODE_DEFAULT_TELEMETRY_SOURCE,
    TELEMETRY_SOURCES,
    ScoredRow,
    TeacherStudentObjective,
    build_objective,
)
from glean_gepa.objectives.protocol import (
    REQUIRED_ATTRIBUTES,
    REQUIRED_METHODS,
    ObjectiveProtocol,
    check_objective_contract,
    scored_rows_are_normalized,
)
from glean_gepa.objectives.registry import (
    BUILTIN_OBJECTIVES,
    VALID_MODES,
    ObjectiveRegistrationError,
    validate_objective_class,
)


def test_every_catalog_spec_loads_and_registers() -> None:
    registered = registry.registered()
    for spec in BUILTIN_OBJECTIVES:
        assert registered[(spec.mode, spec.source)] is spec.load(), spec.source
    assert len(registered) >= len(BUILTIN_OBJECTIVES)


def test_catalog_has_one_default_per_mode_and_base_aliases_follow_it() -> None:
    defaults = {spec.mode for spec in BUILTIN_OBJECTIVES if spec.default}
    assert defaults == VALID_MODES
    assert MODE_DEFAULT_TELEMETRY_SOURCE == {"teacher_student": "tool_match", "single_model": "shell_telemetry"}
    assert TELEMETRY_SOURCES is registry.registered()


def test_catalog_sources_are_unique_per_mode() -> None:
    keys = [(spec.mode, spec.source) for spec in BUILTIN_OBJECTIVES]
    assert len(keys) == len(set(keys))


def test_builtins_pass_validation_and_instantiate() -> None:
    for spec in BUILTIN_OBJECTIVES:
        cls = spec.load()
        assert validate_objective_class(cls, mode=spec.mode, source=spec.source) == [], spec.source
        # Single-model objectives require a client; any object satisfies the constructor check.
        assert isinstance(cls(bigquery_client=object(), lookback_days=1), ObjectiveProtocol), spec.source


def test_a_registered_second_source_is_selectable_from_config(tmp_path) -> None:
    class Dummy:
        name = "dummy_alignment"
        telemetry_dimensions = ("dummy_alignment",)

        def __init__(self, *, bigquery_client=None, lookback_days: int = 1):
            pass

    yaml = tmp_path / "mode.yaml"
    yaml.write_text(
        "schema_version: 1\nmode: teacher_student\nsignals:\n  - name: dummy_alignment\n    source: dummy_trace\n"
        "objective:\n  primary: dummy_alignment\n  composite:\n    dummy_alignment: 1.0\n"
    )
    with pytest.raises(ExperimentConfigError, match="cannot score signal"):
        load_experiment_config(yaml)
    register_telemetry_source("teacher_student", "dummy_trace", Dummy, validate=False)
    try:
        config = load_experiment_config(yaml)
        assert isinstance(build_objective("teacher_student", config.signals), Dummy)
    finally:
        unregister_telemetry_source("teacher_student", "dummy_trace")


def test_register_rejects_incomplete_class_with_every_problem_listed() -> None:
    class Incomplete(TeacherStudentObjective):
        name = "incomplete"
        telemetry_dimensions = ("incomplete",)
        focused_bucket_type = "QUERY_CANONICAL"

        def __init__(self) -> None:  # wrong constructor on purpose
            pass

    with pytest.raises(ObjectiveRegistrationError) as excinfo:
        registry.register("teacher_student", "incomplete", Incomplete)
    message = str(excinfo.value)
    assert "missing method analyze()" in message
    assert "missing method entry_row()" in message
    assert "__init__ must accept keyword 'bigquery_client'" in message
    assert not registry.is_registered("teacher_student", "incomplete")


def test_register_rejects_wrong_mode_base_and_bad_source() -> None:
    shell = registry.resolve("single_model", "shell_telemetry")
    problems = validate_objective_class(shell, mode="teacher_student", source="Bad-Source")
    assert any("must subclass TeacherStudentObjective" in p for p in problems)
    assert any("snake_case" in p for p in problems)


def test_register_refuses_silent_override_but_allows_replace() -> None:
    tool_match = registry.resolve("teacher_student", "tool_match")
    citation = registry.resolve("teacher_student", "citation_match")
    registry.register("teacher_student", "tool_match", tool_match)  # same class: no-op
    with pytest.raises(ObjectiveRegistrationError, match="already registered"):
        registry.register("teacher_student", "tool_match", citation)
    try:
        registry.register("teacher_student", "tool_match", citation, replace=True)
        assert registry.resolve("teacher_student", "tool_match") is citation
    finally:
        registry.register("teacher_student", "tool_match", tool_match, replace=True)


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


def test_reflective_example_helper_builds_the_shared_frame() -> None:
    """Objectives supply four slots; the helper owns Inputs, Metrics, defaults, and evidence caps."""
    from unittest.mock import MagicMock

    from glean_gepa.objectives.base import REFLECTION_EVIDENCE_LIMIT
    from glean_gepa.objectives.shell import ShellSuccessObjective

    objective = ShellSuccessObjective(bigquery_client=MagicMock())
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
