"""The objective catalog lists valid classes, ``build_objective`` picks the right one, and it stays a leaf module."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from glean_gepa.experiment_config import ExperimentConfigError, load_experiment_config
from glean_gepa.objectives.base import SingleModelObjective, TeacherStudentObjective
from glean_gepa.objectives.loop import LoopEfficiencyObjective
from glean_gepa.objectives.registry import DEFAULT_SOURCE, OBJECTIVES, build_objective, is_known_source, is_registered
from glean_gepa.objectives.shell import ShellSuccessObjective
from glean_gepa.objectives.tool_match import FirstToolMatchObjective

MODE_BASES = {"teacher_student": TeacherStudentObjective, "single_model": SingleModelObjective}
CATALOG = [(mode, source, cls) for mode, sources in OBJECTIVES.items() for source, cls in sources.items()]
# Abstract bases don't enforce class attributes, so the catalog test checks them explicitly.
REQUIRED_CLASS_ATTRIBUTES = ("name", "telemetry_dimensions", "focused_bucket_type", "failure_label")


def test_catalog_covers_both_modes() -> None:
    assert set(OBJECTIVES) == set(MODE_BASES)


@pytest.mark.parametrize(("mode", "source", "cls"), CATALOG, ids=[f"{m}/{s}" for m, s, _ in CATALOG])
def test_every_catalog_entry_is_a_complete_objective_for_its_mode(mode: str, source: str, cls: type) -> None:
    # shell and loop reject a None client, so every entry gets a fake one.
    objective = cls(bigquery_client=MagicMock(), lookback_days=1)
    assert isinstance(objective, MODE_BASES[mode]), source
    for attribute in REQUIRED_CLASS_ATTRIBUTES:
        assert hasattr(objective, attribute), f"{source} has no {attribute!r}"


def test_every_mode_default_is_in_the_catalog() -> None:
    assert set(DEFAULT_SOURCE) == set(OBJECTIVES)
    for mode, source in DEFAULT_SOURCE.items():
        assert source in OBJECTIVES[mode], mode


def test_is_registered_is_per_mode_and_is_known_source_spans_modes() -> None:
    assert is_registered("teacher_student", "tool_match")
    assert not is_registered("teacher_student", "loop_telemetry")
    assert not is_registered("single_model", None)
    assert is_known_source("loop_telemetry") and is_known_source("tool_match")
    assert not is_known_source("") and not is_known_source(None) and not is_known_source("cortex_judge")


def test_build_objective_picks_first_enabled_signal_registered_for_the_mode() -> None:
    client = MagicMock()
    signals = [
        {"name": "off", "source": "citation_match", "enabled": False},
        {"name": "judge", "source": "cortex_judge"},
        {"name": "other_mode", "source": "loop_telemetry"},
        {"name": "loop", "source": "loop_telemetry"},
    ]
    assert isinstance(build_objective("single_model", signals, bigquery_client=client), LoopEfficiencyObjective)
    # citation_match is disabled and loop_telemetry is single_model only, so teacher_student falls back.
    assert type(build_objective("teacher_student", signals)) is FirstToolMatchObjective
    assert type(build_objective("single_model", None, bigquery_client=client)) is ShellSuccessObjective
    skipped = [{"name": "loop", "source": "loop_telemetry", "enabled": False}]
    assert type(build_objective("single_model", skipped, bigquery_client=client)) is ShellSuccessObjective


def test_build_objective_applies_experiment_config() -> None:
    experiment = {
        "objective": {"params": {"k": 3}},
        "reflection": {"failure_label": "CUSTOM FAILURES"},
        "screening": {"high_signal": "first_tool_match"},
        "signals": [{"name": "first_tool_match", "source": "tool_match"}],
    }
    objective = build_objective("teacher_student", experiment["signals"], lookback_days=4, experiment=experiment)
    assert objective.params == {"k": 3}
    assert objective.failure_label == "CUSTOM FAILURES"
    assert objective.high_signal == "first_tool_match"
    assert objective.signal_names == ("first_tool_match",)
    assert type(objective).failure_label != "CUSTOM FAILURES"


def test_a_catalog_entry_is_selectable_from_config(tmp_path, monkeypatch) -> None:
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
    monkeypatch.setitem(OBJECTIVES["teacher_student"], "dummy_trace", Dummy)
    config = load_experiment_config(yaml)
    assert isinstance(build_objective("teacher_student", config.signals), Dummy)


@pytest.mark.parametrize("base", [TeacherStudentObjective, SingleModelObjective])
def test_subclass_missing_abstract_methods_cannot_be_instantiated(base: type) -> None:
    incomplete = type(
        "Incomplete",
        (base,),
        {"name": "incomplete", "telemetry_dimensions": ("incomplete",), "focused_bucket_type": "QUERY_CANONICAL"},
    )
    with pytest.raises(TypeError, match="abstract"):
        incomplete()


_IMPORT_ORDER_MODULES = (
    "glean_gepa.run_log",
    "glean_gepa.focused_evalset",
    "glean_gepa.objectives",
    "glean_gepa.objectives.tool_match",
    "glean_gepa.experiment_config",
    "glean_gepa.objectives.registry",
)
# Each module first, then the rest in order, plus the whole list reversed.
_IMPORT_ORDERS = [
    (first, *(module for module in _IMPORT_ORDER_MODULES if module != first)) for first in _IMPORT_ORDER_MODULES
] + [tuple(reversed(_IMPORT_ORDER_MODULES))]


@pytest.mark.parametrize("order", _IMPORT_ORDERS, ids=[order[0] for order in _IMPORT_ORDERS[:-1]] + ["reversed"])
def test_registry_is_a_leaf_module_so_any_import_order_works(order: tuple[str, ...]) -> None:
    code = "\n".join(f"import {module}" for module in order)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_reflective_example_helper_builds_the_shared_frame() -> None:
    """Objectives supply four slots; the helper owns Inputs, Metrics, defaults, and evidence caps."""
    from glean_gepa.objectives.base import REFLECTION_EVIDENCE_LIMIT

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
    assert example["Inputs"]["entry_id"] == "e1" and example["Inputs"]["eval_trace_id"] == "trace-1"
    # Partial ``generated`` is merged over empty defaults, so the reflector can always read every key.
    assert example["Generated Outputs"]["student_answer"] == "ans"
    assert example["Generated Outputs"]["teacher_answer"] == "" and example["Generated Outputs"]["teacher_tools"] == []
    assert len(example["Action Inputs"]) == REFLECTION_EVIDENCE_LIMIT
    assert example["Execution Errors"] == ["boom"] and example["Metrics"]["score"] == 0.5

    # Optional ids are omitted, not set to None.
    bare = objective.reflective_example({**trajectory, "data": {"eval_set_name": "set"}}, feedback="f")
    assert "eval_run_id" not in bare["Inputs"]
    assert bare["Action Inputs"] == [] and bare["Execution Errors"] == []
