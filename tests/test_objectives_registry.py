from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from glean_gepa.experiment_config import ExperimentConfigError, load_experiment_config
from glean_gepa.objectives import (
    build_objective,
    is_registered_telemetry_source,
    register_telemetry_source,
    unregister_telemetry_source,
)
from glean_gepa.objectives.shell import ShellSuccessObjective
from glean_gepa.objectives.tool_match import FirstToolMatchObjective


class _DummyTeacherStudentObjective:
    name = "dummy_alignment"
    telemetry_dimensions = ("dummy_alignment",)

    def __init__(self, *, bigquery_client=None, lookback_days: int = 1):
        self.bigquery_client = bigquery_client
        self.lookback_days = lookback_days


def test_build_objective_defaults_to_the_builtin_source_for_each_mode():
    teacher_student = build_objective("teacher_student")
    single_model = build_objective("single_model", bigquery_client=MagicMock())

    assert isinstance(teacher_student, FirstToolMatchObjective)
    assert isinstance(single_model, ShellSuccessObjective)


_DUMMY_EXPERIMENT = (
    "schema_version: 1\n"
    "mode: teacher_student\n"
    "signals:\n"
    "  - name: dummy_alignment\n"
    "    source: dummy_trace\n"
    "objective:\n"
    "  primary: dummy_alignment\n"
    "  composite:\n"
    "    dummy_alignment: 1.0\n"
)


def test_a_second_registered_source_is_constructible(tmp_path):
    # A bare stub; skip contract validation so this test only covers config wiring.
    register_telemetry_source("teacher_student", "dummy_trace", _DummyTeacherStudentObjective, validate=False)
    try:
        assert is_registered_telemetry_source("teacher_student", "dummy_trace")
        mode = tmp_path / "mode.yaml"
        mode.write_text(_DUMMY_EXPERIMENT)

        config = load_experiment_config(mode)
        objective = build_objective("teacher_student", config.signals, bigquery_client=MagicMock())

        assert config.primary_objective == "dummy_alignment"
        assert isinstance(objective, _DummyTeacherStudentObjective)
    finally:
        unregister_telemetry_source("teacher_student", "dummy_trace")


def test_unregistered_second_source_still_fails_the_load(tmp_path):
    mode = tmp_path / "mode.yaml"
    mode.write_text(_DUMMY_EXPERIMENT)

    with pytest.raises(ExperimentConfigError, match="cannot score signal"):
        load_experiment_config(mode)
