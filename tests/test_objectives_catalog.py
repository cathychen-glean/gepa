"""The objective catalog is the single registration point and rejects bad entries."""

from __future__ import annotations

import pytest

from glean_gepa.objectives import registry
from glean_gepa.objectives.base import (
    MODE_DEFAULT_TELEMETRY_SOURCE,
    TELEMETRY_SOURCES,
    TeacherStudentObjective,
    build_objective,
)
from glean_gepa.objectives.registry import (
    BUILTIN_OBJECTIVES,
    VALID_MODES,
    ObjectiveRegistrationError,
    ObjectiveSpec,
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


def test_builtins_pass_validation() -> None:
    for spec in BUILTIN_OBJECTIVES:
        assert validate_objective_class(spec.load(), mode=spec.mode, source=spec.source) == [], spec.source


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


def test_resolve_error_lists_valid_sources() -> None:
    with pytest.raises(KeyError, match="valid sources for single_model: loop_telemetry, shell_telemetry"):
        registry.resolve("single_model", "downvote_feedback")


def test_build_objective_uses_catalog_default_when_no_signal_matches() -> None:
    objective = build_objective("teacher_student", [{"name": "x", "source": "not_registered"}])
    assert type(objective) is registry.resolve("teacher_student", "tool_match")


def test_spec_load_rejects_malformed_class_path() -> None:
    spec = ObjectiveSpec(mode="single_model", source="x", class_path="no_colon", summary="")
    with pytest.raises(ObjectiveRegistrationError, match="module:Class"):
        spec.load()


def test_describe_lists_every_source() -> None:
    table = registry.describe()
    for spec in BUILTIN_OBJECTIVES:
        assert spec.source in table
