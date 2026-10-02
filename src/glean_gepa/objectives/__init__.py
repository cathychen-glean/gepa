"""Eval-topology-agnostic scoring plugins for the Glean adapters.

The base contracts live in :mod:`glean_gepa.objectives.base` and the shared
member contract in :mod:`glean_gepa.objectives.protocol`. Every objective is
listed once in ``registry.BUILTIN_OBJECTIVES``; add a row there to register a
new one. This package re-exports the base API so
``from glean_gepa.objectives import build_objective`` keeps working.
"""

from __future__ import annotations

from glean_gepa.objectives.base import (
    MODE_DEFAULT_TELEMETRY_SOURCE,
    TELEMETRY_SOURCES,
    AnalysisDetail,
    AnalysisRequest,
    ScoredRow,
    ScoringContext,
    SingleModelObjective,
    TeacherStudentObjective,
    TelemetryPendingError,
    build_objective,
    configure_objective,
    is_registered_telemetry_source,
    is_telemetry_source,
    register_telemetry_source,
    unregister_telemetry_source,
)

__all__ = [
    "MODE_DEFAULT_TELEMETRY_SOURCE",
    "AnalysisDetail",
    "AnalysisRequest",
    "ScoredRow",
    "ScoringContext",
    "SingleModelObjective",
    "TelemetryPendingError",
    "TELEMETRY_SOURCES",
    "TeacherStudentObjective",
    "build_objective",
    "configure_objective",
    "is_registered_telemetry_source",
    "is_telemetry_source",
    "register_telemetry_source",
    "unregister_telemetry_source",
]
