"""Eval-topology-agnostic scoring plugins for the Glean adapters.

The base contracts and registry live in :mod:`glean_gepa.objectives.base`; each
concrete metric is a sibling module (``tool_match``, ``citation_match``,
``loop``, ``shell``) that self-registers on import. This package re-exports the
base API so ``from glean_gepa.objectives import build_objective`` keeps working.
"""

from __future__ import annotations

from glean_gepa.objectives.base import (
    MODE_DEFAULT_PACK,
    MODE_DEFAULT_TELEMETRY_SOURCE,
    TELEMETRY_SOURCES,
    ScoredRow,
    SingleModelObjective,
    TeacherStudentObjective,
    build_objective,
    is_registered_telemetry_source,
    is_telemetry_source,
    register_telemetry_source,
    unregister_telemetry_source,
)

__all__ = [
    "MODE_DEFAULT_PACK",
    "MODE_DEFAULT_TELEMETRY_SOURCE",
    "ScoredRow",
    "SingleModelObjective",
    "TELEMETRY_SOURCES",
    "TeacherStudentObjective",
    "build_objective",
    "is_registered_telemetry_source",
    "is_telemetry_source",
    "register_telemetry_source",
    "unregister_telemetry_source",
]
