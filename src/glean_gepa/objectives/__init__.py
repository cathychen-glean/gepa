"""Eval-topology-agnostic scoring plugins for the Glean adapters.

The two abstract bases, ``TeacherStudentObjective`` and ``SingleModelObjective``, and the
types they share live in :mod:`glean_gepa.objectives.base`. Every selectable objective is
one entry in ``OBJECTIVES`` in :mod:`glean_gepa.objectives.registry`, which also holds
``build_objective``.
"""

from __future__ import annotations

# Never import ``registry`` here: it imports the objective classes, which import al_adapter, which
# imports this package through run_log and focused_evalset.
from glean_gepa.objectives.base import (
    AnalysisDetail,
    AnalysisRequest,
    ScoredRow,
    ScoringContext,
    SingleModelObjective,
    TeacherStudentObjective,
    TelemetryPendingError,
    configure_objective,
)

__all__ = [
    "AnalysisDetail",
    "AnalysisRequest",
    "ScoredRow",
    "ScoringContext",
    "SingleModelObjective",
    "TelemetryPendingError",
    "TeacherStudentObjective",
    "configure_objective",
]
