"""Every objective an experiment YAML can select, keyed by mode and ``signals[].source``.

To add an objective, implement a ``TeacherStudentObjective`` or ``SingleModelObjective``
subclass, add one entry to :data:`OBJECTIVES`, and declare a signal with that ``source``
in an experiment YAML (``configs/*.yaml``).

This module is a leaf: nothing inside ``glean_gepa.objectives`` may import it. The
objective classes import ``al_adapter``, which imports ``run_log`` and ``focused_evalset``,
which import ``glean_gepa.objectives.utils``; importing the catalog from the package
would close that cycle. Callers import it directly as ``glean_gepa.objectives.registry``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from glean_gepa.adapter_types import JudgingMode
from glean_gepa.objectives.agentic_preference import AgenticPreferenceObjective
from glean_gepa.objectives.base import SingleModelObjective, TeacherStudentObjective, configure_objective
from glean_gepa.objectives.citation_match import CitationMatchObjective
from glean_gepa.objectives.escalation_match import EscalationMatchObjective
from glean_gepa.objectives.golden_escalation import GoldenEscalationMatchObjective
from glean_gepa.objectives.loop import LoopEfficiencyObjective
from glean_gepa.objectives.shell import ShellSuccessObjective
from glean_gepa.objectives.tool_match import FirstToolMatchObjective

OBJECTIVES: dict[JudgingMode, dict[str, type[TeacherStudentObjective] | type[SingleModelObjective]]] = {
    "teacher_student": {
        "tool_match": FirstToolMatchObjective,
        "citation_match": CitationMatchObjective,
        "agentic_preference": AgenticPreferenceObjective,
        "escalation_match": EscalationMatchObjective,
    },
    "single_model": {
        "shell_telemetry": ShellSuccessObjective,
        "loop_telemetry": LoopEfficiencyObjective,
        "golden_escalation": GoldenEscalationMatchObjective,
    },
}
# Objective used when a config for the mode declares no scorable signal.
DEFAULT_SOURCE: dict[JudgingMode, str] = {"teacher_student": "tool_match", "single_model": "shell_telemetry"}


def is_registered(mode: JudgingMode, source: str | None) -> bool:
    return bool(source) and source in OBJECTIVES.get(mode, {})


def is_known_source(source: str | None) -> bool:
    """True when ``source`` is registered under any mode."""
    return bool(source) and any(source in sources for sources in OBJECTIVES.values())


def build_objective(
    mode: JudgingMode,
    signals: Sequence[Mapping[str, Any]] | None = None,
    *,
    bigquery_client: Any | None = None,
    lookback_days: int = 1,
    experiment: Mapping[str, Any] | None = None,
) -> TeacherStudentObjective | SingleModelObjective:
    """Construct the objective for the first enabled signal registered for ``mode``, else the mode default."""
    source = DEFAULT_SOURCE[mode]
    for signal in signals or ():
        if signal.get("enabled", True) is False:
            continue
        candidate = signal.get("source")
        if isinstance(candidate, str) and is_registered(mode, candidate):
            source = candidate
            break
    objective = OBJECTIVES[mode][source](bigquery_client=bigquery_client, lookback_days=lookback_days)
    configure_objective(objective, experiment)
    return objective


__all__ = ["DEFAULT_SOURCE", "OBJECTIVES", "build_objective", "is_known_source", "is_registered"]
