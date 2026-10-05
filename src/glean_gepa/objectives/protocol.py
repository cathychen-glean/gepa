"""The shared contract every objective must satisfy.

``TeacherStudentObjective`` and ``SingleModelObjective`` differ in how they
fetch and score an eval, but both adapters wire the same set of members into
``ALAdapter`` and the reflection loop. This module names that shared surface
once, so a new objective author can read one file to learn what to implement,
and the registry can reject an incomplete class before a run starts.

Mode-specific members (``scored_rows``, ``analyze`` arity, pending-telemetry and
cache hooks) stay on the two ABCs in :mod:`glean_gepa.objectives.base`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

if TYPE_CHECKING:
    from glean_gepa.al_adapter import ReflectiveExample, ReflectiveExampleMetrics
    from glean_gepa.objectives.base import ScoredRow, SingleModelObjective, TeacherStudentObjective

    TelemetryObjective: TypeAlias = TeacherStudentObjective | SingleModelObjective
else:  # pragma: no cover - typing alias only
    TelemetryObjective: TypeAlias = Any


@runtime_checkable
class ObjectiveAnalysis(Protocol):
    """Result of ``objective.analyze(...)``.

    ``aggregate`` carries run-level counts and rates. The objective ``name``
    should be a float field on it so the adapter can read the primary score
    generically. ``per_entry`` maps ``entry_id`` to that entry's analysis and is
    what focused evals and reflection read.
    """

    aggregate: Any
    per_entry: Mapping[str, Any]


@runtime_checkable
class ObjectiveProtocol(Protocol):
    """Members every objective exposes, regardless of eval topology.

    Attributes
    ----------
    name
        Metric key. Appears in ``objective.composite``, ``signals[].name``,
        ``objective_scores`` on each trajectory, and the run summary.
    telemetry_dimensions
        Per-row score keys this objective emits in ``ScoredRow.dimension_scores``.
    focused_bucket_type
        How a focused eval set is bucketed; one of ``FOCUSED_BUCKET_TYPES``.
    failure_label
        Heading for the high-signal failure block in the reflection prompt.
    """

    name: str
    telemetry_dimensions: tuple[str, ...]
    focused_bucket_type: str
    failure_label: str

    # --- scoring -----------------------------------------------------------

    def focused_pass_rate(self, analysis: Any, requested_entry_ids: Sequence[str]) -> float:
        """Score in ``[0, 1]`` over ``requested_entry_ids`` only. Higher is better."""
        ...

    # --- reflection --------------------------------------------------------

    def failure_pattern(self, component_name: str, trajectory: Any) -> tuple[Any, ...]:
        """Hashable signature used to group similar failures for ``component_name``."""
        ...

    def build_reflective_example(
        self,
        component_name: str,
        trajectory: Any,
        candidate: dict[str, str],
    ) -> ReflectiveExample:
        """One reflection example for ``component_name`` from a scored trajectory."""
        ...

    def reflection_prompt(self, module_name: str) -> str:
        """Responsibility text the reflector receives for ``module_name``."""
        ...

    def format_reflective_metrics(self, metrics: ReflectiveExampleMetrics) -> str | None:
        """Render the metric line shown under each reflective example."""
        ...


@runtime_checkable
class CacheableObjective(Protocol):
    """Optional: persist analysis between runs. ``SingleModelAdapter`` calls these."""

    def cache_payload(self) -> dict[str, Any]: ...

    def load_cache(self, raw_cache: Any) -> None: ...


# Members the registry requires. Kept as data so the error message can list them.
REQUIRED_ATTRIBUTES: tuple[str, ...] = (
    "name",
    "telemetry_dimensions",
    "focused_bucket_type",
    "failure_label",
)
REQUIRED_METHODS: tuple[str, ...] = (
    "analyze",
    "focused_pass_rate",
    "entry_row",
    "aggregate_row",
    "failure_pattern",
    "build_reflective_example",
    "reflection_prompt",
    "format_reflective_metrics",
)


def check_objective_contract(cls: type) -> list[str]:
    """Return the shared-contract members ``cls`` does not define.

    Attributes count as defined when set on the class. Methods count as
    defined when present and not still abstract. An empty list means the
    class satisfies :class:`ObjectiveProtocol`.
    """
    abstract = set(getattr(cls, "__abstractmethods__", ()))
    missing: list[str] = []
    for attribute in REQUIRED_ATTRIBUTES:
        if not hasattr(cls, attribute):
            missing.append(f"attribute {attribute!r}")
    for method in REQUIRED_METHODS:
        if method in abstract or not callable(getattr(cls, method, None)):
            missing.append(f"method {method}()")
    return missing


def require_objective_contract(cls: type, *, source: str | None = None) -> None:
    """Raise ``TypeError`` naming every missing member. Intended for registration time."""
    missing = check_objective_contract(cls)
    if not missing:
        return
    label = f"{cls.__module__}.{cls.__qualname__}"
    if source:
        label = f"{label} (source={source!r})"
    raise TypeError(f"{label} does not satisfy ObjectiveProtocol; missing: {', '.join(missing)}")


def scored_rows_are_normalized(rows: Sequence[ScoredRow]) -> bool:
    """True when every dimension score is a finite float in ``[0, 1]``."""
    for row in rows:
        for value in row.dimension_scores.values():
            if isinstance(value, bool) or not isinstance(value, int | float):
                return False
            if not 0.0 <= float(value) <= 1.0:
                return False
    return True


__all__ = [
    "REQUIRED_ATTRIBUTES",
    "REQUIRED_METHODS",
    "CacheableObjective",
    "ObjectiveAnalysis",
    "ObjectiveProtocol",
    "TelemetryObjective",
    "check_objective_contract",
    "require_objective_contract",
    "scored_rows_are_normalized",
]
