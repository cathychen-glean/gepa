"""Eval-topology-agnostic scoring plugins for the Glean adapters.

``TeacherStudentAdapter`` and ``SingleModelAdapter`` own how evals are run.
An objective owns the metric: fetching telemetry, scoring rows, high-signal
selection, and reflection. Register a new ``(mode, source)`` pair to add a
metric without forking an adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from glean_gepa.adapter_types import JudgingMode
from glean_gepa.objectives.utils.mismatch import select_mismatch_groups
from glean_gepa.prompt_constants import CORE_TOOL_KEYS
from glean_gepa.reflection_prompts import DEFAULT_MODULE_RESPONSIBILITY, core_tool_reflection_prompt

if TYPE_CHECKING:
    from gepa.core.adapter import EvaluationBatch
    from glean_gepa.adapter_types import ALRolloutOutput, ALTrajectory
    from glean_gepa.al_adapter import ReflectiveExample, ReflectiveExampleMetrics

TELEMETRY_SOURCES: dict[tuple[JudgingMode, str], type] = {}
MODE_DEFAULT_PACK: dict[JudgingMode, str] = {
    "teacher_student": "tools",
    "single_model": "shell",
}
MODE_DEFAULT_TELEMETRY_SOURCE: dict[JudgingMode, str] = {
    "teacher_student": "tool_match",
    "single_model": "shell_telemetry",
}


@dataclass(frozen=True)
class ScoredRow:
    """One adapter-facing score row produced from an objective's analysis."""

    entry_id: str | None
    dimension_scores: dict[str, float]
    output: Mapping[str, Any]
    data_overrides: Mapping[str, Any] = field(default_factory=dict)


def module_responsibility(
    module_name: str,
    *,
    responsibilities: Mapping[str, str],
) -> str:
    """Resolve the responsibility text the reflector gets for ``module_name``."""
    responsibility = responsibilities.get(module_name)
    if responsibility is not None:
        return responsibility
    if module_name in CORE_TOOL_KEYS:
        return core_tool_reflection_prompt(module_name)
    return DEFAULT_MODULE_RESPONSIBILITY


class PackConfigurable:
    """Pack YAML knobs overlaid onto an objective instance after construction."""

    params: dict[str, Any]

    def pack_param(self, key: str, default: Any) -> Any:
        return (getattr(self, "params", None) or {}).get(key, default)

    def reflection_prompt(self, module_name: str) -> str:
        return module_responsibility(
            module_name,
            responsibilities=getattr(self, "module_responsibilities", {}) or {},
        )


def configure_objective(objective: Any, pack: Mapping[str, Any] | None) -> None:
    """Apply pack YAML knobs. Class attributes stay the unconfigured defaults."""
    if not hasattr(objective, "params"):
        objective.params = {}
    if not pack:
        return
    objective_cfg = pack.get("objective") or {}
    reflection = pack.get("reflection") or {}
    bucket = objective_cfg.get("focused_bucket_type")
    if bucket is not None:
        from glean_gepa.focused_evalset import FOCUSED_BUCKET_TYPES

        if str(bucket) not in FOCUSED_BUCKET_TYPES:
            allowed = ", ".join(sorted(FOCUSED_BUCKET_TYPES))
            raise ValueError(f"focused_bucket_type must be one of {allowed}, got {bucket!r}")
        objective.focused_bucket_type = str(bucket)
    params = objective_cfg.get("params")
    if isinstance(params, Mapping):
        objective.params = dict(params)
    label = reflection.get("failure_label")
    if label is not None:
        objective.failure_label = str(label)
    title = reflection.get("report_title")
    if title is not None and hasattr(type(objective), "reflection_report_title"):
        objective.reflection_report_title = str(title)
    modules = reflection.get("modules")
    if isinstance(modules, Mapping) and modules:
        base = dict(getattr(type(objective), "module_responsibilities", {}) or {})
        base.update({str(name): str(text) for name, text in modules.items()})
        objective.module_responsibilities = base


class TeacherStudentObjective(PackConfigurable, ABC):
    """Paired teacher-vs-student trace comparison."""

    name: str
    telemetry_dimensions: tuple[str, ...]
    focused_bucket_type: str
    failure_label: str = "HIGH-SIGNAL FAILURES"
    reflection_report_title: str = "REFLECTION: teacher vs student traces"
    teacher_compared_key: str
    student_compared_key: str
    mismatch_pair: Callable[[Any, Any], tuple[str, str] | None]
    module_responsibilities: ClassVar[Mapping[str, str]] = {}
    bigquery_client: Any | None = None
    evalcli: Any | None = None
    lookback_days: int = 1
    _paired_analysis_cache: dict[tuple[str, str], Any]

    @property
    def analysis_cache(self) -> dict[tuple[str, str], Any]:
        """Objective-agnostic accessor for the paired-analysis cache."""
        return getattr(self, "_paired_analysis_cache", {})

    @abstractmethod
    def analyze(self, teacher_eval_id: str, student_eval_id: str) -> Any: ...

    def cached_paired_analysis(
        self,
        teacher_eval_id: str,
        student_eval_id: str,
        *,
        cache: dict[tuple[str, str], Any],
        fetch: Callable[..., Any],
        empty: Callable[[str, str], Any],
        label: str,
    ) -> Any:
        """Fetch (or reuse a cached) paired analysis with shared HIT/MISS logging.

        ``fetch`` and ``empty`` are passed in from the concrete objective's module
        so unit tests can still patch the module-level fetch function.
        """
        cache_key = (teacher_eval_id, student_eval_id)
        cached = cache.get(cache_key)
        if cached is not None:
            print(f"[Cache HIT] Using cached {label} for {teacher_eval_id} vs {student_eval_id}")
            return cached
        if self.bigquery_client is None:
            analysis = empty(teacher_eval_id, student_eval_id)
        else:
            analysis = fetch(
                self.bigquery_client,
                teacher_eval_id=teacher_eval_id,
                student_eval_id=student_eval_id,
                lookback_days=self.lookback_days,
                evalcli=self.evalcli,
            )
        cache[cache_key] = analysis
        print(f"[Cache MISS] Fetched {label} for {teacher_eval_id} vs {student_eval_id}")
        return analysis

    @abstractmethod
    def validate_full_eval(self, analysis: Any) -> None: ...

    @abstractmethod
    def focused_pass_rate(self, analysis: Any, requested_entry_ids: Sequence[str]) -> float: ...

    @abstractmethod
    def scored_rows(
        self,
        analysis: Any,
        *,
        focused: bool,
        capture_traces: bool,
        query: str,
        deployment_id: str,
    ) -> list[ScoredRow]: ...

    def is_high_signal(self, output: Mapping[str, Any]) -> bool:
        return self._mismatch_key(output) is not None

    def _mismatch_key(self, output: Mapping[str, Any]) -> tuple[str, str] | None:
        """Signature of the teacher/student divergence in ``output``, or ``None`` when aligned."""
        return type(self).mismatch_pair(
            output.get(self.teacher_compared_key),
            output.get(self.student_compared_key),
        )

    def _select_mismatch_groups(
        self, mismatch_keys: Sequence[tuple[str, str] | None]
    ) -> tuple[list[int], list[tuple[str, str, int]]]:
        return select_mismatch_groups(mismatch_keys)

    def _component_trajectories(
        self,
        component_name: str,
        selected: list[Any],
        selected_keys: list[tuple[str, str] | None],
        *,
        trajectories: list[Any],
        mismatch_keys: list[tuple[str, str] | None],
    ) -> list[Any]:
        """Trajectories to reflect on for ``component_name`` (default: all selected)."""
        del component_name, selected_keys, trajectories, mismatch_keys
        return selected

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[ALTrajectory, ALRolloutOutput],
        components_to_update: list[str],
        build_example: Callable[[str, Any, dict[str, str]], ReflectiveExample],
    ) -> dict[str, list[ReflectiveExample]]:
        """Reflect on the most frequent teacher/student mismatch groups in the batch."""
        from glean_gepa.run_log import (
            format_eval_entry_report,
            format_high_signal_selection_report,
            log_section,
            selected_entry_ids_from_examples,
        )

        if not eval_batch.trajectories:
            return {comp: [] for comp in components_to_update}

        trajectories = list(eval_batch.trajectories)
        mismatch_keys = [self._mismatch_key(trajectory["output"]) for trajectory in trajectories]
        selected_indices, selected_groups = self._select_mismatch_groups(mismatch_keys)
        selected = [trajectories[index] for index in selected_indices]
        selected_keys = [mismatch_keys[index] for index in selected_indices]
        examples: dict[str, list[ReflectiveExample]] = {}
        for component_name in components_to_update:
            chosen = self._component_trajectories(
                component_name,
                selected,
                selected_keys,
                trajectories=trajectories,
                mismatch_keys=mismatch_keys,
            )
            examples[component_name] = [build_example(component_name, trajectory, candidate) for trajectory in chosen]
        mismatch_count = sum(key is not None for key in mismatch_keys)
        selected_entry_ids = [
            str(trajectory["output"].get("entry_id", ""))
            for trajectory in selected
            if trajectory["output"].get("entry_id")
        ]
        log_section(self.reflection_report_title, format_eval_entry_report(trajectories))
        log_section(
            "REFLECTION: high-signal dataset",
            format_high_signal_selection_report(
                selected_groups=selected_groups,
                selected_entry_ids=selected_entry_ids,
                selected_count=len(selected_indices),
                total_mismatch_count=mismatch_count,
                module_entry_ids={
                    module: selected_entry_ids_from_examples(module_examples)
                    for module, module_examples in examples.items()
                },
            ),
        )
        return examples

    @abstractmethod
    def failure_pattern(self, component_name: str, trajectory: Any) -> tuple[Any, ...]: ...

    @abstractmethod
    def build_reflective_example(
        self,
        component_name: str,
        trajectory: Any,
        candidate: dict[str, str],
    ) -> ReflectiveExample: ...

    def format_reflective_metrics(self, metrics: ReflectiveExampleMetrics) -> str | None:
        return None

    def high_signal_core_tool_keys(self, trajectories: Sequence[Any] | None) -> list[str]:
        del trajectories
        return []


class TelemetryPendingError(RuntimeError):
    """Raised when an eval has no scorable telemetry yet; callers should retry, not score 0/0."""


class SingleModelObjective(PackConfigurable, ABC):
    """Student-only BigQuery / agentspan metric."""

    name: str
    telemetry_dimensions: tuple[str, ...]
    focused_bucket_type: str
    failure_label: str = "HIGH-SIGNAL FAILURES"
    # Human-readable telemetry name for pending/read logs; empty falls back to ``name``.
    pending_telemetry_label: str = ""
    module_responsibilities: ClassVar[Mapping[str, str]] = {}

    @abstractmethod
    def analyze(
        self,
        eval_id: str,
        *,
        include_error_examples: bool = True,
        include_per_entry: bool = True,
        evalcli: Any | None = None,
        include_action_inputs: bool = True,
    ) -> Any: ...

    @abstractmethod
    def is_pending(self, analysis: Any) -> bool: ...

    @abstractmethod
    def aggregate_score(self, analysis: Any) -> float: ...

    @abstractmethod
    def focused_pass_rate(self, analysis: Any, requested_entry_ids: Sequence[str]) -> float: ...

    @abstractmethod
    def entry_ids_to_score(self, analysis: Any, requested_entry_ids: Sequence[str] | None) -> tuple[str, ...]: ...

    @abstractmethod
    def log_analysis(self, analysis: Any) -> None: ...

    @abstractmethod
    def scored_rows(
        self,
        analysis: Any,
        *,
        al_data_inst: Mapping[str, Any],
        student_eval_id: str,
        eval_set_name: str,
        eval_set_version: str,
        deployment_ids: Sequence[str],
        requested_entry_ids: Sequence[str] | None,
        is_focused_eval: bool,
        capture_traces: bool,
    ) -> list[ScoredRow]: ...

    @abstractmethod
    def prepare_focused_source_entries(
        self,
        *,
        eval_set_name: str,
        eval_set_version: str,
        eval_run_id: str,
        entry_ids: Sequence[str],
        deployment_ids: Sequence[str],
    ) -> list[dict[str, Any]] | None: ...

    @abstractmethod
    def failure_pattern(self, component_name: str, trajectory: Any) -> tuple[Any, ...]: ...

    @abstractmethod
    def build_reflective_example(
        self,
        component_name: str,
        trajectory: Any,
        candidate: dict[str, str],
    ) -> ReflectiveExample: ...

    def format_reflective_metrics(self, metrics: ReflectiveExampleMetrics) -> str | None:
        return None

    def cache_payload(self) -> dict[str, Any]:
        return {}

    def load_cache(self, raw_cache: Any) -> None:
        del raw_cache


def register_telemetry_source(mode: JudgingMode, source: str, cls: type) -> None:
    TELEMETRY_SOURCES[(mode, source)] = cls


def unregister_telemetry_source(mode: JudgingMode, source: str) -> None:
    TELEMETRY_SOURCES.pop((mode, source), None)


def _ensure_builtin_objectives_registered() -> None:
    from glean_gepa.objectives.citation_match import CitationMatchObjective
    from glean_gepa.objectives.loop import LoopEfficiencyObjective
    from glean_gepa.objectives.shell import ShellSuccessObjective
    from glean_gepa.objectives.tool_match import FirstToolMatchObjective

    TELEMETRY_SOURCES.setdefault(("teacher_student", "tool_match"), FirstToolMatchObjective)
    TELEMETRY_SOURCES.setdefault(("teacher_student", "citation_match"), CitationMatchObjective)
    TELEMETRY_SOURCES.setdefault(("single_model", "shell_telemetry"), ShellSuccessObjective)
    TELEMETRY_SOURCES.setdefault(("single_model", "loop_telemetry"), LoopEfficiencyObjective)


def is_registered_telemetry_source(mode: JudgingMode, source: str | None) -> bool:
    if not source:
        return False
    _ensure_builtin_objectives_registered()
    return (mode, source) in TELEMETRY_SOURCES


def is_telemetry_source(source: str | None) -> bool:
    if not source:
        return False
    _ensure_builtin_objectives_registered()
    return any(registered_source == source for _mode, registered_source in TELEMETRY_SOURCES)


def build_objective(
    mode: JudgingMode,
    signals: Sequence[Mapping[str, Any]] | None = None,
    *,
    bigquery_client: Any | None = None,
    lookback_days: int = 1,
    pack: Mapping[str, Any] | None = None,
) -> TeacherStudentObjective | SingleModelObjective:
    """Construct the telemetry objective registered for ``mode`` and the pack source."""
    _ensure_builtin_objectives_registered()
    source = MODE_DEFAULT_TELEMETRY_SOURCE[mode]
    if signals:
        for signal in signals:
            if signal.get("enabled", True) is False:
                continue
            candidate = signal.get("source")
            if isinstance(candidate, str) and is_registered_telemetry_source(mode, candidate):
                source = candidate
                break
    cls = TELEMETRY_SOURCES[(mode, source)]
    objective = cls(bigquery_client=bigquery_client, lookback_days=lookback_days)
    configure_objective(objective, pack)
    return objective


__all__ = [
    "MODE_DEFAULT_PACK",
    "MODE_DEFAULT_TELEMETRY_SOURCE",
    "ScoredRow",
    "SingleModelObjective",
    "TELEMETRY_SOURCES",
    "TeacherStudentObjective",
    "TelemetryPendingError",
    "build_objective",
    "configure_objective",
    "is_registered_telemetry_source",
    "is_telemetry_source",
    "register_telemetry_source",
    "unregister_telemetry_source",
]
