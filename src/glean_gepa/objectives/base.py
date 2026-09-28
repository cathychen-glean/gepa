"""Eval-topology-agnostic scoring plugins for the Glean adapters.

``TeacherStudentAdapter`` and ``SingleModelAdapter`` own how evals are run.
An objective owns the metric: fetching telemetry, scoring rows, high-signal
selection, and reflection. Register a new ``(mode, source)`` pair to add a
metric without forking an adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, cast

import glean_gepa.objectives.registry as _registry
from glean_gepa.adapter_types import JudgingMode
from glean_gepa.objectives.utils.mismatch import REFLECTION_HIGH_SIGNAL_ENTRY_LIMIT, select_mismatch_groups
from glean_gepa.prompt_constants import CORE_TOOL_KEYS
from glean_gepa.reflection_prompts import DEFAULT_MODULE_RESPONSIBILITY, core_tool_reflection_prompt

if TYPE_CHECKING:
    from gepa.core.adapter import EvaluationBatch
    from glean_gepa.adapter_types import ALRolloutOutput, ALTrajectory
    from glean_gepa.al_adapter import ReflectiveExample, ReflectiveExampleMetrics

# ``make_reflective_dataset(k=None)`` means YAML ``all``; omitting ``k`` uses the
# objective's ``reflection_entry_limit``.
_REFLECTION_K_DEFAULT = object()

# Reflection surfaces at most this many tool payloads / error strings per example.
REFLECTION_EVIDENCE_LIMIT = 5

# The catalog and registry live in ``glean_gepa.objectives.registry``. These
# names are kept so existing imports keep working; they read from that module.
TELEMETRY_SOURCES: dict[tuple[JudgingMode, str], type] = _registry._REGISTRY
MODE_DEFAULT_PACK: dict[JudgingMode, str] = {
    spec.mode: spec.default_pack for spec in _registry.BUILTIN_OBJECTIVES if spec.default_pack
}
MODE_DEFAULT_TELEMETRY_SOURCE: dict[JudgingMode, str] = {
    spec.mode: spec.source for spec in _registry.BUILTIN_OBJECTIVES if spec.default_pack
}


AnalysisDetail = Literal["aggregate", "per_entry", "traces"]


@dataclass(frozen=True)
class AnalysisRequest:
    """What the adapter needs from one ``objective.analyze(...)`` call.

    Built once per fetch by the adapter. Objectives read it; they are not
    mutated between calls.

    Attributes
    ----------
    evalcli
        Client for eval-analysis views and detailed traces. ``None`` skips
        every evalcli-backed enrichment.
    detail
        How much to fetch. ``aggregate`` is a run-level number for validation.
        ``per_entry`` adds per-entry rows for focused evals. ``traces`` adds
        per-entry evidence (error examples, tool payloads) for reflection.
        An objective that has one fetch shape may ignore this.
    hydrate_action_inputs
        Attach tool payloads from detailed traces to high-signal entries.
        False for validation-only batches, which never feed reflection.
    """

    evalcli: Any | None = None
    detail: AnalysisDetail = "per_entry"
    hydrate_action_inputs: bool = True

    @property
    def wants_per_entry(self) -> bool:
        return self.detail != "aggregate"

    @property
    def wants_traces(self) -> bool:
        return self.detail == "traces"


@dataclass(frozen=True)
class ScoredRow:
    """One adapter-facing score row produced from an objective's analysis."""

    entry_id: str | None
    dimension_scores: dict[str, float]
    output: Mapping[str, Any]
    data_overrides: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoringContext:
    """Batch-level facts every ``ScoredRow`` in one eval shares.

    Built by the base ``scored_rows`` and handed to the ``entry_row`` /
    ``aggregate_row`` hooks so objectives do not re-derive them.

    Attributes
    ----------
    query
        ``"{eval_set_name}:{eval_set_version}"``. Also the ``entry_id`` of an
        aggregate row that must be keyed to the eval set.
    deployment_id
        First configured deployment, or ``""``.
    is_focused
        The batch is a focused high-signal eval, so per-entry scores are
        pass/fail rather than rates.
    capture_traces
        Rows will become reflection trajectories; attach evidence.
    student_eval_id
        Student eval run id. Single-model only; teacher/student analyses carry
        their run ids on the analysis object.
    """

    query: str
    deployment_id: str
    is_focused: bool
    capture_traces: bool
    student_eval_id: str = ""

    def entry_query(self, entry_id: str) -> str:
        return f"{self.query} entry={entry_id}"


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
    # Set from ``screening.high_signal`` and ``signals[].name`` by ``configure_objective``.
    high_signal: str | None = None
    signal_names: tuple[str, ...] = ()

    def pack_param(self, key: str, default: Any) -> Any:
        return (getattr(self, "params", None) or {}).get(key, default)

    def format_reflective_metrics(self, metrics: ReflectiveExampleMetrics) -> str | None:
        """Render ``score``, ``screening.high_signal``, then each other ``signals.name``.

        A signal is omitted when that metric was not scored. No ``high_signal`` omits the line.
        """
        high_signal = self.high_signal
        if not high_signal:
            return None
        parts = [
            f"score={metrics['score']:.2f}",
            f"{high_signal}={metrics.get(high_signal, metrics['score']):.2f}",
        ]
        names = self.signal_names
        seen = {high_signal}
        for name in names:
            if name in seen:
                continue
            other = metrics.get(name)
            if other is None:
                continue
            parts.append(f"{name}={other:.2f}")
            seen.add(name)
        return ", ".join(parts)

    def reflective_metrics(self, trajectory: Mapping[str, Any]) -> ReflectiveExampleMetrics:
        """Score plus each signal this run is configured to report.

        The objective's own metric is always included. Any other ``signals.name``
        from the pack YAML, such as a pairwise correctness judge, is included
        only when that trajectory actually scored it.
        """
        from glean_gepa.al_adapter import ReflectiveExampleMetrics

        objective_scores = trajectory.get("objective_scores") or {}
        metrics: dict[str, float] = {"score": float(trajectory["score"])}
        objective_name = getattr(self, "name", None)
        ordered: list[str] = []
        if isinstance(objective_name, str) and objective_name:
            ordered.append(objective_name)
        if self.high_signal and self.high_signal not in ordered:
            ordered.append(self.high_signal)
        for name in self.signal_names:
            if name not in ordered:
                ordered.append(name)
        for name in ordered:
            raw = objective_scores.get(name)
            if raw is None and name == objective_name:
                raw = trajectory.get("score")
            if isinstance(raw, bool) or not isinstance(raw, int | float):
                continue
            metrics[name] = float(raw)
        return cast(ReflectiveExampleMetrics, metrics)

    def reflective_example(
        self,
        trajectory: Mapping[str, Any],
        *,
        feedback: str,
        generated: Mapping[str, Any] | None = None,
        action_inputs: Sequence[str] = (),
        execution_errors: Sequence[str] = (),
    ) -> ReflectiveExample:
        """Assemble one ``ReflectiveExample`` from the parts an objective decides.

        The frame (``Inputs``, ``Metrics``, evidence caps) is the same for every
        objective; ``build_reflective_example`` supplies the four slots here.
        ``generated`` fills the ``Generated Outputs`` block; missing keys default
        to empty so a student-only objective can omit it.
        """
        from glean_gepa.al_adapter import ReflectiveExampleInputs, ReflectiveExampleOutputs

        output = trajectory["output"]
        data = trajectory["data"]
        inputs: ReflectiveExampleInputs = {
            "eval_set": data["eval_set_name"],
            "entry_id": output["entry_id"],
            "deployment_id": output["deployment_id"],
            "query": output["query"],
        }
        if eval_run_id := data.get("eval_run_id"):
            inputs["eval_run_id"] = eval_run_id
        if eval_trace_id := data.get("eval_trace_id"):
            inputs["eval_trace_id"] = eval_trace_id
        outputs: ReflectiveExampleOutputs = {
            "student_answer": "",
            "teacher_answer": "",
            "student_tools": [],
            "teacher_tools": [],
        }
        if generated:
            outputs.update(cast(Any, generated))
        return {
            "Inputs": inputs,
            "Generated Outputs": outputs,
            "Action Inputs": list(action_inputs)[:REFLECTION_EVIDENCE_LIMIT],
            "Execution Errors": list(execution_errors)[:REFLECTION_EVIDENCE_LIMIT],
            "Feedback": feedback,
            "Metrics": self.reflective_metrics(trajectory),
        }

    def wired_signal_issues(self, objective_scores: Mapping[str, Any]) -> list[str]:
        """Feedback lines for wired judge signals that scored below their floor.

        The objective's own metric is omitted; its example already describes that failure.
        """
        from glean_gepa.judge_metrics_util import JUDGE_SPECS

        objective_name = getattr(self, "name", None)
        parts: list[str] = []
        for name in self.signal_names:
            if name == objective_name:
                continue
            spec = JUDGE_SPECS.get(name)
            raw = objective_scores.get(name)
            if spec is None or isinstance(raw, bool) or not isinstance(raw, int | float):
                continue
            score = float(raw)
            if score < spec.default_min:
                parts.append(f"{name.replace('_', ' ').capitalize()} issue: score={score:.2f}.")
        return parts

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
    screening = pack.get("screening") or {}
    high_signal = screening.get("high_signal")
    if high_signal:
        objective.high_signal = str(high_signal)
    signals = pack.get("signals")
    if isinstance(signals, Sequence) and not isinstance(signals, str | bytes):
        objective.signal_names = tuple(
            str(signal["name"]) for signal in signals if isinstance(signal, Mapping) and signal.get("name")
        )


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
    reflection_entry_limit: ClassVar[int] = REFLECTION_HIGH_SIGNAL_ENTRY_LIMIT
    bigquery_client: Any | None = None
    # Set once by the adapter after construction; used by reflection-time hydration.
    evalcli: Any | None = None
    lookback_days: int = 1
    _paired_analysis_cache: dict[tuple[str, str], Any]
    # Pairs stored from a call that did not request action inputs. A seeded
    # cache entry is absent here and is treated as hydrated.
    _unhydrated_pairs: set[tuple[str, str]]

    @property
    def analysis_cache(self) -> dict[tuple[str, str], Any]:
        """Objective-agnostic accessor for the paired-analysis cache."""
        return getattr(self, "_paired_analysis_cache", {})

    @abstractmethod
    def analyze(self, teacher_eval_id: str, student_eval_id: str, *, request: AnalysisRequest) -> Any: ...

    def _unhydrated_pair_keys(self) -> set[tuple[str, str]]:
        pairs = getattr(self, "_unhydrated_pairs", None)
        if not isinstance(pairs, set):
            pairs = set()
            self._unhydrated_pairs = pairs
        return pairs

    def analysis_is_cacheable(self, analysis: Any) -> bool:
        """False for a provisional empty comparison, so a later call can fetch again."""
        aggregate = getattr(analysis, "aggregate", None)
        if aggregate is not None and hasattr(aggregate, "compared_entries"):
            return aggregate.compared_entries > 0
        return bool(getattr(analysis, "per_entry", None))

    def cached_paired_analysis(
        self,
        teacher_eval_id: str,
        student_eval_id: str,
        *,
        request: AnalysisRequest,
        cache: dict[tuple[str, str], Any],
        fetch: Callable[..., Any],
        empty: Callable[[str, str], Any],
        label: str,
    ) -> Any:
        """Fetch (or reuse a cached) paired analysis with shared HIT/MISS logging.

        ``fetch`` and ``empty`` are passed in from the concrete objective's module
        so unit tests can still patch the module-level fetch function.

        An empty comparison is not stored. An empty refetch leaves any entry
        already cached in place and returns that entry. A request without
        ``hydrate_action_inputs`` marks the pair unhydrated; a later call that
        requests action inputs fetches again and replaces that entry.
        """
        cache_key = (teacher_eval_id, student_eval_id)
        unhydrated = self._unhydrated_pair_keys()
        cached = cache.get(cache_key)
        if cached is not None and (not request.hydrate_action_inputs or cache_key not in unhydrated):
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
                evalcli=request.evalcli,
                include_action_inputs=request.hydrate_action_inputs,
            )
        if not self.analysis_is_cacheable(analysis):
            print(f"[Cache] Not caching provisional empty {label} for {teacher_eval_id} vs {student_eval_id}")
            return cache.get(cache_key, analysis)
        cache[cache_key] = analysis
        if request.hydrate_action_inputs:
            unhydrated.discard(cache_key)
        else:
            unhydrated.add(cache_key)
        print(f"[Cache MISS] Fetched {label} for {teacher_eval_id} vs {student_eval_id}")
        return analysis

    def require_compared_entries(self, analysis: Any) -> None:
        """Reject a 0/0 comparison. Objectives without a compared-entry count do nothing."""
        del analysis

    @abstractmethod
    def validate_full_eval(self, analysis: Any) -> None: ...

    @abstractmethod
    def focused_pass_rate(self, analysis: Any, requested_entry_ids: Sequence[str]) -> float: ...

    def scored_rows(
        self,
        analysis: Any,
        *,
        focused: bool,
        capture_traces: bool,
        query: str,
        deployment_id: str,
    ) -> list[ScoredRow]:
        """Rows for one paired eval. Objectives implement ``entry_row`` and ``aggregate_row``.

        - Focused eval with no telemetry: nothing to score, return ``[]``.
        - Validation (not focused, no traces): one aggregate row, ``entry_id=None``.
        - Otherwise: one row per compared entry.
        """
        ctx = ScoringContext(
            query=query, deployment_id=deployment_id, is_focused=focused, capture_traces=capture_traces
        )
        if focused and not analysis.per_entry:
            return []
        if not focused and not capture_traces:
            return [self.aggregate_row(analysis, ctx)]
        return [self.entry_row(entry_id, metrics, analysis, ctx) for entry_id, metrics in analysis.per_entry.items()]

    @abstractmethod
    def entry_row(self, entry_id: str, metrics: Any, analysis: Any, ctx: ScoringContext) -> ScoredRow:
        """Score and rollout output for one compared entry. ``ScoredRow.entry_id`` must be ``entry_id``."""

    @abstractmethod
    def aggregate_row(self, analysis: Any, ctx: ScoringContext) -> ScoredRow:
        """Run-level score for validation batches. ``ScoredRow.entry_id`` must be ``None``."""

    def is_high_signal(self, output: Mapping[str, Any]) -> bool:
        return self._mismatch_key(output) is not None

    def _mismatch_key(self, output: Mapping[str, Any]) -> tuple[str, str] | None:
        """Signature of the teacher/student divergence in ``output``, or ``None`` when aligned."""
        return type(self).mismatch_pair(
            output.get(self.teacher_compared_key),
            output.get(self.student_compared_key),
        )

    def _select_mismatch_groups(
        self,
        mismatch_keys: Sequence[tuple[str, str] | None],
        *,
        trajectories: Sequence[Any] = (),
        max_entries: int | None = REFLECTION_HIGH_SIGNAL_ENTRY_LIMIT,
    ) -> tuple[list[int], list[tuple[str, str, int]]]:
        """Pick the entries to reflect on, by descending mismatch-group frequency."""
        del trajectories
        if max_entries is None:
            max_entries = len(mismatch_keys)
        return select_mismatch_groups(mismatch_keys, max_entries=max_entries)

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

    def hydrate_reflective_trajectories(self, selected: list[Any]) -> None:
        """Add per-entry context to the trajectories reflection is about to read."""
        del selected

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[ALTrajectory, ALRolloutOutput],
        components_to_update: list[str],
        build_example: Callable[[str, Any, dict[str, str]], ReflectiveExample],
        k: Any = _REFLECTION_K_DEFAULT,
    ) -> dict[str, list[ReflectiveExample]]:
        """Reflect on high-signal teacher/student mismatches in the batch.

        ``k`` is ``search.reflection_samples``: an integer caps the set, ``None``
        (YAML ``all``) keeps every mismatch. Omit it to use ``reflection_entry_limit``.
        """
        from glean_gepa.run_log import (
            format_eval_entry_report,
            format_high_signal_selection_report,
            log_section,
            selected_entry_ids_from_examples,
        )

        if not eval_batch.trajectories:
            return {comp: [] for comp in components_to_update}

        if k is _REFLECTION_K_DEFAULT:
            max_entries: int | None = getattr(type(self), "reflection_entry_limit", REFLECTION_HIGH_SIGNAL_ENTRY_LIMIT)
        else:
            max_entries = k
        trajectories = list(eval_batch.trajectories)
        mismatch_keys = [self._mismatch_key(trajectory["output"]) for trajectory in trajectories]
        selected_indices, selected_groups = self._select_mismatch_groups(
            mismatch_keys,
            trajectories=trajectories,
            max_entries=max_entries,
        )
        selected = [trajectories[index] for index in selected_indices]
        selected_keys = [mismatch_keys[index] for index in selected_indices]
        self.hydrate_reflective_trajectories(selected)
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
                cap=max_entries if max_entries is not None else mismatch_count,
                justification=getattr(self, "reflection_selection_justification", None),
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
    # Aggregate count that stays 0 until scorable telemetry has landed.
    pending_count: str
    module_responsibilities: ClassVar[Mapping[str, str]] = {}
    # Per-eval cache state. Read through ``analysis_cache`` / ``unhydrated_eval_ids``,
    # which create these lazily so concrete ``__init__`` need not declare them.
    _eval_analysis_cache: dict[str, Any]
    _unhydrated_eval_ids: set[str]

    @abstractmethod
    def analyze(self, eval_id: str, *, request: AnalysisRequest) -> Any: ...

    # --- shared per-eval cache -------------------------------------------

    @property
    def analysis_cache(self) -> dict[str, Any]:
        return self.__dict__.setdefault("_eval_analysis_cache", {})

    @property
    def unhydrated_eval_ids(self) -> set[str]:
        return self.__dict__.setdefault("_unhydrated_eval_ids", set())

    def cache_hit_is_sufficient(self, cached: Any, request: AnalysisRequest) -> bool:
        """Can ``cached`` serve ``request`` without a refetch? Override to demand more detail."""
        del cached, request
        return True

    def analysis_is_cacheable(self, analysis: Any, request: AnalysisRequest) -> bool:
        """False for a provisional result, so a later call fetches again."""
        del request
        return not self.is_pending(analysis)

    def cached_eval_analysis(
        self,
        eval_id: str,
        *,
        request: AnalysisRequest,
        fetch: Callable[[AnalysisRequest], Any],
        label: str,
    ) -> Any:
        """Fetch (or reuse a cached) analysis for one eval with shared HIT/MISS logging.

        ``fetch`` is called with the request only; the objective's closure supplies
        its own client and parameters so tests can still patch the module fetch.

        Rules, mirroring ``TeacherStudentObjective.cached_paired_analysis``:

        - A hit is served unless ``cache_hit_is_sufficient`` says no, or the entry
          was stored without action inputs and this request wants them.
        - A result that is not cacheable is returned but not stored; if an entry
          was already cached, that entry is returned instead.
        - Storing records whether action inputs were hydrated so a later
          hydrating request refetches rather than serving payload-less entries.
        """
        cache = self.analysis_cache
        unhydrated = self.unhydrated_eval_ids
        cached = cache.get(eval_id)
        if cached is not None:
            needs_hydration = request.hydrate_action_inputs and eval_id in unhydrated
            if not needs_hydration and self.cache_hit_is_sufficient(cached, request):
                print(f"[Cache HIT] Using cached {label} for eval_id: {eval_id}")
                return cached
            reason = "with action inputs" if needs_hydration else "with more detail"
            print(f"[Cache] Refetching {label} {reason} for eval_id: {eval_id}")
        analysis = fetch(request)
        if not self.analysis_is_cacheable(analysis, request):
            print(f"[Cache] Not caching provisional {label} for eval_id: {eval_id}")
            return cached if cached is not None else analysis
        cache[eval_id] = analysis
        if request.hydrate_action_inputs:
            unhydrated.discard(eval_id)
        else:
            unhydrated.add(eval_id)
        return analysis

    def is_pending(self, analysis: Any) -> bool:
        """Telemetry has not landed while ``pending_count`` on the aggregate is 0."""
        return getattr(analysis.aggregate, self.pending_count) == 0

    def aggregate_score(self, analysis: Any) -> float:
        """The objective ``name`` is the float field on ``analysis.aggregate``."""
        return float(getattr(analysis.aggregate, self.name))

    @abstractmethod
    def focused_pass_rate(self, analysis: Any, requested_entry_ids: Sequence[str]) -> float: ...

    def entry_ids_to_score(self, analysis: Any, requested_entry_ids: Sequence[str] | None) -> tuple[str, ...]:
        """Which entries get a per-entry ``ScoredRow``.

        Focused eval (``requested_entry_ids`` given): the requested ids that have
        telemetry, in request order. Never a superset of the request. Requested
        ids with no telemetry are omitted here and count as failures in
        ``focused_pass_rate``, which divides by the full request.

        Full eval (``None``): the objective's high-signal set.
        """
        if requested_entry_ids:
            return tuple(entry_id for entry_id in requested_entry_ids if entry_id in analysis.per_entry)
        return tuple(analysis.high_signal_entry_ids)

    @abstractmethod
    def log_analysis(self, analysis: Any) -> None: ...

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
    ) -> list[ScoredRow]:
        """Rows for one student eval. Objectives implement ``entry_row`` and ``aggregate_row``.

        - Validation (not focused, no traces): one aggregate row, ``entry_id=None``.
        - Nothing to surface per entry: the aggregate row keyed to the eval set.
        - Otherwise: one row per id from ``entry_ids_to_score``.
        """
        del al_data_inst
        ctx = ScoringContext(
            query=f"{eval_set_name}:{eval_set_version}",
            deployment_id=deployment_ids[0] if deployment_ids else "",
            is_focused=is_focused_eval,
            capture_traces=capture_traces,
            student_eval_id=student_eval_id,
        )
        if not is_focused_eval and not capture_traces:
            return [self.aggregate_row(analysis, ctx)]
        entry_ids = self.entry_ids_to_score(analysis, requested_entry_ids)
        if not entry_ids:
            return [replace(self.aggregate_row(analysis, ctx), entry_id=ctx.query)]
        # ``entry_ids_to_score`` only yields ids present in ``per_entry``.
        return [self.entry_row(entry_id, analysis.per_entry[entry_id], analysis, ctx) for entry_id in entry_ids]

    @abstractmethod
    def entry_row(self, entry_id: str, metrics: Any, analysis: Any, ctx: ScoringContext) -> ScoredRow:
        """Score and rollout output for one entry. ``ScoredRow.entry_id`` must be ``entry_id``."""

    @abstractmethod
    def aggregate_row(self, analysis: Any, ctx: ScoringContext) -> ScoredRow:
        """Run-level score. ``ScoredRow.entry_id`` must be ``None``; the base re-keys it when needed."""

    @abstractmethod
    def failure_pattern(self, component_name: str, trajectory: Any) -> tuple[Any, ...]: ...

    @abstractmethod
    def build_reflective_example(
        self,
        component_name: str,
        trajectory: Any,
        candidate: dict[str, str],
    ) -> ReflectiveExample: ...

    def cache_payload(self) -> dict[str, Any]:
        return {}

    def load_cache(self, raw_cache: Any) -> None:
        del raw_cache


def register_telemetry_source(
    mode: JudgingMode,
    source: str,
    cls: type,
    *,
    replace: bool = False,
    validate: bool = True,
) -> None:
    """Register an out-of-tree objective. Built-ins are listed in ``registry.BUILTIN_OBJECTIVES``."""
    _registry.register(mode, source, cls, replace=replace, validate=validate)


def unregister_telemetry_source(mode: JudgingMode, source: str) -> None:
    _registry.unregister(mode, source)


def _ensure_builtin_objectives_registered() -> None:
    _registry.load_builtins()


def is_registered_telemetry_source(mode: JudgingMode, source: str | None) -> bool:
    return _registry.is_registered(mode, source)


def is_telemetry_source(source: str | None) -> bool:
    return _registry.is_known_source(source)


def build_objective(
    mode: JudgingMode,
    signals: Sequence[Mapping[str, Any]] | None = None,
    *,
    bigquery_client: Any | None = None,
    lookback_days: int = 1,
    pack: Mapping[str, Any] | None = None,
) -> TeacherStudentObjective | SingleModelObjective:
    """Construct the telemetry objective registered for ``mode`` and the pack source."""
    source = _registry.default_source(mode)
    if signals:
        for signal in signals:
            if signal.get("enabled", True) is False:
                continue
            candidate = signal.get("source")
            if isinstance(candidate, str) and _registry.is_registered(mode, candidate):
                source = candidate
                break
    cls = _registry.resolve(mode, source)
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
