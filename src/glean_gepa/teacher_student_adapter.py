"""Adapter for teacher-vs-student Glean evaluations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from gepa.core.adapter import EvaluationBatch
from glean_gepa.adapter_types import (
    ALDataInst,
    ALRolloutOutput,
    ALTrajectory,
    PairwiseJudge,
    PointwiseJudge,
    TeacherStudentALDataInst,
    TeacherStudentALRolloutOutput,
    TeacherStudentALTrajectory,
)
from glean_gepa.al_adapter import (
    ALRunner,
    GleanAdapterBase,
    ReflectiveExample,
    Thresholds,
)
from glean_gepa.batch import EvalRunIds, GleanEvaluationBatch
from glean_gepa.evalcli_client import (
    AGENTIC_JUDGE_TYPE,
    COMPLETENESS_JUDGE_TYPE,
    CORRECTNESS_INPUT_MAPPINGS,
    CORRECTNESS_JUDGE_TYPE,
    CORRECTNESS_RUN_PARAMS,
)
from glean_gepa.focused_evalset import resolve_eval_run_target
from glean_gepa.judge_metrics_util import (
    JudgeAnalysis,
    wait_for_all_judge_metrics,
)
from glean_gepa.objectives import TeacherStudentObjective
from glean_gepa.objectives.tool_match import FirstToolMatchObjective
from glean_gepa.prompt import compile_encoded_prompt
from glean_gepa.prompt_constants import WRITING_CODE_KEY

CORRECTNESS_DIMENSION = "correctness"
CORRECTNESS_JUDGE = PairwiseJudge(
    CORRECTNESS_DIMENSION,
    CORRECTNESS_JUDGE_TYPE,
    CORRECTNESS_RUN_PARAMS,
    CORRECTNESS_INPUT_MAPPINGS,
)
POINTWISE_JUDGES: tuple[PointwiseJudge, ...] = ()
PAIRWISE_JUDGES: tuple[PairwiseJudge, ...] = ()
_ANALYSIS_FIELDS = frozenset({"aggregate", "per_entry", "judge_run_id"})


def _payloads_by_base(payload: Any) -> dict[str, Any]:
    """Normalize cached judge payloads keyed by baseline eval id.

    Older caches stored a run id or analysis dict directly under the judge type.
    Pairwise results are nested ``{base_eval_id: payload}``; pointwise uses ``""``.
    """
    if isinstance(payload, str):
        return {"": payload}
    if not isinstance(payload, dict):
        return {}
    if _ANALYSIS_FIELDS & payload.keys():
        return {"": payload}
    return {str(base): inner for base, inner in payload.items()}


def _same_screen_batch(left: Sequence[ALDataInst], right: Sequence[ALDataInst]) -> bool:
    """True when two child screens target the same focused eval set.

    Cached eval-run ids are attached per child, so identity/`==` on the batch
    dicts would miss the overlap and fall back to serial `evaluate()`.
    """

    def identity(batch: Sequence[ALDataInst]) -> list[tuple[Any, ...]]:
        return [
            (
                item.get("eval_set_name"),
                item.get("eval_set_version"),
                tuple(item.get("deployment_ids") or []),
                tuple(item.get("eval_entry_ids") or []),
            )
            for item in batch
        ]

    return identity(left) == identity(right)


def _entry_queries_from_listing(entries: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Map ``entry_id -> user query`` over listed eval-set entries."""
    resolved: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        entry_id = str(entry.get("id") or "")
        if not entry_id:
            continue
        entry_input = entry.get("input")
        raw = entry.get("query") or (entry_input.get("query") if isinstance(entry_input, Mapping) else None)
        if isinstance(raw, str) and raw.strip():
            resolved[entry_id] = raw.strip()
    return resolved


def _fetch_entry_queries(
    evalcli: Any,
    *,
    eval_set_name: str,
    eval_set_version: str,
    deployment_ids: Sequence[str],
) -> dict[str, str]:
    """List one eval-set version and return its ``entry_id -> query`` map."""
    if evalcli is None or not eval_set_name or not eval_set_version:
        return {}
    try:
        entries = evalcli.list_eval_set_entries(
            eval_set_name=eval_set_name,
            eval_set_version=eval_set_version,
            deployment_ids=list(deployment_ids),
        )
    except Exception as exc:
        print(f"[Entry Queries] Could not list entries for {eval_set_name}:{eval_set_version}: {exc}")
        return {}
    resolved = _entry_queries_from_listing(entries or [])
    print(f"[Entry Queries] Resolved {len(resolved)} user queries for {eval_set_name}:{eval_set_version}")
    return resolved


def _per_entry_or_aggregate(analysis: JudgeAnalysis, entry_id: str | None) -> float | None:
    """Eval-level rows use the aggregate; per-entry rows use a real entry score.

    An empty ``per_entry`` map is not a copy of the aggregate onto every row.
    Missing entries stay unscored so high-signal selection cannot treat the
    eval rate as every student's preference.
    """
    if entry_id is None:
        return analysis.aggregate
    if entry_id in analysis.per_entry:
        return analysis.per_entry[entry_id]
    if analysis.per_entry:
        return None
    return analysis.aggregate


@dataclass(frozen=True)
class _StartedPair:
    al_data_inst: TeacherStudentALDataInst
    teacher_eval_id: str
    student_eval_id: str
    eval_set_name: str = ""
    eval_set_version: str = ""


class TeacherStudentAdapter(GleanAdapterBase):
    """Optimize instructions from paired teacher-vs-student evaluations."""

    supports_high_signal_eval = True

    def __init__(
        self,
        runner: ALRunner,
        teacher_model: str,
        thresholds: Thresholds,
        student_model: str,
        *,
        bigquery_client: Any | None = None,
        agentspan_lookback_days: int = 1,
        editable_modules: list[str] | None = None,
        cache_file: str | None = None,
        primary_objective: str | None = None,
        default_frontier_type: str = "hybrid",
        composite_weights: dict[str, float] | None = None,
        constant_scores: dict[str, float] | None = None,
        pointwise_judges: Sequence[PointwiseJudge] | None = None,
        pairwise_judges: Sequence[PairwiseJudge] | None = None,
        objective: TeacherStudentObjective | None = None,
        screening_kind: str | None = None,
    ):
        self.pointwise_judges = tuple(pointwise_judges) if pointwise_judges is not None else POINTWISE_JUDGES
        self.pairwise_judges = tuple(pairwise_judges) if pairwise_judges is not None else PAIRWISE_JUDGES
        self.screening_kind = screening_kind
        self.teacher_model = teacher_model
        self.bigquery_client = bigquery_client
        self.agentspan_lookback_days = agentspan_lookback_days
        self.objective = objective or FirstToolMatchObjective(
            bigquery_client=bigquery_client,
            lookback_days=agentspan_lookback_days,
        )
        self.telemetry_dimensions = self.objective.telemetry_dimensions
        self._judge_runs: dict[tuple[str, str, str], str] = {}
        self._judge_cache: dict[tuple[str, str, str], JudgeAnalysis] = {}
        self._entry_query_cache: dict[tuple[str, str], dict[str, str]] = {}
        super().__init__(
            runner=runner,
            thresholds=thresholds,
            student_model=student_model,
            evaluate_fn=self._evaluate_teacher_student,
            failure_pattern_fn=self.objective.failure_pattern,
            reflective_example_fn=self.objective.build_reflective_example,
            reflection_prompt_fn=self.objective.reflection_prompt,
            reflective_metrics_fn=self.objective.format_reflective_metrics,
            failure_label=self.objective.failure_label,
            primary_objective=primary_objective or self.objective.name,
            default_frontier_type=default_frontier_type,
            editable_modules=list(editable_modules) if editable_modules else [WRITING_CODE_KEY],
            composite_weights=dict({self.objective.name: 1.0} if composite_weights is None else composite_weights),
            constant_scores=dict(constant_scores or {}),
            extra_scorable_dimensions={
                *(judge.name for judge in self.pointwise_judges),
                *(judge.name for judge in self.pairwise_judges),
            },
            cache_file=cache_file,
        )

    @property
    def _analysis_cache(self) -> dict[tuple[str, str], Any]:
        return self.objective.analysis_cache

    def _get_or_fetch_analysis(
        self,
        teacher_eval_id: str,
        student_eval_id: str,
        *,
        include_action_inputs: bool = True,
    ):
        self.objective.bigquery_client = self.bigquery_client
        # Validation skips trace hydration. evalcli stays available; the requested
        # flag decides whether this fetch hydrates and whether the cache hits.
        self.objective.include_action_inputs = include_action_inputs
        self.objective.evalcli = self.runner.evalcli
        self.objective.lookback_days = self.agentspan_lookback_days
        analysis = self.objective.analyze(teacher_eval_id, student_eval_id)
        self._save_cache()
        return analysis

    def _evaluate_teacher_student(
        self,
        batch: list[ALDataInst],
        candidate: dict[str, str],
        capture_traces: bool,
    ) -> GleanEvaluationBatch:
        typed_batch = cast(list[TeacherStudentALDataInst], batch)
        if not typed_batch:
            return GleanEvaluationBatch(outputs=[], scores=[], trajectories=None, objective_scores=[], summary=None)
        started, pending_waits = self._start_batch_evals(typed_batch, candidate)
        self._await_judge_metrics(self._wait_pending_evals(pending_waits, [started]))
        return self._finish_batch_evals(started, capture_traces)

    def _get_or_start_eval(
        self,
        *,
        model: str,
        system_prompt: str,
        eval_set_name: str,
        eval_set_version: str,
        deployment_ids: list[str],
        role: str,
        run_label: str = "gepa",
    ) -> tuple[str, bool]:
        """Return a cached eval id, or start a new run without waiting."""
        eval_id, wait_required = self.runner.start(
            model,
            system_prompt=system_prompt,
            eval_set_name=eval_set_name,
            eval_set_version=eval_set_version,
            deployment_ids=deployment_ids,
            run_label=run_label,
        )
        if wait_required:
            print(f"Waiting on {role} eval_id: {eval_id}")
        else:
            print(f"[Cache HIT] Using cached {role} eval_id: {eval_id} ({run_label})")
        return eval_id, wait_required

    def _start_batch_evals(
        self,
        batch: list[TeacherStudentALDataInst],
        candidate: dict[str, str],
    ) -> tuple[list[_StartedPair], dict[str, str]]:
        system_prompt = compile_encoded_prompt(candidate)
        started: list[_StartedPair] = []
        pending_waits: dict[str, str] = {}
        for al_data_inst in batch:
            target = resolve_eval_run_target(
                self.runner.evalcli,
                al_data_inst,
                bigquery_client=self.bigquery_client,
                bucket_type=self.objective.focused_bucket_type,
            )
            if target is None:
                print(
                    "[Focused eval set] Could not prepare high-signal eval set for "
                    f"{al_data_inst.get('eval_set_name')}:{al_data_inst.get('eval_set_version')}; skipping"
                )
                continue
            eval_set_name = target.eval_set_name
            eval_set_version = target.eval_set_version
            deployment_ids = al_data_inst.get("deployment_ids", [])
            run_label = target.run_label
            teacher_eval_id = al_data_inst.get("cached_teacher_eval_run_id")
            wait_teacher = False
            if teacher_eval_id:
                print(f"[Child cache HIT] Using cached teacher eval_id: {teacher_eval_id}")
            else:
                teacher_eval_id, wait_teacher = self._get_or_start_eval(
                    model=self.teacher_model,
                    system_prompt="<<TEACHER_PROD_PROMPT>>",
                    eval_set_name=eval_set_name,
                    eval_set_version=eval_set_version,
                    deployment_ids=deployment_ids,
                    role="teacher",
                    run_label=run_label,
                )
            student_eval_id = al_data_inst.get("cached_student_eval_run_id")
            wait_student = False
            if student_eval_id:
                print(f"[Child cache HIT] Using cached student eval_id: {student_eval_id}")
            else:
                student_eval_id, wait_student = self._get_or_start_eval(
                    model=self.student_model,
                    system_prompt=system_prompt,
                    eval_set_name=eval_set_name,
                    eval_set_version=eval_set_version,
                    deployment_ids=deployment_ids,
                    role="student",
                    run_label=run_label,
                )
            if wait_teacher:
                pending_waits[teacher_eval_id] = "teacher"
            if wait_student:
                pending_waits[student_eval_id] = "student"
            started.append(
                _StartedPair(
                    al_data_inst=al_data_inst,
                    teacher_eval_id=teacher_eval_id,
                    student_eval_id=student_eval_id,
                    eval_set_name=eval_set_name,
                    eval_set_version=eval_set_version,
                )
            )
        return started, pending_waits

    def _wait_pending_evals(
        self,
        pending_waits: dict[str, str],
        started_groups: Sequence[Sequence[_StartedPair]],
    ) -> list[tuple[str, str, str, str | None]]:
        """Wait for evals and start each pair's judges as soon as both sides finish."""
        completed = {
            eval_id
            for started in started_groups
            for pair in started
            for eval_id in (pair.teacher_eval_id, pair.student_eval_id)
            if eval_id not in pending_waits
        }
        pending: list[tuple[str, str, str, str | None]] = []
        seen: set[tuple[str, str, str]] = set()

        def start_ready_judges() -> None:
            for started in started_groups:
                pending.extend(self._start_judges(started, ready_eval_ids=completed, seen=seen))

        start_ready_judges()
        for eval_id, role in pending_waits.items():
            self.runner.wait(eval_id)
            print(f"Recorded completed {role} eval_id: {eval_id}")
            completed.add(eval_id)
            start_ready_judges()
        return pending

    @staticmethod
    def _judge_key(eval_id: str, judge_type: str, base_eval_run_id: str | None = None) -> tuple[str, str, str]:
        return (eval_id, base_eval_run_id or "", judge_type)

    def _extra_cache_payload(self) -> dict[str, Any]:
        judge_runs: dict[str, dict[str, dict[str, str]]] = {}
        for (eval_id, base_eval_id, judge_type), run_id in self._judge_runs.items():
            judge_runs.setdefault(eval_id, {}).setdefault(judge_type, {})[base_eval_id] = run_id
        judge_cache: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        for (eval_id, base_eval_id, judge_type), analysis in self._judge_cache.items():
            judge_cache.setdefault(eval_id, {}).setdefault(judge_type, {})[base_eval_id] = {
                "aggregate": analysis.aggregate,
                "per_entry": analysis.per_entry,
                "judge_run_id": analysis.judge_run_id,
                "per_entry_feedback": analysis.per_entry_feedback,
            }
        return {"judge_runs": judge_runs, "judge_cache": judge_cache}

    def _load_extra_cache(self, data: dict[str, Any]) -> None:
        self._judge_runs = {}
        raw_runs = data.get("judge_runs")
        if isinstance(raw_runs, dict):
            for eval_id, by_type in raw_runs.items():
                if not isinstance(by_type, dict):
                    continue
                for judge_type, payload in by_type.items():
                    for base_eval_id, run_id in _payloads_by_base(payload).items():
                        self._judge_runs[self._judge_key(str(eval_id), str(judge_type), base_eval_id)] = str(run_id)
        else:
            for eval_id, run_id in (data.get("completeness_judge_runs") or {}).items():
                self._judge_runs[self._judge_key(str(eval_id), COMPLETENESS_JUDGE_TYPE)] = str(run_id)

        raw_cache = data.get("judge_cache")
        if not isinstance(raw_cache, dict):
            raw_cache = {
                eval_id: {COMPLETENESS_JUDGE_TYPE: raw}
                for eval_id, raw in (data.get("completeness_cache") or {}).items()
                if isinstance(raw, dict)
            }
        self._judge_cache = {}
        for eval_id, by_type in raw_cache.items():
            if not isinstance(by_type, dict):
                continue
            for judge_type, payload in by_type.items():
                for base_eval_id, raw in _payloads_by_base(payload).items():
                    if not isinstance(raw, dict):
                        continue
                    per_entry = {
                        str(entry_id): float(score) for entry_id, score in (raw.get("per_entry") or {}).items()
                    }
                    aggregate_raw = raw.get("aggregate")
                    aggregate = 0.0 if aggregate_raw is None else float(aggregate_raw)
                    if per_entry and raw.get("aggregate") is None:
                        aggregate = sum(per_entry.values()) / len(per_entry)
                    self._judge_cache[self._judge_key(str(eval_id), str(judge_type), base_eval_id)] = JudgeAnalysis(
                        eval_id=str(eval_id),
                        aggregate=aggregate,
                        per_entry=per_entry,
                        judge_run_id=str(raw["judge_run_id"]) if raw.get("judge_run_id") else None,
                        judge_type=str(judge_type),
                        per_entry_feedback={
                            str(entry_id): str(text) for entry_id, text in (raw.get("per_entry_feedback") or {}).items()
                        },
                    )

    def _ensure_judge(
        self,
        eval_id: str,
        *,
        judge_type: str,
        run_params: str,
        base_eval_run_id: str | None = None,
        input_mappings: str | None = None,
    ) -> str:
        cache_key = self._judge_key(eval_id, judge_type, base_eval_run_id)
        judge_run_id = self._judge_runs.get(cache_key)
        if not judge_run_id:
            existing = self.runner.evalcli.find_judge_run_id(
                eval_id, judge_type=judge_type, base_eval_run_id=base_eval_run_id
            )
            if isinstance(existing, str) and existing:
                print(f"[{judge_type}] Reusing judge run {existing} for eval {eval_id}")
                judge_run_id = existing
            else:
                judge_run_id = self.runner.evalcli.create_judge_run(
                    eval_run_id=eval_id,
                    judge_type=judge_type,
                    run_params=run_params,
                    base_eval_run_id=base_eval_run_id,
                    input_mappings=input_mappings,
                )
                if not isinstance(judge_run_id, str) or not judge_run_id:
                    raise TypeError(f"{judge_type} judge create for {eval_id} returned {judge_run_id!r}")
                print(f"[{judge_type}] Started judge run {judge_run_id} for eval {eval_id}")
        if self._judge_runs.get(cache_key) != judge_run_id:
            self._judge_runs[cache_key] = judge_run_id
            self._save_cache()
        return judge_run_id

    def _pairwise_judges_for_pair(self, pair: _StartedPair) -> tuple[PairwiseJudge, ...]:
        """Start every configured pairwise judge, except the correctness-floor split.

        ``screening.kind=correctness_floor`` still runs CORRECTNESS on focused
        screens and AGENTIC on full/val evals. The agentic pack screens with
        AGENTIC_JUDGE on the high-signal slice instead, so it does not split.
        """
        if self.screening_kind != "correctness_floor":
            return self.pairwise_judges
        wanted_type = CORRECTNESS_JUDGE_TYPE if pair.al_data_inst.get("eval_entry_ids") else AGENTIC_JUDGE_TYPE
        return tuple(judge for judge in self.pairwise_judges if judge.judge_type == wanted_type)

    def _start_judges(
        self,
        started: Sequence[_StartedPair],
        *,
        ready_eval_ids: set[str] | None = None,
        seen: set[tuple[str, str, str]] | None = None,
    ) -> list[tuple[str, str, str, str | None]]:
        """Create Cortex judge runs for pairs whose evals are done. Do not wait for metrics."""
        pending: list[tuple[str, str, str, str | None]] = []
        seen_keys = seen if seen is not None else set()
        for pair in started:
            pair_ready = ready_eval_ids is None or (
                pair.teacher_eval_id in ready_eval_ids and pair.student_eval_id in ready_eval_ids
            )
            if pair_ready:
                for judge in self._pairwise_judges_for_pair(pair):
                    cache_key = self._judge_key(pair.student_eval_id, judge.judge_type, pair.teacher_eval_id)
                    if cache_key in seen_keys or cache_key in self._judge_cache:
                        continue
                    seen_keys.add(cache_key)
                    judge_run_id = self._ensure_judge(
                        pair.student_eval_id,
                        judge_type=judge.judge_type,
                        run_params=judge.run_params,
                        base_eval_run_id=pair.teacher_eval_id,
                        input_mappings=judge.input_mappings,
                    )
                    pending.append((pair.student_eval_id, judge.judge_type, judge_run_id, pair.teacher_eval_id))
            for eval_id in (pair.teacher_eval_id, pair.student_eval_id):
                if ready_eval_ids is not None and eval_id not in ready_eval_ids:
                    continue
                for judge in self.pointwise_judges:
                    cache_key = self._judge_key(eval_id, judge.judge_type)
                    if cache_key in seen_keys or cache_key in self._judge_cache:
                        continue
                    seen_keys.add(cache_key)
                    judge_run_id = self._ensure_judge(eval_id, judge_type=judge.judge_type, run_params=judge.run_params)
                    pending.append((eval_id, judge.judge_type, judge_run_id, None))
        return pending

    def _await_judge_metrics(self, pending: Sequence[tuple[str, str, str | None, str | None]]) -> None:
        """Block until every started judge has finished, then read all scores."""
        analyses = wait_for_all_judge_metrics(self.runner.evalcli, pending)
        for eval_id, judge_type, _judge_run_id, base_eval_id in pending:
            analysis = analyses.get((eval_id, judge_type, base_eval_id))
            if analysis is None:
                continue
            self._judge_cache[self._judge_key(eval_id, judge_type, base_eval_id)] = analysis
        if pending:
            self._save_cache()

    def _judge_for(self, eval_id: str, *, judge_type: str, base_eval_run_id: str | None = None) -> JudgeAnalysis:
        cached = self._judge_cache.get(self._judge_key(eval_id, judge_type, base_eval_run_id))
        if cached is not None:
            return cached
        return JudgeAnalysis(eval_id=eval_id, aggregate=0.0, per_entry={}, judge_type=judge_type)

    def batch_evaluate(
        self,
        items: list[tuple[dict[str, str], list[ALDataInst]]],
        *,
        capture_traces: bool = True,
    ) -> list[GleanEvaluationBatch]:
        """Overlap shared-batch screens; keep per-child cached eval-run ids."""
        if not items:
            return []
        first_batch = items[0][1]
        if not all(_same_screen_batch(batch, first_batch) for _candidate, batch in items):
            return [self.evaluate(batch, candidate, capture_traces=capture_traces) for candidate, batch in items]
        all_started: list[list[_StartedPair]] = []
        pending_waits: dict[str, str] = {}
        for candidate, batch in items:
            started, candidate_pending = self._start_batch_evals(
                cast(list[TeacherStudentALDataInst], batch), candidate
            )
            all_started.append(started)
            pending_waits.update(candidate_pending)
        self._await_judge_metrics(self._wait_pending_evals(pending_waits, all_started))
        return [self._finish_batch_evals(started, capture_traces) for started in all_started]

    def evaluate_many(
        self,
        batch: list[ALDataInst],
        candidates: list[dict[str, str]],
        capture_traces: bool = False,
    ) -> list[GleanEvaluationBatch]:
        typed_batch = cast(list[TeacherStudentALDataInst], batch)
        all_started: list[list[_StartedPair]] = []
        pending_waits: dict[str, str] = {}
        for candidate in candidates:
            started, candidate_pending = self._start_batch_evals(typed_batch, candidate)
            all_started.append(started)
            pending_waits.update(candidate_pending)
        self._await_judge_metrics(self._wait_pending_evals(pending_waits, all_started))
        return [self._finish_batch_evals(started, capture_traces) for started in all_started]

    def high_signal_batch(self, eval_batch: GleanEvaluationBatch) -> list[ALDataInst]:
        """Keep every parent entry the objective marks as high-signal."""
        grouped: dict[tuple[str, str, tuple[str, ...]], list[str]] = {}
        seen: set[str] = set()
        for trajectory in eval_batch.trajectories or []:
            data = trajectory["data"]
            output = trajectory["output"]
            entry_id = output.get("entry_id")
            if not entry_id or entry_id in seen:
                continue
            if not self.objective.is_high_signal(output):
                continue
            seen.add(entry_id)
            key = (data["eval_set_name"], data["eval_set_version"], tuple(data["deployment_ids"]))
            grouped.setdefault(key, []).append(entry_id)
        if grouped:
            count = sum(len(ids) for ids in grouped.values())
            print(f"[High-signal] Selected {count} entries for screening")
        return [
            {
                "eval_set_name": eval_set_name,
                "eval_set_version": eval_set_version,
                "deployment_ids": list(deployment_ids),
                "status": "active",
                "eval_entry_ids": entry_ids,
            }
            for (eval_set_name, eval_set_version, deployment_ids), entry_ids in grouped.items()
        ]

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[ALTrajectory, ALRolloutOutput],
        components_to_update: list[str],
        k: int | None,
        error_hamming_distance_k: int | None = None,
    ) -> dict[str, list[ReflectiveExample]]:
        """Build reflection examples from the objective's high-signal selection.

        ``k`` is ``search.reflection_samples`` (CLI ``--reflection_samples``): an
        integer caps the set, ``None`` keeps every high-signal entry. Hamming
        dedupe is not applied on this path.
        """
        del error_hamming_distance_k
        return self.objective.make_reflective_dataset(
            candidate,
            eval_batch,
            components_to_update,
            self.objective.build_reflective_example,
            k=k,
        )

    def high_signal_core_tool_keys(self, trajectories: Sequence[Any] | None) -> list[str]:
        return self.objective.high_signal_core_tool_keys(trajectories)

    def _finish_batch_evals(
        self,
        started: list[_StartedPair],
        capture_traces: bool,
    ) -> GleanEvaluationBatch[TeacherStudentALTrajectory, TeacherStudentALRolloutOutput]:
        all_outputs: list[TeacherStudentALRolloutOutput] = []
        all_scores: list[float] = []
        all_trajectories: list[TeacherStudentALTrajectory] | None = [] if capture_traces else None
        all_objective_scores: list[dict[str, float]] = []
        focused_alignment_rates: list[float] = []
        all_eval_run_ids: list[EvalRunIds] = []

        for pair in started:
            al_data_inst = pair.al_data_inst
            all_eval_run_ids.append(
                {
                    "eval_set_name": str(al_data_inst.get("eval_set_name", "")),
                    "eval_set_version": str(al_data_inst.get("eval_set_version", "")),
                    "student_eval_run_id": pair.student_eval_id,
                    "teacher_eval_run_id": pair.teacher_eval_id,
                }
            )
            analysis = self._get_or_fetch_analysis(
                pair.teacher_eval_id,
                pair.student_eval_id,
                include_action_inputs=not bool(al_data_inst.get("validation_only")),
            )
            requested_entry_ids = al_data_inst.get("eval_entry_ids") or []
            is_focused_eval = bool(requested_entry_ids)
            primary_from_pairwise_judge = any(judge.name == self.objective.name for judge in self.pairwise_judges)
            if is_focused_eval:
                # Pairwise-judge primaries (agentic preference) already land in
                # summary via the judge overlay. Overwriting with focused_pass_rate
                # would zero that screen. Trace-based primaries still need it.
                if self.screening_kind != "correctness_floor" and not primary_from_pairwise_judge:
                    focused_alignment_rates.append(self.objective.focused_pass_rate(analysis, requested_entry_ids))
            else:
                self.objective.validate_full_eval(analysis)
            deployment_id = (al_data_inst.get("deployment_ids") or [""])[0]
            query = f"{al_data_inst.get('eval_set_name', '')}:{al_data_inst.get('eval_set_version', '')}"
            # Validation eval sets are PII-gated and never feed reflection, so they are
            # not listed at all. An empty result is cached to avoid re-listing.
            entry_queries: dict[str, str] = {}
            if not al_data_inst.get("validation_only"):
                cache_key = (pair.eval_set_name, pair.eval_set_version)
                cached = self._entry_query_cache.get(cache_key)
                if cached is None:
                    cached = _fetch_entry_queries(
                        self.runner.evalcli,
                        eval_set_name=pair.eval_set_name,
                        eval_set_version=pair.eval_set_version,
                        deployment_ids=al_data_inst.get("deployment_ids") or [],
                    )
                    self._entry_query_cache[cache_key] = cached
                entry_queries = cached
            pairwise_for_pair = self._pairwise_judges_for_pair(pair)
            student_judges = {
                **{
                    judge.name: self._judge_for(pair.student_eval_id, judge_type=judge.judge_type)
                    for judge in self.pointwise_judges
                },
                **{
                    judge.name: self._judge_for(
                        pair.student_eval_id,
                        judge_type=judge.judge_type,
                        base_eval_run_id=pair.teacher_eval_id,
                    )
                    for judge in pairwise_for_pair
                },
            }
            for judge in self.pointwise_judges:
                teacher_analysis = self._judge_for(pair.teacher_eval_id, judge_type=judge.judge_type)
                print(
                    f"[{judge.judge_type}] student {pair.student_eval_id}="
                    f"{student_judges[judge.name].aggregate:.2f} "
                    f"teacher {pair.teacher_eval_id}={teacher_analysis.aggregate:.2f}"
                )
            for judge in pairwise_for_pair:
                print(
                    f"[{judge.judge_type}] student {pair.student_eval_id} vs teacher "
                    f"{pair.teacher_eval_id}={student_judges[judge.name].aggregate:.2f}"
                )

            scored_rows = self.objective.scored_rows(
                analysis,
                focused=is_focused_eval,
                capture_traces=capture_traces,
                query=query,
                deployment_id=deployment_id,
            )
            if not scored_rows:
                continue

            primary_judge = student_judges.get(self.objective.name)
            primary_judge_run_id = primary_judge.judge_run_id if primary_judge is not None else None
            for row in scored_rows:
                output = cast(TeacherStudentALRolloutOutput, dict(row.output))
                if row.entry_id and (entry_query := entry_queries.get(row.entry_id)):
                    output["query"] = entry_query
                if primary_judge_run_id:
                    output["judge_run_id"] = primary_judge_run_id
                judge_scores: dict[str, float] = {}
                for name, judge_analysis in student_judges.items():
                    score = _per_entry_or_aggregate(judge_analysis, row.entry_id)
                    if score is None:
                        continue
                    judge_scores[name] = score
                    if row.entry_id is not None and row.entry_id in judge_analysis.per_entry:
                        output[name] = score
                        feedback = judge_analysis.per_entry_feedback.get(row.entry_id)
                        if feedback:
                            output[f"{name}_feedback"] = feedback
                all_outputs.append(output)
                objective_score = {
                    **self.constant_scores,
                    **row.dimension_scores,
                    **judge_scores,
                }
                score = self.composite_score(objective_score)
                all_scores.append(score)
                all_objective_scores.append(objective_score)
                if capture_traces and all_trajectories is not None:
                    trajectory_data = {**al_data_inst, **row.data_overrides}
                    trajectory: TeacherStudentALTrajectory = {
                        "data": cast(TeacherStudentALDataInst, trajectory_data),
                        "output": output,
                        "score": score,
                        "objective_scores": objective_score,
                    }
                    all_trajectories.append(trajectory)

        summary = None
        if all_objective_scores:
            summary = {}
            all_dims: set[str] = set()
            for obj_score in all_objective_scores:
                all_dims.update(obj_score.keys())
            for dim in all_dims:
                values = [obj_score.get(dim, 0.0) for obj_score in all_objective_scores if dim in obj_score]
                summary[dim] = sum(values) / len(values) if values else 0.0
        if focused_alignment_rates:
            if summary is None:
                summary = {judge.name: 0.0 for judge in (*self.pointwise_judges, *self.pairwise_judges)}
                summary.update(self.constant_scores)
            summary[self.objective.name] = sum(focused_alignment_rates) / len(focused_alignment_rates)
        if summary is not None and started:
            for judge in self.pointwise_judges:
                teacher_scores = [
                    self._judge_for(pair.teacher_eval_id, judge_type=judge.judge_type).aggregate for pair in started
                ]
                summary[f"teacher_{judge.name}"] = sum(teacher_scores) / len(teacher_scores)

        return GleanEvaluationBatch(
            outputs=all_outputs,
            scores=all_scores,
            trajectories=all_trajectories,
            objective_scores=all_objective_scores,
            summary=summary,
            eval_run_ids=all_eval_run_ids,
        )


__all__ = [
    "TeacherStudentALDataInst",
    "TeacherStudentALRolloutOutput",
    "TeacherStudentALTrajectory",
    "TeacherStudentAdapter",
]
