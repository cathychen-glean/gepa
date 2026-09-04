"""Adapter for teacher-vs-student Glean evaluations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, cast

from gepa.core.adapter import EvaluationBatch
from glean_gepa.adapter_types import (
    ALDataInst,
    ALRolloutOutput,
    ALTrajectory,
    TeacherStudentALDataInst,
    TeacherStudentALRolloutOutput,
    TeacherStudentALTrajectory,
)
from glean_gepa.al_adapter import (
    ALRunner,
    GleanAdapterBase,
    ReflectiveExample,
    ReflectiveExampleInputs,
    ReflectiveExampleMetrics,
    Thresholds,
)
from glean_gepa.batch import EvalRunIds, GleanEvaluationBatch
from glean_gepa.core_tools import (
    CORE_TOOL_KEYS,
    core_tool_reflection_prompt,
    is_core_tool_span,
    tool_description_override_key,
)
from glean_gepa.evalcli_client import COMPLETENESS_JUDGE_TYPE, COMPLETENESS_RUN_PARAMS
from glean_gepa.focused_evalset import resolve_eval_run_target
from glean_gepa.judge_metrics_util import (
    JudgeAnalysis,
    wait_for_judge_metrics,
)
from glean_gepa.prompt import FULL_PROMPT_KEY, RULES_EXT_KEY, WRITING_CODE_KEY, compile_encoded_prompt
from glean_gepa.run_log import (
    format_eval_entry_report,
    format_high_signal_selection_report,
    log_section,
    selected_entry_ids_from_examples,
)
from glean_gepa.tool_match_util import (
    PREFIX_K_GRADUATION_RATE,
    PREFIX_K_SHORTAGE_ENTRIES,
    PREFIX_K_TOP_CANDIDATES,
    EvalRunToolMatchAnalysis,
    empty_tool_match_analysis,
    exact_prefix_match,
    fetch_eval_run_tool_match_analysis,
    first_disagreeing_mismatch,
    first_tool_mismatch_pair,
    intersect_compared_entry_ids,
    log_tool_match_analysis,
    prefix_tools,
    require_compared_eval_entries,
    restrict_tool_match_analysis,
    select_first_tool_mismatch_groups,
    tool_prefix_alignment,
)

PRIMARY_OBJECTIVE = "tool_alignment"
ADAPTER_STATE_PREFIX_K = "tool_prefix_k"
COMPLETENESS_WEIGHT = 0.5
TOOL_ALIGNMENT_WEIGHT = 0.5
POINTWISE_JUDGES: tuple[tuple[str, str], ...] = ((COMPLETENESS_JUDGE_TYPE, COMPLETENESS_RUN_PARAMS),)


@dataclass(frozen=True)
class _StartedPair:
    al_data_inst: TeacherStudentALDataInst
    teacher_eval_id: str
    student_eval_id: str


class TeacherStudentAdapter(GleanAdapterBase):
    """Optimize instructions from teacher-vs-student tool-usage comparisons."""

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
        val_eval_versions: list[str] | None = None,
        prefix_k: int = 1,
    ):
        self.teacher_model = teacher_model
        self.bigquery_client = bigquery_client
        self.agentspan_lookback_days = agentspan_lookback_days
        self._tool_match_cache: dict[tuple[str, str], EvalRunToolMatchAnalysis] = {}
        self._judge_runs: dict[tuple[str, str], str] = {}
        self._judge_cache: dict[tuple[str, str], JudgeAnalysis] = {}
        self.val_eval_versions = {str(version) for version in (val_eval_versions or [])}
        self.prefix_k = max(1, int(prefix_k))
        self._val_prefix_sequences: dict[str, dict[str, list[tuple[tuple[str, ...], tuple[str, ...]]]]] = {}
        self._last_train_low_alignment_count: int | None = None
        super().__init__(
            runner=runner,
            thresholds=thresholds,
            student_model=student_model,
            evaluate_fn=self._evaluate_teacher_student,
            failure_pattern_fn=self._create_failure_pattern,
            reflective_example_fn=self._build_reflective_example,
            reflection_prompt_fn=self._reflection_prompt,
            reflective_metrics_fn=self._format_reflective_metrics,
            failure_label="HIGH-SIGNAL FAILURES (teacher vs student tool match)",
            primary_objective=PRIMARY_OBJECTIVE,
            default_frontier_type="hybrid",
            editable_modules=list(editable_modules) if editable_modules else [WRITING_CODE_KEY],
            cache_file=cache_file,
        )

    def _get_or_fetch_tool_match_analysis(self, teacher_eval_id: str, student_eval_id: str) -> EvalRunToolMatchAnalysis:
        cache_key = (teacher_eval_id, student_eval_id)
        cached = self._tool_match_cache.get(cache_key)
        if cached is not None:
            print(f"[Cache HIT] Using cached tool match analysis for {teacher_eval_id} vs {student_eval_id}")
            return cached
        if self.bigquery_client is None:
            analysis = empty_tool_match_analysis(teacher_eval_id, student_eval_id)
        else:
            analysis = fetch_eval_run_tool_match_analysis(
                self.bigquery_client,
                teacher_eval_id=teacher_eval_id,
                student_eval_id=student_eval_id,
                lookback_days=self.agentspan_lookback_days,
            )
        self._tool_match_cache[cache_key] = analysis
        self._save_cache()
        print(f"[Cache MISS] Fetched tool match analysis for {teacher_eval_id} vs {student_eval_id}")
        return analysis

    @staticmethod
    def _candidate_key(candidate: dict[str, str]) -> str:
        return hashlib.md5(json.dumps(candidate, sort_keys=True).encode()).hexdigest()

    def get_adapter_state(self) -> dict[str, Any]:
        return {
            ADAPTER_STATE_PREFIX_K: self.prefix_k,
            "val_prefix_sequences": {
                candidate_key: {
                    version: [{"teacher": list(teacher), "student": list(student)} for teacher, student in sequences]
                    for version, sequences in by_version.items()
                }
                for candidate_key, by_version in self._val_prefix_sequences.items()
            },
            "train_low_alignment_count": self._last_train_low_alignment_count,
        }

    def set_adapter_state(self, state: dict[str, Any]) -> None:
        if not state:
            return
        prefix_k = state.get(ADAPTER_STATE_PREFIX_K)
        if prefix_k is not None:
            self.prefix_k = max(self.prefix_k, max(1, int(prefix_k)))
        self._last_train_low_alignment_count = (
            int(state["train_low_alignment_count"]) if state.get("train_low_alignment_count") is not None else None
        )
        raw_sequences = state.get("val_prefix_sequences") or {}
        if isinstance(raw_sequences, dict):
            restored: dict[str, dict[str, list[tuple[tuple[str, ...], tuple[str, ...]]]]] = {}
            for candidate_key, by_version in raw_sequences.items():
                if not isinstance(by_version, dict):
                    continue
                restored[str(candidate_key)] = {}
                for version, rows in by_version.items():
                    sequences: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
                    for row in rows or []:
                        if not isinstance(row, dict):
                            continue
                        sequences.append((tuple(row.get("teacher") or []), tuple(row.get("student") or [])))
                    restored[str(candidate_key)][str(version)] = sequences
            self._val_prefix_sequences = restored

    def record_train_low_alignment_count(self, eval_batch: GleanEvaluationBatch) -> int:
        count = sum(
            1
            for trajectory in eval_batch.trajectories or []
            if float((trajectory.get("objective_scores") or {}).get("tool_alignment", trajectory.get("score", 1.0)))
            < 1.0
        )
        self._last_train_low_alignment_count = count
        return count

    def pooled_val_prefix_match_rate(self, candidate: dict[str, str], *, prefix_k: int | None = None) -> float | None:
        by_version = self._val_prefix_sequences.get(self._candidate_key(candidate))
        if not by_version:
            return None
        k = self.prefix_k if prefix_k is None else prefix_k
        versions = self.val_eval_versions or set(by_version)
        sequences = [pair for version, rows in by_version.items() if version in versions for pair in rows]
        if not sequences:
            return None
        matches = sum(1 for teacher, student in sequences if exact_prefix_match(teacher, student, k))
        return matches / len(sequences)

    def maybe_graduate_prefix_k(self, *, iteration: int, candidates: list[dict[str, str]]) -> bool:
        """Advance sticky prefix depth after an iteration's val, never before iter 1."""
        if iteration < 2:
            return False
        rates = [
            rate
            for candidate in candidates
            for rate in [self.pooled_val_prefix_match_rate(candidate)]
            if rate is not None
        ]
        rates.sort(reverse=True)
        top = rates[:PREFIX_K_TOP_CANDIDATES]
        val_trigger = bool(top) and all(rate > PREFIX_K_GRADUATION_RATE for rate in top)
        shortage = (
            self._last_train_low_alignment_count is not None
            and self._last_train_low_alignment_count < PREFIX_K_SHORTAGE_ENTRIES
        )
        if not val_trigger and not shortage:
            return False
        self.prefix_k += 1
        self._save_cache()
        reason = []
        if val_trigger:
            formatted = ", ".join(f"{rate:.1%}" for rate in top)
            reason.append(
                f"top {len(top)} pooled prefix-{self.prefix_k - 1} match rates [{formatted}] "
                f"> {PREFIX_K_GRADUATION_RATE:.0%}"
            )
        if shortage:
            reason.append(
                f"train low-alignment entries={self._last_train_low_alignment_count} < {PREFIX_K_SHORTAGE_ENTRIES}"
            )
        print(f"[Prefix k] Graduated to k={self.prefix_k} ({'; '.join(reason)})")
        return True

    def high_signal_fix_rate(self, parent_eval: GleanEvaluationBatch, child_eval: GleanEvaluationBatch) -> float:
        """Fraction of parent alignment<1.0 entries whose child alignment improved."""
        parent_by_id: dict[str, float] = {}
        for trajectory in parent_eval.trajectories or []:
            entry_id = str((trajectory.get("output") or {}).get("entry_id") or "")
            if not entry_id or entry_id in parent_by_id:
                continue
            parent_by_id[entry_id] = float(
                (trajectory.get("objective_scores") or {}).get("tool_alignment", trajectory.get("score", 0.0))
            )
        failures = {entry_id: alignment for entry_id, alignment in parent_by_id.items() if alignment < 1.0}
        if not failures:
            return 0.0
        child_by_id: dict[str, float] = {}
        for trajectory in child_eval.trajectories or []:
            entry_id = str((trajectory.get("output") or {}).get("entry_id") or "")
            if not entry_id:
                continue
            child_by_id[entry_id] = float(
                (trajectory.get("objective_scores") or {}).get("tool_alignment", trajectory.get("score", 0.0))
            )
        improved = sum(
            1 for entry_id, parent_alignment in failures.items() if child_by_id.get(entry_id, 0.0) > parent_alignment
        )
        return improved / len(failures)

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
        self._wait_pending_evals(pending_waits)
        self._run_judges(started)
        return self._finish_batch_evals(started, capture_traces, candidate=candidate)

    def _get_or_start_eval(
        self,
        *,
        cache_key: tuple[str, ...],
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
    ) -> tuple[list[_StartedPair], dict[str, tuple[tuple[str, ...], str]]]:
        system_prompt = compile_encoded_prompt(candidate)
        started: list[_StartedPair] = []
        pending_waits: dict[str, tuple[tuple[str, ...], str]] = {}
        for al_data_inst in batch:
            target = resolve_eval_run_target(
                self.runner.evalcli,
                al_data_inst,
                bigquery_client=self.bigquery_client,
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
            teacher_prompt_hash = hashlib.md5(b"<<TEACHER_PROD_PROMPT>>").hexdigest()[:16]
            student_prompt_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:16]
            teacher_cache_key = (
                eval_set_name,
                eval_set_version,
                self.teacher_model,
                teacher_prompt_hash,
                run_label,
            )
            student_cache_key = (
                eval_set_name,
                eval_set_version,
                self.student_model,
                student_prompt_hash,
                run_label,
            )
            teacher_eval_id = al_data_inst.get("cached_teacher_eval_run_id")
            wait_teacher = False
            if teacher_eval_id:
                print(f"[Child cache HIT] Using cached teacher eval_id: {teacher_eval_id}")
            else:
                teacher_eval_id, wait_teacher = self._get_or_start_eval(
                    cache_key=teacher_cache_key,
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
                    cache_key=student_cache_key,
                    model=self.student_model,
                    system_prompt=system_prompt,
                    eval_set_name=eval_set_name,
                    eval_set_version=eval_set_version,
                    deployment_ids=deployment_ids,
                    role="student",
                    run_label=run_label,
                )
            if wait_teacher:
                pending_waits[teacher_eval_id] = (teacher_cache_key, "teacher")
            if wait_student:
                pending_waits[student_eval_id] = (student_cache_key, "student")
            started.append(
                _StartedPair(
                    al_data_inst=al_data_inst,
                    teacher_eval_id=teacher_eval_id,
                    student_eval_id=student_eval_id,
                )
            )
        return started, pending_waits

    def _wait_pending_evals(self, pending_waits: dict[str, tuple[tuple[str, ...], str]]) -> None:
        for eval_id, (_cache_key, role) in pending_waits.items():
            self.runner.wait(eval_id)
            print(f"Recorded completed {role} eval_id: {eval_id}")
            for judge_type, run_params in POINTWISE_JUDGES:
                self._ensure_judge(eval_id, judge_type=judge_type, run_params=run_params)

    def _extra_cache_payload(self) -> dict[str, Any]:
        judge_runs: dict[str, dict[str, str]] = {}
        for (eval_id, judge_type), run_id in self._judge_runs.items():
            judge_runs.setdefault(eval_id, {})[judge_type] = run_id
        judge_cache: dict[str, dict[str, dict[str, Any]]] = {}
        for (eval_id, judge_type), analysis in self._judge_cache.items():
            judge_cache.setdefault(eval_id, {})[judge_type] = {
                "aggregate": analysis.aggregate,
                "per_entry": analysis.per_entry,
                "judge_run_id": analysis.judge_run_id,
            }
        return {"judge_runs": judge_runs, "judge_cache": judge_cache, **self.get_adapter_state()}

    def _load_extra_cache(self, data: dict[str, Any]) -> None:
        self._judge_runs = {}
        raw_runs = data.get("judge_runs")
        if isinstance(raw_runs, dict):
            for eval_id, by_type in raw_runs.items():
                if not isinstance(by_type, dict):
                    continue
                for judge_type, run_id in by_type.items():
                    self._judge_runs[(str(eval_id), str(judge_type))] = str(run_id)
        else:
            for eval_id, run_id in (data.get("completeness_judge_runs") or {}).items():
                self._judge_runs[(str(eval_id), COMPLETENESS_JUDGE_TYPE)] = str(run_id)

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
            for judge_type, raw in by_type.items():
                if not isinstance(raw, dict):
                    continue
                per_entry = {str(entry_id): float(score) for entry_id, score in (raw.get("per_entry") or {}).items()}
                aggregate_raw = raw.get("aggregate")
                aggregate = 0.0 if aggregate_raw is None else float(aggregate_raw)
                if per_entry and raw.get("aggregate") is None:
                    aggregate = sum(per_entry.values()) / len(per_entry)
                self._judge_cache[(str(eval_id), str(judge_type))] = JudgeAnalysis(
                    eval_id=str(eval_id),
                    aggregate=aggregate,
                    per_entry=per_entry,
                    judge_run_id=str(raw["judge_run_id"]) if raw.get("judge_run_id") else None,
                    judge_type=str(judge_type),
                )
        self.set_adapter_state(data)

    def _ensure_judge(self, eval_id: str, *, judge_type: str, run_params: str) -> str:
        cache_key = (eval_id, judge_type)
        judge_run_id = self._judge_runs.get(cache_key)
        if not judge_run_id:
            existing = self.runner.evalcli.find_judge_run_id(eval_id, judge_type=judge_type)
            if isinstance(existing, str) and existing:
                print(f"[{judge_type}] Reusing judge run {existing} for eval {eval_id}")
                judge_run_id = existing
            else:
                judge_run_id = self.runner.evalcli.create_judge_run(
                    eval_run_id=eval_id,
                    judge_type=judge_type,
                    run_params=run_params,
                )
                if not isinstance(judge_run_id, str) or not judge_run_id:
                    raise TypeError(f"{judge_type} judge create for {eval_id} returned {judge_run_id!r}")
                print(f"[{judge_type}] Started judge run {judge_run_id} for eval {eval_id}")
        if self._judge_runs.get(cache_key) != judge_run_id:
            self._judge_runs[cache_key] = judge_run_id
            self._save_cache()
        return judge_run_id

    def _run_judges(self, started: list[_StartedPair]) -> None:
        """Trigger pointwise judges after evals finish; reuse cached teacher/seed runs."""
        pending: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for pair in started:
            for eval_id in (pair.teacher_eval_id, pair.student_eval_id):
                for judge_type, run_params in POINTWISE_JUDGES:
                    cache_key = (eval_id, judge_type)
                    if cache_key in seen:
                        continue
                    seen.add(cache_key)
                    if cache_key in self._judge_cache:
                        continue
                    judge_run_id = self._ensure_judge(eval_id, judge_type=judge_type, run_params=run_params)
                    pending.append((eval_id, judge_type, judge_run_id))
        for eval_id, judge_type, judge_run_id in pending:
            analysis = wait_for_judge_metrics(
                self.runner.evalcli,
                eval_id=eval_id,
                judge_type=judge_type,
                judge_run_id=judge_run_id,
            )
            self._judge_cache[(eval_id, judge_type)] = analysis
        if pending:
            self._save_cache()

    def _judge_for(self, eval_id: str, *, judge_type: str) -> JudgeAnalysis:
        cached = self._judge_cache.get((eval_id, judge_type))
        if cached is not None:
            return cached
        return JudgeAnalysis(eval_id=eval_id, aggregate=0.0, per_entry={}, judge_type=judge_type)

    def batch_evaluate(
        self,
        items: list[tuple[dict[str, str], list[ALDataInst]]],
        *,
        capture_traces: bool = True,
    ) -> list[GleanEvaluationBatch]:
        """Overlap shared-batch screens through evaluate_many instead of threading evaluate()."""
        if not items:
            return []
        first_batch = items[0][1]
        if all(batch is first_batch or batch == first_batch for _candidate, batch in items):
            return self.evaluate_many(
                first_batch,
                [candidate for candidate, _batch in items],
                capture_traces,
            )
        return [self.evaluate(batch, candidate, capture_traces=capture_traces) for candidate, batch in items]

    def evaluate_many(
        self,
        batch: list[ALDataInst],
        candidates: list[dict[str, str]],
        capture_traces: bool = False,
    ) -> list[GleanEvaluationBatch]:
        typed_batch = cast(list[TeacherStudentALDataInst], batch)
        all_started: list[list[_StartedPair]] = []
        pending_waits: dict[str, tuple[tuple[str, ...], str]] = {}
        for candidate in candidates:
            started, candidate_pending = self._start_batch_evals(typed_batch, candidate)
            all_started.append(started)
            pending_waits.update(candidate_pending)
        self._wait_pending_evals(pending_waits)
        for started in all_started:
            self._run_judges(started)
        shared_ids = self._inner_join_compared_entry_ids(typed_batch, all_started)
        return [
            self._finish_batch_evals(
                started,
                capture_traces,
                candidate=candidate,
                compared_entry_ids_by_version=shared_ids,
            )
            for started, candidate in zip(all_started, candidates, strict=True)
        ]

    def _inner_join_compared_entry_ids(
        self,
        batch: list[TeacherStudentALDataInst],
        all_started: list[list[_StartedPair]],
    ) -> dict[str, set[str]] | None:
        """Intersect compared entries across children on the same full eval set.

        Focused screens already share ``eval_entry_ids`` as the denominator.
        A single candidate has nothing to join.
        """
        if len(all_started) < 2:
            return None
        if any(bool(item.get("eval_entry_ids")) for item in batch):
            return None
        by_version: dict[str, list[EvalRunToolMatchAnalysis]] = {}
        for started in all_started:
            for pair in started:
                version = str(pair.al_data_inst.get("eval_set_version", ""))
                by_version.setdefault(version, []).append(
                    self._get_or_fetch_tool_match_analysis(pair.teacher_eval_id, pair.student_eval_id)
                )
        shared: dict[str, set[str]] = {}
        for version, analyses in by_version.items():
            entry_ids = intersect_compared_entry_ids(analyses)
            shared[version] = entry_ids
            print(
                f"[Tool Match] Inner join {version}: {len(entry_ids)} entries "
                f"across {len(analyses)} candidates"
            )
        return shared

    def high_signal_batch(self, eval_batch: GleanEvaluationBatch) -> list[ALDataInst]:
        """Keep every parent entry whose current-k tool alignment is below 1.0."""
        grouped: dict[tuple[str, str, tuple[str, ...]], list[str]] = {}
        seen: set[str] = set()
        for trajectory in eval_batch.trajectories or []:
            data = trajectory["data"]
            output = trajectory["output"]
            entry_id = output.get("entry_id")
            if not entry_id or entry_id in seen:
                continue
            alignment = tool_prefix_alignment(
                output.get("teacher_tool_events"),
                output.get("student_tool_events"),
                self.prefix_k,
            )
            if alignment >= 1.0:
                continue
            seen.add(entry_id)
            key = (data["eval_set_name"], data["eval_set_version"], tuple(data["deployment_ids"]))
            grouped.setdefault(key, []).append(entry_id)
        if grouped:
            count = sum(len(ids) for ids in grouped.values())
            print(f"[High-signal] Selected {count} alignment<1.0 entries for screening (prefix_k={self.prefix_k})")
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
        """Build reflection examples from the largest first-disagreement clusters.

        ``k`` and ``error_hamming_distance_k`` are ignored: the proposer only sees
        this frequency-capped mismatch set (at most 20 entries, or the full
        most-frequent group if that group is larger).
        """
        del k, error_hamming_distance_k
        if not eval_batch.trajectories:
            return {comp: [] for comp in components_to_update}

        trajectories = [cast(TeacherStudentALTrajectory, trajectory) for trajectory in eval_batch.trajectories]
        mismatch_keys = [
            first_disagreeing_mismatch(
                trajectory["output"].get("teacher_tool_events"),
                trajectory["output"].get("student_tool_events"),
                prefix_k=self.prefix_k,
            )
            for trajectory in trajectories
        ]
        selected_indices, selected_groups = select_first_tool_mismatch_groups(mismatch_keys)
        selected = [trajectories[index] for index in selected_indices]
        examples: dict[str, list[ReflectiveExample]] = {}
        for component_name in components_to_update:
            chosen = selected
            if component_name in CORE_TOOL_KEYS:
                chosen = [
                    trajectory
                    for trajectory in selected
                    if component_name
                    in {
                        tool_description_override_key(name)
                        for name in (
                            *prefix_tools(trajectory["output"].get("teacher_tool_events"), self.prefix_k),
                            *prefix_tools(trajectory["output"].get("student_tool_events"), self.prefix_k),
                        )
                    }
                ]
            elif component_name == RULES_EXT_KEY:
                chosen = []
                for trajectory in selected:
                    mismatch = first_disagreeing_mismatch(
                        trajectory["output"].get("teacher_tool_events"),
                        trajectory["output"].get("student_tool_events"),
                        prefix_k=self.prefix_k,
                    )
                    if mismatch is None:
                        continue
                    if any(is_core_tool_span(name) for name in mismatch[1:] if name):
                        continue
                    chosen.append(trajectory)
            examples[component_name] = [
                self._build_reflective_example(component_name, trajectory, candidate) for trajectory in chosen
            ]
        mismatch_count = sum(pair is not None for pair in mismatch_keys)
        selected_entry_ids = [
            str(trajectory["output"].get("entry_id", ""))
            for trajectory in selected
            if trajectory["output"].get("entry_id")
        ]
        log_section(
            "REFLECTION: teacher vs student tool sequences",
            format_eval_entry_report(trajectories),
        )
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

    def _create_failure_pattern(self, component_name: str, trajectory: TeacherStudentALTrajectory) -> tuple[Any, ...]:
        output = trajectory["output"]
        tool_alignment = trajectory.get("objective_scores", {}).get("tool_alignment", 1.0)
        return (
            int(tool_alignment < 0.7),
            int(
                first_tool_mismatch_pair(output.get("teacher_tool_events"), output.get("student_tool_events"))
                is not None
            ),
            int(output.get("student_tool_errors", 0) > 0),
        )

    def _build_reflective_example(
        self,
        component_name: str,
        trajectory: TeacherStudentALTrajectory,
        candidate: dict[str, str],
    ) -> ReflectiveExample:
        output = trajectory["output"]
        objective_scores = trajectory.get("objective_scores", {})
        tool_alignment = objective_scores.get("tool_alignment", trajectory["score"])
        completeness = objective_scores.get("completeness", 0.0)
        student_tools = output.get("student_tool_events", [])
        teacher_tools = output.get("teacher_tool_events", [])
        mismatch = first_disagreeing_mismatch(teacher_tools, student_tools, prefix_k=self.prefix_k)
        feedback_parts = []
        if mismatch is not None:
            slot, teacher_name, student_name = mismatch
            if slot == 0:
                feedback_parts.append(
                    f"First-tool mismatch: teacher used {teacher_name or '(none)'} "
                    f"and student used {student_name or '(none)'}."
                )
            else:
                feedback_parts.append(
                    f"Prefix mismatch at tool {slot + 1}: teacher used {teacher_name or '(none)'} "
                    f"and student used {student_name or '(none)'}."
                )
        if tool_alignment < 1.0:
            feedback_parts.append(f"Tool alignment issue: score={tool_alignment:.2f}.")
        if completeness < 0.7:
            feedback_parts.append(f"Completeness issue: score={completeness:.2f}.")

        inputs: ReflectiveExampleInputs = {
            "eval_set": trajectory["data"]["eval_set_name"],
            "entry_id": output["entry_id"],
            "deployment_id": output["deployment_id"],
            "query": output["query"],
        }
        return {
            "Inputs": inputs,
            "Generated Outputs": {
                "student_answer": output.get("student_answer", ""),
                "teacher_answer": output.get("teacher_answer", ""),
                "student_tools": student_tools,
                "teacher_tools": teacher_tools,
            },
            "Action Inputs": [],
            "Execution Errors": [],
            "Feedback": " ".join(feedback_parts) if feedback_parts else "General teacher/student tool divergence.",
            "Metrics": {
                "score": trajectory["score"],
                "tool_alignment": tool_alignment,
                "completeness": completeness,
            },
        }

    @staticmethod
    def _reflection_prompt(module_name: str) -> str:
        if module_name == FULL_PROMPT_KEY:
            return (
                "You are editing the ENTIRE student system prompt as a single string. Any section "
                "may change — routing, execution discipline, coding instructions, tool surface, "
                "or response guidelines — if it improves first-tool alignment with the teacher. "
                "Preserve [[placeholder]] tokens. Propose a complete updated prompt with minimal deltas."
            )
        if module_name == RULES_EXT_KEY:
            return (
                "You are writing at most two markdown bullets that will be appended after the existing "
                "**Rules:** list in Writing Code. Each line must start with '- '. Do not repeat those "
                "existing Rules, do not add a heading, and do not exceed two bullets. Target first-tool "
                "mismatches whose tools are not core tools (for example Write vs (none)). Keep each "
                "bullet operational and concise."
            )
        if module_name in CORE_TOOL_KEYS:
            return core_tool_reflection_prompt(module_name)
        return "Focus only on this module's responsibilities."

    @staticmethod
    def _format_reflective_metrics(metrics: ReflectiveExampleMetrics) -> str:
        return (
            f"score={metrics['score']:.2f}, tool_alignment={metrics.get('tool_alignment', metrics['score']):.2f}, "
            f"completeness={metrics.get('completeness', 0.0):.2f}"
        )

    def _finish_batch_evals(
        self,
        started: list[_StartedPair],
        capture_traces: bool,
        *,
        candidate: dict[str, str] | None = None,
        compared_entry_ids_by_version: dict[str, set[str]] | None = None,
    ) -> GleanEvaluationBatch[TeacherStudentALTrajectory, TeacherStudentALRolloutOutput]:
        all_outputs: list[TeacherStudentALRolloutOutput] = []
        all_scores: list[float] = []
        all_trajectories: list[TeacherStudentALTrajectory] | None = [] if capture_traces else None
        all_objective_scores: list[dict[str, float]] = []
        focused_alignment_rates: list[float] = []
        all_eval_run_ids: list[EvalRunIds] = []
        recorded_val_sequences = False

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
            tool_match_analysis = self._get_or_fetch_tool_match_analysis(pair.teacher_eval_id, pair.student_eval_id)
            version = str(al_data_inst.get("eval_set_version", ""))
            if compared_entry_ids_by_version is not None and version in compared_entry_ids_by_version:
                tool_match_analysis = restrict_tool_match_analysis(
                    tool_match_analysis, compared_entry_ids_by_version[version]
                )
            requested_entry_ids = al_data_inst.get("eval_entry_ids") or []
            is_focused_eval = bool(requested_entry_ids)
            if is_focused_eval:
                matching = sum(
                    1
                    for metrics in tool_match_analysis.per_entry.values()
                    if exact_prefix_match(metrics.teacher_tools, metrics.student_tools, self.prefix_k)
                )
                focused_alignment_rates.append(matching / len(requested_entry_ids))
                if not tool_match_analysis.per_entry:
                    continue
            else:
                require_compared_eval_entries(tool_match_analysis)
                log_tool_match_analysis(tool_match_analysis)
            deployment_id = (al_data_inst.get("deployment_ids") or [""])[0]
            query = f"{al_data_inst.get('eval_set_name', '')}:{al_data_inst.get('eval_set_version', '')}"
            student_completeness = self._judge_for(pair.student_eval_id, judge_type=COMPLETENESS_JUDGE_TYPE)
            teacher_completeness = self._judge_for(pair.teacher_eval_id, judge_type=COMPLETENESS_JUDGE_TYPE)
            print(
                f"[{COMPLETENESS_JUDGE_TYPE}] student {pair.student_eval_id}={student_completeness.aggregate:.2f} "
                f"teacher {pair.teacher_eval_id}={teacher_completeness.aggregate:.2f}"
            )

            record_val = (
                candidate is not None
                and not is_focused_eval
                and (not self.val_eval_versions or version in self.val_eval_versions)
            )
            if record_val:
                assert candidate is not None
                candidate_key = self._candidate_key(candidate)
                self._val_prefix_sequences.setdefault(candidate_key, {})[version] = [
                    (metrics.teacher_tools, metrics.student_tools) for metrics in tool_match_analysis.per_entry.values()
                ]
                recorded_val_sequences = True

            for entry_id, tool_match in tool_match_analysis.per_entry.items():
                student_tools = list(tool_match.student_tools)
                teacher_tools = list(tool_match.teacher_tools)
                tool_alignment = tool_prefix_alignment(teacher_tools, student_tools, self.prefix_k)
                completeness = student_completeness.per_entry.get(entry_id, student_completeness.aggregate)
                output: TeacherStudentALRolloutOutput = {
                    "deployment_id": deployment_id,
                    "query": query,
                    "student_answer": "",
                    "student_tool_events": student_tools,
                    "student_loops": 0,
                    "student_tool_calls": len(student_tools),
                    "student_tool_errors": 0,
                    "student_input_tokens": 0,
                    "student_output_tokens": 0,
                    "student_latency_ms": None,
                    "teacher_answer": "",
                    "teacher_tool_events": teacher_tools,
                    "teacher_loops": 0,
                    "teacher_tool_calls": len(teacher_tools),
                    "teacher_input_tokens": 0,
                    "teacher_output_tokens": 0,
                    "entry_id": entry_id,
                }
                all_outputs.append(output)
                score = COMPLETENESS_WEIGHT * completeness + TOOL_ALIGNMENT_WEIGHT * tool_alignment
                all_scores.append(score)
                objective_score = {
                    "completeness": completeness,
                    "tool_alignment": tool_alignment,
                }
                all_objective_scores.append(objective_score)
                if capture_traces and all_trajectories is not None:
                    trajectory: TeacherStudentALTrajectory = {
                        "data": al_data_inst,
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
                summary = {"completeness": 0.0}
            summary["tool_alignment"] = sum(focused_alignment_rates) / len(focused_alignment_rates)
        if summary is not None and started:
            teacher_scores = [
                self._judge_for(pair.teacher_eval_id, judge_type=COMPLETENESS_JUDGE_TYPE).aggregate for pair in started
            ]
            summary["teacher_completeness"] = sum(teacher_scores) / len(teacher_scores)
        if recorded_val_sequences:
            self._save_cache()

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
