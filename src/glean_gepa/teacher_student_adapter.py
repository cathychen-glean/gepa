"""Adapter for teacher-vs-student Glean evaluations."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from glean_gepa.adapter_types import (
    ALDataInst,
    ALTrajectory,
    PointwiseJudge,
    TeacherStudentALDataInst,
    TeacherStudentALRolloutOutput,
    TeacherStudentALTrajectory,
)
from glean_gepa.al_adapter import (
    ALRunner,
    GleanAdapterBase,
    ReflectiveExample,
    ReflectiveSelection,
    Thresholds,
)
from glean_gepa.batch import EvalRunIds, GleanEvaluationBatch
from glean_gepa.evalcli_client import COMPLETENESS_JUDGE_TYPE, COMPLETENESS_RUN_PARAMS
from glean_gepa.focused_evalset import QUERY_CANONICAL_BUCKET_TYPE, resolve_eval_run_target
from glean_gepa.judge_metrics_util import (
    JudgeAnalysis,
    wait_for_judge_metrics,
)
from glean_gepa.prompt import compile_encoded_prompt, is_core_tool_span, tool_description_override_key
from glean_gepa.prompt_constants import CORE_TOOL_KEYS, RULES_EXT_KEY, WRITING_CODE_KEY
from glean_gepa.reflection_examples import (
    build_teacher_student_reflective_example,
    format_teacher_student_reflective_metrics,
)
from glean_gepa.reflection_prompts import teacher_student_reflection_prompt
from glean_gepa.run_log import (
    format_high_signal_selection_report,
    selected_entry_ids_from_examples,
)
from glean_gepa.tool_match_util import (
    TOOL_ALIGNMENT_OBJECTIVE,
    EvalRunTeacherStudentMatchAnalysis,
    empty_teacher_student_match_analysis,
    fetch_eval_run_teacher_student_match_analysis,
    first_tool_mismatch_pair,
    focused_match_rate,
    log_teacher_student_match_analysis,
    parse_teacher_student_match_analysis,
    require_compared_eval_entries,
    select_first_tool_mismatch_groups,
    serialize_teacher_student_match_analysis,
    tool_alignment_entries,
)

PRIMARY_OBJECTIVE = TOOL_ALIGNMENT_OBJECTIVE
COMPLETENESS_DIMENSION = "completeness"
GROUNDING_DIMENSION = "grounding"
FIXED_GROUNDING = 1.0
COMPLETENESS_WEIGHT = 0.5
TOOL_ALIGNMENT_WEIGHT = 0.3
GROUNDING_WEIGHT = 0.2
POINTWISE_JUDGES: tuple[PointwiseJudge, ...] = (
    PointwiseJudge(COMPLETENESS_DIMENSION, COMPLETENESS_JUDGE_TYPE, COMPLETENESS_RUN_PARAMS),
)
DEFAULT_COMPOSITE_WEIGHTS = {
    COMPLETENESS_DIMENSION: COMPLETENESS_WEIGHT,
    TOOL_ALIGNMENT_OBJECTIVE: TOOL_ALIGNMENT_WEIGHT,
    GROUNDING_DIMENSION: GROUNDING_WEIGHT,
}
DEFAULT_CONSTANT_SCORES = {GROUNDING_DIMENSION: FIXED_GROUNDING}


@dataclass(frozen=True)
class _StartedPair:
    al_data_inst: TeacherStudentALDataInst
    teacher_eval_id: str
    student_eval_id: str


class TeacherStudentAdapter(GleanAdapterBase):
    """Optimize instructions from teacher-vs-student tool-usage comparisons."""

    supports_high_signal_eval = True
    telemetry_dimensions = (TOOL_ALIGNMENT_OBJECTIVE,)

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
        primary_objective: str = PRIMARY_OBJECTIVE,
        default_frontier_type: str = "hybrid",
        composite_weights: dict[str, float] | None = None,
        pointwise_judges: Sequence[PointwiseJudge] | None = None,
        constant_scores: dict[str, float] | None = None,
    ):
        self.teacher_model = teacher_model
        self.bigquery_client = bigquery_client
        self.agentspan_lookback_days = agentspan_lookback_days
        # Set before super().__init__ so the base can validate composite weights
        # against the judge dimensions scorable_dimensions() adds below.
        self.pointwise_judges = tuple(pointwise_judges) if pointwise_judges is not None else POINTWISE_JUDGES
        self._teacher_student_match_cache: dict[
            tuple[str, str, tuple[tuple[str, str], ...]], EvalRunTeacherStudentMatchAnalysis
        ] = {}
        self._judge_runs: dict[tuple[str, str], str] = {}
        self._judge_cache: dict[tuple[str, str], JudgeAnalysis] = {}
        super().__init__(
            runner=runner,
            thresholds=thresholds,
            student_model=student_model,
            evaluate_fn=self._evaluate_teacher_student,
            failure_pattern_fn=self._create_failure_pattern,
            reflective_example_fn=build_teacher_student_reflective_example,
            reflection_prompt_fn=teacher_student_reflection_prompt,
            reflective_metrics_fn=format_teacher_student_reflective_metrics,
            failure_label="HIGH-SIGNAL FAILURES (teacher vs student)",
            primary_objective=primary_objective,
            default_frontier_type=default_frontier_type,
            editable_modules=list(editable_modules) if editable_modules else [WRITING_CODE_KEY],
            composite_weights=composite_weights if composite_weights is not None else DEFAULT_COMPOSITE_WEIGHTS,
            constant_scores=constant_scores if constant_scores is not None else DEFAULT_CONSTANT_SCORES,
            cache_file=cache_file,
            diversify_reflective_examples=False,
            dedupe_reflective_examples=False,
        )

    def scorable_dimensions(self) -> set[str]:
        return super().scorable_dimensions() | {judge.name for judge in self.pointwise_judges}

    def _get_or_fetch_teacher_student_match_analysis(
        self,
        teacher_eval_id: str,
        student_eval_id: str,
        *,
        entry_id_pairs: Sequence[tuple[str, str]] | None = None,
    ) -> EvalRunTeacherStudentMatchAnalysis:
        # Focused runs remap entry ids, so the pairs belong in the key.
        cache_key = (teacher_eval_id, student_eval_id, tuple(sorted(entry_id_pairs or ())))
        cached = self._cached_analysis(self._teacher_student_match_cache, cache_key)
        if cached is not None:
            print(f"[Cache HIT] Using cached teacher-student analysis for {teacher_eval_id} vs {student_eval_id}")
            return cached
        if self.bigquery_client is None:
            analysis = empty_teacher_student_match_analysis(teacher_eval_id, student_eval_id)
        else:
            analysis = fetch_eval_run_teacher_student_match_analysis(
                self.bigquery_client,
                teacher_eval_id=teacher_eval_id,
                student_eval_id=student_eval_id,
                lookback_days=self.agentspan_lookback_days,
                entry_id_pairs=cache_key[2] or None,
            )
        self._store_analysis(self._teacher_student_match_cache, cache_key, analysis)
        print(f"[Cache MISS] Fetched teacher-student analysis for {teacher_eval_id} vs {student_eval_id}")
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
        self._wait_pending_evals(pending_waits)
        self._run_judges(started)
        return self._finish_batch_evals(started, capture_traces)

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
                bucket_type=QUERY_CANONICAL_BUCKET_TYPE,
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
            source_teacher_eval_id = al_data_inst.get("source_teacher_eval_run_id")
            teacher_eval_id = source_teacher_eval_id or al_data_inst.get("cached_teacher_eval_run_id")
            wait_teacher = False
            if target.is_focused:
                if not source_teacher_eval_id:
                    raise RuntimeError(
                        "High-signal teacher-student screening needs the parent teacher eval id; "
                        "attach source_teacher_eval_run_id in high_signal_batch."
                    )
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
            for judge in self.pointwise_judges:
                self._ensure_judge(eval_id, judge_type=judge.judge_type, run_params=judge.run_params)

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
        # Stored as records rather than a keyed map: the cache key is a tuple that
        # includes the entry-id pairs, which JSON cannot use as an object key. The
        # "tool_match_cache" key keeps its original name so caches already on disk
        # still load after the rename.
        tool_match_cache = [
            {
                "teacher_eval_id": teacher_eval_id,
                "student_eval_id": student_eval_id,
                "entry_id_pairs": [list(pair) for pair in pairs],
                "analysis": serialize_teacher_student_match_analysis(analysis),
            }
            for (teacher_eval_id, student_eval_id, pairs), analysis in self._teacher_student_match_cache.items()
        ]
        return {"judge_runs": judge_runs, "judge_cache": judge_cache, "tool_match_cache": tool_match_cache}

    def _load_extra_cache(self, data: dict[str, Any]) -> None:
        self._teacher_student_match_cache = {}
        for record in data.get("tool_match_cache") or []:
            if not isinstance(record, dict):
                continue
            analysis = parse_teacher_student_match_analysis(record.get("analysis"))
            if analysis is None:
                continue
            pairs = [(str(pair[0]), str(pair[1])) for pair in record.get("entry_id_pairs") or [] if len(pair) == 2]
            key = (
                str(record.get("teacher_eval_id") or ""),
                str(record.get("student_eval_id") or ""),
                tuple(sorted(pairs)),
            )
            self._teacher_student_match_cache[key] = analysis

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
                for judge in self.pointwise_judges:
                    cache_key = (eval_id, judge.judge_type)
                    if cache_key in seen:
                        continue
                    seen.add(cache_key)
                    if cache_key in self._judge_cache:
                        continue
                    judge_run_id = self._ensure_judge(eval_id, judge_type=judge.judge_type, run_params=judge.run_params)
                    pending.append((eval_id, judge.judge_type, judge_run_id))
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
        return [self._finish_batch_evals(started, capture_traces) for started in all_started]

    def high_signal_batch(self, eval_batch: GleanEvaluationBatch) -> list[ALDataInst]:
        """Keep every parent entry whose first scored tools disagree."""
        grouped: dict[tuple[str, str, tuple[str, ...]], list[str]] = {}
        seen: set[str] = set()
        teacher_by_eval_set: dict[tuple[str, str], str] = {}
        for record in eval_batch.eval_run_ids or []:
            teacher_eval_id = record.get("teacher_eval_run_id")
            if teacher_eval_id:
                teacher_by_eval_set[(record["eval_set_name"], record["eval_set_version"])] = teacher_eval_id
        for trajectory in eval_batch.trajectories or []:
            data = trajectory["data"]
            output = trajectory["output"]
            teacher_eval_id = output.get("teacher_eval_run_id")
            if teacher_eval_id:
                teacher_by_eval_set.setdefault((data["eval_set_name"], data["eval_set_version"]), teacher_eval_id)
            entry_id = output.get("entry_id")
            if not entry_id or entry_id in seen:
                continue
            if (
                first_tool_mismatch_pair(
                    output.get("teacher_tool_events"),
                    output.get("student_tool_events"),
                )
                is None
            ):
                continue
            seen.add(entry_id)
            key = (data["eval_set_name"], data["eval_set_version"], tuple(data["deployment_ids"]))
            grouped.setdefault(key, []).append(entry_id)
        if grouped:
            count = sum(len(ids) for ids in grouped.values())
            print(f"[High-signal] Selected {count} first-tool mismatch entries for screening")
        batch: list[ALDataInst] = []
        for (eval_set_name, eval_set_version, deployment_ids), entry_ids in grouped.items():
            item: dict[str, Any] = {
                "eval_set_name": eval_set_name,
                "eval_set_version": eval_set_version,
                "deployment_ids": list(deployment_ids),
                "status": "active",
                "eval_entry_ids": entry_ids,
            }
            teacher_eval_id = teacher_by_eval_set.get((eval_set_name, eval_set_version))
            if teacher_eval_id:
                item["source_teacher_eval_run_id"] = teacher_eval_id
            batch.append(cast(ALDataInst, item))
        return batch

    def _reflective_trajectory_pool(self, trajectories: list[ALTrajectory]) -> ReflectiveSelection:
        """Keep the most frequent first-tool mismatch groups (at most 20 entries)."""
        mismatch_keys = [
            first_tool_mismatch_pair(
                trajectory["output"].get("teacher_tool_events"),
                trajectory["output"].get("student_tool_events"),
            )
            for trajectory in trajectories
        ]
        selected_indices, selected_groups = select_first_tool_mismatch_groups(mismatch_keys)
        selected = [trajectories[index] for index in selected_indices]
        selected_pairs = [mismatch_keys[index] for index in selected_indices]
        selected_entry_ids = [
            str(trajectory["output"].get("entry_id", ""))
            for trajectory in selected
            if trajectory["output"].get("entry_id")
        ]
        return ReflectiveSelection(
            trajectories=selected,
            metadata={
                "selected_pairs": selected_pairs,
                "selected_groups": selected_groups,
                "selected_entry_ids": selected_entry_ids,
                "mismatch_count": sum(pair is not None for pair in mismatch_keys),
            },
        )

    def _filter_reflective_trajectories(
        self,
        component_name: str,
        selection: ReflectiveSelection,
    ) -> list[ALTrajectory]:
        selected = selection.trajectories
        selected_pairs = selection.metadata.get("selected_pairs") or [None] * len(selected)
        if component_name in CORE_TOOL_KEYS:
            return [
                trajectory
                for trajectory, pair in zip(selected, selected_pairs, strict=True)
                if pair is not None
                and any(tool_description_override_key(name) == component_name for name in pair if name)
            ]
        if component_name == RULES_EXT_KEY:
            return [
                trajectory
                for trajectory, pair in zip(selected, selected_pairs, strict=True)
                if pair is not None and not any(is_core_tool_span(name) for name in pair if name)
            ]
        return selected

    def _reflective_eval_entries_log_title(self) -> str:
        return "REFLECTION: teacher vs student tool sequences"

    def _reflective_dataset_justification(
        self,
        *,
        k: int | None,
        error_hamming_distance_k: int | None,
        selection: ReflectiveSelection,
        examples: dict[str, list[ReflectiveExample]],
    ) -> str:
        del k, error_hamming_distance_k
        return format_high_signal_selection_report(
            selected_groups=selection.metadata.get("selected_groups") or [],
            selected_entry_ids=selection.metadata.get("selected_entry_ids") or [],
            selected_count=len(selection.trajectories),
            total_mismatch_count=int(selection.metadata.get("mismatch_count") or 0),
            module_entry_ids={
                module: selected_entry_ids_from_examples(module_examples)
                for module, module_examples in examples.items()
            },
        )

    def _create_failure_pattern(self, component_name: str, trajectory: TeacherStudentALTrajectory) -> tuple[Any, ...]:
        output = trajectory["output"]
        tool_alignment = trajectory.get("objective_scores", {}).get(TOOL_ALIGNMENT_OBJECTIVE, 1.0)
        return (
            int(tool_alignment < 0.7),
            int(
                first_tool_mismatch_pair(output.get("teacher_tool_events"), output.get("student_tool_events"))
                is not None
            ),
            int(output.get("student_tool_errors", 0) > 0),
        )

    def _finish_batch_evals(
        self,
        started: list[_StartedPair],
        capture_traces: bool,
    ) -> GleanEvaluationBatch[TeacherStudentALTrajectory, TeacherStudentALRolloutOutput]:
        """Compose per-entry telemetry, judges, and constants into scored results.

        Objective-specific work is delegated to a producer such as
        ``tool_alignment_entries``; this method stays generic over whatever
        dimensions those producers report.
        """
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
            teacher_student_match_analysis = self._get_or_fetch_teacher_student_match_analysis(
                pair.teacher_eval_id,
                pair.student_eval_id,
                entry_id_pairs=_entry_id_pairs(al_data_inst),
            )
            requested_entry_ids = al_data_inst.get("eval_entry_ids") or []
            is_focused_eval = bool(requested_entry_ids)
            if is_focused_eval:
                focused_alignment_rates.append(focused_match_rate(teacher_student_match_analysis, requested_entry_ids))
                if not teacher_student_match_analysis.per_entry:
                    continue
            else:
                require_compared_eval_entries(teacher_student_match_analysis)
                log_teacher_student_match_analysis(teacher_student_match_analysis)
            deployment_id = (al_data_inst.get("deployment_ids") or [""])[0]
            query = f"{al_data_inst.get('eval_set_name', '')}:{al_data_inst.get('eval_set_version', '')}"
            student_judges = {
                judge.name: self._judge_for(pair.student_eval_id, judge_type=judge.judge_type)
                for judge in self.pointwise_judges
            }
            for judge in self.pointwise_judges:
                teacher_analysis = self._judge_for(pair.teacher_eval_id, judge_type=judge.judge_type)
                print(
                    f"[{judge.judge_type}] student {pair.student_eval_id}="
                    f"{student_judges[judge.name].aggregate:.2f} "
                    f"teacher {pair.teacher_eval_id}={teacher_analysis.aggregate:.2f}"
                )

            entries = tool_alignment_entries(
                teacher_student_match_analysis,
                teacher_eval_id=pair.teacher_eval_id,
                student_eval_id=pair.student_eval_id,
                deployment_id=deployment_id,
                query=query,
            )
            for entry in entries:
                all_outputs.append(entry.output)
                # Report every dimension this adapter can resolve, but weight only
                # the ones objective.composite asks for.
                objective_score = {
                    **self.constant_scores,
                    **entry.objective_scores,
                    **{
                        name: analysis.per_entry.get(entry.student_entry_id, analysis.aggregate)
                        for name, analysis in student_judges.items()
                    },
                }
                score = self.composite_score(objective_score)
                all_scores.append(score)
                all_objective_scores.append(objective_score)
                if capture_traces and all_trajectories is not None:
                    trajectory: TeacherStudentALTrajectory = {
                        "data": al_data_inst,
                        "output": entry.output,
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
                summary = {judge.name: 0.0 for judge in self.pointwise_judges}
                summary.update(self.constant_scores)
            summary[TOOL_ALIGNMENT_OBJECTIVE] = sum(focused_alignment_rates) / len(focused_alignment_rates)
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


def _entry_id_pairs(al_data_inst: TeacherStudentALDataInst) -> list[tuple[str, str]] | None:
    mapping = al_data_inst.get("source_entry_ids_by_focused_id") or {}
    if not mapping:
        return None
    return [(source_id, focused_id) for focused_id, source_id in mapping.items()]


__all__ = [
    "TeacherStudentALDataInst",
    "TeacherStudentALRolloutOutput",
    "TeacherStudentALTrajectory",
    "TeacherStudentAdapter",
]
