"""Adapter for teacher-vs-student Glean evaluations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, cast

from glean_gepa.adapter_types import (
    ALDataInst,
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
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evalcli_client import COMPLETENESS_JUDGE_TYPE, COMPLETENESS_RUN_PARAMS
from glean_gepa.focused_evalset import resolve_eval_run_target
from glean_gepa.judge_metrics_util import (
    JudgeAnalysis,
    wait_for_judge_metrics,
)
from glean_gepa.prompt import FULL_PROMPT_KEY, WRITING_CODE_KEY, compile_encoded_prompt
from glean_gepa.tool_match_util import (
    EvalRunToolMatchAnalysis,
    empty_tool_match_analysis,
    fetch_eval_run_tool_match_analysis,
    log_tool_match_analysis,
    require_compared_eval_entries,
)

PRIMARY_OBJECTIVE = "tool_alignment"
FIXED_GROUNDING = 1.0
COMPLETENESS_WEIGHT = 0.5
TOOL_ALIGNMENT_WEIGHT = 0.3
GROUNDING_WEIGHT = 0.2
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
    ):
        self.teacher_model = teacher_model
        self.bigquery_client = bigquery_client
        self.agentspan_lookback_days = agentspan_lookback_days
        self._tool_match_cache: dict[tuple[str, str], EvalRunToolMatchAnalysis] = {}
        self._judge_runs: dict[tuple[str, str], str] = {}
        self._judge_cache: dict[tuple[str, str], JudgeAnalysis] = {}
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
        print(f"[Cache MISS] Cached tool match analysis for {teacher_eval_id} vs {student_eval_id}")
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
        cached = self._lookup_eval_id(cache_key)
        if cached is not None:
            eval_id, wait_required = cached
            hit_kind = "Soft cache HIT" if wait_required else "Cache HIT"
            print(f"[{hit_kind}] Using cached {role} eval_id: {eval_id} ({run_label})")
            return eval_id, wait_required

        eval_id, wait_required = self.runner.start(
            model,
            system_prompt=system_prompt,
            eval_set_name=eval_set_name,
            eval_set_version=eval_set_version,
            deployment_ids=deployment_ids,
            run_label=run_label,
        )
        self._record_eval_id(cache_key, eval_id, completed=not wait_required)
        if wait_required:
            print(f"[Cache MISS] Started {role} eval_id: {eval_id}")
        else:
            print(f"[Cache HIT] Using cached {role} eval_id: {eval_id}")
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
            target = resolve_eval_run_target(self.runner.evalcli, al_data_inst)
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
        for eval_id, (cache_key, role) in pending_waits.items():
            self.runner.wait(eval_id)
            self._record_eval_id(cache_key, eval_id, completed=True)
            print(f"[Cache MISS] Cached {role} eval_id: {eval_id}")
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
        return {"judge_runs": judge_runs, "judge_cache": judge_cache}

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
                per_entry = {
                    str(entry_id): float(score) for entry_id, score in (raw.get("per_entry") or {}).items()
                }
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
        return [self._finish_batch_evals(started, capture_traces) for started in all_started]

    @staticmethod
    def _check_primary_tool_mismatch(output: TeacherStudentALRolloutOutput) -> bool:
        student_tools = output.get("student_tool_events", [])
        teacher_tools = output.get("teacher_tool_events", [])
        if not student_tools or not teacher_tools:
            return len(student_tools) != len(teacher_tools)
        return student_tools[0] != teacher_tools[0]

    def _create_failure_pattern(self, component_name: str, trajectory: TeacherStudentALTrajectory) -> tuple[Any, ...]:
        output = trajectory["output"]
        tool_alignment = trajectory.get("objective_scores", {}).get("tool_alignment", 1.0)
        return (
            int(tool_alignment < 0.7),
            int(self._check_primary_tool_mismatch(output)),
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
        feedback_parts = []
        if self._check_primary_tool_mismatch(output):
            feedback_parts.append(f"Tool mismatch: student used {student_tools[:3]} vs teacher {teacher_tools[:3]}.")
        if tool_alignment < 0.7:
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
                "or response guidelines — if it improves tool-choice alignment with the teacher. "
                "Preserve [[placeholder]] tokens. Propose a complete updated prompt with minimal deltas."
            )
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
    ) -> GleanEvaluationBatch[TeacherStudentALTrajectory, TeacherStudentALRolloutOutput]:
        all_outputs: list[TeacherStudentALRolloutOutput] = []
        all_scores: list[float] = []
        all_trajectories: list[TeacherStudentALTrajectory] | None = [] if capture_traces else None
        all_objective_scores: list[dict[str, float]] = []
        focused_alignment_rates: list[float] = []

        for pair in started:
            al_data_inst = pair.al_data_inst
            tool_match_analysis = self._get_or_fetch_tool_match_analysis(pair.teacher_eval_id, pair.student_eval_id)
            requested_entry_ids = al_data_inst.get("eval_entry_ids") or []
            is_focused_eval = bool(requested_entry_ids)
            if is_focused_eval:
                matching = sum(1 for metrics in tool_match_analysis.per_entry.values() if metrics.tools_match)
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

            for entry_id, tool_match in tool_match_analysis.per_entry.items():
                student_tools = list(tool_match.student_tools)
                teacher_tools = list(tool_match.teacher_tools)
                tool_alignment = float(tool_match.tools_match) if is_focused_eval else tool_match.tool_match_score
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
                score = (
                    COMPLETENESS_WEIGHT * completeness
                    + TOOL_ALIGNMENT_WEIGHT * tool_alignment
                    + GROUNDING_WEIGHT * FIXED_GROUNDING
                )
                all_scores.append(score)
                objective_score = {
                    "completeness": completeness,
                    "tool_alignment": tool_alignment,
                    "grounding": FIXED_GROUNDING,
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
                summary = {"completeness": 0.0, "grounding": FIXED_GROUNDING}
            summary["tool_alignment"] = sum(focused_alignment_rates) / len(focused_alignment_rates)
        if summary is not None and started:
            teacher_scores = [
                self._judge_for(pair.teacher_eval_id, judge_type=COMPLETENESS_JUDGE_TYPE).aggregate
                for pair in started
            ]
            summary["teacher_completeness"] = sum(teacher_scores) / len(teacher_scores)

        return GleanEvaluationBatch(
            outputs=all_outputs,
            scores=all_scores,
            trajectories=all_trajectories,
            objective_scores=all_objective_scores,
            summary=summary,
        )


__all__ = [
    "TeacherStudentALDataInst",
    "TeacherStudentALRolloutOutput",
    "TeacherStudentALTrajectory",
    "TeacherStudentAdapter",
]
