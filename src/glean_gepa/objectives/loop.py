"""Loop-efficiency objective: fewer agent loops without dropping correctness."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date
from typing import Any, ClassVar

from glean_gepa.adapter_types import SingleModelALRolloutOutput, SingleModelALTrajectory
from glean_gepa.al_adapter import ReflectiveExample, ReflectiveExampleInputs, ReflectiveExampleMetrics
from glean_gepa.focused_evalset import QUERY_CANONICAL_BUCKET_TYPE
from glean_gepa.objectives.base import ScoredRow, SingleModelObjective, register_telemetry_source
from glean_gepa.objectives.utils.loop_count_util import (
    CORRECTNESS_PASS,
    LOOP_EFFICIENCY_OBJECTIVE,
    TARGET_LOOP_COUNT,
    EvalRunLoopCountAnalysis,
    LoopCountEntryMetrics,
    aggregate_loop_count_metrics,
    fetch_eval_run_loop_count_analysis,
    log_loop_count_analysis,
)
from glean_gepa.prompt_constants import WRITING_CODE_KEY
from glean_gepa.reflection_prompts import CONDITIONAL_PRESERVE_RULE

WRITING_CODE_RESPONSIBILITY = (
    "Focus ONLY on coding and execution-discipline instructions that reduce extra agent loops "
    "without lowering correctness. Batch independent SDK calls, stop once the answer is grounded, "
    "and never skip the search or citation work the question requires. "
    f"{CONDITIONAL_PRESERVE_RULE} Propose minimal deltas."
)

EVAL_LOOP_CACHE_SCHEMA_VERSION = 2


def _serialize_eval_analysis(analysis: EvalRunLoopCountAnalysis) -> dict[str, Any]:
    return {
        "schema_version": EVAL_LOOP_CACHE_SCHEMA_VERSION,
        "eval_id": analysis.eval_id,
        "start_date": analysis.start_date.isoformat(),
        "end_date": analysis.end_date.isoformat(),
        "aggregate": asdict(analysis.aggregate),
        "per_entry": {entry_id: asdict(metrics) for entry_id, metrics in analysis.per_entry.items()},
        "high_signal_entry_ids": list(analysis.high_signal_entry_ids),
    }


def _parse_eval_analysis_cache(
    raw_cache: Any,
    *,
    target_loops: int = TARGET_LOOP_COUNT,
    correctness_pass: float = CORRECTNESS_PASS,
) -> dict[str, EvalRunLoopCountAnalysis]:
    parsed: dict[str, EvalRunLoopCountAnalysis] = {}
    if not isinstance(raw_cache, dict):
        return parsed
    for eval_id, raw in raw_cache.items():
        try:
            if not isinstance(raw, dict) or raw.get("schema_version") != EVAL_LOOP_CACHE_SCHEMA_VERSION:
                continue
            per_entry = {
                str(entry_id): LoopCountEntryMetrics(
                    entry_id=str(metrics.get("entry_id") or entry_id),
                    loop_count=int(metrics.get("loop_count") or 0),
                    correctness=float(metrics.get("correctness") or 0.0),
                    has_error=bool(metrics.get("has_error")),
                    action_inputs=tuple(metrics.get("action_inputs") or ()),
                    target_loops=target_loops,
                    correctness_pass=correctness_pass,
                )
                for entry_id, metrics in (raw.get("per_entry") or {}).items()
                if isinstance(metrics, dict)
            }
            parsed[str(eval_id)] = EvalRunLoopCountAnalysis(
                eval_id=str(raw.get("eval_id") or eval_id),
                start_date=date.fromisoformat(raw["start_date"]),
                end_date=date.fromisoformat(raw["end_date"]),
                aggregate=aggregate_loop_count_metrics(str(raw.get("eval_id") or eval_id), per_entry),
                per_entry=per_entry,
                high_signal_entry_ids=tuple(
                    sorted(entry_id for entry_id, metrics in per_entry.items() if metrics.loop_efficiency < 1.0)
                ),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return parsed


def _rollout_output(
    *,
    deployment_id: str,
    query: str,
    entry_id: str,
    student_eval_id: str,
    loop_count: int,
    correctness: float,
    action_inputs: list[str] | None = None,
    correctness_pass: float = CORRECTNESS_PASS,
) -> SingleModelALRolloutOutput:
    output: SingleModelALRolloutOutput = {
        "deployment_id": deployment_id,
        "query": query,
        "student_tool_calls": loop_count,
        "student_tool_errors": int(correctness < correctness_pass),
        "entry_id": entry_id,
        "shell_error_messages": [],
        "student_eval_run_id": student_eval_id,
        "student_loops": loop_count,
        "correctness": correctness,
    }
    if action_inputs:
        output["action_inputs"] = list(action_inputs)
    return output


class LoopEfficiencyObjective(SingleModelObjective):
    """Score student evals by inverted loop count, gated on correctness."""

    name = LOOP_EFFICIENCY_OBJECTIVE
    telemetry_dimensions = (LOOP_EFFICIENCY_OBJECTIVE,)
    focused_bucket_type = QUERY_CANONICAL_BUCKET_TYPE
    failure_label = "HIGH-SIGNAL FAILURES (extra loops or incorrect)"
    pending_telemetry_label = "loop_efficiency"
    module_responsibilities: ClassVar[Mapping[str, str]] = {
        WRITING_CODE_KEY: WRITING_CODE_RESPONSIBILITY,
    }

    def __init__(self, *, bigquery_client: Any | None = None, lookback_days: int = 1):
        if bigquery_client is None:
            raise ValueError("bigquery_client is required")
        self.bigquery_client = bigquery_client
        self.lookback_days = lookback_days
        self.params: dict[str, Any] = {}
        self._eval_analysis_cache: dict[str, EvalRunLoopCountAnalysis] = {}

    def _target_loop_count(self) -> int:
        return int(self.pack_param("target_loop_count", TARGET_LOOP_COUNT))

    def _correctness_pass(self) -> float:
        return float(self.pack_param("correctness_pass", CORRECTNESS_PASS))

    def analyze(
        self,
        eval_id: str,
        *,
        include_error_examples: bool = True,
        include_per_entry: bool = True,
        evalcli: Any | None = None,
        include_action_inputs: bool = True,
    ) -> EvalRunLoopCountAnalysis:
        del include_error_examples
        cached = self._eval_analysis_cache.get(eval_id)
        if cached is not None:
            missing_entry_breakdown = (
                include_per_entry and not cached.per_entry and cached.aggregate.compared_entries > 0
            )
            if not missing_entry_breakdown:
                print(f"[Cache HIT] Using cached loop-count analysis for eval_id: {eval_id}")
                return cached
            print(f"[Cache] Refetching loop-count analysis with per-entry metrics for eval_id: {eval_id}")
        analysis = fetch_eval_run_loop_count_analysis(
            self.bigquery_client,
            eval_id=eval_id,
            lookback_days=self.lookback_days,
            evalcli=evalcli,
            include_action_inputs=include_action_inputs,
            target_loops=self._target_loop_count(),
            correctness_pass=self._correctness_pass(),
        )
        if analysis.aggregate.compared_entries == 0:
            print(f"[Cache] Not caching provisional empty loop analysis for eval_id: {eval_id}")
            return analysis
        self._eval_analysis_cache[eval_id] = analysis
        return analysis

    def is_pending(self, analysis: EvalRunLoopCountAnalysis) -> bool:
        return analysis.aggregate.compared_entries == 0

    def aggregate_score(self, analysis: EvalRunLoopCountAnalysis) -> float:
        return analysis.aggregate.loop_efficiency

    def focused_pass_rate(self, analysis: EvalRunLoopCountAnalysis, requested_entry_ids: Sequence[str]) -> float:
        passed = sum(
            1
            for entry_id in requested_entry_ids
            if (metrics := analysis.per_entry.get(entry_id)) is not None and metrics.loop_efficiency >= 1.0
        )
        return passed / len(requested_entry_ids)

    def entry_ids_to_score(
        self, analysis: EvalRunLoopCountAnalysis, requested_entry_ids: Sequence[str] | None
    ) -> tuple[str, ...]:
        if requested_entry_ids:
            return tuple(analysis.per_entry) or tuple(requested_entry_ids)
        return analysis.high_signal_entry_ids

    def log_analysis(self, analysis: EvalRunLoopCountAnalysis) -> None:
        log_loop_count_analysis(analysis)

    def scored_rows(
        self,
        analysis: EvalRunLoopCountAnalysis,
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
        del al_data_inst
        query = f"{eval_set_name}:{eval_set_version}"
        deployment_id = deployment_ids[0] if deployment_ids else ""
        correctness_pass = self._correctness_pass()
        if not is_focused_eval and not capture_traces:
            return [
                ScoredRow(
                    entry_id=None,
                    dimension_scores={LOOP_EFFICIENCY_OBJECTIVE: analysis.aggregate.loop_efficiency},
                    output=_rollout_output(
                        deployment_id=deployment_id,
                        query=query,
                        entry_id=query,
                        student_eval_id=student_eval_id,
                        loop_count=round(analysis.aggregate.mean_loop_count),
                        correctness=analysis.aggregate.mean_correctness,
                        correctness_pass=correctness_pass,
                    ),
                )
            ]

        high_signal_entry_ids = self.entry_ids_to_score(analysis, requested_entry_ids)
        if not high_signal_entry_ids:
            return [
                ScoredRow(
                    entry_id=query,
                    dimension_scores={LOOP_EFFICIENCY_OBJECTIVE: analysis.aggregate.loop_efficiency},
                    output=_rollout_output(
                        deployment_id=deployment_id,
                        query=query,
                        entry_id=query,
                        student_eval_id=student_eval_id,
                        loop_count=round(analysis.aggregate.mean_loop_count),
                        correctness=analysis.aggregate.mean_correctness,
                        correctness_pass=correctness_pass,
                    ),
                )
            ]

        rows: list[ScoredRow] = []
        for entry_id in high_signal_entry_ids:
            metrics = analysis.per_entry.get(entry_id)
            loop_count = metrics.loop_count if metrics else 0
            correctness = metrics.correctness if metrics else 0.0
            efficiency = (
                float(metrics.loop_efficiency >= 1.0)
                if is_focused_eval and metrics is not None
                else (metrics.loop_efficiency if metrics else 0.0)
            )
            rows.append(
                ScoredRow(
                    entry_id=entry_id,
                    dimension_scores={LOOP_EFFICIENCY_OBJECTIVE: efficiency},
                    output=_rollout_output(
                        deployment_id=deployment_id,
                        query=f"{query} entry={entry_id}",
                        entry_id=entry_id,
                        student_eval_id=student_eval_id,
                        loop_count=loop_count,
                        correctness=correctness,
                        action_inputs=list(metrics.action_inputs) if metrics else None,
                        correctness_pass=correctness_pass,
                    ),
                    data_overrides={"eval_entry_id": entry_id, "eval_run_id": student_eval_id},
                )
            )
        return rows

    def prepare_focused_source_entries(
        self,
        *,
        eval_set_name: str,
        eval_set_version: str,
        eval_run_id: str,
        entry_ids: Sequence[str],
        deployment_ids: Sequence[str],
    ) -> list[dict[str, Any]] | None:
        del eval_set_name, eval_set_version, eval_run_id, entry_ids, deployment_ids
        return None

    def failure_pattern(self, component_name: str, trajectory: SingleModelALTrajectory) -> tuple[Any, ...]:
        del component_name
        output = trajectory["output"]
        efficiency = trajectory.get("objective_scores", {}).get(self.name, 1.0)
        target_loops = self._target_loop_count()
        return (
            int(efficiency < float(self.pack_param("failure_score_below", 1.0))),
            int((output.get("correctness") or 1.0) < self._correctness_pass()),
            int((output.get("student_loops") or 0) > target_loops),
        )

    def build_reflective_example(
        self,
        component_name: str,
        trajectory: SingleModelALTrajectory,
        candidate: dict[str, str],
    ) -> ReflectiveExample:
        del component_name, candidate
        output = trajectory["output"]
        efficiency = trajectory.get("objective_scores", {}).get(self.name, 1.0)
        loops = int(output.get("student_loops") or 0)
        correctness = float(output.get("correctness") or 0.0)
        target_loops = self._target_loop_count()
        feedback_parts = []
        if correctness < self._correctness_pass():
            feedback_parts.append(
                f"Incorrect answer (correctness={correctness:.2f}); do not reduce loops by skipping work."
            )
        if loops > target_loops:
            feedback_parts.append(
                f"Used {loops} loops; stay at or below {target_loops} by batching independent calls "
                "and stopping once the answer is grounded."
            )
        if not feedback_parts:
            feedback_parts.append("Reduce extra agent loops without lowering correctness.")

        inputs: ReflectiveExampleInputs = {
            "eval_set": trajectory["data"]["eval_set_name"],
            "entry_id": output["entry_id"],
            "deployment_id": output["deployment_id"],
            "query": output["query"],
        }
        if eval_run_id := trajectory["data"].get("eval_run_id"):
            inputs["eval_run_id"] = eval_run_id
        # The student's own tool payloads across loops show what work it did,
        # which is the per-entry context the scrubbed query field cannot give.
        action_inputs = output.get("action_inputs") or []
        return {
            "Inputs": inputs,
            "Generated Outputs": {
                "student_answer": "",
                "teacher_answer": "",
                "student_tools": [],
                "teacher_tools": [],
            },
            "Action Inputs": list(action_inputs[:5]),
            "Execution Errors": [],
            "Feedback": " ".join(feedback_parts),
            "Metrics": {
                "score": trajectory["score"],
                "loop_efficiency": efficiency,
                "correctness": correctness,
            },
        }

    def format_reflective_metrics(self, metrics: ReflectiveExampleMetrics) -> str | None:
        return (
            f"score={metrics['score']:.2f}, "
            f"loop_efficiency={metrics.get('loop_efficiency', metrics['score']):.2f}, "
            f"correctness={metrics.get('correctness', 0.0):.2f}"
        )

    def cache_payload(self) -> dict[str, Any]:
        return {eval_id: _serialize_eval_analysis(analysis) for eval_id, analysis in self._eval_analysis_cache.items()}

    def load_cache(self, raw_cache: Any) -> None:
        self._eval_analysis_cache = _parse_eval_analysis_cache(
            raw_cache,
            target_loops=self._target_loop_count(),
            correctness_pass=self._correctness_pass(),
        )


register_telemetry_source("single_model", "loop_telemetry", LoopEfficiencyObjective)

__all__ = [
    "EVAL_LOOP_CACHE_SCHEMA_VERSION",
    "LoopEfficiencyObjective",
]
