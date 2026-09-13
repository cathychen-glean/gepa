"""Loop-efficiency objective: fewer agent loops without dropping correctness."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date
from typing import Any

from glean_gepa.adapter_types import SingleModelALRolloutOutput, SingleModelALTrajectory
from glean_gepa.al_adapter import ReflectiveExample, ReflectiveExampleInputs, ReflectiveExampleMetrics
from glean_gepa.focused_evalset import QUERY_CANONICAL_BUCKET_TYPE
from glean_gepa.objectives.base import ScoredRow, SingleModelObjective, register_telemetry_source
from glean_gepa.objectives.utils.loop_count_util import (
    LOOP_EFFICIENCY_OBJECTIVE,
    TARGET_LOOP_COUNT,
    EvalRunLoopCountAnalysis,
    LoopCountEntryMetrics,
    LoopCountMetrics,
    fetch_eval_run_loop_count_analysis,
    log_loop_count_analysis,
)
from glean_gepa.reflection_prompts import single_model_loop_reflection_prompt

EVAL_LOOP_CACHE_SCHEMA_VERSION = 2
CORRECTNESS_PASS_FOR_FEEDBACK = 0.5


class LoopTelemetryPendingError(RuntimeError):
    """Raised when an eval has not yet emitted loop-count telemetry."""


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


def _parse_eval_analysis_cache(raw_cache: Any) -> dict[str, EvalRunLoopCountAnalysis]:
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
                )
                for entry_id, metrics in (raw.get("per_entry") or {}).items()
                if isinstance(metrics, dict)
            }
            aggregate_raw = raw.get("aggregate") or {}
            parsed[str(eval_id)] = EvalRunLoopCountAnalysis(
                eval_id=str(raw.get("eval_id") or eval_id),
                start_date=date.fromisoformat(raw["start_date"]),
                end_date=date.fromisoformat(raw["end_date"]),
                aggregate=LoopCountMetrics(
                    eval_id=str(aggregate_raw.get("eval_id") or eval_id),
                    compared_entries=int(aggregate_raw.get("compared_entries") or 0),
                    matching_entries=int(aggregate_raw.get("matching_entries") or 0),
                    mean_loop_count=float(aggregate_raw.get("mean_loop_count") or 0.0),
                    loop_efficiency=float(aggregate_raw.get("loop_efficiency") or 0.0),
                    mean_correctness=float(aggregate_raw.get("mean_correctness") or 0.0),
                ),
                per_entry=per_entry,
                high_signal_entry_ids=tuple(raw.get("high_signal_entry_ids") or ()),
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
) -> SingleModelALRolloutOutput:
    output: SingleModelALRolloutOutput = {
        "deployment_id": deployment_id,
        "query": query,
        "student_tool_calls": loop_count,
        "student_tool_errors": int(correctness < 0.5),
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
    pending_error_type = LoopTelemetryPendingError
    pending_telemetry_label = "loop_efficiency"

    def __init__(self, *, bigquery_client: Any | None = None, lookback_days: int = 1):
        if bigquery_client is None:
            raise ValueError("bigquery_client is required")
        self.bigquery_client = bigquery_client
        self.lookback_days = lookback_days
        self._eval_analysis_cache: dict[str, EvalRunLoopCountAnalysis] = {}

    def analyze(
        self,
        eval_id: str,
        *,
        include_error_examples: bool = True,
        include_per_entry: bool = True,
        evalcli: Any | None = None,
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

    def reflection_prompt(self, module_name: str) -> str:
        return single_model_loop_reflection_prompt(module_name)

    def failure_pattern(self, component_name: str, trajectory: SingleModelALTrajectory) -> tuple[Any, ...]:
        del component_name
        output = trajectory["output"]
        efficiency = trajectory.get("objective_scores", {}).get(self.name, 1.0)
        return (
            int(efficiency < 1.0),
            int((output.get("correctness") or 1.0) < 0.5),
            int((output.get("student_loops") or 0) > TARGET_LOOP_COUNT),
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
        feedback_parts = []
        if correctness < CORRECTNESS_PASS_FOR_FEEDBACK:
            feedback_parts.append(
                f"Incorrect answer (correctness={correctness:.2f}); do not reduce loops by skipping work."
            )
        if loops > TARGET_LOOP_COUNT:
            feedback_parts.append(
                f"Used {loops} loops; stay at or below {TARGET_LOOP_COUNT} by batching independent calls "
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
        self._eval_analysis_cache = _parse_eval_analysis_cache(raw_cache)


register_telemetry_source("single_model", "loop_telemetry", LoopEfficiencyObjective)

__all__ = [
    "EVAL_LOOP_CACHE_SCHEMA_VERSION",
    "LoopEfficiencyObjective",
    "LoopTelemetryPendingError",
]
