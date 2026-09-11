"""Shell-tool success objective for student-only evals."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, timezone
from datetime import time as datetime_time
from typing import Any

from glean_gepa.adapter_types import SingleModelALRolloutOutput, SingleModelALTrajectory
from glean_gepa.al_adapter import (
    ReflectiveExample,
    ReflectiveExampleInputs,
    ReflectiveExampleMetrics,
    extract_shell_action_inputs,
)
from glean_gepa.debug import debug_print
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.focused_evalset import SESSION_BUCKET_TYPE
from glean_gepa.objectives.base import ScoredRow, SingleModelObjective, register_telemetry_source
from glean_gepa.objectives.utils.shell_tool_error_util import (
    SHELL_SUCCESS_OBJECTIVE,
    EvalRunShellToolErrorAnalysis,
    fetch_eval_run_shell_tool_error_analysis,
    fetch_high_signal_evalset_entries,
    parse_shell_tool_error_entry_metrics,
    parse_shell_tool_error_metrics,
)
from glean_gepa.reflection_prompts import single_model_reflection_prompt
from glean_gepa.reflection_sampling import strip_stdout_sections

EVAL_ANALYSIS_CACHE_SCHEMA_VERSION = 9


class ShellToolTelemetryPendingError(RuntimeError):
    """Raised when an eval has not yet emitted shell telemetry."""


def log_shell_tool_error_analysis(analysis: EvalRunShellToolErrorAnalysis) -> None:
    """Log the fetched shell-tool error rate and recent error details."""
    aggregate = analysis.aggregate
    print(
        f"[Shell Tool] Fetched error rate for eval {analysis.eval_id}: "
        f"{aggregate.shell_error_pct:.2f}% "
        f"({aggregate.shell_errors}/{aggregate.shell_executions})"
    )
    for example in aggregate.recent_error_examples:
        if example.action_input:
            debug_print(f"[Shell Tool] Action input for eval {analysis.eval_id}: {example.action_input}")
        if example.error_str:
            debug_print(f"[Shell Tool] Error for eval {analysis.eval_id}: {example.error_str}")


def _timestamp_millis(value: str | None) -> int | None:
    if not value:
        return None
    normalized = value.replace(" UTC", "+00:00")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def enrich_shell_error_action_inputs(
    evalcli: EvalCliClient,
    analysis: EvalRunShellToolErrorAnalysis,
) -> EvalRunShellToolErrorAnalysis:
    """Fetch detailed traces and attach serialized Shell inputs to failed actions."""
    examples = list(analysis.aggregate.recent_error_examples)
    for metrics in analysis.per_entry.values():
        examples.extend(metrics.recent_error_examples)

    grouped: defaultdict[tuple[str, str], list[Any]] = defaultdict(list)
    for example in examples:
        if example.project_id and example.trace_id and example.action_run_id and not example.action_input:
            grouped[(example.project_id, example.trace_id)].append(example)
    if not grouped:
        return analysis

    print(f"[Shell Tool] Fetching action inputs for {len(grouped)} traces on eval {analysis.eval_id}")
    resolved: dict[tuple[str, str, str], str] = {}
    for (deployment_id, trace_id), trace_examples in grouped.items():
        timestamps = [
            timestamp
            for example in trace_examples
            for timestamp in [_timestamp_millis(example.started_at)]
            if timestamp is not None
        ]
        if timestamps:
            start_time_millis = min(timestamps) - int(timedelta(hours=1).total_seconds() * 1000)
            end_time_millis = max(timestamps) + 1000
        else:
            start_dt = datetime.combine(analysis.start_date, datetime_time.min, tzinfo=timezone.utc)
            end_dt = datetime.combine(analysis.end_date + timedelta(days=1), datetime_time.min, tzinfo=timezone.utc)
            start_time_millis = int(start_dt.timestamp() * 1000)
            end_time_millis = int(end_dt.timestamp() * 1000)
        try:
            detailed_trace = evalcli.get_analysis_trace(
                deployment_id=deployment_id,
                trace_id=trace_id,
                start_time_millis=start_time_millis,
                end_time_millis=end_time_millis,
            )
        except Exception as exc:
            print(f"[Shell Tool] Failed to fetch action inputs for trace {trace_id}: {exc}")
            continue
        for action_run_id, action_input in extract_shell_action_inputs(detailed_trace).items():
            resolved[(deployment_id, trace_id, action_run_id)] = action_input

    def enrich_example(example: Any) -> Any:
        if example.action_input or not (example.project_id and example.trace_id and example.action_run_id):
            return example
        action_input = resolved.get((example.project_id, example.trace_id, example.action_run_id))
        return replace(example, action_input=action_input) if action_input else example

    if not resolved:
        return analysis

    aggregate = replace(
        analysis.aggregate,
        recent_error_examples=tuple(enrich_example(example) for example in analysis.aggregate.recent_error_examples),
    )
    per_entry = {
        entry_id: replace(
            metrics,
            recent_error_examples=tuple(enrich_example(example) for example in metrics.recent_error_examples),
        )
        for entry_id, metrics in analysis.per_entry.items()
    }
    return replace(analysis, aggregate=aggregate, per_entry=per_entry)


def _serialize_eval_analysis(analysis: EvalRunShellToolErrorAnalysis) -> dict[str, Any]:
    def metrics_dict(metrics: Any) -> dict[str, Any]:
        return {
            "eval_id": getattr(metrics, "eval_id", None),
            "entry_id": getattr(metrics, "entry_id", None),
            "shell_executions": metrics.shell_executions,
            "shell_errors": metrics.shell_errors,
            "shell_error_rate": metrics.shell_error_rate,
            "shell_error_pct": metrics.shell_error_pct,
            "trace_ids": list(getattr(metrics, "trace_ids", ())),
            "session_tracking_tokens": list(getattr(metrics, "session_tracking_tokens", ())),
            "recent_error_examples": [asdict(example) for example in metrics.recent_error_examples],
        }

    return {
        "schema_version": EVAL_ANALYSIS_CACHE_SCHEMA_VERSION,
        "eval_id": analysis.eval_id,
        "start_date": analysis.start_date.isoformat(),
        "end_date": analysis.end_date.isoformat(),
        "aggregate": metrics_dict(analysis.aggregate),
        "per_entry": {entry_id: metrics_dict(metrics) for entry_id, metrics in analysis.per_entry.items()},
        "high_signal_entry_ids": list(analysis.high_signal_entry_ids),
    }


def _parse_eval_analysis_cache(raw_cache: Any) -> dict[str, EvalRunShellToolErrorAnalysis]:
    parsed: dict[str, EvalRunShellToolErrorAnalysis] = {}
    if not isinstance(raw_cache, dict):
        return parsed
    for eval_id, raw in raw_cache.items():
        try:
            if not isinstance(raw, dict):
                continue
            if raw.get("schema_version") != EVAL_ANALYSIS_CACHE_SCHEMA_VERSION:
                print(f"[Cache] Refreshing legacy shell error analysis for eval_id: {eval_id}")
                continue
            aggregate = parse_shell_tool_error_metrics(raw["aggregate"])
            if aggregate.shell_executions == 0:
                print(f"[Cache] Refreshing provisional 0/0 shell analysis for eval_id: {eval_id}")
                continue
            per_entry = {
                entry_id: parse_shell_tool_error_entry_metrics(metrics)
                for entry_id, metrics in (raw.get("per_entry") or {}).items()
            }
            parsed[str(eval_id)] = EvalRunShellToolErrorAnalysis(
                eval_id=str(raw.get("eval_id") or eval_id),
                start_date=date.fromisoformat(raw["start_date"]),
                end_date=date.fromisoformat(raw["end_date"]),
                aggregate=aggregate,
                per_entry=per_entry,
                high_signal_entry_ids=tuple(raw.get("high_signal_entry_ids") or ()),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return parsed


class ShellSuccessObjective(SingleModelObjective):
    """Score student evals by shell-tool success rate from Agentspan."""

    name = SHELL_SUCCESS_OBJECTIVE
    telemetry_dimensions = (SHELL_SUCCESS_OBJECTIVE,)
    focused_bucket_type = SESSION_BUCKET_TYPE
    failure_label = "HIGH-SIGNAL FAILURES"
    pending_error_type = ShellToolTelemetryPendingError
    pending_telemetry_label = "shell"

    def __init__(self, *, bigquery_client: Any | None = None, lookback_days: int = 1):
        if bigquery_client is None:
            raise ValueError("bigquery_client is required")
        self.bigquery_client = bigquery_client
        self.lookback_days = lookback_days
        self._eval_analysis_cache: dict[str, EvalRunShellToolErrorAnalysis] = {}

    def analyze(
        self,
        eval_id: str,
        *,
        include_error_examples: bool = True,
        include_per_entry: bool = True,
        evalcli: Any | None = None,
    ) -> EvalRunShellToolErrorAnalysis:
        cached = self._eval_analysis_cache.get(eval_id)
        if cached is not None:
            missing_entry_breakdown = (
                include_per_entry and not cached.per_entry and cached.aggregate.shell_executions > 0
            )
            if not missing_entry_breakdown:
                print(f"[Cache HIT] Using cached shell error analysis for eval_id: {eval_id}")
                return cached
            print(f"[Cache] Refetching shell error analysis with per-entry metrics for eval_id: {eval_id}")
        analysis = fetch_eval_run_shell_tool_error_analysis(
            self.bigquery_client,
            eval_id=eval_id,
            lookback_days=self.lookback_days,
            include_error_examples=include_error_examples,
            include_per_entry=include_per_entry,
        )
        if include_error_examples:
            if evalcli is not None:
                analysis = enrich_shell_error_action_inputs(evalcli, analysis)
            if analysis.aggregate.shell_executions == 0:
                print(f"[Cache] Not caching provisional 0/0 shell analysis for eval_id: {eval_id}")
                return analysis
            self._eval_analysis_cache[eval_id] = analysis
        return analysis

    def is_pending(self, analysis: EvalRunShellToolErrorAnalysis) -> bool:
        return analysis.aggregate.shell_executions == 0

    def aggregate_score(self, analysis: EvalRunShellToolErrorAnalysis) -> float:
        return analysis.aggregate.shell_success_rate

    def focused_pass_rate(self, analysis: EvalRunShellToolErrorAnalysis, requested_entry_ids: Sequence[str]) -> float:
        passed_entries = sum(1 for entry_metrics in analysis.per_entry.values() if entry_metrics.shell_errors == 0)
        return passed_entries / len(requested_entry_ids)

    def entry_ids_to_score(
        self, analysis: EvalRunShellToolErrorAnalysis, requested_entry_ids: Sequence[str] | None
    ) -> tuple[str, ...]:
        if requested_entry_ids:
            return tuple(analysis.per_entry) or tuple(requested_entry_ids)
        return analysis.high_signal_entry_ids

    def log_analysis(self, analysis: EvalRunShellToolErrorAnalysis) -> None:
        log_shell_tool_error_analysis(analysis)

    def scored_rows(
        self,
        analysis: EvalRunShellToolErrorAnalysis,
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
            output: SingleModelALRolloutOutput = {
                "deployment_id": deployment_id,
                "query": query,
                "student_tool_calls": analysis.aggregate.shell_executions,
                "student_tool_errors": analysis.aggregate.shell_errors,
                "entry_id": query,
                "shell_error_messages": [
                    example.error_str for example in analysis.aggregate.recent_error_examples if example.error_str
                ],
                "student_eval_run_id": student_eval_id,
            }
            shell_action_inputs = [
                example.action_input for example in analysis.aggregate.recent_error_examples if example.action_input
            ]
            if shell_action_inputs:
                output["shell_action_inputs"] = shell_action_inputs
            return [
                ScoredRow(
                    entry_id=None,
                    dimension_scores={SHELL_SUCCESS_OBJECTIVE: analysis.aggregate.shell_success_rate},
                    output=output,
                )
            ]

        high_signal_entry_ids = self.entry_ids_to_score(analysis, requested_entry_ids)
        if not high_signal_entry_ids:
            shell_error_messages = [
                example.error_str for example in analysis.aggregate.recent_error_examples if example.error_str
            ]
            shell_action_inputs = [
                example.action_input for example in analysis.aggregate.recent_error_examples if example.action_input
            ]
            output = {
                "deployment_id": deployment_id,
                "query": query,
                "student_tool_calls": analysis.aggregate.shell_executions,
                "student_tool_errors": analysis.aggregate.shell_errors,
                "entry_id": query,
                "shell_error_messages": shell_error_messages,
                "student_eval_run_id": student_eval_id,
            }
            if shell_action_inputs:
                output["shell_action_inputs"] = shell_action_inputs
            return [
                ScoredRow(
                    entry_id=query,
                    dimension_scores={SHELL_SUCCESS_OBJECTIVE: analysis.aggregate.shell_success_rate},
                    output=output,
                )
            ]

        rows: list[ScoredRow] = []
        for entry_id in high_signal_entry_ids:
            entry_metrics = analysis.per_entry.get(entry_id, analysis.aggregate)
            failed_eval_example = next(
                (example for example in entry_metrics.recent_error_examples if example.trace_id),
                None,
            )
            eval_trace_id = failed_eval_example.trace_id if failed_eval_example else None
            eval_trace_examples = [
                example
                for example in entry_metrics.recent_error_examples
                if eval_trace_id is None or example.trace_id == eval_trace_id
            ]
            shell_error_messages = [example.error_str for example in eval_trace_examples if example.error_str]
            shell_action_inputs = [example.action_input for example in eval_trace_examples if example.action_input]
            entry_output: SingleModelALRolloutOutput = {
                "deployment_id": deployment_id,
                "query": f"{query} entry={entry_id}",
                "student_tool_calls": entry_metrics.shell_executions,
                "student_tool_errors": entry_metrics.shell_errors,
                "entry_id": entry_id,
                "shell_error_messages": shell_error_messages,
                "student_eval_run_id": student_eval_id,
            }
            if shell_action_inputs:
                entry_output["shell_action_inputs"] = shell_action_inputs
            if eval_trace_id:
                entry_output["eval_trace_id"] = eval_trace_id
            data_overrides: dict[str, Any] = {
                "eval_entry_id": entry_id,
                "eval_run_id": student_eval_id,
            }
            if eval_trace_id:
                data_overrides["eval_trace_id"] = eval_trace_id
            shell_success = (
                float(entry_id in analysis.per_entry and entry_metrics.shell_errors == 0)
                if is_focused_eval
                else entry_metrics.shell_success_rate
            )
            rows.append(
                ScoredRow(
                    entry_id=entry_id,
                    dimension_scores={SHELL_SUCCESS_OBJECTIVE: shell_success},
                    output=entry_output,
                    data_overrides=data_overrides,
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
        return fetch_high_signal_evalset_entries(
            self.bigquery_client,
            eval_set_name=eval_set_name,
            eval_set_version=eval_set_version,
            eval_run_id=eval_run_id,
            entry_ids=list(entry_ids),
            deployment_ids=list(deployment_ids),
        )

    def reflection_prompt(self, module_name: str) -> str:
        return single_model_reflection_prompt(module_name)

    def failure_pattern(self, component_name: str, trajectory: SingleModelALTrajectory) -> tuple[Any, ...]:
        del component_name
        output = trajectory["output"]
        shell_success_rate = trajectory.get("objective_scores", {}).get(self.name, 1.0)
        return (
            int(shell_success_rate < 0.9),
            int(output.get("student_tool_errors", 0) > 0),
            len(output.get("shell_error_messages", [])),
        )

    def build_reflective_example(
        self,
        component_name: str,
        trajectory: SingleModelALTrajectory,
        candidate: dict[str, str],
    ) -> ReflectiveExample:
        del component_name, candidate
        output = trajectory["output"]
        shell_success_rate = trajectory.get("objective_scores", {}).get(self.name, 1.0)
        shell_error_messages = [
            sanitized for error in output.get("shell_error_messages", []) if (sanitized := strip_stdout_sections(error))
        ]
        if shell_error_messages:
            feedback = "Resolve the shell execution failures shown above."
        elif output.get("student_tool_errors", 0) > 0:
            feedback = f"Tool errors: Student encountered {output.get('student_tool_errors', 0)} shell tool errors."
        else:
            feedback = "General shell tool reliability issue."

        inputs: ReflectiveExampleInputs = {
            "eval_set": trajectory["data"]["eval_set_name"],
            "entry_id": output["entry_id"],
            "deployment_id": output["deployment_id"],
            "query": output["query"],
        }
        if eval_run_id := trajectory["data"].get("eval_run_id"):
            inputs["eval_run_id"] = eval_run_id
        if eval_trace_id := trajectory["data"].get("eval_trace_id"):
            inputs["eval_trace_id"] = eval_trace_id

        return {
            "Inputs": inputs,
            "Generated Outputs": {
                "student_answer": "",
                "teacher_answer": "",
                "student_tools": [],
                "teacher_tools": [],
            },
            "Action Inputs": output.get("shell_action_inputs", [])[:5],
            "Execution Errors": shell_error_messages[:5],
            "Feedback": feedback,
            "Metrics": {"score": trajectory["score"], "shell_success_rate": shell_success_rate},
        }

    def format_reflective_metrics(self, metrics: ReflectiveExampleMetrics) -> str | None:
        del metrics
        return None

    def cache_payload(self) -> dict[str, Any]:
        return {eval_id: _serialize_eval_analysis(analysis) for eval_id, analysis in self._eval_analysis_cache.items()}

    def load_cache(self, raw_cache: Any) -> None:
        self._eval_analysis_cache = _parse_eval_analysis_cache(raw_cache)


register_telemetry_source("single_model", "shell_telemetry", ShellSuccessObjective)

__all__ = [
    "EVAL_ANALYSIS_CACHE_SCHEMA_VERSION",
    "ShellSuccessObjective",
    "ShellToolTelemetryPendingError",
    "enrich_shell_error_action_inputs",
    "log_shell_tool_error_analysis",
]
