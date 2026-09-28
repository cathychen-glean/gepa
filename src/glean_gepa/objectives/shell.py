"""Shell-tool success objective for student-only evals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date
from typing import Any, ClassVar

from glean_gepa.adapter_types import SingleModelALRolloutOutput, SingleModelALTrajectory
from glean_gepa.al_adapter import ReflectiveExample
from glean_gepa.focused_evalset import SESSION_BUCKET_TYPE
from glean_gepa.objectives.base import AnalysisRequest, ScoredRow, ScoringContext, SingleModelObjective
from glean_gepa.objectives.utils.shell_tool_error_util import (
    SHELL_SUCCESS_OBJECTIVE,
    EvalRunShellToolErrorAnalysis,
    ShellToolErrorEntryMetrics,
    enrich_shell_error_action_inputs,
    fetch_eval_run_shell_tool_error_analysis,
    log_shell_tool_error_analysis,
    parse_shell_tool_error_entry_metrics,
    parse_shell_tool_error_metrics,
)
from glean_gepa.prompt_constants import WRITING_CODE_KEY
from glean_gepa.reflection_prompts import CONDITIONAL_PRESERVE_RULE
from glean_gepa.reflection_sampling import strip_stdout_sections

WRITING_CODE_RESPONSIBILITY = (
    "Focus ONLY on coding instructions that affect shell tool reliability: SDK call patterns, "
    "ToolResult handling, parallelism via asyncio.gather, sandbox rules, and when to print vs extract. "
    f"Use shell error examples as evidence. {CONDITIONAL_PRESERVE_RULE} Propose minimal deltas."
)

EVAL_ANALYSIS_CACHE_SCHEMA_VERSION = 9


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
    pending_telemetry_label = "shell"
    pending_count = "shell_executions"
    module_responsibilities: ClassVar[Mapping[str, str]] = {WRITING_CODE_KEY: WRITING_CODE_RESPONSIBILITY}

    def __init__(self, *, bigquery_client: Any | None = None, lookback_days: int = 1):
        if bigquery_client is None:
            raise ValueError("bigquery_client is required")
        self.bigquery_client = bigquery_client
        self.lookback_days = lookback_days
        self.params: dict[str, Any] = {}
        self._eval_analysis_cache: dict[str, EvalRunShellToolErrorAnalysis] = {}

    def analyze(self, eval_id: str, *, request: AnalysisRequest) -> EvalRunShellToolErrorAnalysis:
        def fetch(req: AnalysisRequest) -> EvalRunShellToolErrorAnalysis:
            # Error examples are trace-level evidence; the adapter asks for them only on
            # full trace evals, not focused ones.
            analysis = fetch_eval_run_shell_tool_error_analysis(
                self.bigquery_client,
                eval_id=eval_id,
                lookback_days=self.lookback_days,
                include_error_examples=req.wants_traces,
                include_per_entry=req.wants_per_entry,
            )
            if req.wants_traces and req.evalcli is not None:
                analysis = enrich_shell_error_action_inputs(req.evalcli, analysis)
            return analysis

        return self.cached_eval_analysis(eval_id, request=request, fetch=fetch, label="shell error analysis")

    def cache_hit_is_sufficient(self, cached: EvalRunShellToolErrorAnalysis, request: AnalysisRequest) -> bool:
        """An aggregate-only entry cannot serve a request that needs per-entry rows."""
        return not (request.wants_per_entry and not cached.per_entry and cached.aggregate.shell_executions > 0)

    def analysis_is_cacheable(self, analysis: EvalRunShellToolErrorAnalysis, request: AnalysisRequest) -> bool:
        """Only the full trace fetch is worth keeping; it carries the error examples reflection needs."""
        return request.wants_traces and not self.is_pending(analysis)

    def focused_pass_rate(self, analysis: EvalRunShellToolErrorAnalysis, requested_entry_ids: Sequence[str]) -> float:
        passed_entries = sum(1 for entry_metrics in analysis.per_entry.values() if entry_metrics.shell_errors == 0)
        return passed_entries / len(requested_entry_ids)

    def log_analysis(self, analysis: EvalRunShellToolErrorAnalysis) -> None:
        log_shell_tool_error_analysis(analysis)

    def aggregate_row(self, analysis: EvalRunShellToolErrorAnalysis, ctx: ScoringContext) -> ScoredRow:
        aggregate = analysis.aggregate
        output: SingleModelALRolloutOutput = {
            "deployment_id": ctx.deployment_id,
            "query": ctx.query,
            "entry_id": ctx.query,
            "student_tool_calls": aggregate.shell_executions,
            "student_tool_errors": aggregate.shell_errors,
            "shell_error_messages": [e.error_str for e in aggregate.recent_error_examples if e.error_str],
            "student_eval_run_id": ctx.student_eval_id,
        }
        if action_inputs := [e.action_input for e in aggregate.recent_error_examples if e.action_input]:
            output["shell_action_inputs"] = action_inputs
        return ScoredRow(
            entry_id=None,
            dimension_scores={SHELL_SUCCESS_OBJECTIVE: aggregate.shell_success_rate},
            output=output,
        )

    def entry_row(
        self,
        entry_id: str,
        metrics: ShellToolErrorEntryMetrics,
        analysis: EvalRunShellToolErrorAnalysis,
        ctx: ScoringContext,
    ) -> ScoredRow:
        del analysis
        # Surface evidence from the one failed eval trace, when there is one.
        failed = next((e for e in metrics.recent_error_examples if e.trace_id), None)
        eval_trace_id = failed.trace_id if failed else None
        examples = [e for e in metrics.recent_error_examples if eval_trace_id is None or e.trace_id == eval_trace_id]
        output: SingleModelALRolloutOutput = {
            "deployment_id": ctx.deployment_id,
            "query": ctx.entry_query(entry_id),
            "entry_id": entry_id,
            "student_tool_calls": metrics.shell_executions,
            "student_tool_errors": metrics.shell_errors,
            "shell_error_messages": [e.error_str for e in examples if e.error_str],
            "student_eval_run_id": ctx.student_eval_id,
        }
        if action_inputs := [e.action_input for e in examples if e.action_input]:
            output["shell_action_inputs"] = action_inputs
        data_overrides: dict[str, Any] = {"eval_entry_id": entry_id, "eval_run_id": ctx.student_eval_id}
        if eval_trace_id:
            output["eval_trace_id"] = eval_trace_id
            data_overrides["eval_trace_id"] = eval_trace_id
        # Focused evals are pass/fail per entry; full evals keep the call-level rate.
        score = float(metrics.shell_errors == 0) if ctx.is_focused else metrics.shell_success_rate
        return ScoredRow(
            entry_id=entry_id,
            dimension_scores={SHELL_SUCCESS_OBJECTIVE: score},
            output=output,
            data_overrides=data_overrides,
        )

    def failure_pattern(self, component_name: str, trajectory: SingleModelALTrajectory) -> tuple[Any, ...]:
        del component_name
        output = trajectory["output"]
        shell_success_rate = trajectory.get("objective_scores", {}).get(self.name, 1.0)
        return (
            int(shell_success_rate < float(self.pack_param("failure_score_below", 0.9))),
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
        errors = [s for e in output.get("shell_error_messages", []) if (s := strip_stdout_sections(e))]
        if errors:
            feedback = "Resolve the shell execution failures shown above."
        elif output.get("student_tool_errors", 0) > 0:
            feedback = f"Tool errors: Student encountered {output.get('student_tool_errors', 0)} shell tool errors."
        else:
            feedback = "General shell tool reliability issue."
        return self.reflective_example(
            trajectory,
            feedback=feedback,
            action_inputs=output.get("shell_action_inputs", []),
            execution_errors=errors,
        )

    def cache_payload(self) -> dict[str, Any]:
        return {eval_id: _serialize_eval_analysis(analysis) for eval_id, analysis in self._eval_analysis_cache.items()}

    def load_cache(self, raw_cache: Any) -> None:
        self._eval_analysis_cache = _parse_eval_analysis_cache(raw_cache)


__all__ = [
    "EVAL_ANALYSIS_CACHE_SCHEMA_VERSION",
    "ShellSuccessObjective",
]
