"""Loop-efficiency objective: reward a single model for answering in few agent loops.

Score per entry is ``1 / (1 + extra_loops)`` where ``extra_loops`` is how many
loops the student used beyond ``target_loop_count`` (pack param, default 2).
A direct answer or one within the cap scores 1.0. The run-level score is the
mean over entries.

Telemetry (BigQuery SQL, evalcli overlay, trace enrichment) lives in
:mod:`glean_gepa.objectives.utils.loop_count_util`. This module only maps that
analysis onto the objective contract in :mod:`glean_gepa.objectives.protocol`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date
from typing import Any, ClassVar

from glean_gepa.adapter_types import SingleModelALRolloutOutput, SingleModelALTrajectory
from glean_gepa.al_adapter import ReflectiveExample
from glean_gepa.focused_evalset import QUERY_CANONICAL_BUCKET_TYPE
from glean_gepa.objectives.base import AnalysisRequest, ScoredRow, ScoringContext, SingleModelObjective
from glean_gepa.objectives.utils.loop_count_util import (
    LOOP_EFFICIENCY_OBJECTIVE,
    TARGET_LOOP_COUNT,
    EvalRunLoopCountAnalysis,
    LoopCountEntryMetrics,
    aggregate_loop_count_metrics,
    fetch_eval_run_loop_count_analysis,
    log_loop_count_analysis,
    loop_count_entry_from_cache,
    loop_reflection_feedback,
)
from glean_gepa.prompt_constants import WRITING_CODE_KEY
from glean_gepa.reflection_prompts import CONDITIONAL_PRESERVE_RULE

WRITING_CODE_RESPONSIBILITY = (
    "Focus ONLY on instructions that reduce extra agent loops: batch independent tool calls in one "
    "turn, stop once the answer is grounded, and avoid re-running searches that already returned the "
    f"needed evidence. Use the loop counts and tool payloads above as evidence. {CONDITIONAL_PRESERVE_RULE} "
    "Propose minimal deltas."
)

CACHE_SCHEMA_VERSION = 1


class LoopEfficiencyObjective(SingleModelObjective):
    """Score student evals by how few agent loops each entry needed."""

    name = LOOP_EFFICIENCY_OBJECTIVE
    telemetry_dimensions = (LOOP_EFFICIENCY_OBJECTIVE,)
    focused_bucket_type = QUERY_CANONICAL_BUCKET_TYPE
    failure_label = "HIGH-SIGNAL FAILURES (extra loops)"
    pending_telemetry_label = "loop count"
    # ``compared_entries`` is 0 until Execute Action spans have landed for the run.
    pending_count = "compared_entries"
    module_responsibilities: ClassVar[Mapping[str, str]] = {WRITING_CODE_KEY: WRITING_CODE_RESPONSIBILITY}

    def __init__(self, *, bigquery_client: Any | None = None, lookback_days: int = 1):
        if bigquery_client is None:
            raise ValueError("bigquery_client is required")
        self.bigquery_client = bigquery_client
        self.lookback_days = lookback_days
        self.params: dict[str, Any] = {}
        self._eval_analysis_cache: dict[str, EvalRunLoopCountAnalysis] = {}
        # Eval ids cached from a fetch that skipped action-input hydration. A later
        # hydrating request refetches these instead of serving payload-less entries.
        self._unhydrated_eval_ids: set[str] = set()

    # --- configuration -----------------------------------------------------

    @property
    def target_loops(self) -> int:
        return int(self.pack_param("target_loop_count", TARGET_LOOP_COUNT))

    # --- scoring -----------------------------------------------------------

    def analyze(self, eval_id: str, *, request: AnalysisRequest) -> EvalRunLoopCountAnalysis:
        """Loop counts are one fetch shape; ``request.detail`` does not change the query.

        Only ``request.hydrate_action_inputs`` does, and the base cache handles that.
        """

        def fetch(req: AnalysisRequest) -> EvalRunLoopCountAnalysis:
            return fetch_eval_run_loop_count_analysis(
                self.bigquery_client,
                eval_id=eval_id,
                lookback_days=self.lookback_days,
                evalcli=req.evalcli,
                include_action_inputs=req.hydrate_action_inputs,
                target_loops=self.target_loops,
            )

        return self.cached_eval_analysis(eval_id, request=request, fetch=fetch, label="loop count analysis")

    def focused_pass_rate(self, analysis: EvalRunLoopCountAnalysis, requested_entry_ids: Sequence[str]) -> float:
        if not requested_entry_ids:
            return 0.0
        passed = sum(
            1
            for entry_id in requested_entry_ids
            if (metrics := analysis.per_entry.get(entry_id)) is not None and metrics.loop_efficiency >= 1.0
        )
        return passed / len(requested_entry_ids)

    def log_analysis(self, analysis: EvalRunLoopCountAnalysis) -> None:
        log_loop_count_analysis(analysis)

    def aggregate_row(self, analysis: EvalRunLoopCountAnalysis, ctx: ScoringContext) -> ScoredRow:
        aggregate = analysis.aggregate
        output: SingleModelALRolloutOutput = {
            "deployment_id": ctx.deployment_id,
            "query": ctx.query,
            "entry_id": ctx.query,
            "student_tool_calls": round(aggregate.mean_loop_count * max(aggregate.compared_entries, 1)),
            "student_tool_errors": 0,
            "shell_error_messages": [],
            "student_eval_run_id": ctx.student_eval_id,
            "student_loops": round(aggregate.mean_loop_count),
        }
        return ScoredRow(entry_id=None, dimension_scores={self.name: aggregate.loop_efficiency}, output=output)

    def entry_row(
        self,
        entry_id: str,
        metrics: LoopCountEntryMetrics,
        analysis: EvalRunLoopCountAnalysis,
        ctx: ScoringContext,
    ) -> ScoredRow:
        del analysis
        output: SingleModelALRolloutOutput = {
            "deployment_id": ctx.deployment_id,
            "query": ctx.entry_query(entry_id),
            "entry_id": entry_id,
            "student_tool_calls": metrics.loop_count,
            "student_tool_errors": int(metrics.has_error),
            "shell_error_messages": [],
            "student_eval_run_id": ctx.student_eval_id,
            "student_loops": metrics.loop_count,
        }
        if metrics.action_inputs:
            output["action_inputs"] = list(metrics.action_inputs)
        return ScoredRow(
            entry_id=entry_id,
            dimension_scores={self.name: metrics.loop_efficiency},
            output=output,
            data_overrides={"eval_entry_id": entry_id, "eval_run_id": ctx.student_eval_id},
        )

    # --- reflection --------------------------------------------------------

    def failure_pattern(self, component_name: str, trajectory: SingleModelALTrajectory) -> tuple[Any, ...]:
        del component_name
        loops = int(trajectory["output"].get("student_loops", 0))
        score = trajectory.get("objective_scores", {}).get(self.name, 1.0)
        return (
            int(score < float(self.pack_param("failure_score_below", 1.0))),
            max(0, loops - self.target_loops),
        )

    def build_reflective_example(
        self,
        component_name: str,
        trajectory: SingleModelALTrajectory,
        candidate: dict[str, str],
    ) -> ReflectiveExample:
        del component_name, candidate
        output = trajectory["output"]
        return self.reflective_example(
            trajectory,
            feedback=loop_reflection_feedback(
                loops=int(output.get("student_loops", 0)), target_loops=self.target_loops
            ),
            action_inputs=output.get("action_inputs", []),
        )

    # --- cache -------------------------------------------------------------

    def cache_payload(self) -> dict[str, Any]:
        return {
            eval_id: {
                "schema_version": CACHE_SCHEMA_VERSION,
                "eval_id": analysis.eval_id,
                "start_date": analysis.start_date.isoformat(),
                "end_date": analysis.end_date.isoformat(),
                "action_inputs_hydrated": eval_id not in self._unhydrated_eval_ids,
                "per_entry": {
                    entry_id: {
                        "entry_id": m.entry_id,
                        "loop_count": m.loop_count,
                        "correctness": m.correctness,
                        "has_error": m.has_error,
                        "action_inputs": list(m.action_inputs),
                    }
                    for entry_id, m in analysis.per_entry.items()
                },
            }
            for eval_id, analysis in self._eval_analysis_cache.items()
        }

    def load_cache(self, raw_cache: Any) -> None:
        """Rebuild analyses from cached loop counts under the *current* ``target_loop_count``.

        Scores and the high-signal list are derived, not stored, so retuning the
        cap between runs takes effect on resume.
        """
        self._eval_analysis_cache = {}
        self._unhydrated_eval_ids = set()
        if not isinstance(raw_cache, dict):
            return
        target = self.target_loops
        for eval_id, raw in raw_cache.items():
            try:
                if not isinstance(raw, dict) or raw.get("schema_version") != CACHE_SCHEMA_VERSION:
                    print(f"[Cache] Refreshing legacy loop analysis for eval_id: {eval_id}")
                    continue
                per_entry: dict[str, LoopCountEntryMetrics] = {
                    entry_id: loop_count_entry_from_cache(entry_id, metrics, target_loops=target)
                    for entry_id, metrics in (raw.get("per_entry") or {}).items()
                }
                if not per_entry:
                    continue
                key = str(eval_id)
                self._eval_analysis_cache[key] = self._rebuild(str(raw.get("eval_id") or eval_id), raw, per_entry)
                # Absent flag (pre-fix payload) is treated as unhydrated so the next
                # trace fetch refreshes rather than trusting an unknown state.
                if not raw.get("action_inputs_hydrated", False):
                    self._unhydrated_eval_ids.add(key)
            except (KeyError, TypeError, ValueError):
                continue

    def _rebuild(
        self, eval_id: str, raw: Mapping[str, Any], per_entry: dict[str, LoopCountEntryMetrics]
    ) -> EvalRunLoopCountAnalysis:
        per_entry = {entry_id: replace(m, target_loops=self.target_loops) for entry_id, m in per_entry.items()}
        return EvalRunLoopCountAnalysis(
            eval_id=eval_id,
            start_date=date.fromisoformat(raw["start_date"]),
            end_date=date.fromisoformat(raw["end_date"]),
            aggregate=aggregate_loop_count_metrics(eval_id, per_entry),
            per_entry=per_entry,
            high_signal_entry_ids=tuple(sorted(e for e, m in per_entry.items() if m.loop_efficiency < 1.0)),
        )


__all__ = ["CACHE_SCHEMA_VERSION", "LoopEfficiencyObjective"]
