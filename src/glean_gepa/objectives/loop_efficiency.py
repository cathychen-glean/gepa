"""Loop-efficiency objective: reward a single model for answering in few agent loops.

Score per entry is ``1 / (1 + extra_loops)`` where ``extra_loops`` is how many
loops the student used beyond ``target_loop_count`` (pack param, default 2).
A direct answer or one within the cap scores 1.0. The run-level score is the
mean over entries.

Layout, top to bottom: the entry and aggregate types, row parsing, the
``eval_spans`` SQL, the fetch (with the EvalCLI overlay that prefers the eval's
own ``loopCount`` / CORRECTNESS), then the objective class that maps the
analysis onto the contract in :mod:`glean_gepa.objectives.protocol`. Shared
plumbing (bounds query, shard window, trace enrichment, frame) comes from
``objectives/utils``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, ClassVar

from glean_gepa.adapter_types import SingleModelALRolloutOutput, SingleModelALTrajectory
from glean_gepa.al_adapter import ReflectiveExample
from glean_gepa.focused_evalset import QUERY_CANONICAL_BUCKET_TYPE
from glean_gepa.objectives.base import AnalysisRequest, ScoredRow, ScoringContext, SingleModelObjective
from glean_gepa.objectives.utils.agentspan import Rows, bounds_query, fetch_agentspan_analysis
from glean_gepa.objectives.utils.agentspan_query import (
    DEFAULT_AGENTS_SPAN_TABLE,
    DEFAULT_LOOKBACK_DAYS,
    EVAL_ENTRY_ID_EXPR,
    EXECUTE_ACTION_FILTER,
    default_date_range,
    wildcard_shard_filter,
)
from glean_gepa.objectives.utils.core import RunAnalysis, log_analysis
from glean_gepa.objectives.utils.traces import FetchedByRole, enrich_action_inputs
from glean_gepa.prompt_constants import WRITING_CODE_KEY
from glean_gepa.reflection_prompts import CONDITIONAL_PRESERVE_RULE

LOOP_EFFICIENCY_OBJECTIVE = "loop_efficiency"
# 1.0 when the student finishes in this many loops or fewer. Extra loops decay
# as 1 / (1 + extra).
TARGET_LOOP_COUNT = 2


# ---------------------------------------------------------------------------
# Slot 1: one entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopCountEntryMetrics:
    entry_id: str
    loop_count: int
    correctness: float
    has_error: bool
    action_inputs: tuple[str, ...] = ()
    target_loops: int = TARGET_LOOP_COUNT

    @property
    def loop_efficiency(self) -> float:
        """Higher-is-better score: 1.0 at or below the loop cap, decaying as loops grow."""
        extra = max(0, self.loop_count - self.target_loops)
        return 1.0 / (1.0 + extra)

    @property
    def passed(self) -> bool:
        return self.loop_efficiency >= 1.0

    @property
    def score(self) -> float:
        return self.loop_efficiency


@dataclass(frozen=True)
class LoopCountMetrics:
    compared_entries: int
    matching_entries: int
    mean_loop_count: float
    loop_efficiency: float
    mean_correctness: float


class EvalRunLoopCountAnalysis(RunAnalysis[LoopCountMetrics, LoopCountEntryMetrics]):
    """Agentspan analyses always carry a resolved shard window."""

    start_date: date
    end_date: date


# ---------------------------------------------------------------------------
# Slot 2: rows -> entry
# ---------------------------------------------------------------------------


def loop_count_entry_from_cache(
    entry_id: str, metrics: Mapping[str, Any], *, target_loops: int
) -> LoopCountEntryMetrics:
    return LoopCountEntryMetrics(
        entry_id=str(metrics.get("entry_id") or entry_id),
        loop_count=int(metrics.get("loop_count") or 0),
        correctness=float(metrics.get("correctness") or 0.0),
        has_error=bool(metrics.get("has_error")),
        action_inputs=tuple(metrics.get("action_inputs") or ()),
        target_loops=target_loops,
    )


def parse_loop_count_entry_metrics(
    row: Mapping[str, Any], *, target_loops: int = TARGET_LOOP_COUNT
) -> LoopCountEntryMetrics:
    loop_count = int(row.get("loop_count") or 0)
    has_error = bool(row.get("has_error"))
    raw_correctness = row.get("correctness")
    if raw_correctness is None:
        correctness = 0.0 if has_error else 1.0
    else:
        correctness = float(raw_correctness)
        if has_error:
            correctness = min(correctness, 0.0)
    return LoopCountEntryMetrics(
        entry_id=str(row.get("entry_id") or ""),
        loop_count=loop_count,
        correctness=correctness,
        has_error=has_error,
        target_loops=target_loops,
    )


def aggregate_loop_count_metrics(per_entry: Mapping[str, LoopCountEntryMetrics]) -> LoopCountMetrics:
    compared = len(per_entry)
    matching = sum(1 for metrics in per_entry.values() if metrics.loop_efficiency >= 1.0)
    mean_loops = (sum(metrics.loop_count for metrics in per_entry.values()) / compared) if compared else 0.0
    mean_correctness = (sum(metrics.correctness for metrics in per_entry.values()) / compared) if compared else 0.0
    efficiency = (sum(metrics.loop_efficiency for metrics in per_entry.values()) / compared) if compared else 0.0
    return LoopCountMetrics(
        compared_entries=compared,
        matching_entries=matching,
        mean_loop_count=mean_loops,
        loop_efficiency=efficiency,
        mean_correctness=mean_correctness,
    )


# ---------------------------------------------------------------------------
# Slot 3: SQL
# ---------------------------------------------------------------------------


def build_loop_count_per_entry_query(*, agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE) -> str:
    """Count tool loops per eval entry, keeping 0-loop answers in the comparison.

    Loop count is the number of Execute Action EXECUTE spans. Overlay
    ``metadata.loopCount`` from the eval analysis view when that is available.
    Entries with no Execute Action spans still appear so a direct (0-loop)
    answer is scored instead of looking like missing telemetry.
    """
    return f"""
WITH eval_spans AS (
  SELECT
    {EVAL_ENTRY_ID_EXPR} AS entry_id,
    {EXECUTE_ACTION_FILTER} AS is_loop,
    (
      jsonPayload.action.execution_status = 'ERROR'
      OR jsonPayload.span_info.execution_status.code = 'ERROR'
    ) AS is_error,
    jsonPayload.context.agent_trace.trace_id AS trace_id,
    resource.labels.project_id AS deployment_id,
    SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64) AS start_ms
  FROM `{agentspan_table}`
  WHERE {wildcard_shard_filter("start_date", "end_date")}
    AND jsonPayload.context.eval.eval_id = @eval_id
)
SELECT
  entry_id,
  COUNTIF(is_loop) AS loop_count,
  COUNTIF(is_error) > 0 AS has_error,
  ANY_VALUE(trace_id) AS trace_id,
  ANY_VALUE(deployment_id) AS deployment_id,
  MIN(start_ms) AS min_start_ms,
  MAX(start_ms) AS max_start_ms
FROM eval_spans
WHERE entry_id IS NOT NULL
GROUP BY entry_id
ORDER BY entry_id
""".strip()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def empty_loop_count_analysis(
    eval_id: str, *, lookback_days: int = DEFAULT_LOOKBACK_DAYS, end_date: date | None = None
) -> EvalRunLoopCountAnalysis:
    start_date, resolved_end = default_date_range(lookback_days=lookback_days, end_date=end_date)
    return EvalRunLoopCountAnalysis(
        eval_ids=(eval_id,),
        aggregate=aggregate_loop_count_metrics({}),
        start_date=start_date,
        end_date=resolved_end,
    )


def fetch_eval_run_loop_count_analysis(
    client: Any,
    *,
    eval_id: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: date | None = None,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
    evalcli: Any | None = None,
    include_action_inputs: bool = True,
    target_loops: int = TARGET_LOOP_COUNT,
) -> EvalRunLoopCountAnalysis:
    def parse(row: Mapping[str, Any]) -> LoopCountEntryMetrics | None:
        metrics = parse_loop_count_entry_metrics(row, target_loops=target_loops)
        return metrics if metrics.entry_id else None

    def overlay(per_entry: Mapping[str, LoopCountEntryMetrics]) -> Mapping[str, LoopCountEntryMetrics]:
        return overlay_evalcli_loop_and_correctness(evalcli, eval_id, dict(per_entry), target_loops=target_loops)

    def enrich(
        per_entry: Mapping[str, LoopCountEntryMetrics], rows: Rows, high_signal: tuple[str, ...]
    ) -> Mapping[str, LoopCountEntryMetrics]:
        return _enrich_action_inputs(evalcli, per_entry, rows, high_signal)

    analysis = fetch_agentspan_analysis(
        client,
        eval_ids=(eval_id,),
        bounds_sql=bounds_query(eval_id_predicate="= @eval_id", agentspan_table=agentspan_table),
        per_entry_sql=build_loop_count_per_entry_query(agentspan_table=agentspan_table),
        parse_row=parse,
        aggregate=lambda _ids, per_entry, _dropped: aggregate_loop_count_metrics(per_entry),
        is_high_signal=lambda m: m.loop_efficiency < 1.0,
        post_parse=overlay if evalcli is not None else None,
        enrich=enrich if evalcli is not None and include_action_inputs else None,
        lookback_days=lookback_days,
        end_date=end_date,
    )
    if analysis.start_date is None:
        return empty_loop_count_analysis(eval_id, lookback_days=lookback_days, end_date=end_date)
    return EvalRunLoopCountAnalysis(
        eval_ids=analysis.eval_ids,
        aggregate=analysis.aggregate,
        per_entry=analysis.per_entry,
        high_signal_entry_ids=analysis.high_signal_entry_ids,
        start_date=analysis.start_date,
        end_date=analysis.end_date or analysis.start_date,
    )


def _enrich_action_inputs(
    evalcli: Any,
    per_entry: Mapping[str, LoopCountEntryMetrics],
    rows: Rows,
    high_signal_entry_ids: tuple[str, ...],
) -> dict[str, LoopCountEntryMetrics]:
    """Attach the student's tool payloads to high-signal entries from their traces."""

    def apply(metrics: LoopCountEntryMetrics, fetched: FetchedByRole, entry_id: str) -> LoopCountEntryMetrics:
        inputs = fetched.get("student", {}).get(entry_id)
        return replace(metrics, action_inputs=inputs) if inputs is not None else metrics

    return enrich_action_inputs(evalcli, per_entry, rows, high_signal_entry_ids, apply=apply)


def overlay_evalcli_loop_and_correctness(
    evalcli: Any,
    eval_id: str,
    per_entry: dict[str, LoopCountEntryMetrics],
    *,
    target_loops: int = TARGET_LOOP_COUNT,
) -> dict[str, LoopCountEntryMetrics]:
    """Prefer eval ``loopCount`` and CORRECTNESS judge scores when the view has them."""
    get_view = getattr(evalcli, "get_analysis_view", None)
    if not callable(get_view):
        return per_entry
    try:
        view = get_view(eval_id)
    except Exception as exc:
        print(f"[Loop Count] Skipping evalcli overlay for {eval_id}: {exc}")
        return per_entry
    if not isinstance(view, dict):
        return per_entry

    loops_by_entry, correctness_by_entry = _parse_analysis_view(view, eval_id)
    if not loops_by_entry and not correctness_by_entry:
        return per_entry

    updated = dict(per_entry)
    for entry_id in {*updated, *loops_by_entry, *correctness_by_entry}:
        current = updated.get(entry_id)
        loop_count = loops_by_entry.get(entry_id, current.loop_count if current else 0)
        has_error = current.has_error if current else False
        correctness = correctness_by_entry.get(
            entry_id,
            current.correctness if current is not None else (0.0 if has_error else 1.0),
        )
        updated[entry_id] = LoopCountEntryMetrics(
            entry_id=entry_id,
            loop_count=int(loop_count),
            correctness=float(correctness),
            has_error=has_error,
            action_inputs=current.action_inputs if current else (),
            target_loops=target_loops,
        )
    return updated


def _parse_analysis_view(view: dict[str, Any], eval_id: str) -> tuple[dict[str, int], dict[str, float]]:
    loops_by_entry: dict[str, int] = {}
    correctness_by_entry: dict[str, float] = {}
    for entry in view.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        entry_id = str(entry.get("entryId") or entry.get("entry_id") or "")
        if not entry_id:
            continue
        for eval_run_entry in entry.get("evalRunEntries") or []:
            if not isinstance(eval_run_entry, dict):
                continue
            run_id = str(eval_run_entry.get("evalRunId") or "")
            if run_id and run_id != eval_id:
                continue
            metadata = eval_run_entry.get("metadata") or {}
            if "loopCount" in metadata:
                loops_by_entry[entry_id] = int(metadata.get("loopCount") or 0)
        for judge_entry in entry.get("judgeRunEntries") or []:
            if not isinstance(judge_entry, dict):
                continue
            for output in judge_entry.get("outputs") or []:
                if isinstance(output, dict) and str(output.get("name") or "").upper() == "CORRECTNESS":
                    correctness_by_entry[entry_id] = float(output.get("score") or 0.0)
    return loops_by_entry, correctness_by_entry


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

WRITING_CODE_RESPONSIBILITY = (
    "Focus ONLY on instructions that reduce extra agent loops: batch independent tool calls in one "
    "turn, stop once the answer is grounded, and avoid re-running searches that already returned the "
    f"needed evidence. Use the loop counts and tool payloads above as evidence. {CONDITIONAL_PRESERVE_RULE} "
    "Propose minimal deltas."
)

CACHE_SCHEMA_VERSION = 1


class LoopEfficiencyObjective(SingleModelObjective[EvalRunLoopCountAnalysis]):
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
        aggregate = analysis.aggregate
        target_loops = next(iter(analysis.per_entry.values())).target_loops if analysis.per_entry else TARGET_LOOP_COUNT
        log_analysis(
            analysis,
            label="Loop Count",
            headline=(
                f"efficiency={aggregate.loop_efficiency:.2%} "
                f"({aggregate.matching_entries}/{aggregate.compared_entries} at <= {target_loops} loops), "
                f"mean_loops={aggregate.mean_loop_count:.2f}, correctness={aggregate.mean_correctness:.2f}"
            ),
            entry_line=lambda m: f"loops={m.loop_count} correctness={m.correctness:.2f} score={m.loop_efficiency:.2f}",
        )

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
        loops = int(output.get("student_loops", 0))
        if loops > self.target_loops:
            feedback = (
                f"Used {loops} loops; stay at or below {self.target_loops} by batching independent calls "
                "and stopping once the answer is grounded."
            )
        else:
            feedback = "Reduce extra agent loops."
        return self.reflective_example(trajectory, feedback=feedback, action_inputs=output.get("action_inputs", []))

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
            eval_ids=(eval_id,),
            aggregate=aggregate_loop_count_metrics(per_entry),
            per_entry=per_entry,
            high_signal_entry_ids=tuple(sorted(e for e, m in per_entry.items() if m.loop_efficiency < 1.0)),
            start_date=date.fromisoformat(raw["start_date"]),
            end_date=date.fromisoformat(raw["end_date"]),
        )


__all__ = ["CACHE_SCHEMA_VERSION", "LoopEfficiencyObjective"]
