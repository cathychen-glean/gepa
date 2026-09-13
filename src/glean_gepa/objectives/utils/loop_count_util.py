"""Student-only loop-count metrics derived from Glean evaluation spans."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Any

from glean_gepa.objectives.utils.action_input_trace import (
    build_trace_locator,
    fetch_action_inputs_by_entry,
)
from glean_gepa.objectives.utils.agentspan_query import (
    DEFAULT_AGENTS_SPAN_TABLE,
    DEFAULT_LOOKBACK_DAYS,
    EXECUTE_ACTION_FILTER,
    QueryParameter,
    default_date_range,
    run_windowed_per_entry_query,
    wildcard_shard_filter,
)

# Reflection surfaces at most this many tool payloads per entry.
ACTION_INPUT_SURFACE_LIMIT = 5

LOOP_EFFICIENCY_OBJECTIVE = "loop_efficiency"
# 1.0 when the student finishes in this many loops or fewer. Extra loops decay
# as 1 / (1 + extra). Incorrect entries score 0 so fewer loops cannot hide a
# worse answer.
TARGET_LOOP_COUNT = 2
CORRECTNESS_PASS = 0.5


@dataclass(frozen=True)
class LoopCountEntryMetrics:
    entry_id: str
    loop_count: int
    correctness: float
    has_error: bool
    action_inputs: tuple[str, ...] = ()

    @property
    def loop_efficiency(self) -> float:
        if self.has_error:
            return 0.0
        return loop_efficiency_score(self.loop_count, self.correctness)


@dataclass(frozen=True)
class LoopCountMetrics:
    eval_id: str
    compared_entries: int
    matching_entries: int
    mean_loop_count: float
    loop_efficiency: float
    mean_correctness: float


@dataclass(frozen=True)
class EvalRunLoopCountAnalysis:
    eval_id: str
    start_date: date
    end_date: date
    aggregate: LoopCountMetrics
    per_entry: dict[str, LoopCountEntryMetrics]
    high_signal_entry_ids: tuple[str, ...]


def loop_efficiency_score(
    loop_count: int,
    correctness: float,
    *,
    target_loops: int = TARGET_LOOP_COUNT,
    correctness_pass: float = CORRECTNESS_PASS,
) -> float:
    """Higher-is-better score: 0 if incorrect, else 1.0 at or below the loop cap."""
    if correctness < correctness_pass:
        return 0.0
    extra = max(0, int(loop_count) - target_loops)
    return 1.0 / (1.0 + extra)


def parse_loop_count_entry_metrics(row: dict[str, Any]) -> LoopCountEntryMetrics:
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
    )


def aggregate_loop_count_metrics(eval_id: str, per_entry: dict[str, LoopCountEntryMetrics]) -> LoopCountMetrics:
    compared = len(per_entry)
    matching = sum(1 for metrics in per_entry.values() if metrics.loop_efficiency >= 1.0)
    mean_loops = (sum(metrics.loop_count for metrics in per_entry.values()) / compared) if compared else 0.0
    mean_correctness = (sum(metrics.correctness for metrics in per_entry.values()) / compared) if compared else 0.0
    efficiency = (sum(metrics.loop_efficiency for metrics in per_entry.values()) / compared) if compared else 0.0
    return LoopCountMetrics(
        eval_id=eval_id,
        compared_entries=compared,
        matching_entries=matching,
        mean_loop_count=mean_loops,
        loop_efficiency=efficiency,
        mean_correctness=mean_correctness,
    )


def empty_loop_count_analysis(
    eval_id: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: date | None = None,
) -> EvalRunLoopCountAnalysis:
    start_date, resolved_end = default_date_range(lookback_days=lookback_days, end_date=end_date)
    return EvalRunLoopCountAnalysis(
        eval_id=eval_id,
        start_date=start_date,
        end_date=resolved_end,
        aggregate=aggregate_loop_count_metrics(eval_id, {}),
        per_entry={},
        high_signal_entry_ids=(),
    )


def build_loop_count_time_bounds_query(
    *,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> str:
    """Find min/max span timestamps for one student eval run."""
    return f"""
SELECT
  MIN(SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64)) AS min_start_ms,
  MAX(SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64)) AS max_start_ms
FROM `{agentspan_table}`
WHERE {wildcard_shard_filter("search_start_date", "search_end_date")}
  AND jsonPayload.context.eval.eval_id = @eval_id
""".strip()


def build_loop_count_per_entry_query(
    *,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> str:
    """Count tool loops per eval entry, keeping 0-loop answers in the comparison.

    Loop count is the number of Execute Action EXECUTE spans. Overlay
    ``metadata.loopCount`` from the eval analysis view when that is available.
    Entries with no Execute Action spans still appear so a direct (0-loop)
    answer is scored instead of looking like missing telemetry.
    """
    return f"""
WITH eval_spans AS (
  SELECT
    COALESCE(
      jsonPayload.context.eval.entry_uuid,
      CAST(jsonPayload.context.eval.entry_id AS STRING)
    ) AS entry_id,
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


def fetch_eval_run_loop_count_analysis(
    client: Any,
    *,
    eval_id: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: date | None = None,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
    evalcli: Any | None = None,
) -> EvalRunLoopCountAnalysis:
    result = run_windowed_per_entry_query(
        client,
        bounds_query=build_loop_count_time_bounds_query(agentspan_table=agentspan_table),
        per_entry_query=build_loop_count_per_entry_query(agentspan_table=agentspan_table),
        bounds_params=lambda search_start, search_end: [
            QueryParameter("eval_id", "STRING", eval_id),
            QueryParameter("search_start_date", "DATE", search_start.isoformat()),
            QueryParameter("search_end_date", "DATE", search_end.isoformat()),
        ],
        per_entry_params=lambda start_date, resolved_end: [
            QueryParameter("eval_id", "STRING", eval_id),
            QueryParameter("start_date", "DATE", start_date.isoformat()),
            QueryParameter("end_date", "DATE", resolved_end.isoformat()),
        ],
        lookback_days=lookback_days,
        end_date=end_date,
    )
    if result is None:
        return empty_loop_count_analysis(eval_id, lookback_days=lookback_days, end_date=end_date)

    start_date, resolved_end, per_entry_rows = result
    per_entry = {
        metrics.entry_id: metrics
        for row in per_entry_rows
        for metrics in [parse_loop_count_entry_metrics(row)]
        if metrics.entry_id
    }
    if evalcli is not None:
        per_entry = overlay_evalcli_loop_and_correctness(evalcli, eval_id, per_entry)
    high_signal_entry_ids = tuple(
        sorted(entry_id for entry_id, metrics in per_entry.items() if metrics.loop_efficiency < 1.0)
    )
    if evalcli is not None:
        per_entry = _enrich_action_inputs(evalcli, per_entry, per_entry_rows, high_signal_entry_ids)
    return EvalRunLoopCountAnalysis(
        eval_id=eval_id,
        start_date=start_date,
        end_date=resolved_end,
        aggregate=aggregate_loop_count_metrics(eval_id, per_entry),
        per_entry=per_entry,
        high_signal_entry_ids=high_signal_entry_ids,
    )


def _enrich_action_inputs(
    evalcli: Any,
    per_entry: dict[str, LoopCountEntryMetrics],
    per_entry_rows: list[dict[str, Any]],
    high_signal_entry_ids: tuple[str, ...],
) -> dict[str, LoopCountEntryMetrics]:
    """Attach the student's tool payloads to high-signal entries from their traces.

    The scrubbed table cannot serve tool payloads, so resolve them from the detailed
    trace located by the scrub-safe ids returned alongside the loop counts.
    """
    high_signal = set(high_signal_entry_ids)
    locators = [
        locator
        for row in per_entry_rows
        if str(row.get("entry_id") or "") in high_signal
        for locator in [
            build_trace_locator(
                entry_id=str(row.get("entry_id") or ""),
                deployment_id=row.get("deployment_id"),
                trace_id=row.get("trace_id"),
                min_start_ms=row.get("min_start_ms"),
                max_start_ms=row.get("max_start_ms"),
            )
        ]
        if locator is not None
    ]
    if not locators:
        return per_entry
    action_inputs_by_entry = fetch_action_inputs_by_entry(
        evalcli, locators, limit=ACTION_INPUT_SURFACE_LIMIT, role_label="student"
    )
    if not action_inputs_by_entry:
        return per_entry
    return {
        entry_id: (
            replace(metrics, action_inputs=action_inputs_by_entry[entry_id])
            if entry_id in action_inputs_by_entry
            else metrics
        )
        for entry_id, metrics in per_entry.items()
    }


def overlay_evalcli_loop_and_correctness(
    evalcli: Any,
    eval_id: str,
    per_entry: dict[str, LoopCountEntryMetrics],
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
        )
    return updated


def log_loop_count_analysis(analysis: EvalRunLoopCountAnalysis) -> None:
    aggregate = analysis.aggregate
    print(
        f"[Loop Count] {analysis.eval_id}: efficiency={aggregate.loop_efficiency:.2%} "
        f"({aggregate.matching_entries}/{aggregate.compared_entries} at <= {TARGET_LOOP_COUNT} loops), "
        f"mean_loops={aggregate.mean_loop_count:.2f}, correctness={aggregate.mean_correctness:.2f}"
    )
    for entry_id in analysis.high_signal_entry_ids[:5]:
        metrics = analysis.per_entry[entry_id]
        print(
            f"[Loop Count] High-signal entry={entry_id}: loops={metrics.loop_count} "
            f"correctness={metrics.correctness:.2f} score={metrics.loop_efficiency:.2f}"
        )


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
