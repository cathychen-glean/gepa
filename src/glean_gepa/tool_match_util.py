"""Teacher-vs-student tool-match metrics derived from Glean evaluation spans."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

from glean_gepa.shell_tool_error_util import (
    DEFAULT_AGENTS_SPAN_TABLE,
    DEFAULT_LOOKBACK_DAYS,
    QueryParameter,
    default_date_range,
    resolve_eval_run_date_range,
)

SKIPPED_TOOL_NAMES = frozenset({"Personal Knowledge Vault Retrieve"})


class NoComparedEvalEntriesError(RuntimeError):
    """Raised when teacher/student evals produced no comparable entries."""


@dataclass(frozen=True)
class ToolMatchEntryMetrics:
    entry_id: str
    student_tools: tuple[str, ...]
    teacher_tools: tuple[str, ...]
    tools_match: bool
    tool_match_score: float
    student_trace_id: str | None = None
    teacher_trace_id: str | None = None

    @property
    def has_mismatch(self) -> bool:
        return not self.tools_match


@dataclass(frozen=True)
class ToolMatchMetrics:
    teacher_eval_id: str
    student_eval_id: str
    compared_entries: int
    matching_entries: int
    mismatching_entries: int
    tool_match_rate: float


@dataclass(frozen=True)
class EvalRunToolMatchAnalysis:
    teacher_eval_id: str
    student_eval_id: str
    start_date: date
    end_date: date
    aggregate: ToolMatchMetrics
    per_entry: dict[str, ToolMatchEntryMetrics]
    high_signal_entry_ids: tuple[str, ...]


def _execute_action_filter_sql() -> str:
    return (
        "STARTS_WITH(jsonPayload.span_info.span_name, 'Execute Action:') "
        "AND jsonPayload.action.execution_mode = 'EXECUTE'"
    )


def build_tool_match_time_bounds_query(
    *,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> str:
    """Find min/max Execute Action timestamps for a teacher/student eval pair."""
    action_filter = _execute_action_filter_sql()
    return f"""
SELECT
  MIN(SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64)) AS min_start_ms,
  MAX(SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64)) AS max_start_ms
FROM `{agentspan_table}`
WHERE PARSE_DATE('%Y%m%d', _TABLE_SUFFIX)
  BETWEEN @search_start_date AND @search_end_date
  AND jsonPayload.context.eval.eval_id IN UNNEST(@eval_ids)
  AND {action_filter}
""".strip()


def build_tool_match_per_entry_query(
    *,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> str:
    """Build SQL that pairs teacher and student tool sequences per eval entry."""
    action_filter = _execute_action_filter_sql()
    skipped = ", ".join(f"'{name}'" for name in sorted(SKIPPED_TOOL_NAMES))
    return f"""
WITH tool_spans AS (
  SELECT
    jsonPayload.context.eval.eval_id AS eval_id,
    COALESCE(
      jsonPayload.context.eval.entry_uuid,
      CAST(jsonPayload.context.eval.entry_id AS STRING)
    ) AS entry_id,
    jsonPayload.context.agent_trace.trace_id AS trace_id,
    REGEXP_REPLACE(jsonPayload.span_info.span_name, r'^Execute Action: ', '') AS tool_name,
    SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64) AS start_ms
  FROM `{agentspan_table}`
  WHERE PARSE_DATE('%Y%m%d', _TABLE_SUFFIX)
    BETWEEN @start_date AND @end_date
    AND jsonPayload.context.eval.eval_id IN UNNEST(@eval_ids)
    AND {action_filter}
    AND REGEXP_REPLACE(jsonPayload.span_info.span_name, r'^Execute Action: ', '') NOT IN ({skipped})
),
per_role AS (
  SELECT
    entry_id,
    eval_id,
    ARRAY_AGG(tool_name IGNORE NULLS ORDER BY start_ms) AS tools,
    ARRAY_AGG(trace_id IGNORE NULLS ORDER BY start_ms LIMIT 1)[SAFE_OFFSET(0)] AS trace_id
  FROM tool_spans
  WHERE entry_id IS NOT NULL
  GROUP BY entry_id, eval_id
),
student AS (
  SELECT entry_id, tools, trace_id FROM per_role WHERE eval_id = @student_eval_id
),
teacher AS (
  SELECT entry_id, tools, trace_id FROM per_role WHERE eval_id = @teacher_eval_id
)
SELECT
  COALESCE(student.entry_id, teacher.entry_id) AS entry_id,
  IFNULL(student.tools, ARRAY<STRING>[]) AS student_tools,
  IFNULL(teacher.tools, ARRAY<STRING>[]) AS teacher_tools,
  student.trace_id AS student_trace_id,
  teacher.trace_id AS teacher_trace_id
FROM student
FULL OUTER JOIN teacher
  ON student.entry_id = teacher.entry_id
ORDER BY entry_id
""".strip()


def build_tool_match_search_params(
    *,
    teacher_eval_id: str,
    student_eval_id: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: date | None = None,
) -> list[QueryParameter]:
    search_start, search_end = default_date_range(lookback_days=lookback_days, end_date=end_date)
    return [
        QueryParameter("eval_ids", "STRING", [teacher_eval_id, student_eval_id]),
        QueryParameter("search_start_date", "DATE", search_start.isoformat()),
        QueryParameter("search_end_date", "DATE", search_end.isoformat()),
    ]


def build_tool_match_query_params(
    *,
    teacher_eval_id: str,
    student_eval_id: str,
    start_date: date,
    end_date: date,
) -> list[QueryParameter]:
    return [
        QueryParameter("eval_ids", "STRING", [teacher_eval_id, student_eval_id]),
        QueryParameter("student_eval_id", "STRING", student_eval_id),
        QueryParameter("teacher_eval_id", "STRING", teacher_eval_id),
        QueryParameter("start_date", "DATE", start_date.isoformat()),
        QueryParameter("end_date", "DATE", end_date.isoformat()),
    ]


def compute_tool_match_score(student_tools: Sequence[str], teacher_tools: Sequence[str]) -> float:
    """Return a 0-1 score for ordered tool-name alignment."""
    if not student_tools and not teacher_tools:
        return 1.0
    if not student_tools or not teacher_tools:
        return 0.0
    if tuple(student_tools) == tuple(teacher_tools):
        return 1.0
    compared = max(len(student_tools), len(teacher_tools))
    matches = 0
    for index, student_tool in enumerate(student_tools):
        if index < len(teacher_tools) and student_tool == teacher_tools[index]:
            matches += 1
    return matches / compared


def parse_tool_match_entry_metrics(row: dict[str, Any]) -> ToolMatchEntryMetrics:
    student_tools = tuple(str(name) for name in (row.get("student_tools") or []) if name)
    teacher_tools = tuple(str(name) for name in (row.get("teacher_tools") or []) if name)
    return ToolMatchEntryMetrics(
        entry_id=str(row.get("entry_id") or ""),
        student_tools=student_tools,
        teacher_tools=teacher_tools,
        tools_match=student_tools == teacher_tools,
        tool_match_score=compute_tool_match_score(student_tools, teacher_tools),
        student_trace_id=str(row["student_trace_id"]) if row.get("student_trace_id") else None,
        teacher_trace_id=str(row["teacher_trace_id"]) if row.get("teacher_trace_id") else None,
    )


def aggregate_tool_match_metrics(
    teacher_eval_id: str,
    student_eval_id: str,
    per_entry: dict[str, ToolMatchEntryMetrics],
) -> ToolMatchMetrics:
    compared = len(per_entry)
    matching = sum(1 for metrics in per_entry.values() if metrics.tools_match)
    return ToolMatchMetrics(
        teacher_eval_id=teacher_eval_id,
        student_eval_id=student_eval_id,
        compared_entries=compared,
        matching_entries=matching,
        mismatching_entries=compared - matching,
        tool_match_rate=(matching / compared) if compared else 0.0,
    )


def empty_tool_match_analysis(
    teacher_eval_id: str,
    student_eval_id: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: date | None = None,
) -> EvalRunToolMatchAnalysis:
    start_date, resolved_end = default_date_range(lookback_days=lookback_days, end_date=end_date)
    return EvalRunToolMatchAnalysis(
        teacher_eval_id=teacher_eval_id,
        student_eval_id=student_eval_id,
        start_date=start_date,
        end_date=resolved_end,
        aggregate=aggregate_tool_match_metrics(teacher_eval_id, student_eval_id, {}),
        per_entry={},
        high_signal_entry_ids=(),
    )


def fetch_eval_run_tool_match_analysis(
    client: Any,
    *,
    teacher_eval_id: str,
    student_eval_id: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: date | None = None,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> EvalRunToolMatchAnalysis:
    bounds_rows = client.query(
        build_tool_match_time_bounds_query(agentspan_table=agentspan_table),
        params=build_tool_match_search_params(
            teacher_eval_id=teacher_eval_id,
            student_eval_id=student_eval_id,
            lookback_days=lookback_days,
            end_date=end_date,
        ),
    )
    date_range = resolve_eval_run_date_range(
        bounds_rows[0] if bounds_rows else None,
        lookback_days=lookback_days,
        end_date=end_date,
    )
    if date_range is None:
        return empty_tool_match_analysis(
            teacher_eval_id,
            student_eval_id,
            lookback_days=lookback_days,
            end_date=end_date,
        )

    start_date, resolved_end = date_range
    per_entry_rows = client.query(
        build_tool_match_per_entry_query(agentspan_table=agentspan_table),
        params=build_tool_match_query_params(
            teacher_eval_id=teacher_eval_id,
            student_eval_id=student_eval_id,
            start_date=start_date,
            end_date=resolved_end,
        ),
    )
    per_entry = {
        metrics.entry_id: metrics
        for row in per_entry_rows
        for metrics in [parse_tool_match_entry_metrics(row)]
        if metrics.entry_id
    }
    return EvalRunToolMatchAnalysis(
        teacher_eval_id=teacher_eval_id,
        student_eval_id=student_eval_id,
        start_date=start_date,
        end_date=resolved_end,
        aggregate=aggregate_tool_match_metrics(teacher_eval_id, student_eval_id, per_entry),
        per_entry=per_entry,
        high_signal_entry_ids=tuple(
            sorted(entry_id for entry_id, metrics in per_entry.items() if metrics.has_mismatch)
        ),
    )


def require_compared_eval_entries(analysis: EvalRunToolMatchAnalysis) -> None:
    """Reject an analysis with zero compared entries.

    A 0/0 comparison is not a 100% match: it means neither eval produced
    comparable Execute Action spans, so tool alignment is undefined.
    """
    if analysis.aggregate.compared_entries > 0:
        return
    raise NoComparedEvalEntriesError(
        f"No eval entries were compared for student {analysis.student_eval_id} vs "
        f"teacher {analysis.teacher_eval_id}. Wait for agentspan ingest or "
        f"check that the eval runs actually executed entries."
    )


def log_tool_match_analysis(analysis: EvalRunToolMatchAnalysis) -> None:
    aggregate = analysis.aggregate
    print(
        f"[Tool Match] {analysis.student_eval_id} vs {analysis.teacher_eval_id}: "
        f"{aggregate.tool_match_rate:.2%} exact match "
        f"({aggregate.matching_entries}/{aggregate.compared_entries})"
    )
    for entry_id in analysis.high_signal_entry_ids[:5]:
        metrics = analysis.per_entry[entry_id]
        print(
            f"[Tool Match] Mismatch entry={entry_id}: "
            f"student={list(metrics.student_tools[:5])} teacher={list(metrics.teacher_tools[:5])}"
        )
