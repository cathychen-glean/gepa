"""Teacher-vs-student tool-match metrics derived from Glean evaluation spans."""

from __future__ import annotations

from collections.abc import Sequence
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

TOOL_ALIGNMENT_OBJECTIVE = "tool_alignment"
SKIPPED_TOOL_NAMES = frozenset({"Personal Knowledge Vault Retrieve", "Shell", "Shell Tool"})
# Reflection surfaces at most this many tool payloads per entry.
ACTION_INPUT_SURFACE_LIMIT = 5


class NoComparedEvalEntriesError(RuntimeError):
    """Raised when teacher/student evals produced no comparable entries."""


@dataclass(frozen=True)
class ToolMatchEntryMetrics:
    entry_id: str
    student_tools: tuple[str, ...]
    teacher_tools: tuple[str, ...]
    tools_match: bool
    student_action_inputs: tuple[str, ...] = ()
    teacher_action_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolMatchMetrics:
    teacher_eval_id: str
    student_eval_id: str
    compared_entries: int
    matching_entries: int
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


def build_tool_match_time_bounds_query(
    *,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> str:
    """Find min/max Execute Action timestamps for a teacher/student eval pair."""
    return f"""
SELECT
  MIN(SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64)) AS min_start_ms,
  MAX(SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64)) AS max_start_ms
FROM `{agentspan_table}`
WHERE {wildcard_shard_filter("search_start_date", "search_end_date")}
  AND jsonPayload.context.eval.eval_id IN UNNEST(@eval_ids)
  AND {EXECUTE_ACTION_FILTER}
""".strip()


def build_tool_match_per_entry_query(
    *,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> str:
    """Build SQL that pairs teacher and student tool sequences per eval entry."""
    skipped = ", ".join(f"'{name}'" for name in sorted(SKIPPED_TOOL_NAMES))
    return f"""
WITH tool_spans AS (
  SELECT
    jsonPayload.context.eval.eval_id AS eval_id,
    COALESCE(
      jsonPayload.context.eval.entry_uuid,
      CAST(jsonPayload.context.eval.entry_id AS STRING)
    ) AS entry_id,
    REGEXP_REPLACE(jsonPayload.span_info.span_name, r'^Execute Action: ', '') AS tool_name,
    jsonPayload.context.agent_trace.trace_id AS trace_id,
    resource.labels.project_id AS deployment_id,
    SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64) AS start_ms
  FROM `{agentspan_table}`
  WHERE {wildcard_shard_filter("start_date", "end_date")}
    AND jsonPayload.context.eval.eval_id IN UNNEST(@eval_ids)
    AND {EXECUTE_ACTION_FILTER}
    AND REGEXP_REPLACE(jsonPayload.span_info.span_name, r'^Execute Action: ', '') NOT IN ({skipped})
),
per_role AS (
  SELECT
    entry_id,
    eval_id,
    ARRAY_AGG(tool_name IGNORE NULLS ORDER BY start_ms) AS tools,
    ANY_VALUE(trace_id) AS trace_id,
    ANY_VALUE(deployment_id) AS deployment_id,
    MIN(start_ms) AS min_start_ms,
    MAX(start_ms) AS max_start_ms
  FROM tool_spans
  WHERE entry_id IS NOT NULL
  GROUP BY entry_id, eval_id
),
student AS (
  SELECT entry_id, tools, trace_id, deployment_id, min_start_ms, max_start_ms
  FROM per_role WHERE eval_id = @student_eval_id
),
teacher AS (
  SELECT entry_id, tools, trace_id, deployment_id, min_start_ms, max_start_ms
  FROM per_role WHERE eval_id = @teacher_eval_id
)
SELECT
  COALESCE(student.entry_id, teacher.entry_id) AS entry_id,
  IFNULL(student.tools, ARRAY<STRING>[]) AS student_tools,
  IFNULL(teacher.tools, ARRAY<STRING>[]) AS teacher_tools,
  student.trace_id AS student_trace_id,
  student.deployment_id AS student_deployment_id,
  student.min_start_ms AS student_min_start_ms,
  student.max_start_ms AS student_max_start_ms,
  teacher.trace_id AS teacher_trace_id,
  teacher.deployment_id AS teacher_deployment_id,
  teacher.min_start_ms AS teacher_min_start_ms,
  teacher.max_start_ms AS teacher_max_start_ms
FROM student
FULL OUTER JOIN teacher
  ON student.entry_id = teacher.entry_id
ORDER BY entry_id
""".strip()


def scored_tool_sequence(tools: Sequence[str] | None) -> tuple[str, ...]:
    """Return tool names used for sequence matching, dropping Shell and other skipped tools."""
    return tuple(str(name) for name in (tools or []) if name and str(name) not in SKIPPED_TOOL_NAMES)


def first_tool_name(tools: Sequence[str] | None) -> str:
    """Return the first scored tool name, or an empty string when none remain."""
    scored = scored_tool_sequence(tools)
    return scored[0] if scored else ""


def first_tool_mismatch_pair(
    teacher_tools: Sequence[str] | None,
    student_tools: Sequence[str] | None,
) -> tuple[str, str] | None:
    """Return ``(teacher_first, student_first)`` when they differ, else ``None``."""
    teacher = first_tool_name(teacher_tools)
    student = first_tool_name(student_tools)
    if teacher == student:
        return None
    return (teacher, student)


def parse_tool_match_entry_metrics(row: dict[str, Any]) -> ToolMatchEntryMetrics:
    student_tools = scored_tool_sequence(row.get("student_tools"))
    teacher_tools = scored_tool_sequence(row.get("teacher_tools"))
    return ToolMatchEntryMetrics(
        entry_id=str(row.get("entry_id") or ""),
        student_tools=student_tools,
        teacher_tools=teacher_tools,
        tools_match=(student_tools[:1] == teacher_tools[:1]),
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
    evalcli: Any | None = None,
) -> EvalRunToolMatchAnalysis:
    eval_ids = [teacher_eval_id, student_eval_id]
    result = run_windowed_per_entry_query(
        client,
        bounds_query=build_tool_match_time_bounds_query(agentspan_table=agentspan_table),
        per_entry_query=build_tool_match_per_entry_query(agentspan_table=agentspan_table),
        bounds_params=lambda search_start, search_end: [
            QueryParameter("eval_ids", "STRING", eval_ids),
            QueryParameter("search_start_date", "DATE", search_start.isoformat()),
            QueryParameter("search_end_date", "DATE", search_end.isoformat()),
        ],
        per_entry_params=lambda start_date, resolved_end: [
            QueryParameter("eval_ids", "STRING", eval_ids),
            QueryParameter("student_eval_id", "STRING", student_eval_id),
            QueryParameter("teacher_eval_id", "STRING", teacher_eval_id),
            QueryParameter("start_date", "DATE", start_date.isoformat()),
            QueryParameter("end_date", "DATE", resolved_end.isoformat()),
        ],
        lookback_days=lookback_days,
        end_date=end_date,
    )
    if result is None:
        return empty_tool_match_analysis(
            teacher_eval_id,
            student_eval_id,
            lookback_days=lookback_days,
            end_date=end_date,
        )

    start_date, resolved_end, per_entry_rows = result
    per_entry = {
        metrics.entry_id: metrics
        for row in per_entry_rows
        for metrics in [parse_tool_match_entry_metrics(row)]
        if metrics.entry_id
    }
    high_signal_entry_ids = tuple(
        sorted(entry_id for entry_id, metrics in per_entry.items() if not metrics.tools_match)
    )
    if evalcli is not None:
        per_entry = _enrich_action_inputs(evalcli, per_entry, per_entry_rows, high_signal_entry_ids)
    return EvalRunToolMatchAnalysis(
        teacher_eval_id=teacher_eval_id,
        student_eval_id=student_eval_id,
        start_date=start_date,
        end_date=resolved_end,
        aggregate=aggregate_tool_match_metrics(teacher_eval_id, student_eval_id, per_entry),
        per_entry=per_entry,
        high_signal_entry_ids=high_signal_entry_ids,
    )


def _enrich_action_inputs(
    evalcli: Any,
    per_entry: dict[str, ToolMatchEntryMetrics],
    per_entry_rows: list[dict[str, Any]],
    high_signal_entry_ids: tuple[str, ...],
) -> dict[str, ToolMatchEntryMetrics]:
    """Attach teacher and student tool payloads to high-signal entries from traces.

    The scrubbed table cannot serve tool payloads, so resolve them from each role's
    detailed trace located by the scrub-safe ids returned alongside the tool names.
    """
    high_signal = set(high_signal_entry_ids)
    rows = [row for row in per_entry_rows if str(row.get("entry_id") or "") in high_signal]
    if not rows:
        return per_entry

    def _locators(role: str) -> list[Any]:
        collected = []
        for row in rows:
            locator = build_trace_locator(
                entry_id=str(row.get("entry_id") or ""),
                deployment_id=row.get(f"{role}_deployment_id"),
                trace_id=row.get(f"{role}_trace_id"),
                min_start_ms=row.get(f"{role}_min_start_ms"),
                max_start_ms=row.get(f"{role}_max_start_ms"),
            )
            if locator is not None:
                collected.append(locator)
        return collected

    student_inputs = fetch_action_inputs_by_entry(
        evalcli,
        _locators("student"),
        skip_tools=SKIPPED_TOOL_NAMES,
        limit=ACTION_INPUT_SURFACE_LIMIT,
        role_label="student",
    )
    teacher_inputs = fetch_action_inputs_by_entry(
        evalcli,
        _locators("teacher"),
        skip_tools=SKIPPED_TOOL_NAMES,
        limit=ACTION_INPUT_SURFACE_LIMIT,
        role_label="teacher",
    )
    if not student_inputs and not teacher_inputs:
        return per_entry
    return {
        entry_id: replace(
            metrics,
            student_action_inputs=student_inputs.get(entry_id, metrics.student_action_inputs),
            teacher_action_inputs=teacher_inputs.get(entry_id, metrics.teacher_action_inputs),
        )
        for entry_id, metrics in per_entry.items()
    }


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
        f"{aggregate.tool_match_rate:.2%} first-tool match "
        f"({aggregate.matching_entries}/{aggregate.compared_entries})"
    )
    for entry_id in analysis.high_signal_entry_ids[:5]:
        metrics = analysis.per_entry[entry_id]
        print(
            f"[Tool Match] Mismatch entry={entry_id}: "
            f"student={list(metrics.student_tools[:5])} teacher={list(metrics.teacher_tools[:5])}"
        )
