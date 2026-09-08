"""Teacher-vs-student tool-match metrics derived from Glean evaluation spans."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from glean_gepa.adapter_types import EntryTelemetry, TeacherStudentALRolloutOutput
from glean_gepa.shell_tool_error_util import (
    DEFAULT_AGENTS_SPAN_TABLE,
    DEFAULT_LOOKBACK_DAYS,
    QueryParameter,
    default_date_range,
    resolve_eval_run_date_range,
    wildcard_shard_filter,
)

TOOL_ALIGNMENT_OBJECTIVE = "tool_alignment"
# Bump when the persisted analysis shape changes; older entries are then refetched.
TEACHER_STUDENT_MATCH_CACHE_SCHEMA_VERSION = 1
REFLECTION_HIGH_SIGNAL_ENTRY_LIMIT = 20
SKIPPED_TOOL_NAMES = frozenset({"Personal Knowledge Vault Retrieve", "Shell", "Shell Tool"})
_EXECUTE_ACTION_FILTER = (
    "STARTS_WITH(jsonPayload.span_info.span_name, 'Execute Action:') AND jsonPayload.action.execution_mode = 'EXECUTE'"
)


class NoComparedEvalEntriesError(RuntimeError):
    """Raised when teacher/student evals produced no comparable entries."""


@dataclass(frozen=True)
class TeacherStudentMatchEntryMetrics:
    entry_id: str
    student_tools: tuple[str, ...]
    teacher_tools: tuple[str, ...]
    tools_match: bool
    student_entry_id: str = ""


@dataclass(frozen=True)
class TeacherStudentMatchMetrics:
    teacher_eval_id: str
    student_eval_id: str
    compared_entries: int
    matching_entries: int
    tool_match_rate: float


@dataclass(frozen=True)
class EvalRunTeacherStudentMatchAnalysis:
    teacher_eval_id: str
    student_eval_id: str
    start_date: date
    end_date: date
    aggregate: TeacherStudentMatchMetrics
    per_entry: dict[str, TeacherStudentMatchEntryMetrics]
    high_signal_entry_ids: tuple[str, ...]


def focused_match_rate(
    analysis: EvalRunTeacherStudentMatchAnalysis,
    requested_entry_ids: Sequence[str],
) -> float:
    """Share of the requested focused entries whose first tools matched.

    The denominator is what was asked for, not what came back: an entry missing
    from the trace query has not been fixed and must not be silently dropped.
    """
    if not requested_entry_ids:
        return 0.0
    matching = sum(1 for metrics in analysis.per_entry.values() if metrics.tools_match)
    return matching / len(requested_entry_ids)


def tool_alignment_entries(
    analysis: EvalRunTeacherStudentMatchAnalysis,
    *,
    teacher_eval_id: str,
    student_eval_id: str,
    deployment_id: str,
    query: str,
) -> list[EntryTelemetry]:
    """Turn a teacher-student comparison into per-entry tool-alignment telemetry.

    This is the tool-alignment half of the adapter's scoring pipeline: every read
    of a teacher-student match field lives here, so the adapter only composes the
    result with judges and constants and stays generic over the objective.
    """
    entries: list[EntryTelemetry] = []
    for entry_id, entry_match in analysis.per_entry.items():
        student_tools = list(entry_match.student_tools)
        teacher_tools = list(entry_match.teacher_tools)
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
            "student_eval_run_id": student_eval_id,
            "teacher_eval_run_id": teacher_eval_id,
        }
        entries.append(
            EntryTelemetry(
                entry_id=entry_id,
                student_entry_id=entry_match.student_entry_id or entry_id,
                objective_scores={TOOL_ALIGNMENT_OBJECTIVE: float(entry_match.tools_match)},
                output=output,
            )
        )
    return entries


def serialize_teacher_student_match_analysis(analysis: EvalRunTeacherStudentMatchAnalysis) -> dict[str, Any]:
    """Render an analysis as JSON so it survives the process that fetched it.

    The Agentspan shard window is measured back from today, so an unpersisted
    analysis has to be re-queried on every start and eventually falls outside the
    lookback even though the eval run itself never changes.
    """
    return {
        "schema_version": TEACHER_STUDENT_MATCH_CACHE_SCHEMA_VERSION,
        "teacher_eval_id": analysis.teacher_eval_id,
        "student_eval_id": analysis.student_eval_id,
        "start_date": analysis.start_date.isoformat(),
        "end_date": analysis.end_date.isoformat(),
        "aggregate": asdict(analysis.aggregate),
        "per_entry": {entry_id: asdict(metrics) for entry_id, metrics in analysis.per_entry.items()},
        "high_signal_entry_ids": list(analysis.high_signal_entry_ids),
    }


def parse_teacher_student_match_analysis(raw: Any) -> EvalRunTeacherStudentMatchAnalysis | None:
    """Rebuild a persisted analysis, or None when it is unusable or stale."""
    if not isinstance(raw, dict) or raw.get("schema_version") != TEACHER_STUDENT_MATCH_CACHE_SCHEMA_VERSION:
        return None
    try:
        aggregate = TeacherStudentMatchMetrics(
            teacher_eval_id=str(raw["aggregate"]["teacher_eval_id"]),
            student_eval_id=str(raw["aggregate"]["student_eval_id"]),
            compared_entries=int(raw["aggregate"]["compared_entries"]),
            matching_entries=int(raw["aggregate"]["matching_entries"]),
            tool_match_rate=float(raw["aggregate"]["tool_match_rate"]),
        )
        # A zero-entry analysis means the shard window missed the run rather than
        # that the models disagreed on nothing. Refetch instead of pinning it.
        if aggregate.compared_entries == 0:
            return None
        per_entry = {
            str(entry_id): TeacherStudentMatchEntryMetrics(
                entry_id=str(metrics["entry_id"]),
                student_tools=tuple(metrics.get("student_tools") or ()),
                teacher_tools=tuple(metrics.get("teacher_tools") or ()),
                tools_match=bool(metrics["tools_match"]),
                student_entry_id=str(metrics.get("student_entry_id") or ""),
            )
            for entry_id, metrics in (raw.get("per_entry") or {}).items()
        }
        return EvalRunTeacherStudentMatchAnalysis(
            teacher_eval_id=str(raw["teacher_eval_id"]),
            student_eval_id=str(raw["student_eval_id"]),
            start_date=date.fromisoformat(raw["start_date"]),
            end_date=date.fromisoformat(raw["end_date"]),
            aggregate=aggregate,
            per_entry=per_entry,
            high_signal_entry_ids=tuple(raw.get("high_signal_entry_ids") or ()),
        )
    except (KeyError, TypeError, ValueError):
        return None


def build_teacher_student_match_time_bounds_query(
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
  AND {_EXECUTE_ACTION_FILTER}
""".strip()


def build_teacher_student_match_per_entry_query(
    *,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
    remap_focused_entry_ids: bool = False,
) -> str:
    """Build SQL that pairs teacher and student tool sequences per eval entry.

    Same-eval-set runs share ``entry_id`` and join directly. High-signal screens
    run the student on a focused upload (new entry IDs) against the original
    teacher eval, so the join uses ``@source_entry_ids`` / ``@focused_entry_ids``.
    """
    skipped = ", ".join(f"'{name}'" for name in sorted(SKIPPED_TOOL_NAMES))
    tool_spans = f"""
WITH tool_spans AS (
  SELECT
    jsonPayload.context.eval.eval_id AS eval_id,
    COALESCE(
      jsonPayload.context.eval.entry_uuid,
      CAST(jsonPayload.context.eval.entry_id AS STRING)
    ) AS entry_id,
    REGEXP_REPLACE(jsonPayload.span_info.span_name, r'^Execute Action: ', '') AS tool_name,
    SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64) AS start_ms
  FROM `{agentspan_table}`
  WHERE {wildcard_shard_filter("start_date", "end_date")}
    AND jsonPayload.context.eval.eval_id IN UNNEST(@eval_ids)
    AND {_EXECUTE_ACTION_FILTER}
    AND REGEXP_REPLACE(jsonPayload.span_info.span_name, r'^Execute Action: ', '') NOT IN ({skipped})
),
per_role AS (
  SELECT
    entry_id,
    eval_id,
    ARRAY_AGG(tool_name IGNORE NULLS ORDER BY start_ms) AS tools
  FROM tool_spans
  WHERE entry_id IS NOT NULL
  GROUP BY entry_id, eval_id
),
student AS (
  SELECT entry_id, tools FROM per_role WHERE eval_id = @student_eval_id
),
teacher AS (
  SELECT entry_id, tools FROM per_role WHERE eval_id = @teacher_eval_id
)
""".strip()
    if not remap_focused_entry_ids:
        return f"""
{tool_spans}
SELECT
  COALESCE(student.entry_id, teacher.entry_id) AS entry_id,
  COALESCE(student.entry_id, teacher.entry_id) AS student_entry_id,
  IFNULL(student.tools, ARRAY<STRING>[]) AS student_tools,
  IFNULL(teacher.tools, ARRAY<STRING>[]) AS teacher_tools
FROM student
FULL OUTER JOIN teacher
  ON student.entry_id = teacher.entry_id
ORDER BY entry_id
""".strip()
    return f"""
{tool_spans},
mapping AS (
  SELECT
    @source_entry_ids[OFFSET(i)] AS source_entry_id,
    @focused_entry_ids[OFFSET(i)] AS focused_entry_id
  FROM UNNEST(GENERATE_ARRAY(0, ARRAY_LENGTH(@source_entry_ids) - 1)) AS i
)
SELECT
  mapping.source_entry_id AS entry_id,
  mapping.focused_entry_id AS student_entry_id,
  IFNULL(student.tools, ARRAY<STRING>[]) AS student_tools,
  IFNULL(teacher.tools, ARRAY<STRING>[]) AS teacher_tools
FROM mapping
LEFT JOIN student
  ON student.entry_id = mapping.focused_entry_id
LEFT JOIN teacher
  ON teacher.entry_id = mapping.source_entry_id
ORDER BY entry_id
""".strip()


def scored_tool_sequence(tools: Sequence[str] | None) -> tuple[str, ...]:
    """Return tool names used for sequence matching, dropping Shell and other skipped tools."""
    return tuple(str(name) for name in (tools or []) if name and str(name) not in SKIPPED_TOOL_NAMES)


def first_tool_mismatch_pair(
    teacher_tools: Sequence[str] | None,
    student_tools: Sequence[str] | None,
) -> tuple[str, str] | None:
    """Return ``(teacher_first, student_first)`` when they differ, else ``None``.

    A side that scores no tools at all is reported as an empty name, which
    mismatches any real tool.
    """
    teacher_scored = scored_tool_sequence(teacher_tools)
    student_scored = scored_tool_sequence(student_tools)
    teacher = teacher_scored[0] if teacher_scored else ""
    student = student_scored[0] if student_scored else ""
    if teacher == student:
        return None
    return (teacher, student)


def select_first_tool_mismatch_groups(
    mismatch_keys: Sequence[tuple[str, str] | None],
    *,
    max_entries: int = REFLECTION_HIGH_SIGNAL_ENTRY_LIMIT,
) -> tuple[list[int], list[tuple[str, str, int]]]:
    """Select first-tool mismatch indices by descending ``(teacher, student)`` frequency.

    The most frequent group is always included in full, even when it exceeds
    ``max_entries``. Later whole groups are added while they still fit in the
    cap; groups that would overflow are skipped so later smaller groups can
    still be included.

    Returns ``(selected_indices, selected_groups)`` where each group is
    ``(teacher_tool, student_tool, taken_count)``.
    """
    if max_entries < 0:
        raise ValueError("max_entries must be non-negative")
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, key in enumerate(mismatch_keys):
        if key is None:
            continue
        groups[key].append(index)
    ranked = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1]))
    selected: list[int] = []
    selected_groups: list[tuple[str, str, int]] = []
    for (teacher_tool, student_tool), indices in ranked:
        if selected and len(selected) + len(indices) > max_entries:
            continue
        selected.extend(indices)
        selected_groups.append((teacher_tool, student_tool, len(indices)))
        if len(selected) >= max_entries:
            break
    return selected, selected_groups


def parse_teacher_student_match_entry_metrics(row: dict[str, Any]) -> TeacherStudentMatchEntryMetrics:
    student_tools = scored_tool_sequence(row.get("student_tools"))
    teacher_tools = scored_tool_sequence(row.get("teacher_tools"))
    entry_id = str(row.get("entry_id") or "")
    student_entry_id = str(row.get("student_entry_id") or entry_id)
    return TeacherStudentMatchEntryMetrics(
        entry_id=entry_id,
        student_tools=student_tools,
        teacher_tools=teacher_tools,
        tools_match=(student_tools[:1] == teacher_tools[:1]),
        student_entry_id=student_entry_id,
    )


def aggregate_teacher_student_match_metrics(
    teacher_eval_id: str,
    student_eval_id: str,
    per_entry: dict[str, TeacherStudentMatchEntryMetrics],
) -> TeacherStudentMatchMetrics:
    compared = len(per_entry)
    matching = sum(1 for metrics in per_entry.values() if metrics.tools_match)
    return TeacherStudentMatchMetrics(
        teacher_eval_id=teacher_eval_id,
        student_eval_id=student_eval_id,
        compared_entries=compared,
        matching_entries=matching,
        tool_match_rate=(matching / compared) if compared else 0.0,
    )


def empty_teacher_student_match_analysis(
    teacher_eval_id: str,
    student_eval_id: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: date | None = None,
) -> EvalRunTeacherStudentMatchAnalysis:
    start_date, resolved_end = default_date_range(lookback_days=lookback_days, end_date=end_date)
    return EvalRunTeacherStudentMatchAnalysis(
        teacher_eval_id=teacher_eval_id,
        student_eval_id=student_eval_id,
        start_date=start_date,
        end_date=resolved_end,
        aggregate=aggregate_teacher_student_match_metrics(teacher_eval_id, student_eval_id, {}),
        per_entry={},
        high_signal_entry_ids=(),
    )


def fetch_eval_run_teacher_student_match_analysis(
    client: Any,
    *,
    teacher_eval_id: str,
    student_eval_id: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: date | None = None,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
    entry_id_pairs: Sequence[tuple[str, str]] | None = None,
) -> EvalRunTeacherStudentMatchAnalysis:
    search_start, search_end = default_date_range(lookback_days=lookback_days, end_date=end_date)
    bounds_rows = client.query(
        build_teacher_student_match_time_bounds_query(agentspan_table=agentspan_table),
        params=[
            QueryParameter("eval_ids", "STRING", [teacher_eval_id, student_eval_id]),
            QueryParameter("search_start_date", "DATE", search_start.isoformat()),
            QueryParameter("search_end_date", "DATE", search_end.isoformat()),
        ],
    )
    date_range = resolve_eval_run_date_range(
        bounds_rows[0] if bounds_rows else None,
        lookback_days=lookback_days,
        end_date=end_date,
    )
    if date_range is None:
        return empty_teacher_student_match_analysis(
            teacher_eval_id,
            student_eval_id,
            lookback_days=lookback_days,
            end_date=end_date,
        )

    start_date, resolved_end = date_range
    remap = bool(entry_id_pairs)
    per_entry_params = [
        QueryParameter("eval_ids", "STRING", [teacher_eval_id, student_eval_id]),
        QueryParameter("student_eval_id", "STRING", student_eval_id),
        QueryParameter("teacher_eval_id", "STRING", teacher_eval_id),
        QueryParameter("start_date", "DATE", start_date.isoformat()),
        QueryParameter("end_date", "DATE", resolved_end.isoformat()),
    ]
    if remap:
        source_entry_ids = [source_id for source_id, _focused_id in entry_id_pairs or ()]
        focused_entry_ids = [focused_id for _source_id, focused_id in entry_id_pairs or ()]
        per_entry_params.extend(
            [
                QueryParameter("source_entry_ids", "STRING", source_entry_ids),
                QueryParameter("focused_entry_ids", "STRING", focused_entry_ids),
            ]
        )
    per_entry_rows = client.query(
        build_teacher_student_match_per_entry_query(
            agentspan_table=agentspan_table,
            remap_focused_entry_ids=remap,
        ),
        params=per_entry_params,
    )
    per_entry = {
        metrics.entry_id: metrics
        for row in per_entry_rows
        for metrics in [parse_teacher_student_match_entry_metrics(row)]
        if metrics.entry_id
    }
    return EvalRunTeacherStudentMatchAnalysis(
        teacher_eval_id=teacher_eval_id,
        student_eval_id=student_eval_id,
        start_date=start_date,
        end_date=resolved_end,
        aggregate=aggregate_teacher_student_match_metrics(teacher_eval_id, student_eval_id, per_entry),
        per_entry=per_entry,
        high_signal_entry_ids=tuple(
            sorted(entry_id for entry_id, metrics in per_entry.items() if not metrics.tools_match)
        ),
    )


def require_compared_eval_entries(analysis: EvalRunTeacherStudentMatchAnalysis) -> None:
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


def log_teacher_student_match_analysis(analysis: EvalRunTeacherStudentMatchAnalysis) -> None:
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
