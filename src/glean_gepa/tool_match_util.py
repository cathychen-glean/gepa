"""Teacher-vs-student tool-match metrics derived from Glean evaluation spans."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from glean_gepa.shell_tool_error_util import (
    DEFAULT_AGENTS_SPAN_TABLE,
    DEFAULT_LOOKBACK_DAYS,
    QueryParameter,
    default_date_range,
    resolve_eval_run_date_range,
    wildcard_shard_filter,
)

REFLECTION_HIGH_SIGNAL_ENTRY_LIMIT = 20
PREFIX_K_GRADUATION_RATE = 0.75
PREFIX_K_TOP_CANDIDATES = 2
PREFIX_K_SHORTAGE_ENTRIES = 20
SKIPPED_TOOL_NAMES = frozenset({"Personal Knowledge Vault Retrieve", "Shell", "Shell Tool"})
_EXECUTE_ACTION_FILTER = (
    "STARTS_WITH(jsonPayload.span_info.span_name, 'Execute Action:') AND jsonPayload.action.execution_mode = 'EXECUTE'"
)


class NoComparedEvalEntriesError(RuntimeError):
    """Raised when teacher/student evals produced no comparable entries."""


@dataclass(frozen=True)
class ToolMatchEntryMetrics:
    entry_id: str
    student_tools: tuple[str, ...]
    teacher_tools: tuple[str, ...]
    tools_match: bool


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
  AND {_EXECUTE_ACTION_FILTER}
""".strip()


def build_tool_match_per_entry_query(
    *,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> str:
    """Build SQL that pairs teacher and student tool sequences per eval entry.

    The teacher side is the denominator: every teacher entry is kept, and a
    missing student run is an empty tool list (a mismatch). Student-only
    extras are dropped so every child compared to the same teacher uses the
    same entry set.
    """
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
SELECT
  teacher.entry_id AS entry_id,
  IFNULL(student.tools, ARRAY<STRING>[]) AS student_tools,
  teacher.tools AS teacher_tools
FROM teacher
LEFT JOIN student
  ON student.entry_id = teacher.entry_id
ORDER BY entry_id
""".strip()


def scored_tool_sequence(tools: Sequence[str] | None) -> tuple[str, ...]:
    """Return tool names used for sequence matching, dropping Shell and other skipped tools."""
    return tuple(str(name) for name in (tools or []) if name and str(name) not in SKIPPED_TOOL_NAMES)


def collapse_consecutive_tools(tools: Sequence[str]) -> tuple[str, ...]:
    """Collapse adjacent duplicate tool names. ``[Write, Search, Write]`` stays three steps."""
    collapsed: list[str] = []
    for name in tools:
        if not collapsed or collapsed[-1] != name:
            collapsed.append(name)
    return tuple(collapsed)


def match_tool_sequence(tools: Sequence[str] | None) -> tuple[str, ...]:
    """Strip skipped tools, then collapse consecutive duplicates."""
    return collapse_consecutive_tools(scored_tool_sequence(tools))


def prefix_slot_weights(prefix_k: int) -> tuple[float, ...]:
    """Return unnormalized slot weights for a prefix of length ``prefix_k``."""
    if prefix_k < 1:
        raise ValueError("prefix_k must be >= 1")
    if prefix_k == 1:
        return (1.0,)
    if prefix_k == 2:
        return (0.75, 0.25)
    later = 0.5 / (prefix_k - 1)
    return (0.5,) + (later,) * (prefix_k - 1)


def tool_prefix_alignment(
    teacher_tools: Sequence[str] | None,
    student_tools: Sequence[str] | None,
    prefix_k: int = 1,
) -> float:
    """Weighted prefix alignment after consecutive-run collapse.

    Slots past both sequence lengths are skipped and remaining weights are
    renormalized. A miss at slot i zeros later slots. Matching every tool that
    exists scores 1.0.
    """
    if prefix_k < 1:
        raise ValueError("prefix_k must be >= 1")
    teacher = match_tool_sequence(teacher_tools)
    student = match_tool_sequence(student_tools)
    weights = prefix_slot_weights(prefix_k)
    used_weights: list[float] = []
    used_matches: list[float] = []
    for index in range(prefix_k):
        teacher_has = index < len(teacher)
        student_has = index < len(student)
        if not teacher_has and not student_has:
            continue
        matched = teacher_has and student_has and teacher[index] == student[index]
        used_weights.append(weights[index])
        used_matches.append(1.0 if matched else 0.0)
        if not matched:
            break
    if not used_weights:
        return 1.0
    return sum(match * weight for match, weight in zip(used_matches, used_weights, strict=True)) / sum(used_weights)


def exact_prefix_match(
    teacher_tools: Sequence[str] | None,
    student_tools: Sequence[str] | None,
    prefix_k: int = 1,
) -> bool:
    """True when every scored prefix slot that exists matches."""
    return tool_prefix_alignment(teacher_tools, student_tools, prefix_k) >= 1.0


def first_tool_name(tools: Sequence[str] | None) -> str:
    """Return the first scored tool name, or an empty string when none remain."""
    scored = match_tool_sequence(tools)
    return scored[0] if scored else ""


def first_disagreeing_mismatch(
    teacher_tools: Sequence[str] | None,
    student_tools: Sequence[str] | None,
    prefix_k: int = 1,
) -> tuple[int, str, str] | None:
    """Return ``(0-based slot, teacher_tool, student_tool)`` at the first prefix miss."""
    if prefix_k < 1:
        raise ValueError("prefix_k must be >= 1")
    teacher = match_tool_sequence(teacher_tools)
    student = match_tool_sequence(student_tools)
    for index in range(prefix_k):
        teacher_has = index < len(teacher)
        student_has = index < len(student)
        if not teacher_has and not student_has:
            continue
        teacher_name = teacher[index] if teacher_has else ""
        student_name = student[index] if student_has else ""
        if teacher_name != student_name:
            return (index, teacher_name, student_name)
    return None


def first_tool_mismatch_pair(
    teacher_tools: Sequence[str] | None,
    student_tools: Sequence[str] | None,
) -> tuple[str, str] | None:
    """Return ``(teacher_first, student_first)`` when they differ, else ``None``."""
    mismatch = first_disagreeing_mismatch(teacher_tools, student_tools, prefix_k=1)
    if mismatch is None:
        return None
    return (mismatch[1], mismatch[2])


def prefix_tools(
    tools: Sequence[str] | None,
    prefix_k: int,
) -> tuple[str, ...]:
    """Return the collapsed tool prefix of length ``prefix_k``."""
    if prefix_k < 1:
        return ()
    return match_tool_sequence(tools)[:prefix_k]


def select_first_tool_mismatch_groups(
    mismatch_keys: Sequence[tuple[int, str, str] | tuple[str, str] | None],
    *,
    max_entries: int = REFLECTION_HIGH_SIGNAL_ENTRY_LIMIT,
) -> tuple[list[int], list[tuple[int, str, str, int]]]:
    """Select low-alignment rows by descending first-disagreement frequency.

    Keys are ``(slot, teacher_tool, student_tool)``. A 2-tuple
    ``(teacher_tool, student_tool)`` is treated as slot 0. The most frequent
    group is always included in full, even when it exceeds ``max_entries``.
    Later whole groups are added while they still fit in the cap; groups that
    would overflow are skipped so later smaller groups can still be included.

    Returns ``(selected_indices, selected_groups)`` where each group is
    ``(slot, teacher_tool, student_tool, taken_count)``.
    """
    if max_entries < 0:
        raise ValueError("max_entries must be non-negative")
    groups: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for index, key in enumerate(mismatch_keys):
        if key is None:
            continue
        if len(key) == 2:
            normalized = (0, key[0], key[1])
        else:
            normalized = (int(key[0]), str(key[1]), str(key[2]))
        groups[normalized].append(index)
    ranked = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), item[0][0], item[0][1], item[0][2]),
    )
    selected: list[int] = []
    selected_groups: list[tuple[int, str, str, int]] = []
    for (slot, teacher_tool, student_tool), indices in ranked:
        if selected and len(selected) + len(indices) > max_entries:
            continue
        selected.extend(indices)
        selected_groups.append((slot, teacher_tool, student_tool, len(indices)))
        if len(selected) >= max_entries:
            break
    return selected, selected_groups


def parse_tool_match_entry_metrics(row: dict[str, Any]) -> ToolMatchEntryMetrics:
    student_tools = scored_tool_sequence(row.get("student_tools"))
    teacher_tools = scored_tool_sequence(row.get("teacher_tools"))
    return ToolMatchEntryMetrics(
        entry_id=str(row.get("entry_id") or ""),
        student_tools=student_tools,
        teacher_tools=teacher_tools,
        tools_match=exact_prefix_match(teacher_tools, student_tools, prefix_k=1),
    )


def restrict_tool_match_analysis(
    analysis: EvalRunToolMatchAnalysis,
    entry_ids: set[str],
) -> EvalRunToolMatchAnalysis:
    """Keep only ``entry_ids`` and recompute aggregate / high-signal fields."""
    per_entry = {entry_id: metrics for entry_id, metrics in analysis.per_entry.items() if entry_id in entry_ids}
    return EvalRunToolMatchAnalysis(
        teacher_eval_id=analysis.teacher_eval_id,
        student_eval_id=analysis.student_eval_id,
        start_date=analysis.start_date,
        end_date=analysis.end_date,
        aggregate=aggregate_tool_match_metrics(analysis.teacher_eval_id, analysis.student_eval_id, per_entry),
        per_entry=per_entry,
        high_signal_entry_ids=tuple(
            sorted(entry_id for entry_id, metrics in per_entry.items() if not metrics.tools_match)
        ),
    )


def intersect_compared_entry_ids(analyses: Sequence[EvalRunToolMatchAnalysis]) -> set[str]:
    """Return entry IDs present in every analysis, or empty when none are given."""
    if not analyses:
        return set()
    shared = set(analyses[0].per_entry)
    for analysis in analyses[1:]:
        shared &= set(analysis.per_entry)
    return shared


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
) -> EvalRunToolMatchAnalysis:
    search_start, search_end = default_date_range(lookback_days=lookback_days, end_date=end_date)
    # The default 1-day lookback is already two UTC shards. A bounds pre-scan
    # reads the same jsonPayload days again and does not shrink the window.
    if lookback_days <= 1:
        start_date, resolved_end = search_start, search_end
    else:
        bounds_rows = client.query(
            build_tool_match_time_bounds_query(agentspan_table=agentspan_table),
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
            return empty_tool_match_analysis(
                teacher_eval_id,
                student_eval_id,
                lookback_days=lookback_days,
                end_date=end_date,
            )
        start_date, resolved_end = date_range
    per_entry_rows = client.query(
        build_tool_match_per_entry_query(agentspan_table=agentspan_table),
        params=[
            QueryParameter("eval_ids", "STRING", [teacher_eval_id, student_eval_id]),
            QueryParameter("student_eval_id", "STRING", student_eval_id),
            QueryParameter("teacher_eval_id", "STRING", teacher_eval_id),
            QueryParameter("start_date", "DATE", start_date.isoformat()),
            QueryParameter("end_date", "DATE", resolved_end.isoformat()),
        ],
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
            sorted(entry_id for entry_id, metrics in per_entry.items() if not metrics.tools_match)
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
        f"{aggregate.tool_match_rate:.2%} first-tool match "
        f"({aggregate.matching_entries}/{aggregate.compared_entries})"
    )
    for entry_id in analysis.high_signal_entry_ids[:5]:
        metrics = analysis.per_entry[entry_id]
        print(
            f"[Tool Match] Mismatch entry={entry_id}: "
            f"student={list(metrics.student_tools[:5])} teacher={list(metrics.teacher_tools[:5])}"
        )
