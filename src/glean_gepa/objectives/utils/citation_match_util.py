"""Teacher-vs-student citation-set metrics derived from Glean evaluation spans."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from glean_gepa.objectives.utils.agentspan_query import (
    DEFAULT_AGENTS_SPAN_TABLE,
    DEFAULT_LOOKBACK_DAYS,
    QueryParameter,
    default_date_range,
    run_windowed_per_entry_query,
    wildcard_shard_filter,
)

CITATION_MATCH_OBJECTIVE = "citation_match"
# Pull citation IDs out of whatever JSON path the span uses. Confirm against a
# real agentspan row if this over- or under-matches.
_CITATION_ID_JSON_REGEX = r'"(?:citationId|citation_id)"\\s*:\\s*"([^"]+)"'


class NoComparedCitationEntriesError(RuntimeError):
    """Raised when teacher/student evals produced no comparable citation rows."""


@dataclass(frozen=True)
class CitationMatchEntryMetrics:
    entry_id: str
    student_citations: tuple[str, ...]
    teacher_citations: tuple[str, ...]
    citations_match: bool

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(sorted(frozenset(self.teacher_citations) - frozenset(self.student_citations)))

    @property
    def extra(self) -> tuple[str, ...]:
        return tuple(sorted(frozenset(self.student_citations) - frozenset(self.teacher_citations)))


@dataclass(frozen=True)
class CitationMatchMetrics:
    teacher_eval_id: str
    student_eval_id: str
    compared_entries: int
    matching_entries: int
    citation_match_rate: float


@dataclass(frozen=True)
class EvalRunCitationMatchAnalysis:
    teacher_eval_id: str
    student_eval_id: str
    start_date: date
    end_date: date
    aggregate: CitationMatchMetrics
    per_entry: dict[str, CitationMatchEntryMetrics]
    high_signal_entry_ids: tuple[str, ...]


def scored_citation_ids(citations: Sequence[str] | None) -> tuple[str, ...]:
    """Return unique citation IDs in first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in citations or []:
        citation_id = str(raw).strip()
        if not citation_id or citation_id in seen:
            continue
        seen.add(citation_id)
        ordered.append(citation_id)
    return tuple(ordered)


def citation_mismatch_pair(
    teacher_citations: Sequence[str] | None,
    student_citations: Sequence[str] | None,
) -> tuple[str, str] | None:
    """Return ``(missing_sig, extra_sig)`` when citation sets differ, else ``None``."""
    teacher = frozenset(scored_citation_ids(teacher_citations))
    student = frozenset(scored_citation_ids(student_citations))
    if teacher == student:
        return None
    return (
        _citation_group_sig(sorted(teacher - student), prefix="missing"),
        _citation_group_sig(sorted(student - teacher), prefix="extra"),
    )


def parse_citation_match_entry_metrics(row: dict[str, Any]) -> CitationMatchEntryMetrics:
    student_citations = scored_citation_ids(row.get("student_citations"))
    teacher_citations = scored_citation_ids(row.get("teacher_citations"))
    return CitationMatchEntryMetrics(
        entry_id=str(row.get("entry_id") or ""),
        student_citations=student_citations,
        teacher_citations=teacher_citations,
        citations_match=frozenset(student_citations) == frozenset(teacher_citations),
    )


def aggregate_citation_match_metrics(
    teacher_eval_id: str,
    student_eval_id: str,
    per_entry: dict[str, CitationMatchEntryMetrics],
) -> CitationMatchMetrics:
    compared = len(per_entry)
    matching = sum(1 for metrics in per_entry.values() if metrics.citations_match)
    return CitationMatchMetrics(
        teacher_eval_id=teacher_eval_id,
        student_eval_id=student_eval_id,
        compared_entries=compared,
        matching_entries=matching,
        citation_match_rate=(matching / compared) if compared else 0.0,
    )


def empty_citation_match_analysis(
    teacher_eval_id: str,
    student_eval_id: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: date | None = None,
) -> EvalRunCitationMatchAnalysis:
    start_date, resolved_end = default_date_range(lookback_days=lookback_days, end_date=end_date)
    return EvalRunCitationMatchAnalysis(
        teacher_eval_id=teacher_eval_id,
        student_eval_id=student_eval_id,
        start_date=start_date,
        end_date=resolved_end,
        aggregate=aggregate_citation_match_metrics(teacher_eval_id, student_eval_id, {}),
        per_entry={},
        high_signal_entry_ids=(),
    )


def build_citation_match_time_bounds_query(
    *,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> str:
    """Find min/max span timestamps for a teacher/student eval pair."""
    return f"""
SELECT
  MIN(SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64)) AS min_start_ms,
  MAX(SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64)) AS max_start_ms
FROM `{agentspan_table}`
WHERE {wildcard_shard_filter("search_start_date", "search_end_date")}
  AND jsonPayload.context.eval.eval_id IN UNNEST(@eval_ids)
""".strip()


def build_citation_match_per_entry_query(
    *,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> str:
    """Pair teacher and student citation-id sets per eval entry.

    Citation IDs are scraped from the span JSON rather than a single nested
    field, so this still works if Cito/MCP stores them under different paths.
    Swap the ``REGEXP_EXTRACT_ALL`` expression if a structured array is better.
    """
    return f"""
WITH citation_spans AS (
  SELECT
    jsonPayload.context.eval.eval_id AS eval_id,
    COALESCE(
      jsonPayload.context.eval.entry_uuid,
      CAST(jsonPayload.context.eval.entry_id AS STRING)
    ) AS entry_id,
    REGEXP_EXTRACT_ALL(TO_JSON_STRING(jsonPayload), r'{_CITATION_ID_JSON_REGEX}') AS citation_ids
  FROM `{agentspan_table}`
  WHERE {wildcard_shard_filter("start_date", "end_date")}
    AND jsonPayload.context.eval.eval_id IN UNNEST(@eval_ids)
),
per_role AS (
  SELECT
    entry_id,
    eval_id,
    ARRAY(
      SELECT DISTINCT citation_id
      FROM UNNEST(ARRAY_CONCAT_AGG(citation_ids)) AS citation_id
      WHERE citation_id IS NOT NULL AND citation_id != ''
      ORDER BY citation_id
    ) AS citations
  FROM citation_spans
  WHERE entry_id IS NOT NULL
  GROUP BY entry_id, eval_id
),
student AS (
  SELECT entry_id, citations FROM per_role WHERE eval_id = @student_eval_id
),
teacher AS (
  SELECT entry_id, citations FROM per_role WHERE eval_id = @teacher_eval_id
)
SELECT
  COALESCE(student.entry_id, teacher.entry_id) AS entry_id,
  IFNULL(student.citations, ARRAY<STRING>[]) AS student_citations,
  IFNULL(teacher.citations, ARRAY<STRING>[]) AS teacher_citations
FROM student
FULL OUTER JOIN teacher
  ON student.entry_id = teacher.entry_id
ORDER BY entry_id
""".strip()


def fetch_eval_run_citation_match_analysis(
    client: Any,
    *,
    teacher_eval_id: str,
    student_eval_id: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: date | None = None,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> EvalRunCitationMatchAnalysis:
    eval_ids = [teacher_eval_id, student_eval_id]
    result = run_windowed_per_entry_query(
        client,
        bounds_query=build_citation_match_time_bounds_query(agentspan_table=agentspan_table),
        per_entry_query=build_citation_match_per_entry_query(agentspan_table=agentspan_table),
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
        return empty_citation_match_analysis(
            teacher_eval_id,
            student_eval_id,
            lookback_days=lookback_days,
            end_date=end_date,
        )

    start_date, resolved_end, per_entry_rows = result
    per_entry = {
        metrics.entry_id: metrics
        for row in per_entry_rows
        for metrics in [parse_citation_match_entry_metrics(row)]
        if metrics.entry_id
    }
    return EvalRunCitationMatchAnalysis(
        teacher_eval_id=teacher_eval_id,
        student_eval_id=student_eval_id,
        start_date=start_date,
        end_date=resolved_end,
        aggregate=aggregate_citation_match_metrics(teacher_eval_id, student_eval_id, per_entry),
        per_entry=per_entry,
        high_signal_entry_ids=tuple(
            sorted(entry_id for entry_id, metrics in per_entry.items() if not metrics.citations_match)
        ),
    )


def require_compared_citation_entries(analysis: EvalRunCitationMatchAnalysis) -> None:
    """Reject an analysis with zero compared entries."""
    if analysis.aggregate.compared_entries > 0:
        return
    raise NoComparedCitationEntriesError(
        f"No eval entries were compared for student {analysis.student_eval_id} vs "
        f"teacher {analysis.teacher_eval_id}. Wait for agentspan ingest or "
        f"check that the eval runs actually emitted citation payloads."
    )


def log_citation_match_analysis(analysis: EvalRunCitationMatchAnalysis) -> None:
    aggregate = analysis.aggregate
    print(
        f"[Citation Match] {analysis.student_eval_id} vs {analysis.teacher_eval_id}: "
        f"{aggregate.citation_match_rate:.2%} citation-set match "
        f"({aggregate.matching_entries}/{aggregate.compared_entries})"
    )
    for entry_id in analysis.high_signal_entry_ids[:5]:
        metrics = analysis.per_entry[entry_id]
        print(
            f"[Citation Match] Mismatch entry={entry_id}: "
            f"student={list(metrics.student_citations[:5])} "
            f"teacher={list(metrics.teacher_citations[:5])}"
        )


def _citation_group_sig(ids: Sequence[str], *, prefix: str) -> str:
    if not ids:
        return f"{prefix}:(none)"
    shown = ",".join(ids[:8])
    overflow = f"+{len(ids) - 8}" if len(ids) > 8 else ""
    return f"{prefix}:{shown}{overflow}"
