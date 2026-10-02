"""Citation-match objective: does the student cite the same document set the teacher did?

Layout, top to bottom: the entry and aggregate types, row parsing, the
``citation_spans`` SQL, the paired fetch, then the objective class that maps
the analysis onto the contract in :mod:`glean_gepa.objectives.protocol`. Shared
plumbing (bounds query, shard window, paired FULL OUTER JOIN scaffold, trace
enrichment, frame) comes from ``objectives/utils``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, ClassVar

from glean_gepa.adapter_types import TeacherStudentALTrajectory, paired_rollout_output
from glean_gepa.al_adapter import ReflectiveExample
from glean_gepa.focused_evalset import QUERY_CANONICAL_BUCKET_TYPE
from glean_gepa.objectives.base import AnalysisRequest, ScoredRow, ScoringContext, TeacherStudentObjective
from glean_gepa.objectives.utils.agentspan import (
    Rows,
    bounds_query,
    fetch_agentspan_analysis,
    paired_role_query,
)
from glean_gepa.objectives.utils.agentspan_query import (
    DEFAULT_AGENTS_SPAN_TABLE,
    DEFAULT_LOOKBACK_DAYS,
    EVAL_ENTRY_ID_EXPR,
    default_date_range,
    wildcard_shard_filter,
)
from glean_gepa.objectives.utils.core import (
    EVIDENCE_LIMIT,
    NoComparedEntriesError,
    PairedRunAnalysis,
    log_analysis,
    pass_rate,
)
from glean_gepa.objectives.utils.traces import FetchedByRole, enrich_action_inputs
from glean_gepa.prompt_constants import RULES_EXT_KEY, WRITING_CODE_KEY
from glean_gepa.reflection_prompts import GENERALITY_RULES, RULES_EXT_FRAME, compose_responsibility

CITATION_MATCH_OBJECTIVE = "citation_match"
CITATION_ARRAY_PATH = "jsonPayload.agent_run.citations_data.positioned_citations"
CITATION_ID_FIELD = "doc_id"


# ---------------------------------------------------------------------------
# Slot 1: one entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CitationMatchEntryMetrics:
    entry_id: str
    student_citations: tuple[str, ...]
    teacher_citations: tuple[str, ...]
    citations_match: bool
    student_action_inputs: tuple[str, ...] = ()
    teacher_action_inputs: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.citations_match

    @property
    def score(self) -> float:
        return 1.0 if self.citations_match else 0.0

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(sorted(frozenset(self.teacher_citations) - frozenset(self.student_citations)))

    @property
    def extra(self) -> tuple[str, ...]:
        return tuple(sorted(frozenset(self.student_citations) - frozenset(self.teacher_citations)))


@dataclass(frozen=True)
class CitationMatchMetrics:
    compared_entries: int
    matching_entries: int
    citation_match_rate: float


class EvalRunCitationMatchAnalysis(PairedRunAnalysis[CitationMatchMetrics, CitationMatchEntryMetrics]):
    """Agentspan analyses always carry a resolved shard window."""

    start_date: date
    end_date: date


# ---------------------------------------------------------------------------
# Slot 2: rows -> entry
# ---------------------------------------------------------------------------


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


def parse_citation_match_entry_metrics(row: Mapping[str, Any]) -> CitationMatchEntryMetrics:
    student_citations = scored_citation_ids(row.get("student_citations"))
    teacher_citations = scored_citation_ids(row.get("teacher_citations"))
    return CitationMatchEntryMetrics(
        entry_id=str(row.get("entry_id") or ""),
        student_citations=student_citations,
        teacher_citations=teacher_citations,
        citations_match=frozenset(student_citations) == frozenset(teacher_citations),
    )


def aggregate_citation_match_metrics(per_entry: Mapping[str, CitationMatchEntryMetrics]) -> CitationMatchMetrics:
    return CitationMatchMetrics(
        compared_entries=len(per_entry),
        matching_entries=sum(1 for m in per_entry.values() if m.passed),
        citation_match_rate=pass_rate(per_entry),
    )


# ---------------------------------------------------------------------------
# Slot 3: SQL
# ---------------------------------------------------------------------------


def build_citation_match_per_entry_query(*, agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE) -> str:
    """Pair teacher and student citation-id sets per eval entry.

    Reads ``agent_run`` rather than ``agent_step``: the run-level array is the
    answer's final cited set and is a superset of the per-step arrays, so
    unioning the two adds nothing.
    """
    per_role = f"""
citation_spans AS (
  SELECT
    jsonPayload.context.eval.eval_id AS eval_id,
    {EVAL_ENTRY_ID_EXPR} AS entry_id,
    jsonPayload.context.agent_trace.trace_id AS trace_id,
    resource.labels.project_id AS deployment_id,
    SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64) AS start_ms,
    ARRAY(
      SELECT citation.{CITATION_ID_FIELD}
      FROM UNNEST({CITATION_ARRAY_PATH}) AS citation
      WHERE citation.{CITATION_ID_FIELD} IS NOT NULL AND citation.{CITATION_ID_FIELD} != ''
    ) AS citation_ids
  FROM `{agentspan_table}`
  WHERE {wildcard_shard_filter("start_date", "end_date")}
    AND jsonPayload.context.eval.eval_id IN UNNEST(@eval_ids)
),
per_role_agg AS (
  SELECT
    entry_id,
    eval_id,
    ARRAY_CONCAT_AGG(citation_ids) AS citation_ids,
    ANY_VALUE(trace_id) AS trace_id,
    ANY_VALUE(deployment_id) AS deployment_id,
    MIN(start_ms) AS min_start_ms,
    MAX(start_ms) AS max_start_ms
  FROM citation_spans
  WHERE entry_id IS NOT NULL
  GROUP BY entry_id, eval_id
),
per_role AS (
  SELECT
    entry_id,
    eval_id,
    ARRAY(
      SELECT DISTINCT citation_id
      FROM UNNEST(citation_ids) AS citation_id
      WHERE citation_id IS NOT NULL AND citation_id != ''
      ORDER BY citation_id
    ) AS citations,
    trace_id,
    deployment_id,
    min_start_ms,
    max_start_ms
  FROM per_role_agg
)"""
    return paired_role_query(per_role_cte=per_role, signal_column="citations")


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def empty_citation_match_analysis(
    teacher_eval_id: str,
    student_eval_id: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: date | None = None,
) -> EvalRunCitationMatchAnalysis:
    start_date, resolved_end = default_date_range(lookback_days=lookback_days, end_date=end_date)
    return EvalRunCitationMatchAnalysis(
        eval_ids=(teacher_eval_id, student_eval_id),
        aggregate=aggregate_citation_match_metrics({}),
        start_date=start_date,
        end_date=resolved_end,
    )


def fetch_eval_run_citation_match_analysis(
    client: Any,
    *,
    teacher_eval_id: str,
    student_eval_id: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: date | None = None,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
    evalcli: Any | None = None,
    include_action_inputs: bool = True,
) -> EvalRunCitationMatchAnalysis:
    def parse(row: Mapping[str, Any]) -> CitationMatchEntryMetrics | None:
        metrics = parse_citation_match_entry_metrics(row)
        return metrics if metrics.entry_id else None

    def enrich(
        per_entry: Mapping[str, CitationMatchEntryMetrics], rows: Rows, high_signal: tuple[str, ...]
    ) -> Mapping[str, CitationMatchEntryMetrics]:
        return _enrich_action_inputs(evalcli, per_entry, rows, high_signal)

    analysis = fetch_agentspan_analysis(
        client,
        eval_ids=(teacher_eval_id, student_eval_id),
        bounds_sql=bounds_query(eval_id_predicate="IN UNNEST(@eval_ids)", agentspan_table=agentspan_table),
        per_entry_sql=build_citation_match_per_entry_query(agentspan_table=agentspan_table),
        parse_row=parse,
        aggregate=lambda per_entry, _dropped: aggregate_citation_match_metrics(per_entry),
        is_high_signal=lambda m: not m.citations_match,
        enrich=enrich if evalcli is not None and include_action_inputs else None,
        lookback_days=lookback_days,
        end_date=end_date,
    )
    if analysis.start_date is None:
        return empty_citation_match_analysis(
            teacher_eval_id, student_eval_id, lookback_days=lookback_days, end_date=end_date
        )
    return EvalRunCitationMatchAnalysis(
        eval_ids=analysis.eval_ids,
        aggregate=analysis.aggregate,
        per_entry=analysis.per_entry,
        high_signal_entry_ids=analysis.high_signal_entry_ids,
        start_date=analysis.start_date,
        end_date=analysis.end_date or analysis.start_date,
    )


def _enrich_action_inputs(
    evalcli: Any,
    per_entry: Mapping[str, CitationMatchEntryMetrics],
    rows: Rows,
    high_signal_entry_ids: tuple[str, ...],
) -> dict[str, CitationMatchEntryMetrics]:
    """Attach teacher and student tool payloads to high-signal entries from traces."""

    def apply(metrics: CitationMatchEntryMetrics, fetched: FetchedByRole, entry_id: str) -> CitationMatchEntryMetrics:
        student = fetched.get("student", {}).get(entry_id, metrics.student_action_inputs)
        teacher = fetched.get("teacher", {}).get(entry_id, metrics.teacher_action_inputs)
        return replace(metrics, student_action_inputs=student, teacher_action_inputs=teacher)

    return enrich_action_inputs(
        evalcli, per_entry, rows, high_signal_entry_ids, apply=apply, roles=("student", "teacher")
    )


def _citation_group_sig(ids: Sequence[str], *, prefix: str) -> str:
    if not ids:
        return f"{prefix}:(none)"
    shown = ",".join(ids[:8])
    overflow = f"+{len(ids) - 8}" if len(ids) > 8 else ""
    return f"{prefix}:{shown}{overflow}"


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

WRITING_CODE_RESPONSIBILITY = (
    "Focus ONLY on the coding and execution instructions that decide which sources reach the answer: "
    "printing raw SDK results, carrying each result's `citationId` through any filtering, ranking, or "
    "summarizing step, and not truncating output the answer still has to cite. Change other guidance "
    "only where it produces missing or extra citations. "
    f"{GENERALITY_RULES} Propose minimal deltas."
)

RULES_EXT_RESPONSIBILITY = compose_responsibility(
    RULES_EXT_FRAME,
    "Target citation mismatches (missing teacher sources, extra student sources, or dropped "
    "citationId values after filtering SDK results). Keep each bullet operational and concise.",
)


def _rollout_output(
    *,
    entry_id: str,
    deployment_id: str,
    query: str,
    student_citations: list[str],
    teacher_citations: list[str],
    student_action_inputs: list[str] | None = None,
    teacher_action_inputs: list[str] | None = None,
):
    """One rollout row. Citation IDs ride on optional output fields plus answers."""
    output = paired_rollout_output(
        deployment_id=deployment_id,
        query=query,
        entry_id=entry_id,
        student_answer="citations=" + (", ".join(student_citations) if student_citations else "(none)"),
        teacher_answer="citations=" + (", ".join(teacher_citations) if teacher_citations else "(none)"),
    )
    output["student_citations"] = student_citations
    output["teacher_citations"] = teacher_citations
    if student_action_inputs:
        output["student_action_inputs"] = list(student_action_inputs)
    if teacher_action_inputs:
        output["teacher_action_inputs"] = list(teacher_action_inputs)
    return output


class CitationMatchObjective(TeacherStudentObjective[EvalRunCitationMatchAnalysis]):
    """Score the student's cited source set against the teacher's."""

    name = CITATION_MATCH_OBJECTIVE
    telemetry_dimensions = (CITATION_MATCH_OBJECTIVE,)
    focused_bucket_type = QUERY_CANONICAL_BUCKET_TYPE
    failure_label = "HIGH-SIGNAL FAILURES (teacher vs student citation match)"
    reflection_report_title = "REFLECTION: teacher vs student citation sets"
    teacher_compared_key = "teacher_citations"
    student_compared_key = "student_citations"
    mismatch_pair = citation_mismatch_pair
    module_responsibilities: ClassVar[Mapping[str, str]] = {
        WRITING_CODE_KEY: WRITING_CODE_RESPONSIBILITY,
        RULES_EXT_KEY: RULES_EXT_RESPONSIBILITY,
    }

    def __init__(self, *, bigquery_client: Any | None = None, lookback_days: int = 1):
        self.bigquery_client = bigquery_client
        self.lookback_days = lookback_days
        self.params: dict[str, Any] = {}
        self._paired_analysis_cache: dict[tuple[str, str], EvalRunCitationMatchAnalysis] = {}

    def analyze(
        self, teacher_eval_id: str, student_eval_id: str, *, request: AnalysisRequest
    ) -> EvalRunCitationMatchAnalysis:
        return self.cached_paired_analysis(
            teacher_eval_id,
            student_eval_id,
            request=request,
            cache=self._paired_analysis_cache,
            fetch=fetch_eval_run_citation_match_analysis,
            empty=empty_citation_match_analysis,
            label="citation match analysis",
        )

    def require_compared_entries(self, analysis: EvalRunCitationMatchAnalysis) -> None:
        """Reject an analysis with zero compared entries."""
        if analysis.aggregate.compared_entries > 0:
            return
        raise NoComparedEntriesError(
            f"No eval entries were compared for student {analysis.student_eval_id} vs "
            f"teacher {analysis.teacher_eval_id}. Wait for agentspan ingest or "
            f"check that the eval runs actually emitted citation payloads."
        )

    def validate_full_eval(self, analysis: EvalRunCitationMatchAnalysis) -> None:
        aggregate = analysis.aggregate
        log_analysis(
            analysis,
            label="Citation Match",
            headline=(
                f"vs {analysis.teacher_eval_id}: {aggregate.citation_match_rate:.2%} citation-set match "
                f"({aggregate.matching_entries}/{aggregate.compared_entries})"
            ),
            entry_line=lambda m: (
                f"student={list(m.student_citations[:EVIDENCE_LIMIT])} teacher={list(m.teacher_citations[:EVIDENCE_LIMIT])}"
            ),
        )

    def focused_pass_rate(self, analysis: EvalRunCitationMatchAnalysis, requested_entry_ids: Sequence[str]) -> float:
        matching = sum(1 for metrics in analysis.per_entry.values() if metrics.citations_match)
        return matching / len(requested_entry_ids)

    def aggregate_row(self, analysis: EvalRunCitationMatchAnalysis, ctx: ScoringContext) -> ScoredRow:
        return ScoredRow(
            entry_id=None,
            dimension_scores={CITATION_MATCH_OBJECTIVE: analysis.aggregate.citation_match_rate},
            output=_rollout_output(
                entry_id=ctx.query,
                deployment_id=ctx.deployment_id,
                query=ctx.query,
                student_citations=[],
                teacher_citations=[],
            ),
        )

    def entry_row(
        self,
        entry_id: str,
        metrics: CitationMatchEntryMetrics,
        analysis: EvalRunCitationMatchAnalysis,
        ctx: ScoringContext,
    ) -> ScoredRow:
        del analysis
        return ScoredRow(
            entry_id=entry_id,
            dimension_scores={CITATION_MATCH_OBJECTIVE: float(metrics.citations_match)},
            output=_rollout_output(
                entry_id=entry_id,
                deployment_id=ctx.deployment_id,
                query=ctx.query,
                student_citations=list(metrics.student_citations),
                teacher_citations=list(metrics.teacher_citations),
                student_action_inputs=list(metrics.student_action_inputs),
                teacher_action_inputs=list(metrics.teacher_action_inputs),
            ),
        )

    def failure_pattern(self, component_name: str, trajectory: TeacherStudentALTrajectory) -> tuple[Any, ...]:
        del component_name
        output = trajectory["output"]
        citation_match = trajectory.get("objective_scores", {}).get(self.name, 1.0)
        return (
            int(citation_match < float(self.experiment_param("failure_score_below", 1.0))),
            int(self._mismatch_key(output) is not None),
        )

    def build_reflective_example(
        self,
        component_name: str,
        trajectory: TeacherStudentALTrajectory,
        candidate: dict[str, str],
    ) -> ReflectiveExample:
        del component_name, candidate
        output = trajectory["output"]
        objective_scores = trajectory.get("objective_scores", {})
        citation_match = objective_scores.get(self.name, trajectory["score"])
        student_citations = list(output.get("student_citations") or [])
        teacher_citations = list(output.get("teacher_citations") or [])
        mismatch = self._mismatch_key(output)
        feedback_parts = []
        if mismatch is not None:
            missing, extra = mismatch
            feedback_parts.append(f"Citation-set mismatch: {missing}; {extra}.")
        if citation_match < 1.0:
            feedback_parts.append(f"Citation match issue: score={citation_match:.2f}.")
        feedback_parts.extend(self.wired_signal_issues(objective_scores))
        return self.reflective_example(
            trajectory,
            feedback=" ".join(feedback_parts) if feedback_parts else "General teacher/student citation divergence.",
            generated={
                "student_answer": output.get("student_answer", ""),
                "teacher_answer": output.get("teacher_answer", ""),
                "student_tools": student_citations,
                "teacher_tools": teacher_citations,
                "student_citations": student_citations,
                "teacher_citations": teacher_citations,
            },
            # The raw user query is scrubbed, so surface the teacher's tool payloads
            # (the searches it ran) as the intent signal behind the cited sources.
            action_inputs=output.get("teacher_action_inputs") or [],
        )


__all__ = ["CitationMatchObjective"]
