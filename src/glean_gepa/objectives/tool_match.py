"""First-tool-match objective: does the student pick the same first tool the teacher did?

Layout, top to bottom: the entry and aggregate types, row parsing, the
``tool_spans`` SQL (with the failed-runs prelude that drops entries whose
teacher or student run errored), the paired fetch, then the objective class
that maps the analysis onto the contract in :mod:`glean_gepa.objectives.protocol`.
Shared plumbing (bounds query, shard window, paired FULL OUTER JOIN scaffold,
trace enrichment, frame) comes from ``objectives/utils``. Pure tool-name helpers
are in ``objectives/utils/tool_names`` so ``prompt`` can import them too."""

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
    AGENT_RUN_FAILURE_FILTER,
    DEFAULT_AGENTS_SPAN_TABLE,
    DEFAULT_LOOKBACK_DAYS,
    EVAL_ENTRY_ID_EXPR,
    EXECUTE_ACTION_FILTER,
    QueryParameter,
    default_date_range,
    wildcard_shard_filter,
)
from glean_gepa.objectives.utils.core import (
    EVIDENCE_LIMIT,
    NoComparedEntriesError,
    PairedRunAnalysis,
    log_analysis,
)
from glean_gepa.objectives.utils.mismatch import select_mismatch_groups
from glean_gepa.objectives.utils.tool_names import (
    SKIPPED_TOOL_NAMES,
    first_tool_mismatch_pair,
    scored_tool_sequence,
)
from glean_gepa.objectives.utils.traces import FetchedByRole, enrich_action_inputs
from glean_gepa.prompt import high_signal_core_tool_keys, is_core_tool_span, tool_description_override_key
from glean_gepa.prompt_constants import CORE_TOOL_KEYS, EXECUTION_DISCIPLINE_KEY, RULES_EXT_KEY
from glean_gepa.reflection_prompts import NO_EXAMPLE_SPECIFICS_RULE, TEACHER_IS_OFFLINE_RULE

TOOL_ALIGNMENT_OBJECTIVE = "tool_alignment"


# ---------------------------------------------------------------------------
# Slot 1: one entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolMatchEntryMetrics:
    """One entry's first-tool comparison.

    This objective scores only the first tool each role picked, so the payloads are
    likewise the first call's -- surfacing later calls would invite reflection to
    rewrite a prompt over a decision that was never scored.
    """

    entry_id: str
    student_tools: tuple[str, ...]
    teacher_tools: tuple[str, ...]
    tools_match: bool
    student_first_tool_input: tuple[str, str] | None = None
    teacher_first_tool_input: tuple[str, str] | None = None

    @property
    def passed(self) -> bool:
        return self.tools_match

    @property
    def score(self) -> float:
        return 1.0 if self.tools_match else 0.0


@dataclass(frozen=True)
class ToolMatchMetrics:
    teacher_eval_id: str
    student_eval_id: str
    compared_entries: int
    matching_entries: int
    tool_match_rate: float
    excluded_failed_runs: int = 0


class EvalRunToolMatchAnalysis(PairedRunAnalysis[ToolMatchMetrics, ToolMatchEntryMetrics]):
    """Agentspan analyses always carry a resolved shard window."""

    start_date: date
    end_date: date


# ---------------------------------------------------------------------------
# Slot 2: rows -> entry
# ---------------------------------------------------------------------------


def entry_run_failed(row: Mapping[str, Any]) -> bool:
    """Whether either role's run died on this entry, leaving no comparable trajectory."""
    return bool(row.get("run_failed"))


def parse_tool_match_entry_metrics(
    row: Mapping[str, Any], skip_tools: frozenset[str] | None = None
) -> ToolMatchEntryMetrics:
    student_tools = scored_tool_sequence(row.get("student_tools"), skip_tools=skip_tools)
    teacher_tools = scored_tool_sequence(row.get("teacher_tools"), skip_tools=skip_tools)
    return ToolMatchEntryMetrics(
        entry_id=str(row.get("entry_id") or ""),
        student_tools=student_tools,
        teacher_tools=teacher_tools,
        tools_match=(student_tools[:1] == teacher_tools[:1]),
    )


def aggregate_tool_match_metrics(
    teacher_eval_id: str,
    student_eval_id: str,
    per_entry: Mapping[str, ToolMatchEntryMetrics],
    *,
    excluded_failed_runs: int = 0,
) -> ToolMatchMetrics:
    compared = len(per_entry)
    matching = sum(1 for metrics in per_entry.values() if metrics.tools_match)
    return ToolMatchMetrics(
        teacher_eval_id=teacher_eval_id,
        student_eval_id=student_eval_id,
        compared_entries=compared,
        matching_entries=matching,
        tool_match_rate=(matching / compared) if compared else 0.0,
        excluded_failed_runs=excluded_failed_runs,
    )


# ---------------------------------------------------------------------------
# Slot 3: SQL
# ---------------------------------------------------------------------------


def build_tool_match_per_entry_query(*, agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE) -> str:
    """Build SQL that pairs teacher and student tool sequences per eval entry."""
    shard = wildcard_shard_filter("start_date", "end_date")
    failed_runs = f"""
failed_runs AS (
  SELECT DISTINCT
    {EVAL_ENTRY_ID_EXPR} AS entry_id
  FROM `{agentspan_table}`
  WHERE {shard}
    AND jsonPayload.context.eval.eval_id IN UNNEST(@eval_ids)
    AND {AGENT_RUN_FAILURE_FILTER}
    -- @eval_ids is exactly the teacher/student pair, so any hit means one of the two
    -- roles died on this entry. A NULL here would make the IN below return NULL.
    AND {EVAL_ENTRY_ID_EXPR} IS NOT NULL
)"""
    per_role = f"""
tool_spans AS (
  SELECT
    jsonPayload.context.eval.eval_id AS eval_id,
    {EVAL_ENTRY_ID_EXPR} AS entry_id,
    REGEXP_REPLACE(jsonPayload.span_info.span_name, r'^Execute Action: ', '') AS tool_name,
    jsonPayload.context.agent_trace.trace_id AS trace_id,
    resource.labels.project_id AS deployment_id,
    SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64) AS start_ms
  FROM `{agentspan_table}`
  WHERE {shard}
    AND jsonPayload.context.eval.eval_id IN UNNEST(@eval_ids)
    AND {EXECUTE_ACTION_FILTER}
    AND REGEXP_REPLACE(jsonPayload.span_info.span_name, r'^Execute Action: ', '') NOT IN UNNEST(@skipped_tools)
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
)"""
    return paired_role_query(
        per_role_cte=per_role,
        signal_column="tools",
        prelude_ctes=failed_runs,
        extra_select="  COALESCE(student.entry_id, teacher.entry_id) IN (SELECT entry_id FROM failed_runs) AS run_failed",
    )


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def empty_tool_match_analysis(
    teacher_eval_id: str,
    student_eval_id: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: date | None = None,
) -> EvalRunToolMatchAnalysis:
    start_date, resolved_end = default_date_range(lookback_days=lookback_days, end_date=end_date)
    return EvalRunToolMatchAnalysis(
        eval_ids=(teacher_eval_id, student_eval_id),
        aggregate=aggregate_tool_match_metrics(teacher_eval_id, student_eval_id, {}),
        start_date=start_date,
        end_date=resolved_end,
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
    include_action_inputs: bool = True,
    skip_tools: frozenset[str] | None = None,
) -> EvalRunToolMatchAnalysis:
    skipped = SKIPPED_TOOL_NAMES if skip_tools is None else skip_tools

    def parse(row: Mapping[str, Any]) -> ToolMatchEntryMetrics | None:
        metrics = parse_tool_match_entry_metrics(row, skip_tools=skip_tools)
        return metrics if metrics.entry_id else None

    def aggregate(
        ids: tuple[str, ...], per_entry: Mapping[str, ToolMatchEntryMetrics], dropped: int
    ) -> ToolMatchMetrics:
        return aggregate_tool_match_metrics(ids[0], ids[-1], per_entry, excluded_failed_runs=dropped)

    def enrich(
        per_entry: Mapping[str, ToolMatchEntryMetrics], rows: Rows, high_signal: tuple[str, ...]
    ) -> Mapping[str, ToolMatchEntryMetrics]:
        return _enrich_action_inputs(evalcli, per_entry, rows, high_signal, skip_tools=skipped)

    analysis = fetch_agentspan_analysis(
        client,
        eval_ids=(teacher_eval_id, student_eval_id),
        bounds_sql=bounds_query(
            eval_id_predicate="IN UNNEST(@eval_ids)", span_filter=EXECUTE_ACTION_FILTER, agentspan_table=agentspan_table
        ),
        per_entry_sql=build_tool_match_per_entry_query(agentspan_table=agentspan_table),
        parse_row=parse,
        aggregate=aggregate,
        is_high_signal=lambda m: not m.tools_match,
        filter_rows=lambda rows: [row for row in rows if not entry_run_failed(row)],
        enrich=enrich if evalcli is not None and include_action_inputs else None,
        extra_params=[QueryParameter("skipped_tools", "STRING", list(skipped))],
        lookback_days=lookback_days,
        end_date=end_date,
    )
    if analysis.start_date is None:
        return empty_tool_match_analysis(
            teacher_eval_id, student_eval_id, lookback_days=lookback_days, end_date=end_date
        )
    return EvalRunToolMatchAnalysis(
        eval_ids=analysis.eval_ids,
        aggregate=analysis.aggregate,
        per_entry=analysis.per_entry,
        high_signal_entry_ids=analysis.high_signal_entry_ids,
        start_date=analysis.start_date,
        end_date=analysis.end_date or analysis.start_date,
    )


def _enrich_action_inputs(
    evalcli: Any,
    per_entry: Mapping[str, ToolMatchEntryMetrics],
    rows: Rows,
    high_signal_entry_ids: tuple[str, ...],
    *,
    skip_tools: frozenset[str],
) -> dict[str, ToolMatchEntryMetrics]:
    """Attach each role's first tool call to high-signal entries from traces."""

    def apply(metrics: ToolMatchEntryMetrics, fetched: FetchedByRole, entry_id: str) -> ToolMatchEntryMetrics:
        student = fetched.get("student", {}).get(entry_id, metrics.student_first_tool_input)
        teacher = fetched.get("teacher", {}).get(entry_id, metrics.teacher_first_tool_input)
        return replace(metrics, student_first_tool_input=student, teacher_first_tool_input=teacher)

    return enrich_action_inputs(
        evalcli,
        per_entry,
        rows,
        high_signal_entry_ids,
        apply=apply,
        roles=("student", "teacher"),
        first_tool_only=True,
        skip_tools=skip_tools,
    )


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

EXECUTION_DISCIPLINE_RESPONSIBILITY = (
    "You are rewriting the bullets under '### Execution Discipline', which set how much effort "
    "the assistant spends before answering: how many tool loops to use, how many queries to "
    "issue, whether to retry after an empty result, and when to stop searching and respond. "
    "Each line must start with '- '. Do not add a heading. This module governs effort and "
    "stopping conditions only: leave response formatting, citation mechanics, and shell or SDK "
    "syntax to other modules. Prefer stating the condition under which more work is warranted "
    "over raising a numeric cap, so the rule generalizes to requests of different sizes. "
    f"{NO_EXAMPLE_SPECIFICS_RULE} {TEACHER_IS_OFFLINE_RULE}"
)

RULES_EXT_RESPONSIBILITY = (
    "You are writing at most two markdown bullets that will be appended after the existing "
    "**Rules:** list in Writing Code. Each line must start with '- '. Do not repeat those "
    "existing Rules, do not add a heading, and do not exceed two bullets. Target first-tool "
    "mismatches whose tools are not core tools (for example Write vs (none)). Keep each "
    f"bullet operational and concise. {NO_EXAMPLE_SPECIFICS_RULE} {TEACHER_IS_OFFLINE_RULE}"
)


def _rollout_output(
    *,
    entry_id: str,
    deployment_id: str,
    query: str,
    student_tools: list[str],
    teacher_tools: list[str],
    student_tool_calls: int | None = None,
    teacher_tool_calls: int | None = None,
    student_first_tool_input: tuple[str, str] | None = None,
    teacher_first_tool_input: tuple[str, str] | None = None,
):
    """One rollout row. Tool-call counts default to the listed events."""
    output = paired_rollout_output(
        deployment_id=deployment_id,
        query=query,
        entry_id=entry_id,
        student_tool_events=student_tools,
        teacher_tool_events=teacher_tools,
        student_tool_calls=student_tool_calls,
        teacher_tool_calls=teacher_tool_calls,
    )
    if student_first_tool_input:
        output["student_first_tool_input"] = list(student_first_tool_input)
    if teacher_first_tool_input:
        output["teacher_first_tool_input"] = list(teacher_first_tool_input)
    return output


class FirstToolMatchObjective(TeacherStudentObjective[EvalRunToolMatchAnalysis]):
    """Score the student's first tool call against the teacher's."""

    name = TOOL_ALIGNMENT_OBJECTIVE
    telemetry_dimensions = (TOOL_ALIGNMENT_OBJECTIVE,)
    focused_bucket_type = QUERY_CANONICAL_BUCKET_TYPE
    failure_label = "HIGH-SIGNAL FAILURES (teacher vs student tool match)"
    reflection_report_title = "REFLECTION: teacher vs student tool sequences"
    teacher_compared_key = "teacher_tool_events"
    student_compared_key = "student_tool_events"
    mismatch_pair = first_tool_mismatch_pair
    module_responsibilities: ClassVar[Mapping[str, str]] = {
        RULES_EXT_KEY: RULES_EXT_RESPONSIBILITY,
        EXECUTION_DISCIPLINE_KEY: EXECUTION_DISCIPLINE_RESPONSIBILITY,
    }

    def __init__(self, *, bigquery_client: Any | None = None, lookback_days: int = 1):
        self.bigquery_client = bigquery_client
        self.lookback_days = lookback_days
        self.params: dict[str, Any] = {}
        self._paired_analysis_cache: dict[tuple[str, str], EvalRunToolMatchAnalysis] = {}

    def _skipped_tools(self) -> frozenset[str]:
        raw = self.pack_param("skipped_tools", None)
        if raw is None:
            return SKIPPED_TOOL_NAMES
        return frozenset(str(name) for name in raw)

    def _mismatch_key(self, output: Mapping[str, Any]) -> tuple[str, str] | None:
        return first_tool_mismatch_pair(
            output.get(self.teacher_compared_key),
            output.get(self.student_compared_key),
            skip_tools=self._skipped_tools(),
        )

    def analyze(
        self, teacher_eval_id: str, student_eval_id: str, *, request: AnalysisRequest
    ) -> EvalRunToolMatchAnalysis:
        skipped = self._skipped_tools()

        def fetch(client: Any, **kwargs: Any) -> EvalRunToolMatchAnalysis:
            return fetch_eval_run_tool_match_analysis(client, skip_tools=skipped, **kwargs)

        return self.cached_paired_analysis(
            teacher_eval_id,
            student_eval_id,
            request=request,
            cache=self._paired_analysis_cache,
            fetch=fetch,
            empty=empty_tool_match_analysis,
            label="tool match analysis",
        )

    def require_compared_entries(self, analysis: EvalRunToolMatchAnalysis) -> None:
        """Reject an analysis with zero compared entries.

        A 0/0 comparison is not a 100% match: it means neither eval produced
        comparable Execute Action spans, so tool alignment is undefined.
        """
        if analysis.aggregate.compared_entries > 0:
            return
        excluded = analysis.aggregate.excluded_failed_runs
        reason = (
            f"all {excluded} candidate entries were dropped because a teacher or student run failed"
            if excluded
            else "wait for agentspan ingest or check that the eval runs actually executed entries"
        )
        raise NoComparedEntriesError(
            f"No eval entries were compared for student {analysis.student_eval_id} vs "
            f"teacher {analysis.teacher_eval_id}: {reason}."
        )

    def validate_full_eval(self, analysis: EvalRunToolMatchAnalysis) -> None:
        aggregate = analysis.aggregate
        if aggregate.excluded_failed_runs:
            print(
                f"[Tool Match] Excluded {aggregate.excluded_failed_runs} entries whose teacher or "
                "student run failed; check per-deployment error rates if this is a large share"
            )
        log_analysis(
            analysis,
            label="Tool Match",
            headline=(
                f"vs {analysis.teacher_eval_id}: {aggregate.tool_match_rate:.2%} first-tool match "
                f"({aggregate.matching_entries}/{aggregate.compared_entries})"
            ),
            entry_line=lambda m: (
                f"student={list(m.student_tools[:EVIDENCE_LIMIT])} teacher={list(m.teacher_tools[:EVIDENCE_LIMIT])}"
            ),
        )

    def focused_pass_rate(self, analysis: EvalRunToolMatchAnalysis, requested_entry_ids: Sequence[str]) -> float:
        matching = sum(1 for metrics in analysis.per_entry.values() if metrics.tools_match)
        return matching / len(requested_entry_ids)

    def aggregate_row(self, analysis: EvalRunToolMatchAnalysis, ctx: ScoringContext) -> ScoredRow:
        return ScoredRow(
            entry_id=None,
            dimension_scores={TOOL_ALIGNMENT_OBJECTIVE: analysis.aggregate.tool_match_rate},
            output=_rollout_output(
                entry_id=ctx.query,
                deployment_id=ctx.deployment_id,
                query=ctx.query,
                student_tools=[],
                teacher_tools=[],
                student_tool_calls=sum(len(m.student_tools) for m in analysis.per_entry.values()),
                teacher_tool_calls=sum(len(m.teacher_tools) for m in analysis.per_entry.values()),
            ),
        )

    def entry_row(
        self, entry_id: str, metrics: ToolMatchEntryMetrics, analysis: EvalRunToolMatchAnalysis, ctx: ScoringContext
    ) -> ScoredRow:
        del analysis
        return ScoredRow(
            entry_id=entry_id,
            dimension_scores={TOOL_ALIGNMENT_OBJECTIVE: float(metrics.tools_match)},
            output=_rollout_output(
                entry_id=entry_id,
                deployment_id=ctx.deployment_id,
                query=ctx.query,
                student_tools=list(metrics.student_tools),
                teacher_tools=list(metrics.teacher_tools),
                student_first_tool_input=metrics.student_first_tool_input,
                teacher_first_tool_input=metrics.teacher_first_tool_input,
            ),
        )

    def _component_trajectories(
        self,
        component_name: str,
        selected: list[Any],
        selected_keys: list[tuple[str, str] | None],
        *,
        trajectories: list[Any],
        mismatch_keys: list[tuple[str, str] | None],
    ) -> list[Any]:
        """Route mismatches to the tool-description or rules module they implicate."""
        if component_name in CORE_TOOL_KEYS:
            return [
                trajectory
                for trajectory, pair in zip(selected, selected_keys, strict=True)
                if pair is not None
                and any(tool_description_override_key(name) == component_name for name in pair if name)
            ]
        if component_name == RULES_EXT_KEY:
            non_core = [
                (trajectory, key)
                for trajectory, key in zip(trajectories, mismatch_keys, strict=True)
                if key is not None and not any(is_core_tool_span(name) for name in key if name)
            ]
            indices, _ = select_mismatch_groups([key for _, key in non_core])
            return [non_core[index][0] for index in indices]
        return selected

    def failure_pattern(self, component_name: str, trajectory: TeacherStudentALTrajectory) -> tuple[Any, ...]:
        del component_name
        output = trajectory["output"]
        tool_alignment = trajectory.get("objective_scores", {}).get(self.name, 1.0)
        return (
            int(tool_alignment < float(self.pack_param("failure_score_below", 0.7))),
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
        tool_alignment = objective_scores.get(self.name, trajectory["score"])
        student_tools = output.get("student_tool_events", [])
        teacher_tools = output.get("teacher_tool_events", [])
        mismatch = self._mismatch_key(output)
        feedback_parts = []
        if mismatch is not None:
            teacher_first, student_first = mismatch
            # An empty scored sequence is a real choice, not missing trace data.
            no_tool = (
                "called no scored tool, emitting only skipped steps such as the automatic vault retrieval or shell"
            )
            teacher_phrase = f"used {teacher_first}" if teacher_first else no_tool
            student_phrase = f"used {student_first}" if student_first else no_tool
            feedback_parts.append(f"First-tool mismatch: teacher {teacher_phrase} and student {student_phrase}.")
        if tool_alignment < 1.0:
            feedback_parts.append(f"Tool alignment issue: score={tool_alignment:.2f}.")
        feedback_parts.extend(self.wired_signal_issues(objective_scores))

        action_inputs: list[str] = []
        for role in ("teacher", "student"):
            pair = output.get(f"{role}_first_tool_input")
            if isinstance(pair, list | tuple) and len(pair) == 2 and pair[1]:
                tool, payload = pair
                payload_text = str(payload)
                if len(payload_text) > 240:
                    payload_text = payload_text[:240] + "... (truncated)"
                action_inputs = [f"{role} first tool ({tool or 'unknown'}): {payload_text}"]
                break
        return self.reflective_example(
            trajectory,
            feedback=" ".join(feedback_parts) if feedback_parts else "General teacher/student tool divergence.",
            generated={
                "student_answer": output.get("student_answer", ""),
                "teacher_answer": output.get("teacher_answer", ""),
                "student_tools": student_tools,
                "teacher_tools": teacher_tools,
            },
            action_inputs=action_inputs,
        )

    def high_signal_core_tool_keys(self, trajectories: Sequence[Any] | None) -> list[str]:
        return high_signal_core_tool_keys(trajectories)


__all__ = ["FirstToolMatchObjective"]
