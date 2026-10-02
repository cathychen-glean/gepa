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
from glean_gepa.judge_metrics_util import CUSTOMER_AGENTIC_PREFERENCE_METRIC
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
    pass_rate,
)
from glean_gepa.objectives.utils.mismatch import REFLECTION_HIGH_SIGNAL_ENTRY_LIMIT
from glean_gepa.objectives.utils.tool_names import (
    SKIPPED_TOOL_NAMES,
    first_tool_mismatch_pair,
    scored_tool_sequence,
)
from glean_gepa.objectives.utils.traces import FetchedByRole, enrich_action_inputs
from glean_gepa.prompt_constants import CORE_TOOLS, EXECUTION_DISCIPLINE_KEY, RULES_EXT_KEY
from glean_gepa.reflection_prompts import (
    EXECUTION_DISCIPLINE_FRAME,
    GENERALITY_RULES,
    RULES_EXT_FRAME,
    compose_responsibility,
    core_tool_frame,
)

TOOL_ALIGNMENT_OBJECTIVE = "tool_alignment"


# ---------------------------------------------------------------------------
# Slot 1: one entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolMatchEntryMetrics:
    """One entry's first-tool comparison.

    Scoring uses only the first tool each role picked. Reflection evidence keeps a
    short prefix of each role's scored calls, including the arguments.
    """

    entry_id: str
    student_tools: tuple[str, ...]
    teacher_tools: tuple[str, ...]
    tools_match: bool
    student_tool_inputs: tuple[tuple[str, str], ...] = ()
    teacher_tool_inputs: tuple[tuple[str, str], ...] = ()
    student_trace_id: str = ""
    teacher_trace_id: str = ""
    student_deployment_id: str = ""
    teacher_deployment_id: str = ""
    student_min_start_ms: int = 0
    student_max_start_ms: int = 0
    teacher_min_start_ms: int = 0
    teacher_max_start_ms: int = 0

    @property
    def passed(self) -> bool:
        return self.tools_match

    @property
    def score(self) -> float:
        return 1.0 if self.tools_match else 0.0


@dataclass(frozen=True)
class ToolMatchMetrics:
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
        student_trace_id=str(row.get("student_trace_id") or ""),
        teacher_trace_id=str(row.get("teacher_trace_id") or ""),
        student_deployment_id=str(row.get("student_deployment_id") or ""),
        teacher_deployment_id=str(row.get("teacher_deployment_id") or ""),
        student_min_start_ms=_millis(row.get("student_min_start_ms")),
        student_max_start_ms=_millis(row.get("student_max_start_ms")),
        teacher_min_start_ms=_millis(row.get("teacher_min_start_ms")),
        teacher_max_start_ms=_millis(row.get("teacher_max_start_ms")),
    )


def _millis(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


_TOOL_INPUT_CHARS = 240
# Each role's scored calls shown to the reflector, kept in sequence order.
TOOL_INPUT_LIMIT = 3


def format_tool_input_lines(role: str, pairs: Sequence[Sequence[str]]) -> list[str]:
    """One reflector line per scored call: ``role tool: payload``."""
    lines: list[str] = []
    for pair in pairs:
        if not isinstance(pair, list | tuple) or len(pair) != 2 or not pair[1]:
            continue
        tool, payload = pair
        text = str(payload)
        if len(text) > _TOOL_INPUT_CHARS:
            text = text[:_TOOL_INPUT_CHARS] + "... (truncated)"
        lines.append(f"{role} {tool or 'unknown'}: {text}")
    return lines


def tool_input_evidence(output: Mapping[str, Any]) -> list[str]:
    """Teacher calls, then student calls, for one reflective example.

    At most ``TOOL_INPUT_LIMIT`` calls per role, in sequence order, so the
    arguments sit next to the tool sequence without crowding the prompt.
    """
    lines: list[str] = []
    for role in ("teacher", "student"):
        raw = output.get(f"{role}_tool_inputs") or ()
        lines.extend(format_tool_input_lines(role, raw)[:TOOL_INPUT_LIMIT])
    return lines


def aggregate_tool_match_metrics(
    per_entry: Mapping[str, ToolMatchEntryMetrics], *, excluded_failed_runs: int = 0
) -> ToolMatchMetrics:
    return ToolMatchMetrics(
        compared_entries=len(per_entry),
        matching_entries=sum(1 for m in per_entry.values() if m.passed),
        tool_match_rate=pass_rate(per_entry),
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
        aggregate=aggregate_tool_match_metrics({}),
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

    def aggregate(per_entry: Mapping[str, ToolMatchEntryMetrics], dropped: int) -> ToolMatchMetrics:
        return aggregate_tool_match_metrics(per_entry, excluded_failed_runs=dropped)

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
        student = fetched.get("student", {}).get(entry_id, metrics.student_tool_inputs)
        teacher = fetched.get("teacher", {}).get(entry_id, metrics.teacher_tool_inputs)
        return replace(
            metrics,
            student_tool_inputs=tuple(student),
            teacher_tool_inputs=tuple(teacher),
        )

    return enrich_action_inputs(
        evalcli,
        per_entry,
        rows,
        high_signal_entry_ids,
        apply=apply,
        roles=("student", "teacher"),
        named_pairs=True,
        skip_tools=skip_tools,
        limit=TOOL_INPUT_LIMIT,
    )


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

# How the first-tool score is decided, so the reflector targets the one decision that moves it.
SCORE_MODEL_NOTE = (
    "HOW THE SCORE IS DECIDED. An entry scores 1 when the first scored tool the student called is "
    "the tool the teacher called first, otherwise 0. Calls to skipped tools (the automatic vault "
    "retrieval, shell, skill lookup) are ignored, so the 'first tool' is the first call that reaches "
    "a search, reader, lister, discovery, or action tool. Nothing after that first call counts, "
    "answer quality does not count, a different but reasonable tool still scores 0, and calling no "
    "scored tool at all is itself a first-tool choice (it matches only when the teacher also called "
    "none). The only lever is therefore the decision the student makes before any tool result "
    "exists, using the user's request, earlier turns, and preloaded context. A rule helps only if "
    "it changes that one decision for the class of requests in the evidence."
)

# What kinds of prompt edits have changed first-tool behavior, and what has not.
EDIT_EFFECTIVENESS_NOTE = (
    "WHAT KIND OF EDIT WORKS. A rule changes the first call only if its trigger is something the "
    "student can see in the request before acting: a tagged or pasted URL or attachment, a request "
    "for the contents of a folder, channel, or project, a verb that names an action in another "
    "system (send, create, update, book), a question about a specific named person, account, "
    "system, or ticket, a request with several independent parts. State the trigger and the tool "
    "together: 'when the request contains a document URL, open it before searching for it'. "
    "Abstract policy ('choose the most appropriate tool', 'consider whether search is needed', 'be "
    "thorough') does not change behavior. In measured runs, rewording a description without adding "
    "or removing a trigger condition left the student's invocation counts for that tool flat, so do "
    "not spend a rewrite on synonyms for text the student already follows. Competing instructions "
    "cancel new rules: if the current text tells the student to answer from preloaded context, "
    "minimize tool loops, cap searches, or try reasoning before tools, and the evidence shows the "
    "teacher calling a tool first, remove or invert that wording rather than appending a counter-rule "
    "beside it. When a tool description and an Execution Discipline bullet disagreed, the student "
    "followed the tool description at the moment of the call, so put the rule in the text that is "
    "read at that moment. Each rewrite should address one or two mismatch mechanisms, name the "
    "trigger and the tool, and stay short; long additions dilute the module. Do not add a rule for a "
    "mechanism that appears in only one supplied example, and do not add a rule for a request class "
    "the student already routes correctly: most entries already match, and a trigger stated too "
    "broadly breaks those matches."
)

# Procedure the reflector should follow before proposing text.
ANALYSIS_PROCEDURE_NOTE = (
    "HOW TO READ EACH EXAMPLE. First read FEEDBACK for the pair: which tool the teacher called "
    "first and which the student called first, or that one side called no scored tool. Then read "
    "the user query and any earlier turns for the cue that should have selected the teacher's tool: "
    "a URL or attachment, a container reference, a verb that names an action, a named entity, a "
    "multi-part ask. Then compare the ACTION_INPUT arguments: the teacher's first call shows what it "
    "searched for, opened, listed, or looked up; the student's first call shows what it did instead. "
    "Classify the mismatch by mechanism, not by tool names alone: a substitute (searching for a "
    "document it was given the URL of), a skip (answering from preloaded context or memory with no "
    "scored call), or an extra step (discovering, listing, or searching before the tool the teacher "
    "went to directly). Only then tally the mechanisms across all supplied examples and state the "
    "tally, with counts, in your diagnosis. A rule needs at least two examples showing the same "
    "mechanism."
)

TOOL_MATCH_GAP_ANALYSIS_GUIDE = "\n\n".join([SCORE_MODEL_NOTE, EDIT_EFFECTIVENESS_NOTE, ANALYSIS_PROCEDURE_NOTE])

EXECUTION_DISCIPLINE_RESPONSIBILITY = compose_responsibility(
    EXECUTION_DISCIPLINE_FRAME,
    "That covers whether to call any tool at all or answer from preloaded context and memory, whether "
    "to retrieve before acting, how many loops and queries to spend, whether to retry after an empty "
    "result, and when to stop and respond. For first-tool matching this module decides one thing: "
    "whether a scored tool gets called at all. It is the place to fix mismatches where the student "
    "called no scored tool and the teacher did (or the reverse), by stating the condition under which "
    "retrieval or lookup is warranted before answering. It does not pick which tool: do not list tool "
    "names with conditions or restate a tool's description here, because the choice between tools is "
    "made by the descriptions the student reads at the call. If the current text tells the student to "
    "answer from context, minimize loops, cap searches, or reason before using tools, and the evidence "
    "shows the teacher calling a tool first, remove or invert that wording. Preserve factual "
    "search-first. Prefer stating the condition under which more work is warranted over raising a "
    "numeric cap, so the rule generalizes to requests of different sizes. Leave response formatting, "
    "citation mechanics, and shell or SDK syntax to other modules.",
    guide=TOOL_MATCH_GAP_ANALYSIS_GUIDE,
)

RULES_EXT_RESPONSIBILITY = compose_responsibility(
    RULES_EXT_FRAME,
    "These bullets govern how the student composes its first shell script: which SDK call it issues "
    "first when several are possible, and that a resource the request already identifies (a URL, an "
    "id, a container) is passed straight to the tool that handles it rather than searched for or "
    "rediscovered first. Use them only when a mismatch is caused by how the first script is written "
    "rather than by a wrong tool choice or by not calling a tool at all; the former belongs in the "
    "tool description, the latter in Execution Discipline. Do not restate either of those here. Keep "
    "each bullet operational, checkable, and concise.",
    guide=TOOL_MATCH_GAP_ANALYSIS_GUIDE,
)


def core_tool_responsibility(tool_name: str) -> str:
    """Reflection instructions for one core-tool ``schema.description`` under first-tool matching."""
    return compose_responsibility(
        core_tool_frame(tool_name),
        "The description acts at one moment: when the student decides which tool to call first. "
        "Rewrite it so the student calls this tool first exactly when the request carries the cue the "
        "teacher responded to, and does not call it first when the teacher chose another tool. Say "
        "what in the request selects this tool; if the evidence shows the student reaching for this "
        "tool as a substitute for another, say which kind of request belongs to the other tool so this "
        "one is not chosen for it. Do not add rules about how many loops to run, when to stop, or "
        "whether to retrieve at all — those belong in Execution Discipline. Do not rewrite merely to "
        "match the preferred run's later tool order; only the first call is scored.",
        guide=TOOL_MATCH_GAP_ANALYSIS_GUIDE,
        closing=f"Keep the text operational and concise. {GENERALITY_RULES}",
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
    student_tool_inputs: Sequence[Sequence[str]] = (),
    teacher_tool_inputs: Sequence[Sequence[str]] = (),
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
    if student_tool_inputs:
        output["student_tool_inputs"] = [list(pair) for pair in student_tool_inputs]
    if teacher_tool_inputs:
        output["teacher_tool_inputs"] = [list(pair) for pair in teacher_tool_inputs]
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
    reflection_selection_justification: ClassVar[str] = (
        "Justification: high-signal first-tool mismatches in eval order, up to the reflection cap. "
        "Entries are not bucketed by teacher/student tool pair, and that pair does not decide "
        "which tool descriptions can be edited."
    )
    module_responsibilities: ClassVar[Mapping[str, str]] = {
        RULES_EXT_KEY: RULES_EXT_RESPONSIBILITY,
        EXECUTION_DISCIPLINE_KEY: EXECUTION_DISCIPLINE_RESPONSIBILITY,
        **{tool: core_tool_responsibility(tool) for tool in CORE_TOOLS},
    }

    def __init__(self, *, bigquery_client: Any | None = None, lookback_days: int = 1):
        self.bigquery_client = bigquery_client
        self.lookback_days = lookback_days
        self.params: dict[str, Any] = {}
        self._paired_analysis_cache: dict[tuple[str, str], EvalRunToolMatchAnalysis] = {}

    def _skipped_tools(self) -> frozenset[str]:
        raw = self.experiment_param("skipped_tools", None)
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
                student_tool_inputs=metrics.student_tool_inputs,
                teacher_tool_inputs=metrics.teacher_tool_inputs,
            ),
        )

    def _select_mismatch_groups(
        self,
        mismatch_keys: Sequence[tuple[str, str] | None],
        *,
        trajectories: Sequence[Any] = (),
        max_entries: int | None = REFLECTION_HIGH_SIGNAL_ENTRY_LIMIT,
    ) -> tuple[list[int], list[tuple[str, str, int]]]:
        """Take high-signal entries in order, up to the cap. No frequency grouping."""
        del trajectories
        if max_entries is None:
            max_entries = len(mismatch_keys)
        selected = [index for index, key in enumerate(mismatch_keys) if key is not None]
        return selected[:max_entries], []

    def failure_pattern(self, component_name: str, trajectory: TeacherStudentALTrajectory) -> tuple[Any, ...]:
        del component_name
        output = trajectory["output"]
        tool_alignment = trajectory.get("objective_scores", {}).get(self.name, 1.0)
        return (
            int(tool_alignment < float(self.experiment_param("failure_score_below", 0.7))),
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
        verdict = output.get(f"{CUSTOMER_AGENTIC_PREFERENCE_METRIC}_feedback")
        if isinstance(verdict, str) and verdict.strip():
            feedback_parts.append(f"Agentic judge verdict:\n{verdict.strip()}")

        return self.reflective_example(
            trajectory,
            feedback=" ".join(feedback_parts) if feedback_parts else "General teacher/student tool divergence.",
            generated={
                "student_answer": output.get("student_answer", ""),
                "teacher_answer": output.get("teacher_answer", ""),
                "student_tools": student_tools,
                "teacher_tools": teacher_tools,
            },
            action_inputs=tool_input_evidence(output),
            action_input_limit=2 * TOOL_INPUT_LIMIT,
        )


__all__ = ["FirstToolMatchObjective"]
