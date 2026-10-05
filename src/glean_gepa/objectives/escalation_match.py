"""Escalation-match objective: does the student's Waldo router escalate on the entries the teacher's does?

Waldo is the pre-loop router. On each request it either answers with its attached tools
(``SAW_READY``) or hands off to the full agent. The deliberate hand-off is
``SAW_INSUFFICIENT_TOOLS``. Every other termination (``NO_SAMPLE``, ``MAX_LOOPS_EXHAUSTED``,
``NO_TOOL_CALLS``) also reaches the full agent, but as a fallback, not a decision.

The score is strict escalation agreement: both runs end ``SAW_INSUFFICIENT_TOOLS`` or neither
does, with fallback terminations counted as not escalating. ``handoff_match`` (agreement on
"anything but ``SAW_READY``") is logged on the aggregate only. It is not emitted as a score
dimension, so it never becomes a Pareto frontier key. Only entries where Waldo ran in both runs
are scored. Skips are input-driven, so they are counted and checked for teacher/student parity
but not scored.

Do not read ``auto_mode_escalated``: that is Auto Mode fast-to-thinking, not Waldo. The inner
loop's own IC Preloop span is excluded by span name.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, ClassVar

from glean_gepa.adapter_types import TeacherStudentALRolloutOutput, TeacherStudentALTrajectory, paired_rollout_output
from glean_gepa.al_adapter import ReflectiveExample
from glean_gepa.focused_evalset import QUERY_CANONICAL_BUCKET_TYPE
from glean_gepa.judge_metrics_util import CUSTOMER_AGENTIC_PREFERENCE_METRIC
from glean_gepa.objectives.base import AnalysisRequest, ScoredRow, ScoringContext, TeacherStudentObjective
from glean_gepa.objectives.utils.agentspan import Rows, bounds_query, fetch_agentspan_analysis
from glean_gepa.objectives.utils.agentspan_query import (
    DEFAULT_AGENTS_SPAN_TABLE,
    DEFAULT_LOOKBACK_DAYS,
    EVAL_ENTRY_ID_EXPR,
    default_date_range,
    wildcard_shard_filter,
)
from glean_gepa.objectives.utils.core import NoComparedEntriesError, PairedRunAnalysis, log_analysis
from glean_gepa.objectives.utils.mismatch import REFLECTION_HIGH_SIGNAL_ENTRY_LIMIT
from glean_gepa.reflection_prompts import GENERALITY_RULES, compose_responsibility
from glean_gepa.waldo_prompt_constants import WALDO_ROUTING_KEY, WALDO_TOOL_USAGE_KEY

ESCALATION_MATCH_OBJECTIVE = "escalation_match"

WALDO_AGENT_SPAN = "Agent Run: WaldoAgent"
WALDO_SPAN = "Waldo"
WALDO_REFUSAL_SPAN = "Waldo escalation: first-sentence refusal"
WALDO_SPAN_FILTER = f"jsonPayload.span_info.span_name IN ('{WALDO_AGENT_SPAN}', '{WALDO_SPAN}', '{WALDO_REFUSAL_SPAN}')"

SAW_INSUFFICIENT_TOOLS = "SAW_INSUFFICIENT_TOOLS"
SAW_READY = "SAW_READY"
SKIPPED_MESSAGE_PREFIX = "skipped:"

# Per-run Waldo outcomes. Skips are ``skip:<reason>``.
ESCALATED = "escalated"
ANSWERED = "answered"
OTHER = "other"
SKIP_PREFIX = "skip:"
# No Waldo span at all in this run: the entry never ran, or ingest has not landed.
ABSENT = "absent"
# Waldo spans exist but carry neither a skip nor a termination, e.g. the WaldoAgent run failed.
INCOMPLETE = "incomplete"

_RAN_OUTCOMES = frozenset({ESCALATED, ANSWERED, OTHER})
_ROLE_COLUMNS = ("agent_message", "termination", "waldo_summary", "first_sentence_refusal", "trace_id")


# ---------------------------------------------------------------------------
# Slot 1: one entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WaldoDecision:
    """One run's Waldo outcome on one entry, read from its latest trace."""

    outcome: str
    termination: str = ""
    # The Waldo span's status message, e.g. ``termination=SAW_READY, loops=2, tool_calls=2``.
    summary: str = ""
    first_sentence_refusal: bool = False

    @property
    def present(self) -> bool:
        return self.outcome != ABSENT

    @property
    def ran(self) -> bool:
        return self.outcome in _RAN_OUTCOMES

    @property
    def escalated(self) -> bool:
        return self.outcome == ESCALATED

    @property
    def handed_off(self) -> bool:
        return self.outcome in (ESCALATED, OTHER)

    @property
    def skip_reason(self) -> str:
        return self.outcome[len(SKIP_PREFIX) :] if self.outcome.startswith(SKIP_PREFIX) else ""


def classify_waldo_decision(
    agent_message: Any,
    termination: Any,
    *,
    present: bool = True,
    summary: Any = "",
    first_sentence_refusal: Any = False,
) -> WaldoDecision:
    """Classify one run's latest trace. A WaldoAgent ``skipped:*`` message wins over any termination."""
    message = str(agent_message or "").strip()
    code = str(termination or "").strip()
    if not present:
        outcome = ABSENT
    elif message.startswith(SKIPPED_MESSAGE_PREFIX):
        outcome = SKIP_PREFIX + message[len(SKIPPED_MESSAGE_PREFIX) :]
    elif not code:
        outcome = INCOMPLETE
    elif code == SAW_INSUFFICIENT_TOOLS:
        outcome = ESCALATED
    elif code == SAW_READY:
        outcome = ANSWERED
    else:
        outcome = OTHER
    return WaldoDecision(
        outcome=outcome,
        termination=code,
        summary=str(summary or "").strip(),
        first_sentence_refusal=bool(first_sentence_refusal),
    )


@dataclass(frozen=True)
class EscalationMatchEntryMetrics:
    entry_id: str
    teacher: WaldoDecision
    student: WaldoDecision

    @property
    def both_ran(self) -> bool:
        return self.teacher.ran and self.student.ran

    @property
    def escalation_match(self) -> bool:
        return self.teacher.escalated == self.student.escalated

    @property
    def handoff_match(self) -> bool:
        return self.teacher.handed_off == self.student.handed_off

    @property
    def passed(self) -> bool:
        return self.escalation_match

    @property
    def score(self) -> float:
        return 1.0 if self.escalation_match else 0.0


@dataclass(frozen=True)
class EscalationConfusion:
    """Teacher-vs-student counts for one escalation definition over both-ran entries."""

    both: int = 0
    teacher_only: int = 0
    student_only: int = 0
    neither: int = 0

    @property
    def total(self) -> int:
        return self.both + self.teacher_only + self.student_only + self.neither

    @property
    def teacher_rate(self) -> float:
        return (self.both + self.teacher_only) / self.total if self.total else 0.0

    @property
    def student_rate(self) -> float:
        return (self.both + self.student_only) / self.total if self.total else 0.0

    @property
    def agreement(self) -> float:
        return (self.both + self.neither) / self.total if self.total else 0.0

    @property
    def overlap(self) -> float:
        """Jaccard of the two escalated sets; 0.0 when neither run escalated anywhere."""
        union = self.both + self.teacher_only + self.student_only
        return self.both / union if union else 0.0

    def describe(self) -> str:
        return (
            f"teacher={self.teacher_rate:.1%} student={self.student_rate:.1%}; both={self.both} "
            f"teacher-only={self.teacher_only} student-only={self.student_only} neither={self.neither}; "
            f"agreement={self.agreement:.1%} overlap={self.overlap:.1%}"
        )


@dataclass(frozen=True)
class WaldoCoverage:
    """How paired entries split before scoring."""

    # Entries with Waldo spans in both runs.
    common_entries: int = 0
    # Entries with Waldo spans in only one run.
    unpaired_entries: int = 0
    # Common entries where a run has Waldo spans but neither a skip nor a termination.
    incomplete_entries: int = 0
    # Common entries Waldo skipped in one run only. These change which entries get scored.
    skip_status_mismatches: int = 0
    # Common entries both runs skipped, for different reasons.
    skip_reason_mismatches: int = 0
    teacher_skip_reasons: Mapping[str, int] = field(default_factory=dict)
    student_skip_reasons: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class EscalationMatchMetrics:
    compared_entries: int
    escalation_match: float
    handoff_match: float
    strict: EscalationConfusion = field(default_factory=EscalationConfusion)
    effective: EscalationConfusion = field(default_factory=EscalationConfusion)
    coverage: WaldoCoverage = field(default_factory=WaldoCoverage)


class EvalRunEscalationMatchAnalysis(PairedRunAnalysis[EscalationMatchMetrics, EscalationMatchEntryMetrics]):
    """Agentspan analyses always carry a resolved shard window."""

    start_date: date
    end_date: date


# ---------------------------------------------------------------------------
# Slot 2: rows -> entry, entries -> aggregate
# ---------------------------------------------------------------------------


def _role_decision(row: Mapping[str, Any], role: str) -> WaldoDecision:
    return classify_waldo_decision(
        row.get(f"{role}_agent_message"),
        row.get(f"{role}_termination"),
        present=bool(row.get(f"{role}_trace_id")),
        summary=row.get(f"{role}_waldo_summary"),
        first_sentence_refusal=row.get(f"{role}_first_sentence_refusal"),
    )


def parse_escalation_match_row(row: Mapping[str, Any]) -> EscalationMatchEntryMetrics | None:
    entry_id = str(row.get("entry_id") or "")
    if not entry_id:
        return None
    return EscalationMatchEntryMetrics(
        entry_id=entry_id,
        teacher=_role_decision(row, "teacher"),
        student=_role_decision(row, "student"),
    )


def escalation_confusion(
    entries: Iterable[EscalationMatchEntryMetrics], decided: Callable[[WaldoDecision], bool]
) -> EscalationConfusion:
    counts = Counter((decided(m.teacher), decided(m.student)) for m in entries)
    return EscalationConfusion(
        both=counts[(True, True)],
        teacher_only=counts[(True, False)],
        student_only=counts[(False, True)],
        neither=counts[(False, False)],
    )


def summarize_coverage(entries: Sequence[EscalationMatchEntryMetrics]) -> WaldoCoverage:
    common = [m for m in entries if m.teacher.present and m.student.present]
    return WaldoCoverage(
        common_entries=len(common),
        unpaired_entries=len(entries) - len(common),
        incomplete_entries=sum(1 for m in common if INCOMPLETE in (m.teacher.outcome, m.student.outcome)),
        skip_status_mismatches=sum(1 for m in common if bool(m.teacher.skip_reason) != bool(m.student.skip_reason)),
        skip_reason_mismatches=sum(
            1
            for m in common
            if m.teacher.skip_reason and m.student.skip_reason and m.teacher.skip_reason != m.student.skip_reason
        ),
        teacher_skip_reasons=dict(Counter(m.teacher.skip_reason for m in common if m.teacher.skip_reason)),
        student_skip_reasons=dict(Counter(m.student.skip_reason for m in common if m.student.skip_reason)),
    )


def aggregate_escalation_match_metrics(
    per_entry: Mapping[str, EscalationMatchEntryMetrics], *, coverage: WaldoCoverage | None = None
) -> EscalationMatchMetrics:
    strict = escalation_confusion(per_entry.values(), lambda d: d.escalated)
    effective = escalation_confusion(per_entry.values(), lambda d: d.handed_off)
    return EscalationMatchMetrics(
        compared_entries=len(per_entry),
        escalation_match=strict.agreement,
        handoff_match=effective.agreement,
        strict=strict,
        effective=effective,
        coverage=coverage or WaldoCoverage(),
    )


def escalation_mismatch_pair(teacher_termination: Any, student_termination: Any) -> tuple[str, str] | None:
    """``(teacher, student)`` terminations when exactly one run escalated, else ``None``."""
    teacher = str(teacher_termination or "")
    student = str(student_termination or "")
    if (teacher == SAW_INSUFFICIENT_TOOLS) == (student == SAW_INSUFFICIENT_TOOLS):
        return None
    return teacher, student


# ---------------------------------------------------------------------------
# Slot 3: SQL
# ---------------------------------------------------------------------------


def _latest(column: str, span_name: str) -> str:
    return (
        f"ARRAY_AGG(IF(s.span_name = '{span_name}', s.{column}, NULL) IGNORE NULLS "
        "ORDER BY s.start_ms DESC LIMIT 1)[SAFE_OFFSET(0)]"
    )


def build_escalation_match_per_entry_query(*, agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE) -> str:
    """Pair each entry's Waldo outcome across the teacher and student runs.

    An entry can have several traces (retries); only the trace holding the entry's latest
    Waldo span is read, which drops failed attempts. Within that trace the latest WaldoAgent
    message and Waldo termination win.
    """
    role_cols = ",\n".join(
        f"  {role}.{col} AS {role}_{col}" for role in ("student", "teacher") for col in _ROLE_COLUMNS
    )
    return f"""
WITH waldo_spans AS (
  SELECT
    jsonPayload.context.eval.eval_id AS eval_id,
    {EVAL_ENTRY_ID_EXPR} AS entry_id,
    jsonPayload.context.agent_trace.trace_id AS trace_id,
    jsonPayload.span_info.span_name AS span_name,
    jsonPayload.span_info.execution_status.message AS message,
    jsonPayload.ic_preloop.termination AS termination,
    SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64) AS start_ms
  FROM `{agentspan_table}`
  WHERE {wildcard_shard_filter("start_date", "end_date")}
    AND jsonPayload.context.eval.eval_id IN UNNEST(@eval_ids)
    AND {WALDO_SPAN_FILTER}
),
latest_trace AS (
  SELECT
    eval_id,
    entry_id,
    ARRAY_AGG(trace_id ORDER BY start_ms DESC LIMIT 1)[OFFSET(0)] AS trace_id
  FROM waldo_spans
  WHERE entry_id IS NOT NULL AND trace_id IS NOT NULL
  GROUP BY eval_id, entry_id
),
per_role AS (
  SELECT
    s.eval_id,
    s.entry_id,
    ANY_VALUE(s.trace_id) AS trace_id,
    {_latest("message", WALDO_AGENT_SPAN)} AS agent_message,
    {_latest("termination", WALDO_SPAN)} AS termination,
    {_latest("message", WALDO_SPAN)} AS waldo_summary,
    LOGICAL_OR(s.span_name = '{WALDO_REFUSAL_SPAN}') AS first_sentence_refusal
  FROM waldo_spans AS s
  JOIN latest_trace AS t
    ON s.eval_id = t.eval_id AND s.entry_id = t.entry_id AND s.trace_id = t.trace_id
  GROUP BY s.eval_id, s.entry_id
),
student AS (SELECT * FROM per_role WHERE eval_id = @student_eval_id),
teacher AS (SELECT * FROM per_role WHERE eval_id = @teacher_eval_id)
SELECT
  COALESCE(student.entry_id, teacher.entry_id) AS entry_id,
{role_cols}
FROM student
FULL OUTER JOIN teacher
  ON student.entry_id = teacher.entry_id
ORDER BY entry_id
""".strip()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def empty_escalation_match_analysis(
    teacher_eval_id: str,
    student_eval_id: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: date | None = None,
) -> EvalRunEscalationMatchAnalysis:
    start_date, resolved_end = default_date_range(lookback_days=lookback_days, end_date=end_date)
    return EvalRunEscalationMatchAnalysis(
        eval_ids=(teacher_eval_id, student_eval_id),
        aggregate=aggregate_escalation_match_metrics({}),
        start_date=start_date,
        end_date=resolved_end,
    )


def fetch_eval_run_escalation_match_analysis(
    client: Any,
    *,
    teacher_eval_id: str,
    student_eval_id: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: date | None = None,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
    evalcli: Any | None = None,
    include_action_inputs: bool = True,
) -> EvalRunEscalationMatchAnalysis:
    # Waldo's decision and the user query are the evidence; no tool payloads are hydrated.
    del evalcli, include_action_inputs
    coverage: dict[str, WaldoCoverage] = {}

    def keep_both_ran(rows: Rows) -> Rows:
        parsed = [parse_escalation_match_row(row) for row in rows]
        coverage["value"] = summarize_coverage([m for m in parsed if m is not None])
        return [row for row, m in zip(rows, parsed, strict=True) if m is not None and m.both_ran]

    def aggregate(per_entry: Mapping[str, EscalationMatchEntryMetrics], dropped: int) -> EscalationMatchMetrics:
        del dropped
        return aggregate_escalation_match_metrics(per_entry, coverage=coverage.get("value"))

    analysis = fetch_agentspan_analysis(
        client,
        eval_ids=(teacher_eval_id, student_eval_id),
        bounds_sql=bounds_query(
            eval_id_predicate="IN UNNEST(@eval_ids)", span_filter=WALDO_SPAN_FILTER, agentspan_table=agentspan_table
        ),
        per_entry_sql=build_escalation_match_per_entry_query(agentspan_table=agentspan_table),
        parse_row=parse_escalation_match_row,
        aggregate=aggregate,
        is_high_signal=lambda m: not m.escalation_match,
        filter_rows=keep_both_ran,
        lookback_days=lookback_days,
        end_date=end_date,
    )
    if analysis.start_date is None:
        return empty_escalation_match_analysis(
            teacher_eval_id, student_eval_id, lookback_days=lookback_days, end_date=end_date
        )
    return EvalRunEscalationMatchAnalysis(
        eval_ids=analysis.eval_ids,
        aggregate=analysis.aggregate,
        per_entry=analysis.per_entry,
        high_signal_entry_ids=analysis.high_signal_entry_ids,
        start_date=analysis.start_date,
        end_date=analysis.end_date or analysis.start_date,
    )


def interleave_mismatch_groups(
    mismatch_keys: Sequence[tuple[str, str] | None], *, max_entries: int | None
) -> tuple[list[int], list[tuple[str, str, int]]]:
    """Round-robin across mismatch groups, most frequent first, up to ``max_entries``.

    Unlike ``select_mismatch_groups``, the dominant group is not taken whole, so the rarer
    direction (usually over-escalation) still reaches the reflector.
    """
    groups: dict[tuple[str, str], list[int]] = {}
    for index, key in enumerate(mismatch_keys):
        if key is not None:
            groups.setdefault(key, []).append(index)
    ranked = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    limit = sum(len(indices) for indices in groups.values()) if max_entries is None else max_entries
    taken: dict[tuple[str, str], int] = {key: 0 for key, _ in ranked}
    selected: list[int] = []
    depth = 0
    while len(selected) < limit:
        progressed = False
        for key, indices in ranked:
            if depth < len(indices) and len(selected) < limit:
                selected.append(indices[depth])
                taken[key] += 1
                progressed = True
        if not progressed:
            break
        depth += 1
    return sorted(selected), [(a, b, count) for (a, b), count in taken.items() if count]


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

WALDO_TOOL_USAGE_FRAME = (
    "You are rewriting the 'Tool Usage Guidelines' section of the Waldo router prompt. It has two "
    "conditional branches, <<<[[has_search_tools]] ... >>> and <<<[[no_search_tools]] ... >>>; scio "
    "renders exactly one per request. Keep both branches and every [[...]] placeholder."
)

SCORE_MODEL_NOTE = (
    "HOW THE SCORE IS DECIDED. Waldo either answers with its attached tools or escalates to the full "
    "agent by calling discover, which ends it with SAW_INSUFFICIENT_TOOLS. An entry scores 1 when the "
    "student makes the same escalation decision as the teacher: both escalate, or neither does. "
    "Answering (SAW_READY) and ending any other way (NO_SAMPLE timeouts, MAX_LOOPS_EXHAUSTED, "
    "NO_TOOL_CALLS) both count as not escalating, so running out of loops or time is never a correct "
    "escalation even though the request reaches the full agent. Entries where Waldo was skipped are "
    "not scored. Answer quality, which attached tool was used, and how many loops ran do not count; "
    "the only lever is when this section tells Waldo to call discover."
)

EDIT_EFFECTIVENESS_NOTE = (
    "WHAT KIND OF EDIT WORKS. Mismatches go both ways: under-escalation (teacher escalated, student "
    "did not) and over-escalation (student escalated, teacher did not). Tally them separately. A rule "
    "changes the decision only if its trigger is visible in the request before any tool runs: the kind "
    "of lookup asked for, whether it needs reasoning across several sources, whether it asks for an "
    "action, analysis, or artifact the attached tools cannot produce. State the trigger and the "
    "decision together, such as 'when the request asks for X, call discover rather than searching'. "
    "Blanket policy ('escalate when unsure', 'prefer answering') moves both directions at once and "
    "breaks entries that already match. When the student ran several searches and then answered or "
    "ran out of loops, it judged the attached tools sufficient: describe the request class where they "
    "are not, and say to escalate before searching rather than after."
)

ANALYSIS_PROCEDURE_NOTE = (
    "HOW TO READ EACH EXAMPLE. Read FEEDBACK for the direction of the mismatch and how each side "
    "ended, including its loop and tool-call counts. Then read the user query for the cue the teacher "
    "responded to. Classify each example by mechanism: answered from attached tools when the request "
    "needed more, kept searching until loops or time ran out instead of escalating, or escalated a "
    "simple lookup the attached tools cover. State the tally, with counts, in your diagnosis. A rule "
    "needs at least two examples showing the same mechanism."
)

ESCALATION_MATCH_GAP_ANALYSIS_GUIDE = "\n\n".join([SCORE_MODEL_NOTE, EDIT_EFFECTIVENESS_NOTE, ANALYSIS_PROCEDURE_NOTE])

ESCALATION_TASK = (
    "This section decides when Waldo calls discover (escalate to the full agent) versus using the "
    "attached search tools and answering directly. Rewrite it so the student escalates on the kinds "
    "of request the teacher escalates on, and answers the kinds it answers. State rules in terms of "
    "request type, never specific documents, people, or customers from the examples. Keep the text "
    "operational and concise."
)

WALDO_TOOL_USAGE_RESPONSIBILITY = compose_responsibility(
    WALDO_TOOL_USAGE_FRAME,
    ESCALATION_TASK,
    guide=ESCALATION_MATCH_GAP_ANALYSIS_GUIDE,
    closing=GENERALITY_RULES,
)

WALDO_ROUTING_FRAME = (
    "You are rewriting the routing core at the top of the Waldo router prompt: the First-Action "
    "Decision, the tool-state constraints, the grounding, capability, and Mandatory Discover gates, and "
    "the Routing section. It has conditional branches, <<<[[has_search_tools]] ... >>> and "
    "<<<[[no_search_tools]] ... >>>; scio renders exactly one per request. Keep both branches and every "
    "[[...]] placeholder. The same routing rule is restated in several subsections; when you change "
    "one, change every restatement so the prompt does not contradict itself. Sections after this one, "
    "which you cannot edit, refer back to 'easy lookup' and to 'First-Action Decision' by name: keep "
    "that term and heading. 'Source Selection for Easy Lookups' picks internal versus web search and "
    "does not affect escalation; leave it unchanged."
)

WALDO_ROUTING_RESPONSIBILITY = compose_responsibility(
    WALDO_ROUTING_FRAME,
    ESCALATION_TASK,
    guide=ESCALATION_MATCH_GAP_ANALYSIS_GUIDE,
    closing=GENERALITY_RULES,
)


def _role_outcome_phrase(termination: str) -> str:
    if termination == SAW_INSUFFICIENT_TOOLS:
        return "escalated to the full agent (SAW_INSUFFICIENT_TOOLS)"
    if termination == SAW_READY:
        return "answered with its attached tools (SAW_READY)"
    return (
        f"did not escalate and ended with {termination or 'no termination'}, reaching the full agent only as a fallback"
    )


def escalation_feedback(output: Mapping[str, Any]) -> str:
    """One mismatch sentence plus each side's Waldo span summary."""
    teacher = str(output.get("teacher_waldo_termination") or "")
    student = str(output.get("student_waldo_termination") or "")
    if escalation_mismatch_pair(teacher, student) is None:
        return ""
    direction = "Under-escalation" if teacher == SAW_INSUFFICIENT_TOOLS else "Over-escalation"
    parts = [
        f"{direction}: the teacher {_role_outcome_phrase(teacher)} and the student {_role_outcome_phrase(student)}."
    ]
    for role in ("teacher", "student"):
        summary = str(output.get(f"{role}_waldo_summary") or "")
        refusal = (
            " The escalation came from a first-sentence refusal."
            if output.get(f"{role}_first_sentence_refusal")
            else ""
        )
        if summary or refusal:
            parts.append(f"{role.capitalize()} Waldo: {summary or '(no summary)'}.{refusal}")
    return " ".join(parts)


def _rollout_output(
    *,
    entry_id: str,
    deployment_id: str,
    query: str,
    teacher: WaldoDecision | None = None,
    student: WaldoDecision | None = None,
) -> TeacherStudentALRolloutOutput:
    output = paired_rollout_output(
        deployment_id=deployment_id,
        query=query,
        entry_id=entry_id,
        teacher_answer=(teacher.summary or teacher.termination) if teacher else "",
        student_answer=(student.summary or student.termination) if student else "",
    )
    if teacher is not None:
        output["teacher_waldo_termination"] = teacher.termination
        output["teacher_waldo_summary"] = teacher.summary
        output["teacher_first_sentence_refusal"] = teacher.first_sentence_refusal
    if student is not None:
        output["student_waldo_termination"] = student.termination
        output["student_waldo_summary"] = student.summary
        output["student_first_sentence_refusal"] = student.first_sentence_refusal
    return output


class EscalationMatchObjective(TeacherStudentObjective[EvalRunEscalationMatchAnalysis]):
    """Score whether the student's Waldo escalates on the same entries as the teacher's."""

    name = ESCALATION_MATCH_OBJECTIVE
    telemetry_dimensions = (ESCALATION_MATCH_OBJECTIVE,)
    focused_bucket_type = QUERY_CANONICAL_BUCKET_TYPE
    failure_label = "HIGH-SIGNAL FAILURES (teacher vs student Waldo escalation)"
    reflection_report_title = "REFLECTION: teacher vs student Waldo escalation"
    teacher_compared_key = "teacher_waldo_termination"
    student_compared_key = "student_waldo_termination"
    mismatch_pair = escalation_mismatch_pair
    reflection_selection_justification: ClassVar[str] = (
        "Justification: escalation mismatches interleaved across (teacher, student) termination "
        "groups, most frequent first, up to the reflection cap, so over-escalation examples are "
        "not crowded out by the usually larger under-escalation group."
    )
    module_responsibilities: ClassVar[Mapping[str, str]] = {
        WALDO_ROUTING_KEY: WALDO_ROUTING_RESPONSIBILITY,
        WALDO_TOOL_USAGE_KEY: WALDO_TOOL_USAGE_RESPONSIBILITY,
    }

    def __init__(self, *, bigquery_client: Any | None = None, lookback_days: int = 1):
        self.bigquery_client = bigquery_client
        self.lookback_days = lookback_days
        self.params: dict[str, Any] = {}
        self._paired_analysis_cache: dict[tuple[str, str], EvalRunEscalationMatchAnalysis] = {}

    def analyze(
        self, teacher_eval_id: str, student_eval_id: str, *, request: AnalysisRequest
    ) -> EvalRunEscalationMatchAnalysis:
        return self.cached_paired_analysis(
            teacher_eval_id,
            student_eval_id,
            request=request,
            cache=self._paired_analysis_cache,
            fetch=fetch_eval_run_escalation_match_analysis,
            empty=empty_escalation_match_analysis,
            label="escalation match analysis",
        )

    def require_compared_entries(self, analysis: EvalRunEscalationMatchAnalysis) -> None:
        """Reject a 0/0 comparison: no entry ran Waldo in both runs."""
        if analysis.aggregate.compared_entries > 0:
            return
        coverage = analysis.aggregate.coverage
        reason = (
            f"none of the {coverage.common_entries} common entries ran Waldo in both runs "
            f"(teacher skips {dict(coverage.teacher_skip_reasons)}, student skips "
            f"{dict(coverage.student_skip_reasons)}, incomplete {coverage.incomplete_entries})"
            if coverage.common_entries
            else "wait for agentspan ingest or check that both runs enabled the Waldo router"
        )
        raise NoComparedEntriesError(
            f"No eval entries were compared for student {analysis.student_eval_id} vs "
            f"teacher {analysis.teacher_eval_id}: {reason}."
        )

    def validate_full_eval(self, analysis: EvalRunEscalationMatchAnalysis) -> None:
        aggregate = analysis.aggregate
        coverage = aggregate.coverage
        print(
            f"[Escalation Match] {coverage.common_entries} common entries, {aggregate.compared_entries} "
            f"both-ran, {coverage.unpaired_entries} in one run only, {coverage.incomplete_entries} incomplete"
        )
        print(
            f"[Escalation Match] Skip reasons: teacher {dict(coverage.teacher_skip_reasons)}, "
            f"student {dict(coverage.student_skip_reasons)}"
        )
        if coverage.skip_status_mismatches:
            print(
                f"[Escalation Match] WARNING: {coverage.skip_status_mismatches} common entries skipped Waldo "
                "in one run only. Skips are input-driven, so check that both runs share harness params."
            )
        if coverage.skip_reason_mismatches:
            print(
                f"[Escalation Match] {coverage.skip_reason_mismatches} entries were skipped by both runs "
                "for different reasons; they are not scored either way."
            )
        print(f"[Escalation Match] Effective handoff: {aggregate.effective.describe()}")
        log_analysis(
            analysis,
            label="Escalation Match",
            headline=f"vs {analysis.teacher_eval_id}: strict escalation {aggregate.strict.describe()}",
            entry_line=lambda m: f"teacher={m.teacher.termination} student={m.student.termination}",
        )

    def focused_pass_rate(self, analysis: EvalRunEscalationMatchAnalysis, requested_entry_ids: Sequence[str]) -> float:
        if not requested_entry_ids:
            return 0.0
        matching = sum(
            1 for entry_id in requested_entry_ids if (m := analysis.per_entry.get(entry_id)) and m.escalation_match
        )
        return matching / len(requested_entry_ids)

    def aggregate_row(self, analysis: EvalRunEscalationMatchAnalysis, ctx: ScoringContext) -> ScoredRow:
        return ScoredRow(
            entry_id=None,
            dimension_scores={ESCALATION_MATCH_OBJECTIVE: analysis.aggregate.escalation_match},
            output=_rollout_output(entry_id=ctx.query, deployment_id=ctx.deployment_id, query=ctx.query),
        )

    def entry_row(
        self,
        entry_id: str,
        metrics: EscalationMatchEntryMetrics,
        analysis: EvalRunEscalationMatchAnalysis,
        ctx: ScoringContext,
    ) -> ScoredRow:
        del analysis
        return ScoredRow(
            entry_id=entry_id,
            dimension_scores={ESCALATION_MATCH_OBJECTIVE: float(metrics.escalation_match)},
            output=_rollout_output(
                entry_id=entry_id,
                deployment_id=ctx.deployment_id,
                query=ctx.query,
                teacher=metrics.teacher,
                student=metrics.student,
            ),
        )

    def _select_mismatch_groups(
        self,
        mismatch_keys: Sequence[tuple[str, str] | None],
        *,
        trajectories: Sequence[Any] = (),
        max_entries: int | None = REFLECTION_HIGH_SIGNAL_ENTRY_LIMIT,
    ) -> tuple[list[int], list[tuple[str, str, int]]]:
        del trajectories
        return interleave_mismatch_groups(mismatch_keys, max_entries=max_entries)

    def failure_pattern(self, component_name: str, trajectory: TeacherStudentALTrajectory) -> tuple[Any, ...]:
        del component_name
        output = trajectory["output"]
        escalation_match = trajectory.get("objective_scores", {}).get(self.name, 1.0)
        return (
            int(escalation_match < float(self.experiment_param("failure_score_below", 1.0))),
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
        feedback_parts = [escalation_feedback(output)] if self._mismatch_key(output) is not None else []
        feedback_parts.extend(self.wired_signal_issues(objective_scores))
        verdict = output.get(f"{CUSTOMER_AGENTIC_PREFERENCE_METRIC}_feedback")
        if isinstance(verdict, str) and verdict.strip():
            feedback_parts.append(f"Agentic judge verdict:\n{verdict.strip()}")
        return self.reflective_example(
            trajectory,
            feedback=" ".join(part for part in feedback_parts if part) or "Teacher and student escalation agree.",
            generated={
                "student_answer": output.get("student_answer", ""),
                "teacher_answer": output.get("teacher_answer", ""),
            },
        )


__all__ = ["EscalationMatchObjective"]
