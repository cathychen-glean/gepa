"""Golden escalation objective: does one model's Waldo router take each request's labeled route?

Waldo is the pre-loop router. On each request it either answers with its attached tools
(``SAW_READY``) or hands off to the full agent. The deliberate hand-off is
``SAW_INSUFFICIENT_TOOLS``. Every other termination (``NO_SAMPLE``, ``MAX_LOOPS_EXHAUSTED``,
``NO_TOOL_CALLS``) also reaches the full agent, but as a fallback, not a decision.

Each eval-set entry carries a human-reviewed route in ``expectedOutput.canonicalAnswer``, a JSON
row whose ``target_route`` is ``discover`` (escalate) or ``direct_response`` (answer). An entry
scores 1 when the student's latest Waldo trace takes that route. A fallback scores 0 whichever the
label. Skipped, incomplete, and unlabeled entries are counted but not scored.

Labels are joined on query text. A QUERY_CANONICAL focused copy keeps each query but neither its
``canonicalAnswer`` nor its entry id, so an entry without a label of its own takes the label of the
same query in any labeled set listed earlier or named in ``objective.params.label_eval_sets``. The
adapter passes the run's eval set in ``AnalysisRequest``; nothing else records it.

Do not read ``auto_mode_escalated``: that is Auto Mode fast-to-thinking, not Waldo. The inner
loop's own IC Preloop span is excluded by span name.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, ClassVar

from glean_gepa.adapter_types import SingleModelALRolloutOutput, SingleModelALTrajectory
from glean_gepa.al_adapter import ReflectiveExample
from glean_gepa.focused_evalset import QUERY_CANONICAL_BUCKET_TYPE
from glean_gepa.objectives.base import AnalysisRequest, ScoredRow, ScoringContext, SingleModelObjective
from glean_gepa.objectives.utils.agentspan import Rows, bounds_query, fetch_agentspan_analysis
from glean_gepa.objectives.utils.agentspan_query import (
    DEFAULT_AGENTS_SPAN_TABLE,
    DEFAULT_LOOKBACK_DAYS,
    EVAL_ENTRY_ID_EXPR,
    default_date_range,
    wildcard_shard_filter,
)
from glean_gepa.objectives.utils.core import RunAnalysis, log_analysis
from glean_gepa.prompt_constants import WALDO_ROUTING_KEY, WALDO_SYSTEM_KEY, WALDO_TOOL_USAGE_KEY
from glean_gepa.reflection_prompts import NO_EXAMPLE_SPECIFICS_RULE, compose_responsibility

GOLDEN_ESCALATION_MATCH_OBJECTIVE = "golden_escalation_match"

ESCALATE_ROUTE = "discover"
ANSWER_ROUTE = "direct_response"
KNOWN_ROUTES = frozenset({ESCALATE_ROUTE, ANSWER_ROUTE})
# Label fields shown to the reflector beside the route, in this order.
_LABEL_NOTE_FIELDS = ("review_notes", "failure_mode", "human_comment", "label_caveat")
_LABEL_NOTE_LIMIT = 300

WALDO_AGENT_SPAN = "Agent Run: WaldoAgent"
WALDO_SPAN = "Waldo"
WALDO_REFUSAL_SPAN = "Waldo escalation: first-sentence refusal"
WALDO_SPAN_FILTER = f"jsonPayload.span_info.span_name IN ('{WALDO_AGENT_SPAN}', '{WALDO_SPAN}', '{WALDO_REFUSAL_SPAN}')"

SAW_INSUFFICIENT_TOOLS = "SAW_INSUFFICIENT_TOOLS"
SAW_READY = "SAW_READY"
SKIPPED_MESSAGE_PREFIX = "skipped:"

ESCALATED = "escalated"
ANSWERED = "answered"
# Waldo ended without escalating or answering: a fallback to the full agent.
OTHER = "other"
SKIP_PREFIX = "skip:"
# Waldo spans exist but carry neither a skip nor a termination, e.g. the WaldoAgent run failed.
INCOMPLETE = "incomplete"
_RAN_OUTCOMES = frozenset({ESCALATED, ANSWERED, OTHER})

UNDER_ESCALATION = "under-escalation"
OVER_ESCALATION = "over-escalation"
MISSED_ANSWER = "missed answer"


# ---------------------------------------------------------------------------
# Slot 1: one entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenLabel:
    route: str
    # Reviewer notes from the label row, for reflection only.
    notes: str = ""

    @property
    def escalate(self) -> bool:
        return self.route == ESCALATE_ROUTE


@dataclass(frozen=True)
class WaldoDecision:
    """The student's Waldo outcome on one entry, read from its latest trace."""

    outcome: str
    termination: str = ""
    # The Waldo span's status message, e.g. ``termination=SAW_READY, loops=2, tool_calls=2``.
    summary: str = ""
    first_sentence_refusal: bool = False

    @property
    def ran(self) -> bool:
        return self.outcome in _RAN_OUTCOMES

    @property
    def escalated(self) -> bool:
        return self.outcome == ESCALATED

    @property
    def skip_reason(self) -> str:
        return self.outcome[len(SKIP_PREFIX) :] if self.outcome.startswith(SKIP_PREFIX) else ""


def classify_waldo_decision(
    agent_message: Any, termination: Any, *, summary: Any = "", first_sentence_refusal: Any = False
) -> WaldoDecision:
    """Classify one latest trace. A WaldoAgent ``skipped:*`` message wins over any termination."""
    message = str(agent_message or "").strip()
    code = str(termination or "").strip()
    if message.startswith(SKIPPED_MESSAGE_PREFIX):
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


def takes_golden_route(label: GoldenLabel, decision: WaldoDecision) -> bool:
    """Escalate where the label is discover, answer where it is direct_response. A fallback never matches."""
    return decision.escalated if label.escalate else decision.outcome == ANSWERED


def mismatch_direction(label: GoldenLabel, decision: WaldoDecision) -> str:
    """``""`` when the route matches; otherwise which way Waldo went wrong."""
    if takes_golden_route(label, decision):
        return ""
    if label.escalate:
        return UNDER_ESCALATION
    return OVER_ESCALATION if decision.escalated else MISSED_ANSWER


@dataclass(frozen=True)
class GoldenEscalationEntryMetrics:
    entry_id: str
    query: str
    label: GoldenLabel
    decision: WaldoDecision

    @property
    def golden_escalation_match(self) -> bool:
        return takes_golden_route(self.label, self.decision)

    @property
    def direction(self) -> str:
        return mismatch_direction(self.label, self.decision)

    @property
    def passed(self) -> bool:
        return self.golden_escalation_match

    @property
    def score(self) -> float:
        return 1.0 if self.golden_escalation_match else 0.0


@dataclass(frozen=True)
class GoldenCoverage:
    """How the run's Waldo entries split before scoring."""

    # Labeled entries of the run's eval set with no Waldo span: never ran, or ingest has not landed.
    labeled_without_waldo: int = 0
    # Entries with Waldo spans but no label (not in the set, no canonicalAnswer, no query match).
    unlabeled_entries: int = 0
    incomplete_entries: int = 0
    skip_reasons: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class GoldenEscalationMetrics:
    # Entries with any Waldo span. Zero while agentspan ingest has not landed.
    waldo_entries: int
    compared_entries: int
    golden_escalation_match: float
    under_escalations: int = 0
    over_escalations: int = 0
    missed_answers: int = 0
    # Scored entries that ended in a fallback (NO_SAMPLE, MAX_LOOPS_EXHAUSTED, NO_TOOL_CALLS).
    student_fallbacks: int = 0
    routes: Mapping[str, int] = field(default_factory=dict)
    coverage: GoldenCoverage = field(default_factory=GoldenCoverage)


class EvalRunGoldenEscalationAnalysis(RunAnalysis[GoldenEscalationMetrics, GoldenEscalationEntryMetrics]):
    """Agentspan analyses always carry a resolved shard window."""

    start_date: date
    end_date: date


# ---------------------------------------------------------------------------
# Labels: canonicalAnswer rows, joined on query text
# ---------------------------------------------------------------------------

EntryLabels = Mapping[str, tuple[str, GoldenLabel | None]]


def parse_golden_label(canonical: Any) -> GoldenLabel | None:
    """A ``canonicalAnswer`` JSON row with a known ``target_route``, else ``None``."""
    if isinstance(canonical, str):
        try:
            canonical = json.loads(canonical) if canonical.strip() else None
        except ValueError:
            return None
    if not isinstance(canonical, Mapping):
        return None
    route = str(canonical.get("target_route") or "").strip()
    if route not in KNOWN_ROUTES:
        return None
    notes = "; ".join(
        str(canonical[key]).strip() for key in _LABEL_NOTE_FIELDS if str(canonical.get(key) or "").strip()
    )
    return GoldenLabel(route=route, notes=notes[:_LABEL_NOTE_LIMIT])


def _entry_query(entry: Mapping[str, Any]) -> str:
    entry_input = entry.get("input")
    raw = entry.get("query") or (entry_input.get("query") if isinstance(entry_input, Mapping) else None)
    return str(raw).strip() if raw else ""


def entries_with_labels(entries: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, GoldenLabel | None]]:
    """``entry_id -> (query, own label)`` over listed eval-set entries."""
    resolved: dict[str, tuple[str, GoldenLabel | None]] = {}
    for entry in entries:
        entry_id = str(entry.get("id") or "")
        if not entry_id:
            continue
        expected = entry.get("expectedOutput")
        canonical = expected.get("canonicalAnswer") if isinstance(expected, Mapping) else None
        resolved[entry_id] = (_entry_query(entry), parse_golden_label(canonical))
    return resolved


def parse_label_eval_sets(raw: Any) -> list[tuple[str, str]]:
    """``objective.params.label_eval_sets``: a list of ``"name:version"`` strings."""
    if raw is None:
        return []
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise ValueError(f"label_eval_sets must be a list of 'name:version' strings, got {raw!r}")
    sets: list[tuple[str, str]] = []
    for item in raw:
        name, sep, version = str(item).rpartition(":")
        if not sep or not name or not version:
            raise ValueError(f"label_eval_sets entries must be 'name:version', got {item!r}")
        sets.append((name, version))
    return sets


# ---------------------------------------------------------------------------
# Slot 2: rows -> entry, entries -> aggregate
# ---------------------------------------------------------------------------


def decision_from_row(row: Mapping[str, Any]) -> WaldoDecision:
    return classify_waldo_decision(
        row.get("agent_message"),
        row.get("termination"),
        summary=row.get("waldo_summary"),
        first_sentence_refusal=row.get("first_sentence_refusal"),
    )


def summarize_coverage(decisions: Mapping[str, WaldoDecision], entry_labels: EntryLabels) -> GoldenCoverage:
    def label_of(entry_id: str) -> GoldenLabel | None:
        return entry_labels.get(entry_id, ("", None))[1]

    return GoldenCoverage(
        labeled_without_waldo=sum(
            1 for entry_id, (_q, label) in entry_labels.items() if label and entry_id not in decisions
        ),
        unlabeled_entries=sum(1 for entry_id in decisions if label_of(entry_id) is None),
        incomplete_entries=sum(1 for d in decisions.values() if d.outcome == INCOMPLETE),
        skip_reasons=dict(Counter(d.skip_reason for d in decisions.values() if d.skip_reason)),
    )


def aggregate_golden_escalation_metrics(
    per_entry: Mapping[str, GoldenEscalationEntryMetrics],
    *,
    waldo_entries: int | None = None,
    coverage: GoldenCoverage | None = None,
) -> GoldenEscalationMetrics:
    entries = list(per_entry.values())
    directions = Counter(m.direction for m in entries)
    return GoldenEscalationMetrics(
        waldo_entries=len(entries) if waldo_entries is None else waldo_entries,
        compared_entries=len(entries),
        golden_escalation_match=sum(m.score for m in entries) / len(entries) if entries else 0.0,
        under_escalations=directions[UNDER_ESCALATION],
        over_escalations=directions[OVER_ESCALATION],
        missed_answers=directions[MISSED_ANSWER],
        student_fallbacks=sum(1 for m in entries if m.decision.outcome == OTHER),
        routes=dict(Counter(m.label.route for m in entries)),
        coverage=coverage or GoldenCoverage(),
    )


def interleave_by_direction(
    per_entry: Mapping[str, GoldenEscalationEntryMetrics], entry_ids: Sequence[str]
) -> tuple[str, ...]:
    """Round-robin the failing ids across mismatch directions, largest group first.

    Single-model reflection keeps this order, so the rarer direction still reaches the reflector.
    """
    groups: dict[str, list[str]] = {}
    for entry_id in entry_ids:
        groups.setdefault(per_entry[entry_id].direction, []).append(entry_id)
    ranked = sorted(groups.values(), key=lambda ids: -len(ids))
    ordered: list[str] = []
    for depth in range(max((len(ids) for ids in ranked), default=0)):
        ordered.extend(ids[depth] for ids in ranked if depth < len(ids))
    return tuple(ordered)


# ---------------------------------------------------------------------------
# Slot 3: SQL
# ---------------------------------------------------------------------------


def _latest(column: str, span_name: str) -> str:
    return (
        f"ARRAY_AGG(IF(s.span_name = '{span_name}', s.{column}, NULL) IGNORE NULLS "
        "ORDER BY s.start_ms DESC LIMIT 1)[SAFE_OFFSET(0)]"
    )


def build_golden_escalation_per_entry_query(*, agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE) -> str:
    """One row per entry with the student's Waldo outcome from its latest trace.

    An entry can have several traces (retries); only the trace holding the entry's latest Waldo
    span is read, which drops failed attempts. Within that trace the latest WaldoAgent message and
    Waldo termination win.
    """
    return f"""
WITH waldo_spans AS (
  SELECT
    {EVAL_ENTRY_ID_EXPR} AS entry_id,
    jsonPayload.context.agent_trace.trace_id AS trace_id,
    jsonPayload.span_info.span_name AS span_name,
    jsonPayload.span_info.execution_status.message AS message,
    jsonPayload.ic_preloop.termination AS termination,
    SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64) AS start_ms
  FROM `{agentspan_table}`
  WHERE {wildcard_shard_filter("start_date", "end_date")}
    AND jsonPayload.context.eval.eval_id = @eval_id
    AND {WALDO_SPAN_FILTER}
),
latest_trace AS (
  SELECT
    entry_id,
    ARRAY_AGG(trace_id ORDER BY start_ms DESC LIMIT 1)[OFFSET(0)] AS trace_id
  FROM waldo_spans
  WHERE entry_id IS NOT NULL AND trace_id IS NOT NULL
  GROUP BY entry_id
)
SELECT
  s.entry_id,
  ANY_VALUE(s.trace_id) AS trace_id,
  {_latest("message", WALDO_AGENT_SPAN)} AS agent_message,
  {_latest("termination", WALDO_SPAN)} AS termination,
  {_latest("message", WALDO_SPAN)} AS waldo_summary,
  LOGICAL_OR(s.span_name = '{WALDO_REFUSAL_SPAN}') AS first_sentence_refusal
FROM waldo_spans AS s
JOIN latest_trace AS t
  ON s.entry_id = t.entry_id AND s.trace_id = t.trace_id
GROUP BY s.entry_id
ORDER BY s.entry_id
""".strip()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def empty_golden_escalation_analysis(
    eval_id: str, *, lookback_days: int = DEFAULT_LOOKBACK_DAYS, end_date: date | None = None
) -> EvalRunGoldenEscalationAnalysis:
    start_date, resolved_end = default_date_range(lookback_days=lookback_days, end_date=end_date)
    return EvalRunGoldenEscalationAnalysis(
        eval_ids=(eval_id,),
        aggregate=aggregate_golden_escalation_metrics({}, waldo_entries=0),
        start_date=start_date,
        end_date=resolved_end,
    )


def fetch_eval_run_golden_escalation_analysis(
    client: Any,
    *,
    eval_id: str,
    entry_labels: EntryLabels,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: date | None = None,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> EvalRunGoldenEscalationAnalysis:
    """Score the student's Waldo decision on every labeled entry where Waldo ran."""
    seen: dict[str, Any] = {}

    def keep_labeled_and_ran(rows: Rows) -> Rows:
        decisions = {str(row.get("entry_id") or ""): decision_from_row(row) for row in rows}
        decisions.pop("", None)
        seen["waldo_entries"] = len(decisions)
        seen["coverage"] = summarize_coverage(decisions, entry_labels)
        return [
            row
            for row in rows
            if (entry_id := str(row.get("entry_id") or ""))
            and decisions[entry_id].ran
            and entry_labels.get(entry_id, ("", None))[1] is not None
        ]

    def parse(row: Mapping[str, Any]) -> GoldenEscalationEntryMetrics | None:
        entry_id = str(row.get("entry_id") or "")
        query, label = entry_labels.get(entry_id, ("", None))
        if not entry_id or label is None:
            return None
        return GoldenEscalationEntryMetrics(
            entry_id=entry_id, query=query, label=label, decision=decision_from_row(row)
        )

    analysis = fetch_agentspan_analysis(
        client,
        eval_ids=(eval_id,),
        bounds_sql=bounds_query(
            eval_id_predicate="= @eval_id", span_filter=WALDO_SPAN_FILTER, agentspan_table=agentspan_table
        ),
        per_entry_sql=build_golden_escalation_per_entry_query(agentspan_table=agentspan_table),
        parse_row=parse,
        aggregate=lambda per_entry, _dropped: aggregate_golden_escalation_metrics(
            per_entry, waldo_entries=seen.get("waldo_entries", 0), coverage=seen.get("coverage")
        ),
        filter_rows=keep_labeled_and_ran,
        lookback_days=lookback_days,
        end_date=end_date,
    )
    if analysis.start_date is None:
        return empty_golden_escalation_analysis(eval_id, lookback_days=lookback_days, end_date=end_date)
    return EvalRunGoldenEscalationAnalysis(
        eval_ids=analysis.eval_ids,
        aggregate=analysis.aggregate,
        per_entry=analysis.per_entry,
        high_signal_entry_ids=interleave_by_direction(analysis.per_entry, analysis.high_signal_entry_ids),
        start_date=analysis.start_date,
        end_date=analysis.end_date or analysis.start_date,
    )


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

SCORE_MODEL_NOTE = (
    "HOW THE SCORE IS DECIDED. Waldo either answers with its attached tools or escalates to the full "
    "agent by calling discover, which ends it with SAW_INSUFFICIENT_TOOLS. Every request has a "
    "human-reviewed golden route. An entry scores 1 when Waldo takes it: it escalates where the label is "
    "discover, and answers (SAW_READY) where the label is direct_response. Ending any other way "
    "(NO_SAMPLE timeouts, NO_TOOL_CALLS, or running out of loops) is a fallback, not a decision, so it "
    "scores 0 whichever the label, even though the request reaches the full agent. Entries where Waldo "
    "was skipped are not scored. Answer quality and which attached tool was used do not count. Loop "
    "count counts only through the budget: Waldo has 3 turns, so a rule that keeps it searching past "
    "that turns an answer or an escalation into a fallback."
)

EDIT_EFFECTIVENESS_NOTE = (
    "WHAT KIND OF EDIT WORKS. Mismatches come in three kinds: under-escalation (labeled discover, Waldo "
    "did not escalate), over-escalation (labeled direct_response, Waldo escalated), and missed answer "
    "(labeled direct_response, Waldo fell back without escalating). Tally them separately. A rule "
    "changes the decision only if its trigger is visible in the request before any tool runs: the kind "
    "of lookup asked for, whether it needs reasoning across several sources, whether it asks for an "
    "action, analysis, or artifact the attached tools cannot produce. State the trigger and the "
    "decision together, such as 'when the request asks for X, call discover rather than searching'. "
    "Blanket policy ('escalate when unsure', 'prefer answering') moves both directions at once and "
    "breaks entries that already match. When Waldo ran several searches and then answered or ran out "
    "of loops on a discover request, it judged the attached tools sufficient: describe the request "
    "class where they are not, and say to escalate before searching rather than after."
)

ANALYSIS_PROCEDURE_NOTE = (
    "HOW TO READ EACH EXAMPLE. Read FEEDBACK for the direction of the mismatch, how Waldo ended "
    "(including its loop and tool-call counts), and the reviewer's note when there is one. Then read the "
    "user query for the cue that decides the route. Classify each example by mechanism: answered from "
    "attached tools when the request needed more, kept searching until loops or time ran out instead "
    "of escalating, or escalated a simple lookup or rewrite the attached tools cover. State the tally, "
    "with counts, in your diagnosis. A rule needs at least two examples showing the same mechanism."
)

GOLDEN_ESCALATION_GAP_ANALYSIS_GUIDE = "\n\n".join([SCORE_MODEL_NOTE, EDIT_EFFECTIVENESS_NOTE, ANALYSIS_PROCEDURE_NOTE])

ESCALATION_GOAL = (
    "Rewrite it so Waldo escalates on the kinds of request labeled discover, and answers the kinds "
    "labeled direct_response. State rules in terms of request type, never specific documents, people, "
    "or customers from the examples. Keep the text operational and concise."
)

ESCALATION_TASK = (
    "This section decides when Waldo calls discover (escalate to the full agent) versus using the "
    f"attached search tools and answering directly. {ESCALATION_GOAL}"
)

RESTATED_ROUTING_TASK = (
    "These lines restate the routing decision after the routing core, and Waldo obeys whichever "
    "restatement is strictest: a line here that sends a request to discover overrides an easy-lookup "
    "rule in the routing core. Loosen or tighten them together with what the evidence shows, rather "
    f"than adding exceptions elsewhere. {ESCALATION_GOAL}"
)

WALDO_ROUTING_FRAME = (
    "You are rewriting the routing core at the top of the Waldo router prompt: '### Role & "
    "Capabilities' and '### Routing', which lists the requests that go straight to discover (written "
    "`[[waldo_discover_tool_name]]`) and the easy lookups Waldo answers with its attached tools. It has "
    "conditional branches, <<<[[has_search_tools]] ... >>> and <<<[[no_search_tools]] ... >>>; scio "
    "renders exactly one per request. Keep both branches and every [[...]] placeholder. Role & "
    "Capabilities and Routing state the same decision; when you change one, change the other so the "
    "prompt does not contradict itself. Sections after this one are edited separately and refer back to "
    "'the easy lookups above': keep the term 'easy lookup'."
)

WALDO_SYSTEM_FRAME = (
    "You are rewriting the Waldo router template. The routing core and the tool-usage section are "
    "separate modules spliced in at their slots. The escalation decision is restated here in the "
    "'### Available Tools' paragraph, which says when to write the final answer and to call discover "
    "when the attached tools cannot complete the task: that is the line that moves escalation. Keep the "
    "term 'easy lookup', which the routing core defines. Response Guidelines, Hallucination Prevention, "
    "Confidentiality, and User Information shape the answer, not the escalation decision: leave them "
    "unchanged."
)

WALDO_TOOL_USAGE_FRAME = (
    "You are rewriting the Waldo tool-usage section, spliced into the template under '### Tool Usage "
    "Guidelines'. Every line here moves escalation: which requests the attached tools are for, how long "
    "to keep searching before handing off ('Be persistent on info-seeking lookups'), when one or two "
    "searches may come before discover, and discover for unsupported filters or tools. It has "
    "conditional branches, <<<[[has_search_tools]] ... >>> and <<<[[no_search_tools]] ... >>>; keep "
    "both, every [[...]] placeholder, and the term 'easy lookup'."
)


LABELS_ARE_OFFLINE_RULE = (
    "The golden labels and reviewer notes are an offline scoring reference, not something Waldo can "
    "see at runtime. Never mention labels, reviewers, or golden routes in the prompt text. Instead, "
    "work out which feature of the request decides its route and state that as a standalone rule "
    "Waldo can follow using only the user's request and its own tool results."
)


def _golden_responsibility(frame: str, task: str) -> str:
    return compose_responsibility(
        frame,
        task,
        guide=GOLDEN_ESCALATION_GAP_ANALYSIS_GUIDE,
        closing=f"{NO_EXAMPLE_SPECIFICS_RULE} {LABELS_ARE_OFFLINE_RULE}",
    )


WALDO_ROUTING_RESPONSIBILITY = _golden_responsibility(WALDO_ROUTING_FRAME, ESCALATION_TASK)
WALDO_SYSTEM_RESPONSIBILITY = _golden_responsibility(WALDO_SYSTEM_FRAME, RESTATED_ROUTING_TASK)
WALDO_TOOL_USAGE_RESPONSIBILITY = _golden_responsibility(WALDO_TOOL_USAGE_FRAME, RESTATED_ROUTING_TASK)


def _outcome_phrase(termination: str) -> str:
    if termination == SAW_INSUFFICIENT_TOOLS:
        return "escalated to the full agent (SAW_INSUFFICIENT_TOOLS)"
    if termination == SAW_READY:
        return "answered with its attached tools (SAW_READY)"
    return (
        f"did not escalate and ended with {termination or 'no termination'}, reaching the full agent only as a fallback"
    )


def golden_feedback(output: Mapping[str, Any]) -> str:
    """One mismatch sentence, Waldo's span summary, and the reviewer's note."""
    route = str(output.get("golden_route") or "")
    if route not in KNOWN_ROUTES:
        return ""
    label = GoldenLabel(route=route)
    termination = str(output.get("student_waldo_termination") or "")
    decision = classify_waldo_decision("", termination)
    direction = mismatch_direction(label, decision)
    if not direction:
        return f"Matches the golden route ({route})."
    parts = [f"{direction.capitalize()}: the golden route is {route} and Waldo {_outcome_phrase(termination)}."]
    summary = str(output.get("student_waldo_summary") or "")
    refusal = (
        " The escalation came from a first-sentence refusal." if output.get("student_first_sentence_refusal") else ""
    )
    if summary or refusal:
        parts.append(f"Waldo: {summary or '(no summary)'}.{refusal}")
    notes = str(output.get("golden_notes") or "")
    if notes:
        parts.append(f"Reviewer note: {notes}")
    return " ".join(parts)


class GoldenEscalationMatchObjective(SingleModelObjective[EvalRunGoldenEscalationAnalysis]):
    """Score whether one model's Waldo takes each entry's human-labeled route."""

    name = GOLDEN_ESCALATION_MATCH_OBJECTIVE
    telemetry_dimensions = (GOLDEN_ESCALATION_MATCH_OBJECTIVE,)
    focused_bucket_type = QUERY_CANONICAL_BUCKET_TYPE
    failure_label = "HIGH-SIGNAL FAILURES (Waldo route vs golden label)"
    pending_telemetry_label = "Waldo decision"
    # ``waldo_entries`` is 0 until the run's Waldo spans have landed.
    pending_count = "waldo_entries"
    module_responsibilities: ClassVar[Mapping[str, str]] = {
        WALDO_ROUTING_KEY: WALDO_ROUTING_RESPONSIBILITY,
        WALDO_SYSTEM_KEY: WALDO_SYSTEM_RESPONSIBILITY,
        WALDO_TOOL_USAGE_KEY: WALDO_TOOL_USAGE_RESPONSIBILITY,
    }

    def __init__(self, *, bigquery_client: Any | None = None, lookback_days: int = 1):
        if bigquery_client is None:
            raise ValueError("bigquery_client is required")
        self.bigquery_client = bigquery_client
        self.lookback_days = lookback_days
        self.params: dict[str, Any] = {}
        self._eval_analysis_cache: dict[str, EvalRunGoldenEscalationAnalysis] = {}
        self._entries_by_set: dict[tuple[str, str], dict[str, tuple[str, GoldenLabel | None]]] = {}
        self._labels_by_query: dict[str, GoldenLabel] = {}
        self._label_sets_loaded = False

    # --- labels -------------------------------------------------------------

    def _index_labels(self, entries: Mapping[str, tuple[str, GoldenLabel | None]], *, source: str) -> None:
        for query, label in entries.values():
            if not query or label is None:
                continue
            prior = self._labels_by_query.setdefault(query, label)
            if prior.route != label.route:
                raise ValueError(
                    f"Conflicting golden routes for one query ({prior.route} vs {label.route} in {source}): {query[:120]!r}"
                )

    def _list_set(
        self, evalcli: Any, name: str, version: str, deployment_ids: Sequence[str]
    ) -> dict[str, tuple[str, GoldenLabel | None]]:
        key = (name, version)
        if key not in self._entries_by_set:
            listed = entries_with_labels(
                evalcli.list_eval_set_entries(
                    eval_set_name=name, eval_set_version=version, deployment_ids=list(deployment_ids)
                )
            )
            labeled = sum(1 for _query, label in listed.values() if label is not None)
            print(
                f"[Golden Escalation] Listed {name}:{version}: {len(listed)} entries, {labeled} with a canonicalAnswer route"
            )
            self._index_labels(listed, source=f"{name}:{version}")
            self._entries_by_set[key] = listed
        return self._entries_by_set[key]

    def entry_labels(self, request: AnalysisRequest) -> dict[str, tuple[str, GoldenLabel | None]]:
        """``entry_id -> (query, label)`` for the run's eval set; own label first, then by query."""
        if not request.eval_set_name or not request.eval_set_version:
            raise ValueError(f"{type(self).__name__} needs the run's eval set in AnalysisRequest to join labels")
        if request.evalcli is None:
            raise ValueError(f"{type(self).__name__} needs an EvalCLI client to list eval-set entries")
        if not self._label_sets_loaded:
            for name, version in parse_label_eval_sets(self.experiment_param("label_eval_sets", None)):
                self._list_set(request.evalcli, name, version, request.deployment_ids)
            self._label_sets_loaded = True
        listed = self._list_set(
            request.evalcli, request.eval_set_name, request.eval_set_version, request.deployment_ids
        )
        resolved = {
            entry_id: (query, own or self._labels_by_query.get(query)) for entry_id, (query, own) in listed.items()
        }
        if listed and not any(label for _query, label in resolved.values()):
            raise ValueError(
                f"No entry of {request.eval_set_name}:{request.eval_set_version} has a golden route: set "
                "expectedOutput.canonicalAnswer on the entries or list the labeled sets in "
                "objective.params.label_eval_sets."
            )
        return resolved

    # --- scoring -----------------------------------------------------------

    def analyze(self, eval_id: str, *, request: AnalysisRequest) -> EvalRunGoldenEscalationAnalysis:
        """One fetch shape: per-entry decisions are always read, since the aggregate is their mean."""

        def fetch(req: AnalysisRequest) -> EvalRunGoldenEscalationAnalysis:
            return fetch_eval_run_golden_escalation_analysis(
                self.bigquery_client,
                eval_id=eval_id,
                entry_labels=self.entry_labels(req),
                lookback_days=self.lookback_days,
            )

        return self.cached_eval_analysis(eval_id, request=request, fetch=fetch, label="golden escalation analysis")

    def focused_pass_rate(self, analysis: EvalRunGoldenEscalationAnalysis, requested_entry_ids: Sequence[str]) -> float:
        """Share of the requested entries where Waldo now takes the golden route. Unscored entries count 0."""
        if not requested_entry_ids:
            return 0.0
        matching = sum(
            1
            for entry_id in requested_entry_ids
            if (m := analysis.per_entry.get(entry_id)) and m.golden_escalation_match
        )
        missing = sum(1 for entry_id in requested_entry_ids if entry_id not in analysis.per_entry)
        print(
            f"[Golden Escalation] Screen: {matching}/{len(requested_entry_ids)} now take the golden route; "
            f"{missing} have no scored Waldo decision"
        )
        return matching / len(requested_entry_ids)

    def log_analysis(self, analysis: EvalRunGoldenEscalationAnalysis) -> None:
        aggregate = analysis.aggregate
        coverage = aggregate.coverage
        print(
            f"[Golden Escalation] {aggregate.waldo_entries} entries with Waldo spans, {aggregate.compared_entries} "
            f"scored (routes {dict(aggregate.routes)}); {coverage.unlabeled_entries} unlabeled, "
            f"{coverage.incomplete_entries} incomplete, skips {dict(coverage.skip_reasons)}, "
            f"{coverage.labeled_without_waldo} labeled entries without Waldo spans"
        )
        log_analysis(
            analysis,
            label="Golden Escalation",
            headline=(
                f"match={aggregate.golden_escalation_match:.1%}; under-escalated {aggregate.under_escalations}, "
                f"over-escalated {aggregate.over_escalations}, missed answers {aggregate.missed_answers} "
                f"({aggregate.student_fallbacks} fallbacks)"
            ),
            entry_line=lambda m: f"golden={m.label.route} waldo={m.decision.termination or m.decision.outcome}",
        )

    def _output(self, ctx: ScoringContext, *, entry_id: str, query: str) -> SingleModelALRolloutOutput:
        return {
            "deployment_id": ctx.deployment_id,
            "query": query,
            "entry_id": entry_id,
            "student_tool_calls": 0,
            "student_tool_errors": 0,
            "shell_error_messages": [],
            "student_eval_run_id": ctx.student_eval_id,
        }

    def aggregate_row(self, analysis: EvalRunGoldenEscalationAnalysis, ctx: ScoringContext) -> ScoredRow:
        return ScoredRow(
            entry_id=None,
            dimension_scores={self.name: analysis.aggregate.golden_escalation_match},
            output=self._output(ctx, entry_id=ctx.query, query=ctx.query),
        )

    def entry_row(
        self,
        entry_id: str,
        metrics: GoldenEscalationEntryMetrics,
        analysis: EvalRunGoldenEscalationAnalysis,
        ctx: ScoringContext,
    ) -> ScoredRow:
        del analysis
        output = self._output(ctx, entry_id=entry_id, query=metrics.query or ctx.entry_query(entry_id))
        output["golden_route"] = metrics.label.route
        output["golden_notes"] = metrics.label.notes
        output["student_waldo_termination"] = metrics.decision.termination
        output["student_waldo_summary"] = metrics.decision.summary
        output["student_first_sentence_refusal"] = metrics.decision.first_sentence_refusal
        return ScoredRow(
            entry_id=entry_id,
            dimension_scores={self.name: metrics.score},
            output=output,
            data_overrides={"eval_entry_id": entry_id, "eval_run_id": ctx.student_eval_id},
        )

    # --- reflection --------------------------------------------------------

    def failure_pattern(self, component_name: str, trajectory: SingleModelALTrajectory) -> tuple[Any, ...]:
        del component_name
        output = trajectory["output"]
        route = str(output.get("golden_route") or "")
        termination = str(output.get("student_waldo_termination") or "")
        if route not in KNOWN_ROUTES:
            return ()
        return (mismatch_direction(GoldenLabel(route=route), classify_waldo_decision("", termination)), termination)

    def build_reflective_example(
        self,
        component_name: str,
        trajectory: SingleModelALTrajectory,
        candidate: dict[str, str],
    ) -> ReflectiveExample:
        del component_name, candidate
        output = trajectory["output"]
        return self.reflective_example(
            trajectory,
            feedback=golden_feedback(output) or "No golden route for this entry.",
            generated={
                "student_answer": output.get("student_waldo_summary") or output.get("student_waldo_termination", "")
            },
        )


__all__ = ["GoldenEscalationMatchObjective"]
