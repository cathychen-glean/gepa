"""Agentic-preference objective: pairwise judge preference vs the teacher as the primary teacher-student metric.

Trace loading and judge-rationale parsing live in
:mod:`glean_gepa.objectives.agentic_preference_traces`. This module maps that
analysis onto the contract in :mod:`glean_gepa.objectives.protocol`.

There is no SQL of its own and no aggregate on the frame: the score is
computed from judge output by the adapter, not here."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from glean_gepa.adapter_types import TeacherStudentALTrajectory, paired_rollout_output
from glean_gepa.al_adapter import ReflectiveExample
from glean_gepa.focused_evalset import QUERY_CANONICAL_BUCKET_TYPE
from glean_gepa.judge_metrics_util import CUSTOMER_AGENTIC_PREFERENCE_METRIC, PREFERENCE_TIE
from glean_gepa.objectives.agentic_preference_traces import (
    AgenticPreferenceAnalysis,
    AgenticPreferenceEntry,
    decisive_dimensions,
    empty_agentic_preference_analysis,
    fetch_paired_preference_traces,
    fetch_preference_rationales,
    student_behavior_flags,
)
from glean_gepa.objectives.base import AnalysisRequest, ScoredRow, ScoringContext, TeacherStudentObjective
from glean_gepa.objectives.tool_match import TOOL_INPUT_LIMIT, tool_input_evidence
from glean_gepa.objectives.utils.action_input_trace import build_trace_locator, fetch_named_tool_inputs_by_entry
from glean_gepa.objectives.utils.tool_names import SKIPPED_TOOL_NAMES
from glean_gepa.prompt_constants import (
    CORE_TOOLS,
    EXECUTION_DISCIPLINE_KEY,
    RULES_EXT_KEY,
    WRITING_CODE_KEY,
)
from glean_gepa.reflection_prompts import (
    EXECUTION_DISCIPLINE_FRAME,
    GENERALITY_RULES,
    RULES_EXT_FRAME,
    WRITING_CODE_FRAME,
    compose_responsibility,
    core_tool_frame,
)

AGENTIC_PREFERENCE_OBJECTIVE = CUSTOMER_AGENTIC_PREFERENCE_METRIC

TEACHER_PREFERRED_KEY = ("teacher_preferred", "")
STUDENT_PREFERRED_KEY = ("student_preferred", "keep")
REFLECTION_RANDOM_SAMPLE_SIZE = 25
REFLECTION_SAMPLE_SEED = 0
# Cortex overall 6/10. Ties at 0.5 are excluded so keep examples are decisive wins.
KEEP_PREFERENCE_THRESHOLD = 0.6
KEEP_SAMPLE_SIZE = 15

KEEP_PRESERVE_RULE = (
    "Some supplied examples are student-preferred KEEP cases "
    f"(preference >= {KEEP_PREFERENCE_THRESHOLD}): the student already beat the teacher there. "
    "Read their FEEDBACK to see which dimension the student won and why (usually: it delivered "
    "concisely and conservatively, stayed on the requested scope, or delivered when the other run "
    "asked or stalled). Any rule you add must leave that behavior intact: do not push the student "
    "toward longer answers, extra sections, tangents, hedges, or clarifying questions."
)

# How the pairwise agentic judge decides, so the rewriter targets what actually moves the verdict.
JUDGE_MODEL_NOTE = (
    "HOW THE JUDGE DECIDES. The pairwise agentic judge scores three dimensions and synthesizes them "
    "in a fixed order. (1) task_completion decides first: did the run deliver every action the user "
    "asked for, in the requested form? Asking a clarifying question, offering a next step ('want me "
    "to dig deeper / pull the full contents?'), or answering a narrower question than the one asked "
    "counts as not completing the task, and a delivered-but-imperfect result beats an undelivered one. "
    "(2) correctness overrides when the outcome-central claim is wrong: the judge has tools and "
    "re-verifies specific claims (numbers, dates, names, titles, statuses, steps, mechanisms) against "
    "the retrieved sources, so a specific stated from memory that no source supports is caught and "
    "loses even when the rest of the answer is fine; hedging an unsupported specific does not save it, "
    "omitting it does. Tool or permission failures are not penalized; asserting something the trace "
    "did not establish is. (3) output_readiness breaks ties: is the message forwardable as-is — no "
    "preamble ('Here's...', 'Great question'), no closing offers, no unfilled placeholders, no raw "
    "tags or citation markup, professional register, focused on what was asked, and for artifact "
    "tasks does the file actually contain the claimed edit. Readiness is weighted higher for "
    "artifact tasks. Over-long answers with unrelated extras lose readiness too, so concision is an "
    "asset to protect, not a defect to fix."
)

# What kinds of prompt edits have changed student behavior in past rounds, and what has not.
EDIT_EFFECTIVENESS_NOTE = (
    "WHAT KIND OF EDIT WORKS. Rules with a chance of changing behavior are mechanical and triggered "
    "by a condition the student can observe at the moment of acting: 'when the question names a "
    "specific person, account, system or ticket and you only have snippets, run your own search and "
    "open the top document before answering'; 'if you are about to write \"would you like me to…\" "
    'or "could you clarify…", pick the most plausible target from context, do that step, and state '
    "the assumption in one line'; 'before stating a number, date, name or title, find it in a "
    "retrieved result or leave it out'; 'after editing a file, print the changed region and report "
    "only what you saw'. Rules that did NOT change behavior were abstract policy that requires the "
    "student to judge its own perception ('verify when evidence is indirect', 'consider whether the "
    "entity matches', 'be thorough'). Measured so far: rewording a tool description without "
    "changing how often the student invokes that tool did not move tool usage or the preference "
    "rate (reader and search descriptions were rewritten five ways; invocation counts stayed flat "
    "and paired deltas sat inside judge noise), and rewriting the ask_user_questions description "
    "changed nothing because the student asks in prose. A rule only works if it lives in the text "
    "the student reads at the moment the behavior happens: Execution Discipline for what the final "
    "message may contain and when to ask, a tool description only for when to call that tool. "
    "Instructions that push the opposite way cancel new rules: if the current text tells the student "
    "to minimize tool loops, cap searches, not search for confirmation, or to ask about audience, "
    "tone, depth or format, remove or invert that wording rather than appending a counter-rule "
    "beside it. Each rewrite should address one or two mechanisms, name the trigger and the action, "
    "and stay short; long additions dilute the module and have not moved the preference rate. Do not "
    "add rules about anything that appears in fewer than three LOSS examples, and do not add a rule "
    "that a KEEP example shows the student already getting right."
)

# Procedure the reflector should follow before proposing text.
ANALYSIS_PROCEDURE_NOTE = (
    "HOW TO READ EACH EXAMPLE. First read FEEDBACK: the preference score, the DECIDED BY line "
    "(which dimension carried the verdict and by how much), the judge's per-dimension explanation, "
    "the verified_claims lines (which specific claims were unsupported and on which side), the "
    "'user requested' vs 'Student … / Teacher …' action lines, and the STUDENT BEHAVIOR FLAGS. Then "
    "compare STUDENT_TOOLS and TEACHER_TOOLS with the ACTION_INPUT arguments to see which retrieval "
    "or writing step the preferred run took that the student skipped, and whether the student had "
    "the information it needed but stopped early. Then read STUDENT_ANSWER for the moment it stopped "
    "short: an offer, a question, a hedge, a generic list where a specific was needed, a preamble. "
    "Only then name the behavior that explains the loss. Before proposing, tally those behaviors "
    "across all LOSS examples and check the KEEP examples for anything your rule would break. State "
    "the tally in your diagnosis."
)

AGENTIC_GAP_ANALYSIS_GUIDE = "\n\n".join([JUDGE_MODEL_NOTE, EDIT_EFFECTIVENESS_NOTE, ANALYSIS_PROCEDURE_NOTE])

WRITING_CODE_RESPONSIBILITY = compose_responsibility(
    WRITING_CODE_FRAME,
    "Rewrite those coding instructions so the student produces the requested deliverable in the "
    "sandbox (Write or Edit a file, run the skill, print then extract) instead of answering from "
    "snippets, truncated ToolResults, or memory, and so that it re-reads the changed region of a file "
    "before reporting an edit and never leaves placeholder text in a deliverable. Do not add yield, "
    "ask, retry, or search-budget rules — those belong in Execution Discipline. Do not rewrite a named "
    "native tool's description. Leave response formatting and citation mechanics to other modules.",
    guide=AGENTIC_GAP_ANALYSIS_GUIDE,
    closing=f"{KEEP_PRESERVE_RULE} {GENERALITY_RULES} Propose minimal deltas.",
)

RULES_EXT_RESPONSIBILITY = compose_responsibility(
    RULES_EXT_FRAME,
    "These bullets govern how the student works inside the sandbox on artifact and coding tasks: "
    "after writing or editing a file, print the changed region and report only what is actually in "
    "the file; on a follow-up turn edit the existing output file instead of creating a new one; fill "
    "every placeholder from context or drop the line; keep raw tool output and markup out of "
    "deliverables. Do not restate Execution Discipline (when to yield, ask, retry, or stop) or a "
    "core-tool description.",
    guide=AGENTIC_GAP_ANALYSIS_GUIDE,
    closing=f"{KEEP_PRESERVE_RULE} Keep each bullet operational, checkable, and concise. {GENERALITY_RULES}",
)

EXECUTION_DISCIPLINE_RESPONSIBILITY = compose_responsibility(
    EXECUTION_DISCIPLINE_FRAME,
    "That covers whether to run its own retrieval or answer from what is already in context, whether "
    "to open a document or stop at snippets, whether to ask or to assume and deliver, whether to "
    "retry after an empty result and how, and what the final message may contain. Concretely: when a question "
    "names a specific entity and the student only has preloaded snippets, it should run its own "
    "search and open the primary document before answering, not offer to; when a detail it is "
    "about to state is not in a retrieved result, it should leave the detail out rather than hedge; "
    "when a request is ambiguous only in shaping (tone, depth, format, audience, which of several "
    "plausible targets), it should choose from context, deliver, and state the choice in one line; "
    "it should ask only when the target of the action is missing and one search does not find it; "
    "it should never end with an offer to do a step it could do now; the message should open with "
    "the answer and carry no preamble or closing offer. The student asks in prose in its final "
    "message far more often than through ask_user_questions, so this module — not that tool's "
    "description — is where the ask-or-deliver rule takes effect. If the current text minimizes "
    "tool loops, caps searches, discourages confirming searches, or tells the student to ask when a "
    "shaping choice (audience, tone, depth, format) is missing, remove or invert that wording — "
    "appending a counter-rule beside it does not work. Preserve factual search-first; do not add a hard stop "
    "after a fixed number of searches; do not tell the student to reason without tools as a "
    "default, or to keep searching until it is sure. This module governs effort, stopping "
    "conditions, and the final handoff only: leave SDK syntax, the Writing Code **Rules:** list, and "
    "named-tool descriptions to other modules.",
    guide=AGENTIC_GAP_ANALYSIS_GUIDE,
    closing=f"{KEEP_PRESERVE_RULE} {GENERALITY_RULES}",
)

# Tool-specific guidance: which gap mechanism each core tool's description can fix.
_TOOL_GAP_NOTES: Mapping[str, str] = {
    "ask_user_questions": (
        "This description governs only the entries where the student actually calls the tool, "
        "which is rare (about 2% of entries); most asking is prose in the final message "
        "and belongs to Execution Discipline, so only edit this if the supplied examples show the "
        "tool itself being called for a shaping choice. When they do: the tool should be used only "
        "when the request cannot be acted on at all — the target of the action (recipient, meeting, "
        "document, account, ticket) is missing and neither context nor one search identifies it — "
        "and not for tone, audience, depth or format, not after a search already produced a usable "
        "target, and never as a closing menu of options after an answer. Keep the call syntax and "
        "the note that answers arrive next turn."
    ),
    "glean_document_reader": (
        "The student stops at search snippets while the "
        "preferred run opens the one document that answers the question. Say when to open the "
        "primary document the snippets point to (the question needs a list, steps, numbers, dates, "
        "owners, settings, or the exact wording) and that several partial snippets are not a "
        "substitute for reading it. Keep the existing raw-bytes and multi-URL guidance intact."
    ),
    "glean_search": (
        "The student answers from "
        "preloaded context instead of running its own search when the question names a specific "
        "person, account, customer, system, process, ticket, or document. Say that preloaded "
        "results and earlier turns locate the answer but are not the answer, and that after an empty "
        "result the retry should drop filters or broaden, not reword the same filtered query. Also "
        "guard entity substitution: results are evidence only for the entity actually named."
    ),
    "glean_container_lister": (
        "Use this description to make the student enumerate a container (folder, channel, project) "
        "when the request is about its contents, instead of guessing from a search snippet."
    ),
    "tool_search": (
        "Use this description to make the student find the action tool that performs the requested "
        "step (send, create, update) instead of describing the step or asking whether to do it."
    ),
    "todo_write": (
        "Use this description so a multi-part request is tracked and every part is delivered before "
        "yielding; an omitted part is a task-completion loss."
    ),
    "delegate": (
        "Use this description so delegation is a way to finish a multi-part deliverable, not a way "
        "to stop before the result exists."
    ),
    "discover": (
        "Use this description so discovery is followed by the action it enables; discovering "
        "instead of acting is a task-completion loss."
    ),
}


def core_tool_responsibility(tool_name: str) -> str:
    """Reflection instructions for one core-tool ``schema.description``."""
    body = (
        "A description acts only at the moment the student decides whether to call this tool, so the "
        "edit must change that decision: state the trigger condition and the action in the description "
        "itself, and do not spend the rewrite on synonyms for text the student already follows. Rewrite "
        "it so the student uses this tool when it is the step "
        "that obtains or produces the requested deliverable, and does not use it as a substitute for "
        "that step (searching again, asking instead of drafting, or discovering instead of acting). A "
        "tool-sequence difference is extra evidence only when it caused an unfinished or thinner "
        "deliverable; do not rewrite merely to copy the preferred run's tool order. Do not add yield, "
        "search-budget, or stopping-condition rules — those belong in Execution Discipline."
    )
    if specific := _TOOL_GAP_NOTES.get(tool_name, ""):
        body = f"{body} {specific}"
    return compose_responsibility(
        core_tool_frame(tool_name),
        body,
        guide=AGENTIC_GAP_ANALYSIS_GUIDE,
        closing=f"{KEEP_PRESERVE_RULE} Keep the text operational and concise. {GENERALITY_RULES}",
    )


def _rollout_output(
    *,
    entry_id: str,
    deployment_id: str,
    query: str,
    student_answer: str = "",
    teacher_answer: str = "",
    student_tools: Sequence[str] = (),
    teacher_tools: Sequence[str] = (),
    student_tool_inputs: Sequence[Sequence[str]] = (),
    teacher_tool_inputs: Sequence[Sequence[str]] = (),
    student_eval_run_id: str = "",
    teacher_eval_run_id: str = "",
    trace: AgenticPreferenceEntry | None = None,
):
    output = paired_rollout_output(
        deployment_id=deployment_id,
        query=query,
        entry_id=entry_id,
        student_answer=student_answer,
        teacher_answer=teacher_answer,
        student_tool_events=student_tools,
        teacher_tool_events=teacher_tools,
        student_eval_run_id=student_eval_run_id,
        teacher_eval_run_id=teacher_eval_run_id,
    )
    if student_tool_inputs:
        output["student_tool_inputs"] = [list(pair) for pair in student_tool_inputs]
    if teacher_tool_inputs:
        output["teacher_tool_inputs"] = [list(pair) for pair in teacher_tool_inputs]
    if trace is not None:
        for role in ("student", "teacher"):
            output[f"{role}_trace_id"] = getattr(trace, f"{role}_trace_id")
            output[f"{role}_deployment_id"] = getattr(trace, f"{role}_deployment_id")
            output[f"{role}_min_start_ms"] = getattr(trace, f"{role}_min_start_ms")
            output[f"{role}_max_start_ms"] = getattr(trace, f"{role}_max_start_ms")
    return output


def _attach_missing_tool_inputs(evalcli: Any, trajectories: Sequence[Any], *, skip_tools: frozenset[str]) -> None:
    """Fill tool-call arguments for reflection entries that do not have them yet."""
    for role in ("teacher", "student"):
        locators = []
        pending: list[Any] = []
        for trajectory in trajectories:
            output = trajectory["output"]
            if output.get(f"{role}_tool_inputs"):
                continue
            locator = build_trace_locator(
                entry_id=str(output.get("entry_id") or ""),
                deployment_id=output.get(f"{role}_deployment_id"),
                trace_id=output.get(f"{role}_trace_id"),
                min_start_ms=output.get(f"{role}_min_start_ms"),
                max_start_ms=output.get(f"{role}_max_start_ms"),
            )
            if locator is None:
                continue
            locators.append(locator)
            pending.append(output)
        if not locators:
            continue
        fetched = fetch_named_tool_inputs_by_entry(
            evalcli, locators, skip_tools=skip_tools, limit=TOOL_INPUT_LIMIT, role_label=role
        )
        for output in pending:
            pairs = fetched.get(str(output.get("entry_id") or ""))
            if pairs:
                output[f"{role}_tool_inputs"] = [list(pair) for pair in pairs]


def _rationale_cache_key(output: Mapping[str, Any]) -> tuple[str, str, str]:
    """Scope cached rationales to the judge run (or teacher eval) that produced the score."""
    return (
        str(output.get("student_eval_run_id") or ""),
        str(output.get("entry_id") or ""),
        str(output.get("judge_run_id") or output.get("teacher_eval_run_id") or ""),
    )


def _sample_reflection_indices(
    indices: Sequence[int],
    *,
    trajectories: Sequence[Any],
    cap: int | None,
) -> list[int]:
    """Seeded uniform sample, or the full list when it already fits ``cap``.

    A fresh ``Random(REFLECTION_SAMPLE_SEED)`` per call keeps the loss draw
    identical to the keep draw's seed without consuming RNG state across sets.
    """
    if not indices:
        return []
    if cap is None or len(indices) <= cap:
        return list(indices)

    def entry_id(index: int) -> str:
        trajectory = trajectories[index]
        output = trajectory["output"] if isinstance(trajectory, Mapping) else {}
        return str(output.get("entry_id") or "")

    ordered = sorted(indices, key=entry_id)
    selected = random.Random(REFLECTION_SAMPLE_SEED).sample(ordered, cap)
    selected.sort(key=entry_id)
    return selected


class AgenticPreferenceObjective(TeacherStudentObjective[AgenticPreferenceAnalysis]):
    """Score the student by whether the pairwise agentic judge preferred it over the teacher."""

    name = AGENTIC_PREFERENCE_OBJECTIVE
    telemetry_dimensions = (AGENTIC_PREFERENCE_OBJECTIVE,)
    focused_bucket_type = QUERY_CANONICAL_BUCKET_TYPE
    failure_label = "REFLECTION EXAMPLES (teacher-preferred LOSS and student-preferred KEEP)"
    reflection_report_title = "REFLECTION: teacher vs student agentic preference"
    reflection_entry_limit: ClassVar[int] = REFLECTION_RANDOM_SAMPLE_SIZE
    reflection_selection_justification: ClassVar[str] = (
        "Justification: seeded random sample of teacher-preferred losses, plus a keep set of "
        f"student-preferred wins (preference >= {KEEP_PREFERENCE_THRESHOLD}) so the rewriter "
        "does not regress finished outputs. The keep budget is independent of the loss cap."
    )
    module_responsibilities: ClassVar[Mapping[str, str]] = {
        WRITING_CODE_KEY: WRITING_CODE_RESPONSIBILITY,
        RULES_EXT_KEY: RULES_EXT_RESPONSIBILITY,
        EXECUTION_DISCIPLINE_KEY: EXECUTION_DISCIPLINE_RESPONSIBILITY,
        **{tool: core_tool_responsibility(tool) for tool in CORE_TOOLS},
    }

    def __init__(self, *, bigquery_client: Any | None = None, lookback_days: int = 1):
        self.bigquery_client = bigquery_client
        self.lookback_days = lookback_days
        self.params: dict[str, Any] = {}
        self._paired_analysis_cache: dict[tuple[str, str], AgenticPreferenceAnalysis] = {}
        self._rationale_cache: dict[tuple[str, str, str], str] = {}

    def analyze(
        self, teacher_eval_id: str, student_eval_id: str, *, request: AnalysisRequest
    ) -> AgenticPreferenceAnalysis:
        return self.cached_paired_analysis(
            teacher_eval_id,
            student_eval_id,
            request=request,
            cache=self._paired_analysis_cache,
            fetch=fetch_paired_preference_traces,
            empty=empty_agentic_preference_analysis,
            label="agentic preference traces",
        )

    def validate_full_eval(self, analysis: AgenticPreferenceAnalysis) -> None:
        print(
            f"[agentic preference] {analysis.student_eval_id} vs {analysis.teacher_eval_id}: "
            f"{analysis.compared_entries} paired traces"
        )

    def focused_pass_rate(self, analysis: AgenticPreferenceAnalysis, requested_entry_ids: Sequence[str]) -> float:
        """Unused: the adapter screens on the AGENTIC_JUDGE overlay, not traces."""
        del analysis, requested_entry_ids
        return 0.0

    # The score is a placeholder: the adapter overlays the pairwise judge's
    # per-entry verdicts after these rows are built.

    def aggregate_row(self, analysis: AgenticPreferenceAnalysis, ctx: ScoringContext) -> ScoredRow:
        return ScoredRow(
            entry_id=None,
            dimension_scores={AGENTIC_PREFERENCE_OBJECTIVE: 0.0},
            output=_rollout_output(
                entry_id=ctx.query,
                deployment_id=ctx.deployment_id,
                query=ctx.query,
                student_eval_run_id=analysis.student_eval_id,
                teacher_eval_run_id=analysis.teacher_eval_id,
            ),
        )

    def entry_row(
        self,
        entry_id: str,
        metrics: AgenticPreferenceEntry,
        analysis: AgenticPreferenceAnalysis,
        ctx: ScoringContext,
    ) -> ScoredRow:
        return ScoredRow(
            entry_id=entry_id,
            dimension_scores={AGENTIC_PREFERENCE_OBJECTIVE: 0.0},
            output=_rollout_output(
                entry_id=entry_id,
                deployment_id=ctx.deployment_id,
                query=ctx.query,
                student_answer=metrics.student_answer,
                teacher_answer=metrics.teacher_answer,
                student_tools=metrics.student_tools,
                teacher_tools=metrics.teacher_tools,
                student_tool_inputs=metrics.student_tool_inputs,
                teacher_tool_inputs=metrics.teacher_tool_inputs,
                student_eval_run_id=analysis.student_eval_id,
                teacher_eval_run_id=analysis.teacher_eval_id,
                trace=metrics,
            ),
        )

    def hydrate_reflective_trajectories(self, selected: list[Any]) -> None:
        """Attach judge verdicts and tool-call arguments to the entries reflection will read."""
        if self.evalcli is None:
            return
        raw_skipped = self.experiment_param("skipped_tools", None)
        skipped = SKIPPED_TOOL_NAMES if raw_skipped is None else frozenset(str(name) for name in raw_skipped)
        _attach_missing_tool_inputs(self.evalcli, selected, skip_tools=skipped)
        feedback_key = f"{self.name}_feedback"
        pending: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
        for trajectory in selected:
            output = trajectory["output"]
            output.pop(feedback_key, None)
            entry_id = str(output.get("entry_id") or "")
            student_eval_id = str(output.get("student_eval_run_id") or "")
            deployment_id = str(output.get("deployment_id") or "")
            if not entry_id or not student_eval_id or not deployment_id:
                continue
            cache_key = _rationale_cache_key(output)
            if cached := self._rationale_cache.get(cache_key):
                output[feedback_key] = cached
                continue
            pending[(student_eval_id, deployment_id, cache_key[2])].append(trajectory)
        for (student_eval_id, deployment_id, cache_scope), trajectories in pending.items():
            judge_run_id = str(trajectories[0]["output"].get("judge_run_id") or "") or None
            rationales = fetch_preference_rationales(
                self.evalcli,
                entry_ids=[str(trajectory["output"]["entry_id"]) for trajectory in trajectories],
                student_eval_id=student_eval_id,
                deployment_id=deployment_id,
                judge_run_id=judge_run_id,
            )
            print(f"[agentic preference] Loaded {len(rationales)}/{len(trajectories)} judge rationales")
            for trajectory in trajectories:
                output = trajectory["output"]
                entry_id = str(output.get("entry_id") or "")
                if rationale := rationales.get(entry_id):
                    output[feedback_key] = rationale
                    self._rationale_cache[(student_eval_id, entry_id, cache_scope)] = rationale

    def is_high_signal(self, output: Mapping[str, Any]) -> bool:
        score = output.get(self.name)
        return score is not None and score < PREFERENCE_TIE

    def _mismatch_key(self, output: Mapping[str, Any]) -> tuple[str, str] | None:
        if self.is_high_signal(output):
            return TEACHER_PREFERRED_KEY
        score = output.get(self.name)
        if score is not None and score >= KEEP_PREFERENCE_THRESHOLD:
            return STUDENT_PREFERRED_KEY
        return None

    def _select_mismatch_groups(
        self,
        mismatch_keys: Sequence[tuple[str, str] | None],
        *,
        trajectories: Sequence[Any] = (),
        max_entries: int | None = REFLECTION_RANDOM_SAMPLE_SIZE,
    ) -> tuple[list[int], list[tuple[str, str, int]]]:
        """Reflect on teacher-preferred losses plus a student-preferred keep set."""
        if len(trajectories) != len(mismatch_keys):
            return super()._select_mismatch_groups(mismatch_keys, trajectories=trajectories, max_entries=max_entries)

        losses = [index for index, key in enumerate(mismatch_keys) if key == TEACHER_PREFERRED_KEY]
        keeps = [index for index, key in enumerate(mismatch_keys) if key == STUDENT_PREFERRED_KEY]
        selected_losses = _sample_reflection_indices(losses, trajectories=trajectories, cap=max_entries)
        keep_cap = None if max_entries is None else KEEP_SAMPLE_SIZE
        selected_keeps = _sample_reflection_indices(keeps, trajectories=trajectories, cap=keep_cap)
        selected = selected_losses + selected_keeps
        groups: list[tuple[str, str, int]] = []
        if selected_losses:
            groups.append((*TEACHER_PREFERRED_KEY, len(selected_losses)))
        if selected_keeps:
            groups.append((*STUDENT_PREFERRED_KEY, len(selected_keeps)))
        return selected, groups

    def failure_pattern(self, component_name: str, trajectory: TeacherStudentALTrajectory) -> tuple[Any, ...]:
        del component_name
        return (int(self.is_high_signal(trajectory["output"])),)

    def build_reflective_example(
        self,
        component_name: str,
        trajectory: TeacherStudentALTrajectory,
        candidate: dict[str, str],
    ) -> ReflectiveExample:
        del component_name, candidate
        output = trajectory["output"]
        objective_scores = trajectory.get("objective_scores", {})
        preference = output.get(self.name)
        if preference is None:
            preference = objective_scores.get(self.name, trajectory["score"])
        teacher_tools = [str(name) for name in output.get("teacher_tool_events") or [] if name]
        student_tools = [str(name) for name in output.get("student_tool_events") or [] if name]
        student_answer = str(output.get("student_answer") or "")
        teacher_answer = str(output.get("teacher_answer") or "")
        rationale = output.get(f"{self.name}_feedback")
        rationale = rationale.strip() if isinstance(rationale, str) else ""
        flags = student_behavior_flags(
            student_answer=student_answer,
            teacher_answer=teacher_answer,
            student_tools=student_tools,
            teacher_tools=teacher_tools,
        )
        if preference >= KEEP_PREFERENCE_THRESHOLD:
            won = decisive_dimensions(rationale, side="student")
            feedback_parts = [
                f"KEEP: the pairwise agentic judge already preferred the student "
                f"(preference={preference:.2f}; tie is {PREFERENCE_TIE:.2f}).",
                "WON ON: "
                + (", ".join(f"{name} (gap {gap:g})" for name, gap in won) if won else "overall verdict")
                + ". Read the judge's explanation for what the student did right here and keep that behavior; "
                "a rule that makes the student longer, more hedged, more exploratory, or more likely to ask "
                "would regress this entry.",
            ]
        else:
            lost = decisive_dimensions(rationale, side="teacher")
            feedback_parts = [
                f"LOSS: the pairwise agentic judge preferred the teacher "
                f"(preference={preference:.2f}; tie is {PREFERENCE_TIE:.2f}).",
                "DECIDED BY: "
                + (
                    ", ".join(f"{name} (gap {gap:g})" for name, gap in lost)
                    if lost
                    else "overall verdict (no dimension-level split available)"
                )
                + ". Task completion decides first, correctness overrides when the central claim is wrong, "
                "readiness breaks ties.",
                f"Preferred run tools: {' -> '.join(teacher_tools) if teacher_tools else '(none)'}",
                f"Student tools: {' -> '.join(student_tools) if student_tools else '(none)'}",
            ]
        if flags:
            feedback_parts.append("STUDENT BEHAVIOR FLAGS:\n" + "\n".join(f"- {flag}" for flag in flags))
        if rationale:
            feedback_parts.append(
                "Judge verdict by dimension (gap = how decisive; 'claim [...]' lines are the judge's own "
                f"fact checks; 'user requested' vs actions lines show coverage):\n{rationale}"
            )
        return self.reflective_example(
            trajectory,
            feedback="\n".join(feedback_parts),
            generated={
                "student_answer": output.get("student_answer", ""),
                "teacher_answer": output.get("teacher_answer", ""),
                "student_tools": list(output.get("student_tool_events") or []),
                "teacher_tools": list(output.get("teacher_tool_events") or []),
            },
            action_inputs=tool_input_evidence(output),
            action_input_limit=2 * TOOL_INPUT_LIMIT,
        )


__all__ = ["AgenticPreferenceObjective"]
