"""Pairwise agentic preference vs the teacher as the primary teacher-student metric."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from glean_gepa.adapter_types import (
    TeacherStudentALRolloutOutput,
    TeacherStudentALTrajectory,
)
from glean_gepa.al_adapter import ReflectiveExample, ReflectiveExampleInputs, ReflectiveExampleMetrics
from glean_gepa.focused_evalset import QUERY_CANONICAL_BUCKET_TYPE
from glean_gepa.judge_metrics_util import PREFERENCE_TIE
from glean_gepa.objectives.base import ScoredRow, TeacherStudentObjective, register_telemetry_source
from glean_gepa.objectives.utils.agentic_preference_util import (
    AGENTIC_PREFERENCE_OBJECTIVE,
    AgenticPreferenceAnalysis,
    empty_agentic_preference_analysis,
    fetch_paired_preference_traces,
    fetch_preference_rationales,
    log_agentic_preference_analysis,
)
from glean_gepa.objectives.utils.tool_match_util import first_tool_mismatch_pair
from glean_gepa.prompt import high_signal_core_tool_keys
from glean_gepa.prompt_constants import (
    CORE_TOOLS,
    EXECUTION_DISCIPLINE_KEY,
    RULES_EXT_KEY,
    WRITING_CODE_KEY,
)
from glean_gepa.reflection_prompts import NO_EXAMPLE_SPECIFICS_RULE, TEACHER_IS_OFFLINE_RULE

TEACHER_PREFERRED_KEY = ("teacher_preferred", "")
STUDENT_PREFERRED_KEY = ("student_preferred", "keep")
REFLECTION_RANDOM_SAMPLE_SIZE = 25
REFLECTION_SAMPLE_SEED = 0
# Cortex overall 6/10. Ties at 0.5 are excluded so keep examples are decisive wins.
KEEP_PREFERENCE_THRESHOLD = 0.6
KEEP_SAMPLE_SIZE = 15

KEEP_PRESERVE_RULE = (
    "Some supplied examples are student-preferred keep cases "
    f"(preference >= {KEEP_PREFERENCE_THRESHOLD}): the student "
    "already beat the teacher with a finished, forwardable output. Do not regress those."
)

KEEP_SEARCH_BUDGET_RULE = (
    "Preserve search-then-cite and artifact polish; do not add a hard stop after a fixed number "
    "of searches or drop factual search-first to fix other losses."
)

WRITING_CODE_RESPONSIBILITY = (
    "You are rewriting the ## Writing Code body: SDK call patterns, ToolResult handling, "
    "the **Rules:** list, and sandbox privacy. Do not add a heading. Rewrite those coding "
    "instructions so the student produces the requested deliverable in the sandbox (Write or "
    "Edit a file, run the skill, print then extract) instead of answering from snippets, "
    "truncated ToolResults, or memory. Do not add yield, ask, retry, or search-budget rules — "
    "those belong in Execution Discipline. Do not rewrite a named native tool's description. "
    "Leave response formatting and citation mechanics to other modules. "
    f"{KEEP_PRESERVE_RULE} {NO_EXAMPLE_SPECIFICS_RULE} {TEACHER_IS_OFFLINE_RULE} Propose minimal deltas."
)

RULES_EXT_RESPONSIBILITY = (
    "You are writing at most two markdown bullets that will be appended after the existing "
    "**Rules:** list in Writing Code. Each line must start with '- '. Do not repeat those "
    "existing Rules, do not add a heading, and do not exceed two bullets. Close the coding-"
    "rules gap: produce the requested file or result (Write, Edit, or the relevant skill) "
    "instead of answering from snippets. Do not restate Execution Discipline (when to yield, "
    "ask, retry, or stop) or a core-tool description. "
    f"{KEEP_PRESERVE_RULE} Keep each bullet operational and concise. "
    f"{NO_EXAMPLE_SPECIFICS_RULE} {TEACHER_IS_OFFLINE_RULE}"
)

EXECUTION_DISCIPLINE_RESPONSIBILITY = (
    "You are rewriting the bullets under '### Execution Discipline', which set when the "
    "assistant may yield: whether to keep working, whether to ask, whether to retry after an "
    "empty result, and which kind of next step to take. Each line must start with '- '. Do not "
    "add a heading. Nudge the student to complete a shareable, polished deliverable before "
    "yielding. If the query is merely ambiguous, assume from user context and proceed; ask only "
    "after a usable first version exists, or if the request is impossible to interpret. After a "
    "failed or empty step, switch tool class rather than issuing another similar search. Stop "
    "when the output is forwardable, not when snippets seem sufficient and not after a numeric "
    "search cap. Preserve factual search-first; do not add a hard stop after a fixed number of "
    "searches. Do not tell the student to reason without tools as a default, or to keep "
    "searching until it is sure. This module governs effort and stopping conditions only: leave "
    "SDK syntax, the Writing Code **Rules:** list, and named-tool descriptions to other modules. "
    f"{KEEP_PRESERVE_RULE} {KEEP_SEARCH_BUDGET_RULE} {NO_EXAMPLE_SPECIFICS_RULE} {TEACHER_IS_OFFLINE_RULE}"
)


def core_tool_responsibility(tool_name: str) -> str:
    """Reflection instructions for one core-tool ``schema.description``."""
    return (
        f"You are editing only the prompt-visible schema.description for the core tool `{tool_name}`. "
        "The override replaces description text only — not the tool signature, parameters, or Returns. "
        "Rewrite the description so the student uses this tool when it is the step that obtains or "
        "produces the requested deliverable, and does not use it as a substitute for that step "
        "(searching again, asking instead of drafting, or discovering instead of acting). An "
        "opening-tool difference is extra evidence only when it caused an unfinished artifact; do "
        "not rewrite merely to copy the preferred run's first action. Do not add yield, "
        "search-budget, or stopping-condition rules — those belong in Execution Discipline. "
        f"{KEEP_PRESERVE_RULE} Keep the text operational and concise. "
        f"{NO_EXAMPLE_SPECIFICS_RULE} {TEACHER_IS_OFFLINE_RULE}"
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
    student_eval_run_id: str = "",
    teacher_eval_run_id: str = "",
) -> TeacherStudentALRolloutOutput:
    return {
        "deployment_id": deployment_id,
        "query": query,
        "student_eval_run_id": student_eval_run_id,
        "teacher_eval_run_id": teacher_eval_run_id,
        "student_answer": student_answer,
        "student_tool_events": list(student_tools),
        "student_loops": 0,
        "student_tool_calls": len(student_tools),
        "student_tool_errors": 0,
        "student_input_tokens": 0,
        "student_output_tokens": 0,
        "student_latency_ms": None,
        "teacher_answer": teacher_answer,
        "teacher_tool_events": list(teacher_tools),
        "teacher_loops": 0,
        "teacher_tool_calls": len(teacher_tools),
        "teacher_input_tokens": 0,
        "teacher_output_tokens": 0,
        "entry_id": entry_id,
    }


def _preference_score(output: Mapping[str, Any]) -> float | None:
    raw = output.get(AGENTIC_PREFERENCE_OBJECTIVE)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


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


class AgenticPreferenceObjective(TeacherStudentObjective):
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

    def analyze(self, teacher_eval_id: str, student_eval_id: str) -> AgenticPreferenceAnalysis:
        return self.cached_paired_analysis(
            teacher_eval_id,
            student_eval_id,
            cache=self._paired_analysis_cache,
            fetch=fetch_paired_preference_traces,
            empty=empty_agentic_preference_analysis,
            label="agentic preference traces",
        )

    def validate_full_eval(self, analysis: AgenticPreferenceAnalysis) -> None:
        log_agentic_preference_analysis(analysis)

    def focused_pass_rate(self, analysis: AgenticPreferenceAnalysis, requested_entry_ids: Sequence[str]) -> float:
        """Unused: the adapter screens on the AGENTIC_JUDGE overlay, not traces."""
        del analysis, requested_entry_ids
        return 0.0

    def scored_rows(
        self,
        analysis: AgenticPreferenceAnalysis,
        *,
        focused: bool,
        capture_traces: bool,
        query: str,
        deployment_id: str,
    ) -> list[ScoredRow]:
        if focused and not analysis.per_entry:
            return []
        if not focused and not capture_traces:
            return [
                ScoredRow(
                    entry_id=None,
                    dimension_scores={AGENTIC_PREFERENCE_OBJECTIVE: 0.0},
                    output=_rollout_output(
                        entry_id=query,
                        deployment_id=deployment_id,
                        query=query,
                        student_eval_run_id=analysis.student_eval_id,
                        teacher_eval_run_id=analysis.teacher_eval_id,
                    ),
                )
            ]
        return [
            ScoredRow(
                entry_id=entry_id,
                dimension_scores={AGENTIC_PREFERENCE_OBJECTIVE: 0.0},
                output=_rollout_output(
                    entry_id=entry_id,
                    deployment_id=deployment_id,
                    query=query,
                    student_answer=metrics.student_answer,
                    teacher_answer=metrics.teacher_answer,
                    student_tools=metrics.student_tools,
                    teacher_tools=metrics.teacher_tools,
                    student_eval_run_id=analysis.student_eval_id,
                    teacher_eval_run_id=analysis.teacher_eval_id,
                ),
            )
            for entry_id, metrics in analysis.per_entry.items()
        ]

    def hydrate_reflective_trajectories(self, selected: list[Any]) -> None:
        """Attach the judge's prose verdict to the entries reflection will read."""
        if self.evalcli is None:
            return
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
        score = _preference_score(output)
        return score is not None and score < PREFERENCE_TIE

    def _mismatch_key(self, output: Mapping[str, Any]) -> tuple[str, str] | None:
        if self.is_high_signal(output):
            return TEACHER_PREFERRED_KEY
        score = _preference_score(output)
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
        """Reflect on teacher-preferred losses plus a student-preferred keep set.

        Losses are a seeded uniform sample of size ``max_entries`` (or every
        loss when ``max_entries is None`` / YAML ``reflection_samples: all``).
        Keeps are sampled independently with the same seed so adding them does
        not change which 25 losses reflection already saw. Screening still
        uses ``is_high_signal`` and stays loss-only.
        """
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

    def high_signal_core_tool_keys(self, trajectories: Sequence[Any] | None) -> list[str]:
        """Core tools whose description a loss actually implicates.

        The proposer drops every core-tool module that this does not name, so
        returning nothing would leave them listed as editable but never edited.
        Narrow to the judge's losses first: a first-tool divergence on an entry
        the student won is not evidence that a description misled it.
        """
        losses = [
            trajectory
            for trajectory in trajectories or []
            if isinstance(trajectory, Mapping) and self.is_high_signal(trajectory.get("output") or {})
        ]
        return high_signal_core_tool_keys(losses)

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
        preference = _preference_score(output)
        if preference is None:
            preference = float(objective_scores.get(self.name, trajectory["score"]))
        if preference >= KEEP_PREFERENCE_THRESHOLD:
            feedback_parts = [
                f"KEEP: the pairwise agentic judge already preferred the student "
                f"(preference={preference:.2f}; tie is {PREFERENCE_TIE:.2f}). Do not regress this "
                "finished, forwardable output."
            ]
        else:
            feedback_parts = [
                f"LOSS: the pairwise agentic judge preferred the teacher "
                f"(preference={preference:.2f}; tie is {PREFERENCE_TIE:.2f})."
            ]
            if (
                pair := first_tool_mismatch_pair(output.get("teacher_tool_events"), output.get("student_tool_events"))
            ) is not None:
                teacher_first, student_first = pair
                teacher_phrase = f"opened with {teacher_first}" if teacher_first else "called no scored tool"
                student_phrase = f"opened with {student_first}" if student_first else "called no scored tool"
                feedback_parts.append(f"The preferred run {teacher_phrase} and the student {student_phrase}.")
        rationale = output.get(f"{self.name}_feedback")
        if isinstance(rationale, str) and rationale.strip():
            feedback_parts.append(f"Judge verdict by dimension:\n{rationale.strip()}")
        inputs: ReflectiveExampleInputs = {
            "eval_set": trajectory["data"]["eval_set_name"],
            "entry_id": output["entry_id"],
            "deployment_id": output["deployment_id"],
            "query": output["query"],
        }
        metrics: ReflectiveExampleMetrics = {
            "score": trajectory["score"],
            "agentic_preference_rate": preference,
        }
        correctness = objective_scores.get("correctness")
        if correctness is not None:
            metrics["correctness"] = correctness
        return {
            "Inputs": inputs,
            "Generated Outputs": {
                "student_answer": output.get("student_answer", ""),
                "teacher_answer": output.get("teacher_answer", ""),
                "student_tools": list(output.get("student_tool_events") or []),
                "teacher_tools": list(output.get("teacher_tool_events") or []),
            },
            "Action Inputs": [],
            "Execution Errors": [],
            "Feedback": "\n".join(feedback_parts),
            "Metrics": metrics,
        }

    def format_reflective_metrics(self, metrics: ReflectiveExampleMetrics) -> str:
        parts = [
            f"score={metrics['score']:.2f}",
            f"agentic_preference_rate={metrics.get('agentic_preference_rate', metrics['score']):.2f}",
        ]
        correctness = metrics.get("correctness")
        if correctness is not None:
            parts.append(f"correctness={correctness:.2f}")
        return ", ".join(parts)


register_telemetry_source("teacher_student", "agentic_preference", AgenticPreferenceObjective)

__all__ = ["AgenticPreferenceObjective"]
