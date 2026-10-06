"""Typed evaluation records for the Glean adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple, NotRequired, TypeAlias, TypedDict

JudgingMode: TypeAlias = Literal["teacher_student", "single_model"]


class PointwiseJudge(NamedTuple):
    name: str
    judge_type: str
    run_params: str


class PairwiseJudge(NamedTuple):
    name: str
    #: Adapter-side key for caching and dedupe. Usually the Cortex judge type.
    judge_type: str
    run_params: str
    input_mappings: str
    #: Judge type sent to Cortex when it differs from ``judge_type``. Several
    #: judges (e.g. the agentic multi-dimension and correctness-only skills) share
    #: Cortex type AGENTIC_JUDGE and differ only by ``judge_skill_name``.
    cortex_type_override: str | None = None
    judge_skill_name: str | None = None

    @property
    def cortex_judge_type(self) -> str:
        return self.cortex_type_override or self.judge_type


class EvalHarness(NamedTuple):
    """Per-experiment overrides for eval-run creation (the ``eval:`` config section).

    ``runner_type`` is the EvalCLI ``--runner-type`` (e.g. ``GLEAN_CHAT``). ``sc_params``
    replaces the model alias's base harness preset when set. ``None`` uses the alias default.

    ``extra_sc_params`` and ``drop_sc_params`` come from the prompt targets a run edits
    (``render:`` in their ``target.yaml``): entries added after the preset, and preset
    entries removed, on every eval so those prompts render for teacher and student alike.
    """

    runner_type: str | None = None
    sc_params: str | None = None
    extra_sc_params: tuple[str, ...] = ()
    drop_sc_params: tuple[str, ...] = ()


class EvalSetALDataInst(TypedDict):
    """Configuration shared by both Glean evaluation adapters.

    ``validation_only`` marks an eval set that may only be scored from its eval
    run's metrics. Validation runs on customer deployments, whose eval-set
    entries are PII-gated and unreadable outside the Cortex UI, so reflection
    and focused high-signal screening must stay on the training deployments.
    """

    eval_set_name: str
    eval_set_version: str
    deployment_ids: list[str]
    status: str
    validation_only: NotRequired[bool]
    eval_entry_ids: NotRequired[list[str]]
    focused_eval_set_name: NotRequired[str]
    focused_eval_set_version: NotRequired[str]
    focused_entry_ids: NotRequired[dict[str, str]]
    screen_references: NotRequired[dict[str, str]]


class SingleModelALDataInst(EvalSetALDataInst):
    """Single-model eval data enriched with the failed eval execution identity."""

    eval_entry_id: NotRequired[str]
    eval_run_id: NotRequired[str]
    source_eval_run_id: NotRequired[str]
    eval_trace_id: NotRequired[str]
    eval_entry_ids: NotRequired[list[str]]
    focused_eval_set_name: NotRequired[str]
    focused_eval_set_version: NotRequired[str]
    cached_student_eval_run_id: NotRequired[str]


class TeacherStudentALDataInst(EvalSetALDataInst):
    """Eval-set configuration for a paired teacher/student comparison."""

    cached_student_eval_run_id: NotRequired[str]
    cached_teacher_eval_run_id: NotRequired[str]


class BaseALRolloutOutput(TypedDict):
    """Fields shared by all Glean rollout outputs."""

    deployment_id: str
    query: str
    entry_id: str


class SingleModelALRolloutOutput(BaseALRolloutOutput):
    """Shell reliability evidence from one evaluated student model."""

    student_tool_calls: int
    student_tool_errors: int
    shell_error_messages: list[str]
    student_eval_run_id: str
    shell_action_inputs: NotRequired[list[str]]
    action_inputs: NotRequired[list[str]]
    eval_trace_id: NotRequired[str]
    student_loops: NotRequired[int]
    correctness: NotRequired[float]


class TeacherStudentALRolloutOutput(BaseALRolloutOutput):
    """Paired answers and tool traces used by teacher/student judging."""

    student_answer: str
    student_tool_events: list[str]
    student_tool_calls: int
    teacher_answer: str
    teacher_tool_events: list[str]
    teacher_tool_calls: int
    student_eval_run_id: NotRequired[str]
    teacher_eval_run_id: NotRequired[str]
    judge_run_id: NotRequired[str]
    student_citations: NotRequired[list[str]]
    teacher_citations: NotRequired[list[str]]
    student_action_inputs: NotRequired[list[str]]
    teacher_action_inputs: NotRequired[list[str]]
    student_tool_inputs: NotRequired[list[list[str]]]
    teacher_tool_inputs: NotRequired[list[list[str]]]
    student_trace_id: NotRequired[str]
    teacher_trace_id: NotRequired[str]
    student_deployment_id: NotRequired[str]
    teacher_deployment_id: NotRequired[str]
    student_min_start_ms: NotRequired[int]
    student_max_start_ms: NotRequired[int]
    teacher_min_start_ms: NotRequired[int]
    teacher_max_start_ms: NotRequired[int]
    student_waldo_termination: NotRequired[str]
    teacher_waldo_termination: NotRequired[str]
    student_waldo_summary: NotRequired[str]
    teacher_waldo_summary: NotRequired[str]
    student_first_sentence_refusal: NotRequired[bool]
    teacher_first_sentence_refusal: NotRequired[bool]
    agentic_preference_rate: NotRequired[float]
    agentic_preference_rate_feedback: NotRequired[str]


def paired_rollout_output(
    *,
    deployment_id: str,
    query: str,
    entry_id: str,
    student_answer: str = "",
    teacher_answer: str = "",
    student_tool_events: Sequence[str] = (),
    teacher_tool_events: Sequence[str] = (),
    student_tool_calls: int | None = None,
    teacher_tool_calls: int | None = None,
    student_eval_run_id: str | None = None,
    teacher_eval_run_id: str | None = None,
) -> TeacherStudentALRolloutOutput:
    """Paired rollout. Objectives pass the answers, tools, and eval ids they score."""
    student_events = list(student_tool_events)
    teacher_events = list(teacher_tool_events)
    output: TeacherStudentALRolloutOutput = {
        "deployment_id": deployment_id,
        "query": query,
        "entry_id": entry_id,
        "student_answer": student_answer,
        "student_tool_events": student_events,
        "student_tool_calls": len(student_events) if student_tool_calls is None else student_tool_calls,
        "teacher_answer": teacher_answer,
        "teacher_tool_events": teacher_events,
        "teacher_tool_calls": len(teacher_events) if teacher_tool_calls is None else teacher_tool_calls,
    }
    if student_eval_run_id is not None:
        output["student_eval_run_id"] = student_eval_run_id
    if teacher_eval_run_id is not None:
        output["teacher_eval_run_id"] = teacher_eval_run_id
    return output


class SingleModelALTrajectory(TypedDict):
    data: SingleModelALDataInst
    output: SingleModelALRolloutOutput
    score: float
    objective_scores: dict[str, float]


class TeacherStudentALTrajectory(TypedDict):
    data: TeacherStudentALDataInst
    output: TeacherStudentALRolloutOutput
    score: float
    objective_scores: dict[str, float]


# Shared infrastructure dispatches to one concrete adapter at runtime. Keep
# these unions internal/compatibility-facing; adapter users should use the
# concrete types above.
ALDataInst: TypeAlias = SingleModelALDataInst | TeacherStudentALDataInst
ALRolloutOutput: TypeAlias = SingleModelALRolloutOutput | TeacherStudentALRolloutOutput
ALTrajectory: TypeAlias = SingleModelALTrajectory | TeacherStudentALTrajectory


__all__ = [
    "ALDataInst",
    "ALRolloutOutput",
    "ALTrajectory",
    "EvalSetALDataInst",
    "JudgingMode",
    "PointwiseJudge",
    "paired_rollout_output",
    "SingleModelALDataInst",
    "SingleModelALRolloutOutput",
    "SingleModelALTrajectory",
    "TeacherStudentALDataInst",
    "TeacherStudentALRolloutOutput",
    "TeacherStudentALTrajectory",
]
