"""Typed evaluation records for the Glean adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, NotRequired, TypeAlias, TypedDict

JudgingMode: TypeAlias = Literal["teacher_student", "single_model"]


class PointwiseJudge(NamedTuple):
    """A Cortex pointwise judge and the composite dimension its score feeds.

    ``name`` is the config signal name, which is what ``objective.composite``
    weights; ``judge_type`` is the Cortex-side identity used to start and read
    the judge run.
    """

    name: str
    judge_type: str
    run_params: str


class EvalSetALDataInst(TypedDict):
    """Configuration shared by both Glean evaluation adapters."""

    eval_set_name: str
    eval_set_version: str
    deployment_ids: list[str]
    status: str
    eval_entry_ids: NotRequired[list[str]]
    focused_eval_set_name: NotRequired[str]
    focused_eval_set_version: NotRequired[str]
    source_eval_set_name: NotRequired[str]
    source_eval_set_version: NotRequired[str]
    source_entry_ids_by_focused_id: NotRequired[dict[str, str]]


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
    source_teacher_eval_run_id: NotRequired[str]


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
    eval_trace_id: NotRequired[str]


class TeacherStudentALRolloutOutput(BaseALRolloutOutput):
    """Full paired execution details used by teacher/student judging."""

    student_answer: str
    student_tool_events: list[str]
    student_loops: int
    student_tool_calls: int
    student_tool_errors: int
    student_input_tokens: int
    student_output_tokens: int
    student_latency_ms: int | None
    teacher_answer: str
    teacher_tool_events: list[str]
    teacher_loops: int
    teacher_tool_calls: int
    teacher_input_tokens: int
    teacher_output_tokens: int
    student_eval_run_id: NotRequired[str]
    teacher_eval_run_id: NotRequired[str]


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


@dataclass(frozen=True)
class EntryTelemetry:
    """One eval entry's objective values and the rollout row that explains them.

    ``objective_scores`` holds only the dimensions the producing objective
    measures. The adapter merges in the constants and judge scores that every
    objective shares, so a new objective needs a producer returning these and
    nothing else.
    """

    entry_id: str
    student_entry_id: str
    objective_scores: dict[str, float]
    output: TeacherStudentALRolloutOutput


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
    "EntryTelemetry",
    "EvalSetALDataInst",
    "JudgingMode",
    "SingleModelALDataInst",
    "SingleModelALRolloutOutput",
    "SingleModelALTrajectory",
    "TeacherStudentALDataInst",
    "TeacherStudentALRolloutOutput",
    "TeacherStudentALTrajectory",
]
