"""Mode-specific reflective examples for Glean GEPA.

``make_reflective_example`` assembles the shared envelope. The builders fill in
judging-mode feedback, metrics, and evidence fields.
"""

from __future__ import annotations

from typing import Any

from glean_gepa.adapter_types import ALTrajectory, SingleModelALTrajectory, TeacherStudentALTrajectory
from glean_gepa.al_adapter import (
    ReflectiveExample,
    ReflectiveExampleInputs,
    ReflectiveExampleMetrics,
    ReflectiveExampleOutputs,
)
from glean_gepa.reflection_sampling import strip_stdout_sections
from glean_gepa.shell_tool_error_util import SHELL_SUCCESS_OBJECTIVE
from glean_gepa.tool_match_util import first_tool_mismatch_pair


def make_reflective_example(
    trajectory: ALTrajectory,
    *,
    feedback: str,
    metrics: ReflectiveExampleMetrics,
    generated_outputs: ReflectiveExampleOutputs | None = None,
    action_inputs: list[str] | None = None,
    execution_errors: list[str] | None = None,
) -> ReflectiveExample:
    """Assemble a reflective example envelope from one trajectory."""
    data: Any = trajectory["data"]
    output: Any = trajectory["output"]
    inputs: ReflectiveExampleInputs = {
        "eval_set": data["eval_set_name"],
        "entry_id": output["entry_id"],
        "deployment_id": output["deployment_id"],
        "query": output["query"],
    }
    if eval_run_id := data.get("eval_run_id"):
        inputs["eval_run_id"] = eval_run_id
    if eval_trace_id := data.get("eval_trace_id"):
        inputs["eval_trace_id"] = eval_trace_id
    empty_outputs: ReflectiveExampleOutputs = {
        "student_answer": "",
        "teacher_answer": "",
        "student_tools": [],
        "teacher_tools": [],
    }
    return {
        "Inputs": inputs,
        "Generated Outputs": empty_outputs if generated_outputs is None else generated_outputs,
        "Action Inputs": list(action_inputs or []),
        "Execution Errors": list(execution_errors or []),
        "Feedback": feedback,
        "Metrics": metrics,
    }


def build_single_model_reflective_example(
    _component_name: str,
    trajectory: SingleModelALTrajectory,
    _candidate: dict[str, str],
) -> ReflectiveExample:
    """Build a shell-error reflective example from one single-model trajectory."""
    output = trajectory["output"]
    shell_success_rate = trajectory.get("objective_scores", {}).get(SHELL_SUCCESS_OBJECTIVE, 1.0)
    shell_error_messages = [
        sanitized for error in output.get("shell_error_messages", []) if (sanitized := strip_stdout_sections(error))
    ]
    if shell_error_messages:
        # Keep the concrete text solely in ``Execution Errors``. Repeating
        # it in feedback wastes reflection context without adding signal.
        feedback = "Resolve the shell execution failures shown above."
    elif output.get("student_tool_errors", 0) > 0:
        feedback = f"Tool errors: Student encountered {output.get('student_tool_errors', 0)} shell tool errors."
    else:
        feedback = "General shell tool reliability issue."

    return make_reflective_example(
        trajectory,
        feedback=feedback,
        metrics={"score": trajectory["score"], "shell_success_rate": shell_success_rate},
        action_inputs=output.get("shell_action_inputs", [])[:5],
        execution_errors=shell_error_messages[:5],
    )


def build_teacher_student_reflective_example(
    _component_name: str,
    trajectory: TeacherStudentALTrajectory,
    _candidate: dict[str, str],
) -> ReflectiveExample:
    """Build a first-tool-mismatch reflective example from one teacher-student trajectory."""
    output = trajectory["output"]
    objective_scores = trajectory.get("objective_scores", {})
    tool_alignment = objective_scores.get("tool_alignment", trajectory["score"])
    # None when the completeness judge is not part of the objective. Defaulting to
    # 0.0 would tell reflection every entry has a completeness problem.
    completeness = objective_scores.get("completeness")
    student_tools = output.get("student_tool_events", [])
    teacher_tools = output.get("teacher_tool_events", [])
    mismatch = first_tool_mismatch_pair(teacher_tools, student_tools)
    feedback_parts = []
    if mismatch is not None:
        teacher_first, student_first = mismatch
        feedback_parts.append(
            f"First-tool mismatch: teacher used {teacher_first or '(none)'} "
            f"and student used {student_first or '(none)'}."
        )
    if tool_alignment < 1.0:
        feedback_parts.append(f"Tool alignment issue: score={tool_alignment:.2f}.")
    if completeness is not None and completeness < 0.7:
        feedback_parts.append(f"Completeness issue: score={completeness:.2f}.")

    metrics: ReflectiveExampleMetrics = {
        "score": trajectory["score"],
        "tool_alignment": tool_alignment,
    }
    if completeness is not None:
        metrics["completeness"] = completeness

    return make_reflective_example(
        trajectory,
        feedback=" ".join(feedback_parts) if feedback_parts else "General teacher/student tool divergence.",
        metrics=metrics,
        generated_outputs={
            "student_answer": output.get("student_answer", ""),
            "teacher_answer": output.get("teacher_answer", ""),
            "student_tools": student_tools,
            "teacher_tools": teacher_tools,
        },
    )


def format_single_model_reflective_metrics(_metrics: ReflectiveExampleMetrics) -> str | None:
    """Single-model reflection omits a METRICS line; errors carry the signal."""
    return None


def format_teacher_student_reflective_metrics(metrics: ReflectiveExampleMetrics) -> str:
    """Format teacher-student score and tool alignment, plus completeness when scored."""
    parts = [
        f"score={metrics['score']:.2f}",
        f"tool_alignment={metrics.get('tool_alignment', metrics['score']):.2f}",
    ]
    completeness = metrics.get("completeness")
    if completeness is not None:
        parts.append(f"completeness={completeness:.2f}")
    return ", ".join(parts)


__all__ = [
    "build_single_model_reflective_example",
    "build_teacher_student_reflective_example",
    "format_single_model_reflective_metrics",
    "format_teacher_student_reflective_metrics",
    "make_reflective_example",
]
