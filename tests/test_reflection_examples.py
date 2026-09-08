from glean_gepa.reflection_examples import (
    build_single_model_reflective_example,
    build_teacher_student_reflective_example,
    format_single_model_reflective_metrics,
    format_teacher_student_reflective_metrics,
    make_reflective_example,
)
from glean_gepa.shell_tool_error_util import SHELL_SUCCESS_OBJECTIVE


def _single_model_trajectory(
    *,
    entry_id: str = "entry-1",
    score: float = 0.5,
    shell_success_rate: float = 0.5,
    shell_error_messages: list[str] | None = None,
    shell_action_inputs: list[str] | None = None,
    student_tool_errors: int = 0,
    eval_run_id: str | None = None,
    eval_trace_id: str | None = None,
):
    data = {"eval_set_name": "small-eval-set", "eval_set_version": "20260403"}
    if eval_run_id:
        data["eval_run_id"] = eval_run_id
    if eval_trace_id:
        data["eval_trace_id"] = eval_trace_id
    output = {
        "entry_id": entry_id,
        "deployment_id": "scio-prod",
        "query": "fix the failing command",
        "student_tool_errors": student_tool_errors,
        "shell_error_messages": shell_error_messages or [],
    }
    if shell_action_inputs is not None:
        output["shell_action_inputs"] = shell_action_inputs
    return {
        "data": data,
        "output": output,
        "score": score,
        "objective_scores": {SHELL_SUCCESS_OBJECTIVE: shell_success_rate},
    }


def _teacher_student_trajectory(
    *,
    entry_id: str = "entry-1",
    score: float = 0.4,
    teacher_tools: list[str] | None = None,
    student_tools: list[str] | None = None,
    tool_alignment: float = 0.0,
    completeness: float = 0.5,
):
    return {
        "data": {"eval_set_name": "small-eval-set", "eval_set_version": "20260403"},
        "output": {
            "entry_id": entry_id,
            "deployment_id": "scio-prod",
            "query": "search then read",
            "student_answer": "student",
            "teacher_answer": "teacher",
            "student_tool_events": student_tools or ["Discover"],
            "teacher_tool_events": teacher_tools or ["Glean Search"],
        },
        "score": score,
        "objective_scores": {"tool_alignment": tool_alignment, "completeness": completeness},
    }


def test_make_reflective_example_copies_shared_inputs_and_optional_ids():
    example = make_reflective_example(
        _single_model_trajectory(eval_run_id="run-1", eval_trace_id="trace-1"),
        feedback="Resolve the issue.",
        metrics={"score": 0.5},
        action_inputs=["ls"],
        execution_errors=["boom"],
    )
    assert example["Inputs"] == {
        "eval_set": "small-eval-set",
        "entry_id": "entry-1",
        "deployment_id": "scio-prod",
        "query": "fix the failing command",
        "eval_run_id": "run-1",
        "eval_trace_id": "trace-1",
    }
    assert example["Generated Outputs"] == {
        "student_answer": "",
        "teacher_answer": "",
        "student_tools": [],
        "teacher_tools": [],
    }
    assert example["Action Inputs"] == ["ls"]
    assert example["Execution Errors"] == ["boom"]
    assert example["Feedback"] == "Resolve the issue."
    assert example["Metrics"] == {"score": 0.5}


def test_build_single_model_reflective_example_uses_execution_errors_for_feedback():
    example = build_single_model_reflective_example(
        "WRITING_CODE",
        _single_model_trajectory(
            shell_error_messages=["command failed\nstdout:\nnoisy output\nstderr:\npermission denied"],
            shell_action_inputs=['{"command": "python3 broken.py"}'],
            eval_run_id="run-1",
            eval_trace_id="trace-1",
        ),
        {},
    )
    assert example["Feedback"] == "Resolve the shell execution failures shown above."
    assert example["Execution Errors"] == ["command failed\nstderr:\npermission denied"]
    assert example["Action Inputs"] == ['{"command": "python3 broken.py"}']
    assert example["Inputs"]["eval_run_id"] == "run-1"
    assert example["Inputs"]["eval_trace_id"] == "trace-1"
    assert example["Metrics"] == {"score": 0.5, "shell_success_rate": 0.5}


def test_build_single_model_reflective_example_falls_back_when_errors_lack_messages():
    tool_error = build_single_model_reflective_example(
        "WRITING_CODE",
        _single_model_trajectory(student_tool_errors=2),
        {},
    )
    general = build_single_model_reflective_example("WRITING_CODE", _single_model_trajectory(), {})
    assert tool_error["Feedback"] == "Tool errors: Student encountered 2 shell tool errors."
    assert general["Feedback"] == "General shell tool reliability issue."
    assert format_single_model_reflective_metrics(tool_error["Metrics"]) is None


def test_build_teacher_student_reflective_example_describes_first_tool_mismatch():
    example = build_teacher_student_reflective_example(
        "glean_search",
        _teacher_student_trajectory(),
        {},
    )
    assert example["Feedback"].startswith("First-tool mismatch: teacher used Glean Search and student used Discover.")
    assert "Tool alignment issue: score=0.00." in example["Feedback"]
    assert "Completeness issue: score=0.50." in example["Feedback"]
    assert example["Generated Outputs"]["teacher_tools"] == ["Glean Search"]
    assert example["Generated Outputs"]["student_tools"] == ["Discover"]
    assert format_teacher_student_reflective_metrics(example["Metrics"]) == (
        "score=0.40, tool_alignment=0.00, completeness=0.50"
    )


def test_build_teacher_student_reflective_example_omits_unscored_completeness():
    """With the completeness judge out of the objective, reflection must not be told
    every entry has a completeness problem."""
    trajectory = _teacher_student_trajectory()
    trajectory["objective_scores"] = {"tool_alignment": 0.0, "grounding": 1.0}

    example = build_teacher_student_reflective_example("glean_search", trajectory, {})

    assert "Tool alignment issue: score=0.00." in example["Feedback"]
    assert "Completeness" not in example["Feedback"]
    assert "completeness" not in example["Metrics"]
    assert format_teacher_student_reflective_metrics(example["Metrics"]) == "score=0.40, tool_alignment=0.00"


def test_build_teacher_student_reflective_example_general_feedback_when_aligned():
    example = build_teacher_student_reflective_example(
        "WRITING_CODE",
        _teacher_student_trajectory(
            teacher_tools=["Glean Search"],
            student_tools=["Glean Search"],
            score=1.0,
            tool_alignment=1.0,
            completeness=0.9,
        ),
        {},
    )
    assert example["Feedback"] == "General teacher/student tool divergence."
    assert example["Action Inputs"] == []
    assert example["Execution Errors"] == []
