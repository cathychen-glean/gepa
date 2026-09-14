"""First-tool teacher/student alignment objective."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from glean_gepa.adapter_types import (
    TeacherStudentALRolloutOutput,
    TeacherStudentALTrajectory,
)
from glean_gepa.al_adapter import ReflectiveExample, ReflectiveExampleInputs, ReflectiveExampleMetrics
from glean_gepa.focused_evalset import QUERY_CANONICAL_BUCKET_TYPE
from glean_gepa.objectives.base import ScoredRow, TeacherStudentObjective, register_telemetry_source
from glean_gepa.objectives.utils.tool_match_util import (
    TOOL_ALIGNMENT_OBJECTIVE,
    EvalRunToolMatchAnalysis,
    empty_tool_match_analysis,
    fetch_eval_run_tool_match_analysis,
    first_tool_mismatch_pair,
    log_tool_match_analysis,
    require_compared_eval_entries,
)
from glean_gepa.prompt import high_signal_core_tool_keys, is_core_tool_span, tool_description_override_key
from glean_gepa.prompt_constants import CORE_TOOL_KEYS, RULES_EXT_KEY
from glean_gepa.reflection_prompts import teacher_student_reflection_prompt


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
) -> TeacherStudentALRolloutOutput:
    """One rollout row. Tool-call counts default to the listed events."""
    output: TeacherStudentALRolloutOutput = {
        "deployment_id": deployment_id,
        "query": query,
        "student_answer": "",
        "student_tool_events": student_tools,
        "student_loops": 0,
        "student_tool_calls": len(student_tools) if student_tool_calls is None else student_tool_calls,
        "student_tool_errors": 0,
        "student_input_tokens": 0,
        "student_output_tokens": 0,
        "student_latency_ms": None,
        "teacher_answer": "",
        "teacher_tool_events": teacher_tools,
        "teacher_loops": 0,
        "teacher_tool_calls": len(teacher_tools) if teacher_tool_calls is None else teacher_tool_calls,
        "teacher_input_tokens": 0,
        "teacher_output_tokens": 0,
        "entry_id": entry_id,
    }
    if student_first_tool_input:
        output["student_first_tool_input"] = list(student_first_tool_input)
    if teacher_first_tool_input:
        output["teacher_first_tool_input"] = list(teacher_first_tool_input)
    return output


def _first_tool_phrase(role: str, tool: str) -> str:
    """Describe a role's first tool, spelling out what an absent one means.

    An empty scored sequence is a real choice, not missing data: every span the role
    emitted was a skipped one, so it never reached for a tool this objective scores.
    Reporting that as "(none)" read to reflection like a gap in the trace.
    """
    if tool:
        return f"{role} used {tool}"
    return f"{role} called no scored tool, emitting only skipped steps such as the automatic vault retrieval or shell"


def _first_tool_input_lines(output: Mapping[str, Any]) -> list[str]:
    """The scored first tool call, labelled with the role and tool that issued it.

    Prefers the teacher's call because that is the behaviour being taught, and falls
    back to the student's so reflection still sees what the task was about when the
    teacher answered without calling a tool. An unlabelled payload is worse than
    none here: reflection cannot tell whose call it is reading.
    """
    for role in ("teacher", "student"):
        pair = output.get(f"{role}_first_tool_input")
        if isinstance(pair, list | tuple) and len(pair) == 2 and pair[1]:
            tool, payload = pair
            return [f"{role} first tool ({tool or 'unknown'}): {payload}"]
    return []


class FirstToolMatchObjective(TeacherStudentObjective):
    """Score the student's first tool call against the teacher's."""

    name = TOOL_ALIGNMENT_OBJECTIVE
    telemetry_dimensions = (TOOL_ALIGNMENT_OBJECTIVE,)
    focused_bucket_type = QUERY_CANONICAL_BUCKET_TYPE
    failure_label = "HIGH-SIGNAL FAILURES (teacher vs student tool match)"
    reflection_report_title = "REFLECTION: teacher vs student tool sequences"

    def __init__(self, *, bigquery_client: Any | None = None, lookback_days: int = 1):
        self.bigquery_client = bigquery_client
        self.lookback_days = lookback_days
        self._paired_analysis_cache: dict[tuple[str, str], EvalRunToolMatchAnalysis] = {}

    def analyze(self, teacher_eval_id: str, student_eval_id: str) -> EvalRunToolMatchAnalysis:
        return self.cached_paired_analysis(
            teacher_eval_id,
            student_eval_id,
            cache=self._paired_analysis_cache,
            fetch=fetch_eval_run_tool_match_analysis,
            empty=empty_tool_match_analysis,
            label="tool match analysis",
        )

    def validate_full_eval(self, analysis: EvalRunToolMatchAnalysis) -> None:
        require_compared_eval_entries(analysis)
        log_tool_match_analysis(analysis)

    def focused_pass_rate(self, analysis: EvalRunToolMatchAnalysis, requested_entry_ids: Sequence[str]) -> float:
        matching = sum(1 for metrics in analysis.per_entry.values() if metrics.tools_match)
        return matching / len(requested_entry_ids)

    def scored_rows(
        self,
        analysis: EvalRunToolMatchAnalysis,
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
                    dimension_scores={TOOL_ALIGNMENT_OBJECTIVE: analysis.aggregate.tool_match_rate},
                    output=_rollout_output(
                        entry_id=query,
                        deployment_id=deployment_id,
                        query=query,
                        student_tools=[],
                        teacher_tools=[],
                        student_tool_calls=sum(len(m.student_tools) for m in analysis.per_entry.values()),
                        teacher_tool_calls=sum(len(m.teacher_tools) for m in analysis.per_entry.values()),
                    ),
                )
            ]
        return [
            ScoredRow(
                entry_id=entry_id,
                dimension_scores={TOOL_ALIGNMENT_OBJECTIVE: float(tool_match.tools_match)},
                output=_rollout_output(
                    entry_id=entry_id,
                    deployment_id=deployment_id,
                    query=query,
                    student_tools=list(tool_match.student_tools),
                    teacher_tools=list(tool_match.teacher_tools),
                    student_first_tool_input=tool_match.student_first_tool_input,
                    teacher_first_tool_input=tool_match.teacher_first_tool_input,
                ),
            )
            for entry_id, tool_match in analysis.per_entry.items()
        ]

    def _mismatch_key(self, output: Mapping[str, Any]) -> tuple[str, str] | None:
        return first_tool_mismatch_pair(output.get("teacher_tool_events"), output.get("student_tool_events"))

    def _component_trajectories(
        self,
        component_name: str,
        selected: list[Any],
        selected_keys: list[tuple[str, str] | None],
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
            return [
                trajectory
                for trajectory, pair in zip(selected, selected_keys, strict=True)
                if pair is not None and not any(is_core_tool_span(name) for name in pair if name)
            ]
        return selected

    def reflection_prompt(self, module_name: str) -> str:
        return teacher_student_reflection_prompt(module_name)

    def failure_pattern(self, component_name: str, trajectory: TeacherStudentALTrajectory) -> tuple[Any, ...]:
        del component_name
        output = trajectory["output"]
        tool_alignment = trajectory.get("objective_scores", {}).get(self.name, 1.0)
        return (
            int(tool_alignment < 0.7),
            int(
                first_tool_mismatch_pair(output.get("teacher_tool_events"), output.get("student_tool_events"))
                is not None
            ),
            int(output.get("student_tool_errors", 0) > 0),
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
        # Absent is not zero: the completeness judge is off by default, and defaulting
        # it to 0.0 would report a failed judge on every example.
        completeness = objective_scores.get("completeness")
        student_tools = output.get("student_tool_events", [])
        teacher_tools = output.get("teacher_tool_events", [])
        mismatch = first_tool_mismatch_pair(teacher_tools, student_tools)
        feedback_parts = []
        if mismatch is not None:
            teacher_first, student_first = mismatch
            feedback_parts.append(
                f"First-tool mismatch: {_first_tool_phrase('teacher', teacher_first)} "
                f"and {_first_tool_phrase('student', student_first)}."
            )
        if tool_alignment < 1.0:
            feedback_parts.append(f"Tool alignment issue: score={tool_alignment:.2f}.")
        if completeness is not None and completeness < 0.7:
            feedback_parts.append(f"Completeness issue: score={completeness:.2f}.")

        inputs: ReflectiveExampleInputs = {
            "eval_set": trajectory["data"]["eval_set_name"],
            "entry_id": output["entry_id"],
            "deployment_id": output["deployment_id"],
            "query": output["query"],
        }
        metrics: ReflectiveExampleMetrics = {
            "score": trajectory["score"],
            "tool_alignment": tool_alignment,
        }
        if completeness is not None:
            metrics["completeness"] = completeness
        return {
            "Inputs": inputs,
            "Generated Outputs": {
                "student_answer": output.get("student_answer", ""),
                "teacher_answer": output.get("teacher_answer", ""),
                "student_tools": student_tools,
                "teacher_tools": teacher_tools,
            },
            "Action Inputs": _first_tool_input_lines(output),
            "Execution Errors": [],
            "Feedback": " ".join(feedback_parts) if feedback_parts else "General teacher/student tool divergence.",
            "Metrics": metrics,
        }

    def format_reflective_metrics(self, metrics: ReflectiveExampleMetrics) -> str:
        parts = [
            f"score={metrics['score']:.2f}",
            f"tool_alignment={metrics.get('tool_alignment', metrics['score']):.2f}",
        ]
        completeness = metrics.get("completeness")
        if completeness is not None:
            parts.append(f"completeness={completeness:.2f}")
        return ", ".join(parts)

    def high_signal_core_tool_keys(self, trajectories: Sequence[Any] | None) -> list[str]:
        return high_signal_core_tool_keys(trajectories)


register_telemetry_source("teacher_student", "tool_match", FirstToolMatchObjective)

__all__ = ["FirstToolMatchObjective"]
