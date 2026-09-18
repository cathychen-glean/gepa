"""First-tool teacher/student alignment objective."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from glean_gepa.adapter_types import (
    TeacherStudentALRolloutOutput,
    TeacherStudentALTrajectory,
)
from glean_gepa.al_adapter import ReflectiveExample, ReflectiveExampleInputs, ReflectiveExampleMetrics
from glean_gepa.focused_evalset import QUERY_CANONICAL_BUCKET_TYPE
from glean_gepa.objectives.base import ScoredRow, TeacherStudentObjective, register_telemetry_source
from glean_gepa.objectives.utils.mismatch import select_mismatch_groups
from glean_gepa.objectives.utils.tool_match_util import (
    SKIPPED_TOOL_NAMES,
    TOOL_ALIGNMENT_OBJECTIVE,
    EvalRunToolMatchAnalysis,
    empty_tool_match_analysis,
    fetch_eval_run_tool_match_analysis,
    first_tool_mismatch_pair,
    log_tool_match_analysis,
    require_compared_eval_entries,
)
from glean_gepa.prompt import high_signal_core_tool_keys, is_core_tool_span, tool_description_override_key
from glean_gepa.prompt_constants import CORE_TOOL_KEYS, EXECUTION_DISCIPLINE_KEY, RULES_EXT_KEY
from glean_gepa.reflection_prompts import NO_EXAMPLE_SPECIFICS_RULE, TEACHER_IS_OFFLINE_RULE

EXECUTION_DISCIPLINE_RESPONSIBILITY = (
    "You are rewriting the bullets under '### Execution Discipline', which set how much effort "
    "the assistant spends before answering: how many tool loops to use, how many queries to "
    "issue, whether to retry after an empty result, and when to stop searching and respond. "
    "Each line must start with '- '. Do not add a heading. This module governs effort and "
    "stopping conditions only: leave response formatting, citation mechanics, and shell or SDK "
    "syntax to other modules. Prefer stating the condition under which more work is warranted "
    "over raising a numeric cap, so the rule generalizes to requests of different sizes. "
    f"{NO_EXAMPLE_SPECIFICS_RULE} {TEACHER_IS_OFFLINE_RULE}"
)

RULES_EXT_RESPONSIBILITY = (
    "You are writing at most two markdown bullets that will be appended after the existing "
    "**Rules:** list in Writing Code. Each line must start with '- '. Do not repeat those "
    "existing Rules, do not add a heading, and do not exceed two bullets. Target first-tool "
    "mismatches whose tools are not core tools (for example Write vs (none)). Keep each "
    f"bullet operational and concise. {NO_EXAMPLE_SPECIFICS_RULE} {TEACHER_IS_OFFLINE_RULE}"
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


class FirstToolMatchObjective(TeacherStudentObjective):
    """Score the student's first tool call against the teacher's."""

    name = TOOL_ALIGNMENT_OBJECTIVE
    telemetry_dimensions = (TOOL_ALIGNMENT_OBJECTIVE,)
    focused_bucket_type = QUERY_CANONICAL_BUCKET_TYPE
    failure_label = "HIGH-SIGNAL FAILURES (teacher vs student tool match)"
    reflection_report_title = "REFLECTION: teacher vs student tool sequences"
    teacher_compared_key = "teacher_tool_events"
    student_compared_key = "student_tool_events"
    mismatch_pair = first_tool_mismatch_pair
    module_responsibilities: ClassVar[Mapping[str, str]] = {
        RULES_EXT_KEY: RULES_EXT_RESPONSIBILITY,
        EXECUTION_DISCIPLINE_KEY: EXECUTION_DISCIPLINE_RESPONSIBILITY,
    }

    def __init__(self, *, bigquery_client: Any | None = None, lookback_days: int = 1):
        self.bigquery_client = bigquery_client
        self.lookback_days = lookback_days
        self.params: dict[str, Any] = {}
        self._paired_analysis_cache: dict[tuple[str, str], EvalRunToolMatchAnalysis] = {}

    def _skipped_tools(self) -> frozenset[str]:
        raw = self.pack_param("skipped_tools", None)
        if raw is None:
            return SKIPPED_TOOL_NAMES
        return frozenset(str(name) for name in raw)

    def _mismatch_key(self, output: Mapping[str, Any]) -> tuple[str, str] | None:
        return first_tool_mismatch_pair(
            output.get(self.teacher_compared_key),
            output.get(self.student_compared_key),
            skip_tools=self._skipped_tools(),
        )

    def analyze(self, teacher_eval_id: str, student_eval_id: str) -> EvalRunToolMatchAnalysis:
        skipped = self._skipped_tools()

        def fetch(client: Any, **kwargs: Any) -> EvalRunToolMatchAnalysis:
            return fetch_eval_run_tool_match_analysis(client, skip_tools=skipped, **kwargs)

        return self.cached_paired_analysis(
            teacher_eval_id,
            student_eval_id,
            cache=self._paired_analysis_cache,
            fetch=fetch,
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

    def _component_trajectories(
        self,
        component_name: str,
        selected: list[Any],
        selected_keys: list[tuple[str, str] | None],
        *,
        trajectories: list[Any],
        mismatch_keys: list[tuple[str, str] | None],
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
            non_core = [
                (trajectory, key)
                for trajectory, key in zip(trajectories, mismatch_keys, strict=True)
                if key is not None and not any(is_core_tool_span(name) for name in key if name)
            ]
            indices, _ = select_mismatch_groups([key for _, key in non_core])
            return [non_core[index][0] for index in indices]
        return selected

    def failure_pattern(self, component_name: str, trajectory: TeacherStudentALTrajectory) -> tuple[Any, ...]:
        del component_name
        output = trajectory["output"]
        tool_alignment = trajectory.get("objective_scores", {}).get(self.name, 1.0)
        return (
            int(tool_alignment < float(self.pack_param("failure_score_below", 0.7))),
            int(self._mismatch_key(output) is not None),
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
        completeness = objective_scores.get("completeness")
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
        action_inputs: list[str] = []
        for role in ("teacher", "student"):
            pair = output.get(f"{role}_first_tool_input")
            if isinstance(pair, list | tuple) and len(pair) == 2 and pair[1]:
                tool, payload = pair
                payload_text = str(payload)
                if len(payload_text) > 240:
                    payload_text = payload_text[:240] + "... (truncated)"
                action_inputs = [f"{role} first tool ({tool or 'unknown'}): {payload_text}"]
                break
        return {
            "Inputs": inputs,
            "Generated Outputs": {
                "student_answer": output.get("student_answer", ""),
                "teacher_answer": output.get("teacher_answer", ""),
                "student_tools": student_tools,
                "teacher_tools": teacher_tools,
            },
            "Action Inputs": action_inputs,
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
