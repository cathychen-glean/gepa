"""Citation-set teacher/student alignment objective."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from glean_gepa.adapter_types import TeacherStudentALTrajectory, paired_rollout_output
from glean_gepa.al_adapter import ReflectiveExample, ReflectiveExampleInputs
from glean_gepa.focused_evalset import QUERY_CANONICAL_BUCKET_TYPE
from glean_gepa.objectives.base import ScoredRow, TeacherStudentObjective, register_telemetry_source
from glean_gepa.objectives.utils.citation_match_util import (
    CITATION_MATCH_OBJECTIVE,
    EvalRunCitationMatchAnalysis,
    citation_mismatch_pair,
    empty_citation_match_analysis,
    fetch_eval_run_citation_match_analysis,
    log_citation_match_analysis,
    require_compared_citation_entries,
)
from glean_gepa.prompt_constants import RULES_EXT_KEY, WRITING_CODE_KEY
from glean_gepa.reflection_prompts import NO_EXAMPLE_SPECIFICS_RULE, TEACHER_IS_OFFLINE_RULE

WRITING_CODE_RESPONSIBILITY = (
    "Focus ONLY on the coding and execution instructions that decide which sources reach the answer: "
    "printing raw SDK results, carrying each result's `citationId` through any filtering, ranking, or "
    "summarizing step, and not truncating output the answer still has to cite. Change other guidance "
    "only where it produces missing or extra citations. "
    f"{NO_EXAMPLE_SPECIFICS_RULE} {TEACHER_IS_OFFLINE_RULE} Propose minimal deltas."
)

RULES_EXT_RESPONSIBILITY = (
    "You are writing at most two markdown bullets that will be appended after the existing "
    "**Rules:** list in Writing Code. Each line must start with '- '. Do not repeat those "
    "existing Rules, do not add a heading, and do not exceed two bullets. Target citation "
    "mismatches (missing teacher sources, extra student sources, or dropped citationId values "
    "after filtering SDK results). Keep each bullet operational and concise. "
    f"{NO_EXAMPLE_SPECIFICS_RULE} {TEACHER_IS_OFFLINE_RULE}"
)


def _rollout_output(
    *,
    entry_id: str,
    deployment_id: str,
    query: str,
    student_citations: list[str],
    teacher_citations: list[str],
    student_action_inputs: list[str] | None = None,
    teacher_action_inputs: list[str] | None = None,
):
    """One rollout row. Citation IDs ride on optional output fields plus answers."""
    output = paired_rollout_output(
        deployment_id=deployment_id,
        query=query,
        entry_id=entry_id,
        student_answer="citations=" + (", ".join(student_citations) if student_citations else "(none)"),
        teacher_answer="citations=" + (", ".join(teacher_citations) if teacher_citations else "(none)"),
    )
    output["student_citations"] = student_citations
    output["teacher_citations"] = teacher_citations
    if student_action_inputs:
        output["student_action_inputs"] = list(student_action_inputs)
    if teacher_action_inputs:
        output["teacher_action_inputs"] = list(teacher_action_inputs)
    return output


class CitationMatchObjective(TeacherStudentObjective):
    """Score the student's cited source set against the teacher's."""

    name = CITATION_MATCH_OBJECTIVE
    telemetry_dimensions = (CITATION_MATCH_OBJECTIVE,)
    focused_bucket_type = QUERY_CANONICAL_BUCKET_TYPE
    failure_label = "HIGH-SIGNAL FAILURES (teacher vs student citation match)"
    reflection_report_title = "REFLECTION: teacher vs student citation sets"
    teacher_compared_key = "teacher_citations"
    student_compared_key = "student_citations"
    mismatch_pair = citation_mismatch_pair
    module_responsibilities: ClassVar[Mapping[str, str]] = {
        WRITING_CODE_KEY: WRITING_CODE_RESPONSIBILITY,
        RULES_EXT_KEY: RULES_EXT_RESPONSIBILITY,
    }

    def __init__(self, *, bigquery_client: Any | None = None, lookback_days: int = 1):
        self.bigquery_client = bigquery_client
        self.lookback_days = lookback_days
        self.params: dict[str, Any] = {}
        self._paired_analysis_cache: dict[tuple[str, str], EvalRunCitationMatchAnalysis] = {}

    def analyze(self, teacher_eval_id: str, student_eval_id: str) -> EvalRunCitationMatchAnalysis:
        return self.cached_paired_analysis(
            teacher_eval_id,
            student_eval_id,
            cache=self._paired_analysis_cache,
            fetch=fetch_eval_run_citation_match_analysis,
            empty=empty_citation_match_analysis,
            label="citation match analysis",
        )

    def require_compared_entries(self, analysis: EvalRunCitationMatchAnalysis) -> None:
        require_compared_citation_entries(analysis)

    def validate_full_eval(self, analysis: EvalRunCitationMatchAnalysis) -> None:
        log_citation_match_analysis(analysis)

    def focused_pass_rate(self, analysis: EvalRunCitationMatchAnalysis, requested_entry_ids: Sequence[str]) -> float:
        matching = sum(1 for metrics in analysis.per_entry.values() if metrics.citations_match)
        return matching / len(requested_entry_ids)

    def scored_rows(
        self,
        analysis: EvalRunCitationMatchAnalysis,
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
                    dimension_scores={CITATION_MATCH_OBJECTIVE: analysis.aggregate.citation_match_rate},
                    output=_rollout_output(
                        entry_id=query,
                        deployment_id=deployment_id,
                        query=query,
                        student_citations=[],
                        teacher_citations=[],
                    ),
                )
            ]
        return [
            ScoredRow(
                entry_id=entry_id,
                dimension_scores={CITATION_MATCH_OBJECTIVE: float(citation_match.citations_match)},
                output=_rollout_output(
                    entry_id=entry_id,
                    deployment_id=deployment_id,
                    query=query,
                    student_citations=list(citation_match.student_citations),
                    teacher_citations=list(citation_match.teacher_citations),
                    student_action_inputs=list(citation_match.student_action_inputs),
                    teacher_action_inputs=list(citation_match.teacher_action_inputs),
                ),
            )
            for entry_id, citation_match in analysis.per_entry.items()
        ]

    def failure_pattern(self, component_name: str, trajectory: TeacherStudentALTrajectory) -> tuple[Any, ...]:
        del component_name
        output = trajectory["output"]
        citation_match = trajectory.get("objective_scores", {}).get(self.name, 1.0)
        return (
            int(citation_match < float(self.pack_param("failure_score_below", 1.0))),
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
        citation_match = objective_scores.get(self.name, trajectory["score"])
        student_citations = list(output.get("student_citations") or [])
        teacher_citations = list(output.get("teacher_citations") or [])
        mismatch = self._mismatch_key(output)
        feedback_parts = []
        if mismatch is not None:
            missing, extra = mismatch
            feedback_parts.append(f"Citation-set mismatch: {missing}; {extra}.")
        if citation_match < 1.0:
            feedback_parts.append(f"Citation match issue: score={citation_match:.2f}.")
        feedback_parts.extend(self.wired_signal_issues(objective_scores))

        inputs: ReflectiveExampleInputs = {
            "eval_set": trajectory["data"]["eval_set_name"],
            "entry_id": output["entry_id"],
            "deployment_id": output["deployment_id"],
            "query": output["query"],
        }
        # The raw user query is scrubbed, so surface the teacher's tool payloads
        # (the searches it ran) as the intent signal behind the cited sources.
        teacher_action_inputs = output.get("teacher_action_inputs") or []
        return {
            "Inputs": inputs,
            "Generated Outputs": {
                "student_answer": output.get("student_answer", ""),
                "teacher_answer": output.get("teacher_answer", ""),
                "student_tools": student_citations,
                "teacher_tools": teacher_citations,
                "student_citations": student_citations,
                "teacher_citations": teacher_citations,
            },
            "Action Inputs": list(teacher_action_inputs[:5]),
            "Execution Errors": [],
            "Feedback": " ".join(feedback_parts) if feedback_parts else "General teacher/student citation divergence.",
            "Metrics": self.reflective_metrics(trajectory),
        }


register_telemetry_source("teacher_student", "citation_match", CitationMatchObjective)

__all__ = ["CitationMatchObjective"]
