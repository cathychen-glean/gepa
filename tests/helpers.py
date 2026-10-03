"""Shared fixtures for glean_gepa tests. Not a test module."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from glean_gepa.al_adapter import ALRunner
from glean_gepa.objectives.tool_match import EvalRunToolMatchAnalysis, ToolMatchEntryMetrics, ToolMatchMetrics
from glean_gepa.single_model_adapter import SingleModelAdapter
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter


EVAL_SET = {
    "eval_set_name": "Glean Chat V2 Medium",
    "eval_set_version": "20260806",
    "deployment_ids": ["scio-prod"],
    "status": "active",
}


def teacher_student_adapter(evalcli: MagicMock | None = None, **kwargs) -> TeacherStudentAdapter:
    kwargs.setdefault("teacher_model", "gpt")
    kwargs.setdefault("student_model", "fast")
    return TeacherStudentAdapter(
        runner=ALRunner(evalcli=evalcli if evalcli is not None else MagicMock()), **kwargs
    )


def single_model_adapter(evalcli: MagicMock | None = None, **kwargs) -> SingleModelAdapter:
    kwargs.setdefault("bigquery_client", MagicMock())
    kwargs.setdefault("student_model", "fast")
    return SingleModelAdapter(
        runner=ALRunner(evalcli=evalcli if evalcli is not None else MagicMock()), **kwargs
    )


def evalcli_with_ordered_events(events: list[str]) -> MagicMock:
    """An evalcli mock that records create/wait ordering into ``events``."""
    evalcli = MagicMock()

    def create_eval_run(**kwargs):
        eval_run_id = kwargs["eval_run_id"]
        events.append(f"create:{eval_run_id}")
        return eval_run_id

    def wait_for_eval_run(eval_run_id, **_kwargs):
        events.append(f"wait:{eval_run_id}")

    evalcli.create_eval_run.side_effect = create_eval_run
    evalcli.wait_for_eval_run.side_effect = wait_for_eval_run
    evalcli.find_judge_run_id.return_value = None
    evalcli.create_judge_run.side_effect = lambda **kwargs: f"judge-{kwargs['eval_run_id']}"
    evalcli.get_eval_metrics.return_value = {
        "judgeMetrics": {"totalEntries": 1, "missingEntries": 0, "CORRECTNESS": {"passRate": 0.0, "sampleSize": 1}}
    }
    return evalcli


def tool_match_analysis(
    *, teacher_eval_id: str = "teacher-1", student_eval_id: str = "student-1", compared_entries: int = 1
) -> EvalRunToolMatchAnalysis:
    per_entry = {}
    if compared_entries:
        per_entry["entry-1"] = ToolMatchEntryMetrics(
            entry_id="entry-1", student_tools=("search",), teacher_tools=("read",), tools_match=False
        )
    matching = sum(1 for m in per_entry.values() if m.tools_match)
    return EvalRunToolMatchAnalysis(
        eval_ids=(teacher_eval_id, student_eval_id),
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
        aggregate=ToolMatchMetrics(
            compared_entries=len(per_entry),
            matching_entries=matching,
            tool_match_rate=(matching / len(per_entry)) if per_entry else 0.0,
        ),
        per_entry=per_entry,
        high_signal_entry_ids=tuple(per_entry),
    )
