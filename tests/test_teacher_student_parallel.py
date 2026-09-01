from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from glean_gepa.al_adapter import ALRunner, Thresholds
from glean_gepa.evalcli_client import COMPLETENESS_JUDGE_TYPE
from glean_gepa.judge_metrics_util import JudgeAnalysis
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter, _StartedPair
from glean_gepa.tool_match_util import (
    EvalRunToolMatchAnalysis,
    NoComparedEvalEntriesError,
    ToolMatchEntryMetrics,
    ToolMatchMetrics,
)

EVAL_SET = {
    "eval_set_name": "Glean Chat V2 Medium",
    "eval_set_version": "20260806",
    "deployment_ids": ["scio-prod"],
    "status": "active",
}


def _evalcli_with_ordered_events(events: list[str]) -> MagicMock:
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
    evalcli.get_eval_metrics.return_value = {"judgeMetrics": {"COMPLETENESS": {"passRate": 0.0}}}
    return evalcli


def _teacher_student_adapter(evalcli: MagicMock, cache_file: str | None = None) -> TeacherStudentAdapter:
    return TeacherStudentAdapter(
        runner=ALRunner(evalcli=evalcli),
        teacher_model="gpt",
        student_model="claude",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        cache_file=cache_file,
    )


def _assert_all_creates_before_waits(events: list[str], *, n_creates: int, n_waits: int) -> None:
    create_idxs = [i for i, event in enumerate(events) if event.startswith("create:")]
    wait_idxs = [i for i, event in enumerate(events) if event.startswith("wait:")]
    assert len(create_idxs) == n_creates
    assert len(wait_idxs) == n_waits
    assert max(create_idxs) < min(wait_idxs)


def _tool_match_analysis(
    *,
    teacher_eval_id: str = "teacher-1",
    student_eval_id: str = "student-1",
    compared_entries: int = 1,
) -> EvalRunToolMatchAnalysis:
    per_entry = {}
    if compared_entries:
        per_entry["entry-1"] = ToolMatchEntryMetrics(
            entry_id="entry-1",
            student_tools=("search",),
            teacher_tools=("read",),
            tools_match=False,
            tool_match_score=0.5,
        )
    matching = sum(1 for metrics in per_entry.values() if metrics.tools_match)
    return EvalRunToolMatchAnalysis(
        teacher_eval_id=teacher_eval_id,
        student_eval_id=student_eval_id,
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
        aggregate=ToolMatchMetrics(
            teacher_eval_id=teacher_eval_id,
            student_eval_id=student_eval_id,
            compared_entries=len(per_entry),
            matching_entries=matching,
            mismatching_entries=len(per_entry) - matching,
            tool_match_rate=(matching / len(per_entry)) if per_entry else 0.0,
        ),
        per_entry=per_entry,
        high_signal_entry_ids=tuple(per_entry),
    )


def test_teacher_and_student_runs_are_created_before_waiting():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    adapter = _teacher_student_adapter(evalcli)

    with patch.object(adapter, "_get_or_fetch_tool_match_analysis", return_value=_tool_match_analysis()):
        adapter.evaluate([EVAL_SET], {"WRITING_CODE": "test prompt"}, capture_traces=False)

    _assert_all_creates_before_waits(events, n_creates=2, n_waits=2)


def test_all_eval_set_runs_in_a_batch_are_created_before_waiting():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    adapter = _teacher_student_adapter(evalcli)
    batch = [
        {**EVAL_SET, "eval_set_version": "20260806"},
        {**EVAL_SET, "eval_set_version": "20260807"},
    ]

    with patch.object(adapter, "_get_or_fetch_tool_match_analysis", return_value=_tool_match_analysis()):
        adapter.evaluate(batch, {"WRITING_CODE": "test prompt"}, capture_traces=False)

    _assert_all_creates_before_waits(events, n_creates=4, n_waits=4)


def test_evaluate_many_starts_all_candidate_runs_before_waiting():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    adapter = _teacher_student_adapter(evalcli)

    with patch.object(adapter, "_get_or_fetch_tool_match_analysis", return_value=_tool_match_analysis()):
        adapter.evaluate_many(
            [EVAL_SET],
            [
                {"WRITING_CODE": "prompt a"},
                {"WRITING_CODE": "prompt b"},
            ],
            capture_traces=False,
        )

    # One shared teacher run plus one student run per candidate.
    _assert_all_creates_before_waits(events, n_creates=3, n_waits=3)
    teacher_creates = [event for event in events if event.startswith("create:") and "_gpt_" in event]
    student_creates = [event for event in events if event.startswith("create:") and "_claude_" in event]
    assert len(teacher_creates) == 1
    assert len(student_creates) == 2


def test_batch_evaluate_shares_teacher_run_across_children():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    adapter = _teacher_student_adapter(evalcli)
    batch = [
        {
            **EVAL_SET,
            "eval_entry_ids": ["source-entry"],
            "focused_eval_set_name": "gepa-high-signal-glean-chat-v2-medium",
            "focused_eval_set_version": "20260806_hs_abc",
        }
    ]

    with patch.object(adapter, "_get_or_fetch_tool_match_analysis", return_value=_tool_match_analysis()):
        adapter.batch_evaluate(
            [
                ({"WRITING_CODE": "prompt a"}, batch),
                ({"WRITING_CODE": "prompt b"}, batch),
            ],
            capture_traces=True,
        )

    _assert_all_creates_before_waits(events, n_creates=3, n_waits=3)
    teacher_creates = [event for event in events if event.startswith("create:") and "_gpt_" in event]
    student_creates = [event for event in events if event.startswith("create:") and "_claude_" in event]
    assert len(teacher_creates) == 1
    assert len(student_creates) == 2


def test_al_runner_run_still_waits_before_returning():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    runner = ALRunner(evalcli=evalcli)

    runner.run(
        "gpt",
        "<<TEACHER_PROD_PROMPT>>",
        eval_set_name="Glean Chat V2 Medium",
        eval_set_version="20260806",
        deployment_ids=["scio-prod"],
    )

    assert len(events) == 2
    assert events[0].startswith("create:")
    assert events[1].startswith("wait:")
    assert events[0].split(":", 1)[1] == events[1].split(":", 1)[1]


def test_get_or_fetch_tool_match_analysis_returns_empty_without_bigquery():
    adapter = _teacher_student_adapter(MagicMock())

    analysis = adapter._get_or_fetch_tool_match_analysis("teacher-1", "student-1")

    assert analysis.teacher_eval_id == "teacher-1"
    assert analysis.student_eval_id == "student-1"
    assert analysis.per_entry == {}
    assert analysis.aggregate.compared_entries == 0


def test_get_or_fetch_tool_match_analysis_caches_fetch():
    adapter = _teacher_student_adapter(MagicMock())
    adapter.bigquery_client = MagicMock()
    fetched = EvalRunToolMatchAnalysis(
        teacher_eval_id="teacher-1",
        student_eval_id="student-1",
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
        aggregate=ToolMatchMetrics(
            teacher_eval_id="teacher-1",
            student_eval_id="student-1",
            compared_entries=1,
            matching_entries=0,
            mismatching_entries=1,
            tool_match_rate=0.0,
        ),
        per_entry={},
        high_signal_entry_ids=("entry-1",),
    )

    with patch(
        "glean_gepa.teacher_student_adapter.fetch_eval_run_tool_match_analysis",
        return_value=fetched,
    ) as fetch:
        first = adapter._get_or_fetch_tool_match_analysis("teacher-1", "student-1")
        second = adapter._get_or_fetch_tool_match_analysis("teacher-1", "student-1")

    fetch.assert_called_once_with(
        adapter.bigquery_client,
        teacher_eval_id="teacher-1",
        student_eval_id="student-1",
        lookback_days=adapter.agentspan_lookback_days,
    )
    assert first is fetched
    assert second is fetched


def test_finish_batch_evals_uses_tool_match_and_completeness():
    adapter = _teacher_student_adapter(MagicMock())
    analysis = EvalRunToolMatchAnalysis(
        teacher_eval_id="teacher-1",
        student_eval_id="student-1",
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
        aggregate=ToolMatchMetrics(
            teacher_eval_id="teacher-1",
            student_eval_id="student-1",
            compared_entries=1,
            matching_entries=0,
            mismatching_entries=1,
            tool_match_rate=0.5,
        ),
        per_entry={
            "entry-1": ToolMatchEntryMetrics(
                entry_id="entry-1",
                student_tools=("search",),
                teacher_tools=("read",),
                tools_match=False,
                tool_match_score=0.5,
            )
        },
        high_signal_entry_ids=("entry-1",),
    )
    adapter._tool_match_cache[("teacher-1", "student-1")] = analysis
    adapter._judge_cache[("student-1", COMPLETENESS_JUDGE_TYPE)] = JudgeAnalysis(
        eval_id="student-1", aggregate=1.0, per_entry={"entry-1": 1.0}, judge_type=COMPLETENESS_JUDGE_TYPE
    )
    adapter._judge_cache[("teacher-1", COMPLETENESS_JUDGE_TYPE)] = JudgeAnalysis(
        eval_id="teacher-1", aggregate=0.9, per_entry={"entry-1": 0.9}, judge_type=COMPLETENESS_JUDGE_TYPE
    )
    result = adapter._finish_batch_evals(
        [
            _StartedPair(
                al_data_inst=EVAL_SET,
                teacher_eval_id="teacher-1",
                student_eval_id="student-1",
            )
        ],
        capture_traces=True,
    )

    assert result.scores == pytest.approx([0.85])
    assert result.objective_scores == [{"completeness": 1.0, "tool_alignment": 0.5, "grounding": 1.0}]
    assert result.outputs[0]["student_tool_events"] == ["search"]
    assert result.outputs[0]["teacher_tool_events"] == ["read"]
    assert result.summary == {
        "completeness": 1.0,
        "tool_alignment": 0.5,
        "grounding": 1.0,
        "teacher_completeness": 0.9,
    }
    assert result.trajectories is not None
    assert result.trajectories[0]["score"] == pytest.approx(0.85)


def test_high_signal_eval_runs_teacher_and_student_on_focused_set():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    adapter = _teacher_student_adapter(evalcli)
    batch = [
        {
            **EVAL_SET,
            "eval_entry_ids": ["source-entry"],
            "focused_eval_set_name": "gepa-high-signal-glean-chat-v2-medium",
            "focused_eval_set_version": "20260806_hs_abc",
        }
    ]

    with patch.object(adapter, "_get_or_fetch_tool_match_analysis", return_value=_tool_match_analysis()):
        adapter.evaluate(batch, {"WRITING_CODE": "test prompt"}, capture_traces=False)

    creates = [call.kwargs for call in evalcli.create_eval_run.call_args_list]
    assert len(creates) == 2
    for kwargs in creates:
        assert kwargs["eval_set_name"] == "gepa-high-signal-glean-chat-v2-medium"
        assert kwargs["eval_set_version"] == "20260806_hs_abc"
        assert kwargs["eval_run_id"].startswith("gepa_high_signal_")


def test_finish_focused_eval_uses_requested_entry_denominator():
    adapter = _teacher_student_adapter(MagicMock())
    analysis = EvalRunToolMatchAnalysis(
        teacher_eval_id="teacher-1",
        student_eval_id="student-1",
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
        aggregate=ToolMatchMetrics(
            teacher_eval_id="teacher-1",
            student_eval_id="student-1",
            compared_entries=2,
            matching_entries=1,
            mismatching_entries=1,
            tool_match_rate=0.5,
        ),
        per_entry={
            "fresh-1": ToolMatchEntryMetrics(
                entry_id="fresh-1",
                student_tools=("read",),
                teacher_tools=("read",),
                tools_match=True,
                tool_match_score=1.0,
            ),
            "fresh-2": ToolMatchEntryMetrics(
                entry_id="fresh-2",
                student_tools=("search",),
                teacher_tools=("read",),
                tools_match=False,
                tool_match_score=0.5,
            ),
        },
        high_signal_entry_ids=("fresh-1", "fresh-2"),
    )
    adapter._tool_match_cache[("teacher-1", "student-1")] = analysis
    result = adapter._finish_batch_evals(
        [
            _StartedPair(
                al_data_inst={**EVAL_SET, "eval_entry_ids": ["s1", "s2", "s3"]},
                teacher_eval_id="teacher-1",
                student_eval_id="student-1",
            )
        ],
        capture_traces=True,
    )

    assert result.summary is not None
    assert result.summary["tool_alignment"] == pytest.approx(1 / 3)
    assert [score["tool_alignment"] for score in result.objective_scores] == [1.0, 0.0]


def test_finish_focused_eval_does_not_raise_when_no_entries_were_compared():
    adapter = _teacher_student_adapter(MagicMock())
    adapter._tool_match_cache[("teacher-1", "student-1")] = _tool_match_analysis(compared_entries=0)
    result = adapter._finish_batch_evals(
        [
            _StartedPair(
                al_data_inst={**EVAL_SET, "eval_entry_ids": ["s1"]},
                teacher_eval_id="teacher-1",
                student_eval_id="student-1",
            )
        ],
        capture_traces=False,
    )

    assert result.outputs == []
    assert result.summary == {
        "completeness": 0.0,
        "tool_alignment": 0.0,
        "grounding": 1.0,
        "teacher_completeness": 0.0,
    }


def test_finish_batch_evals_raises_when_no_entries_were_compared():
    adapter = _teacher_student_adapter(MagicMock())
    adapter._tool_match_cache[("teacher-1", "student-1")] = _tool_match_analysis(compared_entries=0)

    with pytest.raises(NoComparedEvalEntriesError, match="No eval entries were compared"):
        adapter._finish_batch_evals(
            [
                _StartedPair(
                    al_data_inst=EVAL_SET,
                    teacher_eval_id="teacher-1",
                    student_eval_id="student-1",
                )
            ],
            capture_traces=False,
        )


def test_launched_eval_ids_are_soft_cached_before_wait(tmp_path):
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    cache_file = tmp_path / "eval_cache.json"
    adapter = _teacher_student_adapter(evalcli, cache_file=str(cache_file))

    with patch.object(adapter.runner, "wait", side_effect=TimeoutError("terminal timed out")):
        with pytest.raises(TimeoutError):
            adapter.evaluate([EVAL_SET], {"WRITING_CODE": "test prompt"}, capture_traces=False)

    saved = json.loads(cache_file.read_text())
    assert saved["eval_cache"] == {}
    assert len(saved["in_flight_eval_cache"]) == 2
    create_ids = [event.split(":", 1)[1] for event in events if event.startswith("create:")]
    assert sorted(saved["in_flight_eval_cache"].values()) == sorted(create_ids)


def test_soft_cached_eval_ids_resume_wait_instead_of_recreating(tmp_path):
    first_events: list[str] = []
    first_evalcli = _evalcli_with_ordered_events(first_events)
    cache_file = tmp_path / "eval_cache.json"
    first = _teacher_student_adapter(first_evalcli, cache_file=str(cache_file))
    with patch.object(first.runner, "wait", side_effect=TimeoutError("terminal timed out")):
        with pytest.raises(TimeoutError):
            first.evaluate([EVAL_SET], {"WRITING_CODE": "test prompt"}, capture_traces=False)
    launched_ids = sorted(json.loads(cache_file.read_text())["in_flight_eval_cache"].values())

    second_events: list[str] = []
    second_evalcli = _evalcli_with_ordered_events(second_events)
    second = _teacher_student_adapter(second_evalcli, cache_file=str(cache_file))
    with patch.object(second, "_get_or_fetch_tool_match_analysis", return_value=_tool_match_analysis()):
        second.evaluate([EVAL_SET], {"WRITING_CODE": "test prompt"}, capture_traces=False)

    assert second_evalcli.create_eval_run.call_count == 0
    waited_ids = sorted(event.split(":", 1)[1] for event in second_events if event.startswith("wait:"))
    assert waited_ids == launched_ids
    saved = json.loads(cache_file.read_text())
    assert saved["in_flight_eval_cache"] == {}
    assert sorted(saved["eval_cache"].values()) == launched_ids


def _stub_completeness_judge(evalcli: MagicMock, events: list[str], *, score: float = 0.8) -> None:
    evalcli.find_judge_run_id.return_value = None

    def create_completeness(**kwargs):
        eval_run_id = kwargs["eval_run_id"]
        judge_id = f"judge-{eval_run_id}"
        events.append(f"judge-create:{eval_run_id}")
        return judge_id

    evalcli.create_judge_run.side_effect = create_completeness
    evalcli.get_eval_metrics.return_value = {"judgeMetrics": {"COMPLETENESS": {"passRate": score}}}


def test_completeness_judges_run_for_teacher_and_student_after_evals():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    _stub_completeness_judge(evalcli, events)
    adapter = _teacher_student_adapter(evalcli)

    with patch.object(adapter, "_get_or_fetch_tool_match_analysis", return_value=_tool_match_analysis()):
        result = adapter.evaluate([EVAL_SET], {"WRITING_CODE": "test prompt"}, capture_traces=False)

    create_idxs = [i for i, event in enumerate(events) if event.startswith("create:")]
    wait_idxs = [i for i, event in enumerate(events) if event.startswith("wait:")]
    assert max(create_idxs) < min(wait_idxs)
    eval_run_ids = [event.split(":", 1)[1] for event in events if event.startswith("create:")]
    judged_eval_ids = [event.split(":", 1)[1] for event in events if event.startswith("judge-create:")]
    assert sorted(judged_eval_ids) == sorted(eval_run_ids)
    for eval_id in eval_run_ids:
        assert events.index(f"wait:{eval_id}") < events.index(f"judge-create:{eval_id}")
    assert result.summary is not None
    assert result.summary["completeness"] == pytest.approx(0.8)
    assert result.summary["teacher_completeness"] == pytest.approx(0.8)
    assert result.objective_scores[0]["completeness"] == pytest.approx(0.8)
    assert "correctness" not in result.objective_scores[0]


def test_completeness_judge_for_teacher_is_created_once_across_candidates():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    _stub_completeness_judge(evalcli, events)
    adapter = _teacher_student_adapter(evalcli)

    with patch.object(adapter, "_get_or_fetch_tool_match_analysis", return_value=_tool_match_analysis()):
        adapter.evaluate_many(
            [EVAL_SET],
            [{"WRITING_CODE": "prompt a"}, {"WRITING_CODE": "prompt b"}],
            capture_traces=False,
        )

    judged_eval_ids = [event.split(":", 1)[1] for event in events if event.startswith("judge-create:")]
    # One shared teacher eval plus one student eval per candidate.
    assert len(judged_eval_ids) == 3
    assert len(set(judged_eval_ids)) == 3

