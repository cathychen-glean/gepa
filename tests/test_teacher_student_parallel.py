from __future__ import annotations

import json
import threading
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from glean_gepa.adapter_types import PointwiseJudge
from glean_gepa.al_adapter import ALRunner, Thresholds
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evalcli_client import COMPLETENESS_JUDGE_TYPE
from glean_gepa.judge_metrics_util import JudgeAnalysis
from glean_gepa.prompt_constants import RULES_EXT_KEY
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter, _StartedPair
from glean_gepa.tool_match_util import (
    EvalRunTeacherStudentMatchAnalysis,
    NoComparedEvalEntriesError,
    TeacherStudentMatchEntryMetrics,
    TeacherStudentMatchMetrics,
)

EVAL_SET = {
    "eval_set_name": "Glean Chat V2 Medium",
    "eval_set_version": "20260806",
    "deployment_ids": ["scio-prod"],
    "status": "active",
}
FOCUSED_BATCH = [
    {
        **EVAL_SET,
        "eval_entry_ids": ["source-entry"],
        "focused_eval_set_name": "gepa-high-signal-glean-chat-v2-medium",
        "focused_eval_set_version": "20260806_hs_abc",
        "source_teacher_eval_run_id": "parent-teacher",
        "source_entry_ids_by_focused_id": {"fresh-1": "source-entry"},
    }
]


def _tool_match_cache_key(
    teacher_eval_id: str,
    student_eval_id: str,
    pairs: tuple[tuple[str, str], ...] = (),
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    """Mirror the key the adapter builds, including the sort on the entry pairs."""
    return (teacher_eval_id, student_eval_id, tuple(sorted(pairs)))


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
        student_model="claude_sonnet",
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
) -> EvalRunTeacherStudentMatchAnalysis:
    per_entry = {}
    if compared_entries:
        per_entry["entry-1"] = TeacherStudentMatchEntryMetrics(
            entry_id="entry-1",
            student_tools=("search",),
            teacher_tools=("read",),
            tools_match=False,
        )
    matching = sum(1 for metrics in per_entry.values() if metrics.tools_match)
    return EvalRunTeacherStudentMatchAnalysis(
        teacher_eval_id=teacher_eval_id,
        student_eval_id=student_eval_id,
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
        aggregate=TeacherStudentMatchMetrics(
            teacher_eval_id=teacher_eval_id,
            student_eval_id=student_eval_id,
            compared_entries=len(per_entry),
            matching_entries=matching,
            tool_match_rate=(matching / len(per_entry)) if per_entry else 0.0,
        ),
        per_entry=per_entry,
        high_signal_entry_ids=tuple(per_entry),
    )


def test_teacher_and_student_runs_are_created_before_waiting():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    adapter = _teacher_student_adapter(evalcli)

    with patch.object(adapter, "_get_or_fetch_teacher_student_match_analysis", return_value=_tool_match_analysis()):
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

    with patch.object(adapter, "_get_or_fetch_teacher_student_match_analysis", return_value=_tool_match_analysis()):
        adapter.evaluate(batch, {"WRITING_CODE": "test prompt"}, capture_traces=False)

    _assert_all_creates_before_waits(events, n_creates=4, n_waits=4)


def test_evaluate_many_starts_all_candidate_runs_before_waiting():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    adapter = _teacher_student_adapter(evalcli)

    with patch.object(adapter, "_get_or_fetch_teacher_student_match_analysis", return_value=_tool_match_analysis()):
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
    student_creates = [event for event in events if event.startswith("create:") and "_claude_sonnet_" in event]
    assert len(teacher_creates) == 1
    assert len(student_creates) == 2


def test_batch_evaluate_reuses_parent_teacher_and_runs_students_on_focused_set():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    adapter = _teacher_student_adapter(evalcli)

    with patch.object(adapter, "_get_or_fetch_teacher_student_match_analysis", return_value=_tool_match_analysis()):
        adapter.batch_evaluate(
            [
                ({"WRITING_CODE": "prompt a"}, FOCUSED_BATCH),
                ({"WRITING_CODE": "prompt b"}, FOCUSED_BATCH),
            ],
            capture_traces=True,
        )

    _assert_all_creates_before_waits(events, n_creates=2, n_waits=2)
    teacher_creates = [event for event in events if event.startswith("create:") and "_gpt_" in event]
    student_creates = [event for event in events if event.startswith("create:") and "_claude_sonnet_" in event]
    assert teacher_creates == []
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


def test_get_or_fetch_teacher_student_match_analysis_returns_empty_without_bigquery():
    adapter = _teacher_student_adapter(MagicMock())

    analysis = adapter._get_or_fetch_teacher_student_match_analysis("teacher-1", "student-1")

    assert analysis.teacher_eval_id == "teacher-1"
    assert analysis.student_eval_id == "student-1"
    assert analysis.per_entry == {}
    assert analysis.aggregate.compared_entries == 0


def test_get_or_fetch_teacher_student_match_analysis_caches_fetch():
    adapter = _teacher_student_adapter(MagicMock())
    adapter.bigquery_client = MagicMock()
    fetched = EvalRunTeacherStudentMatchAnalysis(
        teacher_eval_id="teacher-1",
        student_eval_id="student-1",
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
        aggregate=TeacherStudentMatchMetrics(
            teacher_eval_id="teacher-1",
            student_eval_id="student-1",
            compared_entries=1,
            matching_entries=0,
            tool_match_rate=0.0,
        ),
        per_entry={},
        high_signal_entry_ids=("entry-1",),
    )

    with patch(
        "glean_gepa.teacher_student_adapter.fetch_eval_run_teacher_student_match_analysis",
        return_value=fetched,
    ) as fetch:
        first = adapter._get_or_fetch_teacher_student_match_analysis("teacher-1", "student-1")
        second = adapter._get_or_fetch_teacher_student_match_analysis("teacher-1", "student-1")

    fetch.assert_called_once_with(
        adapter.bigquery_client,
        teacher_eval_id="teacher-1",
        student_eval_id="student-1",
        lookback_days=adapter.agentspan_lookback_days,
        entry_id_pairs=None,
    )
    assert first is fetched
    assert second is fetched


def test_finish_batch_evals_uses_tool_match_and_completeness():
    adapter = _teacher_student_adapter(MagicMock())
    analysis = EvalRunTeacherStudentMatchAnalysis(
        teacher_eval_id="teacher-1",
        student_eval_id="student-1",
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
        aggregate=TeacherStudentMatchMetrics(
            teacher_eval_id="teacher-1",
            student_eval_id="student-1",
            compared_entries=1,
            matching_entries=0,
            tool_match_rate=0.5,
        ),
        per_entry={
            "entry-1": TeacherStudentMatchEntryMetrics(
                entry_id="entry-1",
                student_tools=("search",),
                teacher_tools=("read",),
                tools_match=False,
            )
        },
        high_signal_entry_ids=("entry-1",),
    )
    adapter._teacher_student_match_cache[_tool_match_cache_key("teacher-1", "student-1")] = analysis
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

    assert result.scores == pytest.approx([0.7])
    assert result.objective_scores == [{"completeness": 1.0, "tool_alignment": 0.0, "grounding": 1.0}]
    assert result.outputs[0]["student_tool_events"] == ["search"]
    assert result.outputs[0]["teacher_tool_events"] == ["read"]
    assert result.summary == {
        "completeness": 1.0,
        "tool_alignment": 0.0,
        "grounding": 1.0,
        "teacher_completeness": 0.9,
    }
    assert result.trajectories is not None
    assert result.trajectories[0]["score"] == pytest.approx(0.7)


def test_high_signal_eval_runs_only_student_on_focused_set():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    adapter = _teacher_student_adapter(evalcli)

    with patch.object(
        adapter, "_get_or_fetch_teacher_student_match_analysis", return_value=_tool_match_analysis()
    ) as fetch:
        adapter.evaluate(FOCUSED_BATCH, {"WRITING_CODE": "test prompt"}, capture_traces=False)

    creates = [call.kwargs for call in evalcli.create_eval_run.call_args_list]
    assert len(creates) == 1
    assert creates[0]["eval_set_name"] == "gepa-high-signal-glean-chat-v2-medium"
    assert creates[0]["eval_set_version"] == "20260806_hs_abc"
    assert creates[0]["eval_run_id"].startswith("gepa_high_signal_")
    assert "_claude_sonnet_" in creates[0]["eval_run_id"]
    fetch.assert_called()
    assert fetch.call_args.kwargs["entry_id_pairs"] == [("source-entry", "fresh-1")]


def test_high_signal_eval_requires_parent_teacher_eval_id():
    adapter = _teacher_student_adapter(MagicMock())
    batch = [
        {
            **EVAL_SET,
            "eval_entry_ids": ["source-entry"],
            "focused_eval_set_name": "gepa-high-signal-glean-chat-v2-medium",
            "focused_eval_set_version": "20260806_hs_abc",
        }
    ]

    with pytest.raises(RuntimeError, match="parent teacher eval id"):
        adapter.evaluate(batch, {"WRITING_CODE": "test prompt"}, capture_traces=False)


def test_finish_focused_eval_uses_requested_entry_denominator():
    adapter = _teacher_student_adapter(MagicMock())
    analysis = EvalRunTeacherStudentMatchAnalysis(
        teacher_eval_id="teacher-1",
        student_eval_id="student-1",
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
        aggregate=TeacherStudentMatchMetrics(
            teacher_eval_id="teacher-1",
            student_eval_id="student-1",
            compared_entries=2,
            matching_entries=1,
            tool_match_rate=0.5,
        ),
        per_entry={
            "fresh-1": TeacherStudentMatchEntryMetrics(
                entry_id="fresh-1",
                student_tools=("read",),
                teacher_tools=("read",),
                tools_match=True,
            ),
            "fresh-2": TeacherStudentMatchEntryMetrics(
                entry_id="fresh-2",
                student_tools=("search",),
                teacher_tools=("read",),
                tools_match=False,
            ),
        },
        high_signal_entry_ids=("fresh-1", "fresh-2"),
    )
    adapter._teacher_student_match_cache[_tool_match_cache_key("teacher-1", "student-1")] = analysis
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
    assert "avg_tool_levenshtein" not in result.summary
    assert [score["tool_alignment"] for score in result.objective_scores] == [1.0, 0.0]


def test_finish_focused_eval_uses_student_entry_id_for_completeness():
    adapter = _teacher_student_adapter(MagicMock())
    analysis = EvalRunTeacherStudentMatchAnalysis(
        teacher_eval_id="parent-teacher",
        student_eval_id="student-1",
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
        aggregate=TeacherStudentMatchMetrics(
            teacher_eval_id="parent-teacher",
            student_eval_id="student-1",
            compared_entries=1,
            matching_entries=1,
            tool_match_rate=1.0,
        ),
        per_entry={
            "source-1": TeacherStudentMatchEntryMetrics(
                entry_id="source-1",
                student_tools=("read",),
                teacher_tools=("read",),
                tools_match=True,
                student_entry_id="fresh-1",
            )
        },
        high_signal_entry_ids=(),
    )
    adapter._teacher_student_match_cache[
        _tool_match_cache_key("parent-teacher", "student-1", (("source-1", "fresh-1"),))
    ] = analysis
    adapter._judge_cache[("student-1", COMPLETENESS_JUDGE_TYPE)] = JudgeAnalysis(
        eval_id="student-1", aggregate=0.0, per_entry={"fresh-1": 1.0}, judge_type=COMPLETENESS_JUDGE_TYPE
    )
    adapter._judge_cache[("parent-teacher", COMPLETENESS_JUDGE_TYPE)] = JudgeAnalysis(
        eval_id="parent-teacher", aggregate=0.9, per_entry={"source-1": 0.9}, judge_type=COMPLETENESS_JUDGE_TYPE
    )
    result = adapter._finish_batch_evals(
        [
            _StartedPair(
                al_data_inst={
                    **EVAL_SET,
                    "eval_entry_ids": ["source-1"],
                    "source_entry_ids_by_focused_id": {"fresh-1": "source-1"},
                    "source_teacher_eval_run_id": "parent-teacher",
                },
                teacher_eval_id="parent-teacher",
                student_eval_id="student-1",
            )
        ],
        capture_traces=True,
    )

    assert result.objective_scores == [{"completeness": 1.0, "tool_alignment": 1.0, "grounding": 1.0}]
    assert result.outputs[0]["entry_id"] == "source-1"
    assert result.outputs[0]["teacher_eval_run_id"] == "parent-teacher"
    assert result.summary is not None
    assert result.summary["teacher_completeness"] == pytest.approx(0.9)


def test_tool_match_analysis_survives_a_restart(tmp_path):
    """The Agentspan window is measured back from today, so an unpersisted analysis
    has to be refetched forever and eventually ages out of the lookback."""
    cache_file = str(tmp_path / "adapter.json")
    analysis = _tool_match_analysis()

    first = _teacher_student_adapter(MagicMock(), cache_file=cache_file)
    first._teacher_student_match_cache[_tool_match_cache_key("teacher-1", "student-1")] = analysis
    first._save_cache()

    second = _teacher_student_adapter(MagicMock(), cache_file=cache_file)
    restored = second._teacher_student_match_cache.get(_tool_match_cache_key("teacher-1", "student-1"))

    assert restored is not None
    assert restored.aggregate.compared_entries == analysis.aggregate.compared_entries
    assert restored.aggregate.tool_match_rate == analysis.aggregate.tool_match_rate
    assert restored.start_date == analysis.start_date and restored.end_date == analysis.end_date
    assert set(restored.per_entry) == set(analysis.per_entry)
    entry = restored.per_entry["entry-1"]
    assert entry.student_tools == ("search",) and entry.teacher_tools == ("read",)
    assert entry.tools_match is False


def test_analysis_writes_hold_the_cache_lock(tmp_path):
    """Child screens run in a thread pool while _save_cache iterates these stores.

    An unlocked write races that iteration; _save_cache catches the resulting
    "dictionary changed size during iteration" and only logs it, so the cost is a
    silently dropped cache write. Reproducing that interleaving is timing-dependent,
    so assert the invariant that prevents it instead.
    """

    class _LockAssertingStore(dict):
        def __init__(self, lock):
            super().__init__()
            self._lock = lock
            self.unlocked_writes: list[object] = []

        def __setitem__(self, key, value):
            if not self._lock._is_owned():
                self.unlocked_writes.append(key)
            super().__setitem__(key, value)

    adapter = _teacher_student_adapter(MagicMock(), cache_file=str(tmp_path / "adapter.json"))
    store = _LockAssertingStore(adapter._cache_lock)

    adapter._store_analysis(store, _tool_match_cache_key("teacher-1", "student-1"), _tool_match_analysis())

    assert store.unlocked_writes == []
    assert len(store) == 1


def test_a_zero_entry_tool_match_analysis_is_not_pinned(tmp_path):
    """Zero compared entries means the shard window missed the run, not that the
    models agreed on nothing -- caching it would make the miss permanent."""
    cache_file = str(tmp_path / "adapter.json")
    first = _teacher_student_adapter(MagicMock(), cache_file=cache_file)
    first._teacher_student_match_cache[_tool_match_cache_key("teacher-1", "student-1")] = _tool_match_analysis(
        compared_entries=0
    )
    first._save_cache()

    second = _teacher_student_adapter(MagicMock(), cache_file=cache_file)

    assert second._teacher_student_match_cache == {}


def test_focused_entry_pairs_are_part_of_the_cache_key(tmp_path):
    """A focused run remaps entry ids, so its analysis must not be served to a
    full-eval lookup of the same pair."""
    cache_file = str(tmp_path / "adapter.json")
    first = _teacher_student_adapter(MagicMock(), cache_file=cache_file)
    focused_key = _tool_match_cache_key("teacher-1", "student-1", (("src-1", "fresh-1"),))
    first._teacher_student_match_cache[focused_key] = _tool_match_analysis()
    first._save_cache()

    second = _teacher_student_adapter(MagicMock(), cache_file=cache_file)

    assert second._teacher_student_match_cache.get(_tool_match_cache_key("teacher-1", "student-1")) is None
    assert second._teacher_student_match_cache.get(focused_key) is not None


def test_scoring_follows_the_configured_composite_and_judge_names():
    """A judge signal the adapter has no special knowledge of still scores.

    ``answer_quality`` is fed by the CORRECTNESS judge, so this only works if the
    composite dimension travels with the judge rather than being hardcoded.
    """
    adapter = TeacherStudentAdapter(
        runner=ALRunner(evalcli=MagicMock()),
        teacher_model="gpt",
        student_model="claude_sonnet",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        pointwise_judges=(PointwiseJudge("answer_quality", "CORRECTNESS", "{}"),),
        composite_weights={"answer_quality": 0.75, "tool_alignment": 0.25},
        constant_scores={},
    )
    adapter._teacher_student_match_cache[_tool_match_cache_key("teacher-1", "student-1")] = _tool_match_analysis()
    adapter._judge_cache[("student-1", "CORRECTNESS")] = JudgeAnalysis(
        eval_id="student-1", aggregate=0.0, per_entry={"entry-1": 0.8}, judge_type="CORRECTNESS"
    )

    result = adapter._finish_batch_evals(
        [_StartedPair(al_data_inst={**EVAL_SET}, teacher_eval_id="teacher-1", student_eval_id="student-1")],
        capture_traces=True,
    )

    # tools_match is False, so tool_alignment is 0.0: 0.75 * 0.8 + 0.25 * 0.0.
    assert result.objective_scores == [{"answer_quality": 0.8, "tool_alignment": 0.0}]
    assert result.scores == [pytest.approx(0.6)]
    assert result.summary is not None
    assert result.summary["teacher_answer_quality"] == pytest.approx(0.0)
    assert "grounding" not in result.summary
    assert "teacher_completeness" not in result.summary


def test_composite_weighting_a_dimension_the_adapter_cannot_score_raises():
    with pytest.raises(ValueError, match="cannot score"):
        TeacherStudentAdapter(
            runner=ALRunner(evalcli=MagicMock()),
            teacher_model="gpt",
            student_model="claude_sonnet",
            thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
            composite_weights={"tool_alignment": 0.5, "shell_success_rate": 0.5},
        )


def test_finish_focused_eval_does_not_raise_when_no_entries_were_compared():
    adapter = _teacher_student_adapter(MagicMock())
    adapter._teacher_student_match_cache[_tool_match_cache_key("teacher-1", "student-1")] = _tool_match_analysis(
        compared_entries=0
    )
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
    adapter._teacher_student_match_cache[_tool_match_cache_key("teacher-1", "student-1")] = _tool_match_analysis(
        compared_entries=0
    )

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


def test_launched_eval_ids_are_tracked_in_flight_before_wait():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    adapter = _teacher_student_adapter(evalcli)

    with patch.object(adapter.runner, "wait", side_effect=TimeoutError("terminal timed out")):
        with pytest.raises(TimeoutError):
            adapter.evaluate([EVAL_SET], {"WRITING_CODE": "test prompt"}, capture_traces=False)

    create_ids = [event.split(":", 1)[1] for event in events if event.startswith("create:")]
    assert sorted(adapter.runner._in_flight) == sorted(create_ids)


def test_in_flight_eval_ids_resume_wait_instead_of_recreating():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    adapter = _teacher_student_adapter(evalcli)
    with patch.object(adapter.runner, "wait", side_effect=TimeoutError("terminal timed out")):
        with pytest.raises(TimeoutError):
            adapter.evaluate([EVAL_SET], {"WRITING_CODE": "test prompt"}, capture_traces=False)
    launched_ids = sorted(adapter.runner._in_flight)

    with patch.object(adapter, "_get_or_fetch_teacher_student_match_analysis", return_value=_tool_match_analysis()):
        adapter.evaluate([EVAL_SET], {"WRITING_CODE": "test prompt"}, capture_traces=False)

    assert evalcli.create_eval_run.call_count == 2
    waited_ids = sorted(event.split(":", 1)[1] for event in events if event.startswith("wait:"))
    assert waited_ids == launched_ids
    assert adapter.runner._in_flight == {}


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

    with patch.object(adapter, "_get_or_fetch_teacher_student_match_analysis", return_value=_tool_match_analysis()):
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

    with patch.object(adapter, "_get_or_fetch_teacher_student_match_analysis", return_value=_tool_match_analysis()):
        adapter.evaluate_many(
            [EVAL_SET],
            [{"WRITING_CODE": "prompt a"}, {"WRITING_CODE": "prompt b"}],
            capture_traces=False,
        )

    judged_eval_ids = [event.split(":", 1)[1] for event in events if event.startswith("judge-create:")]
    # One shared teacher eval plus one student eval per candidate.
    assert len(judged_eval_ids) == 3
    assert len(set(judged_eval_ids)) == 3


def test_high_signal_batch_keeps_every_first_tool_mismatch():
    adapter = _teacher_student_adapter(MagicMock())
    trajectories = []
    for index in range(25):
        trajectories.append(
            {
                "data": EVAL_SET,
                "output": {
                    "entry_id": f"mismatch-{index:02d}",
                    "student_tool_events": ["search"],
                    "teacher_tool_events": ["read"],
                },
                "score": 0.5,
                "objective_scores": {"tool_alignment": 0.0},
            }
        )
    trajectories.extend(
        [
            {
                "data": EVAL_SET,
                "output": {
                    "entry_id": "later-tools-differ",
                    "student_tool_events": ["search", "write"],
                    "teacher_tool_events": ["search", "read"],
                },
                "score": 0.5,
                "objective_scores": {"tool_alignment": 1.0},
            },
            {
                "data": EVAL_SET,
                "output": {
                    "entry_id": "perfect",
                    "student_tool_events": ["search"],
                    "teacher_tool_events": ["search"],
                },
                "score": 1.0,
                "objective_scores": {"tool_alignment": 1.0},
            },
        ]
    )
    focused = adapter.high_signal_batch(
        GleanEvaluationBatch(outputs=[], scores=[], trajectories=trajectories, objective_scores=[])
    )

    assert focused[0]["eval_entry_ids"] == [f"mismatch-{index:02d}" for index in range(25)]
    assert "later-tools-differ" not in focused[0]["eval_entry_ids"]
    assert "perfect" not in focused[0]["eval_entry_ids"]
    assert "source_teacher_eval_run_id" not in focused[0]


def test_high_signal_batch_attaches_parent_teacher_eval_id():
    adapter = _teacher_student_adapter(MagicMock())
    focused = adapter.high_signal_batch(
        GleanEvaluationBatch(
            outputs=[],
            scores=[],
            trajectories=[
                {
                    "data": EVAL_SET,
                    "output": {
                        "entry_id": "mismatch-0",
                        "student_tool_events": ["search"],
                        "teacher_tool_events": ["read"],
                        "teacher_eval_run_id": "parent-teacher",
                    },
                    "score": 0.5,
                    "objective_scores": {"tool_alignment": 0.0},
                }
            ],
            objective_scores=[],
            eval_run_ids=[
                {
                    "eval_set_name": EVAL_SET["eval_set_name"],
                    "eval_set_version": EVAL_SET["eval_set_version"],
                    "student_eval_run_id": "parent-student",
                    "teacher_eval_run_id": "parent-teacher",
                }
            ],
        )
    )

    assert focused[0]["eval_entry_ids"] == ["mismatch-0"]
    assert focused[0]["source_teacher_eval_run_id"] == "parent-teacher"


def _mismatch_trajectory(entry_id: str, teacher_tools: list[str], student_tools: list[str], *, score: float = 0.5):
    return {
        "data": EVAL_SET,
        "output": {
            "entry_id": entry_id,
            "deployment_id": "scio-prod",
            "query": "q",
            "student_answer": "",
            "teacher_answer": "",
            "student_tool_events": student_tools,
            "teacher_tool_events": teacher_tools,
        },
        "score": score,
        "objective_scores": {"tool_alignment": 0.0 if teacher_tools[:1] != student_tools[:1] else 1.0},
    }


def test_make_reflective_dataset_uses_most_frequent_first_tool_mismatch_groups():
    adapter = _teacher_student_adapter(MagicMock())
    trajectories = (
        [_mismatch_trajectory(f"xy-{i}", ["x"], ["y"]) for i in range(12)]
        + [_mismatch_trajectory(f"yx-{i}", ["y"], ["x"]) for i in range(8)]
        + [_mismatch_trajectory(f"ab-{i}", ["a"], ["b"]) for i in range(6)]
        + [_mismatch_trajectory(f"cd-{i}", ["c"], ["d"]) for i in range(5)]
        + [_mismatch_trajectory("match", ["search", "read"], ["search", "write"], score=1.0)]
    )
    examples = adapter.make_reflective_dataset(
        {"WRITING_CODE": "prompt"},
        GleanEvaluationBatch(outputs=[], scores=[], trajectories=trajectories, objective_scores=[]),
        ["WRITING_CODE"],
        k=8,
        error_hamming_distance_k=1,
    )["WRITING_CODE"]
    entry_ids = [example["Inputs"]["entry_id"] for example in examples]
    assert len(entry_ids) == 20
    assert all(entry_id.startswith(("xy-", "yx-")) for entry_id in entry_ids)
    assert "match" not in entry_ids
    assert not any(entry_id.startswith(("ab-", "cd-")) for entry_id in entry_ids)
    assert examples[0]["Feedback"].startswith("First-tool mismatch: teacher used x and student used y.")

    oversized = adapter.make_reflective_dataset(
        {"WRITING_CODE": "prompt"},
        GleanEvaluationBatch(
            outputs=[],
            scores=[],
            trajectories=[_mismatch_trajectory(f"xy-{i}", ["x"], ["y"]) for i in range(35)],
            objective_scores=[],
        ),
        ["WRITING_CODE"],
        k=8,
    )["WRITING_CODE"]
    assert [example["Inputs"]["entry_id"] for example in oversized] == [f"xy-{i}" for i in range(35)]


def test_make_reflective_dataset_filters_core_tool_module_to_matching_mismatches():
    adapter = _teacher_student_adapter(MagicMock())
    trajectories = [_mismatch_trajectory(f"search-{i}", ["Glean Search"], ["Discover"]) for i in range(12)] + [
        _mismatch_trajectory(f"read-{i}", ["Glean Document Reader"], ["todo_write"]) for i in range(8)
    ]
    eval_batch = GleanEvaluationBatch(outputs=[], scores=[], trajectories=trajectories, objective_scores=[])

    examples = adapter.make_reflective_dataset(
        {"FULL_PROMPT": "prompt"},
        eval_batch,
        ["FULL_PROMPT", "glean_search", "discover", "glean_document_reader"],
        k=8,
    )

    assert len(examples["FULL_PROMPT"]) == 20
    search_ids = [example["Inputs"]["entry_id"] for example in examples["glean_search"]]
    discover_ids = [example["Inputs"]["entry_id"] for example in examples["discover"]]
    reader_ids = [example["Inputs"]["entry_id"] for example in examples["glean_document_reader"]]
    assert search_ids == [f"search-{i}" for i in range(12)]
    assert discover_ids == search_ids
    assert reader_ids == [f"read-{i}" for i in range(8)]
    assert examples["glean_search"][0]["Feedback"].startswith(
        "First-tool mismatch: teacher used Glean Search and student used Discover."
    )


def test_make_reflective_dataset_filters_rules_ext_to_non_core_mismatches():
    adapter = _teacher_student_adapter(MagicMock())
    trajectories = [_mismatch_trajectory(f"search-{i}", ["Glean Search"], ["Discover"]) for i in range(12)] + [
        _mismatch_trajectory(f"write-{i}", ["Write"], []) for i in range(8)
    ]
    examples = adapter.make_reflective_dataset(
        {"FULL_PROMPT": "prompt", RULES_EXT_KEY: ""},
        GleanEvaluationBatch(outputs=[], scores=[], trajectories=trajectories, objective_scores=[]),
        [RULES_EXT_KEY, "glean_search"],
        k=8,
    )
    write_ids = [example["Inputs"]["entry_id"] for example in examples[RULES_EXT_KEY]]
    search_ids = [example["Inputs"]["entry_id"] for example in examples["glean_search"]]
    assert write_ids == [f"write-{i}" for i in range(8)]
    assert search_ids == [f"search-{i}" for i in range(12)]
    assert examples[RULES_EXT_KEY][0]["Feedback"].startswith(
        "First-tool mismatch: teacher used Write and student used (none)."
    )
