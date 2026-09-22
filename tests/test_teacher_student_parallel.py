from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from glean_gepa.al_adapter import ALRunner, Thresholds
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evalcli_client import CORRECTNESS_JUDGE_TYPE
from glean_gepa.judge_metrics_util import JudgeAnalysis
from glean_gepa.objectives.utils.tool_match_util import (
    SKIPPED_TOOL_NAMES,
    TOOL_ALIGNMENT_OBJECTIVE,
    EvalRunToolMatchAnalysis,
    NoComparedEvalEntriesError,
    ToolMatchEntryMetrics,
    ToolMatchMetrics,
)
from glean_gepa.prompt_constants import RULES_EXT_KEY
from glean_gepa.teacher_student_adapter import (
    CORRECTNESS_DIMENSION,
    CORRECTNESS_JUDGE,
    TeacherStudentAdapter,
    _entry_queries_from_listing,
    _fetch_entry_queries,
    _StartedPair,
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
    evalcli.get_eval_metrics.return_value = {
        "judgeMetrics": {"totalEntries": 1, "missingEntries": 0, "CORRECTNESS": {"passRate": 0.0, "sampleSize": 1}}
    }
    return evalcli


def _teacher_student_adapter(
    evalcli: MagicMock, cache_file: str | None = None, *, judge_correctness: bool = False
) -> TeacherStudentAdapter:
    """Shipped evaluate tests stay on tool_alignment; judge-path tests opt into
    correctness vs the teacher."""
    judge_kwargs = (
        {
            "pairwise_judges": [CORRECTNESS_JUDGE],
            "composite_weights": {CORRECTNESS_DIMENSION: 0.5, TOOL_ALIGNMENT_OBJECTIVE: 0.5},
        }
        if judge_correctness
        else {}
    )
    return TeacherStudentAdapter(
        runner=ALRunner(evalcli=evalcli),
        teacher_model="gpt",
        student_model="claude_sonnet",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        cache_file=cache_file,
        **judge_kwargs,
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
            tool_match_rate=(matching / len(per_entry)) if per_entry else 0.0,
        ),
        per_entry=per_entry,
        high_signal_entry_ids=tuple(per_entry),
    )


def test_teacher_and_student_runs_are_created_before_waiting():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    adapter = _teacher_student_adapter(evalcli)

    with patch.object(adapter, "_get_or_fetch_analysis", return_value=_tool_match_analysis()):
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

    with patch.object(adapter, "_get_or_fetch_analysis", return_value=_tool_match_analysis()):
        adapter.evaluate(batch, {"WRITING_CODE": "test prompt"}, capture_traces=False)

    _assert_all_creates_before_waits(events, n_creates=4, n_waits=4)


def test_evaluate_many_starts_all_candidate_runs_before_waiting():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    adapter = _teacher_student_adapter(evalcli)

    with patch.object(adapter, "_get_or_fetch_analysis", return_value=_tool_match_analysis()):
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

    with patch.object(adapter, "_get_or_fetch_analysis", return_value=_tool_match_analysis()):
        adapter.batch_evaluate(
            [
                ({"WRITING_CODE": "prompt a"}, batch),
                ({"WRITING_CODE": "prompt b"}, batch),
            ],
            capture_traces=True,
        )

    _assert_all_creates_before_waits(events, n_creates=3, n_waits=3)
    teacher_creates = [event for event in events if event.startswith("create:") and "_gpt_" in event]
    student_creates = [event for event in events if event.startswith("create:") and "_claude_sonnet_" in event]
    assert len(teacher_creates) == 1
    assert len(student_creates) == 2


def test_batch_evaluate_overlaps_children_that_only_differ_by_cached_eval_ids():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    adapter = _teacher_student_adapter(evalcli)
    base = {
        **EVAL_SET,
        "eval_entry_ids": ["source-entry"],
        "focused_eval_set_name": "gepa-high-signal-glean-chat-v2-medium",
        "focused_eval_set_version": "20260806_hs_abc",
    }

    with patch.object(adapter, "_get_or_fetch_analysis", return_value=_tool_match_analysis()):
        adapter.batch_evaluate(
            [
                ({"WRITING_CODE": "prompt a"}, [{**base, "cached_student_eval_run_id": "child-a"}]),
                ({"WRITING_CODE": "prompt b"}, [{**base, "cached_student_eval_run_id": "child-b"}]),
            ],
            capture_traces=True,
        )

    _assert_all_creates_before_waits(events, n_creates=1, n_waits=1)


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


def test_get_or_fetch_analysis_returns_empty_without_bigquery():
    adapter = _teacher_student_adapter(MagicMock())

    analysis = adapter._get_or_fetch_analysis("teacher-1", "student-1")

    assert analysis.teacher_eval_id == "teacher-1"
    assert analysis.student_eval_id == "student-1"
    assert analysis.per_entry == {}
    assert analysis.aggregate.compared_entries == 0


def test_get_or_fetch_analysis_caches_fetch():
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
            tool_match_rate=0.0,
        ),
        per_entry={},
        high_signal_entry_ids=("entry-1",),
    )

    with patch(
        "glean_gepa.objectives.tool_match.fetch_eval_run_tool_match_analysis",
        return_value=fetched,
    ) as fetch:
        first = adapter._get_or_fetch_analysis("teacher-1", "student-1")
        second = adapter._get_or_fetch_analysis("teacher-1", "student-1")

    fetch.assert_called_once_with(
        adapter.bigquery_client,
        teacher_eval_id="teacher-1",
        student_eval_id="student-1",
        lookback_days=adapter.agentspan_lookback_days,
        evalcli=adapter.objective.evalcli,
        skip_tools=SKIPPED_TOOL_NAMES,
    )
    assert first is fetched
    assert second is fetched


def test_validation_only_skips_action_input_evalcli():
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
            matching_entries=1,
            tool_match_rate=1.0,
        ),
        per_entry={},
        high_signal_entry_ids=(),
    )

    with patch(
        "glean_gepa.objectives.tool_match.fetch_eval_run_tool_match_analysis",
        return_value=fetched,
    ) as fetch:
        adapter._finish_batch_evals(
            [
                _StartedPair(
                    al_data_inst={**EVAL_SET, "validation_only": True},
                    teacher_eval_id="teacher-1",
                    student_eval_id="student-1",
                )
            ],
            capture_traces=False,
        )

    fetch.assert_called_once_with(
        adapter.bigquery_client,
        teacher_eval_id="teacher-1",
        student_eval_id="student-1",
        lookback_days=adapter.agentspan_lookback_days,
        evalcli=None,
        skip_tools=SKIPPED_TOOL_NAMES,
    )


def test_finish_batch_evals_uses_tool_match_and_correctness():
    adapter = _teacher_student_adapter(MagicMock(), judge_correctness=True)
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
            tool_match_rate=0.5,
        ),
        per_entry={
            "entry-1": ToolMatchEntryMetrics(
                entry_id="entry-1",
                student_tools=("search",),
                teacher_tools=("read",),
                tools_match=False,
            )
        },
        high_signal_entry_ids=("entry-1",),
    )
    adapter._analysis_cache[("teacher-1", "student-1")] = analysis
    adapter._judge_cache[("student-1", "teacher-1", CORRECTNESS_JUDGE_TYPE)] = JudgeAnalysis(
        eval_id="student-1", aggregate=1.0, per_entry={"entry-1": 1.0}, judge_type=CORRECTNESS_JUDGE_TYPE
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

    assert result.scores == pytest.approx([0.5])
    assert result.objective_scores == [{"correctness": 1.0, "tool_alignment": 0.0}]
    assert result.outputs[0]["student_tool_events"] == ["search"]
    assert result.outputs[0]["teacher_tool_events"] == ["read"]
    assert result.summary == {
        "correctness": 1.0,
        "tool_alignment": 0.0,
    }
    assert result.trajectories is not None
    assert result.trajectories[0]["score"] == pytest.approx(0.5)


def _finish_one_pair(adapter, *, validation_only: bool = False):
    adapter._analysis_cache[("teacher-1", "student-1")] = _tool_match_analysis()
    al_data_inst = {**EVAL_SET, **({"validation_only": True} if validation_only else {})}
    return adapter._finish_batch_evals(
        [
            _StartedPair(
                al_data_inst=al_data_inst,
                teacher_eval_id="teacher-1",
                student_eval_id="student-1",
                eval_set_name="Glean Chat V2 Medium",
                eval_set_version="20260907",
            )
        ],
        capture_traces=True,
    )


def test_entry_queries_reads_both_listing_shapes_and_skips_unusable_rows():
    resolved = _entry_queries_from_listing(
        [
            {"id": "entry-1", "input": {"query": "how many employees in Belgium"}},
            # Some listings carry the query at the top level rather than under input.
            {"id": "entry-2", "query": "  draft the BNZ follow-up  "},
            # Unusable: no id to join on, no query text, or a non-string payload.
            {"input": {"query": "orphaned"}},
            {"id": "entry-3", "input": {"query": "   "}},
            {"id": "entry-4", "input": {}},
            {"id": "entry-5", "input": {"query": {"text": "nested"}}},
            "not-a-mapping",
        ]
    )

    assert resolved == {
        "entry-1": "how many employees in Belgium",
        "entry-2": "draft the BNZ follow-up",
    }


def test_fetch_entry_queries_returns_empty_without_a_usable_listing():
    """Customer eval sets are PII-gated, so a refused listing must leave reflection on
    the stand-in rather than fail the batch, and a pair carrying no eval set must not
    spend an RPC to discover there is nothing to list."""
    refused = MagicMock()
    refused.list_eval_set_entries.side_effect = RuntimeError("403 PII-gated")
    assert _fetch_entry_queries(refused, eval_set_name="set", eval_set_version="20260907", deployment_ids=["x"]) == {}

    skipped = MagicMock()
    assert _fetch_entry_queries(skipped, eval_set_name="", eval_set_version="20260907", deployment_ids=[]) == {}
    assert _fetch_entry_queries(None, eval_set_name="set", eval_set_version="20260907", deployment_ids=[]) == {}
    skipped.list_eval_set_entries.assert_not_called()


def test_real_user_queries_join_per_entry_from_the_eval_set_that_ran():
    """Agentspan scrubs the query, so every example used to be labelled with the
    same ``eval_set:version`` string and no task was distinguishable."""
    evalcli = MagicMock()
    evalcli.list_eval_set_entries.return_value = [
        {"id": "entry-1", "input": {"query": "how many employees does Glean have in Belgium"}},
        {"id": "entry-other", "input": {"query": "unrelated"}},
    ]
    adapter = _teacher_student_adapter(evalcli)

    result = _finish_one_pair(adapter)
    _finish_one_pair(adapter)

    assert result.outputs[0]["query"] == "how many employees does Glean have in Belgium"
    assert result.trajectories is not None
    example = adapter.objective.build_reflective_example(RULES_EXT_KEY, result.trajectories[0], {})
    assert example["Inputs"]["query"] == "how many employees does Glean have in Belgium"
    # A focused batch runs against a generated high-signal set and scoring reports that
    # set's entry ids, so listing the batch's base version would not join.
    assert evalcli.list_eval_set_entries.call_args.kwargs == {
        "eval_set_name": "Glean Chat V2 Medium",
        "eval_set_version": "20260907",
        "deployment_ids": ["scio-prod"],
    }
    assert EVAL_SET["eval_set_version"] == "20260806"
    assert evalcli.list_eval_set_entries.call_count == 1, "listed once per version, not once per pair"

    # An entry the listing does not cover keeps the stand-in instead of borrowing
    # another entry's query.
    unlisted = MagicMock()
    unlisted.list_eval_set_entries.return_value = [{"id": "some-other-entry", "input": {"query": "case 007"}}]
    fallback = _finish_one_pair(_teacher_student_adapter(unlisted))
    assert fallback.outputs[0]["query"] == "Glean Chat V2 Medium:20260806"


def test_pii_gated_validation_sets_are_never_listed_and_keep_the_stand_in():
    evalcli = MagicMock()
    adapter = _teacher_student_adapter(evalcli)

    result = _finish_one_pair(adapter, validation_only=True)

    evalcli.list_eval_set_entries.assert_not_called()
    assert result.outputs[0]["query"] == "Glean Chat V2 Medium:20260806"


def test_full_validation_returns_one_row_per_eval_set_not_per_entry():
    """Full validation must stay aligned with the caller's batch.

    The engine has one val ID per eval set and zips it against these rows with
    ``strict=False``, so entry-level rows would record an arbitrary entry's 0/1
    outcome as the whole eval set's validation score.
    """
    adapter = _teacher_student_adapter(MagicMock(), judge_correctness=True)
    adapter._analysis_cache[("teacher-1", "student-1")] = EvalRunToolMatchAnalysis(
        teacher_eval_id="teacher-1",
        student_eval_id="student-1",
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
        aggregate=ToolMatchMetrics(
            teacher_eval_id="teacher-1",
            student_eval_id="student-1",
            compared_entries=4,
            matching_entries=3,
            tool_match_rate=0.75,
        ),
        per_entry={
            # entry-1 matches, so a per-entry row would score 1.0 and hide that
            # the eval set as a whole only reached 0.75.
            "entry-1": ToolMatchEntryMetrics(
                entry_id="entry-1", student_tools=("search",), teacher_tools=("search",), tools_match=True
            ),
            "entry-2": ToolMatchEntryMetrics(
                entry_id="entry-2", student_tools=("search",), teacher_tools=("read",), tools_match=False
            ),
        },
        high_signal_entry_ids=("entry-2",),
    )
    adapter._judge_cache[("student-1", "teacher-1", CORRECTNESS_JUDGE_TYPE)] = JudgeAnalysis(
        eval_id="student-1",
        aggregate=1.0,
        # Per-entry scores differ from the aggregate, so the eval-set row must pick
        # the aggregate rather than fall through to it on a missed lookup.
        per_entry={"entry-1": 0.2, "entry-2": 0.3},
        judge_type=CORRECTNESS_JUDGE_TYPE,
    )
    started = [_StartedPair(al_data_inst=EVAL_SET, teacher_eval_id="teacher-1", student_eval_id="student-1")]

    validation = adapter._finish_batch_evals(started, capture_traces=False)

    assert len(validation.scores) == len(started)
    assert validation.objective_scores == [{"correctness": 1.0, "tool_alignment": 0.75}]
    assert validation.scores == pytest.approx([0.5 * 1.0 + 0.5 * 0.75])
    assert validation.outputs[0]["entry_id"] == f"{EVAL_SET['eval_set_name']}:{EVAL_SET['eval_set_version']}"

    # Reflection still needs entry-level rows, so trace capture is unchanged.
    traced = adapter._finish_batch_evals(started, capture_traces=True)
    assert len(traced.scores) == 2
    assert [obj["tool_alignment"] for obj in traced.objective_scores or []] == [1.0, 0.0]


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

    with patch.object(adapter, "_get_or_fetch_analysis", return_value=_tool_match_analysis()):
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
            tool_match_rate=0.5,
        ),
        per_entry={
            "fresh-1": ToolMatchEntryMetrics(
                entry_id="fresh-1",
                student_tools=("read",),
                teacher_tools=("read",),
                tools_match=True,
            ),
            "fresh-2": ToolMatchEntryMetrics(
                entry_id="fresh-2",
                student_tools=("search",),
                teacher_tools=("read",),
                tools_match=False,
            ),
        },
        high_signal_entry_ids=("fresh-1", "fresh-2"),
    )
    adapter._analysis_cache[("teacher-1", "student-1")] = analysis
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


def test_finish_focused_eval_does_not_raise_when_no_entries_were_compared():
    adapter = _teacher_student_adapter(MagicMock(), judge_correctness=True)
    adapter._analysis_cache[("teacher-1", "student-1")] = _tool_match_analysis(compared_entries=0)
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
        "correctness": 0.0,
        "tool_alignment": 0.0,
    }


def test_finish_batch_evals_raises_when_no_entries_were_compared():
    adapter = _teacher_student_adapter(MagicMock())
    adapter._analysis_cache[("teacher-1", "student-1")] = _tool_match_analysis(compared_entries=0)

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

    with patch.object(adapter, "_get_or_fetch_analysis", return_value=_tool_match_analysis()):
        adapter.evaluate([EVAL_SET], {"WRITING_CODE": "test prompt"}, capture_traces=False)

    assert evalcli.create_eval_run.call_count == 2
    waited_ids = sorted(event.split(":", 1)[1] for event in events if event.startswith("wait:"))
    assert waited_ids == launched_ids
    assert adapter.runner._in_flight == {}


def _stub_correctness_judge(evalcli: MagicMock, events: list[str], *, score: float = 0.8) -> None:
    evalcli.find_judge_run_id.return_value = None

    def create_correctness(**kwargs):
        eval_run_id = kwargs["eval_run_id"]
        events.append(f"judge-create:{eval_run_id}:base={kwargs.get('base_eval_run_id')}")
        return f"judge-{eval_run_id}"

    evalcli.create_judge_run.side_effect = create_correctness
    evalcli.get_eval_metrics.return_value = {
        "judgeMetrics": {
            "totalEntries": 1,
            "missingEntries": 0,
            "CORRECTNESS": {"passRate": score, "sampleSize": 1},
        }
    }


def test_correctness_judge_scores_the_student_against_the_teacher():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    _stub_correctness_judge(evalcli, events)
    adapter = _teacher_student_adapter(evalcli, judge_correctness=True)

    with patch.object(adapter, "_get_or_fetch_analysis", return_value=_tool_match_analysis()):
        result = adapter.evaluate([EVAL_SET], {"WRITING_CODE": "test prompt"}, capture_traces=False)

    create_idxs = [i for i, event in enumerate(events) if event.startswith("create:")]
    wait_idxs = [i for i, event in enumerate(events) if event.startswith("wait:")]
    assert max(create_idxs) < min(wait_idxs)
    eval_run_ids = [event.split(":", 1)[1] for event in events if event.startswith("create:")]
    judge_creates = [event for event in events if event.startswith("judge-create:")]
    assert len(judge_creates) == 1
    assert events.index(f"wait:{eval_run_ids[0]}") < events.index(judge_creates[0])
    assert events.index(f"wait:{eval_run_ids[1]}") < events.index(judge_creates[0])
    create_kwargs = evalcli.create_judge_run.call_args.kwargs
    assert create_kwargs["eval_run_id"] != create_kwargs["base_eval_run_id"]
    assert create_kwargs["judge_type"] == CORRECTNESS_JUDGE_TYPE
    assert create_kwargs["base_eval_run_id"] in eval_run_ids
    assert create_kwargs["eval_run_id"] in eval_run_ids
    assert result.summary is not None
    assert result.summary["correctness"] == pytest.approx(0.8)
    assert "teacher_correctness" not in result.summary
    assert result.objective_scores[0]["correctness"] == pytest.approx(0.8)
    assert "completeness" not in result.objective_scores[0]


def test_correctness_judge_runs_once_per_student_across_candidates():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    _stub_correctness_judge(evalcli, events)
    adapter = _teacher_student_adapter(evalcli, judge_correctness=True)

    with patch.object(adapter, "_get_or_fetch_analysis", return_value=_tool_match_analysis()):
        adapter.evaluate_many(
            [EVAL_SET],
            [{"WRITING_CODE": "prompt a"}, {"WRITING_CODE": "prompt b"}],
            capture_traces=False,
        )

    judged_eval_ids = [event.split(":", 2)[1] for event in events if event.startswith("judge-create:")]
    # One pairwise judge per student eval; the shared teacher is the baseline, not judged.
    assert len(judged_eval_ids) == 2
    assert len(set(judged_eval_ids)) == 2


def test_evaluate_many_starts_each_pairwise_judge_when_that_pair_finishes():
    events: list[str] = []
    evalcli = _evalcli_with_ordered_events(events)
    _stub_correctness_judge(evalcli, events)

    def get_eval_metrics(eval_id, **_kwargs):
        events.append(f"metrics:{eval_id}")
        return {
            "judgeMetrics": {
                "totalEntries": 1,
                "missingEntries": 0,
                "CORRECTNESS": {"passRate": 0.8, "sampleSize": 1},
            }
        }

    evalcli.get_eval_metrics.side_effect = get_eval_metrics
    adapter = _teacher_student_adapter(evalcli, judge_correctness=True)

    with patch.object(adapter, "_get_or_fetch_analysis", return_value=_tool_match_analysis()):
        adapter.evaluate_many(
            [EVAL_SET],
            [{"WRITING_CODE": "prompt a"}, {"WRITING_CODE": "prompt b"}],
            capture_traces=False,
        )

    wait_ids = [event.split(":", 1)[1] for event in events if event.startswith("wait:")]
    teacher_id = next(eval_id for eval_id in wait_ids if "_gpt_" in eval_id)
    student_waits = [eval_id for eval_id in wait_ids if "_claude_sonnet_" in eval_id]
    assert len(student_waits) == 2
    first_student, second_student = student_waits
    assert events.index(f"judge-create:{first_student}:base={teacher_id}") < events.index(f"wait:{second_student}")
    last_judge = max(i for i, event in enumerate(events) if event.startswith("judge-create:"))
    first_metrics = min(i for i, event in enumerate(events) if event.startswith("metrics:"))
    assert last_judge < first_metrics


def test_pairwise_judge_cache_includes_the_teacher_baseline(tmp_path):
    evalcli = MagicMock()
    evalcli.find_judge_run_id.return_value = None
    evalcli.create_judge_run.side_effect = lambda **kwargs: f"judge-{kwargs['base_eval_run_id']}"
    cache_file = str(tmp_path / "adapter.json")
    adapter = _teacher_student_adapter(evalcli, cache_file=cache_file, judge_correctness=True)
    kwargs = {"judge_type": CORRECTNESS_JUDGE_TYPE, "run_params": "{}"}

    assert adapter._ensure_judge("student-1", base_eval_run_id="teacher-1", **kwargs) == "judge-teacher-1"
    assert adapter._ensure_judge("student-1", base_eval_run_id="teacher-2", **kwargs) == "judge-teacher-2"
    assert evalcli.create_judge_run.call_count == 2
    adapter._save_cache()

    reloaded = _teacher_student_adapter(evalcli, cache_file=cache_file, judge_correctness=True)
    assert reloaded._ensure_judge("student-1", base_eval_run_id="teacher-1", **kwargs) == "judge-teacher-1"
    assert evalcli.create_judge_run.call_count == 2


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
        k=20,
        error_hamming_distance_k=1,
    )["WRITING_CODE"]
    entry_ids = [example["Inputs"]["entry_id"] for example in examples]
    assert len(entry_ids) == 20
    assert all(entry_id.startswith(("xy-", "yx-")) for entry_id in entry_ids)
    assert "match" not in entry_ids
    assert not any(entry_id.startswith(("ab-", "cd-")) for entry_id in entry_ids)
    assert examples[0]["Feedback"].startswith("First-tool mismatch: teacher used x and student used y.")

    capped = adapter.make_reflective_dataset(
        {"WRITING_CODE": "prompt"},
        GleanEvaluationBatch(outputs=[], scores=[], trajectories=trajectories, objective_scores=[]),
        ["WRITING_CODE"],
        k=8,
    )["WRITING_CODE"]
    assert [example["Inputs"]["entry_id"] for example in capped] == [f"xy-{i}" for i in range(12)]

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
        {"WRITING_CODE": "prompt"},
        eval_batch,
        ["WRITING_CODE", "glean_search", "discover", "glean_document_reader"],
        k=20,
    )

    assert len(examples["WRITING_CODE"]) == 20
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
    # An empty student sequence means every span was a skipped one, so say so rather
    # than printing "(none)", which reflection read as a hole in the trace.
    assert examples[RULES_EXT_KEY][0]["Feedback"].startswith(
        "First-tool mismatch: teacher used Write and student called no scored tool, "
        "emitting only skipped steps such as the automatic vault retrieval or shell."
    )
