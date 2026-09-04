from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from glean_gepa.tool_match_util import (
    SKIPPED_TOOL_NAMES,
    EvalRunToolMatchAnalysis,
    NoComparedEvalEntriesError,
    ToolMatchEntryMetrics,
    aggregate_tool_match_metrics,
    build_tool_match_per_entry_query,
    collapse_consecutive_tools,
    empty_tool_match_analysis,
    exact_prefix_match,
    fetch_eval_run_tool_match_analysis,
    first_disagreeing_mismatch,
    first_tool_mismatch_pair,
    first_tool_name,
    intersect_compared_entry_ids,
    parse_tool_match_entry_metrics,
    prefix_slot_weights,
    require_compared_eval_entries,
    restrict_tool_match_analysis,
    scored_tool_sequence,
    select_first_tool_mismatch_groups,
    tool_prefix_alignment,
)


def test_first_tool_scoring_strips_shell_and_ignores_later_tools():
    assert scored_tool_sequence(["Shell", "search", "Shell Tool", "read"]) == ("search", "read")
    assert scored_tool_sequence(["Shell", "Shell"]) == ()
    assert first_tool_name(()) == ""
    assert first_tool_name(("search", "read")) == "search"
    assert first_tool_name(("Shell", "search")) == "search"
    assert first_tool_mismatch_pair(("search", "read"), ("search", "write")) is None
    assert first_tool_mismatch_pair(("search",), ()) == ("search", "")
    assert first_tool_mismatch_pair(("Shell", "read"), ("search",)) == ("read", "search")
    assert first_tool_mismatch_pair(("Shell",), ("Shell Tool",)) is None

    match = parse_tool_match_entry_metrics(
        {"entry_id": "entry-1", "student_tools": ["Shell", "search", "read"], "teacher_tools": ["search", "write"]}
    )
    assert match.entry_id == "entry-1"
    assert match.student_tools == ("search", "read")
    assert match.teacher_tools == ("search", "write")
    assert match.tools_match

    mismatch = parse_tool_match_entry_metrics(
        {"entry_id": "entry-2", "student_tools": ["search"], "teacher_tools": ["read"]}
    )
    assert not mismatch.tools_match


def test_consecutive_dedupe_and_prefix_alignment():
    assert collapse_consecutive_tools(("Write", "Write", "Search", "Write")) == ("Write", "Search", "Write")
    assert first_tool_mismatch_pair(("Write", "Write"), ("Write",)) is None
    assert first_disagreeing_mismatch(("Write", "Search"), ("Write",), prefix_k=2) == (1, "Search", "")
    assert prefix_slot_weights(1) == (1.0,)
    assert prefix_slot_weights(2) == (0.75, 0.25)
    assert prefix_slot_weights(3) == (0.5, 0.25, 0.25)

    assert tool_prefix_alignment(("Write", "Write"), ("Write",), prefix_k=2) == 1.0
    assert tool_prefix_alignment(("Write", "Search"), ("Write", "Search"), prefix_k=2) == 1.0
    assert tool_prefix_alignment(("Write", "Search"), ("Write", "Read"), prefix_k=2) == 0.75
    assert tool_prefix_alignment(("Search",), ("Write",), prefix_k=2) == 0.0
    assert tool_prefix_alignment(("Write", "Search"), ("Write", "Search"), prefix_k=3) == 1.0
    assert tool_prefix_alignment(("Write", "Search"), ("Write", "Read"), prefix_k=3) == pytest.approx(0.5 / 0.75)
    assert exact_prefix_match(("Write",), ("Write", "Write"), prefix_k=2)
    assert not exact_prefix_match(("Write", "Search"), ("Write",), prefix_k=2)


def test_tool_match_queries_and_fetch():
    sql = build_tool_match_per_entry_query()
    assert "@student_eval_id" in sql and "@teacher_eval_id" in sql
    assert "Execute Action:" in sql
    assert "student_trace_id" not in sql
    assert "LEFT JOIN student" in sql
    assert "FULL OUTER JOIN" not in sql
    assert "PARSE_DATE" not in sql
    assert "_TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', @start_date)" in sql
    for skipped in SKIPPED_TOOL_NAMES:
        assert skipped in sql

    client = MagicMock()
    client.query.side_effect = [
        [{"min_start_ms": 1_786_363_200_000, "max_start_ms": 1_786_449_600_000}],
        [
            {"entry_id": "entry-1", "student_tools": ["search"], "teacher_tools": ["read"]},
            {"entry_id": "entry-2", "student_tools": [], "teacher_tools": ["search"]},
        ],
    ]
    analysis = fetch_eval_run_tool_match_analysis(
        client,
        teacher_eval_id="teacher",
        student_eval_id="student",
        lookback_days=7,
        end_date=date(2026, 8, 11),
    )
    search_params = {param.name: param.value for param in client.query.call_args_list[0].kwargs["params"]}
    entry_params = {param.name: param.value for param in client.query.call_args_list[1].kwargs["params"]}
    assert search_params["eval_ids"] == ["teacher", "student"]
    assert search_params["search_start_date"] == "2026-08-04"
    assert search_params["search_end_date"] == "2026-08-11"
    assert entry_params["student_eval_id"] == "student"
    assert entry_params["teacher_eval_id"] == "teacher"
    assert analysis.per_entry["entry-1"].tools_match is False
    assert analysis.per_entry["entry-2"].student_tools == ()
    assert analysis.high_signal_entry_ids == ("entry-1", "entry-2")
    assert client.query.call_count == 2


def test_tool_match_fetch_skips_bounds_query_for_default_lookback():
    client = MagicMock()
    client.query.return_value = [
        {"entry_id": "entry-1", "student_tools": ["search"], "teacher_tools": ["search"]},
    ]
    analysis = fetch_eval_run_tool_match_analysis(
        client,
        teacher_eval_id="teacher",
        student_eval_id="student",
        lookback_days=1,
        end_date=date(2026, 8, 11),
    )
    assert client.query.call_count == 1
    params = {param.name: param.value for param in client.query.call_args.kwargs["params"]}
    assert params["start_date"] == "2026-08-10"
    assert params["end_date"] == "2026-08-11"
    assert analysis.aggregate.compared_entries == 1


def test_aggregate_and_empty_analysis():
    per_entry = {
        "a": ToolMatchEntryMetrics("a", ("search",), ("search",), True),
        "b": ToolMatchEntryMetrics("b", ("read",), ("search",), False),
    }
    aggregate = aggregate_tool_match_metrics("teacher", "student", per_entry)
    assert aggregate.compared_entries == 2
    assert aggregate.matching_entries == 1
    assert aggregate.tool_match_rate == 0.5
    empty = aggregate_tool_match_metrics("teacher", "student", {})
    assert empty.compared_entries == 0
    assert empty.tool_match_rate == 0.0
    analysis = empty_tool_match_analysis("teacher-1", "student-1", end_date=date(2026, 8, 11))
    with pytest.raises(NoComparedEvalEntriesError, match="No eval entries were compared"):
        require_compared_eval_entries(analysis)


def test_restrict_and_intersect_compared_entries():
    keep = ToolMatchEntryMetrics("keep", ("search",), ("search",), True)
    drop = ToolMatchEntryMetrics("drop", ("read",), ("search",), False)
    extra = ToolMatchEntryMetrics("extra", ("search",), ("search",), True)
    first = empty_tool_match_analysis("teacher", "student-a", end_date=date(2026, 8, 11))
    first = restrict_tool_match_analysis(
        EvalRunToolMatchAnalysis(
            teacher_eval_id="teacher",
            student_eval_id="student-a",
            start_date=first.start_date,
            end_date=first.end_date,
            aggregate=aggregate_tool_match_metrics("teacher", "student-a", {"keep": keep, "drop": drop}),
            per_entry={"keep": keep, "drop": drop},
            high_signal_entry_ids=("drop",),
        ),
        {"keep"},
    )
    second = EvalRunToolMatchAnalysis(
        teacher_eval_id="teacher",
        student_eval_id="student-b",
        start_date=first.start_date,
        end_date=first.end_date,
        aggregate=aggregate_tool_match_metrics("teacher", "student-b", {"keep": keep, "extra": extra}),
        per_entry={"keep": keep, "extra": extra},
        high_signal_entry_ids=(),
    )
    assert set(first.per_entry) == {"keep"}
    assert first.aggregate.compared_entries == 1
    assert first.aggregate.matching_entries == 1
    assert first.high_signal_entry_ids == ()
    assert intersect_compared_entry_ids([first, second]) == {"keep"}
    assert intersect_compared_entry_ids([]) == set()


def test_select_first_tool_mismatch_groups():
    capped, groups = select_first_tool_mismatch_groups(
        [("x", "y")] * 12 + [("y", "x")] * 8 + [("a", "b")] * 6 + [("c", "d")] * 5 + [None]
    )
    assert len(capped) == 20
    assert groups == [(0, "x", "y", 12), (0, "y", "x", 8)]

    skipped, skipped_groups = select_first_tool_mismatch_groups(
        [("x", "y")] * 10 + [("y", "x")] * 8 + [("a", "b")] * 7 + [("e", "f")] * 2
    )
    assert len(skipped) == 20
    assert skipped_groups == [(0, "x", "y", 10), (0, "y", "x", 8), (0, "e", "f", 2)]

    oversized, oversized_groups = select_first_tool_mismatch_groups([("x", "y")] * 35 + [("a", "b")] * 3)
    assert oversized == list(range(35))
    assert oversized_groups == [(0, "x", "y", 35)]

    by_slot, slot_groups = select_first_tool_mismatch_groups(
        [(1, "Search", "Write")] * 12 + [(0, "Search", "Write")] * 8 + [None]
    )
    assert len(by_slot) == 20
    assert slot_groups == [(1, "Search", "Write", 12), (0, "Search", "Write", 8)]
