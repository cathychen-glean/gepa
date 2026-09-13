from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock

import pytest

from glean_gepa.objectives.utils.tool_match_util import (
    SKIPPED_TOOL_NAMES,
    NoComparedEvalEntriesError,
    ToolMatchEntryMetrics,
    aggregate_tool_match_metrics,
    build_tool_match_per_entry_query,
    build_tool_match_time_bounds_query,
    empty_tool_match_analysis,
    fetch_eval_run_tool_match_analysis,
    first_tool_mismatch_pair,
    first_tool_name,
    parse_tool_match_entry_metrics,
    require_compared_eval_entries,
    scored_tool_sequence,
)
from glean_gepa.objectives.tool_match import FirstToolMatchObjective
from glean_gepa.objectives.utils.mismatch import select_mismatch_groups


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


def _trace_with_action_inputs(*action_inputs: str) -> dict:
    """A detailed-trace payload whose Execute Action spans carry ``action_input``."""
    return {
        "trace": {
            "spans": [
                {
                    "name": "Execute Action: Search",
                    "attributes": {"input": {"strValue": json.dumps({"action_input": action_input})}},
                }
                for action_input in action_inputs
            ]
        }
    }


def test_tool_match_queries_and_fetch():
    bounds_sql = build_tool_match_time_bounds_query()
    sql = build_tool_match_per_entry_query()
    assert "PARSE_DATE" not in bounds_sql
    assert "PARSE_DATE" not in sql
    assert "_TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', @search_start_date)" in bounds_sql
    assert "_TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', @start_date)" in sql
    assert "@student_eval_id" in sql and "@teacher_eval_id" in sql
    assert "Execute Action:" in sql
    assert "FULL OUTER JOIN" in sql
    # Tool payloads are scrubbed from this table, so the query must NOT read them here
    # and must instead carry the scrub-safe locators used to fetch the detailed trace.
    assert "span_info.inputs" not in sql
    assert "agent_trace.trace_id" in sql
    assert "student_trace_id" in sql and "teacher_trace_id" in sql
    assert "student_deployment_id" in sql and "teacher_deployment_id" in sql
    for skipped in SKIPPED_TOOL_NAMES:
        assert skipped in sql

    client = MagicMock()
    client.query.side_effect = [
        [{"min_start_ms": 1_786_363_200_000, "max_start_ms": 1_786_449_600_000}],
        [
            {
                "entry_id": "entry-1",
                "student_tools": ["search"],
                "teacher_tools": ["read"],
                "student_trace_id": "s-trace-1",
                "student_deployment_id": "dep",
                "student_min_start_ms": 1_786_400_000_000,
                "student_max_start_ms": 1_786_400_050_000,
                "teacher_trace_id": "t-trace-1",
                "teacher_deployment_id": "dep",
                "teacher_min_start_ms": 1_786_400_000_000,
                "teacher_max_start_ms": 1_786_400_050_000,
            },
            {"entry_id": "entry-2", "student_tools": [], "teacher_tools": ["search"]},
        ],
    ]

    def _get_trace(*, deployment_id, trace_id, start_time_millis, end_time_millis):
        if trace_id == "t-trace-1":
            # Duplicate payloads collapse; only Execute Action inputs survive.
            return _trace_with_action_inputs('{"query":"case 007"}', '{"query":"case 007"}')
        if trace_id == "s-trace-1":
            return _trace_with_action_inputs('{"query":"stu"}')
        return _trace_with_action_inputs()

    evalcli = MagicMock()
    evalcli.get_analysis_trace.side_effect = _get_trace

    analysis = fetch_eval_run_tool_match_analysis(
        client,
        teacher_eval_id="teacher",
        student_eval_id="student",
        lookback_days=7,
        end_date=date(2026, 8, 11),
        evalcli=evalcli,
    )
    search_params = {param.name: param.value for param in client.query.call_args_list[0].kwargs["params"]}
    entry_params = {param.name: param.value for param in client.query.call_args_list[1].kwargs["params"]}
    assert search_params["eval_ids"] == ["teacher", "student"]
    assert search_params["search_start_date"] == "2026-08-04"
    # One day past end_date: _TABLE_SUFFIX is UTC, so tomorrow's shard is scanned too.
    assert search_params["search_end_date"] == "2026-08-12"
    assert entry_params["student_eval_id"] == "student"
    assert entry_params["teacher_eval_id"] == "teacher"
    assert analysis.per_entry["entry-1"].tools_match is False
    assert analysis.per_entry["entry-2"].student_tools == ()
    # Payloads come from the detailed trace (the scrubbed table serves none), and are
    # de-duplicated in span order.
    assert analysis.per_entry["entry-1"].teacher_action_inputs == ('{"query":"case 007"}',)
    assert analysis.per_entry["entry-1"].student_action_inputs == ('{"query":"stu"}',)
    # entry-2 exposes no trace locator, so it is never fetched and stays empty.
    assert analysis.per_entry["entry-2"].teacher_action_inputs == ()
    # The trace window pads the entry's span bounds by the configured lead/trail.
    teacher_call = next(
        call for call in evalcli.get_analysis_trace.call_args_list if call.kwargs["trace_id"] == "t-trace-1"
    )
    assert teacher_call.kwargs["start_time_millis"] == 1_786_400_000_000 - 3_600_000
    assert teacher_call.kwargs["end_time_millis"] == 1_786_400_050_000 + 60_000
    assert analysis.high_signal_entry_ids == ("entry-1", "entry-2")
    assert client.query.call_count == 2


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


def test_tool_reflective_example_surfaces_teacher_action_inputs():
    objective = FirstToolMatchObjective()
    trajectory = {
        "data": {"eval_set_name": "set"},
        "score": 0.0,
        "objective_scores": {"tool_alignment": 0.0, "completeness": 0.5},
        "output": {
            "entry_id": "e1",
            "deployment_id": "dep",
            "query": "set:v1",
            "student_tool_events": ["search"],
            "teacher_tool_events": ["read"],
            "teacher_action_inputs": [f'{{"query":"q{i}"}}' for i in range(6)],
        },
    }
    example = objective.build_reflective_example("MODULE", trajectory, {})
    assert example["Action Inputs"] == [f'{{"query":"q{i}"}}' for i in range(5)]


def test_select_mismatch_groups():
    capped, groups = select_mismatch_groups(
        [("x", "y")] * 12 + [("y", "x")] * 8 + [("a", "b")] * 6 + [("c", "d")] * 5 + [None]
    )
    assert len(capped) == 20
    assert groups == [("x", "y", 12), ("y", "x", 8)]

    skipped, skipped_groups = select_mismatch_groups(
        [("x", "y")] * 10 + [("y", "x")] * 8 + [("a", "b")] * 7 + [("e", "f")] * 2
    )
    assert len(skipped) == 20
    assert skipped_groups == [("x", "y", 10), ("y", "x", 8), ("e", "f", 2)]

    oversized, oversized_groups = select_mismatch_groups([("x", "y")] * 35 + [("a", "b")] * 3)
    assert oversized == list(range(35))
    assert oversized_groups == [("x", "y", 35)]
