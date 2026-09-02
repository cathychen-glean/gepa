from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from glean_gepa.tool_match_util import (
    SKIPPED_TOOL_NAMES,
    NoComparedEvalEntriesError,
    ToolMatchEntryMetrics,
    aggregate_tool_match_metrics,
    avg_levenshtein_alignment_score,
    build_tool_match_per_entry_query,
    build_tool_match_query_params,
    build_tool_match_search_params,
    compute_tool_match_score,
    empty_tool_match_analysis,
    fetch_eval_run_tool_match_analysis,
    mean_sequence_levenshtein,
    parse_tool_match_entry_metrics,
    require_compared_eval_entries,
    sequence_levenshtein_distance,
)


def test_compute_tool_match_score_identical_sequences():
    assert compute_tool_match_score(("search", "read"), ("search", "read")) == 1.0


def test_compute_tool_match_score_empty_both_sides():
    assert compute_tool_match_score((), ()) == 1.0


def test_compute_tool_match_score_one_side_empty():
    assert compute_tool_match_score(("search",), ()) == 0.0


def test_compute_tool_match_score_partial_prefix():
    assert compute_tool_match_score(("search", "read"), ("search", "write")) == 0.5


@pytest.mark.parametrize(
    ("student", "teacher", "distance"),
    [
        (("search", "read"), ("search", "read"), 0),
        (("search", "read"), ("search", "write"), 1),
        ((), ("search", "read"), 2),
        (("search",), ("search", "read", "write"), 2),
        (("a", "b", "c"), ("a", "x", "c", "d"), 2),
    ],
)
def test_sequence_levenshtein_distance(student, teacher, distance):
    assert sequence_levenshtein_distance(student, teacher) == distance


def test_mean_sequence_levenshtein_averages_requested_entries():
    per_entry = {
        "a": (("search", "read", "write"), ()),
        "b": (("search",), ("search",)),
        "c": (("read",), ("write",)),
    }

    assert mean_sequence_levenshtein(per_entry, ["a", "b"]) == 1.5


def test_mean_sequence_levenshtein_uses_fallback_for_missing_entries():
    per_entry = {"a": (("search", "read"), ("search",))}

    assert mean_sequence_levenshtein(per_entry, ["a", "b"], fallback_distances={"b": 4}) == 2.5
    assert mean_sequence_levenshtein(per_entry, ["a", "b"]) == float("inf")


def test_avg_levenshtein_alignment_score_decreases_with_distance():
    assert avg_levenshtein_alignment_score(0) == 1.0
    assert avg_levenshtein_alignment_score(1) == 0.5
    assert avg_levenshtein_alignment_score(float("inf")) == 0.0
    assert avg_levenshtein_alignment_score(3) < avg_levenshtein_alignment_score(1)


def test_build_tool_match_per_entry_query_pairs_teacher_and_student():
    sql = build_tool_match_per_entry_query()

    assert "@eval_ids" in sql
    assert "@student_eval_id" in sql
    assert "@teacher_eval_id" in sql
    assert "STARTS_WITH" in sql
    assert "Execute Action:" in sql
    assert "student_tools" in sql
    assert "teacher_tools" in sql
    assert "FULL OUTER JOIN" in sql
    assert "IFNULL(student.tools, ARRAY<STRING>[])" in sql
    assert "IFNULL(teacher.tools, ARRAY<STRING>[])" in sql
    for skipped in SKIPPED_TOOL_NAMES:
        assert skipped in sql


def test_build_tool_match_search_params_uses_lookback_window():
    params = build_tool_match_search_params(
        teacher_eval_id="teacher",
        student_eval_id="student",
        lookback_days=3,
        end_date=date(2026, 8, 11),
    )
    param_map = {param.name: param.value for param in params}

    assert param_map["eval_ids"] == ["teacher", "student"]
    assert param_map["search_start_date"] == "2026-08-08"
    assert param_map["search_end_date"] == "2026-08-11"


def test_build_tool_match_query_params_includes_both_eval_ids():
    params = build_tool_match_query_params(
        teacher_eval_id="teacher",
        student_eval_id="student",
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
    )
    param_map = {param.name: param.value for param in params}

    assert param_map["student_eval_id"] == "student"
    assert param_map["teacher_eval_id"] == "teacher"
    assert param_map["eval_ids"] == ["teacher", "student"]


def test_parse_tool_match_entry_metrics_from_bigquery_row():
    metrics = parse_tool_match_entry_metrics(
        {
            "entry_id": "entry-1",
            "student_tools": ["search", "read"],
            "teacher_tools": ["search", "read"],
            "student_trace_id": "s-trace",
            "teacher_trace_id": "t-trace",
        }
    )

    assert metrics.entry_id == "entry-1"
    assert metrics.tools_match
    assert metrics.tool_match_score == 1.0
    assert metrics.student_trace_id == "s-trace"


def test_aggregate_tool_match_metrics():
    per_entry = {
        "a": ToolMatchEntryMetrics(
            entry_id="a",
            student_tools=("search",),
            teacher_tools=("search",),
            tools_match=True,
            tool_match_score=1.0,
        ),
        "b": ToolMatchEntryMetrics(
            entry_id="b",
            student_tools=("read",),
            teacher_tools=("search",),
            tools_match=False,
            tool_match_score=0.0,
        ),
    }

    aggregate = aggregate_tool_match_metrics("teacher", "student", per_entry)

    assert aggregate.compared_entries == 2
    assert aggregate.matching_entries == 1
    assert aggregate.mismatching_entries == 1
    assert aggregate.tool_match_rate == 0.5


def test_aggregate_tool_match_metrics_zero_entries_is_not_perfect():
    aggregate = aggregate_tool_match_metrics("teacher", "student", {})

    assert aggregate.compared_entries == 0
    assert aggregate.tool_match_rate == 0.0


def test_require_compared_eval_entries_raises_on_empty_analysis():
    analysis = empty_tool_match_analysis("teacher-1", "student-1", end_date=date(2026, 8, 11))

    with pytest.raises(NoComparedEvalEntriesError, match="No eval entries were compared"):
        require_compared_eval_entries(analysis)


def test_fetch_eval_run_tool_match_analysis_joins_per_entry_rows():
    client = MagicMock()
    client.query.side_effect = [
        [{"min_start_ms": 1_786_363_200_000, "max_start_ms": 1_786_449_600_000}],
        [
            {
                "entry_id": "entry-1",
                "student_tools": ["search"],
                "teacher_tools": ["read"],
                "student_trace_id": "s",
                "teacher_trace_id": "t",
            },
            {
                "entry_id": "entry-2",
                "student_tools": [],
                "teacher_tools": ["search"],
                "student_trace_id": None,
                "teacher_trace_id": "t2",
            },
        ],
    ]

    analysis = fetch_eval_run_tool_match_analysis(
        client,
        teacher_eval_id="teacher",
        student_eval_id="student",
        lookback_days=7,
        end_date=date(2026, 8, 11),
    )

    assert analysis.per_entry["entry-1"].tools_match is False
    assert analysis.per_entry["entry-2"].student_tools == ()
    assert analysis.per_entry["entry-2"].tool_match_score == 0.0
    assert analysis.high_signal_entry_ids == ("entry-1", "entry-2")
    assert analysis.aggregate.mismatching_entries == 2
    assert client.query.call_count == 2
