from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock

import pytest

from glean_gepa.objectives.tool_match import (
    FirstToolMatchObjective,
    ToolMatchEntryMetrics,
    aggregate_tool_match_metrics,
    build_tool_match_per_entry_query,
    empty_tool_match_analysis,
    fetch_eval_run_tool_match_analysis,
    parse_tool_match_entry_metrics,
)
from glean_gepa.objectives.utils.action_input_trace import extract_trace_tool_inputs
from glean_gepa.objectives.utils.agentspan import bounds_query
from glean_gepa.objectives.utils.agentspan_query import EXECUTE_ACTION_FILTER
from glean_gepa.objectives.utils.core import NoComparedEntriesError
from glean_gepa.objectives.utils.mismatch import select_mismatch_groups
from glean_gepa.objectives.utils.tool_names import (
    SKIPPED_TOOL_NAMES,
    first_tool_mismatch_pair,
    first_tool_name,
    scored_tool_sequence,
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

    assert (
        first_tool_mismatch_pair(
            ("find_skills_assistant", "Discover", "search"),
            ("Discover", "find_skills_assistant", "search"),
        )
        is None
    )
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


def _agent_tool_span(tool: str, action_input: str) -> dict:
    """An agent-side tool span: the whole payload sits in a flat ``action_input``."""
    return {
        "name": f"Execute Action: {tool}",
        "attributes": {"input": {"strValue": json.dumps({"action_input": action_input})}},
    }


def _glean_tool_span(tool: str, arguments: dict) -> dict:
    """A Glean tool span: the call is double-encoded under ``input``, beside metadata."""
    inner = json.dumps({"id": "call-1", "action": tool, "tool_name": tool, **arguments})
    return {
        "name": f"Execute Action: {tool}",
        "attributes": {"input": {"strValue": json.dumps({"input": inner})}},
    }


def _trace(*spans: dict) -> dict:
    return {"trace": {"spans": list(spans)}}


def test_extract_trace_tool_inputs_reads_both_call_envelopes():
    """Glean tools never emit ``action_input``; reading only that key drops them all.

    Their arguments live in a nested, double-encoded ``input`` object under a per-tool
    key, which is why first-tool reflection saw empty payloads for Glean Search.
    """
    trace = _trace(
        _agent_tool_span("Shell", '{"command":"ls"}'),
        _glean_tool_span("Glean Search", {"glean_search_tool_args": {"query": "case 007"}}),
        _glean_tool_span("Glean Search", {"glean_search_tool_args": {"query": "case 007"}}),
        _agent_tool_span("Write", '{"path":"a.txt"}'),
        {"name": "Some Other Span", "attributes": {"input": {"strValue": '{"action_input":"ignored"}'}}},
    )
    assert extract_trace_tool_inputs(trace, skip_tools=SKIPPED_TOOL_NAMES) == (
        ("Glean Search", '{"glean_search_tool_args": {"query": "case 007"}}'),
        ("Write", '{"path":"a.txt"}'),
    )
    # Only the first non-skipped call is kept for first-tool scoring.
    assert extract_trace_tool_inputs(trace, skip_tools=SKIPPED_TOOL_NAMES, limit=1) == (
        ("Glean Search", '{"glean_search_tool_args": {"query": "case 007"}}'),
    )
    # Without a skip list the shell call leads, and identifiers are not payload.
    assert extract_trace_tool_inputs(trace, limit=1) == (("Shell", '{"command":"ls"}'),)
    assert extract_trace_tool_inputs(_trace(_glean_tool_span("Glean Search", {}))) == ()
    assert extract_trace_tool_inputs({"trace": {"spans": [{"name": "Execute Action: X"}]}}) == ()
    assert extract_trace_tool_inputs(None) == ()


def test_tool_match_queries_and_fetch():
    bounds_sql = bounds_query(eval_id_predicate="IN UNNEST(@eval_ids)", span_filter=EXECUTE_ACTION_FILTER)
    sql = build_tool_match_per_entry_query()
    assert "PARSE_DATE" not in bounds_sql
    assert "PARSE_DATE" not in sql
    assert "_TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', @search_start_date)" in bounds_sql
    assert "_TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', @start_date)" in sql
    assert "@student_eval_id" in sql and "@teacher_eval_id" in sql
    assert "Execute Action:" in sql
    assert "FULL OUTER JOIN" in sql
    # A run that died must be flagged so its truncated tool sequence is not scored.
    assert "failed_runs" in sql and "run_failed" in sql
    assert "Agent Run:" in sql
    # Tool payloads are scrubbed from this table, so the query must NOT read them here
    # and must instead carry the scrub-safe locators used to fetch the detailed trace.
    assert "span_info.inputs" not in sql
    assert "agent_trace.trace_id" in sql
    assert "student_trace_id" in sql and "teacher_trace_id" in sql
    assert "student_deployment_id" in sql and "teacher_deployment_id" in sql
    assert "UNNEST(@skipped_tools)" in sql

    client = MagicMock()
    client.query.side_effect = [
        [{"min_start_ms": 1_786_363_200_000, "max_start_ms": 1_786_449_600_000}],
        [
            {
                "entry_id": "entry-1",
                "student_tools": ["search"],
                "teacher_tools": ["read"],
                "student_trace_id": "s-trace-1",
                "student_deployment_id": "scio-prod",
                "student_min_start_ms": 1_786_400_000_000,
                "student_max_start_ms": 1_786_400_050_000,
                "teacher_trace_id": "t-trace-1",
                "teacher_deployment_id": "scio-prod",
                "teacher_min_start_ms": 1_786_400_000_000,
                "teacher_max_start_ms": 1_786_400_050_000,
            },
            {"entry_id": "entry-2", "student_tools": [], "teacher_tools": ["search"]},
            {
                "entry_id": "entry-3",
                "student_tools": ["search"],
                "teacher_tools": ["read"],
                "student_trace_id": "s-trace-cust",
                "student_deployment_id": "glean-televox",
                "student_min_start_ms": 1_786_400_000_000,
                "student_max_start_ms": 1_786_400_050_000,
                "teacher_trace_id": "t-trace-cust",
                "teacher_deployment_id": "glean-televox",
                "teacher_min_start_ms": 1_786_400_000_000,
                "teacher_max_start_ms": 1_786_400_050_000,
            },
        ],
    ]

    def _get_trace(*, deployment_id, trace_id, start_time_millis, end_time_millis):
        if trace_id == "t-trace-1":
            # The skipped shell call must not be mistaken for the scored first tool.
            return _trace(
                _agent_tool_span("Shell", '{"command":"ls"}'),
                _glean_tool_span("Glean Search", {"glean_search_tool_args": {"query": "case 007"}}),
                _glean_tool_span("Glean Document Reader", {"code_search": {"urls": ["a"]}}),
            )
        if trace_id == "s-trace-1":
            return _trace(_agent_tool_span("Write", '{"path":"a.txt"}'))
        return _trace()

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
    # Payloads come from the detailed trace (the scrubbed table serves none). Each
    # role keeps its scored calls, shell excluded, tagged with the tool that issued them.
    assert analysis.per_entry["entry-1"].teacher_tool_inputs == (
        ("Glean Search", '{"glean_search_tool_args": {"query": "case 007"}}'),
        ("Glean Document Reader", '{"code_search": {"urls": ["a"]}}'),
    )
    assert analysis.per_entry["entry-1"].student_tool_inputs == (("Write", '{"path":"a.txt"}'),)
    # entry-2 exposes no trace locator, so it is never fetched and stays empty.
    assert analysis.per_entry["entry-2"].teacher_tool_inputs == ()
    # Customer deployments 403 on analyze-trace; skip them and still score the mismatch.
    assert analysis.per_entry["entry-3"].tools_match is False
    assert analysis.per_entry["entry-3"].teacher_tool_inputs == ()
    assert {call.kwargs["trace_id"] for call in evalcli.get_analysis_trace.call_args_list} == {
        "t-trace-1",
        "s-trace-1",
    }
    # The trace window pads the entry's span bounds by the configured lead/trail.
    teacher_call = next(
        call for call in evalcli.get_analysis_trace.call_args_list if call.kwargs["trace_id"] == "t-trace-1"
    )
    assert teacher_call.kwargs["start_time_millis"] == 1_786_400_000_000 - 3_600_000
    assert teacher_call.kwargs["end_time_millis"] == 1_786_400_050_000 + 60_000
    assert analysis.high_signal_entry_ids == ("entry-1", "entry-2", "entry-3")
    assert client.query.call_count == 2

    objective = FirstToolMatchObjective()
    output = next(
        row.output
        for row in objective.scored_rows(
            analysis, focused=True, capture_traces=True, query="q", deployment_id="scio-prod"
        )
        if row.entry_id == "entry-1"
    )
    example = objective.build_reflective_example(
        "MODULE",
        {
            "data": {"eval_set_name": "set"},
            "score": 0.0,
            "objective_scores": {"tool_alignment": 0.0, "correctness": 0.5},
            "output": output,
        },
        {},
    )
    # Each call is attributed to a role, and later scored calls are kept with the first.
    assert example["Action Inputs"] == [
        'teacher Glean Search: {"glean_search_tool_args": {"query": "case 007"}}',
        'teacher Glean Document Reader: {"code_search": {"urls": ["a"]}}',
        'student Write: {"path":"a.txt"}',
    ]


def test_failed_runs_are_excluded_rather_than_scored_as_mismatches():
    """A dead run has no first tool, so scoring it blames a decision never made."""
    client = MagicMock()
    client.query.side_effect = [
        [{"min_start_ms": 1_786_363_200_000, "max_start_ms": 1_786_449_600_000}],
        [
            # Both roles completed and disagreed: a real, scorable mismatch.
            {"entry_id": "live-mismatch", "run_failed": False, "student_tools": ["search"], "teacher_tools": ["read"]},
            {"entry_id": "live-match", "run_failed": False, "student_tools": ["search"], "teacher_tools": ["search"]},
            # The teacher died: an empty sequence that is not a tool choice.
            {"entry_id": "teacher-died", "run_failed": True, "student_tools": ["search"], "teacher_tools": []},
            # The teacher died after the warm-start tools, so the sequence looks populated.
            {
                "entry_id": "teacher-truncated",
                "run_failed": True,
                "student_tools": ["read"],
                "teacher_tools": ["Discover"],
            },
            # A missing flag must not be read as a failure.
            {"entry_id": "no-flag", "student_tools": ["search"], "teacher_tools": ["search"]},
        ],
    ]

    analysis = fetch_eval_run_tool_match_analysis(
        client,
        teacher_eval_id="teacher",
        student_eval_id="student",
        lookback_days=7,
        end_date=date(2026, 8, 11),
    )
    assert set(analysis.per_entry) == {"live-mismatch", "live-match", "no-flag"}
    assert analysis.aggregate.compared_entries == 3
    assert analysis.aggregate.matching_entries == 2
    assert analysis.aggregate.excluded_failed_runs == 2
    # Without the exclusion this would be 2/5 = 40%, penalising the prompt for a
    # provider rejection that the student had no part in.
    assert analysis.aggregate.tool_match_rate == pytest.approx(2 / 3)
    # Dropped entries must not reach reflection as high-signal failures.
    assert analysis.high_signal_entry_ids == ("live-mismatch",)

    # Every entry dropped is an undefined comparison, not a 0% tool match.
    client.query.side_effect = [
        [{"min_start_ms": 1_786_363_200_000, "max_start_ms": 1_786_449_600_000}],
        [
            {"entry_id": "a", "run_failed": True, "student_tools": ["search"], "teacher_tools": []},
            {"entry_id": "b", "run_failed": True, "student_tools": ["read"], "teacher_tools": []},
        ],
    ]
    all_failed = fetch_eval_run_tool_match_analysis(
        client,
        teacher_eval_id="teacher",
        student_eval_id="student",
        lookback_days=7,
        end_date=date(2026, 8, 11),
    )
    assert all_failed.aggregate.compared_entries == 0
    assert all_failed.aggregate.excluded_failed_runs == 2
    with pytest.raises(NoComparedEntriesError, match="all 2 candidate entries were dropped"):
        FirstToolMatchObjective().require_compared_entries(all_failed)


def _reflective_example(objective, objective_scores: dict, **output_extras) -> dict:
    output = {
        "entry_id": "entry-1",
        "deployment_id": "scio-prod",
        "query": "q",
        "student_tool_events": ["Glean Search"],
        "teacher_tool_events": ["Glean Document Reader"],
        **output_extras,
    }
    return objective.build_reflective_example(
        "MODULE",
        {"data": {"eval_set_name": "set"}, "score": 0.0, "objective_scores": objective_scores, "output": output},
        {},
    )


def test_reflective_example_includes_both_roles_tool_inputs():
    """Both sequences are labeled, so a student payload cannot be read as the teacher's."""
    objective = FirstToolMatchObjective()
    scores = {"tool_alignment": 0.0}

    student_only = _reflective_example(
        objective, scores, student_tool_inputs=[["Glean Search", '{"query": "pto policy"}']]
    )
    assert student_only["Action Inputs"] == ['student Glean Search: {"query": "pto policy"}']

    both = _reflective_example(
        objective,
        scores,
        student_tool_inputs=[["Glean Search", '{"query": "pto policy"}'], ["Write", '{"path":"a.txt"}']],
        teacher_tool_inputs=[["Glean Document Reader", '{"urls": ["x"]}']],
    )
    assert both["Action Inputs"] == [
        'teacher Glean Document Reader: {"urls": ["x"]}',
        'student Glean Search: {"query": "pto policy"}',
        'student Write: {"path":"a.txt"}',
    ]
    assert _reflective_example(objective, scores)["Action Inputs"] == []
    assert _reflective_example(objective, scores, teacher_tool_inputs=[["Glean Search", ""]])["Action Inputs"] == []

    payload = '{"file_path":"SKILL.md","old_string":"' + "x" * 900 + '"}'
    long = _reflective_example(objective, scores, teacher_tool_inputs=[["Edit", payload]])
    assert long["Action Inputs"][0].endswith("... (truncated)")
    assert len(long["Action Inputs"][0]) < len(payload)

    extra = [[f"Tool{i}", f'{{"q": "{i}"}}'] for i in range(4)]
    capped = _reflective_example(objective, scores, teacher_tool_inputs=extra, student_tool_inputs=extra)
    assert capped["Generated Outputs"]["teacher_tools"] == ["Glean Document Reader"]
    assert capped["Generated Outputs"]["student_tools"] == ["Glean Search"]
    assert capped["Action Inputs"] == [
        'teacher Tool0: {"q": "0"}',
        'teacher Tool1: {"q": "1"}',
        'teacher Tool2: {"q": "2"}',
        'student Tool0: {"q": "0"}',
        'student Tool1: {"q": "1"}',
        'student Tool2: {"q": "2"}',
    ]


def test_reflection_selects_mismatches_in_order_without_grouping():
    objective = FirstToolMatchObjective()
    keys = [("Write", "")] * 2 + [("Glean Search", "Discover")] * 10 + [None]
    selected, groups = objective._select_mismatch_groups(keys, max_entries=4)
    assert selected == [0, 1, 2, 3]
    assert groups == []
    all_mismatches, _ = objective._select_mismatch_groups(keys, max_entries=None)
    assert all_mismatches == list(range(12))


def test_unscored_correctness_is_omitted_rather_than_reported_as_zero():
    """Correctness vs the teacher is absent until the pairwise judge has scored."""
    objective = FirstToolMatchObjective()
    objective.high_signal = "tool_alignment"
    objective.signal_names = ("tool_alignment", "correctness")

    absent = _reflective_example(objective, {"tool_alignment": 0.0})
    assert "correctness" not in absent["Metrics"]
    assert "Correctness" not in absent["Feedback"]
    assert objective.format_reflective_metrics(absent["Metrics"]) == "score=0.00, tool_alignment=0.00"

    # A real low score still reports, and a real passing score stays silent.
    scored = _reflective_example(objective, {"tool_alignment": 0.0, "correctness": 0.5})
    assert scored["Metrics"]["correctness"] == 0.5
    assert "Correctness issue: score=0.50." in scored["Feedback"]
    assert objective.format_reflective_metrics(scored["Metrics"]).endswith("correctness=0.50")

    passing = _reflective_example(objective, {"tool_alignment": 0.0, "correctness": 0.9})
    assert "Correctness" not in passing["Feedback"]
    assert passing["Metrics"]["correctness"] == 0.9

    unwired = FirstToolMatchObjective()
    unwired.high_signal = "tool_alignment"
    hidden = _reflective_example(unwired, {"tool_alignment": 0.0, "correctness": 0.5})
    assert "correctness" not in hidden["Metrics"]
    assert "Correctness" not in hidden["Feedback"]


def test_reflective_example_includes_the_agentic_judge_verdict_and_tool_inputs():
    objective = FirstToolMatchObjective()
    example = _reflective_example(
        objective,
        {"tool_alignment": 0.0},
        teacher_tool_inputs=[["Glean Search", '{"query": "pto"}']],
        student_tool_inputs=[["Discover", '{"query": "policy"}']],
        agentic_preference_rate_feedback="task_completion: the student stopped after a search.",
    )
    assert example["Action Inputs"] == [
        'teacher Glean Search: {"query": "pto"}',
        'student Discover: {"query": "policy"}',
    ]
    assert "Agentic judge verdict:\ntask_completion: the student stopped after a search." in example["Feedback"]


def test_aggregate_and_empty_analysis():
    per_entry = {
        "a": ToolMatchEntryMetrics("a", ("search",), ("search",), True),
        "b": ToolMatchEntryMetrics("b", ("read",), ("search",), False),
    }
    aggregate = aggregate_tool_match_metrics(per_entry)
    assert aggregate.compared_entries == 2
    assert aggregate.matching_entries == 1
    assert aggregate.tool_match_rate == 0.5
    empty = aggregate_tool_match_metrics({})
    assert empty.compared_entries == 0
    assert empty.tool_match_rate == 0.0
    analysis = empty_tool_match_analysis("teacher-1", "student-1", end_date=date(2026, 8, 11))
    with pytest.raises(NoComparedEntriesError, match="No eval entries were compared"):
        FirstToolMatchObjective().require_compared_entries(analysis)


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
