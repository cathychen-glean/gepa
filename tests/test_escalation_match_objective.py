from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from glean_gepa.objectives.escalation_match import (
    ABSENT,
    ANSWERED,
    ESCALATED,
    INCOMPLETE,
    OTHER,
    WALDO_ROUTING_FRAME,
    EscalationMatchObjective,
    aggregate_escalation_match_metrics,
    build_escalation_match_per_entry_query,
    classify_waldo_decision,
    escalation_mismatch_pair,
    fetch_eval_run_escalation_match_analysis,
    interleave_mismatch_groups,
    parse_escalation_match_row,
    summarize_coverage,
)
from glean_gepa.objectives.utils.core import NoComparedEntriesError
from glean_gepa.prompt_targets import stock_text

RAN = "Agent execution completed successfully"


def _side(role: str, message: str | None, termination: str | None, *, refusal: bool = False) -> dict[str, Any]:
    if message is None and termination is None:
        return {}
    summary = f"termination={termination}, loops=1, tool_calls=0" if termination else None
    return {
        f"{role}_trace_id": f"{role[0]}-trace",
        f"{role}_agent_message": message,
        f"{role}_termination": termination,
        f"{role}_waldo_summary": summary,
        f"{role}_first_sentence_refusal": refusal,
    }


def _row(
    entry_id: str,
    teacher: tuple[str | None, str | None],
    student: tuple[str | None, str | None],
    *,
    teacher_refusal: bool = False,
) -> dict[str, Any]:
    return {
        "entry_id": entry_id,
        **_side("teacher", *teacher, refusal=teacher_refusal),
        **_side("student", *student),
    }


def _entries(*rows: dict[str, Any]):
    parsed = [parse_escalation_match_row(row) for row in rows]
    return [m for m in parsed if m is not None]


def test_classification_follows_the_escalation_rules():
    assert classify_waldo_decision(RAN, "SAW_INSUFFICIENT_TOOLS").outcome == ESCALATED
    assert classify_waldo_decision(RAN, "SAW_READY").outcome == ANSWERED
    # Fallback terminations are not escalation decisions.
    no_sample = classify_waldo_decision(RAN, "NO_SAMPLE")
    assert no_sample.outcome == OTHER and not no_sample.escalated
    # A WaldoAgent skip wins over a termination left on the same trace by an earlier turn.
    skipped = classify_waldo_decision("skipped:user_turn_limit", "SAW_READY")
    assert skipped.outcome == "skip:user_turn_limit" and skipped.skip_reason == "user_turn_limit"
    assert not skipped.ran
    assert classify_waldo_decision("Span failed with CancelledError", None).outcome == INCOMPLETE
    assert classify_waldo_decision(None, None, present=False).outcome == ABSENT


def test_aggregate_reports_strict_confusion_over_both_ran_entries():
    entries = _entries(
        _row("both", (RAN, "SAW_INSUFFICIENT_TOOLS"), (RAN, "SAW_INSUFFICIENT_TOOLS")),
        _row("teacher-only", (RAN, "SAW_INSUFFICIENT_TOOLS"), (RAN, "SAW_READY")),
        _row("fallback", (RAN, "SAW_INSUFFICIENT_TOOLS"), (RAN, "MAX_LOOPS_EXHAUSTED")),
        _row("student-only", (RAN, "SAW_READY"), (RAN, "SAW_INSUFFICIENT_TOOLS")),
        _row("neither", (RAN, "SAW_READY"), (RAN, "SAW_READY")),
        # The teacher answered and the student fell back: neither escalated, but they do not agree.
        _row("missed-answer", (RAN, "SAW_READY"), (RAN, "MAX_LOOPS_EXHAUSTED")),
    )
    per_entry = {m.entry_id: m for m in entries}
    metrics = aggregate_escalation_match_metrics(per_entry)

    strict = metrics.strict
    assert (strict.both, strict.teacher_only, strict.student_only, strict.neither) == (1, 2, 1, 2)
    assert strict.neither_student_fallback == 1
    assert strict.teacher_rate == pytest.approx(3 / 6)
    assert strict.student_rate == pytest.approx(2 / 6)
    assert metrics.escalation_match == pytest.approx(2 / 6)
    assert strict.agreement == metrics.escalation_match
    assert strict.overlap == pytest.approx(1 / 4)
    assert not per_entry["missed-answer"].escalation_match and per_entry["neither"].escalation_match
    assert aggregate_escalation_match_metrics({}).escalation_match == 0.0


def test_coverage_separates_skip_status_from_skip_reason_mismatches():
    coverage = summarize_coverage(
        _entries(
            _row("ran", (RAN, "SAW_READY"), (RAN, "SAW_READY")),
            _row("same-skip", ("skipped:file_upload", None), ("skipped:file_upload", None)),
            _row("reason-swap", ("skipped:linked_document", None), ("skipped:document_context", None)),
            _row("status", ("skipped:file_upload", None), (RAN, "SAW_READY")),
            _row("incomplete", ("Span failed with CancelledError", None), (RAN, "SAW_READY")),
            _row("teacher-only", (RAN, "SAW_READY"), (None, None)),
        )
    )
    assert coverage.common_entries == 5
    assert coverage.unpaired_entries == 1
    assert coverage.incomplete_entries == 1
    assert coverage.skip_status_mismatches == 1
    assert coverage.skip_reason_mismatches == 1
    assert coverage.teacher_skip_reasons == {"file_upload": 2, "linked_document": 1}
    assert coverage.student_skip_reasons == {"file_upload": 1, "document_context": 1}


def test_query_reads_only_waldo_spans_from_each_entrys_latest_trace():
    sql = build_escalation_match_per_entry_query()
    assert "_TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', @start_date)" in sql and "PARSE_DATE" not in sql
    assert "@student_eval_id" in sql and "@teacher_eval_id" in sql and "UNNEST(@eval_ids)" in sql
    assert "'Agent Run: WaldoAgent'" in sql and "'Waldo'" in sql
    assert "'Waldo escalation: first-sentence refusal'" in sql
    assert "jsonPayload.ic_preloop.termination" in sql
    # Retries: the trace holding the latest span wins, so failed attempts are dropped.
    assert "ARRAY_AGG(trace_id ORDER BY start_ms DESC LIMIT 1)" in sql
    # Auto Mode escalation is a different system.
    assert "auto_mode" not in sql
    for role in ("student", "teacher"):
        for column in ("agent_message", "termination", "waldo_summary", "first_sentence_refusal", "trace_id"):
            assert f"{role}.{column} AS {role}_{column}" in sql


_RUN_DAY = date(2026, 10, 1)
_RUN_MS = int(datetime(2026, 10, 1, 12, tzinfo=timezone.utc).timestamp() * 1000)


def _client(*rows: dict[str, Any]) -> MagicMock:
    client = MagicMock()
    client.query.side_effect = [[{"min_start_ms": _RUN_MS, "max_start_ms": _RUN_MS + 60_000}], list(rows)]
    return client


def _fetch(client: MagicMock):
    return fetch_eval_run_escalation_match_analysis(
        client, teacher_eval_id="teacher", student_eval_id="student", end_date=_RUN_DAY
    )


def test_fetch_scores_only_entries_where_waldo_ran_in_both_runs(capsys):
    client = _client(
        _row("e1", (RAN, "SAW_INSUFFICIENT_TOOLS"), (RAN, "SAW_READY")),
        _row("e2", (RAN, "SAW_READY"), (RAN, "SAW_READY")),
        _row("e3", ("skipped:file_upload", None), ("skipped:file_upload", None)),
        _row("e4", ("skipped:file_upload", None), (RAN, "SAW_INSUFFICIENT_TOOLS")),
        _row("e5", (RAN, "SAW_READY"), (None, None)),
    )
    analysis = _fetch(client)

    assert set(analysis.per_entry) == {"e1", "e2"}
    assert analysis.high_signal_entry_ids == ("e1",)
    assert analysis.aggregate.escalation_match == pytest.approx(0.5)
    assert analysis.aggregate.coverage.common_entries == 4
    assert analysis.aggregate.coverage.skip_status_mismatches == 1
    params = {p.name: p.value for p in client.query.call_args_list[1].kwargs["params"]}
    assert params["teacher_eval_id"] == "teacher" and params["student_eval_id"] == "student"

    EscalationMatchObjective().validate_full_eval(analysis)
    out = capsys.readouterr().out
    assert "WARNING: 1 common entries skipped Waldo in one run only" in out
    assert "teacher=50.0% student=0.0%" in out

    skipped = _fetch(_client(_row("e1", ("skipped:file_upload", None), ("skipped:file_upload", None))))
    with pytest.raises(NoComparedEntriesError, match="file_upload"):
        EscalationMatchObjective().require_compared_entries(skipped)


def test_rows_emit_only_the_strict_score_and_focused_rate_divides_by_request():
    analysis = _fetch(
        _client(
            _row("e1", (RAN, "SAW_INSUFFICIENT_TOOLS"), (RAN, "NO_SAMPLE")),
            _row("e2", (RAN, "SAW_READY"), (RAN, "SAW_READY")),
        )
    )
    objective = EscalationMatchObjective()
    rows = {
        row.entry_id: row
        for row in objective.scored_rows(
            analysis, focused=False, capture_traces=True, query="set:1", deployment_id="scio-prod"
        )
    }
    assert objective.telemetry_dimensions == ("escalation_match",)
    assert rows["e1"].dimension_scores == {"escalation_match": 0.0}
    assert rows["e1"].output["teacher_waldo_termination"] == "SAW_INSUFFICIENT_TOOLS"
    assert rows["e1"].output["student_waldo_termination"] == "NO_SAMPLE"
    assert objective.is_high_signal(rows["e1"].output) and not objective.is_high_signal(rows["e2"].output)

    [aggregate] = objective.scored_rows(
        analysis, focused=False, capture_traces=False, query="set:1", deployment_id="scio-prod"
    )
    assert aggregate.entry_id is None
    assert aggregate.dimension_scores == {"escalation_match": 0.5}
    # A requested entry with no telemetry counts as a miss.
    assert objective.focused_pass_rate(analysis, ["e1", "e2", "missing"]) == pytest.approx(1 / 3)


def test_focused_screen_scores_the_student_against_the_parent_teacher():
    # Rerun pair: the teacher rerun is ignored, even where it ran out of loops or is absent.
    analysis = _fetch(
        _client(
            _row("over-fixed", (RAN, "MAX_LOOPS_EXHAUSTED"), (RAN, "SAW_READY")),
            _row("over-fallback", (RAN, "SAW_READY"), (RAN, "NO_SAMPLE")),
            _row("over-kept", (RAN, "SAW_READY"), (RAN, "SAW_INSUFFICIENT_TOOLS")),
            _row("under-fixed", (None, None), (RAN, "SAW_INSUFFICIENT_TOOLS")),
        )
    )
    objective = EscalationMatchObjective()
    assert objective.screen_reference({"teacher_waldo_termination": "SAW_READY"}) == "SAW_READY"
    assert objective.screen_reference({}) is None
    references = {
        "over-fixed": "SAW_READY",
        "over-fallback": "NO_SAMPLE",
        "over-kept": "SAW_READY",
        "under-fixed": "SAW_INSUFFICIENT_TOOLS",
        "missing": "SAW_READY",
    }
    # Fixed: answering where the parent teacher answered, escalating where it escalated. A
    # student fallback is not an answer, and a requested entry with no student run is a miss.
    rate = objective.focused_reference_pass_rate(analysis, list(references), references)
    assert rate == pytest.approx(2 / 5)

    with pytest.raises(NoComparedEntriesError, match="entry-id mapping"):
        objective.focused_reference_pass_rate(analysis, ["missing"], {"missing": "SAW_READY"})


def test_restated_routing_modules_point_the_reflector_at_their_discover_lines():
    objective = EscalationMatchObjective()
    stock = {key: stock_text(key) or "" for key in ("WALDO_ROUTING", "WALDO_SYSTEM", "WALDO_TOOL_USAGE")}
    routing = objective.reflection_prompt("WALDO_ROUTING")
    for heading in ("### Role & Capabilities", "### Routing"):
        assert heading in routing and heading in stock["WALDO_ROUTING"]
    system = objective.reflection_prompt("WALDO_SYSTEM")
    assert "### Available Tools" in system and "### Available Tools" in stock["WALDO_SYSTEM"]
    tool_usage = objective.reflection_prompt("WALDO_TOOL_USAGE")
    assert "Be persistent on info-seeking lookups" in tool_usage
    assert "Be persistent on info-seeking lookups" in stock["WALDO_TOOL_USAGE"]


def _trajectory(output: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "data": {"eval_set_name": "set", "eval_set_version": "1", "deployment_ids": ["scio-prod"]},
        "output": {"entry_id": "e1", "deployment_id": "scio-prod", "query": "who owns billing?", **output},
        "score": score,
        "objective_scores": {"escalation_match": score},
    }


def test_reflective_feedback_names_the_direction_and_each_sides_waldo_summary():
    objective = EscalationMatchObjective()
    under = objective.build_reflective_example(
        "WALDO_ROUTING",
        _trajectory(
            {
                "teacher_waldo_termination": "SAW_INSUFFICIENT_TOOLS",
                "teacher_waldo_summary": "termination=SAW_INSUFFICIENT_TOOLS, loops=1, tool_calls=0",
                "teacher_first_sentence_refusal": True,
                "student_waldo_termination": "MAX_LOOPS_EXHAUSTED",
                "student_waldo_summary": "termination=MAX_LOOPS_EXHAUSTED, loops=3, tool_calls=4",
            },
            0.0,
        ),
        {},
    )
    feedback = under["Feedback"]
    assert feedback.startswith("Under-escalation: the teacher escalated to the full agent")
    assert "ended with MAX_LOOPS_EXHAUSTED, reaching the full agent only as a fallback" in feedback
    assert "loops=3, tool_calls=4" in feedback
    assert "first-sentence refusal" in feedback

    over = objective.build_reflective_example(
        "WALDO_ROUTING",
        _trajectory(
            {"teacher_waldo_termination": "SAW_READY", "student_waldo_termination": "SAW_INSUFFICIENT_TOOLS"}, 0.0
        ),
        {},
    )
    assert over["Feedback"].startswith("Over-escalation: the teacher answered with its attached tools (SAW_READY)")
    missed = objective.build_reflective_example(
        "WALDO_ROUTING",
        _trajectory(
            {"teacher_waldo_termination": "SAW_READY", "student_waldo_termination": "MAX_LOOPS_EXHAUSTED"}, 0.0
        ),
        {},
    )
    assert missed["Feedback"].startswith("Missed answer: the teacher answered with its attached tools (SAW_READY)")
    assert "ended with MAX_LOOPS_EXHAUSTED" in missed["Feedback"]
    prompt = objective.reflection_prompt("WALDO_ROUTING")
    assert WALDO_ROUTING_FRAME in prompt
    assert "SAW_INSUFFICIENT_TOOLS" in prompt
    assert "discover" in prompt


def test_teacher_loop_exhaustion_and_timeouts_are_left_out_of_the_match():
    analysis = _fetch(
        _client(
            _row("loops", (RAN, "MAX_LOOPS_EXHAUSTED"), (RAN, "SAW_INSUFFICIENT_TOOLS")),
            _row("timeout", (RAN, "NO_SAMPLE"), (RAN, "NO_SAMPLE")),
            _row("answered", (RAN, "SAW_READY"), (RAN, "SAW_INSUFFICIENT_TOOLS")),
        )
    )
    objective = EscalationMatchObjective()
    rows = {
        row.entry_id: row
        for row in objective.scored_rows(
            analysis, focused=False, capture_traces=True, query="set:1", deployment_id="scio-prod"
        )
    }
    for unscored in ("loops", "timeout"):
        assert unscored not in analysis.per_entry and unscored not in rows
    assert analysis.aggregate.compared_entries == 1
    assert analysis.aggregate.coverage.teacher_loop_exhaustion == 1
    assert analysis.aggregate.coverage.teacher_timeouts == 1
    assert analysis.aggregate.escalation_match == 0.0
    assert analysis.high_signal_entry_ids == ("answered",)
    assert objective.is_high_signal(rows["answered"].output)


def test_mismatch_pair_and_reflection_selection_keep_the_rare_direction():
    assert escalation_mismatch_pair("SAW_READY", "SAW_READY") is None
    assert escalation_mismatch_pair("SAW_READY", "NO_SAMPLE") == ("SAW_READY", "NO_SAMPLE")
    assert escalation_mismatch_pair("SAW_INSUFFICIENT_TOOLS", "SAW_READY") == ("SAW_INSUFFICIENT_TOOLS", "SAW_READY")
    assert escalation_mismatch_pair("", "") is None
    assert escalation_mismatch_pair("SAW_READY", "") is None
    # Teacher loop exhaustion and timeouts are not scored or reflected. A student fallback still is.
    assert escalation_mismatch_pair("MAX_LOOPS_EXHAUSTED", "SAW_INSUFFICIENT_TOOLS") is None
    assert escalation_mismatch_pair("NO_SAMPLE", "SAW_INSUFFICIENT_TOOLS") is None
    assert escalation_mismatch_pair("SAW_INSUFFICIENT_TOOLS", "MAX_LOOPS_EXHAUSTED") == (
        "SAW_INSUFFICIENT_TOOLS",
        "MAX_LOOPS_EXHAUSTED",
    )

    under = ("SAW_INSUFFICIENT_TOOLS", "SAW_READY")
    over = ("SAW_READY", "SAW_INSUFFICIENT_TOOLS")
    keys = [under] * 30 + [None, over]
    selected, groups = interleave_mismatch_groups(keys, max_entries=5)
    assert len(selected) == 5 and 31 in selected
    assert groups == [(*under, 4), (*over, 1)]
    everything, _ = interleave_mismatch_groups(keys, max_entries=None)
    assert len(everything) == 31
