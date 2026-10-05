from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from glean_gepa.objectives import registry
from glean_gepa.objectives.escalation_match import (
    ABSENT,
    ANSWERED,
    ESCALATED,
    INCOMPLETE,
    OTHER,
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


def test_classification_follows_the_handoff_rules():
    assert classify_waldo_decision(RAN, "SAW_INSUFFICIENT_TOOLS").outcome == ESCALATED
    assert classify_waldo_decision(RAN, "SAW_READY").outcome == ANSWERED
    # Fallback terminations hand off to the inner loop but are not escalation decisions.
    no_sample = classify_waldo_decision(RAN, "NO_SAMPLE")
    assert no_sample.outcome == OTHER and no_sample.handed_off and not no_sample.escalated
    # A WaldoAgent skip wins over a termination left on the same trace by an earlier turn.
    skipped = classify_waldo_decision("skipped:user_turn_limit", "SAW_READY")
    assert skipped.outcome == "skip:user_turn_limit" and skipped.skip_reason == "user_turn_limit"
    assert not skipped.ran
    assert classify_waldo_decision("Span failed with CancelledError", None).outcome == INCOMPLETE
    assert classify_waldo_decision(None, None, present=False).outcome == ABSENT


def test_aggregate_reports_strict_and_effective_confusion_over_both_ran_entries():
    entries = _entries(
        _row("both", (RAN, "SAW_INSUFFICIENT_TOOLS"), (RAN, "SAW_INSUFFICIENT_TOOLS")),
        _row("teacher-only", (RAN, "SAW_INSUFFICIENT_TOOLS"), (RAN, "SAW_READY")),
        _row("fallback", (RAN, "SAW_INSUFFICIENT_TOOLS"), (RAN, "MAX_LOOPS_EXHAUSTED")),
        _row("student-only", (RAN, "SAW_READY"), (RAN, "SAW_INSUFFICIENT_TOOLS")),
        _row("neither", (RAN, "SAW_READY"), (RAN, "SAW_READY")),
    )
    metrics = aggregate_escalation_match_metrics({m.entry_id: m for m in entries})

    strict = metrics.strict
    assert (strict.both, strict.teacher_only, strict.student_only, strict.neither) == (1, 2, 1, 1)
    assert strict.teacher_rate == pytest.approx(3 / 5)
    assert strict.student_rate == pytest.approx(2 / 5)
    assert metrics.escalation_match == pytest.approx(2 / 5)
    assert strict.overlap == pytest.approx(1 / 4)
    # Effective handoff counts the student's loop exhaustion as a hand-off.
    effective = metrics.effective
    assert (effective.both, effective.teacher_only, effective.student_only, effective.neither) == (2, 1, 1, 1)
    assert metrics.handoff_match == pytest.approx(3 / 5)
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


def test_zero_compared_entries_is_rejected_with_the_skip_breakdown():
    analysis = _fetch(_client(_row("e1", ("skipped:file_upload", None), ("skipped:file_upload", None))))
    with pytest.raises(NoComparedEntriesError, match="file_upload"):
        EscalationMatchObjective().require_compared_entries(analysis)


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
    # Effective handoff would match here (NO_SAMPLE hands off), but it must not become a frontier key.
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
    assert analysis.aggregate.handoff_match == pytest.approx(1.0)
    # A requested entry with no telemetry counts as a miss.
    assert objective.focused_pass_rate(analysis, ["e1", "e2", "missing"]) == pytest.approx(1 / 3)


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
        "WALDO_TOOL_USAGE",
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
        "WALDO_TOOL_USAGE",
        _trajectory(
            {"teacher_waldo_termination": "SAW_READY", "student_waldo_termination": "SAW_INSUFFICIENT_TOOLS"}, 0.0
        ),
        {},
    )
    assert over["Feedback"].startswith("Over-escalation: the teacher answered with its attached tools (SAW_READY)")
    assert "discover" in objective.reflection_prompt("WALDO_TOOL_USAGE")


def test_mismatch_pair_and_reflection_selection_keep_the_rare_direction():
    assert escalation_mismatch_pair("SAW_READY", "NO_SAMPLE") is None
    assert escalation_mismatch_pair("SAW_INSUFFICIENT_TOOLS", "SAW_READY") == ("SAW_INSUFFICIENT_TOOLS", "SAW_READY")
    assert escalation_mismatch_pair("", "") is None

    under = ("SAW_INSUFFICIENT_TOOLS", "SAW_READY")
    over = ("SAW_READY", "SAW_INSUFFICIENT_TOOLS")
    keys = [under] * 30 + [None, over]
    selected, groups = interleave_mismatch_groups(keys, max_entries=5)
    assert len(selected) == 5 and 31 in selected
    assert groups == [(*under, 4), (*over, 1)]
    everything, _ = interleave_mismatch_groups(keys, max_entries=None)
    assert len(everything) == 31


def test_registered_for_teacher_student_mode():
    assert registry.resolve("teacher_student", "escalation_match") is EscalationMatchObjective
    assert (
        registry.validate_objective_class(EscalationMatchObjective, mode="teacher_student", source="escalation_match")
        == []
    )
