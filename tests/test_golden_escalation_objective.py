from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from glean_gepa.experiment_config import experiment_objective_spec, load_experiment_config
from glean_gepa.objectives import AnalysisRequest
from glean_gepa.objectives.registry import build_objective
from glean_gepa.objectives.golden_escalation import (
    ANSWERED,
    ESCALATED,
    INCOMPLETE,
    MISSED_ANSWER,
    OTHER,
    OVER_ESCALATION,
    UNDER_ESCALATION,
    WALDO_ROUTING_FRAME,
    GoldenEscalationMatchObjective,
    GoldenLabel,
    build_golden_escalation_per_entry_query,
    classify_waldo_decision,
    entries_with_labels,
    fetch_eval_run_golden_escalation_analysis,
    mismatch_direction,
    parse_golden_label,
    parse_label_eval_sets,
)

RAN = "Agent execution completed successfully"
DISCOVER = GoldenLabel(route="discover")
DIRECT = GoldenLabel(route="direct_response")


def _canonical(route: str, **notes: str) -> str:
    return json.dumps({"id": "label-1", "query": "q", "target_route": route, **notes})


def _entry(entry_id: str, query: str, route: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"id": entry_id, "input": {"query": query}}
    if route:
        entry["expectedOutput"] = {"canonicalAnswer": _canonical(route)}
    return entry


def _evalcli(sets: dict[tuple[str, str], list[dict[str, Any]]]) -> MagicMock:
    evalcli = MagicMock()
    evalcli.list_eval_set_entries.side_effect = lambda *, eval_set_name, eval_set_version, deployment_ids: sets[
        (eval_set_name, eval_set_version)
    ]
    return evalcli


def _request(evalcli: Any, name: str = "golden", version: str = "train") -> AnalysisRequest:
    return AnalysisRequest(evalcli=evalcli, eval_set_name=name, eval_set_version=version, deployment_ids=("scio-prod",))


def _objective(**params: Any) -> GoldenEscalationMatchObjective:
    objective = GoldenEscalationMatchObjective(bigquery_client=MagicMock(), lookback_days=7)
    objective.params = dict(params)
    return objective


def test_decision_and_direction_follow_the_golden_route():
    escalated = classify_waldo_decision(RAN, "SAW_INSUFFICIENT_TOOLS")
    answered = classify_waldo_decision(RAN, "SAW_READY")
    fallback = classify_waldo_decision(RAN, "NO_SAMPLE")
    assert (escalated.outcome, answered.outcome, fallback.outcome) == (ESCALATED, ANSWERED, OTHER)
    skipped = classify_waldo_decision("skipped:linked_document", "SAW_READY")
    assert skipped.skip_reason == "linked_document" and not skipped.ran
    assert classify_waldo_decision("Span failed with CancelledError", None).outcome == INCOMPLETE

    assert mismatch_direction(DISCOVER, escalated) == ""
    assert mismatch_direction(DIRECT, answered) == ""
    assert mismatch_direction(DISCOVER, answered) == UNDER_ESCALATION
    # A fallback reaches the full agent but is not a decision: it misses both routes.
    assert mismatch_direction(DISCOVER, fallback) == UNDER_ESCALATION
    assert mismatch_direction(DIRECT, escalated) == OVER_ESCALATION
    assert mismatch_direction(DIRECT, fallback) == MISSED_ANSWER


def test_labels_parse_from_canonical_answer_and_carry_reviewer_notes():
    label = parse_golden_label(_canonical("direct_response", review_notes="simple lookup", label_caveat=" "))
    assert label == GoldenLabel(route="direct_response", notes="simple lookup")
    assert parse_golden_label({"target_route": "discover"}) == DISCOVER
    assert parse_golden_label(json.dumps({"target_route": "teacher_answer"})) is None
    assert parse_golden_label("not json") is None
    assert parse_golden_label(None) is None
    assert len(parse_golden_label({"target_route": "discover", "human_comment": "x" * 900}).notes) == 300  # type: ignore[union-attr]

    resolved = entries_with_labels([_entry("e1", " who owns billing? ", "discover"), _entry("e2", "go"), {"id": ""}])
    assert resolved == {"e1": ("who owns billing?", DISCOVER), "e2": ("go", None)}

    assert parse_label_eval_sets(["waldo_golden:holdout_labeled", "a:b:c"]) == [
        ("waldo_golden", "holdout_labeled"),
        ("a:b", "c"),
    ]
    assert parse_label_eval_sets(None) == []
    for bad in ("waldo_golden:holdout", ["no-version"], [":v"]):
        with pytest.raises(ValueError, match="name:version"):
            parse_label_eval_sets(bad)


def test_focused_copy_entries_take_the_label_of_the_same_query():
    evalcli = _evalcli(
        {
            ("golden", "train"): [_entry("t1", "who owns billing?", "discover"), _entry("t2", "go", "direct_response")],
            # A QUERY_CANONICAL copy: new ids, same queries, no canonicalAnswer.
            ("gepa-high-signal-golden", "train_hs"): [
                _entry("f1", "who owns billing?"),
                _entry("f2", "go"),
                _entry("f3", "unlabeled query"),
            ],
        }
    )
    objective = _objective(label_eval_sets=["golden:train"])

    labels = objective.entry_labels(_request(evalcli, "gepa-high-signal-golden", "train_hs"))
    assert labels == {"f1": ("who owns billing?", DISCOVER), "f2": ("go", DIRECT), "f3": ("unlabeled query", None)}
    # Each set is listed once per objective.
    assert objective.entry_labels(_request(evalcli)) == {"t1": ("who owns billing?", DISCOVER), "t2": ("go", DIRECT)}
    assert evalcli.list_eval_set_entries.call_count == 2
    assert evalcli.list_eval_set_entries.call_args_list[0].kwargs["deployment_ids"] == ["scio-prod"]


def test_label_join_refuses_conflicts_and_runs_without_labels():
    conflicting = _evalcli(
        {
            ("golden", "a"): [_entry("a1", "go", "direct_response")],
            ("golden", "b"): [_entry("b1", "go", "discover")],
        }
    )
    with pytest.raises(ValueError, match="Conflicting golden routes"):
        _objective(label_eval_sets=["golden:a", "golden:b"]).entry_labels(_request(conflicting, "golden", "a"))

    unlabeled = _evalcli({("golden", "train"): [_entry("e1", "go")]})
    with pytest.raises(ValueError, match="label_eval_sets"):
        _objective().entry_labels(_request(unlabeled))
    with pytest.raises(ValueError, match="eval set in AnalysisRequest"):
        _objective().entry_labels(AnalysisRequest(evalcli=unlabeled))
    with pytest.raises(ValueError, match="EvalCLI client"):
        _objective().entry_labels(AnalysisRequest(eval_set_name="golden", eval_set_version="train"))


def test_query_reads_only_the_students_waldo_spans_from_each_entrys_latest_trace():
    sql = build_golden_escalation_per_entry_query()
    assert "_TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', @start_date)" in sql
    assert "eval_id = @eval_id" in sql and "teacher" not in sql
    assert "'Agent Run: WaldoAgent'" in sql and "'Waldo'" in sql
    assert "'Waldo escalation: first-sentence refusal'" in sql
    assert "jsonPayload.ic_preloop.termination" in sql
    assert "ARRAY_AGG(trace_id ORDER BY start_ms DESC LIMIT 1)" in sql
    assert "auto_mode" not in sql
    for column in ("agent_message", "termination", "waldo_summary", "first_sentence_refusal"):
        assert f"AS {column}" in sql


_RUN_DAY = date(2026, 10, 1)
_RUN_MS = int(datetime(2026, 10, 1, 12, tzinfo=timezone.utc).timestamp() * 1000)


def _row(entry_id: str, message: str | None, termination: str | None, *, refusal: bool = False) -> dict[str, Any]:
    return {
        "entry_id": entry_id,
        "trace_id": f"trace-{entry_id}",
        "agent_message": message,
        "termination": termination,
        "waldo_summary": f"termination={termination}, loops=1, tool_calls=0" if termination else None,
        "first_sentence_refusal": refusal,
    }


def _client(*rows: dict[str, Any]) -> MagicMock:
    client = MagicMock()
    client.query.side_effect = [[{"min_start_ms": _RUN_MS, "max_start_ms": _RUN_MS + 60_000}], list(rows)]
    return client


_LABELS = {
    "over-1": ("q over 1", DIRECT),
    "over-2": ("q over 2", DIRECT),
    "over-3": ("q over 3", DIRECT),
    "under": ("q under", DISCOVER),
    "missed": ("q missed", DIRECT),
    "match": ("q match", DISCOVER),
    "skipped": ("q skipped", DISCOVER),
    "incomplete": ("q incomplete", DIRECT),
    "no-waldo": ("q no waldo", DISCOVER),
    "unlabeled": ("q unlabeled", None),
}


def _fetch(client: MagicMock):
    return fetch_eval_run_golden_escalation_analysis(client, eval_id="student", entry_labels=_LABELS, end_date=_RUN_DAY)


def _analysis():
    return _fetch(
        _client(
            _row("over-1", RAN, "SAW_INSUFFICIENT_TOOLS"),
            _row("over-2", RAN, "SAW_INSUFFICIENT_TOOLS", refusal=True),
            _row("over-3", RAN, "SAW_INSUFFICIENT_TOOLS"),
            _row("under", RAN, "SAW_READY"),
            _row("missed", RAN, "MAX_LOOPS_EXHAUSTED"),
            _row("match", RAN, "SAW_INSUFFICIENT_TOOLS"),
            _row("skipped", "skipped:linked_document", None),
            _row("incomplete", "Span failed with CancelledError", None),
            _row("unlabeled", RAN, "SAW_READY"),
            _row("not-in-set", RAN, "SAW_READY"),
        )
    )


def test_fetch_scores_labeled_entries_where_waldo_ran_and_interleaves_directions():
    analysis = _analysis()

    assert set(analysis.per_entry) == {"over-1", "over-2", "over-3", "under", "missed", "match"}
    aggregate = analysis.aggregate
    assert aggregate.waldo_entries == 10 and aggregate.compared_entries == 6
    assert aggregate.golden_escalation_match == pytest.approx(1 / 6)
    assert (aggregate.under_escalations, aggregate.over_escalations, aggregate.missed_answers) == (1, 3, 1)
    assert aggregate.student_fallbacks == 1
    assert aggregate.routes == {"direct_response": 4, "discover": 2}
    coverage = aggregate.coverage
    assert coverage.labeled_without_waldo == 1
    assert coverage.unlabeled_entries == 2
    assert coverage.incomplete_entries == 1
    assert coverage.skip_reasons == {"linked_document": 1}
    # The rarer directions reach the reflector before the second over-escalation.
    first, *rest = analysis.high_signal_entry_ids
    assert first.startswith("over-") and set(rest[:2]) == {"under", "missed"}
    assert set(analysis.high_signal_entry_ids) == {"over-1", "over-2", "over-3", "under", "missed"}

    client = _client(_row("match", RAN, "SAW_INSUFFICIENT_TOOLS"))
    _fetch(client)
    params = {p.name: p.value for p in client.query.call_args_list[1].kwargs["params"]}
    assert params["eval_id"] == "student"


def test_analyze_joins_labels_for_the_runs_eval_set():
    evalcli = _evalcli({("golden", "train"): [_entry("e1", "go", "direct_response")]})
    objective = _objective()
    with patch(
        "glean_gepa.objectives.golden_escalation.fetch_eval_run_golden_escalation_analysis",
        return_value=_analysis(),
    ) as fetch:
        first = objective.analyze("student-run", request=_request(evalcli))
        assert objective.analyze("student-run", request=_request(evalcli)) is first

    fetch.assert_called_once()
    assert fetch.call_args.kwargs["eval_id"] == "student-run"
    assert fetch.call_args.kwargs["entry_labels"] == {"e1": ("go", DIRECT)}
    assert fetch.call_args.kwargs["lookback_days"] == 7


def _rows(objective: GoldenEscalationMatchObjective, analysis: Any, **kwargs: Any):
    return objective.scored_rows(
        analysis,
        al_data_inst={},
        student_eval_id="student",
        eval_set_name="golden",
        eval_set_version="train",
        deployment_ids=["scio-prod"],
        **{"requested_entry_ids": None, "is_focused_eval": False, "capture_traces": True, **kwargs},
    )


def test_rows_carry_the_golden_route_and_focused_rate_divides_by_request(capsys):
    analysis = _analysis()
    objective = _objective()

    rows = {row.entry_id: row for row in _rows(objective, analysis)}
    assert set(rows) == {"over-1", "over-2", "over-3", "under", "missed"}
    over = rows["over-2"]
    assert over.dimension_scores == {"golden_escalation_match": 0.0}
    assert over.output["query"] == "q over 2"
    assert over.output["golden_route"] == "direct_response"
    assert over.output["student_waldo_termination"] == "SAW_INSUFFICIENT_TOOLS"
    assert over.output["student_first_sentence_refusal"] is True
    assert over.data_overrides == {"eval_entry_id": "over-2", "eval_run_id": "student"}

    [aggregate] = _rows(objective, analysis, capture_traces=False)
    assert aggregate.entry_id is None
    assert aggregate.dimension_scores == {"golden_escalation_match": pytest.approx(1 / 6)}

    # Focused screen: a requested entry with no scored decision counts as a miss.
    assert objective.focused_pass_rate(analysis, ["match", "under", "skipped", "absent"]) == pytest.approx(1 / 4)
    assert "1/4 now take the golden route; 2 have no scored Waldo decision" in capsys.readouterr().out
    assert objective.focused_pass_rate(analysis, []) == 0.0

    objective.log_analysis(analysis)
    out = capsys.readouterr().out
    assert "10 entries with Waldo spans, 6 scored" in out
    assert "under-escalated 1, over-escalated 3, missed answers 1 (1 fallbacks)" in out


def _trajectory(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "data": {"eval_set_name": "golden", "eval_set_version": "train", "deployment_ids": ["scio-prod"]},
        "output": {"entry_id": "e1", "deployment_id": "scio-prod", "query": "who owns billing?", **output},
        "score": 0.0,
        "objective_scores": {"golden_escalation_match": 0.0},
    }


def test_reflective_feedback_names_the_direction_waldo_summary_and_reviewer_note():
    objective = _objective()
    under = _trajectory(
        {
            "golden_route": "discover",
            "golden_notes": "needs cross-source synthesis",
            "student_waldo_termination": "MAX_LOOPS_EXHAUSTED",
            "student_waldo_summary": "termination=MAX_LOOPS_EXHAUSTED, loops=3, tool_calls=4",
        }
    )
    feedback = objective.build_reflective_example("WALDO_ROUTING", under, {})["Feedback"]
    assert feedback.startswith("Under-escalation: the golden route is discover and Waldo did not escalate")
    assert "loops=3, tool_calls=4" in feedback
    assert "Reviewer note: needs cross-source synthesis" in feedback
    assert objective.failure_pattern("WALDO_ROUTING", under) == (UNDER_ESCALATION, "MAX_LOOPS_EXHAUSTED")  # type: ignore[arg-type]

    over = _trajectory(
        {
            "golden_route": "direct_response",
            "student_waldo_termination": "SAW_INSUFFICIENT_TOOLS",
            "student_first_sentence_refusal": True,
        }
    )
    over_feedback = objective.build_reflective_example("WALDO_ROUTING", over, {})["Feedback"]
    assert over_feedback.startswith("Over-escalation: the golden route is direct_response and Waldo escalated")
    assert "first-sentence refusal" in over_feedback

    missed = _trajectory({"golden_route": "direct_response", "student_waldo_termination": "NO_SAMPLE"})
    assert objective.build_reflective_example("WALDO_ROUTING", missed, {})["Feedback"].startswith("Missed answer")
    assert objective.failure_pattern("WALDO_ROUTING", _trajectory({})) == ()  # type: ignore[arg-type]

    prompt = objective.reflection_prompt("WALDO_ROUTING")
    assert WALDO_ROUTING_FRAME in prompt and "golden route" in prompt and "teacher" not in prompt.lower()


def test_shipped_config_builds_the_golden_objective():
    config = load_experiment_config("single_model_waldo_escalation")
    objective = build_objective(
        config.mode,
        config.signals,
        bigquery_client=MagicMock(),
        lookback_days=7,
        experiment=experiment_objective_spec(config),
    )
    assert isinstance(objective, GoldenEscalationMatchObjective)
    assert objective.focused_bucket_type == "QUERY_CANONICAL"
    assert objective.high_signal == "golden_escalation_match"
    assert parse_label_eval_sets(objective.experiment_param("label_eval_sets", None)) == [
        ("waldo_golden", "le297309_routing_labeled"),
        ("waldo_golden", "holdout_labeled"),
    ]
