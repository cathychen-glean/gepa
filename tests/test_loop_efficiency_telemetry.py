"""Telemetry-layer tests for loop counting. Objective tests live in ``test_loop_objective.py``."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock

from glean_gepa.objectives.loop_efficiency import (
    LoopCountEntryMetrics,
    build_loop_count_per_entry_query,
    fetch_eval_run_loop_count_analysis,
    loop_efficiency_score,
    overlay_evalcli_loop_and_correctness,
    parse_loop_count_entry_metrics,
)


def test_loop_efficiency_is_one_at_or_below_target_and_decays_above_it():
    assert loop_efficiency_score(0) == 1.0
    assert loop_efficiency_score(2) == 1.0
    assert loop_efficiency_score(3) == 0.5
    assert loop_efficiency_score(4) == 1 / 3


def test_parse_uses_errors_as_correctness_when_judge_score_is_missing():
    ok = parse_loop_count_entry_metrics({"entry_id": "e1", "loop_count": 2, "has_error": False})
    assert ok.correctness == 1.0
    assert ok.loop_efficiency == 1.0

    failed = parse_loop_count_entry_metrics({"entry_id": "e2", "loop_count": 1, "has_error": True})
    assert failed.correctness == 0.0
    assert failed.loop_efficiency == 1.0

    judged = parse_loop_count_entry_metrics({"entry_id": "e3", "loop_count": 3, "has_error": False, "correctness": 1.0})
    assert judged.loop_efficiency == 0.5


def _trace_with_action_inputs(*action_inputs: str) -> dict:
    """A detailed-trace payload whose Execute Action spans carry ``action_input``."""
    return {
        "trace": {
            "spans": [
                {
                    "name": "Execute Action: Shell",
                    "attributes": {"input": {"strValue": json.dumps({"action_input": action_input})}},
                }
                for action_input in action_inputs
            ]
        }
    }


def test_loop_count_query_and_fetch():
    sql = build_loop_count_per_entry_query()
    assert "Execute Action:" in sql
    assert "GROUP BY entry_id" in sql
    assert "COUNTIF(is_loop)" in sql
    # Tool payloads are scrubbed from this table, so the query must NOT read them here
    # and must instead carry the scrub-safe locators used to fetch the detailed trace.
    assert "span_info.inputs" not in sql
    assert "agent_trace.trace_id" in sql
    assert "min_start_ms" in sql and "max_start_ms" in sql

    client = MagicMock()
    client.query.side_effect = [
        [{"min_start_ms": 1_786_363_200_000, "max_start_ms": 1_786_449_600_000}],
        [
            {"entry_id": "entry-1", "loop_count": 2, "has_error": False},
            {
                "entry_id": "entry-2",
                "loop_count": 5,
                "has_error": False,
                "trace_id": "trace-2",
                "deployment_id": "scio-prod",
                "min_start_ms": 1_786_400_000_000,
                "max_start_ms": 1_786_400_050_000,
            },
            {"entry_id": "entry-3", "loop_count": 1, "has_error": True},
        ],
    ]

    evalcli = MagicMock()
    evalcli.get_analysis_view.return_value = {}
    evalcli.get_analysis_trace.return_value = _trace_with_action_inputs('{"command":"ls"}')

    analysis = fetch_eval_run_loop_count_analysis(
        client,
        eval_id="student",
        lookback_days=7,
        end_date=date(2026, 8, 11),
        evalcli=evalcli,
    )
    assert analysis.per_entry["entry-1"].loop_efficiency == 1.0
    # entry-1 is not high-signal, so its trace is never fetched.
    assert analysis.per_entry["entry-1"].action_inputs == ()
    assert analysis.per_entry["entry-2"].loop_efficiency == 1 / 4
    # High-signal entry-2's tool payload is resolved from its detailed trace.
    assert analysis.per_entry["entry-2"].action_inputs == ('{"command":"ls"}',)
    # entry-3 errored, but with no floor it scores by its loop count (one loop).
    assert analysis.per_entry["entry-3"].loop_efficiency == 1.0
    assert analysis.per_entry["entry-3"].action_inputs == ()
    assert analysis.high_signal_entry_ids == ("entry-2",)
    assert analysis.aggregate.matching_entries == 2
    assert client.query.call_count == 2

    evalcli.get_analysis_trace.reset_mock()
    client.query.side_effect = [
        [{"min_start_ms": 1_786_363_200_000, "max_start_ms": 1_786_449_600_000}],
        [
            {
                "entry_id": "entry-2",
                "loop_count": 5,
                "has_error": False,
                "trace_id": "trace-2",
                "deployment_id": "scio-prod",
                "min_start_ms": 1_786_400_000_000,
                "max_start_ms": 1_786_400_050_000,
            }
        ],
    ]
    skipped = fetch_eval_run_loop_count_analysis(
        client,
        eval_id="student",
        lookback_days=7,
        end_date=date(2026, 8, 11),
        evalcli=evalcli,
        include_action_inputs=False,
    )
    assert skipped.per_entry["entry-2"].loop_efficiency == 1 / 4
    assert skipped.per_entry["entry-2"].action_inputs == ()
    evalcli.get_analysis_trace.assert_not_called()
    evalcli.get_analysis_view.assert_called()


def test_evalcli_overlay_prefers_loopcount_and_correctness_judge():
    per_entry = {
        "e1": LoopCountEntryMetrics(
            "e1", loop_count=9, correctness=1.0, has_error=False, action_inputs=('{"command":"ls"}',)
        ),
    }
    evalcli = MagicMock()
    evalcli.get_analysis_view.return_value = {
        "entries": [
            {
                "entryId": "e1",
                "evalRunEntries": [
                    {"evalRunId": "run-1", "metadata": {"loopCount": 2}},
                ],
                "judgeRunEntries": [
                    {"outputs": [{"name": "CORRECTNESS", "score": 1.0}]},
                ],
            },
            {
                "entryId": "e2",
                "evalRunEntries": [
                    {"evalRunId": "run-1", "metadata": {"loopCount": 1}},
                ],
                "judgeRunEntries": [
                    {"outputs": [{"name": "CORRECTNESS", "score": 0.0}]},
                ],
            },
        ]
    }
    updated = overlay_evalcli_loop_and_correctness(evalcli, "run-1", per_entry)
    assert updated["e1"].loop_count == 2
    assert updated["e1"].loop_efficiency == 1.0
    # Overlay must not drop the action-input evidence captured from BigQuery.
    assert updated["e1"].action_inputs == ('{"command":"ls"}',)
    assert updated["e2"].loop_count == 1
    assert updated["e2"].loop_efficiency == 1.0
