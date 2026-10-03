from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from glean_gepa.bigquery_client import BigQueryClient, BigQueryError
from glean_gepa.objectives.utils.agentspan_query import default_date_range, resolve_eval_run_date_range
from glean_gepa.objectives.utils.evalset_entries import (
    build_eval_entry_uuid_tracking_query,
    build_high_signal_source_entries_query,
    fetch_evalset_entry_tracking,
    fetch_high_signal_evalset_entries,
)
from glean_gepa.objectives.shell import (
    SHELL_ACTION_IDS,
    SHELL_SPAN_NAMES,
    ShellToolErrorEntryMetrics,
    ShellToolErrorMetrics,
    build_shell_tool_error_per_entry_query,
    build_shell_tool_error_rate_query,
    empty_shell_tool_error_metrics,
    fetch_eval_run_shell_tool_error_analysis,
    is_shell_tool_error,
    parse_shell_tool_error_example,
    parse_shell_tool_error_metrics,
    shell_error_free_rate,
)


def test_is_shell_tool_error():
    assert is_shell_tool_error(
        action_status="ERROR",
        span_status="OK",
        output_status_code="OK",
        provider_status="success",
    )
    assert is_shell_tool_error(
        action_status="SUCCESS",
        span_status="OK",
        output_status_code="OK",
        provider_status="failed",
    )
    assert not is_shell_tool_error(
        action_status="SUCCESS",
        span_status="OK",
        output_status_code="OK",
        provider_status="success",
    )


def test_build_shell_tool_error_rate_query_includes_eval_and_shell_filters():
    sql = build_shell_tool_error_rate_query()

    assert "@eval_id" in sql
    assert "@start_date" in sql
    assert "@end_date" in sql
    assert "scrubbed_agentspan" in sql
    for span_name in SHELL_SPAN_NAMES:
        assert span_name in sql
    for action_id in SHELL_ACTION_IDS:
        assert action_id in sql
    assert "jsonPayload.action.error_str" in sql
    assert "jsonPayload.span_info.execution_status.message" in sql
    assert "jsonPayload.span_info.execution_status.user_message" in sql
    assert "NULLIF(jsonPayload.action.action_run_id, '')" in sql
    assert "NULLIF(jsonPayload.context.agent_trace.span_id, '')" in sql
    assert "TO_JSON_STRING(STRUCT(" in sql
    assert "TO_JSON_STRING(jsonPayload)" not in sql
    assert "PARSE_DATE" not in sql
    assert "_TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', @start_date)" in sql
    assert "action_run_id," in sql
    assert "recent_error_examples" in sql


def test_per_entry_query_collects_only_error_trace_ids_newest_first():
    sql = build_shell_tool_error_per_entry_query()

    assert "IF(is_error, trace_id, NULL)" in sql
    assert "ORDER BY start_ms DESC" in sql
    assert "AS trace_ids" in sql


def test_per_entry_query_can_skip_error_examples_for_high_signal_screening():
    sql = build_shell_tool_error_per_entry_query(include_error_examples=False)

    assert "shell_errors" in sql
    assert "recent_error_examples" not in sql
    assert "AS trace_ids" not in sql


def test_fetch_analysis_can_skip_per_entry_query():
    mock_client = MagicMock()
    mock_client.query.side_effect = [
        [{"min_start_ms": 1_786_363_200_000, "max_start_ms": 1_786_449_600_000}],
        [
            {
                "eval_id": "run_123",
                "shell_executions": 6,
                "shell_errors": 2,
                "shell_error_rate": 1 / 3,
                "shell_error_pct": 33.33,
                "recent_error_examples": [],
            }
        ],
    ]

    analysis = fetch_eval_run_shell_tool_error_analysis(
        mock_client,
        eval_id="run_123",
        end_date=date(2026, 8, 12),
        include_per_entry=False,
    )

    assert analysis.aggregate.shell_executions == 6
    assert analysis.per_entry == {}
    assert mock_client.query.call_count == 2


def test_shell_error_free_rate():
    per_entry = {
        "a": ShellToolErrorEntryMetrics(
            entry_id="a",
            shell_executions=1,
            shell_errors=0,
            shell_error_rate=0.0,
            shell_error_pct=0.0,
            recent_error_examples=(),
        ),
        "b": ShellToolErrorEntryMetrics(
            entry_id="b",
            shell_executions=1,
            shell_errors=1,
            shell_error_rate=1.0,
            shell_error_pct=100.0,
            recent_error_examples=(),
        ),
    }

    assert shell_error_free_rate(per_entry) == 0.5
    assert shell_error_free_rate({}) == 1.0


def test_parse_shell_tool_error_metrics_from_bigquery_row():
    row = {
        "eval_id": "run_123",
        "shell_executions": 10,
        "shell_errors": 2,
        "shell_error_rate": 0.2,
        "shell_error_pct": 20.0,
        "recent_error_examples": [
            {
                "started_at": "2026-08-11 12:00:00 UTC",
                "project_id": "scio-prod",
                "entry_id": "entry-1",
                "eval_id": "run_123",
                "run_id": "workflow-run-1",
                "trace_id": "trace-1",
                "span_id": "span-1",
                "span_name": "Execute Action: Shell",
                "action_id": "Shell",
                "action_run_id": "call-1",
                "action_input": '{"command":"python3 broken.py"}',
                "action_status": "ERROR",
                "span_status": "ERROR",
                "provider_status": "failed",
                "output_status_code": "ERROR",
                "error_str": "command not found: foobar",
            }
        ],
    }

    metrics = parse_shell_tool_error_metrics(row)

    assert metrics.shell_executions == 10
    assert metrics.shell_errors == 2
    assert metrics.shell_error_rate == 0.2
    assert metrics.shell_success_rate == pytest.approx(0.8)
    assert len(metrics.recent_error_examples) == 1
    assert metrics.recent_error_examples[0].error_str == "command not found: foobar"
    assert metrics.recent_error_examples[0].action_run_id == "call-1"
    assert metrics.recent_error_examples[0].action_input == '{"command":"python3 broken.py"}'


def test_parse_shell_tool_error_entry_metrics_includes_trace_ids():
    from glean_gepa.objectives.shell import parse_shell_tool_error_entry_metrics

    metrics = parse_shell_tool_error_entry_metrics(
        {
            "entry_id": "entry-1",
            "shell_executions": 2,
            "shell_errors": 1,
            "shell_error_rate": 0.5,
            "shell_error_pct": 50.0,
            "recent_error_examples": [],
            "trace_ids": ["trace-newest", "trace-older"],
        }
    )

    assert metrics.trace_ids == ("trace-newest", "trace-older")


def test_fetch_shell_tool_error_metrics_returns_parsed_row():
    mock_client = MagicMock()
    mock_client.query.side_effect = [
        [{"min_start_ms": 1_786_363_200_000, "max_start_ms": 1_786_449_600_000}],
        [
            {
                "eval_id": "run_123",
                "shell_executions": 6,
                "shell_errors": 2,
                "shell_error_rate": 1 / 3,
                "shell_error_pct": 33.33,
                "recent_error_examples": [],
            }
        ],
        [
            {
                "entry_id": "entry-1",
                "shell_executions": 4,
                "shell_errors": 1,
                "shell_error_rate": 0.25,
                "shell_error_pct": 25.0,
                "recent_error_examples": [],
            }
        ],
    ]

    metrics = fetch_eval_run_shell_tool_error_analysis(
        mock_client,
        eval_id="run_123",
        end_date=date(2026, 8, 12),
    ).aggregate

    assert metrics.shell_error_rate == pytest.approx(1 / 3)
    assert metrics.shell_errors == 2
    assert mock_client.query.call_count == 3


def test_fetch_shell_error_metrics_keeps_unattributed_shell_spans_in_aggregate():
    mock_client = MagicMock()
    mock_client.query.side_effect = [
        [{"min_start_ms": 1_786_363_200_000, "max_start_ms": 1_786_449_600_000}],
        [
            {
                "eval_id": "run_123",
                "shell_executions": 3,
                "shell_errors": 1,
                "shell_error_rate": 1 / 3,
                "shell_error_pct": 33.33,
                "recent_error_examples": [],
            }
        ],
        [],
    ]

    metrics = fetch_eval_run_shell_tool_error_analysis(
        mock_client,
        eval_id="run_123",
        end_date=date(2026, 8, 12),
    ).aggregate

    assert metrics.shell_executions == 3
    assert metrics.shell_errors == 1
    assert metrics.shell_error_rate == pytest.approx(1 / 3)
    assert mock_client.query.call_count == 3


def test_fetch_shell_tool_error_metrics_returns_empty_when_no_rows():
    mock_client = MagicMock()
    mock_client.query.return_value = []

    metrics = fetch_eval_run_shell_tool_error_analysis(mock_client, eval_id="run_123").aggregate

    assert metrics.shell_executions == 0
    assert metrics.shell_error_rate == 0.0
    mock_client.query.assert_called_once()
