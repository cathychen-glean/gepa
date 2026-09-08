"""Timeout and retry behavior for the BigQuery wrapper.

Three overnight runs hung forever in a blocking TLS read because neither the
request nor the result wait had a deadline.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from glean_gepa.bigquery_client import (
    DEFAULT_REQUEST_TIMEOUT_SEC,
    BigQueryClient,
    BigQueryError,
)


def _row_client(rows: list[dict[str, object]]) -> MagicMock:
    client = MagicMock()
    client.query.return_value.result.return_value = [MagicMock(items=lambda r=row: r.items()) for row in rows]
    return client


def test_query_bounds_both_the_request_and_the_result_wait():
    client = _row_client([{"a": 1}])

    assert BigQueryClient(client=client).query("SELECT 1") == [{"a": 1}]

    assert client.query.call_args.kwargs["timeout"] == DEFAULT_REQUEST_TIMEOUT_SEC
    assert client.query.return_value.result.call_args.kwargs["timeout"] == DEFAULT_REQUEST_TIMEOUT_SEC


def test_query_retries_a_failed_attempt_and_succeeds(monkeypatch):
    monkeypatch.setattr("glean_gepa.bigquery_client.time.sleep", lambda _seconds: None)
    client = MagicMock()
    good = MagicMock()
    good.result.return_value = [MagicMock(items=lambda: {"a": 1}.items())]
    client.query.side_effect = [TimeoutError("read timed out"), good]

    assert BigQueryClient(client=client).query("SELECT 1") == [{"a": 1}]
    assert client.query.call_count == 2


def test_query_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr("glean_gepa.bigquery_client.time.sleep", lambda _seconds: None)
    client = MagicMock()
    client.query.side_effect = TimeoutError("read timed out")

    with pytest.raises(BigQueryError, match="failed after 3 attempt"):
        BigQueryClient(client=client).query("SELECT 1")

    assert client.query.call_count == 3


def test_retry_keeps_an_injected_client_but_drops_an_owned_one(monkeypatch):
    monkeypatch.setattr("glean_gepa.bigquery_client.time.sleep", lambda _seconds: None)
    injected = MagicMock()
    injected.query.side_effect = TimeoutError("read timed out")
    wrapper = BigQueryClient(client=injected)

    with pytest.raises(BigQueryError):
        wrapper.query("SELECT 1")
    # The caller owns this client, so retries must not throw it away.
    assert wrapper._client is injected

    # A client the wrapper built itself may hold dead sockets, so it is discarded.
    owned = BigQueryClient(project_id="p")
    owned._client = injected
    owned._discard_client()
    assert owned._client is None


def test_attempts_is_never_below_one():
    assert BigQueryClient(client=MagicMock(), attempts=0).attempts == 1


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("403 permission denied"),
        RuntimeError("Reauthentication is needed. Please run `gcloud auth application-default login`"),
    ],
)
def test_permanent_failures_are_not_retried(monkeypatch, exc):
    """Retrying an auth or permission failure only delays a message the caller
    has to act on, and burns the backoff waiting."""
    slept: list[float] = []
    monkeypatch.setattr("glean_gepa.bigquery_client.time.sleep", slept.append)
    client = MagicMock()
    client.query.side_effect = exc

    with pytest.raises(BigQueryError, match="failed after 1 attempt"):
        BigQueryClient(client=client).query("SELECT 1")

    assert client.query.call_count == 1
    assert slept == []
