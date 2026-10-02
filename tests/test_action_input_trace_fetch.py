"""Parallel ``analyze trace`` fetching: order, cap, failures, progress, worker count."""

from __future__ import annotations

import threading
import time

import pytest

from glean_gepa.objectives.utils import action_input_trace as ait
from glean_gepa.objectives.utils.action_input_trace import (
    DEFAULT_TRACE_FETCH_WORKERS,
    INTERNAL_TRACE_DEPLOYMENT_ID,
    TraceActionInputLocator,
    trace_fetch_workers,
)


def _loc(entry_id: str, deployment: str = INTERNAL_TRACE_DEPLOYMENT_ID) -> TraceActionInputLocator:
    return TraceActionInputLocator(
        entry_id=entry_id, trace_id=f"t-{entry_id}", deployment_id=deployment, start_ms=0, end_ms=1
    )


class _Evalcli:
    """Fake client: records peak concurrency; ``delay_by`` makes early entries finish last."""

    def __init__(self, *, fail: frozenset[str] = frozenset(), delay_by: dict[str, float] | None = None):
        self.fail, self.delay_by = fail, delay_by or {}
        self.calls: list[str] = []
        self.active = self.peak_active = 0
        self._lock = threading.Lock()

    def get_analysis_trace(self, *, deployment_id, trace_id, start_time_millis, end_time_millis):
        entry_id = trace_id.removeprefix("t-")
        with self._lock:
            self.calls.append(entry_id)
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        try:
            time.sleep(self.delay_by.get(entry_id, 0.0))
            if entry_id in self.fail:
                raise RuntimeError(f"boom {entry_id}")
            return {"trace": entry_id}
        finally:
            with self._lock:
                self.active -= 1


def _fetch(client, locators, **kw):
    return [e for e, _ in ait._iter_entry_traces(client, locators, max_fetches=kw.pop("max_fetches", 60), **kw)]


def test_parallel_fetch_keeps_locator_order_skips_failures_and_logs_progress(monkeypatch, capsys):
    monkeypatch.setenv("GLEAN_GEPA_TRACE_FETCH_WORKERS", "4")
    client = _Evalcli(fail=frozenset({"c"}), delay_by={"a": 0.15, "b": 0.10, "c": 0.05})
    assert _fetch(client, [_loc(x) for x in "abcd"], role_label="student") == ["a", "b", "d"]
    assert client.peak_active > 1
    out = capsys.readouterr().out
    assert "Fetching 4 student traces with 4 worker(s)" in out
    assert "Failed to fetch student trace for entry c" in out
    assert "student traces 4/4 fetched" in out


def test_serial_cap_and_external_skip(monkeypatch, capsys):
    monkeypatch.setenv("GLEAN_GEPA_TRACE_FETCH_WORKERS", "1")
    client = _Evalcli()
    locators = [_loc("a"), _loc("x", deployment="guild"), _loc("b"), _loc("c")]
    assert _fetch(client, locators, max_fetches=2, role_label="") == ["a", "b"]
    assert client.calls == ["a", "b"] and client.peak_active == 1
    assert "Skipping 1 traces on non-scio-prod" in capsys.readouterr().out
    assert _fetch(_Evalcli(), [_loc("x", deployment="guild")], role_label="") == []
    assert _fetch(object(), [_loc("a")], role_label="") == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, DEFAULT_TRACE_FETCH_WORKERS), ("", DEFAULT_TRACE_FETCH_WORKERS), ("3", 3), ("0", 1), ("x", DEFAULT_TRACE_FETCH_WORKERS)],
)
def test_trace_fetch_workers_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("GLEAN_GEPA_TRACE_FETCH_WORKERS", raising=False)
    else:
        monkeypatch.setenv("GLEAN_GEPA_TRACE_FETCH_WORKERS", raw)
    assert trace_fetch_workers() == expected
