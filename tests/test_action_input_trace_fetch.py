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
    """Fake client: records concurrency, finishes later entries first."""

    def __init__(self, *, fail: frozenset[str] = frozenset(), delay_by: dict[str, float] | None = None):
        self.fail = fail
        self.delay_by = delay_by or {}
        self.calls: list[str] = []
        self.active = 0
        self.peak_active = 0
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


def test_fetches_run_in_parallel_and_yield_in_locator_order(monkeypatch):
    monkeypatch.setenv("GLEAN_GEPA_TRACE_FETCH_WORKERS", "4")
    # First entry is slowest, so completion order is reversed from input order.
    client = _Evalcli(delay_by={"a": 0.15, "b": 0.10, "c": 0.05, "d": 0.0})
    out = list(ait._iter_entry_traces(client, [_loc(x) for x in "abcd"], max_fetches=60, role_label="student"))
    assert [e for e, _ in out] == ["a", "b", "c", "d"]
    assert [t["trace"] for _, t in out] == ["a", "b", "c", "d"]
    assert client.peak_active > 1  # actually concurrent


def test_serial_when_one_worker(monkeypatch):
    monkeypatch.setenv("GLEAN_GEPA_TRACE_FETCH_WORKERS", "1")
    client = _Evalcli(delay_by={x: 0.01 for x in "abc"})
    out = list(ait._iter_entry_traces(client, [_loc(x) for x in "abc"], max_fetches=60, role_label=""))
    assert [e for e, _ in out] == ["a", "b", "c"]
    assert client.peak_active == 1
    assert client.calls == ["a", "b", "c"]


def test_failed_fetch_is_skipped_not_fatal(monkeypatch, capsys):
    monkeypatch.setenv("GLEAN_GEPA_TRACE_FETCH_WORKERS", "3")
    client = _Evalcli(fail=frozenset({"b"}))
    out = list(ait._iter_entry_traces(client, [_loc(x) for x in "abc"], max_fetches=60, role_label="teacher"))
    assert [e for e, _ in out] == ["a", "c"]
    assert "Failed to fetch teacher trace for entry b" in capsys.readouterr().out


def test_max_fetches_caps_internal_locators_and_skips_external(monkeypatch, capsys):
    monkeypatch.setenv("GLEAN_GEPA_TRACE_FETCH_WORKERS", "8")
    locators = [_loc("a"), _loc("x", deployment="guild"), _loc("b"), _loc("c"), _loc("d")]
    client = _Evalcli()
    out = list(ait._iter_entry_traces(client, locators, max_fetches=2, role_label=""))
    assert [e for e, _ in out] == ["a", "b"]
    assert sorted(client.calls) == ["a", "b"]  # cap counts only fetched internal traces
    assert "Skipping 1 traces on non-scio-prod" in capsys.readouterr().out


def test_progress_lines_report_start_and_completion(capsys):
    client = _Evalcli()
    list(ait._iter_entry_traces(client, [_loc(x) for x in "abc"], max_fetches=60, role_label="student"))
    out = capsys.readouterr().out
    assert "Fetching 3 student traces with" in out
    assert "student traces 3/3 fetched" in out


def test_no_fetchable_locators_prints_nothing_and_yields_nothing(capsys):
    client = _Evalcli()
    assert list(ait._iter_entry_traces(client, [_loc("x", deployment="guild")], max_fetches=60, role_label="")) == []
    assert "Fetching" not in capsys.readouterr().out
    assert client.calls == []


def test_client_without_get_analysis_trace_is_a_noop():
    assert list(ait._iter_entry_traces(object(), [_loc("a")], max_fetches=60, role_label="")) == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, DEFAULT_TRACE_FETCH_WORKERS),
        ("", DEFAULT_TRACE_FETCH_WORKERS),
        ("3", 3),
        ("0", 1),
        ("-2", 1),
        ("lots", DEFAULT_TRACE_FETCH_WORKERS),
    ],
)
def test_trace_fetch_workers_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("GLEAN_GEPA_TRACE_FETCH_WORKERS", raising=False)
    else:
        monkeypatch.setenv("GLEAN_GEPA_TRACE_FETCH_WORKERS", raw)
    assert trace_fetch_workers() == expected
