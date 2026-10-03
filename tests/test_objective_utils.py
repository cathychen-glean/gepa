"""Contracts of the shared util layers every objective util builds on."""

from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import date
from typing import Any
from unittest.mock import MagicMock
import pytest
from glean_gepa.objectives.utils import agentspan, core, traces
from glean_gepa.objectives.utils.agentspan_query import EXECUTE_ACTION_FILTER, QueryParameter
from datetime import date, datetime, timedelta, timezone
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


@dataclass(frozen=True)
class _Entry:
    entry_id: str
    value: int
    action_inputs: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.value == 0

    @property
    def score(self) -> float:
        return 1.0 if self.passed else 0.5


def _agg(per_entry: Any, dropped: int = 0) -> dict[str, Any]:
    return {"n": len(per_entry), "dropped": dropped}


# --- core -------------------------------------------------------------------


def test_build_analysis_frame_and_derived_properties():
    per_entry = {"b": _Entry("b", 1), "a": _Entry("a", 0), "c": _Entry("c", 2)}
    analysis = core.build_analysis(
        eval_ids=("t", "s"),
        per_entry=per_entry,
        aggregate=_agg,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 2),
        paired=True,
    )
    assert isinstance(analysis, core.PairedRunAnalysis)
    assert (analysis.teacher_eval_id, analysis.student_eval_id, analysis.eval_id) == ("t", "s", "s")
    assert analysis.high_signal_entry_ids == ("b", "c")  # default = not passed, sorted
    assert (analysis.compared_entries, analysis.passed_entries) == (3, 1)
    assert analysis.pass_rate == pytest.approx(1 / 3)
    assert analysis.aggregate == {"n": 3, "dropped": 0}


def test_build_analysis_honours_custom_high_signal_predicate():
    analysis = core.build_analysis(
        eval_ids=("s",),
        per_entry={"a": _Entry("a", 0), "b": _Entry("b", 5)},
        aggregate=lambda pe: None,
        is_high_signal=lambda m: m.value > 3,
    )
    assert analysis.high_signal_entry_ids == ("b",)
    assert not isinstance(analysis, core.PairedRunAnalysis)


def test_parse_rows_skips_none_and_keeps_last_per_entry():
    rows = [{"id": "a", "v": 1}, {"id": "", "v": 9}, {"id": "a", "v": 2}]
    parsed = core.parse_rows(rows, lambda r: _Entry(r["id"], r["v"]) if r["id"] else None)
    assert parsed == {"a": _Entry("a", 2)}


def test_log_analysis_caps_entries_at_evidence_limit(capsys):
    per_entry = {f"e{i}": _Entry(f"e{i}", 1) for i in range(core.EVIDENCE_LIMIT + 3)}
    analysis = core.build_analysis(eval_ids=("run",), per_entry=per_entry, aggregate=lambda p: None)
    core.log_analysis(analysis, label="X", headline="hello", entry_line=lambda m: f"v={m.value}")
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "[X] run: hello"
    assert len(out) == 1 + core.EVIDENCE_LIMIT
    assert all(line.startswith("[X] High-signal entry=e") and line.endswith("v=1") for line in out[1:])


# --- agentspan --------------------------------------------------------------


def test_paired_role_query_emits_role_columns_and_join():
    sql = agentspan.paired_role_query(
        per_role_cte="per_role AS (SELECT 'e' AS entry_id, 's' AS eval_id, ['x'] AS sig, "
        "'t' AS trace_id, 'd' AS deployment_id, 1 AS min_start_ms, 2 AS max_start_ms)",
        signal_column="sig",
        extra_select="  TRUE AS flag",
    )
    assert "FULL OUTER JOIN teacher" in sql
    assert "IFNULL(student.sig, ARRAY<STRING>[]) AS student_sig" in sql
    assert "IFNULL(teacher.sig, ARRAY<STRING>[]) AS teacher_sig" in sql
    for role in ("student", "teacher"):
        for col in agentspan.LOCATOR_COLUMNS:
            assert f"{role}.{col} AS {role}_{col}" in sql
    assert "TRUE AS flag," in sql
    assert "WHERE eval_id = @student_eval_id" in sql and "WHERE eval_id = @teacher_eval_id" in sql


def _client(bounds_row: dict[str, Any] | None, rows: list[dict[str, Any]]) -> MagicMock:
    client = MagicMock()
    client.query.side_effect = [[bounds_row] if bounds_row else [], rows]
    return client


def test_fetch_agentspan_analysis_runs_every_hook_in_order():
    ms = int(date(2026, 8, 10).strftime("%s")) * 1000
    rows = [
        {"entry_id": "a", "value": 0, "drop": False},
        {"entry_id": "b", "value": 3, "drop": False},
        {"entry_id": "c", "value": 1, "drop": True},
    ]
    calls: list[str] = []

    def filter_rows(rs):
        calls.append("filter")
        return [r for r in rs if not r["drop"]]

    def parse(r):
        calls.append("parse")
        return _Entry(r["entry_id"], r["value"])

    def post_parse(pe):
        calls.append("post_parse")
        return {**pe, "z": _Entry("z", 9)}  # overlay may add entries

    def enrich(pe, rs, hs):
        calls.append(f"enrich:{','.join(hs)}")
        return {k: replace(v, action_inputs=("cmd",)) if k in hs else v for k, v in pe.items()}

    analysis = agentspan.fetch_agentspan_analysis(
        _client({"min_start_ms": ms, "max_start_ms": ms}, rows),
        eval_ids=("s",),
        bounds_sql="B",
        per_entry_sql="P",
        parse_row=parse,
        aggregate=_agg,
        filter_rows=filter_rows,
        post_parse=post_parse,
        enrich=enrich,
        extra_params=[QueryParameter("k", "STRING", "v")],
        end_date=date(2026, 8, 10),
    )
    assert calls == ["filter", "parse", "parse", "post_parse", "enrich:b,z"]
    assert analysis.aggregate == {"n": 3, "dropped": 1}
    assert analysis.high_signal_entry_ids == ("b", "z")
    assert analysis.per_entry["b"].action_inputs == ("cmd",)
    assert analysis.per_entry["a"].action_inputs == ()
    assert analysis.start_date == date(2026, 8, 10)


def test_fetch_agentspan_analysis_empty_window_returns_pending_frame():
    client = _client(None, [])
    analysis = agentspan.fetch_agentspan_analysis(
        client, eval_ids=("t", "s"), bounds_sql="B", per_entry_sql="P", parse_row=lambda r: None, aggregate=_agg
    )
    assert isinstance(analysis, core.PairedRunAnalysis)
    assert analysis.compared_entries == 0 and analysis.start_date is None
    client.query.assert_called_once()  # no per-entry round trip


def test_fetch_agentspan_analysis_extra_params_only_on_per_entry_query():
    ms = int(date(2026, 8, 10).strftime("%s")) * 1000
    client = _client({"min_start_ms": ms, "max_start_ms": ms}, [])
    agentspan.fetch_agentspan_analysis(
        client,
        eval_ids=("s",),
        bounds_sql="B",
        per_entry_sql="P",
        parse_row=lambda r: None,
        aggregate=_agg,
        extra_params=[QueryParameter("k", "STRING", "v")],
        end_date=date(2026, 8, 10),
    )
    bounds_names = {p.name for p in client.query.call_args_list[0].kwargs["params"]}
    per_entry_names = {p.name for p in client.query.call_args_list[1].kwargs["params"]}
    assert bounds_names == {"eval_id", "search_start_date", "search_end_date"}
    assert per_entry_names == {"eval_id", "start_date", "end_date", "k"}


# --- traces -----------------------------------------------------------------


def test_enrich_action_inputs_single_role_applies_only_where_fetched(monkeypatch):
    rows = [
        {"entry_id": "a", "trace_id": "ta", "deployment_id": "d", "min_start_ms": 1, "max_start_ms": 2},
        {"entry_id": "b", "trace_id": "tb", "deployment_id": "d", "min_start_ms": 1, "max_start_ms": 2},
    ]
    per_entry = {"a": _Entry("a", 1), "b": _Entry("b", 1), "c": _Entry("c", 0)}
    monkeypatch.setattr(traces, "fetch_action_inputs_by_entry", lambda *a, **k: {"a": ("ls",)})
    seen: list[str | None] = []

    def fake_locators(rs, *, entry_ids, role):
        seen.append(role)
        return [r for r in rs if r["entry_id"] in entry_ids]

    monkeypatch.setattr(traces, "trace_locators_for_rows", fake_locators)

    def apply(m, fetched, entry_id):
        got = fetched.get("student", {}).get(entry_id)
        return replace(m, action_inputs=got) if got else m

    out = traces.enrich_action_inputs(MagicMock(), per_entry, rows, ("a", "b"), apply=apply)
    assert seen == [None]  # single role -> unprefixed locator columns
    assert out["a"].action_inputs == ("ls",)
    assert out["b"].action_inputs == () and out["c"].action_inputs == ()


def test_enrich_action_inputs_paired_roles_and_first_tool(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_first(evalcli, locators, *, skip_tools, role_label):
        calls.append(("first", role_label))
        return {"a": (f"{role_label}_tool", "{}")}

    monkeypatch.setattr(traces, "fetch_first_tool_inputs_by_entry", fake_first)
    monkeypatch.setattr(traces, "trace_locators_for_rows", lambda rs, *, entry_ids, role: [object()])
    fetched_by_role: dict[str, Any] = {}

    def apply(m, fetched, entry_id):
        fetched_by_role.update(fetched)
        return m

    traces.enrich_action_inputs(
        MagicMock(), {"a": _Entry("a", 1)}, [], ("a",), apply=apply, roles=("student", "teacher"), first_tool_only=True
    )
    assert calls == [("first", "student"), ("first", "teacher")]
    assert fetched_by_role == {"student": {"a": ("student_tool", "{}")}, "teacher": {"a": ("teacher_tool", "{}")}}


def test_enrich_action_inputs_no_high_signal_or_no_results_is_identity(monkeypatch):
    per_entry = {"a": _Entry("a", 1)}
    assert traces.enrich_action_inputs(MagicMock(), per_entry, [], (), apply=lambda m, f, e: m) == per_entry
    monkeypatch.setattr(traces, "trace_locators_for_rows", lambda *a, **k: [])
    assert traces.enrich_action_inputs(MagicMock(), per_entry, [], ("a",), apply=lambda m, f, e: m) == per_entry


def test_high_signal_source_entries_query_maps_runtime_entry_uuids_to_source_entries():
    sql = build_high_signal_source_entries_query()

    assert "entry_uuid IN UNNEST(@entry_uuids)" in sql
    assert "WHERE eval_id = @eval_run_id" in sql
    assert "JOIN `scio-apps.fact.evalset_entries` AS evalset_entries USING (stt)" in sql


def test_fetch_high_signal_evalset_entries_resolves_source_trace_by_entry_id():
    client = MagicMock()
    client.query.side_effect = [
        [
            {
                "id": "entry-1",
                "deploymentId": "scio-prod",
                "stt": "source-stt",
                "runId": "source-run",
                "source_date": date(2026, 8, 14),
            }
        ],
        [
            {
                "id": "entry-1",
                "deploymentId": "scio-prod",
                "stt": "source-stt",
                "runId": "source-run",
                "traceId": "source-trace",
            }
        ],
    ]

    entries = fetch_high_signal_evalset_entries(
        client,
        eval_set_name="Glean Chat V2 Medium",
        eval_set_version="20260820",
        eval_run_id="parent-eval-run",
        entry_ids=["entry-1"],
        deployment_ids=["scio-prod"],
    )

    assert entries == [
        {
            "id": "entry-1",
            "deploymentId": "scio-prod",
            "stt": "source-stt",
            "runId": "source-run",
            "traceId": "source-trace",
        }
    ]
    source_params = client.query.call_args_list[0].kwargs["params"]
    assert next(param.value for param in source_params if param.name == "entry_uuids") == ["entry-1"]
    assert next(param.value for param in source_params if param.name == "eval_run_id") == "parent-eval-run"


def _ms(*args) -> int:
    return int(datetime(*args, tzinfo=timezone.utc).timestamp() * 1000)


@pytest.mark.parametrize(
    ("start_ms", "today", "end_date", "expected"),
    [
        # Shards are UTC days; a run just after UTC midnight lands on that UTC day, not the local one.
        (_ms(2026, 8, 30, 0, 30), date(2026, 8, 31), date(2026, 8, 31), (date(2026, 8, 30), date(2026, 8, 30))),
        # The next shard is padded when "today" (UTC) is still behind the run's UTC day ...
        (_ms(2026, 9, 3, 2, 8), date(2026, 9, 2), None, (date(2026, 9, 3), date(2026, 9, 3))),
        # ... and an explicit end_date must not clamp that padded shard away.
        (_ms(2026, 9, 3, 0, 30), date(2026, 9, 2), date(2026, 9, 2), (date(2026, 9, 3), date(2026, 9, 3))),
    ],
    ids=["utc_shard", "pads_next_shard", "end_date_does_not_clamp_pad"],
)
def test_resolve_eval_run_date_range_uses_padded_utc_shards(start_ms, today, end_date, expected):
    with patch("glean_gepa.objectives.utils.agentspan_query.utc_today", return_value=today):
        assert default_date_range(lookback_days=7) == (today - timedelta(days=7), today + timedelta(days=1))
        kwargs = {"end_date": end_date} if end_date else {}
        resolved = resolve_eval_run_date_range({"min_start_ms": start_ms, "max_start_ms": start_ms}, lookback_days=7, **kwargs)
    assert resolved == expected


def test_build_eval_entry_uuid_tracking_query_filters_entry_uuid():
    sql = build_eval_entry_uuid_tracking_query()

    assert "entry_uuid IN UNNEST(@entry_ids)" in sql
    assert "session_tracking_token" in sql
    assert "scrubbed_agentspan" in sql
    assert "PARSE_DATE" not in sql
    assert "_TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', @start_date)" in sql


def test_fetch_evalset_entry_tracking_uses_agentspan_stt():
    mock_client = MagicMock()
    mock_client.query.side_effect = [
        [
            {
                "id": "e1",
                "deploymentId": "prod",
                "stt": "stt-1",
                "runId": "eval-run",
                "traceId": "eval-trace",
            }
        ],
        [],
    ]

    tracking = fetch_evalset_entry_tracking(
        mock_client,
        eval_set_name="Glean Chat V2 Medium",
        eval_set_version="20260817",
        entry_ids=["e1"],
        deployment_ids=["prod"],
    )

    assert tracking == {
        "e1": {
            "deploymentId": "prod",
            "stt": "stt-1",
        }
    }
    assert mock_client.query.call_count == 2
    first_sql = mock_client.query.call_args_list[0].args[0]
    assert "entry_uuid IN UNNEST(@entry_ids)" in first_sql


def test_fetch_evalset_entry_tracking_prefers_original_run_id_from_fact():
    mock_client = MagicMock()
    mock_client.query.side_effect = [
        [
            {
                "id": "e1",
                "deploymentId": "prod",
                "stt": "stt-1",
                "runId": "eval-run",
                "traceId": "eval-trace",
            }
        ],
        [
            {
                "id": "bq-id",
                "deploymentId": "prod",
                "stt": "stt-1",
                "runId": "orig-run",
            }
        ],
    ]

    tracking = fetch_evalset_entry_tracking(
        mock_client,
        eval_set_name="Glean Chat V2 Medium",
        eval_set_version="20260817",
        entry_ids=["e1"],
        deployment_ids=["prod"],
    )

    assert tracking["e1"]["runId"] == "orig-run"
    assert tracking["e1"]["stt"] == "stt-1"
    assert "traceId" not in tracking["e1"]


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
