"""Contracts of the shared util layers every objective util builds on."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Any
from unittest.mock import MagicMock

import pytest

from glean_gepa.objectives.utils import agentspan, core, traces
from glean_gepa.objectives.utils.agentspan_query import EXECUTE_ACTION_FILTER, QueryParameter


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


def _agg(ids: tuple[str, ...], per_entry: Any, dropped: int = 0) -> dict[str, Any]:
    return {"ids": ids, "n": len(per_entry), "dropped": dropped}


# --- core -------------------------------------------------------------------


def test_entry_protocol_is_runtime_checkable():
    assert isinstance(_Entry("a", 0), core.EntryMetricsLike)

    @dataclass(frozen=True)
    class NoScore:
        entry_id: str
        passed: bool

    assert not isinstance(NoScore("a", True), core.EntryMetricsLike)


def test_build_analysis_frame_and_derived_properties():
    per_entry = {"b": _Entry("b", 1), "a": _Entry("a", 0), "c": _Entry("c", 2)}
    analysis = core.build_analysis(
        eval_ids=("t", "s"),
        per_entry=per_entry,
        aggregate=lambda ids, pe: _agg(ids, pe),
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 2),
        paired=True,
    )
    assert isinstance(analysis, core.PairedRunAnalysis)
    assert (analysis.teacher_eval_id, analysis.student_eval_id, analysis.eval_id) == ("t", "s", "s")
    assert analysis.high_signal_entry_ids == ("b", "c")  # default = not passed, sorted
    assert (analysis.compared_entries, analysis.passed_entries) == (3, 1)
    assert analysis.pass_rate == pytest.approx(1 / 3)
    assert analysis.aggregate == {"ids": ("t", "s"), "n": 3, "dropped": 0}


def test_build_analysis_honours_custom_high_signal_predicate():
    analysis = core.build_analysis(
        eval_ids=("s",),
        per_entry={"a": _Entry("a", 0), "b": _Entry("b", 5)},
        aggregate=lambda ids, pe: None,
        is_high_signal=lambda m: m.value > 3,
    )
    assert analysis.high_signal_entry_ids == ("b",)
    assert not isinstance(analysis, core.PairedRunAnalysis)


def test_empty_analysis_is_pending_shaped():
    analysis = core.empty_analysis(eval_ids=("s",), aggregate=lambda ids, pe: _agg(ids, pe))
    assert analysis.compared_entries == 0
    assert analysis.pass_rate == 0.0
    assert analysis.high_signal_entry_ids == ()
    assert analysis.start_date is None


def test_parse_rows_skips_none_and_keeps_last_per_entry():
    rows = [{"id": "a", "v": 1}, {"id": "", "v": 9}, {"id": "a", "v": 2}]
    parsed = core.parse_rows(rows, lambda r: _Entry(r["id"], r["v"]) if r["id"] else None)
    assert parsed == {"a": _Entry("a", 2)}


def test_require_compared_entries_raises_with_hint():
    empty = core.empty_analysis(eval_ids=("t", "s"), aggregate=lambda ids, pe: None, paired=True)
    with pytest.raises(core.NoComparedEntriesError, match=r"eval\(s\) t, s\. wait for ingest"):
        core.require_compared_entries(empty, hint="wait for ingest")
    full = core.build_analysis(eval_ids=("s",), per_entry={"a": _Entry("a", 0)}, aggregate=lambda i, p: None)
    core.require_compared_entries(full, hint="unused")  # no raise


def test_log_analysis_caps_entries_at_evidence_limit(capsys):
    per_entry = {f"e{i}": _Entry(f"e{i}", 1) for i in range(core.EVIDENCE_LIMIT + 3)}
    analysis = core.build_analysis(eval_ids=("run",), per_entry=per_entry, aggregate=lambda i, p: None)
    core.log_analysis(analysis, label="X", headline="hello", entry_line=lambda m: f"v={m.value}")
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "[X] run: hello"
    assert len(out) == 1 + core.EVIDENCE_LIMIT
    assert all(line.startswith("[X] High-signal entry=e") and line.endswith("v=1") for line in out[1:])


# --- agentspan --------------------------------------------------------------


def test_bounds_query_shapes():
    single = agentspan.bounds_query(eval_id_predicate="= @eval_id")
    paired = agentspan.bounds_query(eval_id_predicate="IN UNNEST(@eval_ids)", span_filter=EXECUTE_ACTION_FILTER)
    assert "eval_id = @eval_id" in single and "AND STARTS_WITH" not in single
    assert "eval_id IN UNNEST(@eval_ids)" in paired and EXECUTE_ACTION_FILTER in paired
    for sql in (single, paired):
        assert "MIN(SAFE_CAST" in sql and "MAX(SAFE_CAST" in sql
        assert "@search_start_date" in sql and "@search_end_date" in sql


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


def test_eval_id_params_single_vs_paired():
    assert agentspan.eval_id_params(("s",), paired=False) == [QueryParameter("eval_id", "STRING", "s")]
    paired = {p.name: p.value for p in agentspan.eval_id_params(("t", "s"), paired=True)}
    assert paired == {"eval_ids": ["t", "s"], "teacher_eval_id": "t", "student_eval_id": "s"}


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
    assert analysis.aggregate == {"ids": ("s",), "n": 3, "dropped": 1}
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
