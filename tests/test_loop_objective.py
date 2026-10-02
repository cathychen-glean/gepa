"""``LoopEfficiencyObjective`` maps loop-count telemetry onto the objective contract."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from glean_gepa.al_adapter import ALRunner, Thresholds
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.experiment_config import experiment_objective_spec, load_experiment_config
from glean_gepa.focused_evalset import QUERY_CANONICAL_BUCKET_TYPE
from glean_gepa.objectives import AnalysisRequest, build_objective, registry
from glean_gepa.objectives.loop import (
    LOOP_EFFICIENCY_OBJECTIVE,
    EvalRunLoopCountAnalysis,
    LoopCountEntryMetrics,
    LoopEfficiencyObjective,
    aggregate_loop_count_metrics,
)
from glean_gepa.objectives.protocol import ObjectiveProtocol, check_objective_contract, scored_rows_are_normalized
from glean_gepa.single_model_adapter import SingleModelAdapter


def _analysis(per_entry: dict[str, LoopCountEntryMetrics]) -> EvalRunLoopCountAnalysis:
    return EvalRunLoopCountAnalysis(
        eval_ids=("run",),
        start_date=date(2026, 8, 1),
        end_date=date(2026, 8, 2),
        aggregate=aggregate_loop_count_metrics(per_entry),
        per_entry=per_entry,
        high_signal_entry_ids=tuple(sorted(e for e, m in per_entry.items() if m.loop_efficiency < 1.0)),
    )


def test_satisfies_contract_and_is_registered() -> None:
    assert check_objective_contract(LoopEfficiencyObjective) == []
    assert isinstance(LoopEfficiencyObjective(bigquery_client=MagicMock()), ObjectiveProtocol)
    assert registry.resolve("single_model", "loop_telemetry") is LoopEfficiencyObjective


def test_loop_telemetry_signal_constructs_the_objective(tmp_path):
    mode = tmp_path / "mode.yaml"
    mode.write_text(
        "schema_version: 1\n"
        "mode: single_model\n"
        f"signals:\n  - name: {LOOP_EFFICIENCY_OBJECTIVE}\n    source: loop_telemetry\n"
        f"objective:\n  primary: {LOOP_EFFICIENCY_OBJECTIVE}\n  composite:\n    {LOOP_EFFICIENCY_OBJECTIVE}: 1.0\n"
        f"  focused_bucket_type: {QUERY_CANONICAL_BUCKET_TYPE}\n"
    )
    config = load_experiment_config(mode)
    objective = build_objective(
        "single_model",
        config.signals,
        bigquery_client=MagicMock(),
        experiment=experiment_objective_spec(config),
    )
    assert config.primary_objective == LOOP_EFFICIENCY_OBJECTIVE
    assert isinstance(objective, LoopEfficiencyObjective)
    assert objective.focused_bucket_type == QUERY_CANONICAL_BUCKET_TYPE
    assert "reduce agent loops" in objective.reflection_prompt("WRITING_CODE")


def test_scores_extra_loops_below_one_and_builds_reflection():
    objective = LoopEfficiencyObjective(bigquery_client=MagicMock())
    per_entry = {
        "ok": LoopCountEntryMetrics("ok", 2),
        "slow": LoopCountEntryMetrics("slow", 4, action_inputs=tuple(f"cmd{i}" for i in range(6))),
        "direct": LoopCountEntryMetrics("direct", 0),
    }
    analysis = _analysis(per_entry)
    assert objective.is_pending(analysis) is False
    assert objective.aggregate_score(analysis) == pytest.approx((1 / 3 + 1 / 5 + 1.0) / 3)

    rows = objective.scored_rows(
        analysis,
        al_data_inst={},
        student_eval_id="run",
        eval_set_name="set",
        eval_set_version="v1",
        deployment_ids=["dep"],
        requested_entry_ids=["ok", "slow", "direct"],
        is_focused_eval=True,
        capture_traces=True,
    )
    assert scored_rows_are_normalized(rows)
    by_entry = {row.entry_id: row for row in rows}
    assert by_entry["ok"].dimension_scores[LOOP_EFFICIENCY_OBJECTIVE] == pytest.approx(1 / 3)
    assert by_entry["slow"].dimension_scores[LOOP_EFFICIENCY_OBJECTIVE] == pytest.approx(1 / 5)
    assert by_entry["direct"].dimension_scores[LOOP_EFFICIENCY_OBJECTIVE] == 1.0
    assert by_entry["slow"].output["student_loops"] == 4
    assert objective.focused_pass_rate(analysis, ["ok", "slow", "direct"]) == pytest.approx((1 / 3 + 1 / 5 + 1.0) / 3)

    trajectory = {
        "data": {"eval_set_name": "set", "eval_run_id": "run"},
        "score": 1 / 5,
        "objective_scores": {LOOP_EFFICIENCY_OBJECTIVE: 1 / 5},
        "output": by_entry["slow"].output,
    }
    assert objective.failure_pattern("WRITING_CODE", trajectory) == (4,)
    example = objective.build_reflective_example("WRITING_CODE", trajectory, {})
    assert example["Action Inputs"] == [f"cmd{i}" for i in range(5)]
    assert "Used 4 loops" in example["Feedback"]
    assert example["Inputs"]["eval_run_id"] == "run"


def test_full_eval_without_traces_returns_one_aggregate_row():
    objective = LoopEfficiencyObjective(bigquery_client=MagicMock())
    analysis = _analysis({"a": LoopCountEntryMetrics("a", 3)})
    rows = objective.scored_rows(
        analysis,
        al_data_inst={},
        student_eval_id="run",
        eval_set_name="set",
        eval_set_version="v1",
        deployment_ids=["dep"],
        requested_entry_ids=None,
        is_focused_eval=False,
        capture_traces=False,
    )
    assert len(rows) == 1
    assert rows[0].entry_id is None
    assert rows[0].dimension_scores[LOOP_EFFICIENCY_OBJECTIVE] == 0.25


@pytest.mark.parametrize(
    ("telemetry", "requested", "expected"),
    [
        (["a", "b", "c"], None, "high_signal"),  # full eval: the high-signal set
        (["b", "a"], ["a", "b"], ("a", "b")),  # focused: request order
        (["a"], ["a", "b"], ("a",)),  # telemetry lag: subset, never fabricated
        (["a", "b", "c"], ["a", "b"], ("a", "b")),  # extra telemetry: never a superset
        ([], ["a", "b"], ()),  # nothing landed: adapter's ``is_pending`` handles it
    ],
)
def test_entry_ids_to_score_never_exceeds_the_request(telemetry, requested, expected):
    # Every entry uses 3 loops, so all are high-signal.
    analysis = _analysis({e: LoopCountEntryMetrics(e, 3) for e in telemetry})
    if expected == "high_signal":
        expected = analysis.high_signal_entry_ids
    assert _objective_ids(analysis, requested) == expected


def _objective_ids(analysis, requested):
    return LoopEfficiencyObjective(bigquery_client=MagicMock()).entry_ids_to_score(analysis, requested)


def test_scored_rows_and_pass_rate_agree_on_a_partial_focused_batch():
    """A requested entry with no telemetry gets no row and scores 0."""
    objective = LoopEfficiencyObjective(bigquery_client=MagicMock())
    analysis = _analysis({"ok": LoopCountEntryMetrics("ok", 1)})
    requested = ["ok", "missing"]
    rows = objective.scored_rows(
        analysis,
        al_data_inst={},
        student_eval_id="run",
        eval_set_name="set",
        eval_set_version="v1",
        deployment_ids=["dep"],
        requested_entry_ids=requested,
        is_focused_eval=True,
        capture_traces=True,
    )
    assert [row.entry_id for row in rows] == ["ok"]
    assert objective.focused_pass_rate(analysis, requested) == 0.25


def test_analyze_caches_only_when_telemetry_landed():
    objective = LoopEfficiencyObjective(bigquery_client=MagicMock())
    empty = _analysis({})
    full = _analysis({"a": LoopCountEntryMetrics("a", 1)})
    request = AnalysisRequest()
    with patch("glean_gepa.objectives.loop.fetch_eval_run_loop_count_analysis", side_effect=[empty, full]):
        assert objective.analyze("run", request=request) is empty
        assert objective.is_pending(empty)
        assert objective.analyze("run", request=request) is full
        assert objective.analyze("run", request=request) is full  # cache hit, no third fetch


def test_validation_fetch_then_trace_fetch_rehydrates_action_inputs():
    """An entry cached without payloads must not be served to a reflection-bound request."""
    objective = LoopEfficiencyObjective(bigquery_client=MagicMock())
    bare = _analysis({"a": LoopCountEntryMetrics("a", 3)})
    hydrated = _analysis({"a": LoopCountEntryMetrics("a", 3, action_inputs=("ls",))})
    with patch("glean_gepa.objectives.loop.fetch_eval_run_loop_count_analysis", side_effect=[bare, hydrated]) as fetch:
        first = objective.analyze("run", request=AnalysisRequest(hydrate_action_inputs=False))
        second = objective.analyze("run", request=AnalysisRequest(hydrate_action_inputs=True))
        third = objective.analyze("run", request=AnalysisRequest(hydrate_action_inputs=False))
    assert first is bare
    assert second is hydrated
    assert third is hydrated  # hydrated satisfies both kinds of request
    assert fetch.call_count == 2
    assert [c.kwargs["include_action_inputs"] for c in fetch.call_args_list] == [False, True]
    assert "run" not in objective._unhydrated_eval_ids


def test_trace_fetch_then_validation_fetch_is_one_call():
    objective = LoopEfficiencyObjective(bigquery_client=MagicMock())
    hydrated = _analysis({"a": LoopCountEntryMetrics("a", 3, action_inputs=("ls",))})
    with patch("glean_gepa.objectives.loop.fetch_eval_run_loop_count_analysis", return_value=hydrated) as fetch:
        objective.analyze("run", request=AnalysisRequest(hydrate_action_inputs=True))
        objective.analyze("run", request=AnalysisRequest(hydrate_action_inputs=False))
    fetch.assert_called_once()


def test_empty_rehydration_keeps_the_unhydrated_entry():
    objective = LoopEfficiencyObjective(bigquery_client=MagicMock())
    bare = _analysis({"a": LoopCountEntryMetrics("a", 3)})
    empty = _analysis({})
    with patch("glean_gepa.objectives.loop.fetch_eval_run_loop_count_analysis", side_effect=[bare, empty]):
        objective.analyze("run", request=AnalysisRequest(hydrate_action_inputs=False))
        got = objective.analyze("run", request=AnalysisRequest(hydrate_action_inputs=True))
    assert got is bare
    assert objective._eval_analysis_cache["run"] is bare
    assert "run" in objective._unhydrated_eval_ids  # still owed a hydrating fetch


def test_hydration_state_survives_cache_round_trip():
    """Resume after a validation-only run must still refetch payloads for reflection."""
    source = LoopEfficiencyObjective(bigquery_client=MagicMock())
    bare = _analysis({"a": LoopCountEntryMetrics("a", 3)})
    with patch("glean_gepa.objectives.loop.fetch_eval_run_loop_count_analysis", return_value=bare):
        source.analyze("run", request=AnalysisRequest(hydrate_action_inputs=False))
    payload = source.cache_payload()
    assert payload["run"]["action_inputs_hydrated"] is False

    resumed = LoopEfficiencyObjective(bigquery_client=MagicMock())
    resumed.load_cache(payload)
    assert "run" in resumed._unhydrated_eval_ids
    hydrated = _analysis({"a": LoopCountEntryMetrics("a", 3, action_inputs=("ls",))})
    with patch("glean_gepa.objectives.loop.fetch_eval_run_loop_count_analysis", return_value=hydrated) as fetch:
        got = resumed.analyze("run", request=AnalysisRequest(hydrate_action_inputs=True))
    fetch.assert_called_once()
    assert got.per_entry["a"].action_inputs == ("ls",)
    assert resumed.cache_payload()["run"]["action_inputs_hydrated"] is True

    # A pre-fix payload has no flag; treat as unhydrated rather than trust it.
    del payload["run"]["action_inputs_hydrated"]
    legacy = LoopEfficiencyObjective(bigquery_client=MagicMock())
    legacy.load_cache(payload)
    assert "run" in legacy._unhydrated_eval_ids


def test_cache_rebuilds_scores_from_loop_counts():
    objective = LoopEfficiencyObjective(bigquery_client=MagicMock())
    objective._eval_analysis_cache["run"] = _analysis(
        {
            "A": LoopCountEntryMetrics("A", 1),
            "B": LoopCountEntryMetrics("B", 2),
            "C": LoopCountEntryMetrics("C", 0),
        }
    )
    payload = objective.cache_payload()

    objective.load_cache(payload)
    loaded = objective._eval_analysis_cache["run"]
    assert loaded.high_signal_entry_ids == ("A", "B")
    assert loaded.per_entry["A"].loop_efficiency == 0.5
    assert loaded.per_entry["B"].loop_efficiency == pytest.approx(1 / 3)
    assert loaded.per_entry["C"].loop_efficiency == 1.0

    objective.load_cache({"run": {"schema_version": 0}})
    assert objective._eval_analysis_cache == {}


def test_adapter_uses_query_canonical_focused_sets():
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli")),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        objective=LoopEfficiencyObjective(bigquery_client=MagicMock()),
    )
    batch = [{"eval_set_name": "set", "eval_set_version": "v1", "deployment_ids": ["dep"], "eval_entry_ids": ["e1"]}]
    with patch("glean_gepa.al_adapter.prepare_high_signal_eval_batch", return_value=batch) as prepared:
        assert adapter.prepare_high_signal_batch(batch) == batch
    assert prepared.call_args.kwargs["bucket_type"] == QUERY_CANONICAL_BUCKET_TYPE
