from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

from glean_gepa.al_adapter import ALRunner, Thresholds
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.experiment_config import load_experiment_config
from glean_gepa.focused_evalset import QUERY_CANONICAL_BUCKET_TYPE
from glean_gepa.objectives.utils.loop_count_util import (
    LOOP_EFFICIENCY_OBJECTIVE,
    EvalRunLoopCountAnalysis,
    LoopCountEntryMetrics,
    aggregate_loop_count_metrics,
    build_loop_count_per_entry_query,
    fetch_eval_run_loop_count_analysis,
    loop_efficiency_score,
    overlay_evalcli_loop_and_correctness,
    parse_loop_count_entry_metrics,
)
from glean_gepa.objectives import build_objective
from glean_gepa.objectives.loop import LoopEfficiencyObjective
from glean_gepa.single_model_adapter import SingleModelAdapter


def test_loop_efficiency_is_zero_when_incorrect_and_one_at_or_below_target():
    assert loop_efficiency_score(0, 1.0) == 1.0
    assert loop_efficiency_score(2, 1.0) == 1.0
    assert loop_efficiency_score(3, 1.0) == 0.5
    assert loop_efficiency_score(4, 1.0) == 1 / 3
    assert loop_efficiency_score(1, 0.0) == 0.0
    assert loop_efficiency_score(8, 0.4) == 0.0


def test_parse_uses_errors_as_correctness_when_judge_score_is_missing():
    ok = parse_loop_count_entry_metrics({"entry_id": "e1", "loop_count": 2, "has_error": False})
    assert ok.correctness == 1.0
    assert ok.loop_efficiency == 1.0

    failed = parse_loop_count_entry_metrics({"entry_id": "e2", "loop_count": 1, "has_error": True})
    assert failed.correctness == 0.0
    assert failed.loop_efficiency == 0.0

    judged = parse_loop_count_entry_metrics({"entry_id": "e3", "loop_count": 3, "has_error": False, "correctness": 1.0})
    assert judged.loop_efficiency == 0.5


def test_loop_count_query_and_fetch():
    sql = build_loop_count_per_entry_query()
    assert "Execute Action:" in sql
    assert "GROUP BY entry_id" in sql
    assert "COUNTIF(is_loop)" in sql

    client = MagicMock()
    client.query.side_effect = [
        [{"min_start_ms": 1_786_363_200_000, "max_start_ms": 1_786_449_600_000}],
        [
            {"entry_id": "entry-1", "loop_count": 2, "has_error": False},
            {"entry_id": "entry-2", "loop_count": 5, "has_error": False},
            {"entry_id": "entry-3", "loop_count": 1, "has_error": True},
        ],
    ]
    analysis = fetch_eval_run_loop_count_analysis(
        client,
        eval_id="student",
        lookback_days=7,
        end_date=date(2026, 8, 11),
    )
    assert analysis.per_entry["entry-1"].loop_efficiency == 1.0
    assert analysis.per_entry["entry-2"].loop_efficiency == 1 / 4
    assert analysis.per_entry["entry-3"].loop_efficiency == 0.0
    assert analysis.high_signal_entry_ids == ("entry-2", "entry-3")
    assert analysis.aggregate.matching_entries == 1
    assert client.query.call_count == 2


def test_evalcli_overlay_prefers_loopcount_and_correctness_judge():
    per_entry = {
        "e1": LoopCountEntryMetrics("e1", loop_count=9, correctness=1.0, has_error=False),
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
    assert updated["e2"].loop_count == 1
    assert updated["e2"].loop_efficiency == 0.0


def test_loops_pack_constructs_the_loop_efficiency_objective(tmp_path):
    mode = tmp_path / "mode.yaml"
    mode.write_text("schema_version: 1\nmode: single_model\npacks: [loops]\n")
    config = load_experiment_config(mode)
    objective = build_objective("single_model", config.signals, bigquery_client=MagicMock())
    assert config.primary_objective == LOOP_EFFICIENCY_OBJECTIVE
    assert isinstance(objective, LoopEfficiencyObjective)
    assert objective.focused_bucket_type == QUERY_CANONICAL_BUCKET_TYPE


def test_loop_objective_scores_incorrect_or_extra_loops_below_one():
    objective = LoopEfficiencyObjective(bigquery_client=MagicMock())
    per_entry = {
        "ok": LoopCountEntryMetrics("ok", 2, 1.0, False),
        "slow": LoopCountEntryMetrics("slow", 4, 1.0, False),
        "wrong": LoopCountEntryMetrics("wrong", 1, 0.0, True),
    }
    analysis = EvalRunLoopCountAnalysis(
        eval_id="run",
        start_date=date(2026, 8, 1),
        end_date=date(2026, 8, 2),
        aggregate=aggregate_loop_count_metrics("run", per_entry),
        per_entry=per_entry,
        high_signal_entry_ids=("slow", "wrong"),
    )
    rows = objective.scored_rows(
        analysis,
        al_data_inst={},
        student_eval_id="run",
        eval_set_name="set",
        eval_set_version="v1",
        deployment_ids=["dep"],
        requested_entry_ids=["ok", "slow", "wrong"],
        is_focused_eval=True,
        capture_traces=True,
    )
    by_entry = {row.entry_id: row for row in rows}
    assert by_entry["ok"].dimension_scores[LOOP_EFFICIENCY_OBJECTIVE] == 1.0
    assert by_entry["slow"].dimension_scores[LOOP_EFFICIENCY_OBJECTIVE] == 0.0
    assert by_entry["wrong"].dimension_scores[LOOP_EFFICIENCY_OBJECTIVE] == 0.0
    assert objective.focused_pass_rate(analysis, ["ok", "slow", "wrong"]) == 1 / 3


def test_single_model_adapter_uses_query_canonical_focused_sets_for_loops():
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli")),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        objective=LoopEfficiencyObjective(bigquery_client=MagicMock()),
    )
    batch = [
        {
            "eval_set_name": "set",
            "eval_set_version": "v1",
            "deployment_ids": ["dep"],
            "status": "active",
            "eval_entry_ids": ["e1"],
        }
    ]
    with patch("glean_gepa.al_adapter.prepare_high_signal_eval_batch", return_value=batch) as prepared:
        result = adapter.prepare_high_signal_batch(batch)
    assert result == batch
    prepared.assert_called_once()
    assert prepared.call_args.kwargs["bucket_type"] == QUERY_CANONICAL_BUCKET_TYPE
