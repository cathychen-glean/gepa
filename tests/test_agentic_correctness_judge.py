"""judge_pairwise_agentic_correctness as a validation and screening gate.

It shares Cortex type AGENTIC_JUDGE with the multi-dimension agentic judge, so the
adapter keys it under its own ``judge_type`` and asks Cortex for AGENTIC_JUDGE with
the correctness skill in run_params.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from glean_gepa.al_adapter import ALRunner, Thresholds
from glean_gepa.evalcli_client import (
    AGENTIC_CORRECTNESS_JUDGE_NAME,
    AGENTIC_JUDGE_NAME,
    AGENTIC_JUDGE_TYPE,
    EvalCliClient,
    judge_run_skill_name,
)
from glean_gepa.experiment_config import (
    customer_validation_gates,
    load_experiment_config,
    pairwise_judges,
    screening_weights,
)
from glean_gepa.judge_metrics_util import (
    AGENTIC_CORRECTNESS_JUDGE_TYPE,
    CUSTOMER_AGENTIC_CORRECTNESS_METRIC,
    CUSTOMER_AGENTIC_PREFERENCE_METRIC,
    DEFAULT_CUSTOMER_VALIDATION_GATES,
    JUDGE_SPECS,
    _per_entry_scale,
    per_entry_from_analysis_view,
)
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter

WALDO_CONFIG = Path(__file__).resolve().parents[1] / "src/glean_gepa/configs/teacher_student_waldo.yaml"


def test_spec_is_registered_as_pairwise_gate():
    spec = JUDGE_SPECS[CUSTOMER_AGENTIC_CORRECTNESS_METRIC]
    assert spec.kind == "pairwise"
    assert spec.judge_type == AGENTIC_CORRECTNESS_JUDGE_TYPE
    assert spec.cortex_judge_type == AGENTIC_JUDGE_TYPE
    assert spec.judge_skill_name == AGENTIC_CORRECTNESS_JUDGE_NAME
    assert spec.per_entry_scale == 10.0
    assert spec.default_min == 0.50
    # Opt-in: it duplicates the multi-dimension judge's correctness scorer.
    assert CUSTOMER_AGENTIC_CORRECTNESS_METRIC not in DEFAULT_CUSTOMER_VALIDATION_GATES
    assert CUSTOMER_AGENTIC_PREFERENCE_METRIC in DEFAULT_CUSTOMER_VALIDATION_GATES
    params = json.loads(spec.run_params)
    assert params["judge_skill_name"] == AGENTIC_CORRECTNESS_JUDGE_NAME
    assert params["scoring_mode"] == "randomized_single_0_10"
    assert "dimension_skills" not in params


def test_distinct_adapter_key_from_multi_dimension_judge():
    pref = JUDGE_SPECS[CUSTOMER_AGENTIC_PREFERENCE_METRIC]
    corr = JUDGE_SPECS[CUSTOMER_AGENTIC_CORRECTNESS_METRIC]
    assert pref.cortex_judge_type == corr.cortex_judge_type == AGENTIC_JUDGE_TYPE
    assert pref.judge_type != corr.judge_type
    assert pref.judge_skill_name == AGENTIC_JUDGE_NAME
    assert _per_entry_scale(corr.judge_type) == 10.0


def test_per_entry_scores_normalize_on_0_10():
    view = {
        "entries": [
            {
                "entryId": "e1",
                "evalRunEntries": [{"runId": "ev", "metadata": {"judgeScores": {"j": 7}}}],
            },
            {
                "entryId": "e2",
                "evalRunEntries": [{"runId": "ev", "metadata": {"judgeScores": {"j": 5}}}],
            },
        ]
    }
    scores, _ = per_entry_from_analysis_view(
        view, eval_id="ev", judge_run_id="j", judge_type=AGENTIC_CORRECTNESS_JUDGE_TYPE
    )
    assert scores == {"e1": 0.7, "e2": 0.5}


def test_waldo_config_wires_gate_and_screen():
    config = load_experiment_config(WALDO_CONFIG)
    judges = {j.name: j for j in pairwise_judges(config)}
    judge = judges[CUSTOMER_AGENTIC_CORRECTNESS_METRIC]
    assert judge.judge_type == AGENTIC_CORRECTNESS_JUDGE_TYPE
    assert judge.cortex_judge_type == AGENTIC_JUDGE_TYPE
    assert judge.judge_skill_name == AGENTIC_CORRECTNESS_JUDGE_NAME
    assert json.loads(judge.run_params)["judge_skill_name"] == AGENTIC_CORRECTNESS_JUDGE_NAME
    # 4.8 on the judge's 0-10 scale (5 = parity) -> 0.48 normalized. Validation only.
    assert customer_validation_gates(config) == {CUSTOMER_AGENTIC_CORRECTNESS_METRIC: 0.48}
    assert screening_weights(config) == {}
    assert "floors" not in config.screening


def test_teacher_student_adapter_asks_cortex_for_agentic_judge_with_correctness_skill():
    evalcli = MagicMock()
    evalcli.find_judge_run_id.return_value = None
    evalcli.create_judge_run.return_value = "judge-corr"
    config = load_experiment_config(WALDO_CONFIG)
    adapter = TeacherStudentAdapter(
        runner=ALRunner(evalcli=evalcli),
        teacher_model="gpt",
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        pairwise_judges=pairwise_judges(config),
        screening_kind="high_signal_fix_rate",
    )
    judge = next(j for j in adapter.pairwise_judges if j.name == CUSTOMER_AGENTIC_CORRECTNESS_METRIC)
    run_id = adapter._ensure_judge(
        "student-ev",
        judge_type=judge.judge_type,
        run_params=judge.run_params,
        base_eval_run_id="teacher-ev",
        input_mappings=judge.input_mappings,
        cortex_judge_type=judge.cortex_judge_type,
        judge_skill_name=judge.judge_skill_name,
    )
    assert run_id == "judge-corr"
    find_kwargs = evalcli.find_judge_run_id.call_args.kwargs
    assert find_kwargs["judge_type"] == AGENTIC_JUDGE_TYPE
    assert find_kwargs["judge_skill_name"] == AGENTIC_CORRECTNESS_JUDGE_NAME
    create_kwargs = evalcli.create_judge_run.call_args.kwargs
    assert create_kwargs["judge_type"] == AGENTIC_JUDGE_TYPE
    assert json.loads(create_kwargs["run_params"])["judge_skill_name"] == AGENTIC_CORRECTNESS_JUDGE_NAME
    # Cached under the adapter key, not the Cortex type.
    assert adapter._judge_runs[("student-ev", "teacher-ev", AGENTIC_CORRECTNESS_JUDGE_TYPE)] == "judge-corr"
    assert ("student-ev", "teacher-ev", AGENTIC_JUDGE_TYPE) not in adapter._judge_runs


def test_al_runner_ensure_judge_run_maps_cortex_type():
    evalcli = MagicMock()
    evalcli.find_judge_run_id.return_value = None
    evalcli.create_judge_run.return_value = "judge-corr"
    runner = ALRunner(evalcli=evalcli)
    spec = JUDGE_SPECS[CUSTOMER_AGENTIC_CORRECTNESS_METRIC]
    runner.ensure_judge_run(
        eval_run_id="best",
        judge_type=spec.judge_type,
        run_params=spec.run_params,
        base_eval_run_id="base",
        input_mappings=spec.input_mappings,
        cortex_judge_type=spec.cortex_judge_type,
        judge_skill_name=spec.judge_skill_name,
    )
    assert evalcli.find_judge_run_id.call_args.kwargs["judge_type"] == AGENTIC_JUDGE_TYPE
    assert evalcli.find_judge_run_id.call_args.kwargs["judge_skill_name"] == AGENTIC_CORRECTNESS_JUDGE_NAME
    assert evalcli.create_judge_run.call_args.kwargs["judge_type"] == AGENTIC_JUDGE_TYPE
    assert runner._judge_run_ids[("best", "base", AGENTIC_CORRECTNESS_JUDGE_TYPE)] == "judge-corr"


def _row(run_id: str, skill: str, *, as_json: bool = False) -> dict:
    params = {"judge_skill_name": skill}
    return {
        "id": run_id,
        "evalRunId": "ev",
        "baseEvalRunId": "base",
        "config": {"judgeType": AGENTIC_JUDGE_TYPE, "runParameters": json.dumps(params) if as_json else params},
    }


@pytest.mark.parametrize("as_json", [False, True], ids=["dict", "json-string"])
def test_judge_run_skill_name_reads_config_run_parameters(as_json: bool):
    assert judge_run_skill_name(_row("j", AGENTIC_CORRECTNESS_JUDGE_NAME, as_json=as_json)) == (
        AGENTIC_CORRECTNESS_JUDGE_NAME
    )
    assert judge_run_skill_name({"id": "j"}) is None


def test_find_judge_run_id_filters_by_skill_when_cortex_type_is_shared():
    client = EvalCliClient(binary="/fake/evalcli")
    listing = {
        "judgeRuns": [
            _row("judge-multi", AGENTIC_JUDGE_NAME),
            _row("judge-corr", AGENTIC_CORRECTNESS_JUDGE_NAME),
        ]
    }
    with patch.object(client, "_invoke_json", return_value=listing):
        corr = client.find_judge_run_id(
            "ev",
            judge_type=AGENTIC_JUDGE_TYPE,
            base_eval_run_id="base",
            judge_skill_name=AGENTIC_CORRECTNESS_JUDGE_NAME,
        )
        multi = client.find_judge_run_id(
            "ev", judge_type=AGENTIC_JUDGE_TYPE, base_eval_run_id="base", judge_skill_name=AGENTIC_JUDGE_NAME
        )
        # No skill filter keeps the old behavior: first type match wins.
        unfiltered = client.find_judge_run_id("ev", judge_type=AGENTIC_JUDGE_TYPE, base_eval_run_id="base")
    assert corr == "judge-corr"
    assert multi == "judge-multi"
    assert unfiltered == "judge-multi"



# --- focused slices skip judges the screen does not read ---


def _pair(focused: bool):
    from glean_gepa.teacher_student_adapter import _StartedPair

    inst = {"eval_set_name": "set", "eval_set_version": "v1", "deployment_ids": ["prod"], "status": "active"}
    if focused:
        inst["eval_entry_ids"] = ["e1", "e2"]
    return _StartedPair(al_data_inst=inst, teacher_eval_id="t", student_eval_id="s")  # type: ignore[arg-type]


def test_fix_rate_screen_starts_no_judge_on_focused_slice():
    config = load_experiment_config(WALDO_CONFIG)
    adapter = TeacherStudentAdapter(
        runner=ALRunner(evalcli=MagicMock()),
        teacher_model="gpt",
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        pairwise_judges=pairwise_judges(config),
        screening_kind="high_signal_fix_rate",
    )
    assert [j.name for j in adapter._pairwise_judges_for_pair(_pair(focused=False))] == [
        CUSTOMER_AGENTIC_CORRECTNESS_METRIC
    ]
    assert adapter._pairwise_judges_for_pair(_pair(focused=True)) == ()


def test_weighted_screen_keeps_named_judge_on_focused_slice():
    config = load_experiment_config(WALDO_CONFIG)
    adapter = TeacherStudentAdapter(
        runner=ALRunner(evalcli=MagicMock()),
        teacher_model="gpt",
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        pairwise_judges=pairwise_judges(config),
        screening_kind="high_signal_fix_rate",
        screening_weights={"tool_alignment": 0.5, CUSTOMER_AGENTIC_CORRECTNESS_METRIC: 0.5},
    )
    assert [j.name for j in adapter._pairwise_judges_for_pair(_pair(focused=True))] == [
        CUSTOMER_AGENTIC_CORRECTNESS_METRIC
    ]
