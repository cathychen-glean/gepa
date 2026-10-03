from __future__ import annotations
import hashlib
import json
from unittest.mock import MagicMock
import pytest
from glean_gepa.al_adapter import ALRunner
from glean_gepa.judge_metrics_util import JUDGE_SPECS
from base64 import urlsafe_b64decode
from urllib.parse import unquote_plus
from glean_gepa.al_adapter import ALRunner, Candidate, ModuleSpec, approx_token_len, total_prompt_tokens
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.evolutionary_proposer import (
    modules_after_tool_choice,
    pick_modules_to_edit,
    tool_usage_in_examples,
    tools_with_evidence,
)
from glean_gepa.prompt import (
    compile_encoded_prompt,
    compile_tool_description_overrides,
    is_core_tool_span,
    tool_description_override_key,
)
from glean_gepa.prompt_constants import (
    CORE_TOOL_DESCRIPTIONS,
    CORE_TOOLS,
    EXECUTION_DISCIPLINE_KEY,
    RULES_EXT_KEY,
    TOOL_DESCRIPTION_OVERRIDES_PARAM,
    WRITING_CODE_KEY,
    WRITING_CODE_TOKEN_BUDGET,
)
from glean_gepa.reflection_prompts import parse_chosen_tool_keys
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter
from unittest.mock import MagicMock
import pytest
import yaml
from glean_gepa.adapter_types import EvalHarness
from glean_gepa.al_adapter import CODING_HARNESS_SC_PARAMS, ALRunner
from glean_gepa.experiment_config import ExperimentConfigError, eval_harness, load_experiment_config


def _start(runner: ALRunner) -> tuple[str, bool]:
    return runner.start("fast", "prompt", "set", "v1", ["prod"])


def test_start_writes_in_flight_and_wait_promotes_cancelled(tmp_path):
    cache_file = tmp_path / "eval-runs.json"
    client = MagicMock()
    client.create_eval_run.return_value = "run_abc"
    client.wait_for_eval_run.return_value = [{"taskCountsByStatus": [{"status": "TASK_CANCELLED", "count": 1}]}]
    runner = ALRunner(evalcli=client, cache_file=str(cache_file))

    eval_id, wait_required = _start(runner)
    assert wait_required
    assert eval_id == "run_abc"
    saved = json.loads(cache_file.read_text())
    assert saved["completed"] == {}
    assert saved["in_flight"]["run_abc"][0] == "fast"
    assert saved["in_flight"]["run_abc"][2:] == ["set", "v1", "gepa", "prod"]

    runner.wait(eval_id)
    saved = json.loads(cache_file.read_text())
    assert saved["in_flight"] == {}
    assert "run_abc" in saved["completed"].values()


@pytest.mark.parametrize(
    ("status", "wait_required", "expect_completed"),
    [
        ([{"taskCountsByStatus": [{"status": "TASK_SUBMITTED", "count": 1}]}], True, False),
        ([{"taskCountsByStatus": [{"status": "TASK_SUCCEEDED", "count": 10}]}], False, True),
    ],
)
def test_resumed_runner_reuses_cached_eval(tmp_path, status, wait_required, expect_completed):
    cache_file = tmp_path / "eval-runs.json"
    first = ALRunner(evalcli=MagicMock(create_eval_run=MagicMock(return_value="run_abc")), cache_file=str(cache_file))
    _start(first)

    second_client = MagicMock()
    second_client.get_eval_run_status.return_value = status
    second = ALRunner(evalcli=second_client, cache_file=str(cache_file))
    eval_id, actual_wait = _start(second)

    second_client.create_eval_run.assert_not_called()
    assert eval_id == "run_abc"
    assert actual_wait is wait_required
    saved = json.loads(cache_file.read_text())
    if expect_completed:
        assert "run_abc" in saved["completed"].values()
        assert saved["in_flight"] == {}
    else:
        assert saved["completed"] == {}
        assert "run_abc" in saved["in_flight"]


def test_resumed_runner_recreates_a_cancelled_cached_eval(tmp_path):
    """A cached run that was killed (mostly TASK_CANCELLED) is dropped and relaunched.

    Reusing it would score the handful of entries that finished before the cancel
    as if they were the whole eval.
    """
    cache_file = tmp_path / "eval-runs.json"
    first = ALRunner(evalcli=MagicMock(create_eval_run=MagicMock(return_value="run_abc")), cache_file=str(cache_file))
    _start(first)
    first.wait("run_abc")
    assert "run_abc" in json.loads(cache_file.read_text())["completed"].values()

    second_client = MagicMock(create_eval_run=MagicMock(return_value="run_new"))
    second_client.get_eval_run_status.return_value = [
        {
            "taskCountsByStatus": [
                {"status": "TASK_CANCELLED", "count": 178},
                {"status": "TASK_SUCCEEDED", "count": 14},
                {"status": "TASK_EXECUTING", "count": 1},
            ]
        }
    ]
    second = ALRunner(evalcli=second_client, cache_file=str(cache_file))
    eval_id, wait_required = _start(second)

    second_client.create_eval_run.assert_called_once()
    assert eval_id == "run_new"
    assert wait_required is True
    saved = json.loads(cache_file.read_text())
    assert "run_abc" not in saved["completed"].values()
    assert "run_abc" not in saved["in_flight"]
    assert "run_new" in saved["in_flight"]


def test_wait_polls_when_completed_id_is_probed_ongoing(tmp_path):
    """A completed-map id that Cortex reports as still running must be waited on.

    Otherwise the caller scores half-written telemetry as the final result.
    """
    cache_file = tmp_path / "eval-runs.json"
    first = ALRunner(evalcli=MagicMock(create_eval_run=MagicMock(return_value="run_abc")), cache_file=str(cache_file))
    _start(first)
    first.wait("run_abc")
    assert "run_abc" in json.loads(cache_file.read_text())["completed"].values()

    second_client = MagicMock()
    second_client.get_eval_run_status.return_value = [
        {"taskCountsByStatus": [{"status": "TASK_SUBMITTED", "count": 4}]}
    ]
    second = ALRunner(evalcli=second_client, cache_file=str(cache_file))

    eval_id, wait_required = _start(second)
    assert eval_id == "run_abc"
    assert wait_required is True

    second.wait(eval_id)
    second_client.wait_for_eval_run.assert_called_once_with("run_abc")


def test_flat_cache_still_waits_when_run_is_ongoing(tmp_path):
    """Flat caches load every id as completed; an ongoing probe still wins."""
    cache_file = tmp_path / "eval-runs.json"
    prompt_hash = hashlib.md5(b"prompt").hexdigest()[:16]
    cache_file.write_text(json.dumps({json.dumps(["fast", prompt_hash, "set", "v1", "gepa", "prod"]): "run_old"}))
    client = MagicMock()
    client.get_eval_run_status.return_value = [{"taskCountsByStatus": [{"status": "TASK_RUNNING", "count": 2}]}]
    runner = ALRunner(evalcli=client, cache_file=str(cache_file))

    eval_id, wait_required = _start(runner)
    assert eval_id == "run_old"
    assert wait_required is True

    runner.wait(eval_id)
    client.wait_for_eval_run.assert_called_once_with("run_old")


def test_repeated_start_keeps_waiting_while_in_flight(tmp_path):
    """The verified fast path must not drop the wait on a second start() call."""
    cache_file = tmp_path / "eval-runs.json"
    client = MagicMock(create_eval_run=MagicMock(return_value="run_abc"))
    runner = ALRunner(evalcli=client, cache_file=str(cache_file))

    _, first_wait = _start(runner)
    assert first_wait is True

    _, second_wait = _start(runner)
    assert second_wait is True
    client.create_eval_run.assert_called_once()


def _ensure_correctness_judge(runner: ALRunner) -> str:
    return runner.ensure_judge_run(
        eval_run_id="eval-best",
        judge_type="CORRECTNESS",
        run_params="{}",
        base_eval_run_id="eval-base",
    )


def test_ensure_judge_run_creates_then_persists_for_next_process(tmp_path):
    cache_file = tmp_path / "eval-runs.json"
    client = MagicMock()
    client.find_judge_run_id.return_value = None
    client.create_judge_run.return_value = "judge-1"
    runner = ALRunner(evalcli=client, cache_file=str(cache_file))

    assert _ensure_correctness_judge(runner) == "judge-1"
    client.create_judge_run.assert_called_once()
    saved = json.loads(cache_file.read_text())
    assert saved["judge_runs"][json.dumps(["eval-best", "eval-base", "CORRECTNESS"])] == "judge-1"

    # A fresh process must reuse the cached judge instead of restarting it.
    resumed_client = MagicMock()
    resumed = ALRunner(evalcli=resumed_client, cache_file=str(cache_file))
    assert _ensure_correctness_judge(resumed) == "judge-1"
    resumed_client.create_judge_run.assert_not_called()
    resumed_client.find_judge_run_id.assert_not_called()


def test_ensure_judge_run_adopts_existing_cortex_run_with_empty_cache(tmp_path):
    """A judge that already ran must be adopted even when the local cache is gone."""
    cache_file = tmp_path / "eval-runs.json"
    client = MagicMock()
    client.find_judge_run_id.return_value = "abe8650e-2c44-4dc5-bdd0-a36485f74483"
    runner = ALRunner(evalcli=client, cache_file=str(cache_file))

    assert _ensure_correctness_judge(runner) == "abe8650e-2c44-4dc5-bdd0-a36485f74483"
    client.create_judge_run.assert_not_called()
    client.find_judge_run_id.assert_called_once_with(
        "eval-best",
        judge_type="CORRECTNESS",
        base_eval_run_id="eval-base",
    )
    saved = json.loads(cache_file.read_text())
    assert saved["judge_runs"][json.dumps(["eval-best", "eval-base", "CORRECTNESS"])] == (
        "abe8650e-2c44-4dc5-bdd0-a36485f74483"
    )


@pytest.mark.parametrize("spec", [s for s in JUDGE_SPECS.values() if s.cortex_type_override], ids=lambda s: s.name)
def test_ensure_judge_run_sends_the_cortex_type_but_caches_under_the_adapter_key(spec):
    client = MagicMock()
    client.find_judge_run_id.return_value = None
    client.create_judge_run.return_value = "judge-x"
    runner = ALRunner(evalcli=client)
    runner.ensure_judge_run(
        eval_run_id="best",
        judge_type=spec.judge_type,
        run_params=spec.run_params,
        base_eval_run_id="base",
        input_mappings=spec.input_mappings,
        cortex_judge_type=spec.cortex_judge_type,
        judge_skill_name=spec.judge_skill_name,
    )
    for call in (client.find_judge_run_id, client.create_judge_run):
        assert call.call_args.kwargs["judge_type"] == spec.cortex_judge_type
    assert client.find_judge_run_id.call_args.kwargs["judge_skill_name"] == spec.judge_skill_name
    assert json.loads(client.create_judge_run.call_args.kwargs["run_params"])["judge_skill_name"] == spec.judge_skill_name
    assert runner._judge_run_ids[("best", "base", spec.judge_type)] == "judge-x"
    assert ("best", "base", spec.cortex_judge_type) not in runner._judge_run_ids


def test_dropping_stale_eval_discards_its_judge_runs(tmp_path):
    cache_file = tmp_path / "eval-runs.json"
    client = MagicMock()
    client.find_judge_run_id.return_value = None
    client.create_judge_run.return_value = "judge-1"
    runner = ALRunner(evalcli=client, cache_file=str(cache_file))
    _ensure_correctness_judge(runner)

    runner._drop_eval("eval-base")

    assert json.loads(cache_file.read_text())["judge_runs"] == {}


def test_cache_entry_without_deployments_is_not_reused(tmp_path):
    """A cache entry predating the deployment field must be re-run, not adopted.

    Its deployment subset is unknown, so reusing it can pair a student against a
    teacher that shares none of its entry ids, scoring every entry as a mismatch.
    """
    cache_file = tmp_path / "eval-runs.json"
    prompt_hash = hashlib.md5(b"prompt").hexdigest()[:16]
    cache_file.write_text(json.dumps({json.dumps(["fast", prompt_hash, "set", "v1", "gepa"]): "run_old"}))
    client = MagicMock()
    client.create_eval_run.return_value = "run_new"
    runner = ALRunner(evalcli=client, cache_file=str(cache_file))

    eval_id, wait_required = _start(runner)

    assert eval_id == "run_new"
    assert wait_required is True
    client.create_eval_run.assert_called_once()


def test_the_deployment_set_decides_the_cache_key(tmp_path):
    """Deployments decide which entries run, so a different subset is a different eval.

    The same subset in a different order is not: the signature sorts before joining.
    """
    cache_file = tmp_path / "eval-runs.json"
    first = ALRunner(evalcli=MagicMock(create_eval_run=MagicMock(return_value="run_a")), cache_file=str(cache_file))
    first.start("fast", "prompt", "set", "v1", ["beta", "alpha"])

    other_client = MagicMock(create_eval_run=MagicMock(return_value="run_b"))
    other = ALRunner(evalcli=other_client, cache_file=str(cache_file))
    assert other.start("fast", "prompt", "set", "v1", ["other-deployment"])[0] == "run_b"
    other_client.create_eval_run.assert_called_once()

    reordered_client = MagicMock()
    reordered_client.get_eval_run_status.return_value = [
        {"taskCountsByStatus": [{"status": "TASK_SUCCEEDED", "count": 3}]}
    ]
    reordered = ALRunner(evalcli=reordered_client, cache_file=str(cache_file))
    eval_id, wait_required = reordered.start("fast", "prompt", "set", "v1", ["alpha", "beta"])

    assert eval_id == "run_a"
    assert wait_required is False
    reordered_client.create_eval_run.assert_not_called()


def test_total_prompt_tokens_excludes_core_tool_descriptions():
    candidate = Candidate(
        model="gpt",
        prompt_modules={WRITING_CODE_KEY: "abcd" * 10, "glean_search": "x" * 400},
        module_specs={WRITING_CODE_KEY: ModuleSpec(WRITING_CODE_KEY, "free_text", WRITING_CODE_TOKEN_BUDGET)},
        global_token_cap=4096,
        baseline_prompt_hash="h",
    )
    assert total_prompt_tokens(candidate) == approx_token_len("abcd" * 10)


_BASE_YAML = """
schema_version: 1
mode: teacher_student
signals:
  - name: tool_alignment
    source: tool_match
    lookback_days: 7
objective:
  primary: tool_alignment
  composite:
    tool_alignment: 1.0
  params:
    tool_alignment: {}
screening:
  kind: high_signal_fix_rate
  threshold: 0.25
  high_signal: tool_alignment
"""


def test_eval_section_parsing_and_validation(tmp_path):
    base = yaml.safe_load(_BASE_YAML)

    def write(eval_section):
        base["eval"] = eval_section
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(base))
        return path

    harness = eval_harness(load_experiment_config(write({"runner_type": "GLEAN_CHAT_V2", "sc_params": "a=1,b=2"})))
    assert harness == ("GLEAN_CHAT_V2", "a=1,b=2")
    harness = eval_harness(load_experiment_config(write({"sc_params": ["a=1", " b=2 "]})))
    assert harness == (None, "a=1,b=2")
    with pytest.raises(ExperimentConfigError, match="key=value"):
        load_experiment_config(write({"sc_params": ["novalue"]}))
    with pytest.raises(ExperimentConfigError, match="unknown keys"):
        load_experiment_config(write({"runner": "GLEAN_CHAT"}))


def test_runner_passes_harness_to_evalcli_or_keeps_defaults():
    evalcli = MagicMock()
    evalcli.create_eval_run.return_value = "ev-1"
    with_harness = ALRunner(evalcli=evalcli, harness=EvalHarness(runner_type="GLEAN_CHAT", sc_params="x.y=1"))
    without = ALRunner(evalcli=evalcli)
    for runner in (with_harness, without):
        runner._resolve_cached_eval = lambda key: None  # type: ignore[method-assign]

    with_harness.start("gpt", "", "set", "v1", ["scio-prod"])
    kwargs = evalcli.create_eval_run.call_args.kwargs
    assert kwargs["runner_type"] == "GLEAN_CHAT"
    assert kwargs["sc_params"].startswith("x.y=1") and CODING_HARNESS_SC_PARAMS not in kwargs["sc_params"]
    assert kwargs["eval_params"].endswith("gleanchat_agent=ADVANCED")

    without.start("fast", "", "set", "v1", ["scio-prod"])
    kwargs = evalcli.create_eval_run.call_args.kwargs
    assert "runner_type" not in kwargs  # evalcli's own default applies
    assert kwargs["sc_params"].startswith(CODING_HARNESS_SC_PARAMS)
    assert kwargs["eval_params"].endswith("gleanchat_agent=FAST")
