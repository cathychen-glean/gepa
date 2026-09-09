from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock

import pytest

from glean_gepa.al_adapter import ALRunner


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
    assert saved["in_flight"]["run_abc"][2:] == ["set", "v1", "gepa"]

    runner.wait(eval_id)
    saved = json.loads(cache_file.read_text())
    assert saved["in_flight"] == {}
    assert "run_abc" in saved["completed"].values()


@pytest.mark.parametrize(
    ("status", "wait_required", "expect_completed"),
    [
        ([{"taskCountsByStatus": [{"status": "TASK_SUBMITTED", "count": 1}]}], True, False),
        ([{"taskCountsByStatus": [{"status": "TASK_SUCCEEDED", "count": 10}]}], False, True),
        (
            [
                {
                    "taskCountsByStatus": [
                        {"status": "TASK_SUCCEEDED", "count": 77},
                        {"status": "TASK_CANCELLED", "count": 123},
                    ]
                }
            ],
            False,
            True,
        ),
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


def test_v1_flat_cache_still_waits_when_run_is_ongoing(tmp_path):
    """Legacy flat caches load every id as completed; an ongoing probe still wins."""
    cache_file = tmp_path / "eval-runs.json"
    prompt_hash = hashlib.md5(b"prompt").hexdigest()[:16]
    cache_file.write_text(json.dumps({json.dumps(["fast", prompt_hash, "set", "v1", "gepa"]): "run_old"}))
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


def test_dropping_stale_eval_discards_its_judge_runs(tmp_path):
    cache_file = tmp_path / "eval-runs.json"
    client = MagicMock()
    client.find_judge_run_id.return_value = None
    client.create_judge_run.return_value = "judge-1"
    runner = ALRunner(evalcli=client, cache_file=str(cache_file))
    _ensure_correctness_judge(runner)

    runner._drop_eval("eval-base")

    assert json.loads(cache_file.read_text())["judge_runs"] == {}


def test_v1_eval_run_cache_still_loads(tmp_path):
    cache_file = tmp_path / "eval-runs.json"
    prompt_hash = hashlib.md5(b"prompt").hexdigest()[:16]
    cache_file.write_text(json.dumps({json.dumps(["fast", prompt_hash, "set", "v1", "gepa"]): "run_old"}))
    client = MagicMock()
    client.get_eval_run_status.return_value = [{"taskCountsByStatus": [{"status": "TASK_SUCCEEDED", "count": 3}]}]
    runner = ALRunner(evalcli=client, cache_file=str(cache_file))
    eval_id, wait_required = _start(runner)

    assert eval_id == "run_old"
    assert wait_required is False
    client.create_eval_run.assert_not_called()
