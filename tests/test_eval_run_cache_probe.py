"""Cached eval runs survive a status payload that lacks task counts."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from glean_gepa.al_adapter import ALRunner

KEY = ["gpt", "h", "Set", "v1", "gepa", "scio-prod"]
COMPLETED_ID = "gepa_old_completed"
INFLIGHT_ID = "gepa_old_inflight"


def _runner(tmp_path, *, status, completed=True):
    cache = tmp_path / "cache.json"
    body = {"completed": {}, "in_flight": {}, "judge_runs": {}}
    if completed:
        body["completed"][json.dumps(KEY)] = COMPLETED_ID
    else:
        body["in_flight"][INFLIGHT_ID] = KEY
    cache.write_text(json.dumps(body))
    evalcli = MagicMock()
    evalcli.get_eval_run_status.return_value = status
    evalcli.create_eval_run.return_value = "gepa_new"
    return ALRunner(evalcli=evalcli, cache_file=cache), evalcli


def _key():
    return tuple(KEY)


def test_payload_without_counts_keeps_completed_run(tmp_path, capsys):
    # Regression: this payload shape used to classify as missing and relaunch a 99%-done run.
    runner, evalcli = _runner(tmp_path, status=[{"crossDeploymentUuid": COMPLETED_ID}])
    assert runner._resolve_cached_eval(_key()) == (COMPLETED_ID, False)
    evalcli.create_eval_run.assert_not_called()
    out = capsys.readouterr().out
    assert "status payload had no task counts; keeping cached state" in out
    assert "STALE" not in out


def test_payload_without_counts_keeps_in_flight_run_waiting(tmp_path):
    runner, _ = _runner(tmp_path, status=[{"taskCountsByStatus": []}], completed=False)
    assert runner._resolve_cached_eval(_key()) == (INFLIGHT_ID, True)


def test_explicit_not_found_payload_drops_the_run(tmp_path, capsys):
    status = [{"crossDeploymentUuid": COMPLETED_ID, "errors": ["No EvalRun or JudgeRun found with this ID"]}]
    runner, _ = _runner(tmp_path, status=status)
    assert runner._resolve_cached_eval(_key()) is None
    assert "STALE" in capsys.readouterr().out
    assert COMPLETED_ID not in runner._eval_run_ids.values()


def test_none_status_drops_the_run(tmp_path):
    runner, _ = _runner(tmp_path, status=None)
    assert runner._resolve_cached_eval(_key()) is None


def test_near_finished_in_flight_run_is_promoted_to_usable(tmp_path):
    status = [
        {
            "taskCountsByStatus": [
                {"status": "TASK_SUCCEEDED", "count": 189},
                {"status": "TASK_FAILED", "count": 7},
                {"status": "TASK_EXECUTING", "count": 3},
                {"status": "TASK_IN_QUEUE", "count": 1},
            ]
        }
    ]
    runner, _ = _runner(tmp_path, status=status, completed=False)
    assert runner._resolve_cached_eval(_key()) == (INFLIGHT_ID, False)
    assert runner._eval_run_ids[_key()] == INFLIGHT_ID
