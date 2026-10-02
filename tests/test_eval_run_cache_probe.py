"""Cache resolver: what a ``run status`` payload does to a cached eval run."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from glean_gepa.al_adapter import ALRunner

KEY = ("gpt", "h", "Set", "v1", "gepa", "scio-prod")
NOT_FOUND = [{"errors": ["No EvalRun or JudgeRun found with this ID"]}]
NEAR_DONE = [{"taskCountsByStatus": [{"status": "TASK_SUCCEEDED", "count": 189}, {"status": "TASK_EXECUTING", "count": 4}]}]


@pytest.mark.parametrize(
    ("section", "status", "expected"),
    [
        # No task counts and no not-found error: keep whatever the cache believed.
        ("completed", [{"crossDeploymentUuid": "old"}], ("old", False)),
        ("in_flight", [{"taskCountsByStatus": []}], ("old", True)),
        # Explicit not-found (payload or evalcli None) drops the run.
        ("completed", NOT_FOUND, None),
        ("completed", None, None),
        # A near-finished in-flight run is promoted to usable.
        ("in_flight", NEAR_DONE, ("old", False)),
    ],
)
def test_resolve_cached_eval(tmp_path, section, status, expected):
    cache = tmp_path / "cache.json"
    body = {"completed": {}, "in_flight": {}, "judge_runs": {}}
    if section == "completed":
        body["completed"][json.dumps(list(KEY))] = "old"
    else:
        body["in_flight"]["old"] = list(KEY)
    cache.write_text(json.dumps(body))
    evalcli = MagicMock()
    evalcli.get_eval_run_status.return_value = status
    runner = ALRunner(evalcli=evalcli, cache_file=cache)
    assert runner._resolve_cached_eval(KEY) == expected
    evalcli.create_eval_run.assert_not_called()
