from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from glean_gepa.al_adapter import AGENTIC_LOOP_MODEL_OVERRIDES, CODING_HARNESS_SC_PARAMS, ALRunner
from glean_gepa.evalcli_client import (
    COMPLETENESS_JUDGE_TYPE,
    COMPLETENESS_RUN_PARAMS,
    CORRECTNESS_INPUT_MAPPINGS,
    CORRECTNESS_JUDGE_TYPE,
    CORRECTNESS_RUN_PARAMS,
    EvalCliClient,
    EvalCliError,
    _subprocess_env,
    min_ingested_eval_set_entries,
)
from glean_gepa.judge_metrics_util import (
    judge_pass_rate_from_metrics,
    wait_for_judge_metrics,
)


def test_coding_harness_sc_params_selects_coding_agent_loop():
    runner = ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli"))

    params = runner._build_sc_params("gpt", "")

    assert params.startswith(CODING_HARNESS_SC_PARAMS)
    assert "co.internal_looping_pyagent_default_route_override=coding_agent_loop" in params
    assert "co.py_agent_route_override=o3_agentic_loop" not in params
    assert "co.lo.cao.agentic_loop_sc_params=co.so.enable_for_agentic_loop%3D1%2C" in params
    assert "co.so.ptc_only_tools%3Dglean_search%253Bglean_document_reader" in params
    assert "co.lo.oai_model_for_agentic_loop=" not in params


@pytest.mark.parametrize("alias", sorted(AGENTIC_LOOP_MODEL_OVERRIDES))
def test_build_sc_params_overrides_claude_models(alias: str):
    runner = ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli"))

    params = runner._build_sc_params(alias, "")

    assert f"co.lo.oai_model_for_agentic_loop={AGENTIC_LOOP_MODEL_OVERRIDES[alias]}" in params


def test_build_sc_params_rejects_unknown_and_legacy_claude_alias():
    runner = ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli"))

    with pytest.raises(ValueError, match="Unknown model: claude$"):
        runner._build_sc_params("claude", "")
    with pytest.raises(ValueError, match="Unknown model: gemini$"):
        runner._build_sc_params("gemini", "")


def test_create_eval_run_invokes_evalcli_with_expected_args():
    client = EvalCliClient(binary="/fake/evalcli")
    with patch.object(client, "_invoke_json", return_value={"id": "run_123"}) as mock_invoke:
        run_id = client.create_eval_run(
            eval_run_id="run_123",
            eval_set_name="AI Answers Small",
            eval_set_version="20260403",
            deployment_ids=["scio-prod"],
            description="GEPA eval run for AI Answers Small:20260403",
            sc_params="co.debug_mode=1",
            eval_params="experimental_queue=temp-system-prompt-optimization,gleanchat_agent=FAST",
        )

    assert run_id == "run_123"
    mock_invoke.assert_called_once()
    args = mock_invoke.call_args[0]
    assert args[0:4] == ("run", "create", "--eval-set", "AI Answers Small:20260403")
    preset_idx = args.index("--preset")
    assert args[preset_idx + 1] == "Coding Harness"
    assert "--sc-params" in args
    assert "--eval-params" in args


def test_al_runner_invokes_on_created_before_waiting(tmp_path):
    cache_file = tmp_path / "eval-runs.json"
    client = EvalCliClient(binary="/fake/evalcli")
    runner = ALRunner(evalcli=client, cache_file=str(cache_file))
    events = []

    def on_created(eval_run_id):
        events.append(("created", eval_run_id))
        assert eval_run_id in runner._in_flight
        assert not cache_file.exists()

    def wait_for_eval_run(eval_run_id):
        events.append(("wait", eval_run_id))

    with (
        patch.object(client, "create_eval_run", return_value="run_123"),
        patch.object(client, "wait_for_eval_run", side_effect=wait_for_eval_run),
    ):
        eval_id, wait_required = runner.start(
            "fast",
            "prompt",
            "eval-set",
            "v1",
            ["scio-prod"],
            on_created=on_created,
        )
        assert wait_required
        runner.wait(eval_id)
        assert eval_id == "run_123"

    assert [event[0] for event in events] == ["created", "wait"]
    assert list(json.loads(cache_file.read_text()).values()) == ["run_123"]


def test_create_judge_run_parses_response_list():
    client = EvalCliClient(binary="/fake/evalcli")
    with patch.object(client, "_invoke_json", return_value=[{"id": "judge_456", "status": "SUBMITTED"}]) as mock_invoke:
        judge_id = client.create_judge_run(
            eval_run_id="student",
            judge_type=CORRECTNESS_JUDGE_TYPE,
            run_params=CORRECTNESS_RUN_PARAMS,
            base_eval_run_id="teacher",
            input_mappings=CORRECTNESS_INPUT_MAPPINGS,
        )

    assert judge_id == "judge_456"
    args = mock_invoke.call_args[0]
    assert args[args.index("--judge-type") + 1] == "CORRECTNESS"
    assert args[args.index("--base-eval-run-id") + 1] == "teacher"
    assert "--input-mappings" in args


def test_completeness_evalcli_create_list_and_metrics():
    client = EvalCliClient(binary="/fake/evalcli")
    with patch.object(client, "_invoke_json", return_value={"id": "judge_complete"}) as mock_invoke:
        judge_id = client.create_judge_run(
            eval_run_id="student-run",
            judge_type=COMPLETENESS_JUDGE_TYPE,
            run_params=COMPLETENESS_RUN_PARAMS,
        )
    assert judge_id == "judge_complete"
    create_args = mock_invoke.call_args[0]
    assert create_args[0:2] == ("judge", "create")
    assert create_args[create_args.index("--eval-run-id") + 1] == "student-run"
    assert create_args[create_args.index("--judge-type") + 1] == "COMPLETENESS"
    assert "--base-eval-run-id" not in create_args

    with patch.object(
        client,
        "_invoke_json",
        return_value={"judgeRuns": [{"id": "judge-1", "config": {"judgeType": "COMPLETENESS"}}]},
    ) as mock_invoke:
        found = client.find_judge_run_id("student-run", judge_type="COMPLETENESS")
    assert found == "judge-1"
    list_args = mock_invoke.call_args[0]
    assert list_args[0:2] == ("judge", "list")
    assert "list-for-run" not in list_args
    assert list_args[list_args.index("--eval-run-ids") + 1] == "student-run"

    with patch.object(client, "_invoke_json", return_value={"judgeMetrics": {}}) as mock_invoke:
        client.get_eval_metrics("student-run")
    metrics_args = mock_invoke.call_args[0]
    assert metrics_args[0:2] == ("metrics", "summary")
    assert metrics_args[metrics_args.index("--test-eval-id") + 1] == "student-run"


@pytest.mark.parametrize(
    "payload, expected",
    [
        (
            {
                "judgeMetrics": {
                    "additional_properties": {
                        "COMPLETENESS": {"passRate": 0.8, "sampleSize": 10, "judgeRunId": "judge-1"},
                    }
                }
            },
            0.8,
        ),
        (
            {"judgeMetrics": {"COMPLETENESS": [{"judgeRunId": "judge-1", "test": 0.75, "sampleSize": 4}]}},
            0.75,
        ),
        (
            {"judgeMetrics": {"COMPLETENESS": {"passRate": None, "judgeRunId": "judge-1"}}},
            None,
        ),
    ],
)
def test_judge_pass_rate_from_metrics(payload, expected):
    assert judge_pass_rate_from_metrics(payload, judge_type=COMPLETENESS_JUDGE_TYPE, judge_run_id="judge-1") == expected


def test_wait_for_judge_metrics_ready_and_timeout():
    ready = type("EvalCli", (), {})()
    ready.get_eval_metrics = lambda _eval_id, **_kwargs: {"judgeMetrics": {"COMPLETENESS": {"passRate": 0.6}}}
    analysis = wait_for_judge_metrics(
        ready,
        eval_id="run-1",
        judge_type=COMPLETENESS_JUDGE_TYPE,
        judge_run_id="judge-1",
        poll_interval_sec=0,
    )
    assert analysis.aggregate == 0.6
    assert analysis.eval_id == "run-1"

    stalled = type("EvalCli", (), {})()
    stalled.get_eval_metrics = lambda _eval_id, **_kwargs: {"judgeMetrics": {"COMPLETENESS": {"passRate": None}}}
    with pytest.raises(EvalCliError, match="not ready"):
        wait_for_judge_metrics(
            stalled,
            eval_id="run-1",
            judge_type=COMPLETENESS_JUDGE_TYPE,
            judge_run_id="judge-1",
            poll_interval_sec=0,
            timeout_sec=0,
        )


def test_list_eval_set_versions_returns_matching_and_unspecified_deployments():
    client = EvalCliClient(binary="/fake/evalcli")
    payload = {
        "evalSetVersions": [
            {"version": "20260827", "availableDeploymentIds": ["scio-prod"]},
            {"version": "20260826", "availableDeploymentIds": ["scio-staging"]},
            {"version": "20260825"},
        ]
    }
    with patch.object(client, "_invoke_json", return_value=payload) as mock_invoke:
        rows = client.list_eval_set_versions(eval_set_name="Glean Chat V2 Medium", deployment_ids=["scio-prod"])

    assert [row["version"] for row in rows] == ["20260827", "20260825"]
    assert mock_invoke.call_args[0] == (
        "evalsets",
        "versions",
        "--name",
        "Glean Chat V2 Medium",
        "--page",
        "1",
        "--page-size",
        "100",
    )


def test_list_eval_set_versions_returns_version_rows():
    client = EvalCliClient(binary="/fake/evalcli")
    with patch.object(client, "_invoke_json", return_value={"evalSetVersions": [{"version": "20260827"}]}) as mock_invoke:
        rows = client.list_eval_set_versions(eval_set_name="Glean Chat V2 Medium", deployment_ids=["scio-prod"])

    assert rows == [{"version": "20260827"}]
    assert mock_invoke.call_args[0] == (
        "evalsets",
        "versions",
        "--name",
        "Glean Chat V2 Medium",
        "--page",
        "1",
        "--page-size",
        "100",
    )


def test_wait_for_judge_run_raises_on_failure():
    client = EvalCliClient(binary="/fake/evalcli")
    with patch.object(client, "_invoke_json", return_value={"id": "judge_456", "status": "FAILED"}):
        with pytest.raises(EvalCliError, match="ended with status FAILED"):
            client.wait_for_judge_run("judge_456", poll_interval_sec=0, timeout_sec=1)


def test_subprocess_env_replaces_unreliable_ssl_cert(monkeypatch, tmp_path):
    ca_bundle = tmp_path / "ca.pem"
    ca_bundle.write_text("fake-ca", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", "/var/folders/abc/socketFirewallCa.crt")
    monkeypatch.setenv("SSL_CERT_DIR", "/var/folders/abc")
    monkeypatch.setattr(
        "glean_gepa.evalcli_client._resolve_ca_bundle",
        lambda: str(ca_bundle),
    )

    env = _subprocess_env()

    assert env["SSL_CERT_FILE"] == str(ca_bundle)
    assert env["REQUESTS_CA_BUNDLE"] == str(ca_bundle)
    assert "SSL_CERT_DIR" not in env


def test_subprocess_env_sets_ssl_cert_when_missing(monkeypatch, tmp_path):
    ca_bundle = tmp_path / "ca.pem"
    ca_bundle.write_text("fake-ca", encoding="utf-8")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.setattr(
        "glean_gepa.evalcli_client._resolve_ca_bundle",
        lambda: str(ca_bundle),
    )

    env = _subprocess_env()

    assert env["SSL_CERT_FILE"] == str(ca_bundle)
    assert env["REQUESTS_CA_BUNDLE"] == str(ca_bundle)


def test_subprocess_env_preserves_existing_ssl_cert(monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "/custom/ca.pem")
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)

    env = _subprocess_env()

    assert env["SSL_CERT_FILE"] == "/custom/ca.pem"
    assert "REQUESTS_CA_BUNDLE" not in env


@pytest.mark.parametrize(
    "error",
    [
        EvalCliError("stderr: Error: API request failed: 502\nResponse: Connection refused"),
        EvalCliError("stderr: Error: Could not find valid __Host-GCP_IAP_AUTH_TOKEN_* cookie in any browser."),
    ],
)
def test_wait_for_eval_run_retries_transient_errors(error, capsys):
    client = EvalCliClient(binary="/fake/evalcli")
    in_progress = [{"taskCountsByStatus": [{"status": "TASK_SUBMITTED", "count": 1}]}]
    complete = [{"taskCountsByStatus": [{"status": "TASK_SUCCEEDED", "count": 1}]}]
    with patch.object(
        client,
        "_invoke_json",
        side_effect=[error, in_progress, complete],
    ) as mock_invoke:
        with patch("glean_gepa.evalcli_client.time.sleep"):
            client.wait_for_eval_run("run_123", poll_interval_sec=0)

    assert mock_invoke.call_count == 3
    status_logs = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Eval run run_123 status:")]
    assert len(status_logs) == 2
    assert "TASK_SUBMITTED" in status_logs[0]
    assert "TASK_SUCCEEDED" in status_logs[1]


def test_wait_for_eval_run_honors_timeout():
    client = EvalCliClient(binary="/fake/evalcli")
    with (
        patch.object(client, "_invoke_json") as mock_invoke,
        patch("glean_gepa.evalcli_client.time.monotonic", side_effect=[10.0, 11.0]),
    ):
        with pytest.raises(EvalCliError, match="timed out after 1s"):
            client.wait_for_eval_run("run_123", poll_interval_sec=0, timeout_sec=1)

    mock_invoke.assert_not_called()


def test_wait_for_eval_run_timeout_includes_last_status():
    client = EvalCliClient(binary="/fake/evalcli")
    in_progress = [{"taskCountsByStatus": [{"status": "TASK_SUBMITTED", "count": 2}]}]
    with patch.object(client, "_invoke_json", return_value=in_progress):
        with patch("glean_gepa.evalcli_client.time.sleep"):
            with pytest.raises(EvalCliError, match="timed out after 1s"):
                client.wait_for_eval_run("run_123", poll_interval_sec=1, timeout_sec=1)


def test_wait_for_eval_run_raises_on_non_transient_errors():
    client = EvalCliClient(binary="/fake/evalcli")
    with patch.object(
        client,
        "_invoke_json",
        side_effect=EvalCliError("stderr: auth failed"),
    ):
        with pytest.raises(EvalCliError, match="auth failed"):
            client.wait_for_eval_run("run_123", poll_interval_sec=0)


def test_wait_for_eval_set_entries_accepts_stable_partial_ingest():
    client = EvalCliClient(binary="/fake/evalcli")
    rows = [{"id": f"e{i}"} for i in range(19)]
    with (
        patch.object(client, "list_eval_set_entries", return_value=rows) as list_entries,
        patch("glean_gepa.evalcli_client.time.sleep"),
    ):
        ingested = client.wait_for_eval_set_entries(
            eval_set_name="focused",
            eval_set_version="v1",
            deployment_ids=["prod"],
            expected_count=20,
            poll_interval_sec=1,
            timeout_sec=10,
        )

    assert ingested == rows
    assert list_entries.call_count == 2


def test_wait_for_eval_set_entries_times_out_below_half_ingested():
    client = EvalCliClient(binary="/fake/evalcli")
    rows = [{"id": f"e{i}"} for i in range(9)]
    with (
        patch.object(client, "list_eval_set_entries", return_value=rows),
        patch("glean_gepa.evalcli_client.time.sleep"),
        pytest.raises(EvalCliError, match="need at least 10"),
    ):
        client.wait_for_eval_set_entries(
            eval_set_name="focused",
            eval_set_version="v1",
            deployment_ids=["prod"],
            expected_count=20,
            poll_interval_sec=1,
            timeout_sec=2,
        )


def test_min_ingested_eval_set_entries_is_half_rounded_up():
    assert min_ingested_eval_set_entries(20) == 10
    assert min_ingested_eval_set_entries(19) == 10
    assert min_ingested_eval_set_entries(1) == 1


def test_invoke_raises_on_nonzero_exit():
    client = EvalCliClient(binary="/fake/evalcli")
    with patch("glean_gepa.evalcli_client.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="auth failed")
        with pytest.raises(EvalCliError, match="auth failed"):
            client._invoke("whoami")
