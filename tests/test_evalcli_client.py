from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from glean_gepa.al_adapter import ALRunner
from glean_gepa.coding_harness_params import CODING_HARNESS_SC_PARAMS
from glean_gepa.harnesses import AGENTIC_LOOP_MODEL_OVERRIDES, AGENTIC_LOOP_REASONING_EFFORT
from glean_gepa.bigquery_client import BigQueryClient, BigQueryError
from glean_gepa.debug import set_debug
from glean_gepa.evalcli_client import (
    AGENTIC_CORRECTNESS_JUDGE_NAME,
    AGENTIC_JUDGE_NAME,
    AGENTIC_JUDGE_TYPE,
    COMPLETENESS_JUDGE_TYPE,
    COMPLETENESS_RUN_PARAMS,
    CORRECTNESS_INPUT_MAPPINGS,
    CORRECTNESS_JUDGE_TYPE,
    CORRECTNESS_RUN_PARAMS,
    EVALCLI_TIMEOUT_SEC,
    EvalCliClient,
    EvalCliError,
    classify_eval_run_status,
    judge_run_skill_name,
    min_ingested_eval_set_entries,
)
from glean_gepa.judge_metrics_util import (
    JUDGE_SPECS,
    judge_metrics_snapshot,
    per_entry_from_analysis_view,
    wait_for_all_judge_metrics,
)

OPAQUE_EVALCLI_ERROR = EvalCliError(
    "evalcli failed (exit 1): /bin/evalcli judge create --eval-run-id student-run "
    '--judge-type COMPLETENESS --run-params {"Llm model": "default", "Use Cache": "true"} --json\n'
    "stderr: Error:\n"
    "stdout: "
)


def test_a_hung_evalcli_call_is_killed_and_retried():
    """An evalcli call that never returns must not stall the run forever.

    A `run create` was observed wedged for an hour after the server had already
    created the run, blocking the optimization behind a subprocess that never exited.
    """
    client = EvalCliClient(binary="/fake/evalcli")
    timed_out = subprocess.TimeoutExpired(cmd=["/fake/evalcli", "run", "create"], timeout=EVALCLI_TIMEOUT_SEC)
    with patch("glean_gepa.evalcli_client.subprocess.run", side_effect=timed_out) as mock_run:
        with pytest.raises(EvalCliError, match="evalcli timed out"):
            client._invoke("run", "create")
    assert mock_run.call_args.kwargs["timeout"] == EVALCLI_TIMEOUT_SEC

    # The timeout is transient, so the retry path recovers when the next call returns.
    with patch("glean_gepa.evalcli_client.time.sleep"):
        with patch.object(
            client, "_invoke_json", side_effect=[EvalCliError("evalcli timed out after 900s"), {"id": "r"}]
        ):
            assert client._invoke_json_retrying("run", "create", label="run create r") == {"id": "r"}


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
    assert "co.internal_looping_pyagent_default_route_override=coding_agent_loop" in params
    if alias not in AGENTIC_LOOP_REASONING_EFFORT:
        assert "co.lo.advanced_mode_reasoning_effort=" not in params


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
    assert args[args.index("--runner-type") + 1] == "GLEAN_CHAT"
    assert "--preset" not in args


def test_create_eval_run_adopts_a_run_its_own_retry_already_inserted():
    """A timed-out `run create` can commit the insert before the retry goes out.

    The retry then fails on the primary key it just took, which means the run exists
    rather than that creation failed, so the ID must be adopted instead of raising.
    """
    client = EvalCliClient(binary="/fake/evalcli")
    duplicate = EvalCliError(
        "evalcli failed (exit 1): /bin/evalcli run create --id run_dup --json\n"
        "stderr: Error: API request failed: 500\n"
        'Response: {"detail":"[run_dup] Failed to insert eval run: (pymysql.err.IntegrityError) '
        "(1062, \\\"Duplicate entry 'run_dup' for key 'PRIMARY'\\\")\"}\n"
        "stdout: "
    )

    with patch.object(client, "_invoke_json", side_effect=duplicate):
        assert (
            client.create_eval_run(
                eval_run_id="run_dup",
                eval_set_name="AI Answers Small",
                eval_set_version="20260403",
                deployment_ids=["scio-prod"],
                description="GEPA eval run for AI Answers Small:20260403",
            )
            == "run_dup"
        )


def test_create_eval_run_still_raises_unrelated_failures():
    client = EvalCliClient(binary="/fake/evalcli")
    with (
        patch.object(client, "_invoke_json", side_effect=EvalCliError("evalcli failed (exit 1): no such eval set")),
        pytest.raises(EvalCliError, match="no such eval set"),
    ):
        client.create_eval_run(
            eval_run_id="run_missing",
            eval_set_name="AI Answers Small",
            eval_set_version="20260403",
            deployment_ids=["scio-prod"],
            description="GEPA eval run for AI Answers Small:20260403",
        )


@pytest.mark.parametrize(
    "cmd_prefix, success, expected, call",
    [
        (
            ("run", "create"),
            {"id": "run_ok"},
            "run_ok",
            lambda client: client.create_eval_run(
                eval_run_id="run_ok",
                eval_set_name="AI Answers Small",
                eval_set_version="20260403",
                deployment_ids=["scio-prod"],
                description="GEPA eval run for AI Answers Small:20260403",
            ),
        ),
        (
            ("evalsets", "entries"),
            {"evalSetEntries": [{"id": "e1"}], "pageInfo": {"totalPages": 1}},
            [{"id": "e1"}],
            lambda client: client.list_eval_set_entries(
                eval_set_name="Glean Chat V2 Medium",
                eval_set_version="20260907",
                deployment_ids=["scio-prod"],
            ),
        ),
    ],
    ids=["create_eval_run", "list_eval_set_entries"],
)
def test_evalcli_retries_opaque_errors(cmd_prefix, success, expected, call):
    client = EvalCliClient(binary="/fake/evalcli")
    attempts = {"n": 0}

    def invoke(*args):
        if args[:2] == cmd_prefix:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise OPAQUE_EVALCLI_ERROR
            return success
        raise AssertionError(args)

    with (
        patch.object(client, "_invoke_json", side_effect=invoke),
        patch("glean_gepa.evalcli_client.time.sleep") as sleep,
    ):
        assert call(client) == expected

    assert attempts["n"] == 2
    sleep.assert_called_once()


def test_evalcli_retrying_raises_on_non_transient_errors():
    client = EvalCliClient(binary="/fake/evalcli")
    with patch.object(client, "_invoke_json", side_effect=EvalCliError("stderr: auth failed\nstdout: ")):
        with pytest.raises(EvalCliError, match="auth failed"):
            client._invoke_json_retrying("run", "create", label="run create")


def test_al_runner_invokes_on_created_before_waiting(tmp_path):
    cache_file = tmp_path / "eval-runs.json"
    client = EvalCliClient(binary="/fake/evalcli")
    runner = ALRunner(evalcli=client, cache_file=str(cache_file))
    events = []

    def on_created(eval_run_id):
        events.append(("created", eval_run_id))
        assert eval_run_id in runner._in_flight
        saved = json.loads(cache_file.read_text())
        assert eval_run_id in saved["in_flight"]
        assert eval_run_id not in saved["completed"].values()

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
    saved = json.loads(cache_file.read_text())
    assert list(saved["completed"].values()) == ["run_123"]
    assert saved["in_flight"] == {}


def test_al_runner_waits_for_an_eval_run_cached_by_an_interrupted_process(tmp_path):
    cache_file = tmp_path / "eval-runs.json"
    client = EvalCliClient(binary="/fake/evalcli")
    with (
        patch.object(client, "create_eval_run", return_value="run_123"),
        patch.object(client, "wait_for_eval_run"),
    ):
        ALRunner(evalcli=client, cache_file=str(cache_file)).run("fast", "prompt", "eval-set", "v1", ["scio-prod"])

    resumed = ALRunner(evalcli=client, cache_file=str(cache_file))
    with (
        patch.object(client, "create_eval_run") as create_eval_run,
        patch.object(client, "wait_for_eval_run") as wait_for_eval_run,
    ):
        assert resumed.run("fast", "prompt", "eval-set", "v1", ["scio-prod"]) == "run_123"

    create_eval_run.assert_not_called()
    wait_for_eval_run.assert_not_called()


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
    assert "list-for-run" not in list_args
    assert list_args[list_args.index("--eval-run-ids") + 1] == "student-run"

    with patch.object(client, "_invoke_json", return_value={"judgeMetrics": {}}) as mock_invoke:
        client.get_eval_metrics("student-run")
    metrics_args = mock_invoke.call_args[0]
    assert metrics_args[0:2] == ("metrics", "summary")
    assert metrics_args[metrics_args.index("--test-eval-id") + 1] == "student-run"


def test_create_judge_run_retries_opaque_error():
    client = EvalCliClient(binary="/fake/evalcli")
    creates = {"n": 0}

    def invoke(*args):
        if args[:2] == ("judge", "create"):
            creates["n"] += 1
            if creates["n"] == 1:
                raise OPAQUE_EVALCLI_ERROR
            return {"id": "judge_ok"}
        if args[:2] == ("judge", "list"):
            return []
        raise AssertionError(args)

    with (
        patch.object(client, "_invoke_json", side_effect=invoke),
        patch("glean_gepa.evalcli_client.time.sleep") as sleep,
    ):
        judge_id = client.create_judge_run(
            eval_run_id="student-run",
            judge_type=COMPLETENESS_JUDGE_TYPE,
            run_params=COMPLETENESS_RUN_PARAMS,
        )

    assert judge_id == "judge_ok"
    assert creates["n"] == 2
    sleep.assert_called_once()


@pytest.mark.parametrize(
    ("create_kwargs", "listing", "expected"),
    [
        (
            {"judge_type": COMPLETENESS_JUDGE_TYPE, "run_params": COMPLETENESS_RUN_PARAMS},
            {"judgeRuns": [{"id": "judge-existing", "config": {"judgeType": "COMPLETENESS"}}]},
            "judge-existing",
        ),
        (
            {
                "judge_type": CORRECTNESS_JUDGE_TYPE,
                "run_params": CORRECTNESS_RUN_PARAMS,
                "base_eval_run_id": "teacher-run",
            },
            {
                "judgeRuns": [
                    {"id": "judge-old-teacher", "evalRunId": "student-run", "n": "other-teacher"},
                    {"id": "judge-wanted", "evalRunId": "student-run", "n": "teacher-run"},
                ]
            },
            "judge-wanted",
        ),
    ],
    ids=["pointwise", "pairwise_baseline"],
)
def test_create_judge_run_reuses_existing_after_opaque_create_error(create_kwargs, listing, expected):
    client = EvalCliClient(binary="/fake/evalcli")

    def invoke(*args):
        if args[:2] == ("judge", "create"):
            raise OPAQUE_EVALCLI_ERROR
        if args[:2] == ("judge", "list"):
            return listing
        raise AssertionError(args)

    with patch.object(client, "_invoke_json", side_effect=invoke):
        judge_id = client.create_judge_run(eval_run_id="student-run", **create_kwargs)

    assert judge_id == expected


def _metrics_payload(
    *,
    pass_rate: float | None,
    sample_size: int | None = None,
    total: int | None = None,
    missing: int | None = None,
    judge_run_id: str = "judge-1",
):
    row: dict[str, object] = {"passRate": pass_rate, "judgeRunId": judge_run_id}
    if sample_size is not None:
        row["sampleSize"] = sample_size
    charts: dict[str, object] = {"COMPLETENESS": row}
    if total is not None:
        charts["totalEntries"] = total
    if missing is not None:
        charts["missingEntries"] = missing
    return {"judgeMetrics": charts}


@pytest.mark.parametrize(
    "payload, expected_rate, coverage_complete",
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
            False,
        ),
        (
            {"judgeMetrics": {"COMPLETENESS": [{"judgeRunId": "judge-1", "test": 0.75, "sampleSize": 4}]}},
            0.75,
            False,
        ),
        (
            {"judgeMetrics": {"COMPLETENESS": {"passRate": None, "judgeRunId": "judge-1"}}},
            None,
            False,
        ),
        (
            {"judgeMetrics": {"COMPLETENESS": {"passRate": 0.6, "judgeRunId": "judge-1"}}},
            0.6,
            False,
        ),
        (_metrics_payload(pass_rate=4.31, sample_size=42, total=104, missing=62), 4.31, False),
        (_metrics_payload(pass_rate=4.51, sample_size=104, total=104, missing=0), 4.51, True),
        (_metrics_payload(pass_rate=4.22, sample_size=104), 4.22, False),
    ],
    ids=["wrapped", "test_key", "null_rate", "pass_rate_only", "partial", "finished", "omitted_totals"],
)
def test_judge_metrics_snapshot_reads_pass_rate(payload, expected_rate, coverage_complete):
    snapshot = judge_metrics_snapshot(payload, judge_type=COMPLETENESS_JUDGE_TYPE, judge_run_id="judge-1")
    assert snapshot.rate == expected_rate
    assert snapshot.coverage_complete is coverage_complete


def test_wait_for_judge_metrics_returns_when_coverage_is_complete():
    ready = type("EvalCli", (), {})()
    ready.get_eval_metrics = lambda _eval_id, **_kwargs: _metrics_payload(
        pass_rate=0.6, sample_size=10, total=10, missing=0
    )
    analysis = wait_for_all_judge_metrics(
        ready,
        (("run-1", COMPLETENESS_JUDGE_TYPE, "judge-1", None),),
        poll_interval_sec=0,
    )[("run-1", COMPLETENESS_JUDGE_TYPE, None)]
    assert analysis.aggregate == 0.6
    assert analysis.per_entry == {}


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"judgeMetrics": {"COMPLETENESS": {"passRate": None}}}, "not ready"),
        (_metrics_payload(pass_rate=4.92, sample_size=13, total=104, missing=91), "13/104 scored"),
    ],
    ids=["null_rate", "partial"],
)
def test_wait_for_judge_metrics_times_out(payload, match):
    stalled = type("EvalCli", (), {})()
    stalled.get_eval_metrics = lambda _eval_id, **_kwargs: payload
    with pytest.raises(EvalCliError, match=match):
        wait_for_all_judge_metrics(
            stalled,
            (("run-1", COMPLETENESS_JUDGE_TYPE, "judge-1", None),),
            poll_interval_sec=0,
            timeout_sec=0,
        )


def test_wait_for_judge_metrics_does_not_return_on_partial_pass_rate():
    evalcli = MagicMock()
    evalcli.get_eval_metrics.side_effect = [
        _metrics_payload(pass_rate=4.92, sample_size=1, total=3, missing=2),
        _metrics_payload(pass_rate=4.77, sample_size=3, total=3, missing=0),
    ]
    evalcli.get_analysis_view.return_value = {
        "entries": [
            {
                "entryId": f"e{i}",
                "evalRunEntries": [
                    {"evalRunId": "run-1", "metadata": {"judgeScores": {"judge-1": 5.0}}},
                ],
            }
            for i in range(3)
        ]
    }

    analysis = wait_for_all_judge_metrics(
        evalcli,
        (("run-1", COMPLETENESS_JUDGE_TYPE, "judge-1", None),),
        poll_interval_sec=0,
        timeout_sec=60,
    )[("run-1", COMPLETENESS_JUDGE_TYPE, None)]

    assert evalcli.get_eval_metrics.call_count == 2
    assert analysis.aggregate == 4.77
    assert len(analysis.per_entry) == 3


def test_wait_for_judge_metrics_waits_for_analysis_view_to_catch_sample_size():
    evalcli = MagicMock()
    evalcli.get_eval_metrics.return_value = _metrics_payload(pass_rate=4.51, sample_size=2, total=2, missing=0)
    evalcli.get_analysis_view.side_effect = [
        {
            "entries": [
                {
                    "entryId": "only-one",
                    "evalRunEntries": [
                        {"evalRunId": "run-1", "metadata": {"judgeScores": {"judge-1": 5.0}}},
                    ],
                }
            ]
        },
        {
            "entries": [
                {
                    "entryId": "one",
                    "evalRunEntries": [
                        {"evalRunId": "run-1", "metadata": {"judgeScores": {"judge-1": 4.0}}},
                    ],
                },
                {
                    "entryId": "two",
                    "evalRunEntries": [
                        {"evalRunId": "run-1", "metadata": {"judgeScores": {"judge-1": 6.0}}},
                    ],
                },
            ]
        },
    ]

    analysis = wait_for_all_judge_metrics(
        evalcli,
        (("run-1", COMPLETENESS_JUDGE_TYPE, "judge-1", None),),
        poll_interval_sec=0,
        timeout_sec=60,
    )[("run-1", COMPLETENESS_JUDGE_TYPE, None)]

    assert evalcli.get_analysis_view.call_count == 2
    assert set(analysis.per_entry) == {"one", "two"}


def _view(eval_id: str, judge_id: str, scores: list[float | None], *, unscored: int = 0) -> dict:
    """Analyze view with one run entry per score; ``unscored`` adds entries the judge has not reached yet."""
    return {
        "entries": [
            {
                "entryId": f"e{i}",
                "evalRunEntries": [
                    {"evalRunId": eval_id, "metadata": {"judgeScores": {judge_id: score}}},
                ],
            }
            for i, score in enumerate([*scores, *([None] * unscored)])
        ]
    }


def _wait_without_totals(evalcli, *, timeout_sec: int = 600):
    """The agentic judge reports no coverage totals, so settling is read from the sample size."""
    with patch("glean_gepa.judge_metrics_util.time.sleep") as sleep:
        result = wait_for_all_judge_metrics(
            evalcli,
            (("run-1", COMPLETENESS_JUDGE_TYPE, "judge-1", None),),
            poll_interval_sec=60,
            timeout_sec=timeout_sec,
            settle_sec=120,
        )[("run-1", COMPLETENESS_JUDGE_TYPE, None)]
    return result, sleep.call_count


@pytest.mark.parametrize(
    ("summary_sizes", "view_sizes", "unscored_of", "expected_n", "expected_sleeps", "expected_mean"),
    [
        # A count frozen at 2 of 195 past the settle window is still queuing; it settles once it reaches coverage.
        ([2] * 6 + [190] * 3, [2] * 6 + [190] * 3, 195, 190, 8, 5.0),
        # A growing sample must not be read as final.
        ([7, 90, 190, 190, 190], [7, 90, 190, 190, 190], None, 190, 4, None),
        # A summary frozen at 17 must not be recorded while the view already has 152: use the per-entry mean.
        ([17], [100, 152, 152, 152], None, 152, None, 5.0),
    ],
    ids=["stall_below_coverage_then_settles", "waits_until_sample_stops_growing", "per_entry_mean_when_summary_lags"],
)
def test_wait_for_judge_metrics_without_totals_settles_on_a_stable_sample(
    summary_sizes, view_sizes, unscored_of, expected_n, expected_sleeps, expected_mean
):
    evalcli = MagicMock()
    payloads = [_metrics_payload(pass_rate=5.0, sample_size=n) for n in summary_sizes]
    evalcli.get_eval_metrics.side_effect = payloads if len(payloads) > 1 else None
    evalcli.get_eval_metrics.return_value = payloads[0]
    evalcli.get_analysis_view.side_effect = [
        _view("run-1", "judge-1", [0.5] * n, unscored=(unscored_of - n) if unscored_of else 0) for n in view_sizes
    ]
    analysis, sleeps = _wait_without_totals(evalcli, timeout_sec=3600)
    assert len(analysis.per_entry) == expected_n
    if expected_sleeps is not None:
        assert sleeps == expected_sleeps
    if expected_mean is not None:
        assert analysis.aggregate == pytest.approx(expected_mean)


def _growing_metrics():
    sizes = iter(range(1, 1000))
    return lambda _eval_id, **_kwargs: _metrics_payload(pass_rate=4.0, sample_size=next(sizes))


@pytest.mark.parametrize(
    ("metrics", "view", "match"),
    [
        (
            _metrics_payload(pass_rate=5.5, sample_size=8),
            _view("run-1", "judge-1", [0.55] * 8, unscored=192),
            "8 scored",
        ),
        # No analyze view means nothing to measure coverage against, so a flat count is not proof.
        (
            _metrics_payload(pass_rate=5.0, sample_size=2),
            EvalCliError("503 Unable to load eval run creation dates"),
            "2 scored",
        ),
        (_growing_metrics(), {"entries": []}, "scored"),
    ],
    ids=["flat_below_coverage", "no_view", "sample_keeps_growing"],
)
def test_wait_for_judge_metrics_without_totals_never_settles(metrics, view, match):
    evalcli = MagicMock()
    if callable(metrics):
        evalcli.get_eval_metrics.side_effect = metrics
    else:
        evalcli.get_eval_metrics.return_value = metrics
    if isinstance(view, Exception):
        evalcli.get_analysis_view.side_effect = view
    else:
        evalcli.get_analysis_view.return_value = view
    with pytest.raises(EvalCliError, match=match):
        _wait_without_totals(evalcli, timeout_sec=180)


def test_get_analysis_details_pages_entry_ids_and_retries_transient_pages():
    client = EvalCliClient(binary="/fake/evalcli")
    entry_ids = [f"entry-{i:02d}" for i in range(23)]
    calls: list[list[str]] = []
    failed_once = {"done": False}

    def fake_invoke(*args):
        ids = list(args[args.index("--entry-ids") + 1 : args.index("--eval-run-ids")])
        calls.append(ids)
        if len(calls) == 2 and not failed_once["done"]:
            failed_once["done"] = True
            raise EvalCliError("evalcli failed (exit 1): evalcli analyze details\nstderr: Error:\nstdout: ")
        return [{"evalSetEntry": {"id": entry_id}} for entry_id in ids]

    with (
        patch.object(client, "_invoke_json", side_effect=fake_invoke),
        patch("glean_gepa.evalcli_client.time.sleep"),
    ):
        details = client.get_analysis_details(
            entry_ids=entry_ids, eval_run_ids=["run-1", "run-2"], deployment_id="scio-prod"
        )

    assert [len(ids) for ids in calls] == [10, 10, 10, 3]
    assert [item["evalSetEntry"]["id"] for item in details] == entry_ids


def test_wait_for_all_judge_metrics_reads_views_only_after_every_child_is_complete():
    evalcli = MagicMock()
    polls = {"child-a": 0, "child-b": 0}
    view_when: list[tuple[str, dict[str, int]]] = []

    def get_metrics(eval_id, **_kwargs):
        polls[eval_id] += 1
        if eval_id == "child-a":
            return _metrics_payload(pass_rate=4.22, sample_size=1, total=1, missing=0, judge_run_id="judge-a")
        if polls[eval_id] == 1:
            return _metrics_payload(pass_rate=4.92, sample_size=1, total=104, missing=91, judge_run_id="judge-b")
        return _metrics_payload(pass_rate=4.77, sample_size=1, total=1, missing=0, judge_run_id="judge-b")

    def get_view(eval_id, **_kwargs):
        view_when.append((eval_id, dict(polls)))
        judge_id = "judge-a" if eval_id == "child-a" else "judge-b"
        return {
            "entries": [
                {
                    "entryId": "e0",
                    "evalRunEntries": [
                        {"evalRunId": eval_id, "metadata": {"judgeScores": {judge_id: 5.0}}},
                    ],
                }
            ]
        }

    evalcli.get_eval_metrics.side_effect = get_metrics
    evalcli.get_analysis_view.side_effect = get_view

    analyses = wait_for_all_judge_metrics(
        evalcli,
        (
            ("child-a", COMPLETENESS_JUDGE_TYPE, "judge-a", "teacher-1"),
            ("child-b", COMPLETENESS_JUDGE_TYPE, "judge-b", "teacher-1"),
        ),
        poll_interval_sec=0,
        timeout_sec=60,
    )

    assert polls == {"child-a": 2, "child-b": 2}
    assert all(counts["child-b"] == 2 for _eval_id, counts in view_when)
    assert analyses[("child-a", COMPLETENESS_JUDGE_TYPE, "teacher-1")].aggregate == 4.22
    assert analyses[("child-b", COMPLETENESS_JUDGE_TYPE, "teacher-1")].aggregate == 4.77


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


def test_compare_eval_metrics_uses_pairwise_compare_command():
    client = EvalCliClient(binary="/fake/evalcli")
    payload = {"systemMetrics": {}, "judgeMetrics": {}}
    with patch.object(client, "_invoke_json", return_value=payload) as mock_invoke:
        result = client.compare_eval_metrics("test-run", "base-run")

    assert result == payload
    assert mock_invoke.call_args[0] == (
        "metrics",
        "compare",
        "--test-eval-id",
        "test-run",
        "--base-eval-id",
        "base-run",
    )


@pytest.mark.parametrize(
    "listing, expected",
    [
        (
            {
                "judgeRuns": [
                    {"id": "judge-other-base", "evalRunId": "eval-best", "n": "eval-base-old"},
                    {"id": "judge-wanted", "evalRunId": "eval-best", "n": "eval-base"},
                ]
            },
            "judge-wanted",
        ),
        (
            {"judgeRuns": [{"id": "judge-1", "evalRunId": "some-other-eval", "n": "eval-best"}]},
            None,
        ),
        (
            {"judgeRuns": [{"id": "judge-1", "evalRunId": "eval-best", "baseEvalRunId": "eval-base"}]},
            "judge-1",
        ),
    ],
    ids=["matches_n", "skips_when_eval_is_base", "canonical_base_field"],
)
def test_find_judge_run_id(listing, expected):
    client = EvalCliClient(binary="/fake/evalcli")
    with patch.object(client, "_invoke_json", return_value=listing):
        found = client.find_judge_run_id("eval-best", judge_type="CORRECTNESS", base_eval_run_id="eval-base")
    assert found == expected


def _judge_row(run_id: str, skill: str, *, as_json: bool = False) -> dict:
    params = {"judge_skill_name": skill}
    return {
        "id": run_id,
        "evalRunId": "ev",
        "baseEvalRunId": "base",
        "config": {"judgeType": AGENTIC_JUDGE_TYPE, "runParameters": json.dumps(params) if as_json else params},
    }


@pytest.mark.parametrize("as_json", [False, True], ids=["dict", "json-string"])
def test_judge_run_skill_name_reads_config_run_parameters(as_json: bool):
    assert judge_run_skill_name(_judge_row("j", "skill", as_json=as_json)) == "skill"
    assert judge_run_skill_name({"id": "j"}) is None


def test_find_judge_run_id_filters_by_skill_when_cortex_type_is_shared():
    client = EvalCliClient(binary="/fake/evalcli")
    listing = {
        "judgeRuns": [
            _judge_row("judge-multi", AGENTIC_JUDGE_NAME),
            _judge_row("judge-corr", AGENTIC_CORRECTNESS_JUDGE_NAME),
        ]
    }
    with patch.object(client, "_invoke_json", return_value=listing):
        find = lambda **kw: client.find_judge_run_id("ev", judge_type=AGENTIC_JUDGE_TYPE, base_eval_run_id="base", **kw)
        assert find(judge_skill_name=AGENTIC_CORRECTNESS_JUDGE_NAME) == "judge-corr"
        assert find(judge_skill_name=AGENTIC_JUDGE_NAME) == "judge-multi"
        assert find() == "judge-multi"  # no skill filter: first type match wins


@pytest.mark.parametrize("spec", [s for s in JUDGE_SPECS.values() if s.per_entry_scale != 1.0], ids=lambda s: s.name)
def test_per_entry_scores_normalize_by_the_spec_scale(spec):
    raw = spec.per_entry_scale * 0.7
    view = {
        "entries": [{"entryId": "e1", "evalRunEntries": [{"runId": "ev", "metadata": {"judgeScores": {"j": raw}}}]}]
    }
    scores, _ = per_entry_from_analysis_view(view, eval_id="ev", judge_run_id="j", judge_type=spec.judge_type)
    assert scores == {"e1": pytest.approx(0.7)}


@pytest.mark.parametrize(
    "listing, match",
    [
        ({"judgeRuns": [{"id": "judge_456", "status": "FAILED"}]}, "ended with status FAILED"),
        (
            {
                "judgeRuns": [
                    {"id": "other_judge", "status": "RUNNING"},
                    {"id": "judge_456", "status": "SUCCEEDED"},
                ]
            },
            None,
        ),
    ],
    ids=["failed", "succeeded"],
)
def test_wait_for_judge_run(listing, match):
    client = EvalCliClient(binary="/fake/evalcli")
    with patch.object(client, "_invoke_json", return_value=listing) as mock_invoke:
        if match:
            with pytest.raises(EvalCliError, match=match):
                client.wait_for_judge_run("judge_456", eval_run_id="student-run", poll_interval_sec=0, timeout_sec=1)
        else:
            client.wait_for_judge_run("judge_456", eval_run_id="student-run", poll_interval_sec=0, timeout_sec=1)

    args = mock_invoke.call_args[0]
    assert args[args.index("--eval-run-ids") + 1] == "student-run"


@pytest.mark.parametrize(
    "error",
    [
        EvalCliError("stderr: Error: API request failed: 502\nResponse: Connection refused"),
        EvalCliError("stderr: Error: Could not find valid __Host-GCP_IAP_AUTH_TOKEN_* cookie in any browser."),
        EvalCliError("evalcli failed (exit 1): evalcli run status\nstderr: Error:\nstdout: "),
        EvalCliError(
            "evalcli failed (exit 1): evalcli run status\n"
            "stderr: Error: [Errno 8] nodename nor servname provided, or not known\nstdout: "
        ),
        EvalCliError("stderr: Error: [Errno -3] Temporary failure in name resolution\nstdout: "),
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
    assert "TASK_SUBMITTED" in status_logs[0]
    assert "TASK_SUCCEEDED" in status_logs[1]


def _task_status(*, finished: int, unfinished: int) -> list[dict[str, object]]:
    counts: list[dict[str, object]] = []
    if finished:
        counts.append({"status": "TASK_SUCCEEDED", "count": finished})
    if unfinished:
        counts.append({"status": "TASK_SUBMITTED", "count": unfinished})
    return [{"taskCountsByStatus": counts}]


def _monotonic_clock(*times: float):
    values = list(times)

    def monotonic() -> float:
        if len(values) > 1:
            return values.pop(0)
        return values[0]

    return monotonic


def test_wait_for_eval_run_grace_period(capsys: pytest.CaptureFixture[str]):
    client = EvalCliClient(binary="/fake/evalcli")
    trailing = _task_status(finished=118, unfinished=2)
    done = _task_status(finished=120, unfinished=0)

    with patch.object(client, "_invoke_json", return_value=trailing) as disabled:
        assert client.wait_for_eval_run("run_123", poll_interval_sec=0, grace_period_sec=0) is trailing
    disabled.assert_called_once()
    assert "usable with 2 unfinished, proceeding" in capsys.readouterr().out

    with (
        patch.object(client, "_invoke_json", side_effect=[trailing, done]) as settled,
        patch("glean_gepa.evalcli_client.time.sleep"),
        patch("glean_gepa.evalcli_client.time.monotonic", side_effect=_monotonic_clock(0.0, 1.0, 2.0)),
    ):
        assert client.wait_for_eval_run("run_123", poll_interval_sec=0) is done
    assert settled.call_count == 2

    with (
        patch.object(client, "_invoke_json", return_value=trailing) as expired,
        patch("glean_gepa.evalcli_client.time.sleep"),
        patch("glean_gepa.evalcli_client.time.monotonic", side_effect=_monotonic_clock(0.0, 0.0, 0.0, 1800.0)),
    ):
        assert client.wait_for_eval_run("run_123", poll_interval_sec=0) is trailing
    assert expired.call_count == 2
    assert "grace period expired with 2 unfinished, proceeding" in capsys.readouterr().out

    with (
        patch.object(client, "_invoke_json", return_value=trailing),
        patch("glean_gepa.evalcli_client.time.sleep"),
        patch("glean_gepa.evalcli_client.time.monotonic", side_effect=_monotonic_clock(0.0, 0.0, 0.0, 0.0, 10.0)),
        pytest.raises(EvalCliError, match="timed out after 10s"),
    ):
        client.wait_for_eval_run("run_123", poll_interval_sec=0, timeout_sec=10)


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
    assert (min_ingested_eval_set_entries(20), min_ingested_eval_set_entries(1)) == (10, 1)  # half, rounded up


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


def test_wait_for_eval_set_entries_includes_listing_error_on_timeout():
    client = EvalCliClient(binary="/fake/evalcli")
    with (
        patch.object(
            client,
            "list_eval_set_entries",
            side_effect=EvalCliError('Eval set version "n:v" not found'),
        ),
        patch("glean_gepa.evalcli_client.time.sleep"),
        pytest.raises(EvalCliError, match="Last listing error"),
    ):
        client.wait_for_eval_set_entries(
            eval_set_name="n",
            eval_set_version="v",
            deployment_ids=["prod"],
            expected_count=2,
            poll_interval_sec=1,
            timeout_sec=1,
        )


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        # No counts: unknown unless the payload explicitly says no such run.
        (None, "unknown"),
        ({"taskCountsByStatus": []}, "unknown"),
        ({"errors": ["backend timeout"]}, "unknown"),
        ({"errors": ["No EvalRun or JudgeRun found with this ID"]}, "missing"),
        ({"taskCountsByStatus": [{"status": "TASK_SUBMITTED", "count": 2}]}, "ongoing"),
        (
            {
                "taskCountsByStatus": [
                    {"status": "TASK_SUCCEEDED", "count": 9},
                    {"status": "TASK_EXECUTING", "count": 1},
                ]
            },
            "ongoing",
        ),
        (
            {
                "taskCountsByStatus": [
                    {"status": "TASK_SUCCEEDED", "count": 193},
                    {"status": "TASK_EXECUTING", "count": 2},
                    {"status": "TASK_FAILED", "count": 5},
                ]
            },
            "usable",
        ),
        (
            {
                "taskCountsByStatus": [
                    {"status": "TASK_SUCCEEDED", "count": 90},
                    {"status": "TASK_FAILED", "count": 1},
                    {"status": "TASK_IN_QUEUE", "count": 9},
                    {"status": "TASK_SUBMITTED", "count": 1},
                ]
            },
            "usable",
        ),
        (
            {
                "taskCountsByStatus": [
                    {"status": "TASK_SUCCEEDED", "count": 90},
                    {"status": "TASK_IN_QUEUE", "count": 10},
                ]
            },
            "ongoing",
        ),
        (
            {
                "taskCountsByStatus": [
                    {"status": "TASK_SUCCEEDED", "count": 85},
                    {"status": "TASK_IN_QUEUE", "count": 5},
                    {"status": "TASK_EXECUTING", "count": 5},
                ]
            },
            "ongoing",
        ),
        (
            {
                "taskCountsByStatus": [
                    {"status": "TASK_SUCCEEDED", "count": 80},
                    {"status": "TASK_EXECUTING", "count": 20},
                ]
            },
            "ongoing",
        ),
        ({"taskCountsByStatus": [{"status": "TASK_SUCCEEDED", "count": 10}]}, "usable"),
        (
            {
                "taskCountsByStatus": [
                    {"status": "TASK_SUCCEEDED", "count": 186},
                    {"status": "TASK_FAILED", "count": 14},
                ]
            },
            "usable",
        ),
        (
            {
                "taskCountsByStatus": [
                    {"status": "TASK_SUCCEEDED", "count": 188},
                    {"status": "TASK_FAILED", "count": 11},
                    {"status": "TASK_TIMED_OUT", "count": 1},
                ]
            },
            "usable",
        ),
        # Cancellation-dominated runs are not reusable: recreate rather than score a stub.
        (
            {
                "taskCountsByStatus": [
                    {"status": "TASK_SUCCEEDED", "count": 77},
                    {"status": "TASK_CANCELLED", "count": 123},
                ]
            },
            "missing",
        ),
        ({"taskCountsByStatus": [{"status": "TASK_CANCELLED", "count": 200}]}, "missing"),
        (
            {
                "taskCountsByStatus": [
                    {"status": "TASK_CANCELLED", "count": 178},
                    {"status": "TASK_SUCCEEDED", "count": 14},
                    {"status": "TASK_EXECUTING", "count": 1},
                ]
            },
            "missing",
        ),
        # A handful of cancelled tasks in an otherwise finished run is still usable.
        (
            {
                "taskCountsByStatus": [
                    {"status": "TASK_SUCCEEDED", "count": 190},
                    {"status": "TASK_CANCELLED", "count": 10},
                ]
            },
            "usable",
        ),
    ],
)
def test_classify_eval_run_status(status, expected):
    assert classify_eval_run_status(status) == expected


KEY = ("gpt", "h", "Set", "v1", "gepa", "scio-prod")
NOT_FOUND = [{"errors": ["No EvalRun or JudgeRun found with this ID"]}]
NEAR_DONE = [
    {"taskCountsByStatus": [{"status": "TASK_SUCCEEDED", "count": 189}, {"status": "TASK_EXECUTING", "count": 4}]}
]


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


@pytest.fixture(autouse=True)
def debug_output_disabled():
    set_debug(False)
    yield
    set_debug(False)


def test_evalset_payload_is_hidden_by_default(capsys):
    client = EvalCliClient(binary="/fake/evalcli")
    with patch.object(client, "_invoke", return_value=""):
        client.upload_eval_set({"entries": [{"query": "sensitive prompt"}]})

    assert capsys.readouterr().out == ""


def test_evalset_payload_is_visible_in_debug_mode(capsys):
    set_debug(True)
    client = EvalCliClient(binary="/fake/evalcli")
    with patch.object(client, "_invoke", return_value=""):
        client.upload_eval_set({"entries": [{"query": "sensitive prompt"}]})

    assert "Uploading eval set payload" in capsys.readouterr().out


def test_bigquery_client_query_returns_dict_rows():
    mock_row = MagicMock()
    mock_row.items.return_value = [("eval_id", "run_123"), ("shell_error_rate", 0.1)]
    mock_result = MagicMock()
    mock_result.__iter__.return_value = iter([mock_row])
    mock_client = MagicMock()
    mock_client.query.return_value.result.return_value = mock_result

    client = BigQueryClient(project_id="scio-apps", client=mock_client)
    rows = client.query("SELECT 1")

    assert rows == [{"eval_id": "run_123", "shell_error_rate": 0.1}]
    mock_client.query.assert_called_once()


def test_bigquery_client_raises_import_error_without_dependency():
    client = BigQueryClient(project_id="scio-apps")
    with patch.dict("sys.modules", {"google.cloud": None, "google.cloud.bigquery": None}):
        client._client = None
        with pytest.raises(BigQueryError, match="google-cloud-bigquery is required"):
            client._get_client()


def test_bigquery_client_wraps_query_failures():
    mock_client = MagicMock()
    mock_client.query.side_effect = RuntimeError("permission denied")
    client = BigQueryClient(project_id="scio-apps", client=mock_client)

    with pytest.raises(BigQueryError, match="permission denied"):
        client.query("SELECT 1")
