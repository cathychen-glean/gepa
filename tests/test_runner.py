from __future__ import annotations
import json
from datetime import date, timedelta
from unittest.mock import MagicMock
import pytest
from gepa.core.data_loader import ListDataLoader
from glean_gepa.evalcli_client import (
    AGENTIC_JUDGE_NAME,
    AGENTIC_JUDGE_TYPE,
    AGENTIC_PREFERENCE_RATE_METRIC,
    CORRECTNESS_JUDGE_TYPE,
)
from glean_gepa.evalset_policy import UnseenEvalSetPolicy
from glean_gepa.judge_metrics_util import DEFAULT_CUSTOMER_VALIDATION_GATES
from glean_gepa.experiment_config import load_experiment_config, runner_arg_defaults
from glean_gepa.prompt import compile_encoded_prompt
from glean_gepa.prompt_constants import (
    CORE_TOOLS,
    CORE_TOOLS_GROUP,
    EXECUTION_DISCIPLINE_KEY,
    FULL_PROMPT_KEY,
    RULES_EXT_KEY,
    WALDO_ROUTING_KEY,
    WALDO_SYSTEM_KEY,
    WALDO_TOOL_USAGE_KEY,
    WRITING_CODE_KEY,
)
from glean_gepa.prompt_targets import compile_prompt_text, stock_text
from glean_gepa.runner import (
    CUSTOMER_DEPLOYMENTS_FILENAME,
    CUSTOMER_EVAL_DEPLOYMENT_IDS,
    GLEAN_CHAT_EVAL_SET_NAME,
    MAX_EVAL_RUN_DEPLOYMENTS,
    SCIO_PROD_DEPLOYMENT_IDS,
    TEACHER_STUDENT_DEPLOYMENT_IDS,
    _load_seed_candidate,
    _make_evalset,
    _parse_args,
    _parse_editable_modules,
    _prestart_first_iteration_training,
    _resolve_customer_deployments,
    _resolve_eval_version_split,
    _seed_for_editable_modules,
    _select_covered_dated_versions,
    _select_recent_train_versions,
    _validate_best_candidate_on_customer_eval,
    _verify_customer_eval_metrics,
)
from glean_gepa.remote_job import build_runner_args
from glean_gepa.run_log import (
    capture_run_log,
    format_child_proposal_report,
    format_eval_entry_report,
    format_high_signal_selection_report,
    format_screening_report,
    selected_entry_ids_from_examples,
)
from glean_gepa.runner import RUN_LOG_FILENAME, _resolve_log_file
from unittest.mock import Mock, patch
from glean_gepa.openai_client import create_qe_openai_client, format_exception_chain, get_perfeval_secret
from glean_gepa.runner import _make_reflection_lm


SEED_BOTH = {"WRITING_CODE": "patterns", "FULL_PROMPT": "PREFIX\n{WRITING_CODE}\nSUFFIX"}
SEED_WITH_RULES_SLOT = {**SEED_BOTH, "WRITING_CODE": "patterns\n{RULES_EXT}"}
_PINNED_EXECUTION_DISCIPLINE = {FULL_PROMPT_KEY: "PREFIX\n### Execution Discipline\n- Answer fast.\nSUFFIX"}


def _coding(candidate: dict[str, str]) -> str:
    return compile_prompt_text("coding_agent_loop_system", candidate)


def _waldo(candidate: dict[str, str]) -> str:
    return compile_prompt_text("waldo_system", candidate)


def test_compile_prompt_modules():
    """Writing Code and Execution Discipline splice into FULL_PROMPT; empty
    discipline falls back so the heading is never bare."""
    assert _coding(SEED_BOTH) == "PREFIX\npatterns\nSUFFIX"

    stock_writing_code = stock_text(WRITING_CODE_KEY) or ""
    custom = _coding({WRITING_CODE_KEY: "CUSTOM_PATTERNS"})
    assert "CUSTOM_PATTERNS" in custom
    assert stock_writing_code not in custom
    assert "{WRITING_CODE}" not in custom

    stock_template = stock_text(FULL_PROMPT_KEY) or ""
    assert "{WRITING_CODE}" in stock_template and "{EXECUTION_DISCIPLINE}" in stock_template
    assert "{RULES_EXT}" in stock_writing_code
    assert _coding({}) == stock_template.replace(
        "{EXECUTION_DISCIPLINE}", stock_text(EXECUTION_DISCIPLINE_KEY) or ""
    ).replace("{WRITING_CODE}", stock_writing_code.replace("{RULES_EXT}\n", ""))
    # A pinned template without WRITING_CODE in the candidate still gets stock Writing Code.
    pinned = _coding({FULL_PROMPT_KEY: "PREFIX\n{WRITING_CODE}\nSUFFIX"})
    assert pinned.startswith("PREFIX\n" + stock_writing_code.split("\n", 1)[0])
    assert "{WRITING_CODE}" not in pinned and "{RULES_EXT}" not in pinned

    discipline = _coding({EXECUTION_DISCIPLINE_KEY: "- Keep working until the deliverable is complete."})
    assert "- Keep working until the deliverable is complete." in discipline
    assert "{EXECUTION_DISCIPLINE}" not in discipline
    for candidate in ({EXECUTION_DISCIPLINE_KEY: ""}, {EXECUTION_DISCIPLINE_KEY: "   \n"}):
        assert (stock_text(EXECUTION_DISCIPLINE_KEY) or "") in _coding(candidate)


def test_compile_splices_rules_ext_after_rules():
    writing = "intro\n**Rules:**\n- stock rule\n{RULES_EXT}\n### Sandbox\n"
    compiled = _coding(
        {
            WRITING_CODE_KEY: writing,
            FULL_PROMPT_KEY: "PREFIX\n{WRITING_CODE}\nSUFFIX",
            RULES_EXT_KEY: "- Prefer Write after retrieving sources.\n- Do not skip Write when the teacher writes.",
        }
    )
    assert compiled == (
        "PREFIX\nintro\n**Rules:**\n- stock rule\n"
        "- Prefer Write after retrieving sources.\n- Do not skip Write when the teacher writes."
        "\n### Sandbox\n\nSUFFIX"
    )
    empty = _coding({WRITING_CODE_KEY: writing, FULL_PROMPT_KEY: "PREFIX\n{WRITING_CODE}\nSUFFIX", RULES_EXT_KEY: ""})
    assert empty == "PREFIX\nintro\n**Rules:**\n- stock rule\n### Sandbox\n\nSUFFIX"

    # A rewritten Writing Code may reflow the slot; the literal token must never ship.
    reflowed = _coding(
        {
            WRITING_CODE_KEY: "intro\n**Rules:**\n- stock rule\n{RULES_EXT}  \n### Sandbox\n",
            FULL_PROMPT_KEY: "PREFIX\n{WRITING_CODE}\nSUFFIX",
            RULES_EXT_KEY: "",
        }
    )
    assert reflowed == empty


def test_seed_for_editable_modules():
    # Seed text for the template the editable module renders into is pinned with it.
    assert _seed_for_editable_modules(SEED_BOTH, [WRITING_CODE_KEY]) == SEED_BOTH

    raw = {**SEED_BOTH, "glean_search": "Search less."}
    core_tool_seed = _seed_for_editable_modules(raw, ["glean_search", "discover"])
    assert core_tool_seed == {"glean_search": "Search less.", "discover": stock_text("discover")}

    rules_seed = _seed_for_editable_modules({**SEED_WITH_RULES_SLOT, RULES_EXT_KEY: ""}, [RULES_EXT_KEY])
    assert rules_seed == {**SEED_WITH_RULES_SLOT, RULES_EXT_KEY: ""}

    defaulted = _seed_for_editable_modules({}, [WRITING_CODE_KEY, RULES_EXT_KEY])
    assert defaulted == {WRITING_CODE_KEY: stock_text(WRITING_CODE_KEY), RULES_EXT_KEY: ""}

    # Modules the seed file does not set compile from stock text, so they are not pinned.
    overridden = _seed_for_editable_modules({RULES_EXT_KEY: "- Prefer Write."}, [RULES_EXT_KEY])
    assert overridden == {RULES_EXT_KEY: "- Prefer Write."}
    assert "- Do not use OCR or image-processing scripts; use available tool_sdk tools.\n- Prefer Write.\n" in (
        _coding(overridden)
    )

    stock = _seed_for_editable_modules({}, [EXECUTION_DISCIPLINE_KEY])
    assert stock == {EXECUTION_DISCIPLINE_KEY: stock_text(EXECUTION_DISCIPLINE_KEY)}


@pytest.mark.parametrize(
    ("raw", "modules", "match"),
    [
        (SEED_BOTH, [RULES_EXT_KEY], "no {RULES_EXT} slot"),
        (_PINNED_EXECUTION_DISCIPLINE, [EXECUTION_DISCIPLINE_KEY], "no {EXECUTION_DISCIPLINE} slot"),
    ],
)
def test_editing_a_module_without_a_slot_is_refused(raw, modules, match):
    """A seed with no slot compiles the module away, so the run would evolve text
    the student never sees. Fail at startup instead of burning the budget."""
    with pytest.raises(SystemExit, match=match):
        _seed_for_editable_modules(raw, modules)


def test_parse_editable_modules():
    assert _parse_editable_modules(f"{WRITING_CODE_KEY},{RULES_EXT_KEY}") == [WRITING_CODE_KEY, RULES_EXT_KEY]
    assert _parse_editable_modules(f"{CORE_TOOLS_GROUP},{RULES_EXT_KEY}") == [*CORE_TOOLS, RULES_EXT_KEY]
    assert _parse_editable_modules(f"{CORE_TOOLS_GROUP},glean_search") == list(CORE_TOOLS)
    assert _parse_editable_modules(WALDO_ROUTING_KEY) == [WALDO_ROUTING_KEY]
    with pytest.raises(SystemExit, match="unknown editable_modules"):
        _parse_editable_modules("GLOBAL_ROLE")
    with pytest.raises(SystemExit, match=rf"{FULL_PROMPT_KEY} is not editable.*{EXECUTION_DISCIPLINE_KEY}"):
        _parse_editable_modules(FULL_PROMPT_KEY)


def test_a_rewritten_waldo_routing_core_is_stripped_and_spliced_before_available_tools():
    rewritten = {WALDO_ROUTING_KEY: "## Core Agent Behavior\nCall discover.  \n"}
    assert "Call discover.\n\n### Available Tools" in _waldo(rewritten)


def test_waldo_sections_must_keep_both_search_tool_conditionals():
    stripped = {WALDO_TOOL_USAGE_KEY: (stock_text(WALDO_TOOL_USAGE_KEY) or "").replace("<<<[[no_search_tools]]", "<<<")}
    with pytest.raises(SystemExit, match=r"seed WALDO_TOOL_USAGE is missing required scio markup: <<<\[\[no_search"):
        _seed_for_editable_modules(stripped, [WALDO_TOOL_USAGE_KEY])


def test_waldo_configs_seed_from_the_scio_template_within_the_cap():
    assert ",llmo.per_prompt_overrides.waldo_system=" in compile_encoded_prompt({WALDO_ROUTING_KEY: "route"})

    for config in ("teacher_student_waldo", "teacher_student_waldo_escalation"):
        assert "seed_candidate" not in runner_arg_defaults(load_experiment_config(config))
    defaults = runner_arg_defaults(load_experiment_config("teacher_student_waldo_escalation"))
    editable = [WALDO_ROUTING_KEY, WALDO_SYSTEM_KEY, WALDO_TOOL_USAGE_KEY]
    assert defaults["editable_modules"] == ",".join(editable)
    seed = _seed_for_editable_modules({}, editable)
    assert sum(len(seed[key]) // 4 for key in editable) < defaults["global_token_cap"]


@pytest.mark.parametrize(
    "raw, match",
    [
        ({"GLOBAL_ROLE": "role", "WRITING_CODE": "code instructions"}, "unknown keys"),
        ({"WRITING_CODE": ["code instructions"]}, "WRITING_CODE must be a string"),
        (["WRITING_CODE"], "seed_candidate must be a JSON object"),
    ],
)
def test_load_seed_candidate_rejects_invalid(tmp_path, raw, match):
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(SystemExit, match=match):
        _load_seed_candidate(path)


_FIXED_TODAY = date(2026, 8, 27)


def _version(days_ago: int) -> str:
    return (_FIXED_TODAY - timedelta(days=days_ago)).strftime("%Y%m%d")


@pytest.fixture
def frozen_today(monkeypatch):
    """Pin today so expected versions cannot drift across a midnight boundary."""

    class _FixedDate(date):
        @classmethod
        def today(cls) -> date:
            return _FIXED_TODAY

    monkeypatch.setattr("glean_gepa.runner.date", _FixedDate)


# Two deployments so a version can be published to one and not the other. Independent of
# MAX_EVAL_RUN_DEPLOYMENTS, which decides how many a real run samples.
_SAMPLED_DEPLOYMENTS = sorted(CUSTOMER_EVAL_DEPLOYMENT_IDS[:2])


def _version_row(version: str, deployment_ids=None, size: int = 200):
    """A row as `evalsets versions` returns it, with per-deployment sizes."""
    ids = list(deployment_ids if deployment_ids is not None else _SAMPLED_DEPLOYMENTS)
    return {
        "name": GLEAN_CHAT_EVAL_SET_NAME,
        "version": version,
        "availableDeploymentIds": ids,
        "perDeploymentMetadata": {deployment: {"size": size} for deployment in ids},
    }


def _auto_selected_split(days_back: int | None) -> tuple[list[str], list[str]]:
    """Resolve the automatic split against six consecutive daily versions."""
    argv = ["--seed_candidate", "seed.json"]
    if days_back is not None:
        argv += ["--eval_version_days_back", str(days_back)]
    evalcli = MagicMock()
    evalcli.list_eval_set_versions.return_value = [_version_row(_version(n)) for n in range(6)]
    return _resolve_eval_version_split(_parse_args(argv), evalcli, _SAMPLED_DEPLOYMENTS)


def _select_versions(rows, *, count: int = 1) -> list[str]:
    return _select_covered_dated_versions(
        rows,
        count=count,
        eval_set_name=GLEAN_CHAT_EVAL_SET_NAME,
        deployment_ids=_SAMPLED_DEPLOYMENTS,
    )


def test_eval_version_days_back_holds_the_train_window_still(frozen_today):
    """Without this the train window tracks the calendar, so a new daily version
    misses the eval-run cache. Customer val always uses the latest available versions."""
    train_today, val_today = _auto_selected_split(0)
    train_shifted, val_shifted = _auto_selected_split(2)

    assert train_today == [_version(n) for n in range(5, -1, -1)]
    assert val_today == [_version(1), _version(0)]
    assert train_shifted == [_version(n) for n in range(5, 1, -1)]
    assert val_shifted == [_version(1), _version(0)]
    assert _auto_selected_split(None) == _auto_selected_split(0)


def test_customer_deployment_sampling(tmp_path):
    """Cortex rejects an eval run with more than five deployments. Re-sampling
    would change the valset identity and miss every cached eval run."""
    sampled = _resolve_customer_deployments(tmp_path / CUSTOMER_DEPLOYMENTS_FILENAME, seed=7)
    assert len(sampled) == MAX_EVAL_RUN_DEPLOYMENTS
    assert len(set(sampled)) == MAX_EVAL_RUN_DEPLOYMENTS
    assert set(sampled) <= set(CUSTOMER_EVAL_DEPLOYMENT_IDS)
    ts = _resolve_customer_deployments(tmp_path / "ts.json", seed=7, pool=TEACHER_STUDENT_DEPLOYMENT_IDS)
    assert set(ts) <= set(TEACHER_STUDENT_DEPLOYMENT_IDS)

    state_file = tmp_path / "resume.json"
    first = _resolve_customer_deployments(state_file, seed=1)
    resumed = _resolve_customer_deployments(state_file, seed=2)
    assert resumed == first

    samples = {tuple(_resolve_customer_deployments(tmp_path / f"sample_{index}.json")) for index in range(25)}
    assert len(samples) > 1


@pytest.mark.parametrize(
    ("saved", "pool"),
    [
        (["bill", "not-a-customer"], None),
        ([], None),
        ("bill", None),
        # A run resumed under teacher-student must not carry forward a Claude-gated deployment.
        (["happyreturns", "pricefx-prod", "thoughtworks"], TEACHER_STUDENT_DEPLOYMENT_IDS),
    ],
)
def test_customer_deployment_state_file_rejects_invalid_content(tmp_path, saved, pool):
    state_file = tmp_path / CUSTOMER_DEPLOYMENTS_FILENAME
    state_file.write_text(json.dumps(saved))
    kwargs = {} if pool is None else {"pool": pool}

    with pytest.raises(SystemExit, match="does not hold a list of customer deployments"):
        _resolve_customer_deployments(state_file, **kwargs)


def test_recent_train_versions_cover_the_lookback_window():
    assert _select_recent_train_versions(
        [{"version": "20260813"}, {"version": "20260820"}, {"version": "20260827"}],
        as_of=date(2026, 8, 27),
        lookback_days=14,
    ) == ["20260813", "20260820", "20260827"]

    with pytest.raises(SystemExit, match="at least one scio-prod"):
        _select_recent_train_versions(
            [{"version": "20260820"}],
            as_of=date(2026, 8, 27),
            lookback_days=1,
        )


def test_auto_val_versions_use_customer_coverage_without_listing_entries(frozen_today):
    """`evalsets versions` already reports per-deployment coverage, so selection
    never touches the entries endpoint, which rejects customer deployments."""
    evalcli = MagicMock()

    def list_versions(*, eval_set_name, deployment_ids):
        if deployment_ids == _SAMPLED_DEPLOYMENTS:
            return [
                _version_row("20260820"),
                _version_row("20260827"),
                _version_row("20260813", deployment_ids=["bill"]),
            ]
        return [_version_row(_version(n), deployment_ids=["scio-prod"]) for n in range(6)]

    evalcli.list_eval_set_versions.side_effect = lambda **kwargs: list_versions(**kwargs)
    _, val_versions = _resolve_eval_version_split(
        _parse_args(["--seed_candidate", "seed.json"]), evalcli, _SAMPLED_DEPLOYMENTS
    )
    assert val_versions == ["20260820", "20260827"]

    listed = MagicMock()
    listed.list_eval_set_versions.return_value = [_version_row("20260820"), _version_row("20260827")]
    args = _parse_args(["--seed_candidate", "seed.json", "--val_eval_version_count", "1"])
    _, newest = _resolve_eval_version_split(args, listed, _SAMPLED_DEPLOYMENTS)
    assert newest == ["20260827"]
    listed.list_eval_set_entries.assert_not_called()


def test_val_version_selection_picks_fully_published_dated_versions():
    """Most dated versions ship to scio-prod only; customer-wide ones land
    roughly weekly. Size 0 is published but empty, the 20260812 failure mode."""
    assert _select_versions(
        [_version_row("20260813"), _version_row("20260827"), _version_row("20260820")],
        count=2,
    ) == ["20260820", "20260827"]

    with pytest.raises(SystemExit, match="is published to every deployment"):
        _select_versions([_version_row("latest")])

    partial = _version_row("20260830")
    partial["availableDeploymentIds"] = [_SAMPLED_DEPLOYMENTS[0]]
    partial["perDeploymentMetadata"] = {_SAMPLED_DEPLOYMENTS[0]: {"size": 200}}
    assert _select_versions([_version_row("20260823"), partial]) == ["20260823"]

    empty = _version_row("20260830")
    empty["perDeploymentMetadata"][_SAMPLED_DEPLOYMENTS[1]] = {"size": 0}
    assert _select_versions([_version_row("20260823"), empty]) == ["20260823"]

    evalcli = MagicMock()
    evalcli.list_eval_set_versions.return_value = [partial]
    args = _parse_args(
        [
            "--seed_candidate",
            "seed.json",
            "--train_eval_versions",
            "20260901",
            "--val_eval_versions",
            "20260830",
        ]
    )
    # TEMP: pinned val versions skip the customer publication check.
    assert _resolve_eval_version_split(args, evalcli, _SAMPLED_DEPLOYMENTS) == (["20260901"], ["20260830"])
    evalcli.list_eval_set_versions.assert_not_called()


def _experiment_with_validation(entries):
    experiment = MagicMock()
    experiment.objective = {"validation": entries}
    return experiment


def _passing_customer_metrics():
    return {
        "systemMetrics": {
            "additional_properties": {
                "COST": {
                    "category": "COST",
                    "metrics": [{"metric": "avg_cost_usd", "base": 0.10, "test": 0.11, "pValue": 0.2}],
                },
                "LOOP_COUNT_PERCENTILE": {
                    "category": "LOOP_COUNT_PERCENTILE",
                    "metrics": [{"metric": "avg_al_loops", "base": 2.0, "test": 2.1, "pValue": 0.3}],
                },
                "TOOL_INVOCATION_RATE": {
                    "category": "TOOL_INVOCATION_RATE",
                    "metrics": [
                        {"metric": "glean_search", "base": 0.5, "test": 0.52, "pValue": 0.4},
                        {"metric": "code_search", "base": 0.1, "test": 0.09, "pValue": 0.5},
                    ],
                },
            }
        },
        "judgeMetrics": {
            "additional_properties": {
                "CORRECTNESS": [{"metric": "CORRECTNESS", "base": 0.79, "test": 0.81, "pValue": 0.2}],
                AGENTIC_JUDGE_NAME: [
                    {
                        "metric": "Preference score (5=tie)",
                        "base": 5.0,
                        "test": 5.4,
                        "judgeType": AGENTIC_JUDGE_NAME,
                    },
                    {
                        "metric": AGENTIC_PREFERENCE_RATE_METRIC,
                        "base": 0.5,
                        "test": 0.54,
                        "judgeType": AGENTIC_JUDGE_NAME,
                    },
                ],
            }
        },
    }


def _patched_customer_metrics(
    *,
    cost_p: float | None = None,
    correctness: float | None = None,
    agentic: float | None = None,
    drop_agentic_rate: bool = False,
    clear_judges: bool = False,
) -> dict:
    metrics = _passing_customer_metrics()
    if cost_p is not None:
        metrics["systemMetrics"]["additional_properties"]["COST"]["metrics"][0]["pValue"] = cost_p
    if correctness is not None:
        metrics["judgeMetrics"]["additional_properties"]["CORRECTNESS"][0]["test"] = correctness
    rows = metrics["judgeMetrics"]["additional_properties"][AGENTIC_JUDGE_NAME]
    if drop_agentic_rate:
        del rows[1]
    elif agentic is not None:
        rows[1]["test"] = agentic
    if clear_judges:
        metrics["judgeMetrics"] = {"additional_properties": {}}
    return metrics


@pytest.mark.parametrize(
    ("validation", "expected"),
    [
        ([], []),
        ([{"metric": "correctness", "min": 0.80}], [CORRECTNESS_JUDGE_TYPE]),
        (
            [{"metric": "correctness", "min": 0.80}, {"metric": "agentic_preference_rate", "min": 0.50}],
            [CORRECTNESS_JUDGE_TYPE, AGENTIC_JUDGE_TYPE],
        ),
    ],
)
def test_customer_eval_starts_only_the_configured_judges(validation, expected):
    evalcli = MagicMock()
    evalcli.compare_eval_metrics.return_value = _passing_customer_metrics()
    runner = MagicMock()
    runner.start.side_effect = [("eval-base", True), ("eval-best", True)]
    runner.ensure_judge_run.side_effect = ["judge-1", "judge-agentic"]

    _validate_best_candidate_on_customer_eval(
        runner=runner,
        student_model="gpt",
        baseline_candidate={"WRITING_CODE": "baseline"},
        best_candidate={"WRITING_CODE": "updated"},
        evalcli=evalcli,
        valset=_make_evalset(["20260908"], deployment_ids=_SAMPLED_DEPLOYMENTS),
        experiment=_experiment_with_validation(validation),
    )

    evalcli.list_eval_set_versions.assert_not_called()
    assert runner.start.call_count == 2
    runner.wait.assert_any_call("eval-base")
    runner.wait.assert_any_call("eval-best")
    evalcli.create_judge_run.assert_not_called()
    evalcli.compare_eval_metrics.assert_called_once_with("eval-best", "eval-base")
    assert [call.kwargs["judge_type"] for call in runner.ensure_judge_run.call_args_list] == expected
    if not expected:
        evalcli.wait_for_judge_run.assert_not_called()


def test_customer_eval_skips_when_best_is_still_the_seed():
    evalcli = MagicMock()
    runner = MagicMock()
    seed = {"WRITING_CODE": "unchanged"}

    _validate_best_candidate_on_customer_eval(
        runner=runner,
        student_model="gpt",
        baseline_candidate=seed,
        best_candidate=dict(seed),
        evalcli=evalcli,
        valset=_make_evalset(["20260908"], deployment_ids=_SAMPLED_DEPLOYMENTS),
        experiment=_experiment_with_validation([{"metric": "agentic_preference_rate", "min": 0.50}]),
    )

    runner.start.assert_not_called()
    runner.ensure_judge_run.assert_not_called()
    evalcli.compare_eval_metrics.assert_not_called()


@pytest.mark.parametrize(
    ("patch", "gates", "error", "contains", "omits"),
    [
        ({"cost_p": 0.001}, None, "COST/avg_cost_usd differs significantly", (), ()),
        ({"correctness": 0.8}, DEFAULT_CUSTOMER_VALIDATION_GATES, "correctness 80.00% is not above 80%", (), ()),
        (
            {"agentic": 0.415},
            DEFAULT_CUSTOMER_VALIDATION_GATES,
            "agentic preference rate 41.50% is below 50%",
            (),
            (),
        ),
        ({"agentic": 0.5}, DEFAULT_CUSTOMER_VALIDATION_GATES, None, ("agentic_preference_rate=50.00%",), ()),
        # Same judge run also reports a 0-10 preference score; a missing rate
        # row must fail closed rather than treat that score as the rate.
        (
            {"drop_agentic_rate": True},
            DEFAULT_CUSTOMER_VALIDATION_GATES,
            "did not include judge_pairwise_agentic_multi_dimension",
            (),
            (),
        ),
        ({}, {"correctness": 0.95}, "correctness 81.00% is not above 95%", (), ()),
        (
            {"clear_judges": True},
            {},
            None,
            ("status=PASS",),
            ("correctness=", "agentic_preference_rate="),
        ),
    ],
)
def test_customer_metric_validation(patch, gates, error, contains, omits):
    metrics = _patched_customer_metrics(**patch)
    kwargs = {} if gates is None else {"gates": gates}

    if error:
        with pytest.raises(SystemExit, match=error):
            _verify_customer_eval_metrics(metrics, **kwargs)
        return
    report = _verify_customer_eval_metrics(metrics, **kwargs)
    for snippet in contains:
        assert snippet in report
    for snippet in omits:
        assert snippet not in report


def _prestart(tmp_path, trainset):
    policy = UnseenEvalSetPolicy(state_file=tmp_path / "schedule.json")
    runner = MagicMock()
    runner.start.side_effect = [("teacher-eval", True), ("student-eval", True)]
    _prestart_first_iteration_training(
        runner=runner,
        policy=policy,
        trainset=trainset,
        seed_candidate={"WRITING_CODE": "seed"},
        student_model="gpt6_luna",
        teacher_model="gpt6_sol",
        run_dir=tmp_path,
    )
    return runner


def test_first_iteration_prestarts_the_first_training_slice_unless_resuming(tmp_path):
    base = {"eval_set_name": "Glean Chat V2 Medium", "deployment_ids": ["scio-prod"], "status": "active"}
    trainset = [{**base, "eval_set_version": "20260905"}, {**base, "eval_set_version": "20260904"}]

    runner = _prestart(tmp_path, trainset)
    teacher_call, student_call = runner.start.call_args_list
    assert teacher_call.args[:5] == (
        "gpt6_sol",
        "<<TEACHER_PROD_PROMPT>>",
        "Glean Chat V2 Medium",
        "20260905",
        ["scio-prod"],
    )
    assert student_call.args[0] == "gpt6_luna"
    assert student_call.args[2:5] == ("Glean Chat V2 Medium", "20260905", ["scio-prod"])
    replayed = UnseenEvalSetPolicy(state_file=tmp_path / "schedule.json")
    assert replayed.take_unseen(ListDataLoader(trainset), purpose="reflection", attempt=0) == [0]

    # A resumed run already has state; do not restart the slice or touch the schedule.
    resumed = tmp_path / "resumed"
    resumed.mkdir()
    (resumed / "gepa_state.bin").write_bytes(b"state")
    assert _prestart(resumed, trainset).start.call_count == 0
    assert not (resumed / "schedule.json").exists()


def test_make_evalset():
    """Customer eval-set entries are PII-gated, so validation items must never
    reach the focused high-signal path that lists them."""
    valset = _make_evalset(["20260908"], deployment_ids=_SAMPLED_DEPLOYMENTS, validation_only=True)
    trainset = _make_evalset(["20260901"])

    assert valset[0]["deployment_ids"] == _SAMPLED_DEPLOYMENTS
    assert valset[0]["validation_only"] is True
    assert "validation_only" not in trainset[0]
    assert trainset[0]["deployment_ids"] == list(SCIO_PROD_DEPLOYMENT_IDS)


def test_build_runner_args_uses_cloud_run_execution_and_writes_seed(tmp_path):
    args, run_dir = build_runner_args(
        {
            "CLOUD_RUN_EXECUTION": "gepa-optimize-abc123",
            "GEPA_RUN_ROOT": str(tmp_path),
            "GEPA_RUNNER_ARGS_JSON": json.dumps(["--max_metric_calls", "5"]),
            "GEPA_SEED_CANDIDATE_JSON": json.dumps({"WRITING_CODE": "seed"}),
        }
    )

    assert run_dir == tmp_path / "gepa-optimize-abc123"
    assert args[-4:-2] == ["--run_dir", str(run_dir)]
    assert args[-2:] == ["--seed_candidate", str(run_dir / "seed_candidate.json")]
    assert json.loads((run_dir / "seed_candidate.json").read_text()) == {"WRITING_CODE": "seed"}


def test_build_runner_args_preserves_explicit_run_dir(tmp_path):
    explicit = tmp_path / "explicit"
    args, _run_dir = build_runner_args(
        {
            "GEPA_RUN_ROOT": str(tmp_path),
            "GEPA_RUNNER_ARGS_JSON": json.dumps(["--fake_flow", f"--run_dir={explicit}"]),
        }
    )

    assert args == ["--fake_flow", f"--run_dir={explicit}"]


@pytest.mark.parametrize("value", ["{}", "[1]", "not-json"])
def test_build_runner_args_rejects_invalid_json(value, tmp_path):
    with pytest.raises(ValueError, match="JSON array of strings"):
        build_runner_args({"GEPA_RUN_ROOT": str(tmp_path), "GEPA_RUNNER_ARGS_JSON": value})


def test_format_run_log_reports():
    trajectories = [
        {
            "data": {"eval_set_name": "Chat", "eval_set_version": "v1"},
            "output": {
                "entry_id": "e1",
                "query": "Find the Q3 plan",
                "teacher_tool_events": ["Shell", "Glean Search", "Glean Document Reader"],
                "student_tool_events": ["Discover"],
                "teacher_eval_run_id": "t1",
                "student_eval_run_id": "s1",
            },
            "score": 0.0,
            "objective_scores": {"tool_alignment": 0.0, "correctness": 0.8},
        }
    ]
    report = format_eval_entry_report(trajectories)
    assert "e1" in report and "Glean Search" in report and "Discover" in report and "mismatch" in report

    high_signal = format_high_signal_selection_report(
        selected_groups=[("Glean Search", "Discover", 12), ("Glean Document Reader", "todo_write", 8)],
        selected_entry_ids=["e1", "e2"],
        selected_count=20,
        total_mismatch_count=31,
        module_entry_ids={"glean_search": ["e1"], "WRITING_CODE": ["e1", "e2"]},
    )
    assert "20" in high_signal and "31" in high_signal and "e1, e2" in high_signal and "glean_search" in high_signal

    child = format_child_proposal_report(
        parent_id="parent",
        child_id="child",
        module="glean_search",
        delta="- old\n+ new\n",
        justification="WHY: student used Discover first.",
    )
    assert "glean_search" in child and "student used Discover first." in child and "+ new" in child
    screening = format_screening_report(
        mode="fix-rate",
        entry_ids=["e1", "e2"],
        rows=[("child", 0.4, True, "fix_rate=0.400")],
    )
    assert "PASS" in screening
    assert "e1, e2" in screening

    examples = [
        {"Inputs": {"entry_id": "e1"}},
        {"Inputs": {"entry_id": "e1"}},
        {"Inputs": {"entry_id": "e2"}},
    ]
    assert selected_entry_ids_from_examples(examples) == ["e1", "e2"]


def test_capture_run_log_and_default_path(tmp_path, capsys):
    log_path = tmp_path / "gepa_run.log"
    with capture_run_log(log_path):
        print("hello-run-log")
    captured = capsys.readouterr()
    assert "hello-run-log" in captured.out
    assert "hello-run-log" in log_path.read_text()

    args = _parse_args(["--seed_candidate", "seed.json", "--run_dir", "run_ts8"])
    assert args.log_file is None
    assert _resolve_log_file(args) == args.run_dir / RUN_LOG_FILENAME
    explicit = _parse_args(["--seed_candidate", "seed.json", "--log_file", "/tmp/custom.log"])
    assert _resolve_log_file(explicit).as_posix() == "/tmp/custom.log"


def test_create_qe_openai_client_uses_instance_hostname() -> None:
    ssl_context = Mock()
    http_client = Mock()
    with (
        patch("glean_gepa.openai_client.truststore.SSLContext", return_value=ssl_context),
        patch("glean_gepa.openai_client.openai.DefaultHttpxClient", return_value=http_client) as http_client_cls,
        patch("glean_gepa.openai_client.openai.OpenAI") as openai_cls,
    ):
        create_qe_openai_client("glean-dev")

    http_client_cls.assert_called_once_with(verify=ssl_context)
    openai_cls.assert_called_once_with(
        base_url="https://glean-dev-be.glean.com/qe/llm",
        api_key="dummy",
        timeout=600.0,
        max_retries=5,
        http_client=http_client,
    )


def test_format_exception_chain_includes_transport_root_cause() -> None:
    root = OSError("temporary DNS failure")
    outer = RuntimeError("Connection error")
    outer.__cause__ = root

    assert format_exception_chain(outer) == ("RuntimeError: Connection error <- OSError: temporary DNS failure")


def test_reflection_lm_uses_qe_responses_auth_body(capsys: pytest.CaptureFixture[str]) -> None:
    response = Mock(output_text="ack")
    client = Mock()
    client.responses.create.return_value = response

    with patch("glean_gepa.runner.create_qe_openai_client", return_value=client):
        reflection_lm = _make_reflection_lm(
            "OPEN_AI:GPT5_LATEST",
            qe_project="dev-sandbox-334901",
            qe_instance="glean-dev",
            authenticated_email="cathy.chen@glean.com",
        )

    assert reflection_lm("Just say ack") == "ack"
    assert capsys.readouterr().out == (
        "QE reflection LLM call 1: requesting model=OPEN_AI:GPT5_LATEST, prompt_chars=12\n"
        "QE reflection LLM call 1 prompt:\n"
        "Just say ack\n"
        "QE reflection LLM call 1: received response_chars=3\n"
        "QE reflection LLM call 1 response:\n"
        "ack\n"
    )
    client.responses.create.assert_called_once_with(
        model="OPEN_AI:GPT5_LATEST",
        input="Just say ack",
        max_output_tokens=4096,
        extra_body={
            "perf_eval_secret": get_perfeval_secret("dev-sandbox-334901"),
            "source_info": {
                "clientInitiator": "USER",
                "feature": "INTEGRATION_TEST",
            },
            "authenticated_email": "cathy.chen@glean.com",
        },
    )


def test_reflection_lm_raises_for_an_empty_qe_response() -> None:
    client = Mock()
    client.responses.create.return_value = Mock(output_text="   ")

    with patch("glean_gepa.runner.create_qe_openai_client", return_value=client):
        reflection_lm = _make_reflection_lm(
            "OPEN_AI:GPT5_LATEST",
            qe_project="dev-sandbox-334901",
            qe_instance="glean-dev",
            authenticated_email="cathy.chen@glean.com",
        )

    with pytest.raises(RuntimeError, match="QE reflection LLM returned an empty response"):
        reflection_lm("Just say ack")


def test_reflection_lm_surfaces_qe_request_errors() -> None:
    client = Mock()
    client.responses.create.side_effect = ConnectionError("proxy unavailable")

    with patch("glean_gepa.runner.create_qe_openai_client", return_value=client):
        reflection_lm = _make_reflection_lm(
            "OPEN_AI:GPT5_LATEST",
            qe_project="dev-sandbox-334901",
            qe_instance="glean-dev",
            authenticated_email="cathy.chen@glean.com",
        )

    with pytest.raises(RuntimeError, match="ConnectionError: proxy unavailable"):
        reflection_lm("Just say ack")
