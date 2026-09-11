import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from glean_gepa.prompt import candidate_module_names, compile_system_prompt, materialize_system_prompt
from glean_gepa.prompt_constants import (
    CORE_TOOLS,
    CORE_TOOLS_GROUP,
    DEFAULT_FULL_PROMPT,
    DEFAULT_WRITING_CODE,
    FULL_PROMPT_KEY,
    PROMPT_MODULE_DEFAULTS,
    RULES_EXT_KEY,
    WRITING_CODE_KEY,
)
from glean_gepa.runner import (
    ADAPTER_CACHE_FILENAME,
    CACHE_DIRECTORY_NAME,
    CUSTOMER_DEPLOYMENTS_FILENAME,
    CUSTOMER_EVAL_DEPLOYMENT_IDS,
    GLEAN_CHAT_EVAL_SET_NAME,
    MAX_EVAL_RUN_DEPLOYMENTS,
    SCIO_PROD_DEPLOYMENT_IDS,
    _default_cache_file,
    _load_seed_candidate,
    _make_evalset,
    _parse_args,
    _parse_editable_modules,
    _resolve_customer_deployments,
    _resolve_eval_version_split,
    _seed_for_editable_modules,
    _select_covered_dated_versions,
    _select_recent_train_versions,
    _validate_best_candidate_on_customer_eval,
    _verify_customer_eval_metrics,
)

SEED_BOTH = {"WRITING_CODE": "patterns", "FULL_PROMPT": "PREFIX\n{WRITING_CODE}\nSUFFIX"}


def test_compile_and_materialize_splice_writing_code_when_present():
    assert compile_system_prompt(SEED_BOTH) == "PREFIX\npatterns\nSUFFIX"
    assert materialize_system_prompt(SEED_BOTH) == "PREFIX\npatterns\nSUFFIX"

    stock = compile_system_prompt({WRITING_CODE_KEY: "CUSTOM_PATTERNS"})
    assert "CUSTOM_PATTERNS" in stock
    assert DEFAULT_WRITING_CODE not in stock
    assert "{WRITING_CODE}" not in stock
    assert "## Writing Code" in stock


def test_compile_system_prompt_leaves_writing_code_slot_when_key_absent():
    assert compile_system_prompt({}) == DEFAULT_FULL_PROMPT
    assert (
        compile_system_prompt({FULL_PROMPT_KEY: "PREFIX\n{WRITING_CODE}\nSUFFIX"}) == "PREFIX\n{WRITING_CODE}\nSUFFIX"
    )


def test_materialize_system_prompt_fills_defaults_when_modules_missing():
    prompt = materialize_system_prompt({})

    assert prompt == DEFAULT_FULL_PROMPT.replace("{WRITING_CODE}", DEFAULT_WRITING_CODE)
    assert "{WRITING_CODE}" not in prompt
    assert "{RULES_EXT}" in prompt
    assert "**Rules:**" in prompt


def test_compile_system_prompt_splices_rules_ext_after_rules():
    writing = "intro\n**Rules:**\n- stock rule\n{RULES_EXT}\n### Sandbox\n"
    compiled = compile_system_prompt(
        {
            WRITING_CODE_KEY: writing,
            FULL_PROMPT_KEY: "PREFIX\n{WRITING_CODE}\nSUFFIX",
            RULES_EXT_KEY: "- Prefer Write after retrieving sources.\n- Do not skip Write when the teacher writes.",
        }
    )
    assert "{RULES_EXT}" not in compiled
    assert compiled == (
        "PREFIX\nintro\n**Rules:**\n- stock rule\n"
        "- Prefer Write after retrieving sources.\n- Do not skip Write when the teacher writes."
        "\n### Sandbox\n\nSUFFIX"
    )
    empty = compile_system_prompt(
        {WRITING_CODE_KEY: writing, FULL_PROMPT_KEY: "PREFIX\n{WRITING_CODE}\nSUFFIX", RULES_EXT_KEY: ""}
    )
    assert empty == "PREFIX\nintro\n**Rules:**\n- stock rule\n\n### Sandbox\n\nSUFFIX"


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"WRITING_CODE": "code instructions"},
        {"FULL_PROMPT": "full template"},
        {"WRITING_CODE": "patterns", "FULL_PROMPT": "template"},
        {"glean_search": "Search less."},
        {RULES_EXT_KEY: "- Prefer Write after retrieving sources."},
    ],
)
def test_load_seed_candidate_accepts_known_keys(tmp_path, raw):
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(raw))

    assert _load_seed_candidate(path) == raw


def test_seed_for_editable_modules():
    assert _seed_for_editable_modules(SEED_BOTH, [FULL_PROMPT_KEY]) == {FULL_PROMPT_KEY: "PREFIX\npatterns\nSUFFIX"}
    assert _seed_for_editable_modules(SEED_BOTH, [WRITING_CODE_KEY]) == {WRITING_CODE_KEY: "patterns"}

    raw = {**SEED_BOTH, "glean_search": "Search less."}
    seed = _seed_for_editable_modules(raw, [FULL_PROMPT_KEY])
    assert seed == {FULL_PROMPT_KEY: "PREFIX\npatterns\nSUFFIX"}

    frozen = _seed_for_editable_modules(SEED_BOTH, [])
    assert frozen[FULL_PROMPT_KEY] == "PREFIX\npatterns\nSUFFIX"
    assert WRITING_CODE_KEY not in frozen
    assert "glean_search" not in frozen

    core_tool_seed = _seed_for_editable_modules(raw, ["glean_search", "discover"])
    assert core_tool_seed["glean_search"] == "Search less."
    assert core_tool_seed["discover"] == PROMPT_MODULE_DEFAULTS["discover"]
    assert core_tool_seed[FULL_PROMPT_KEY] == "PREFIX\npatterns\nSUFFIX"

    rules_seed = _seed_for_editable_modules({**SEED_BOTH, RULES_EXT_KEY: ""}, [RULES_EXT_KEY])
    assert rules_seed[RULES_EXT_KEY] == ""
    assert rules_seed[FULL_PROMPT_KEY] == "PREFIX\npatterns\nSUFFIX"

    defaulted = _seed_for_editable_modules({}, [WRITING_CODE_KEY, RULES_EXT_KEY])
    assert defaulted[WRITING_CODE_KEY] == DEFAULT_WRITING_CODE
    assert defaulted[RULES_EXT_KEY] == ""
    assert FULL_PROMPT_KEY not in defaulted

    overridden = _seed_for_editable_modules({RULES_EXT_KEY: "- Prefer Write."}, [RULES_EXT_KEY])
    assert overridden[RULES_EXT_KEY] == "- Prefer Write."
    assert overridden[FULL_PROMPT_KEY] == materialize_system_prompt({})


def test_writing_code_only_does_not_expand_core_tools():
    editable_modules = [WRITING_CODE_KEY]

    assert candidate_module_names(editable_modules) == editable_modules
    assert _seed_for_editable_modules(SEED_BOTH, editable_modules) == {WRITING_CODE_KEY: "patterns"}


def test_parse_editable_modules():
    assert _parse_editable_modules("FULL_PROMPT") == [FULL_PROMPT_KEY]
    assert _parse_editable_modules("WRITING_CODE,FULL_PROMPT") == [WRITING_CODE_KEY, FULL_PROMPT_KEY]
    assert _parse_editable_modules("glean_search") == ["glean_search"]
    assert _parse_editable_modules(RULES_EXT_KEY) == [RULES_EXT_KEY]
    assert _parse_editable_modules(f"{CORE_TOOLS_GROUP},{RULES_EXT_KEY}") == [*CORE_TOOLS, RULES_EXT_KEY]
    assert _parse_editable_modules(CORE_TOOLS_GROUP) == list(CORE_TOOLS)
    assert _parse_editable_modules(f"FULL_PROMPT,{CORE_TOOLS_GROUP}") == [FULL_PROMPT_KEY, *CORE_TOOLS]
    assert _parse_editable_modules(f"{CORE_TOOLS_GROUP},glean_search") == list(CORE_TOOLS)
    with pytest.raises(SystemExit, match="unknown editable_modules"):
        _parse_editable_modules("GLOBAL_ROLE")


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


def test_committed_seed_candidate_pins_only_writing_code():
    """The single seed both configs point at: it overrides WRITING_CODE and leaves
    every other module to PROMPT_MODULE_DEFAULTS."""
    raw = _load_seed_candidate(Path(__file__).resolve().parents[1] / "data" / "seed_candidate.json")

    assert set(raw) == {WRITING_CODE_KEY}
    assert _seed_for_editable_modules(raw, [WRITING_CODE_KEY]) == {WRITING_CODE_KEY: raw[WRITING_CODE_KEY]}
    assert _seed_for_editable_modules(raw, [RULES_EXT_KEY])[RULES_EXT_KEY] == PROMPT_MODULE_DEFAULTS[RULES_EXT_KEY]


def test_parse_args_defaults_editable_modules_to_writing_code():
    args = _parse_args(["--seed_candidate", "seed.json"])

    assert args.editable_modules == WRITING_CODE_KEY


def test_parse_args_accepts_all_reflection_samples_and_hamming_k():
    args = _parse_args(
        [
            "--seed_candidate",
            "seed.json",
            "--reflection_samples",
            "all",
            "--reflection_hamming_distance_k",
            "10",
        ]
    )

    assert args.reflection_samples is None
    assert args.reflection_hamming_distance_k == 10


def test_default_cache_files_live_in_run_cache_directory(tmp_path):
    assert _default_cache_file(tmp_path, ADAPTER_CACHE_FILENAME) == (
        tmp_path / CACHE_DIRECTORY_NAME / ADAPTER_CACHE_FILENAME
    )


def test_default_cache_file_moves_legacy_root_cache_on_resume(tmp_path):
    legacy = tmp_path / ADAPTER_CACHE_FILENAME
    legacy.write_text('{"cached": true}')

    cache_file = _default_cache_file(tmp_path, ADAPTER_CACHE_FILENAME)

    assert cache_file == tmp_path / CACHE_DIRECTORY_NAME / ADAPTER_CACHE_FILENAME
    assert cache_file.read_text() == '{"cached": true}'
    assert not legacy.exists()


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_parse_args_rejects_invalid_reflection_sample_count(value):
    with pytest.raises(SystemExit):
        _parse_args(["--seed_candidate", "seed.json", "--reflection_samples", value])


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


_SAMPLED_DEPLOYMENTS = sorted(CUSTOMER_EVAL_DEPLOYMENT_IDS[:MAX_EVAL_RUN_DEPLOYMENTS])


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


def test_eval_version_days_back_holds_the_train_window_still(frozen_today):
    """Without this the train window tracks the calendar, so a new daily version
    misses the eval-run cache. Customer val always uses the latest available versions."""
    train_today, val_today = _auto_selected_split(0)
    train_shifted, val_shifted = _auto_selected_split(2)

    assert train_today == [_version(n) for n in range(5, -1, -1)]
    assert val_today == [_version(1), _version(0)]
    assert train_shifted == [_version(n) for n in range(5, 1, -1)]
    assert val_shifted == [_version(1), _version(0)]
    # The shifted train window excludes everything newer than the as-of date.
    assert _version(0) not in train_shifted and _version(1) not in train_shifted


def test_eval_version_days_back_defaults_to_today(frozen_today):
    assert _auto_selected_split(None) == _auto_selected_split(0)
    assert _parse_args(["--seed_candidate", "seed.json"]).eval_version_days_back == 0


def test_eval_version_days_back_rejects_negative_values():
    with pytest.raises(SystemExit):
        _parse_args(["--seed_candidate", "seed.json", "--eval_version_days_back", "-1"])


def test_customer_deployments_sample_within_the_eval_run_limit(tmp_path):
    """Cortex rejects an eval run with more than five deployments."""
    sampled = _resolve_customer_deployments(tmp_path / CUSTOMER_DEPLOYMENTS_FILENAME, seed=7)

    assert len(sampled) == MAX_EVAL_RUN_DEPLOYMENTS
    assert len(set(sampled)) == MAX_EVAL_RUN_DEPLOYMENTS
    assert set(sampled) <= set(CUSTOMER_EVAL_DEPLOYMENT_IDS)


def test_customer_deployment_sample_is_reused_on_resume(tmp_path):
    """Re-sampling would change the valset identity and miss every cached eval run."""
    state_file = tmp_path / CUSTOMER_DEPLOYMENTS_FILENAME
    first = _resolve_customer_deployments(state_file, seed=1)
    resumed = _resolve_customer_deployments(state_file, seed=2)

    assert resumed == first
    assert json.loads(state_file.read_text()) == first


def test_customer_deployment_sample_varies_without_a_seed(tmp_path):
    samples = {tuple(_resolve_customer_deployments(tmp_path / f"sample_{index}.json")) for index in range(25)}

    assert len(samples) > 1


@pytest.mark.parametrize("saved", [["bill", "not-a-customer"], [], {"bill": True}, "bill"])
def test_customer_deployment_state_file_rejects_invalid_content(tmp_path, saved):
    state_file = tmp_path / CUSTOMER_DEPLOYMENTS_FILENAME
    state_file.write_text(json.dumps(saved))

    with pytest.raises(SystemExit, match="does not hold a list of customer deployments"):
        _resolve_customer_deployments(state_file)


def test_recent_train_versions_include_the_full_lookback_window():
    train_versions = _select_recent_train_versions(
        [{"version": "20260813"}, {"version": "20260820"}, {"version": "20260827"}],
        as_of=date(2026, 8, 27),
        lookback_days=14,
    )

    assert train_versions == ["20260813", "20260820", "20260827"]


def test_recent_train_versions_fail_when_the_window_is_empty():
    with pytest.raises(SystemExit, match="at least one scio-prod"):
        _select_recent_train_versions(
            [{"version": "20260820"}],
            as_of=date(2026, 8, 27),
            lookback_days=1,
        )


def test_auto_val_versions_come_from_customer_deployments(frozen_today):
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
    train_versions, val_versions = _resolve_eval_version_split(
        _parse_args(["--seed_candidate", "seed.json"]), evalcli, _SAMPLED_DEPLOYMENTS
    )

    assert val_versions == ["20260820", "20260827"]
    assert train_versions[-1] == _version(0)
    evalcli.list_eval_set_versions.assert_any_call(
        eval_set_name=GLEAN_CHAT_EVAL_SET_NAME,
        deployment_ids=_SAMPLED_DEPLOYMENTS,
    )


def test_val_version_selection_never_lists_pii_gated_entries():
    """`evalsets versions` already reports per-deployment coverage, so selection
    never touches the entries endpoint, which rejects customer deployments."""
    evalcli = MagicMock()
    evalcli.list_eval_set_versions.return_value = [_version_row("20260820"), _version_row("20260827")]
    args = _parse_args(["--seed_candidate", "seed.json", "--val_eval_version_count", "1"])

    _, val_versions = _resolve_eval_version_split(args, evalcli, _SAMPLED_DEPLOYMENTS)

    assert val_versions == ["20260827"]
    evalcli.list_eval_set_entries.assert_not_called()


def test_val_version_selection_takes_the_newest_dated_versions():
    selected = _select_covered_dated_versions(
        [_version_row("20260813"), _version_row("20260827"), _version_row("20260820")],
        count=2,
        eval_set_name=GLEAN_CHAT_EVAL_SET_NAME,
        deployment_ids=_SAMPLED_DEPLOYMENTS,
    )

    assert selected == ["20260820", "20260827"]


def test_val_version_selection_ignores_undated_versions():
    with pytest.raises(SystemExit, match="is published to every deployment"):
        _select_covered_dated_versions(
            [_version_row("latest")],
            count=1,
            eval_set_name=GLEAN_CHAT_EVAL_SET_NAME,
            deployment_ids=_SAMPLED_DEPLOYMENTS,
        )


def test_val_version_selection_skips_versions_missing_a_sampled_deployment():
    """Most dated versions ship to scio-prod only; customer-wide ones land
    roughly weekly, so selection must walk back to a fully published one."""
    partial = _version_row("20260830")
    partial["availableDeploymentIds"] = [_SAMPLED_DEPLOYMENTS[0]]
    partial["perDeploymentMetadata"] = {_SAMPLED_DEPLOYMENTS[0]: {"size": 200}}

    selected = _select_covered_dated_versions(
        [_version_row("20260823"), partial],
        count=1,
        eval_set_name=GLEAN_CHAT_EVAL_SET_NAME,
        deployment_ids=_SAMPLED_DEPLOYMENTS,
    )

    assert selected == ["20260823"]


def test_val_version_selection_skips_deployments_with_zero_entries():
    """A published deployment with size 0 has no entries, so the eval run would
    fail exactly the way it did for version 20260812."""
    empty = _version_row("20260830")
    empty["perDeploymentMetadata"][_SAMPLED_DEPLOYMENTS[1]] = {"size": 0}

    selected = _select_covered_dated_versions(
        [_version_row("20260823"), empty],
        count=1,
        eval_set_name=GLEAN_CHAT_EVAL_SET_NAME,
        deployment_ids=_SAMPLED_DEPLOYMENTS,
    )

    assert selected == ["20260823"]


def test_pinned_val_versions_must_be_published_to_every_sampled_deployment():
    partial = _version_row("20260830")
    partial["availableDeploymentIds"] = [_SAMPLED_DEPLOYMENTS[0]]
    partial["perDeploymentMetadata"] = {_SAMPLED_DEPLOYMENTS[0]: {"size": 200}}
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

    with pytest.raises(SystemExit, match="not published to"):
        _resolve_eval_version_split(args, evalcli, _SAMPLED_DEPLOYMENTS)


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
                "CORRECTNESS": [{"metric": "CORRECTNESS", "base": 0.79, "test": 0.81, "pValue": 0.2}]
            }
        },
    }


def test_customer_eval_reuses_the_gepa_valset_for_paired_validation():
    evalcli = MagicMock()
    evalcli.compare_eval_metrics.return_value = _passing_customer_metrics()
    runner = MagicMock()
    runner.start.side_effect = [("eval-base", True), ("eval-best", True)]
    runner.ensure_judge_run.return_value = "judge-1"
    baseline = {"WRITING_CODE": "baseline"}
    best = {"WRITING_CODE": "updated"}
    valset = _make_evalset(["20260908"], deployment_ids=_SAMPLED_DEPLOYMENTS)

    _validate_best_candidate_on_customer_eval(
        runner=runner,
        student_model="gpt",
        baseline_candidate=baseline,
        best_candidate=best,
        evalcli=evalcli,
        valset=valset,
    )

    evalcli.list_eval_set_versions.assert_not_called()
    assert runner.start.call_count == 2
    runner.wait.assert_any_call("eval-base")
    runner.wait.assert_any_call("eval-best")
    # The judge must go through the runner's cache, not straight to create.
    evalcli.create_judge_run.assert_not_called()
    runner.ensure_judge_run.assert_called_once()
    judge_kwargs = runner.ensure_judge_run.call_args.kwargs
    assert judge_kwargs["eval_run_id"] == "eval-best"
    assert judge_kwargs["base_eval_run_id"] == "eval-base"
    evalcli.wait_for_judge_run.assert_called_once_with("judge-1", eval_run_id="eval-best")
    evalcli.compare_eval_metrics.assert_called_once_with("eval-best", "eval-base")
    start_kwargs = runner.start.call_args_list[0].kwargs
    assert start_kwargs["eval_set_version"] == "20260908"
    assert start_kwargs["deployment_ids"] == _SAMPLED_DEPLOYMENTS
    assert len(start_kwargs["deployment_ids"]) <= MAX_EVAL_RUN_DEPLOYMENTS


def test_customer_metric_validation_rejects_significant_system_change():
    metrics = _passing_customer_metrics()
    metrics["systemMetrics"]["additional_properties"]["COST"]["metrics"][0]["pValue"] = 0.001

    with pytest.raises(SystemExit, match="COST/avg_cost_usd differs significantly"):
        _verify_customer_eval_metrics(metrics)


def test_customer_metric_validation_requires_correctness_above_80_percent():
    metrics = _passing_customer_metrics()
    metrics["judgeMetrics"]["additional_properties"]["CORRECTNESS"][0]["test"] = 0.8

    with pytest.raises(SystemExit, match="correctness 80.00% is not above 80%"):
        _verify_customer_eval_metrics(metrics)


def test_make_customer_evalset_uses_the_sampled_customer_deployments():
    assert _make_evalset(["20260908"], deployment_ids=_SAMPLED_DEPLOYMENTS) == [
        {
            "eval_set_name": GLEAN_CHAT_EVAL_SET_NAME,
            "eval_set_version": "20260908",
            "deployment_ids": _SAMPLED_DEPLOYMENTS,
            "status": "active",
        }
    ]


def test_validation_evalset_is_marked_metrics_only():
    """Customer eval-set entries are PII-gated, so validation items must never
    reach the focused high-signal path that lists them."""
    valset = _make_evalset(["20260908"], deployment_ids=_SAMPLED_DEPLOYMENTS, validation_only=True)
    trainset = _make_evalset(["20260901"])

    assert valset[0]["validation_only"] is True
    assert "validation_only" not in trainset[0]
    assert trainset[0]["deployment_ids"] == list(SCIO_PROD_DEPLOYMENT_IDS)
