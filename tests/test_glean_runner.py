import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from glean_gepa.evalcli_client import (
    AGENTIC_JUDGE_NAME,
    AGENTIC_JUDGE_TYPE,
    AGENTIC_PREFERENCE_RATE_METRIC,
    CORRECTNESS_JUDGE_TYPE,
)
from glean_gepa.judge_metrics_util import DEFAULT_CUSTOMER_VALIDATION_GATES
from glean_gepa.prompt import compile_system_prompt, materialize_system_prompt
from glean_gepa.prompt_constants import (
    CORE_TOOLS,
    CORE_TOOLS_GROUP,
    DEFAULT_EXECUTION_DISCIPLINE,
    DEFAULT_FULL_PROMPT,
    DEFAULT_WRITING_CODE,
    EXECUTION_DISCIPLINE_KEY,
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
    TEACHER_STUDENT_DEPLOYMENT_IDS,
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
SEED_WITH_RULES_SLOT = {**SEED_BOTH, "WRITING_CODE": "patterns\n{RULES_EXT}"}
_PINNED_EXECUTION_DISCIPLINE = {FULL_PROMPT_KEY: "PREFIX\n### Execution Discipline\n- Answer fast.\nSUFFIX"}


def test_compile_and_materialize_prompt_modules():
    """Writing Code and Execution Discipline splice into FULL_PROMPT; empty
    discipline falls back so the heading is never bare."""
    assert compile_system_prompt(SEED_BOTH) == "PREFIX\npatterns\nSUFFIX"

    stock = compile_system_prompt({WRITING_CODE_KEY: "CUSTOM_PATTERNS"})
    assert "CUSTOM_PATTERNS" in stock
    assert DEFAULT_WRITING_CODE not in stock
    assert "{WRITING_CODE}" not in stock

    assert compile_system_prompt({}) == DEFAULT_FULL_PROMPT.replace(
        "{EXECUTION_DISCIPLINE}", DEFAULT_EXECUTION_DISCIPLINE
    )
    assert (
        compile_system_prompt({FULL_PROMPT_KEY: "PREFIX\n{WRITING_CODE}\nSUFFIX"}) == "PREFIX\n{WRITING_CODE}\nSUFFIX"
    )

    prompt = materialize_system_prompt({})
    assert prompt == DEFAULT_FULL_PROMPT.replace("{WRITING_CODE}", DEFAULT_WRITING_CODE)
    assert "{RULES_EXT}" in prompt

    custom = compile_system_prompt({EXECUTION_DISCIPLINE_KEY: "- Keep working until the deliverable is complete."})
    assert "- Keep working until the deliverable is complete." in custom
    assert "{EXECUTION_DISCIPLINE}" not in custom
    assert "as few tool loops as possible" not in custom
    for candidate in ({EXECUTION_DISCIPLINE_KEY: ""}, {EXECUTION_DISCIPLINE_KEY: "   \n"}):
        assert DEFAULT_EXECUTION_DISCIPLINE in compile_system_prompt(candidate)


def test_compile_system_prompt_splices_rules_ext_after_rules():
    writing = "intro\n**Rules:**\n- stock rule\n{RULES_EXT}\n### Sandbox\n"
    compiled = compile_system_prompt(
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
    empty = compile_system_prompt(
        {WRITING_CODE_KEY: writing, FULL_PROMPT_KEY: "PREFIX\n{WRITING_CODE}\nSUFFIX", RULES_EXT_KEY: ""}
    )
    assert empty == "PREFIX\nintro\n**Rules:**\n- stock rule\n### Sandbox\n\nSUFFIX"

    # A rewritten Writing Code may reflow the slot; the literal token must never ship.
    reflowed = compile_system_prompt(
        {
            WRITING_CODE_KEY: "intro\n**Rules:**\n- stock rule\n{RULES_EXT}  \n### Sandbox\n",
            FULL_PROMPT_KEY: "PREFIX\n{WRITING_CODE}\nSUFFIX",
            RULES_EXT_KEY: "",
        }
    )
    assert reflowed == empty


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"WRITING_CODE": "code instructions"},
    ],
)
def test_load_seed_candidate_accepts_known_keys(tmp_path, raw):
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(raw))

    assert _load_seed_candidate(path) == raw


def test_seed_for_editable_modules():
    assert _seed_for_editable_modules(SEED_BOTH, [WRITING_CODE_KEY]) == {WRITING_CODE_KEY: "patterns"}

    raw = {**SEED_BOTH, "glean_search": "Search less."}
    frozen = _seed_for_editable_modules(SEED_BOTH, [])
    assert frozen[FULL_PROMPT_KEY] == "PREFIX\npatterns\nSUFFIX"
    assert WRITING_CODE_KEY not in frozen

    core_tool_seed = _seed_for_editable_modules(raw, ["glean_search", "discover"])
    assert core_tool_seed["glean_search"] == "Search less."
    assert core_tool_seed["discover"] == PROMPT_MODULE_DEFAULTS["discover"]

    rules_seed = _seed_for_editable_modules({**SEED_WITH_RULES_SLOT, RULES_EXT_KEY: ""}, [RULES_EXT_KEY])
    assert rules_seed[RULES_EXT_KEY] == ""
    assert rules_seed[FULL_PROMPT_KEY] == "PREFIX\npatterns\n{RULES_EXT}\nSUFFIX"

    defaulted = _seed_for_editable_modules({}, [WRITING_CODE_KEY, RULES_EXT_KEY])
    assert defaulted[WRITING_CODE_KEY] == DEFAULT_WRITING_CODE
    assert defaulted[RULES_EXT_KEY] == ""
    assert FULL_PROMPT_KEY not in defaulted

    overridden = _seed_for_editable_modules({RULES_EXT_KEY: "- Prefer Write."}, [RULES_EXT_KEY])
    assert overridden[RULES_EXT_KEY] == "- Prefer Write."
    assert overridden[FULL_PROMPT_KEY] == materialize_system_prompt({})

    stock = _seed_for_editable_modules({}, [EXECUTION_DISCIPLINE_KEY])
    assert stock[EXECUTION_DISCIPLINE_KEY] == DEFAULT_EXECUTION_DISCIPLINE
    # WRITING_CODE is not editable here, so the frozen prompt must keep the slot open.
    assert "{EXECUTION_DISCIPLINE}" in stock[FULL_PROMPT_KEY]
    assert "{WRITING_CODE}" not in stock[FULL_PROMPT_KEY]


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
    with pytest.raises(SystemExit, match="unknown editable_modules"):
        _parse_editable_modules("GLOBAL_ROLE")
    with pytest.raises(SystemExit, match=rf"{FULL_PROMPT_KEY} is not editable.*{EXECUTION_DISCIPLINE_KEY}"):
        _parse_editable_modules(FULL_PROMPT_KEY)


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


def test_parse_args():
    default = _parse_args(["--seed_candidate", "seed.json"])
    assert default.editable_modules == WRITING_CODE_KEY
    assert default.eval_version_days_back == 0

    args = _parse_args(["--seed_candidate", "seed.json", "--reflection_samples", "all"])
    assert args.reflection_samples is None

    for value in ("0", "not-a-number"):
        with pytest.raises(SystemExit):
            _parse_args(["--seed_candidate", "seed.json", "--reflection_samples", value])
    with pytest.raises(SystemExit):
        _parse_args(["--seed_candidate", "seed.json", "--eval_version_days_back", "-1"])


def test_default_cache_file(tmp_path):
    assert _default_cache_file(tmp_path, ADAPTER_CACHE_FILENAME) == (
        tmp_path / CACHE_DIRECTORY_NAME / ADAPTER_CACHE_FILENAME
    )

    legacy = tmp_path / ADAPTER_CACHE_FILENAME
    legacy.write_text('{"cached": true}')
    cache_file = _default_cache_file(tmp_path, ADAPTER_CACHE_FILENAME)
    assert cache_file == tmp_path / CACHE_DIRECTORY_NAME / ADAPTER_CACHE_FILENAME
    assert cache_file.read_text() == '{"cached": true}'
    assert not legacy.exists()


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
    with pytest.raises(SystemExit, match="not published to"):
        _resolve_eval_version_split(args, evalcli, _SAMPLED_DEPLOYMENTS)


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


def test_make_evalset():
    """Customer eval-set entries are PII-gated, so validation items must never
    reach the focused high-signal path that lists them."""
    valset = _make_evalset(["20260908"], deployment_ids=_SAMPLED_DEPLOYMENTS, validation_only=True)
    trainset = _make_evalset(["20260901"])

    assert valset[0]["deployment_ids"] == _SAMPLED_DEPLOYMENTS
    assert valset[0]["validation_only"] is True
    assert "validation_only" not in trainset[0]
    assert trainset[0]["deployment_ids"] == list(SCIO_PROD_DEPLOYMENT_IDS)
