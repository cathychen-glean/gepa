from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from glean_gepa.experiment_config import (
    ExperimentConfig,
    ExperimentConfigError,
    composite_weights,
    customer_validation_gates,
    experiment_objective_spec,
    load_experiment_config,
    pairwise_judges,
    pointwise_judges,
    resolve_config_path,
    runner_arg_defaults,
    screening_weights,
)
from glean_gepa.judge_metrics_util import DEFAULT_CUSTOMER_VALIDATION_GATES
from glean_gepa.objectives import AnalysisRequest, build_objective
from glean_gepa.objectives.shell import SHELL_SUCCESS_OBJECTIVE
from glean_gepa.objectives.tool_match import empty_tool_match_analysis
from glean_gepa.runner import _parse_args

_POINTWISE_COMPLETENESS = (
    "  - name: completeness\n    source: cortex_judge\n    type: COMPLETENESS\n    kind: pointwise\n"
)
_PAIRWISE_CORRECTNESS = "  - name: correctness\n    source: cortex_judge\n    type: CORRECTNESS\n    kind: pairwise\n"
_UNSCORABLE = r"\(declared but not scorable\)"
_UNDECLARED = r"\(undeclared\)"


def _load_mode(tmp_path, body: str) -> ExperimentConfig:
    mode = tmp_path / "mode.yaml"
    mode.write_text(body)
    return load_experiment_config(mode)


# Minimal self-contained experiments. ``signals`` is appended to the base signal
# list; ``composite`` replaces the base composite; ``objective``/``screening`` and
# other top-level snippets are appended verbatim and so override nothing -- tests
# that need a different primary use ``_objective_yaml`` or build the body by hand.
_BASE_SIGNALS = {
    "teacher_student": "  - name: tool_alignment\n    source: tool_match\n    lookback_days: 7\n",
    "single_model": f"  - name: {SHELL_SUCCESS_OBJECTIVE}\n    source: shell_telemetry\n    lookback_days: 1\n",
}
_BASE_PRIMARY = {"teacher_student": "tool_alignment", "single_model": SHELL_SUCCESS_OBJECTIVE}
_BASE_PARAMS = {
    "teacher_student": "    skipped_tools:\n      - Shell\n      - Shell Tool\n    failure_score_below: 0.7\n",
    "single_model": "    failure_score_below: 0.9\n",
}


def _mode_yaml(
    *,
    mode: str = "teacher_student",
    signals: str = "",
    composite: str = "",
    params: str | None = None,
    objective_extra: str = "",
    screening: str | None = None,
) -> str:
    primary = _BASE_PRIMARY[mode]
    body = f"schema_version: 1\nmode: {mode}\n"
    body += f"signals:\n{_BASE_SIGNALS[mode]}{signals}"
    body += f"objective:\n  primary: {primary}\n  composite:\n{composite or f'    {primary}: 1.0' + chr(10)}"
    body += f"  params:\n{_BASE_PARAMS[mode] if params is None else params}"
    body += objective_extra
    if screening is None:
        screening = f"  kind: high_signal_fix_rate\n  threshold: 0.25\n  high_signal: {primary}\n"
    body += f"screening:\n{screening}"
    return body


def _objective_yaml(snippet: str) -> str:
    """Base teacher_student experiment with ``snippet`` appended under ``objective``."""
    return _mode_yaml(objective_extra=snippet)


def test_screening_weights_blend_the_gate_and_start_the_named_judge(tmp_path):
    config = _load_mode(
        tmp_path,
        _mode_yaml(
            signals=_PAIRWISE_CORRECTNESS,
            screening="  weights:\n    tool_alignment: 0.5\n    agentic_preference_rate: 0.5\n",
        ),
    )
    assert config.primary_objective == "tool_alignment"
    assert screening_weights(config) == {"tool_alignment": 0.5, "agentic_preference_rate": 0.5}
    judges = {judge.name: judge.judge_type for judge in pairwise_judges(config)}
    assert judges["correctness"] == "CORRECTNESS"
    assert judges["agentic_preference_rate"] == "AGENTIC_JUDGE"


def test_tool_experiment_keeps_tool_match_as_parent_and_blends_the_gate():
    config = load_experiment_config("teacher_student_tool")
    assert config.primary_objective == "tool_alignment"
    assert composite_weights(config) == {"tool_alignment": 1.0}
    assert screening_weights(config) == {"tool_alignment": 0.5, "agentic_preference_rate": 0.5}
    assert config.screening["threshold"] == 0.25


def test_correctness_can_be_weighted_into_the_composite(tmp_path):
    # Built from _mode_yaml, not by editing the packaged teacher_student.yaml:
    # that file's composite changed once already and the str.replace silently
    # matched nothing, leaving this test asserting against the unedited config.
    body = _mode_yaml(
        signals=_PAIRWISE_CORRECTNESS,
        composite="    correctness: 0.5\n    tool_alignment: 0.5\n",
    ).replace("  primary: tool_alignment\n", "  primary: correctness\n")
    config = _load_mode(tmp_path, body)

    assert config.primary_objective == "correctness"
    assert composite_weights(config) == {"correctness": 0.5, "tool_alignment": 0.5}
    assert pointwise_judges(config) == ()


def test_resolve_config_path_accepts_packaged_stem_and_file(tmp_path):
    assert resolve_config_path("teacher_student").name == "teacher_student.yaml"
    copied = tmp_path / "custom.yaml"
    copied.write_text(resolve_config_path("single_model_shell").read_text())
    assert resolve_config_path(copied) == copied.resolve()
    with pytest.raises(ExperimentConfigError, match="not found"):
        resolve_config_path("missing_mode")


def test_packs_key_is_rejected(tmp_path):
    with pytest.raises(ExperimentConfigError, match="no longer supported"):
        _load_mode(tmp_path, _mode_yaml() + "packs: [tools]\n")


@pytest.mark.parametrize(("mode", "primary"), list(_BASE_PRIMARY.items()))
def test_bare_mode_without_signals_falls_back_to_the_modes_default_objective(tmp_path, mode, primary):
    """A config with no scorable signal still builds the registry default for its mode."""
    config = _load_mode(tmp_path, f"schema_version: 1\nmode: {mode}\n")
    assert config.signals == ()
    assert config.primary_objective is None
    objective = build_objective(
        mode, config.signals, bigquery_client=MagicMock(), experiment=experiment_objective_spec(config)
    )
    assert objective.name == primary


def test_experiment_sections_configure_the_objective(tmp_path):
    """objective.params, screening.high_signal, reflection.modules and signals[].name all
    reach the constructed objective."""
    overlaid = _load_mode(
        tmp_path,
        _mode_yaml(params="    skipped_tools:\n      - Shell\n    failure_score_below: 0.4\n")
        + "reflection:\n  modules:\n    RULES_EXT: Override the rules module.\n",
    )
    assert overlaid.objective["params"]["failure_score_below"] == 0.4
    assert overlaid.screening["high_signal"] == "tool_alignment"
    assert "Shell" in overlaid.objective["params"]["skipped_tools"]
    assert overlaid.reflection["modules"]["RULES_EXT"] == "Override the rules module."
    objective = build_objective("teacher_student", overlaid.signals, experiment=experiment_objective_spec(overlaid))
    assert objective.experiment_param("failure_score_below", None) == 0.4
    assert objective.high_signal == "tool_alignment"
    assert objective.signal_names == ("tool_alignment",)
    assert (
        objective.format_reflective_metrics({"score": 0.0, "tool_alignment": 0.25, "correctness": 0.5})
        == "score=0.00, tool_alignment=0.25"
    )
    assert objective.reflection_prompt("RULES_EXT") == "Override the rules module."

    correctness_config = _load_mode(tmp_path, _mode_yaml(signals=_PAIRWISE_CORRECTNESS))
    with_correctness = build_objective(
        "teacher_student",
        correctness_config.signals,
        experiment=experiment_objective_spec(correctness_config),
    )
    assert with_correctness.signal_names == ("tool_alignment", "correctness")
    assert (
        with_correctness.format_reflective_metrics({"score": 0.0, "tool_alignment": 0.25, "correctness": 0.5})
        == "score=0.00, tool_alignment=0.25, correctness=0.50"
    )

    skipped_tools = _load_mode(tmp_path, _mode_yaml(params="    skipped_tools:\n      - Shell\n"))
    fetch_objective = build_objective(
        "teacher_student", skipped_tools.signals, experiment=experiment_objective_spec(skipped_tools)
    )
    fetch_objective.bigquery_client = MagicMock()
    with patch(
        "glean_gepa.objectives.tool_match.fetch_eval_run_tool_match_analysis",
        return_value=empty_tool_match_analysis("teacher-1", "student-1"),
    ) as fetch:
        fetch_objective.analyze("teacher-1", "student-1", request=AnalysisRequest())
    skipped = fetch.call_args.kwargs["skip_tools"]
    assert skipped == frozenset({"Shell"})


def test_duplicate_signal_names_fail_the_load(tmp_path):
    with pytest.raises(ExperimentConfigError, match="declared more than once"):
        _load_mode(tmp_path, _mode_yaml(signals="  - name: tool_alignment\n    source: tool_match\n"))


# Every way a config can fail to load. Composite names are checked for
# scorability before the weights are required to distribute, so those cases
# never reach the arithmetic.
_INVALID_CONFIGS = {
    "teacher_student_shell_source": (
        "cannot score signal",
        _mode_yaml(signals="  - name: shell\n    source: shell_telemetry\n"),
    ),
    "teacher_student_loop_source": (
        "cannot score signal",
        _mode_yaml(signals="  - name: loops\n    source: loop_telemetry\n"),
    ),
    "single_model_tool_source": (
        "cannot score signal",
        _mode_yaml(mode="single_model", signals="  - name: tools\n    source: tool_match\n"),
    ),
    "single_model_agentic_source": (
        "cannot score signal",
        _mode_yaml(mode="single_model", signals="  - name: agentic\n    source: agentic_preference\n"),
    ),
    "unknown_source": ("cannot score signal", _mode_yaml(signals="  - name: x\n    source: nope\n")),
    "signals_not_a_list": ("signals must be a list", "schema_version: 1\nmode: teacher_student\nsignals: {}\n"),
    "screening_weights_do_not_sum": (
        "screening.weights must sum to 1",
        _mode_yaml(screening="  weights:\n    tool_alignment: 0.5\n    agentic_preference_rate: 0.2\n"),
    ),
    "screening_weights_unknown": (
        "produce no score",
        _mode_yaml(screening="  weights:\n    not_a_signal: 1.0\n"),
    ),
    "single_model_screening_judge": (
        "produce no score",
        _mode_yaml(
            mode="single_model",
            screening=f"  weights:\n    {SHELL_SUCCESS_OBJECTIVE}: 0.5\n    agentic_preference_rate: 0.5\n",
        ),
    ),
    "packs_key": ("no longer supported", _mode_yaml() + "packs: [tools]\n"),
    "primary_wrong_mode": (
        "cannot score objective.primary",
        _mode_yaml().replace("  primary: tool_alignment\n", f"  primary: {SHELL_SUCCESS_OBJECTIVE}\n"),
    ),
    "primary_disabled_judge": (
        "cannot score objective.primary",
        _mode_yaml(signals=f"{_PAIRWISE_CORRECTNESS}    enabled: false\n").replace(
            "  primary: tool_alignment\n", "  primary: correctness\n"
        ),
    ),
    "single_model_primary_judge": (
        "cannot score objective.primary",
        _mode_yaml(mode="single_model", signals=_PAIRWISE_CORRECTNESS).replace(
            f"  primary: {SHELL_SUCCESS_OBJECTIVE}\n", "  primary: correctness\n"
        ),
    ),
    "unknown_focused_bucket": ("focused_bucket_type", _objective_yaml("  focused_bucket_type: NOT_A_BUCKET\n")),
    "correctness_weighted_while_disabled": (
        rf"correctness {_UNSCORABLE}",
        _mode_yaml(signals=f"{_PAIRWISE_CORRECTNESS}    enabled: false\n", composite="    correctness: 0.5\n"),
    ),
    "undeclared_name": (
        rf"typoed_signal {_UNDECLARED}",
        _mode_yaml(composite="    tool_alignment: 0.5\n    typoed_signal: 0.5\n"),
    ),
    "other_modes_signal": (
        rf"{SHELL_SUCCESS_OBJECTIVE} {_UNDECLARED}",
        _mode_yaml(composite=f"    {SHELL_SUCCESS_OBJECTIVE}: 0.5\n"),
    ),
    "disabled_judge": (
        rf"completeness {_UNSCORABLE}",
        _mode_yaml(signals=f"{_POINTWISE_COMPLETENESS}    enabled: false\n", composite="    completeness: 0.5\n"),
    ),
    "single_model_has_no_judge_plumbing": (
        rf"completeness {_UNSCORABLE}",
        _mode_yaml(mode="single_model", signals=_POINTWISE_COMPLETENESS, composite="    completeness: 0.5\n"),
    ),
    # High-signal selection treats score >= 1.0 as a pass, so an inflated sum
    # would mark a failing entry perfect and drop it from reflection.
    "weights_above_one": ("must sum to 1", _mode_yaml(composite="    tool_alignment: 2.0\n")),
    "weights_below_one": ("must sum to 1", _mode_yaml(composite="    tool_alignment: 0.3\n")),
    "negative_weight": (
        "must be non-negative",
        _mode_yaml(signals=_POINTWISE_COMPLETENESS, composite="    tool_alignment: -1.0\n    completeness: 2.0\n"),
    ),
    "non_numeric_weight": ("must be a number", _mode_yaml(composite="    tool_alignment: high\n")),
    "empty_composite": (
        "must weight at least one signal",
        "schema_version: 1\nmode: teacher_student\nobjective:\n  composite: {}\n",
    ),
    "constant_below_one": (
        "must have value 1.0",
        _mode_yaml(
            signals="  - name: freebie\n    source: constant\n    value: 0.0\n",
            composite="    tool_alignment: 0.5\n    freebie: 0.5\n",
        ),
    ),
    "disabled_judge_without_type": (
        "requires type",
        _mode_yaml(
            signals="  - name: completeness\n    source: cortex_judge\n    kind: pointwise\n    enabled: false\n"
        ),
    ),
    "validation_unknown_metric": (
        "must be one of",
        _objective_yaml("  validation:\n    - metric: nope\n      min: 0.8\n"),
    ),
    "validation_min_not_unit": (
        "between 0 and 1",
        _objective_yaml("  validation:\n    - metric: correctness\n      min: 80\n"),
    ),
    "validation_min_not_a_number": (
        "must be a number",
        _objective_yaml("  validation:\n    - metric: correctness\n      min: high\n"),
    ),
}


@pytest.mark.parametrize(("match", "body"), _INVALID_CONFIGS.values(), ids=_INVALID_CONFIGS)
def test_invalid_config_fails_the_load(tmp_path, match, body):
    with pytest.raises(ExperimentConfigError, match=match):
        _load_mode(tmp_path, body)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (None, {}),
        (_mode_yaml(), {}),
        (_objective_yaml("  validation: []\n"), {}),
        (_objective_yaml("  validation: {}\n"), {}),
        (_objective_yaml("  validation:\n    - metric: correctness\n      min: 0.95\n"), {"correctness": 0.95}),
        (
            _objective_yaml("  validation:\n    - metric: correctness\n"),
            {"correctness": DEFAULT_CUSTOMER_VALIDATION_GATES["correctness"]},
        ),
    ],
)
def test_customer_validation_gates(tmp_path, body, expected):
    config = None if body is None else _load_mode(tmp_path, body)

    assert customer_validation_gates(config) == expected


def test_config_help_shows_the_configs_defaults(capsys):
    """--help must run after the config is applied, not on the --config pre-scan."""
    with pytest.raises(SystemExit):
        _parse_args(["--config", "teacher_student", "--help"])
    with_config = capsys.readouterr().out

    with pytest.raises(SystemExit):
        _parse_args(["--help"])
    without_config = capsys.readouterr().out

    # The config sets student_model; the bare parser's own default is gpt.
    assert "(default: claude_sonnet)" in with_config
    assert "(default: gpt)" in without_config


def test_runner_applies_config_then_cli_overrides(tmp_path):
    bare = _parse_args(["--seed_candidate", "seed.json"])
    assert bare.config is None
    assert bare.experiment is None
    assert bare.judging_mode == "single_model"
    assert bare.student_model == "gpt"

    args = _parse_args(["--config", "teacher_student", "--student_model", "fast"])
    assert args.judging_mode == "teacher_student"
    # Compared against the config, not a literal, so retuning the packaged
    # models does not break this wiring test.
    assert args.student_model == "fast"
    assert args.teacher_model == args.experiment.models["teacher"]
    assert args.run_dir == Path(args.experiment.run["dir"])
    config_lookback = max(
        int(signal["lookback_days"]) for signal in args.experiment.signals if signal.get("lookback_days") is not None
    )
    assert args.agentspan_lookback_days == config_lookback
    assert _parse_args(["--config", "teacher_student", "--global_token_cap", "8192"]).global_token_cap == 8192
    assert (
        runner_arg_defaults(_load_mode(tmp_path, _mode_yaml() + "search:\n  global_token_cap: 2048\n"))[
            "global_token_cap"
        ]
        == 2048
    )

    with pytest.raises(SystemExit, match="conflicts with"):
        _parse_args(["--config", "teacher_student", "--judging_mode", "single_model"])


def test_customer_eval_toggle_yaml_default_and_cli_override(tmp_path):
    # Bare parser: on by default.
    assert _parse_args(["--seed_candidate", "seed.json"]).customer_eval is True
    assert _parse_args(["--seed_candidate", "seed.json", "--no_customer_eval"]).customer_eval is False

    # YAML turns it off; CLI flag turns it back on.
    off = tmp_path / "off.yaml"
    off.write_text(_mode_yaml() + "run:\n  customer_eval: false\n")
    assert runner_arg_defaults(load_experiment_config(off))["customer_eval"] is False
    assert _parse_args(["--config", str(off)]).customer_eval is False
    assert _parse_args(["--config", str(off), "--customer_eval"]).customer_eval is True

    # Absent from YAML: no default injected, parser default (True) applies.
    assert "customer_eval" not in runner_arg_defaults(_load_mode(tmp_path, _mode_yaml()))

    bad = tmp_path / "bad.yaml"
    bad.write_text(_mode_yaml() + "run:\n  customer_eval: nope\n")
    with pytest.raises(ExperimentConfigError, match="run.customer_eval must be true or false"):
        runner_arg_defaults(load_experiment_config(bad))


def test_waldo_config_runs_post_search_eval_on_pinned_internal_val():
    """customer_eval is on, and with val pinned it runs on scio-prod, not a customer deployment."""
    config = load_experiment_config("teacher_student_waldo")
    assert runner_arg_defaults(config)["customer_eval"] is True
    assert _parse_args(["--config", "teacher_student_waldo"]).customer_eval is True
    assert config.data["val_eval_versions"] == [20260909]
    assert config.data["deployment_ids"] == ["scio-prod"]
    # Two pinned train versions so a childless generation does not exhaust the schedule.
    assert config.data["train_eval_versions"] == [20261001, 20260908]
    assert not set(config.data["train_eval_versions"]) & set(config.data["val_eval_versions"])


def test_waldo_global_token_cap_leaves_room_for_the_tool_module_budget():
    """The global cap must not bind before the per-module budget does."""
    import json

    from glean_gepa.al_adapter import approx_token_len
    from glean_gepa.waldo_prompt_constants import WALDO_TOOL_USAGE_TOKEN_BUDGET

    config = load_experiment_config("teacher_student_waldo")
    seed = json.loads(Path(config.run["seed_candidate"]).read_text())

    def find(obj, key):
        if isinstance(obj, dict):
            if isinstance(obj.get(key), str):
                return obj[key]
            for value in obj.values():
                found = find(value, key)
                if found:
                    return found
        return None

    system_tokens = approx_token_len(find(seed, "WALDO_SYSTEM"))
    assert config.search["global_token_cap"] >= system_tokens + WALDO_TOOL_USAGE_TOKEN_BUDGET
