from __future__ import annotations

from pathlib import Path

import pytest

from glean_gepa.adapter_types import PointwiseJudge
from glean_gepa.evalcli_client import COMPLETENESS_JUDGE_TYPE, COMPLETENESS_RUN_PARAMS
from glean_gepa.experiment_config import (
    ExperimentConfig,
    ExperimentConfigError,
    composite_weights,
    load_experiment_config,
    pointwise_judges,
    resolve_config_path,
    runner_arg_defaults,
)
from glean_gepa.runner import _parse_args
from glean_gepa.objectives.utils.shell_tool_error_util import SHELL_SUCCESS_OBJECTIVE
from glean_gepa.objectives.utils.tool_match_util import TOOL_ALIGNMENT_OBJECTIVE
from glean_gepa.teacher_student_adapter import POINTWISE_JUDGES as TEACHER_STUDENT_POINTWISE_JUDGES

# Adapters default the composite to a unit weight on their primary objective;
# these mirror that default to guard against drift from the packaged configs.
SINGLE_MODEL_DEFAULT_WEIGHTS = {SHELL_SUCCESS_OBJECTIVE: 1.0}
TEACHER_STUDENT_DEFAULT_WEIGHTS = {TOOL_ALIGNMENT_OBJECTIVE: 1.0}

_POINTWISE_COMPLETENESS = (
    "  - name: completeness\n    source: cortex_judge\n    type: COMPLETENESS\n    kind: pointwise\n"
)
_PAIRWISE_CORRECTNESS = "  - name: correctness\n    source: cortex_judge\n    type: CORRECTNESS\n    kind: pairwise\n"


def _load_mode(tmp_path, body: str) -> ExperimentConfig:
    mode = tmp_path / "mode.yaml"
    mode.write_text(body)
    return load_experiment_config(mode)


def _mode_yaml(*, mode: str = "teacher_student", packs: str = "[tools]", signals: str = "", composite: str = "") -> str:
    body = f"schema_version: 1\nmode: {mode}\npacks: {packs}\n"
    if signals:
        body += f"signals:\n{signals}"
    if composite:
        body += f"objective:\n  composite:\n{composite}"
    return body


def _packaged_teacher_student(*, enable_completeness: bool = False) -> str:
    """The shipped teacher-student config with completeness weighted into the composite.

    Weighting alone is half the re-enable path; `enable_completeness` supplies the
    other half by clearing the judge's `enabled: false`.
    """
    body = resolve_config_path("teacher_student").read_text()
    body = body.replace(
        "  composite:\n    tool_alignment: 1.0", "  composite:\n    completeness: 0.5\n    tool_alignment: 0.5"
    )
    if enable_completeness:
        body = body.replace("    enabled: false\n    run_params:\n      llm_model", "    run_params:\n      llm_model")
    return body


def test_load_packaged_teacher_student_merges_tools_pack():
    config = load_experiment_config("teacher_student")

    assert config.mode == "teacher_student"
    assert config.packs == ("tools",)
    assert config.primary_objective == "tool_alignment"
    assert config.frontier_type == "hybrid"
    assert config.objective["composite"] == {"tool_alignment": 1.0}
    # The judges stay declared so they can be switched back on, but disabled
    # they start no runs and carry no weight.
    assert [signal["name"] for signal in config.signals] == ["tool_alignment", "completeness", "correctness"]
    assert pointwise_judges(config) == ()
    assert composite_weights(config) == {"tool_alignment": 1.0}
    # The mode overrides only the threshold; kind and high_signal come from the pack.
    assert config.screening["kind"] == "high_signal_fix_rate"
    assert config.screening["high_signal"] == "first_tool_mismatch"


@pytest.mark.parametrize(
    ("config_name", "default_weights", "default_judges"),
    [
        ("teacher_student", TEACHER_STUDENT_DEFAULT_WEIGHTS, TEACHER_STUDENT_POINTWISE_JUDGES),
        ("single_model", SINGLE_MODEL_DEFAULT_WEIGHTS, ()),
    ],
)
def test_adapter_defaults_match_the_packaged_config(config_name, default_weights, default_judges):
    """Where these drift, the same run scores a different objective depending on
    whether it was launched by direct construction or by --config."""
    config = load_experiment_config(config_name)

    assert composite_weights(config) == default_weights
    assert pointwise_judges(config) == default_judges


def test_completeness_can_be_switched_back_on(tmp_path):
    """The documented re-enable path: flip `enabled` and add the composite weight."""
    config = _load_mode(tmp_path, _packaged_teacher_student(enable_completeness=True))

    # The signal name is what objective.composite weights, so it has to travel
    # with the judge type the adapter uses to start the Cortex run.
    assert pointwise_judges(config) == (
        PointwiseJudge("completeness", COMPLETENESS_JUDGE_TYPE, COMPLETENESS_RUN_PARAMS),
    )
    assert composite_weights(config) == {"completeness": 0.5, "tool_alignment": 0.5}


def test_weighting_completeness_without_enabling_it_raises(tmp_path):
    """Half of the re-enable path is a load error rather than a silent zero."""
    with pytest.raises(ExperimentConfigError, match=r"completeness \(declared but not scorable\)"):
        _load_mode(tmp_path, _packaged_teacher_student())


def test_load_packaged_single_model_merges_shell_pack():
    config = load_experiment_config("single_model")

    assert config.mode == "single_model"
    assert config.packs == ("shell",)
    assert config.primary_objective == SHELL_SUCCESS_OBJECTIVE
    assert config.frontier_type == "objective"
    assert [signal["name"] for signal in config.signals] == [SHELL_SUCCESS_OBJECTIVE]
    assert config.screening["high_signal"] == "shell_error_entries"
    assert pointwise_judges(config) == ()


def test_resolve_config_path_accepts_packaged_stem_and_file(tmp_path):
    assert resolve_config_path("teacher_student").name == "teacher_student.yaml"
    copied = tmp_path / "custom.yaml"
    copied.write_text(resolve_config_path("single_model").read_text())
    assert resolve_config_path(copied) == copied.resolve()
    with pytest.raises(ExperimentConfigError, match="not found"):
        resolve_config_path("missing_mode")


@pytest.mark.parametrize(
    ("mode", "packs", "match"),
    [
        ("teacher_student", "[shell]", "cannot score pack"),
        ("teacher_student", "[tools, shell]", "cannot score pack"),
        ("teacher_student", "[loops]", "cannot score pack"),
        ("single_model", "[tools]", "cannot score pack"),
        ("single_model", "[shell, tools]", "cannot score pack"),
        ("teacher_student", "[nope]", "unknown pack"),
        ("teacher_student", "[]", "at least one pack"),
    ],
)
def test_pack_not_scorable_by_mode_raises(tmp_path, mode, packs, match):
    with pytest.raises(ExperimentConfigError, match=match):
        _load_mode(tmp_path, _mode_yaml(mode=mode, packs=packs))


@pytest.mark.parametrize(
    ("mode", "pack", "primary"),
    [
        ("teacher_student", "tools", "tool_alignment"),
        ("single_model", "shell", SHELL_SUCCESS_OBJECTIVE),
    ],
)
def test_omitted_packs_defaults_to_the_modes_pack(tmp_path, mode, pack, primary):
    config = _load_mode(tmp_path, f"schema_version: 1\nmode: {mode}\n")

    assert config.packs == (pack,)
    assert config.primary_objective == primary


def test_primary_objective_override_must_match_mode(tmp_path):
    body = _mode_yaml() + f"objective:\n  primary: {SHELL_SUCCESS_OBJECTIVE}\n"

    with pytest.raises(ExperimentConfigError, match="cannot score objective.primary"):
        _load_mode(tmp_path, body)


def test_mode_composite_replaces_the_packs_composite_wholesale(tmp_path):
    body = _mode_yaml(signals=_POINTWISE_COMPLETENESS, composite="    completeness: 1.0\n")

    config = _load_mode(tmp_path, body)

    # The tools pack declares composite {tool_alignment: 1.0}. Merging would leave
    # that weight in place and push the total to 2.0.
    assert composite_weights(config) == {"completeness": 1.0}


_UNSCORABLE = r"\(declared but not scorable\)"
_UNDECLARED = r"\(undeclared\)"
# Every way a composite can fail to load, keyed by cause. The loader checks that
# each weighted name is scorable before it checks the weights distribute, so the
# first group never reaches the arithmetic.
_INVALID_COMPOSITES = {
    # Paired with a valid weight, so one bad name fails the whole load.
    "undeclared_name": (
        rf"typoed_signal {_UNDECLARED}",
        _mode_yaml(composite="    tool_alignment: 0.5\n    typoed_signal: 0.5\n"),
    ),
    "other_modes_signal": (
        rf"{SHELL_SUCCESS_OBJECTIVE} {_UNDECLARED}",
        _mode_yaml(composite=f"    {SHELL_SUCCESS_OBJECTIVE}: 0.5\n"),
    ),
    "pairwise_judge_has_no_per_entry_score": (
        rf"correctness {_UNSCORABLE}",
        _mode_yaml(signals=_PAIRWISE_CORRECTNESS, composite="    correctness: 0.5\n"),
    ),
    "disabled_judge": (
        rf"completeness {_UNSCORABLE}",
        _mode_yaml(signals=f"{_POINTWISE_COMPLETENESS}    enabled: false\n", composite="    completeness: 0.5\n"),
    ),
    "single_model_has_no_judge_plumbing": (
        rf"completeness {_UNSCORABLE}",
        _mode_yaml(
            mode="single_model", packs="[shell]", signals=_POINTWISE_COMPLETENESS, composite="    completeness: 0.5\n"
        ),
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
}


@pytest.mark.parametrize(("match", "body"), _INVALID_COMPOSITES.values(), ids=_INVALID_COMPOSITES)
def test_invalid_composite_fails_the_load(tmp_path, match, body):
    """A composite that names nothing scorable, or does not distribute over 0..1,
    must fail the load rather than score a silent zero or an inflated pass."""
    with pytest.raises(ExperimentConfigError, match=match):
        _load_mode(tmp_path, body)


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


def test_parsed_args_always_carry_experiment():
    assert _parse_args(["--seed_candidate", "seed.json"]).experiment is None
    assert _parse_args(["--config", "teacher_student"]).experiment is not None


def test_judging_mode_flag_conflicting_with_config_raises():
    with pytest.raises(SystemExit, match="conflicts with"):
        _parse_args(["--config", "teacher_student", "--judging_mode", "single_model"])


def test_runner_config_sets_yaml_defaults_and_cli_overrides():
    args = _parse_args(["--config", "teacher_student", "--student_model", "fast"])

    assert args.judging_mode == "teacher_student"
    # Compared against the config, not a literal, so retuning the packaged
    # models does not break this wiring test.
    assert args.student_model == "fast"
    assert args.teacher_model == args.experiment.models["teacher"]
    assert args.seed_candidate == Path("data/seed_candidate.json")
    assert args.run_dir == Path("run_ts")
    assert args.experiment.primary_objective == "tool_alignment"
    # Derived from the widest lookback_days across the merged signals, so it
    # tracks the pack rather than pinning a value the pack is expected to tune.
    pack_lookback = max(
        int(signal["lookback_days"]) for signal in args.experiment.signals if signal.get("lookback_days") is not None
    )
    assert args.agentspan_lookback_days == pack_lookback


def test_global_token_cap_comes_from_yaml_and_yields_to_the_flag(tmp_path):
    from_yaml = _parse_args(["--config", "teacher_student"])
    assert from_yaml.global_token_cap == 4096

    overridden = _parse_args(["--config", "teacher_student", "--global_token_cap", "8192"])
    assert overridden.global_token_cap == 8192

    body = _mode_yaml() + "search:\n  global_token_cap: 2048\n"
    assert runner_arg_defaults(_load_mode(tmp_path, body))["global_token_cap"] == 2048


def test_mode_signal_override_keeps_the_packs_other_fields(tmp_path):
    """Re-declaring a pack signal adjusts one field; it must not drop `source`,
    which would leave the signal unscorable and fail the load."""
    body = _mode_yaml(signals="  - name: tool_alignment\n    lookback_days: 30\n")

    config = _load_mode(tmp_path, body)

    (tool_alignment,) = [s for s in config.signals if s["name"] == "tool_alignment"]
    assert tool_alignment["lookback_days"] == 30
    assert tool_alignment["source"] == "tool_match"
    assert composite_weights(config) == {"tool_alignment": 1.0}


def test_runner_without_config_keeps_cli_defaults():
    args = _parse_args(["--seed_candidate", "seed.json"])

    assert args.config is None
    assert args.experiment is None
    assert args.judging_mode == "single_model"
    assert args.student_model == "gpt"
