from __future__ import annotations

from pathlib import Path

import pytest

from glean_gepa.adapter_types import PointwiseJudge
from glean_gepa.evalcli_client import COMPLETENESS_JUDGE_TYPE, COMPLETENESS_RUN_PARAMS
from glean_gepa.experiment_config import (
    ExperimentConfigError,
    composite_weights,
    load_experiment_config,
    pointwise_judges,
    resolve_config_path,
    runner_arg_defaults,
    screening_threshold,
)
from glean_gepa.runner import _parse_args
from glean_gepa.shell_tool_error_util import SHELL_SUCCESS_OBJECTIVE

_POINTWISE_COMPLETENESS = (
    "  - name: completeness\n    source: cortex_judge\n    type: COMPLETENESS\n    kind: pointwise\n"
)
_PAIRWISE_CORRECTNESS = "  - name: correctness\n    source: cortex_judge\n    type: CORRECTNESS\n    kind: pairwise\n"


def _write_mode(tmp_path, body: str) -> Path:
    mode = tmp_path / "mode.yaml"
    mode.write_text(body)
    return mode


def _mode_yaml(*, mode: str = "teacher_student", packs: str = "[tools]", signals: str = "", composite: str = "") -> str:
    body = f"schema_version: 1\nmode: {mode}\npacks: {packs}\n"
    if signals:
        body += f"signals:\n{signals}"
    if composite:
        body += f"objective:\n  composite:\n{composite}"
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


def test_completeness_can_be_switched_back_on(tmp_path):
    """The documented re-enable path: flip `enabled` and add the composite weight."""
    packaged = resolve_config_path("teacher_student").read_text()
    reenabled = packaged.replace(
        "    enabled: false\n    run_params:\n      llm_model", "    run_params:\n      llm_model"
    )
    config = load_experiment_config(
        _write_mode(
            tmp_path,
            reenabled.replace(
                "  composite:\n    tool_alignment: 1.0", "  composite:\n    completeness: 0.5\n    tool_alignment: 0.5"
            ),
        )
    )

    # The signal name is what objective.composite weights, so it has to travel
    # with the judge type the adapter uses to start the Cortex run.
    assert pointwise_judges(config) == (
        PointwiseJudge("completeness", COMPLETENESS_JUDGE_TYPE, COMPLETENESS_RUN_PARAMS),
    )
    assert composite_weights(config) == {"completeness": 0.5, "tool_alignment": 0.5}


def test_weighting_completeness_without_enabling_it_raises(tmp_path):
    """Half of the re-enable path is a load error rather than a silent zero."""
    packaged = resolve_config_path("teacher_student").read_text()
    config = _write_mode(
        tmp_path,
        packaged.replace(
            "  composite:\n    tool_alignment: 1.0", "  composite:\n    completeness: 0.5\n    tool_alignment: 0.5"
        ),
    )

    with pytest.raises(ExperimentConfigError, match=r"completeness \(declared but not scorable\)"):
        load_experiment_config(config)


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
    ("mode", "packs"),
    [
        ("teacher_student", "[shell]"),
        ("teacher_student", "[tools, shell]"),
        ("single_model", "[tools]"),
        ("single_model", "[shell, tools]"),
        ("teacher_student", "[nope]"),
        ("teacher_student", "[]"),
    ],
)
def test_pack_not_scorable_by_mode_raises(tmp_path, mode, packs):
    with pytest.raises(ExperimentConfigError, match="supports only packs"):
        load_experiment_config(_write_mode(tmp_path, _mode_yaml(mode=mode, packs=packs)))


@pytest.mark.parametrize(
    ("mode", "pack", "primary"),
    [
        ("teacher_student", "tools", "tool_alignment"),
        ("single_model", "shell", SHELL_SUCCESS_OBJECTIVE),
    ],
)
def test_omitted_packs_defaults_to_the_modes_pack(tmp_path, mode, pack, primary):
    config = load_experiment_config(_write_mode(tmp_path, f"schema_version: 1\nmode: {mode}\n"))

    assert config.packs == (pack,)
    assert config.primary_objective == primary


def test_primary_objective_override_must_match_mode(tmp_path):
    body = _mode_yaml() + f"objective:\n  primary: {SHELL_SUCCESS_OBJECTIVE}\n"

    with pytest.raises(ExperimentConfigError, match="can only score objective.primary=tool_alignment"):
        load_experiment_config(_write_mode(tmp_path, body))


def test_mode_composite_replaces_the_packs_composite_wholesale(tmp_path):
    body = _mode_yaml(signals=_POINTWISE_COMPLETENESS, composite="    completeness: 1.0\n")

    config = load_experiment_config(_write_mode(tmp_path, body))

    # The tools pack declares composite {tool_alignment: 1.0}. Merging would leave
    # that weight in place and push the total to 2.0.
    assert composite_weights(config) == {"completeness": 1.0}


_UNSCORABLE = "declared but not scorable"
# Every way a composite weight can fail to resolve, keyed by why.
_UNSCORABLE_COMPOSITES = {
    # Paired with a valid weight, so one bad name fails the whole load.
    "undeclared_name": (
        "typoed_signal",
        "undeclared",
        _mode_yaml(composite="    tool_alignment: 0.5\n    typoed_signal: 0.5\n"),
    ),
    "other_modes_signal": (
        SHELL_SUCCESS_OBJECTIVE,
        "undeclared",
        _mode_yaml(composite=f"    {SHELL_SUCCESS_OBJECTIVE}: 0.5\n"),
    ),
    "pairwise_judge_has_no_per_entry_score": (
        "correctness",
        _UNSCORABLE,
        _mode_yaml(signals=_PAIRWISE_CORRECTNESS, composite="    correctness: 0.5\n"),
    ),
    "disabled_judge": (
        "completeness",
        _UNSCORABLE,
        _mode_yaml(signals=f"{_POINTWISE_COMPLETENESS}    enabled: false\n", composite="    completeness: 0.5\n"),
    ),
    "single_model_has_no_judge_plumbing": (
        "completeness",
        _UNSCORABLE,
        _mode_yaml(
            mode="single_model", packs="[shell]", signals=_POINTWISE_COMPLETENESS, composite="    completeness: 0.5\n"
        ),
    ),
}


@pytest.mark.parametrize(("signal", "reason", "body"), _UNSCORABLE_COMPOSITES.values(), ids=_UNSCORABLE_COMPOSITES)
def test_composite_weight_nothing_can_score_raises(tmp_path, signal, reason, body):
    """A weight that resolves to nothing must fail the load, never score a silent zero."""
    with pytest.raises(ExperimentConfigError, match=rf"{signal} \({reason}\)"):
        load_experiment_config(_write_mode(tmp_path, body))


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
    assert args.seed_candidate == Path("data/seed_teacher_student.json")
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
    assert runner_arg_defaults(load_experiment_config(_write_mode(tmp_path, body)))["global_token_cap"] == 2048


def test_mode_signal_override_keeps_the_packs_other_fields(tmp_path):
    """Re-declaring a pack signal adjusts one field; it must not drop `source`,
    which would leave the signal unscorable and fail the load."""
    body = _mode_yaml(signals="  - name: tool_alignment\n    lookback_days: 30\n")

    config = load_experiment_config(_write_mode(tmp_path, body))

    (tool_alignment,) = [s for s in config.signals if s["name"] == "tool_alignment"]
    assert tool_alignment["lookback_days"] == 30
    assert tool_alignment["source"] == "tool_match"
    assert composite_weights(config) == {"tool_alignment": 1.0}


@pytest.mark.parametrize(
    ("expected", "composite"),
    [
        # high-signal selection treats score >= 1.0 as a pass, so an inflated sum
        # would mark a failing entry perfect and drop it from reflection.
        ("must sum to 1", "    tool_alignment: 2.0\n"),
        ("must sum to 1", "    tool_alignment: 0.3\n"),
        ("must be non-negative", "    tool_alignment: -1.0\n    completeness: 2.0\n"),
        ("must be a number", "    tool_alignment: high\n"),
    ],
)
def test_composite_weights_must_be_a_normalized_distribution(tmp_path, expected, composite):
    body = _mode_yaml(signals=_POINTWISE_COMPLETENESS, composite=composite)

    with pytest.raises(ExperimentConfigError, match=expected):
        load_experiment_config(_write_mode(tmp_path, body))


def test_runner_without_config_keeps_cli_defaults():
    args = _parse_args(["--seed_candidate", "seed.json"])

    assert args.config is None
    assert args.experiment is None
    assert args.judging_mode == "single_model"
    assert args.student_model == "gpt"
