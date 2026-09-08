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
    screening_threshold,
)
from glean_gepa.runner import _parse_args
from glean_gepa.shell_tool_error_util import SHELL_SUCCESS_OBJECTIVE


def test_load_packaged_teacher_student_merges_tools_pack():
    config = load_experiment_config("teacher_student")

    assert config.mode == "teacher_student"
    assert config.packs == ("tools",)
    assert config.primary_objective == "tool_alignment"
    assert config.frontier_type == "hybrid"
    assert config.objective["composite"] == {"tool_alignment": 1.0}
    # completeness is still declared so it can be switched back on, but it is
    # disabled, so no judge runs are started and it carries no weight.
    signal_names = [signal["name"] for signal in config.signals]
    assert signal_names == ["tool_alignment", "completeness", "correctness", "grounding"]
    assert pointwise_judges(config) == ()
    assert composite_weights(config) == {"tool_alignment": 1.0}
    # The mode file overrides only the pack's threshold; kind and high_signal
    # still come from the pack.
    assert screening_threshold(config) == pytest.approx(0.25)
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


def _write_mode(tmp_path, body: str) -> Path:
    mode = tmp_path / "mode.yaml"
    mode.write_text(body)
    return mode


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
    config = _write_mode(tmp_path, f"schema_version: 1\nmode: {mode}\npacks: {packs}\n")

    with pytest.raises(ExperimentConfigError, match="supports only packs"):
        load_experiment_config(config)


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
    config = _write_mode(
        tmp_path,
        f"schema_version: 1\nmode: teacher_student\npacks: [tools]\nobjective:\n  primary: {SHELL_SUCCESS_OBJECTIVE}\n",
    )

    with pytest.raises(ExperimentConfigError, match="can only score objective.primary=tool_alignment"):
        load_experiment_config(config)


def test_mode_composite_replaces_the_packs_composite_wholesale(tmp_path):
    config = load_experiment_config(
        _write_mode(
            tmp_path,
            "schema_version: 1\nmode: teacher_student\npacks: [tools]\n"
            "signals:\n  - name: completeness\n    source: cortex_judge\n"
            "    type: COMPLETENESS\n    kind: pointwise\n"
            "objective:\n  composite:\n    completeness: 1.0\n",
        )
    )

    # The tools pack declares composite {tool_alignment: 1.0}. Merging would leave
    # that weight in place and push the total to 2.0.
    assert composite_weights(config) == {"completeness": 1.0}


def test_composite_weighting_undeclared_signal_raises(tmp_path):
    config = _write_mode(
        tmp_path,
        "schema_version: 1\nmode: teacher_student\npacks: [tools]\n"
        "objective:\n  composite:\n    tool_alignment: 0.5\n    typoed_signal: 0.5\n",
    )

    with pytest.raises(ExperimentConfigError, match=r"typoed_signal \(undeclared\)"):
        load_experiment_config(config)


def test_composite_weighting_declared_but_unscorable_signal_raises(tmp_path):
    # correctness is a pairwise judge: nothing turns it into a per-entry score.
    config = _write_mode(
        tmp_path,
        "schema_version: 1\nmode: teacher_student\npacks: [tools]\n"
        "signals:\n  - name: correctness\n    source: cortex_judge\n"
        "    type: CORRECTNESS\n    kind: pairwise\n"
        "objective:\n  composite:\n    tool_alignment: 0.5\n    correctness: 0.5\n",
    )

    with pytest.raises(ExperimentConfigError, match=r"correctness \(declared but not scorable\)"):
        load_experiment_config(config)


def test_single_model_cannot_weight_a_pointwise_judge(tmp_path):
    """single_model has no judge plumbing, so a weighted judge must fail at load."""
    config = _write_mode(
        tmp_path,
        "schema_version: 1\nmode: single_model\npacks: [shell]\n"
        "signals:\n  - name: completeness\n    source: cortex_judge\n"
        "    type: COMPLETENESS\n    kind: pointwise\n"
        f"objective:\n  composite:\n    {SHELL_SUCCESS_OBJECTIVE}: 0.5\n    completeness: 0.5\n",
    )

    with pytest.raises(ExperimentConfigError, match=r"completeness \(declared but not scorable\)"):
        load_experiment_config(config)


def test_composite_weighting_disabled_judge_raises(tmp_path):
    config = _write_mode(
        tmp_path,
        "schema_version: 1\nmode: teacher_student\npacks: [tools]\n"
        "signals:\n  - name: completeness\n    source: cortex_judge\n"
        "    type: COMPLETENESS\n    kind: pointwise\n    enabled: false\n"
        "objective:\n  composite:\n    tool_alignment: 0.5\n    completeness: 0.5\n",
    )

    with pytest.raises(ExperimentConfigError, match=r"completeness \(declared but not scorable\)"):
        load_experiment_config(config)


def test_composite_cannot_weight_the_other_modes_signal(tmp_path):
    config = _write_mode(
        tmp_path,
        "schema_version: 1\nmode: teacher_student\npacks: [tools]\n"
        f"objective:\n  composite:\n    tool_alignment: 0.5\n    {SHELL_SUCCESS_OBJECTIVE}: 0.5\n",
    )

    with pytest.raises(ExperimentConfigError, match=rf"{SHELL_SUCCESS_OBJECTIVE} \(undeclared\)"):
        load_experiment_config(config)


def test_judging_mode_flag_conflicting_with_config_raises():
    with pytest.raises(SystemExit, match="conflicts with"):
        _parse_args(["--config", "teacher_student", "--judging_mode", "single_model"])


def test_runner_config_sets_yaml_defaults_and_cli_overrides():
    args = _parse_args(["--config", "teacher_student", "--student_model", "fast"])

    assert args.judging_mode == "teacher_student"
    assert args.student_model == "fast"
    assert args.teacher_model == "gpt"
    assert args.seed_candidate == Path("data/seed_candidate.json")
    assert args.run_dir == Path("run_ts")
    assert args.experiment.primary_objective == "tool_alignment"
    # Derived from the widest lookback_days across the merged signals, so it
    # tracks the pack rather than pinning a value the pack is expected to tune.
    pack_lookback = max(
        int(signal["lookback_days"]) for signal in args.experiment.signals if signal.get("lookback_days") is not None
    )
    assert args.agentspan_lookback_days == pack_lookback


def test_runner_without_config_keeps_cli_defaults():
    args = _parse_args(["--seed_candidate", "seed.json"])

    assert args.config is None
    assert args.experiment is None
    assert args.judging_mode == "single_model"
    assert args.student_model == "gpt"
