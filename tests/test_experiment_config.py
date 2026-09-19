from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from glean_gepa.experiment_config import (
    ExperimentConfig,
    ExperimentConfigError,
    composite_weights,
    customer_validation_gates,
    experiment_objective_pack,
    load_experiment_config,
    pointwise_judges,
    resolve_config_path,
    runner_arg_defaults,
)
from glean_gepa.judge_metrics_util import DEFAULT_CUSTOMER_VALIDATION_GATES
from glean_gepa.objectives import build_objective
from glean_gepa.objectives.utils.shell_tool_error_util import SHELL_SUCCESS_OBJECTIVE
from glean_gepa.objectives.utils.tool_match_util import empty_tool_match_analysis
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


def _mode_yaml(*, mode: str = "teacher_student", packs: str = "[tools]", signals: str = "", composite: str = "") -> str:
    body = f"schema_version: 1\nmode: {mode}\npacks: {packs}\n"
    if signals:
        body += f"signals:\n{signals}"
    if composite:
        body += f"objective:\n  composite:\n{composite}"
    return body


def _objective_yaml(snippet: str) -> str:
    return _mode_yaml() + f"objective:\n{snippet}"


def test_correctness_can_be_weighted_into_the_composite(tmp_path):
    body = (
        resolve_config_path("teacher_student")
        .read_text()
        .replace("  composite:\n    tool_alignment: 1.0", "  composite:\n    correctness: 0.5\n    tool_alignment: 0.5")
    )
    config = _load_mode(tmp_path, body)

    assert composite_weights(config) == {"correctness": 0.5, "tool_alignment": 0.5}
    assert pointwise_judges(config) == ()


def test_resolve_config_path_accepts_packaged_stem_and_file(tmp_path):
    assert resolve_config_path("teacher_student").name == "teacher_student.yaml"
    copied = tmp_path / "custom.yaml"
    copied.write_text(resolve_config_path("single_model").read_text())
    assert resolve_config_path(copied) == copied.resolve()
    with pytest.raises(ExperimentConfigError, match="not found"):
        resolve_config_path("missing_mode")


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


def test_mode_yaml_overlays_the_pack(tmp_path):
    """Mode YAML replaces composite wholesale, patches one signal field, and overlays
    params/modules without dropping the rest of the pack."""
    replaced = _load_mode(tmp_path, _mode_yaml(signals=_POINTWISE_COMPLETENESS, composite="    completeness: 1.0\n"))
    # The tools pack declares composite {tool_alignment: 1.0}. Merging would leave
    # that weight in place and push the total to 2.0.
    assert composite_weights(replaced) == {"completeness": 1.0}

    patched = _load_mode(tmp_path, _mode_yaml(signals="  - name: tool_alignment\n    lookback_days: 30\n"))
    (tool_alignment,) = [s for s in patched.signals if s["name"] == "tool_alignment"]
    assert tool_alignment["lookback_days"] == 30
    assert tool_alignment["source"] == "tool_match"
    assert composite_weights(patched) == {"tool_alignment": 1.0}

    overlaid = _load_mode(
        tmp_path,
        _mode_yaml()
        + "objective:\n  params:\n    failure_score_below: 0.4\n"
        + "reflection:\n  modules:\n    RULES_EXT: Override the rules module.\n",
    )
    assert overlaid.objective["params"]["failure_score_below"] == 0.4
    assert "Shell" in overlaid.objective["params"]["skipped_tools"]
    assert overlaid.reflection["modules"]["RULES_EXT"] == "Override the rules module."
    objective = build_objective("teacher_student", overlaid.signals, pack=experiment_objective_pack(overlaid))
    assert objective.pack_param("failure_score_below", None) == 0.4
    assert objective.reflection_prompt("RULES_EXT") == "Override the rules module."

    skipped_tools = _load_mode(tmp_path, _objective_yaml("  params:\n    skipped_tools:\n      - Shell\n"))
    fetch_objective = build_objective(
        "teacher_student", skipped_tools.signals, pack=experiment_objective_pack(skipped_tools)
    )
    fetch_objective.bigquery_client = MagicMock()
    with patch(
        "glean_gepa.objectives.tool_match.fetch_eval_run_tool_match_analysis",
        return_value=empty_tool_match_analysis("teacher-1", "student-1"),
    ) as fetch:
        fetch_objective.analyze("teacher-1", "student-1")
    skipped = fetch.call_args.kwargs["skip_tools"]
    assert skipped == frozenset({"Shell"})


# Every way a config can fail to load. Composite names are checked for
# scorability before the weights are required to distribute, so those cases
# never reach the arithmetic.
_INVALID_CONFIGS = {
    "teacher_student_shell_pack": ("cannot score pack", _mode_yaml(packs="[shell]")),
    "teacher_student_tools_and_shell": ("cannot score pack", _mode_yaml(packs="[tools, shell]")),
    "teacher_student_loops_pack": ("cannot score pack", _mode_yaml(packs="[loops]")),
    "single_model_tools_pack": ("cannot score pack", _mode_yaml(mode="single_model", packs="[tools]")),
    "single_model_shell_and_tools": ("cannot score pack", _mode_yaml(mode="single_model", packs="[shell, tools]")),
    "unknown_pack": ("unknown pack", _mode_yaml(packs="[nope]")),
    "empty_packs": ("at least one pack", _mode_yaml(packs="[]")),
    "primary_wrong_mode": (
        "cannot score objective.primary",
        _objective_yaml(f"  primary: {SHELL_SUCCESS_OBJECTIVE}\n"),
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
    pack_lookback = max(
        int(signal["lookback_days"]) for signal in args.experiment.signals if signal.get("lookback_days") is not None
    )
    assert args.agentspan_lookback_days == pack_lookback
    assert _parse_args(["--config", "teacher_student", "--global_token_cap", "8192"]).global_token_cap == 8192
    assert (
        runner_arg_defaults(_load_mode(tmp_path, _mode_yaml() + "search:\n  global_token_cap: 2048\n"))[
            "global_token_cap"
        ]
        == 2048
    )

    with pytest.raises(SystemExit, match="conflicts with"):
        _parse_args(["--config", "teacher_student", "--judging_mode", "single_model"])
