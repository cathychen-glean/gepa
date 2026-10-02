"""YAML ``eval:`` section -> EvalHarness -> evalcli create_eval_run arguments."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import yaml

from glean_gepa.adapter_types import EvalHarness
from glean_gepa.al_adapter import CODING_HARNESS_SC_PARAMS, ALRunner
from glean_gepa.experiment_config import ExperimentConfigError, eval_harness, load_experiment_config

_BASE_YAML = """
schema_version: 1
mode: teacher_student
signals:
  - name: tool_alignment
    source: tool_match
    lookback_days: 7
objective:
  primary: tool_alignment
  composite:
    tool_alignment: 1.0
  params:
    tool_alignment: {}
screening:
  kind: high_signal_fix_rate
  threshold: 0.25
  high_signal: tool_alignment
"""


def test_eval_section_parsing_and_validation(tmp_path):
    base = yaml.safe_load(_BASE_YAML)

    def write(eval_section):
        base["eval"] = eval_section
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(base))
        return path

    harness = eval_harness(load_experiment_config(write({"runner_type": "GLEAN_CHAT_V2", "sc_params": "a=1,b=2"})))
    assert harness == ("GLEAN_CHAT_V2", "a=1,b=2")
    harness = eval_harness(load_experiment_config(write({"sc_params": ["a=1", " b=2 "]})))
    assert harness == (None, "a=1,b=2")
    with pytest.raises(ExperimentConfigError, match="key=value"):
        load_experiment_config(write({"sc_params": ["novalue"]}))
    with pytest.raises(ExperimentConfigError, match="unknown keys"):
        load_experiment_config(write({"runner": "GLEAN_CHAT"}))


def test_runner_passes_harness_to_evalcli_or_keeps_defaults():
    evalcli = MagicMock()
    evalcli.create_eval_run.return_value = "ev-1"
    with_harness = ALRunner(evalcli=evalcli, harness=EvalHarness(runner_type="GLEAN_CHAT", sc_params="x.y=1"))
    without = ALRunner(evalcli=evalcli)
    for runner in (with_harness, without):
        runner._resolve_cached_eval = lambda key: None  # type: ignore[method-assign]

    with_harness.start("gpt", "", "set", "v1", ["scio-prod"])
    kwargs = evalcli.create_eval_run.call_args.kwargs
    assert kwargs["runner_type"] == "GLEAN_CHAT"
    assert kwargs["sc_params"].startswith("x.y=1") and CODING_HARNESS_SC_PARAMS not in kwargs["sc_params"]
    assert kwargs["eval_params"].endswith("gleanchat_agent=ADVANCED")

    without.start("fast", "", "set", "v1", ["scio-prod"])
    kwargs = evalcli.create_eval_run.call_args.kwargs
    assert "runner_type" not in kwargs  # evalcli's own default applies
    assert kwargs["sc_params"].startswith(CODING_HARNESS_SC_PARAMS)
    assert kwargs["eval_params"].endswith("gleanchat_agent=FAST")
