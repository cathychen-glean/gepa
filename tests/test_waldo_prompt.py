"""Waldo prompt modules: template slot, conditional preservation, and encoding."""

from __future__ import annotations

import json
from base64 import urlsafe_b64decode
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from glean_gepa.al_adapter import WALDO_DEFAULT_MODEL, WALDO_ENABLE_SC_PARAM, ALRunner, parse_waldo_alias
from glean_gepa.prompt import compile_encoded_prompt, compile_waldo_system_override
from glean_gepa.prompt_constants import EDITABLE_PROMPT_KEYS, KNOWN_PROMPT_KEYS, MODULE_TOKEN_BUDGETS
from glean_gepa.reflection_prompts import conditional_counts, drops_conditional, markup_rule_for
from glean_gepa.runner import _parse_editable_modules, _seed_for_editable_modules
from glean_gepa.waldo_harness_params import WALDO_HARNESS_SC_PARAMS
from glean_gepa.waldo_prompt_constants import (
    DEFAULT_WALDO_SYSTEM,
    DEFAULT_WALDO_TOOL_USAGE,
    WALDO_SYSTEM_KEY,
    WALDO_SYSTEM_OVERRIDE_PARAM,
    WALDO_TOOL_USAGE_CONDITIONALS,
    WALDO_TOOL_USAGE_KEY,
    WALDO_TOOL_USAGE_SLOT,
    compile_waldo_system_prompt,
)


def _decode_override(encoded: str) -> str:
    assert encoded.startswith(WALDO_SYSTEM_OVERRIDE_PARAM + "=")
    return urlsafe_b64decode(encoded.split("=", 1)[1]).decode("utf-8")


def test_waldo_keys_are_registered():
    assert {WALDO_SYSTEM_KEY, WALDO_TOOL_USAGE_KEY} <= KNOWN_PROMPT_KEYS
    assert {WALDO_SYSTEM_KEY, WALDO_TOOL_USAGE_KEY} <= EDITABLE_PROMPT_KEYS
    assert MODULE_TOKEN_BUDGETS[WALDO_TOOL_USAGE_KEY] < MODULE_TOKEN_BUDGETS[WALDO_SYSTEM_KEY]


def test_template_has_slot_and_default_tool_usage_has_both_conditionals():
    assert DEFAULT_WALDO_SYSTEM.count(WALDO_TOOL_USAGE_SLOT) == 1
    counts = conditional_counts(DEFAULT_WALDO_TOOL_USAGE)
    assert set(counts) == set(WALDO_TOOL_USAGE_CONDITIONALS)
    assert all(count >= 1 for count in counts.values())


def test_compile_fills_slot_with_default_when_module_absent():
    compiled = compile_waldo_system_prompt({})
    assert WALDO_TOOL_USAGE_SLOT not in compiled
    assert DEFAULT_WALDO_TOOL_USAGE in compiled
    assert "### Tool Usage Guidelines\n" + DEFAULT_WALDO_TOOL_USAGE in compiled


def test_compile_splices_edited_tool_usage():
    edited = "<<<[[has_search_tools]]A>>>\n<<<[[no_search_tools]]B>>>"
    compiled = compile_waldo_system_prompt({WALDO_TOOL_USAGE_KEY: edited})
    assert edited in compiled
    assert DEFAULT_WALDO_TOOL_USAGE not in compiled
    assert "[[waldo_discover_tool_name]]" in compiled  # rest of template intact


def test_override_is_empty_without_waldo_modules():
    assert compile_waldo_system_override({"WRITING_CODE": "x"}) == ""
    assert WALDO_SYSTEM_OVERRIDE_PARAM not in compile_encoded_prompt({"WRITING_CODE": "x"})


def test_override_encodes_compiled_prompt_for_tool_usage_only_candidate():
    edited = "<<<[[has_search_tools]]A>>>\n<<<[[no_search_tools]]B>>>"
    decoded = _decode_override(compile_waldo_system_override({WALDO_TOOL_USAGE_KEY: edited}))
    assert decoded == compile_waldo_system_prompt({WALDO_TOOL_USAGE_KEY: edited})
    assert WALDO_TOOL_USAGE_SLOT not in decoded
    encoded = compile_encoded_prompt({WALDO_TOOL_USAGE_KEY: edited})
    assert encoded.count(WALDO_SYSTEM_OVERRIDE_PARAM + "=") == 1


def test_seed_for_tool_usage_pins_template():
    mods = _parse_editable_modules(WALDO_TOOL_USAGE_KEY)
    seed = _seed_for_editable_modules({}, mods)
    assert seed[WALDO_TOOL_USAGE_KEY] == DEFAULT_WALDO_TOOL_USAGE
    assert seed[WALDO_SYSTEM_KEY] == DEFAULT_WALDO_SYSTEM


def test_seed_rejects_template_without_slot():
    with pytest.raises(SystemExit, match="no \\{WALDO_TOOL_USAGE\\} slot"):
        _seed_for_editable_modules({WALDO_SYSTEM_KEY: "no slot here"}, [WALDO_TOOL_USAGE_KEY])


def test_seed_rejects_tool_usage_without_conditionals():
    with pytest.raises(SystemExit, match="conditionals"):
        _seed_for_editable_modules({WALDO_TOOL_USAGE_KEY: "plain text"}, [WALDO_TOOL_USAGE_KEY])


@pytest.mark.parametrize(
    "proposed",
    [
        # no_search_tools dropped entirely
        "<<<[[has_search_tools]]Search first.>>>\nThen answer.",
        # has_search_tools dropped entirely
        "<<<[[no_search_tools]]Call discover.>>>",
        # both names gone, text kept
        "Attached tools are only for easy lookups. No search tools: call discover.",
        # unbalanced fences
        "<<<[[has_search_tools]]A>>>\n<<<[[no_search_tools]]B",
    ],
)
def test_drops_conditional_rejects_lost_or_broken_branches(proposed: str):
    assert drops_conditional(proposed, current=DEFAULT_WALDO_TOOL_USAGE)


@pytest.mark.parametrize(
    "proposed",
    [
        DEFAULT_WALDO_TOOL_USAGE,
        # merged into one block per name
        "<<<[[has_search_tools]]Use attached tools for easy lookups. Be persistent.>>>\n"
        "<<<[[no_search_tools]]No search tools. Call `[[waldo_discover_tool_name]]`.>>>\n"
        "Write the final answer when done.",
        # reordered, extra text
        "Always route first.\n<<<[[no_search_tools]]B>>>\n<<<[[has_search_tools]]A>>>\n<<<[[has_search_tools]]C>>>",
    ],
)
def test_drops_conditional_accepts_variants_that_keep_every_name(proposed: str):
    assert not drops_conditional(proposed, current=DEFAULT_WALDO_TOOL_USAGE)


def test_drops_conditional_is_noop_for_plain_modules():
    assert not drops_conditional("anything", current="plain text module")


def test_markup_rule_names_the_conditionals():
    rule = markup_rule_for(DEFAULT_WALDO_TOOL_USAGE)
    assert "<<<[[has_search_tools]] ... >>>" in rule
    assert "<<<[[no_search_tools]] ... >>>" in rule
    assert "discarded" in rule


# --- Waldo model alias + harness scParams ---

SEED_PATH = Path(__file__).resolve().parents[1] / "data" / "waldo_seed_candidate.json"


def test_parse_waldo_alias():
    assert parse_waldo_alias("waldo") == WALDO_DEFAULT_MODEL
    assert parse_waldo_alias("waldo:OPEN_AI:GPT6_LUNA:none") == "OPEN_AI:GPT6_LUNA:none"
    assert parse_waldo_alias("waldo:BASETEN:WALDO") == "BASETEN:WALDO"
    assert parse_waldo_alias("gpt") is None
    assert parse_waldo_alias("waldo_x") is None
    with pytest.raises(ValueError, match="PROVIDER:MODEL"):
        parse_waldo_alias("waldo:")
    with pytest.raises(ValueError, match="PROVIDER:MODEL"):
        parse_waldo_alias("waldo:GPT6_LUNA")


def test_waldo_sc_params_use_waldo_harness_not_coding_harness():
    runner = ALRunner(evalcli=MagicMock())
    prompt_param = compile_encoded_prompt(
        {WALDO_TOOL_USAGE_KEY: "<<<[[has_search_tools]]A>>><<<[[no_search_tools]]B>>>"}
    )
    params = runner._build_sc_params("waldo:OPEN_AI:GPT6_LUNA:none", prompt_param)
    assert params.startswith(WALDO_HARNESS_SC_PARAMS + ",")
    assert f",{WALDO_ENABLE_SC_PARAM}," in params
    assert ",co.lo.icpo.waldo_model=OPEN_AI:GPT6_LUNA:none," in params
    assert "llmo.per_prompt_overrides.waldo_system=" in params
    # Coding-harness-only keys must not leak in.
    assert "co.lo.cao.agentic_loop_sc_params=" not in params
    assert "co.lo.oai_model_for_agentic_loop=" not in params
    # Waldo knobs from the production eval are present.
    for key in (
        "co.lo.icpo.max_loops=",
        "co.lo.icpo.waldo_timeout_seconds=",
        "co.lo.icpo.skip_waldo_escalation_discovery=",
    ):
        assert key in params


def test_waldo_harness_has_no_per_run_keys():
    for key in ("should_run_waldo_qe", "co.lo.icpo.waldo_model=", "waldo_system=", "query_timestamp_override"):
        assert key not in WALDO_HARNESS_SC_PARAMS


def test_bare_waldo_alias_uses_default_model():
    params = ALRunner(evalcli=MagicMock())._build_sc_params("waldo", "")
    assert params.endswith(f",co.lo.icpo.waldo_model={WALDO_DEFAULT_MODEL}")


def test_non_waldo_models_unchanged():
    params = ALRunner(evalcli=MagicMock())._build_sc_params("gpt", "")
    assert "should_run_waldo_qe" not in params
    assert "co.lo.cao.agentic_loop_sc_params=" in params


def test_live_seed_round_trips_and_keeps_conditionals():
    seed = json.loads(SEED_PATH.read_text())
    assert set(seed) == {WALDO_SYSTEM_KEY, WALDO_TOOL_USAGE_KEY}
    assert seed[WALDO_SYSTEM_KEY].count(WALDO_TOOL_USAGE_SLOT) == 1
    assert "### Tool Usage Guidelines\n" + WALDO_TOOL_USAGE_SLOT in seed[WALDO_SYSTEM_KEY]
    counts = conditional_counts(seed[WALDO_TOOL_USAGE_KEY])
    assert set(counts) == set(WALDO_TOOL_USAGE_CONDITIONALS)
    compiled = compile_waldo_system_prompt(seed)
    assert WALDO_TOOL_USAGE_SLOT not in compiled
    assert "### Glean Search Argument Construction" in compiled
    # Live prompt renders discover literally; the stock template placeholder must be gone.
    assert "[[waldo_discover_tool_name]]" not in compiled
    # The runner's seed validation accepts it.
    built = _seed_for_editable_modules(seed, [WALDO_TOOL_USAGE_KEY])
    assert built[WALDO_SYSTEM_KEY] == seed[WALDO_SYSTEM_KEY]


def test_live_seed_fits_module_budgets():
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    seed = json.loads(SEED_PATH.read_text())
    assert len(enc.encode(seed[WALDO_TOOL_USAGE_KEY])) < MODULE_TOKEN_BUDGETS[WALDO_TOOL_USAGE_KEY]
    assert len(enc.encode(compile_waldo_system_prompt(seed))) < MODULE_TOKEN_BUDGETS[WALDO_SYSTEM_KEY]


# --- eval: section -> EvalHarness -> ALRunner ---


def test_waldo_config_eval_section_matches_generated_harness():
    from glean_gepa.experiment_config import eval_harness, load_experiment_config

    config = load_experiment_config(
        Path(__file__).resolve().parents[1] / "src/glean_gepa/configs/teacher_student_waldo.yaml"
    )
    harness = eval_harness(config)
    assert harness.runner_type == "GLEAN_CHAT"
    assert harness.sc_params == WALDO_HARNESS_SC_PARAMS


def test_eval_section_parsing_and_validation(tmp_path):
    import yaml

    from glean_gepa.experiment_config import ExperimentConfigError, eval_harness, load_experiment_config

    base = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "src/glean_gepa/configs/teacher_student_waldo.yaml").read_text()
    )

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


def test_runner_uses_harness_sc_params_and_runner_type():
    from glean_gepa.adapter_types import EvalHarness

    evalcli = MagicMock()
    evalcli.create_eval_run.return_value = "ev-1"
    runner = ALRunner(
        evalcli=evalcli,
        harness=EvalHarness(runner_type="GLEAN_CHAT", sc_params="x.y=1,x.z=2"),
    )
    runner._resolve_cached_eval = lambda key: None  # type: ignore[method-assign]
    runner.start("waldo", "", "set", "v1", ["scio-prod"])
    kwargs = evalcli.create_eval_run.call_args.kwargs
    assert kwargs["runner_type"] == "GLEAN_CHAT"
    assert kwargs["sc_params"].startswith("x.y=1,x.z=2," + WALDO_ENABLE_SC_PARAM + ",")
    assert WALDO_HARNESS_SC_PARAMS not in kwargs["sc_params"]
    assert kwargs["eval_params"] == "experimental_queue=eval-experimental-2,gleanchat_agent=ADVANCED"


def test_runner_without_harness_keeps_defaults():
    evalcli = MagicMock()
    evalcli.create_eval_run.return_value = "ev-1"
    runner = ALRunner(evalcli=evalcli)
    runner._resolve_cached_eval = lambda key: None  # type: ignore[method-assign]
    runner.start("waldo", "", "set", "v1", ["scio-prod"])
    kwargs = evalcli.create_eval_run.call_args.kwargs
    assert "runner_type" not in kwargs  # evalcli's own default (GLEAN_CHAT) applies
    assert kwargs["sc_params"].startswith(WALDO_HARNESS_SC_PARAMS + ",")
    assert kwargs["eval_params"] == "experimental_queue=eval-experimental-2,gleanchat_agent=ADVANCED"
    runner.start("gpt", "", "set", "v1", ["scio-prod"])
    assert "co.lo.cao.agentic_loop_sc_params=" in evalcli.create_eval_run.call_args.kwargs["sc_params"]
