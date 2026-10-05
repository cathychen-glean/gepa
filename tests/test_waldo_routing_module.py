from __future__ import annotations

import json
from pathlib import Path

import pytest

from glean_gepa.experiment_config import load_experiment_config, runner_arg_defaults
from glean_gepa.objectives.escalation_match import WALDO_ROUTING_FRAME, EscalationMatchObjective
from glean_gepa.prompt import compile_waldo_system_override
from glean_gepa.prompt_constants import MODULE_TOKEN_BUDGETS
from glean_gepa.runner import _parse_editable_modules, _seed_for_editable_modules
from glean_gepa.waldo_prompt_constants import (
    DEFAULT_WALDO_ROUTING,
    WALDO_ROUTING_KEY,
    WALDO_ROUTING_SLOT,
    WALDO_ROUTING_TOKEN_BUDGET,
    WALDO_SYSTEM_KEY,
    WALDO_TOOL_USAGE_KEY,
    compile_waldo_system_prompt,
)

DATA = Path(__file__).resolve().parents[1] / "data"


def _seed(name: str) -> dict[str, str]:
    return json.loads((DATA / name).read_text())


def test_routing_seed_compiles_to_the_live_prompt():
    legacy = _seed("waldo_seed_candidate.json")
    routing = _seed("waldo_routing_seed_candidate.json")

    assert routing[WALDO_SYSTEM_KEY].startswith(WALDO_ROUTING_SLOT + "\n\n### Available Tools")
    assert routing[WALDO_ROUTING_KEY].startswith("## First-Action Decision")
    assert routing[WALDO_TOOL_USAGE_KEY] == legacy[WALDO_TOOL_USAGE_KEY]
    assert compile_waldo_system_prompt(routing) == compile_waldo_system_prompt(legacy)


def test_stock_template_fills_routing_slot_with_default():
    compiled = compile_waldo_system_prompt({})

    assert WALDO_ROUTING_SLOT not in compiled
    assert compiled.startswith(DEFAULT_WALDO_ROUTING + "\n\n### Available Tools")


def test_stripped_routing_rewrite_keeps_heading_separated():
    routing = _seed("waldo_routing_seed_candidate.json")
    rewritten = {**routing, WALDO_ROUTING_KEY: "## First-Action Decision\nCall discover.  \n"}

    assert "Call discover.\n\n### Available Tools" in compile_waldo_system_prompt(rewritten)


def test_seed_pins_frozen_waldo_siblings_from_seed_file():
    raw = _seed("waldo_routing_seed_candidate.json")

    seed = _seed_for_editable_modules(raw, [WALDO_ROUTING_KEY])

    assert seed[WALDO_SYSTEM_KEY] == raw[WALDO_SYSTEM_KEY]
    assert seed[WALDO_ROUTING_KEY] == raw[WALDO_ROUTING_KEY]
    assert seed[WALDO_TOOL_USAGE_KEY] == raw[WALDO_TOOL_USAGE_KEY]


def test_legacy_seed_unchanged_for_tool_usage_and_rejected_for_routing():
    legacy = _seed("waldo_seed_candidate.json")

    tool_usage_seed = _seed_for_editable_modules(legacy, [WALDO_TOOL_USAGE_KEY])
    assert WALDO_ROUTING_KEY not in tool_usage_seed
    assert compile_waldo_system_prompt(tool_usage_seed) == compile_waldo_system_prompt(legacy)

    with pytest.raises(
        SystemExit, match=r"WALDO_ROUTING is editable but the seed WALDO_SYSTEM has no \{WALDO_ROUTING\}"
    ):
        _seed_for_editable_modules(legacy, [WALDO_ROUTING_KEY])


def test_routing_module_is_editable_budgeted_and_triggers_override():
    assert _parse_editable_modules(WALDO_ROUTING_KEY) == [WALDO_ROUTING_KEY]
    assert MODULE_TOKEN_BUDGETS[WALDO_ROUTING_KEY] == WALDO_ROUTING_TOKEN_BUDGET
    routing_len = len(_seed("waldo_routing_seed_candidate.json")[WALDO_ROUTING_KEY]) // 4
    assert routing_len < WALDO_ROUTING_TOKEN_BUDGET

    override = compile_waldo_system_override({WALDO_ROUTING_KEY: "route"})
    assert override.startswith("llmo.per_prompt_overrides.waldo_system=")


def test_escalation_objective_briefs_the_routing_module():
    prompt = EscalationMatchObjective(bigquery_client=object()).reflection_prompt(WALDO_ROUTING_KEY)

    assert WALDO_ROUTING_FRAME in prompt
    assert "SAW_INSUFFICIENT_TOOLS" in prompt


def test_escalation_config_edits_only_routing():
    defaults = runner_arg_defaults(load_experiment_config("teacher_student_waldo_escalation"))

    assert Path(defaults["seed_candidate"]) == Path("data/waldo_routing_seed_candidate.json")
    assert defaults["editable_modules"] == WALDO_ROUTING_KEY
    seed = _seed_for_editable_modules(_seed("waldo_routing_seed_candidate.json"), [WALDO_ROUTING_KEY])
    frozen = sum(len(seed[key]) // 4 for key in (WALDO_SYSTEM_KEY, WALDO_TOOL_USAGE_KEY))
    assert frozen + WALDO_ROUTING_TOKEN_BUDGET < defaults["global_token_cap"]
