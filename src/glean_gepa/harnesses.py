"""Eval harnesses: the scParams preset, model routing, and Glean Chat agent mode for a model alias.

``coding`` runs the full agent loop on the Coding Harness preset. ``waldo`` runs the Waldo
router in front of it, selected by a ``waldo[:PROVIDER:MODEL[:effort]]`` alias. Prompt
targets name the harnesses they render under in their ``target.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass

from glean_gepa.coding_harness_params import CODING_HARNESS_SC_PARAMS
from glean_gepa.waldo_harness_params import WALDO_HARNESS_SC_PARAMS

# Models that use the Coding Harness default (no oai_model_for_agentic_loop override).
DEFAULT_AGENTIC_LOOP_MODELS = frozenset({"gpt", "fast"})
# CLI aliases -> co.lo.oai_model_for_agentic_loop ChatModel enum. Older Claude Sonnet
# enums are redlisted off the Coding Harness by QE; only 4.6+ stays on it.
AGENTIC_LOOP_MODEL_OVERRIDES = {
    "claude_sonnet": "CLAUDE_4_6_SONNET_20260217",
    "claude_opus": "CLAUDE_5_OPUS",
    "gpt6_luna": "GPT6_LUNA",
    "gpt6_sol": "GPT6_SOL",
    # GPT-6.1 Sol. "high" is the reasoning effort, pinned below, not a separate enum.
    "gpt6_1_sol_high": "GPT6_1_SOL",
}
# Alias -> reasoning effort override. GPT6_LUNA is already high by default; GPT6_SOL is not.
AGENTIC_LOOP_REASONING_EFFORT = {
    "gpt6_1_sol_high": "high",
}

# Waldo is off in eval runs unless the model alias is `waldo` (default model) or
# `waldo:PROVIDER:MODEL[:effort]`. Teacher and student differ only in `waldo_model` and
# the prompt override.
WALDO_MODEL_ALIAS = "waldo"
WALDO_DEFAULT_MODEL = "BASETEN:WALDO"
WALDO_ENABLE_SC_PARAM = "co.lo.icpo.should_run_waldo_qe=true"
WALDO_MODEL_SC_PARAM = "co.lo.icpo.waldo_model"
# scio's _waldo_skip_reason skips Waldo for the ADVANCED (thinking) agent. AUTO is what
# production Glean Chat and the Coding Harness preset send.
WALDO_GLEANCHAT_AGENT = "AUTO"


def parse_waldo_alias(model: str) -> str | None:
    """Return the Waldo model spec for a ``waldo[:PROVIDER:MODEL[:effort]]`` alias, else ``None``.

    ``waldo`` alone selects :data:`WALDO_DEFAULT_MODEL`. Anything after the first
    colon is passed through to ``co.lo.icpo.waldo_model`` unchanged.
    """
    if model == WALDO_MODEL_ALIAS:
        return WALDO_DEFAULT_MODEL
    prefix = WALDO_MODEL_ALIAS + ":"
    if model.startswith(prefix):
        spec = model[len(prefix) :]
        if not spec or ":" not in spec:
            raise ValueError(f"Waldo model must be PROVIDER:MODEL[:effort], got {spec!r} in {model!r}")
        return spec
    return None


@dataclass(frozen=True)
class Harness:
    name: str
    #: Base scParams preset. An experiment's ``eval.sc_params`` replaces it.
    sc_params: str

    def accepts(self, model: str) -> bool:
        raise NotImplementedError

    def model_sc_params(self, model: str) -> list[str]:
        """Per-model scParams appended after the preset."""
        raise NotImplementedError

    def gleanchat_agent(self, model: str) -> str:
        """EvalCLI ``gleanchat_agent`` mode."""
        raise NotImplementedError

    def cache_tag(self, model: str) -> str:
        """Suffix for the eval cache's prompt key, for settings the prompt hash does not capture."""
        return ""


class CodingHarness(Harness):
    def accepts(self, model: str) -> bool:
        return model in AGENTIC_LOOP_MODEL_OVERRIDES or model in DEFAULT_AGENTIC_LOOP_MODELS

    def model_sc_params(self, model: str) -> list[str]:
        params: list[str] = []
        override = AGENTIC_LOOP_MODEL_OVERRIDES.get(model)
        if override:
            params.append(f"co.lo.oai_model_for_agentic_loop={override}")
        elif model not in DEFAULT_AGENTIC_LOOP_MODELS:
            raise ValueError(f"Unknown model: {model}")
        effort = AGENTIC_LOOP_REASONING_EFFORT.get(model)
        if effort:
            # Non-auto path. The per-model key is the ChatModel enum (model_id).
            params.append(f"co.lo.advanced_mode_reasoning_effort={effort}")
            if override:
                params.append(f"co.lo.advanced_mode_model_reasoning_overrides={override}:{effort}")
            # Auto-routing path. The harness sets use_auto_mode, and that path
            # ignores the advanced-mode fields above.
            for tier in ("economical", "balanced", "frontier"):
                params.append(f"co.lo.mro.{tier}.driver_reasoning_effort={effort}")
        return params

    def gleanchat_agent(self, model: str) -> str:
        return "FAST" if model == "fast" else "ADVANCED"


class WaldoHarness(Harness):
    def accepts(self, model: str) -> bool:
        return parse_waldo_alias(model) is not None

    def model_sc_params(self, model: str) -> list[str]:
        return [WALDO_ENABLE_SC_PARAM, f"{WALDO_MODEL_SC_PARAM}={parse_waldo_alias(model)}"]

    def gleanchat_agent(self, model: str) -> str:
        return WALDO_GLEANCHAT_AGENT

    def cache_tag(self, model: str) -> str:
        # Waldo runs created under ADVANCED skipped Waldo on every entry; keying on the
        # agent mode keeps them from being reused.
        return f"|gleanchat_agent={self.gleanchat_agent(model)}"


CODING_HARNESS = CodingHarness("coding", CODING_HARNESS_SC_PARAMS)
WALDO_HARNESS = WaldoHarness("waldo", WALDO_HARNESS_SC_PARAMS)
HARNESSES: dict[str, Harness] = {harness.name: harness for harness in (CODING_HARNESS, WALDO_HARNESS)}


def harness_for_model(model: str) -> Harness:
    """The harness a model alias runs under. Unknown aliases fall to ``coding``, which rejects them."""
    return WALDO_HARNESS if WALDO_HARNESS.accepts(model) else CODING_HARNESS


def gleanchat_agent(model: str) -> str:
    """EvalCLI ``gleanchat_agent`` mode for a model alias."""
    return harness_for_model(model).gleanchat_agent(model)


def drop_sc_params(preset: str, drops: tuple[str, ...]) -> str:
    """``preset`` without the top-level ``key=value`` entries in ``drops``.

    Splitting on commas is safe: nested values keep their commas percent-encoded.
    """
    if not drops:
        return preset
    return ",".join(param for param in preset.split(",") if param not in drops)
