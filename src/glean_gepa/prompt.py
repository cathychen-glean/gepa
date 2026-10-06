"""Prompt encoding for the Glean assistant."""

from __future__ import annotations

import keyword
import re
from collections.abc import Mapping, Sequence

from glean_gepa.prompt_constants import CORE_TOOL_KEYS
from glean_gepa.prompt_targets import compile_overrides, prompt_targets

_VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")


def compile_encoded_prompt(candidate: Mapping[str, str]) -> str:
    """Compile candidate modules into the scParams fragments every eval of it sends.

    One ``llmo.per_prompt_overrides.<template>=<b64>`` per prompt target the candidate
    carries (the coding-agent prompt always), plus core-tool description overrides.
    """
    return compile_overrides(candidate)


def candidate_module_names(editable_modules: Sequence[str]) -> list[str]:
    """Return the editable candidate modules, de-duplicated in order."""
    return list(dict.fromkeys(editable_modules))


def tool_description_override_key(span_name: str) -> str:
    """Map an Execute Action span name to a ``pyagents_tool_description_overrides`` key."""
    value = span_name.lower()
    if _VALID_IDENTIFIER.fullmatch(value) and not keyword.iskeyword(value):
        return value
    sanitized = _NON_ALNUM.sub("_", value).strip("_")
    if not sanitized:
        return value
    if sanitized[0].isdigit():
        sanitized = "_" + sanitized
    if keyword.iskeyword(sanitized):
        sanitized = sanitized + "_"
    return sanitized


def is_core_tool_span(span_name: str) -> bool:
    """True when an Execute Action span maps to an editable core-tool description."""
    return bool(span_name) and tool_description_override_key(span_name) in CORE_TOOL_KEYS


def compile_tool_description_overrides(candidate: Mapping[str, str]) -> str:
    """Encode core-tool modules as ``co.pyagents_tool_description_overrides=key:b64;...``.

    Empty string means evals keep stock descriptions for every tool. A subset is
    allowed; omitted keys keep stock text.
    """
    return prompt_targets()["core_tool_descriptions"].override(candidate)
