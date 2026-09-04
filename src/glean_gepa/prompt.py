"""Prompt encoding for the Glean assistant."""

from __future__ import annotations

import keyword
import re
from base64 import b64encode, urlsafe_b64encode
from collections.abc import Mapping, Sequence
from typing import Any

from glean_gepa.prompt_constants import (
    CORE_TOOL_DESCRIPTIONS,
    CORE_TOOL_KEYS,
    CORE_TOOLS,
    DEFAULT_FULL_PROMPT,
    DEFAULT_RULES_EXT,
    DEFAULT_WRITING_CODE,
    FULL_PROMPT_KEY,
    RULES_EXT_KEY,
    TOOL_DESCRIPTION_OVERRIDES_PARAM,
    WRITING_CODE_KEY,
)
from glean_gepa.tool_match_util import first_tool_mismatch_pair, select_first_tool_mismatch_groups

_VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")


def materialize_system_prompt(candidate: dict[str, str]) -> str:
    """Inject writing-code into one system prompt, leaving ``{RULES_EXT}`` for compile time.

    Used when ``FULL_PROMPT`` is an editable GEPA module: after this, there is
    no ``{WRITING_CODE}`` slot and GEPA iterates on the full prompt as a single string.
    ``RULES_EXT`` stays a placeholder so later children can rewrite those bullets
    without re-materializing Writing Code.
    """
    template = candidate.get(FULL_PROMPT_KEY, DEFAULT_FULL_PROMPT)
    writing_code = candidate.get(WRITING_CODE_KEY, DEFAULT_WRITING_CODE)
    return template.replace("{WRITING_CODE}", writing_code)


def compile_system_prompt(candidate: dict[str, str]) -> str:
    """Fill the system prompt from a candidate.

    If ``WRITING_CODE`` is present, splice it into the template. Otherwise use
    ``FULL_PROMPT`` as-is (already a materialized full prompt).
    ``RULES_EXT`` is always spliced when the ``{RULES_EXT}`` slot remains.
    Use replace (not str.format) so braces inside module text are preserved.
    """
    template = candidate.get(FULL_PROMPT_KEY, DEFAULT_FULL_PROMPT)
    if WRITING_CODE_KEY in candidate:
        template = template.replace("{WRITING_CODE}", candidate[WRITING_CODE_KEY])
    if "{RULES_EXT}" in template:
        template = template.replace("{RULES_EXT}", candidate.get(RULES_EXT_KEY, DEFAULT_RULES_EXT).strip())
    return template


def compile_encoded_prompt(candidate: dict[str, str]) -> str:
    """Compile candidate modules into encoded scParams fragments.

    Always includes the coding-agent system prompt. Core-tool description
    overrides are appended when the candidate has those modules.
    """
    encoded_system_prompt = b64encode(compile_system_prompt(candidate).encode("utf-8")).decode("ascii")
    parts = ["llmo.per_prompt_overrides.coding_agent_loop_system=" + encoded_system_prompt]
    tool_overrides = compile_tool_description_overrides(candidate)
    if tool_overrides:
        parts.append(tool_overrides)
    return ",".join(parts)


def candidate_module_names(editable_modules: Sequence[str]) -> list[str]:
    """Editable modules plus core-tool description keys, de-duplicated in that order."""
    return list(dict.fromkeys([*editable_modules, *CORE_TOOLS]))


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


def with_core_tool_defaults(prompt_modules: Mapping[str, str]) -> dict[str, str]:
    """Copy ``prompt_modules`` and fill any missing core-tool descriptions from stock text."""
    merged = dict(prompt_modules)
    for key, text in CORE_TOOL_DESCRIPTIONS.items():
        merged.setdefault(key, text)
    return merged


def high_signal_core_tool_keys(trajectories: Sequence[Any] | None) -> list[str]:
    """Core-tool keys that appear in the reflection high-signal first-tool mismatch groups."""
    mismatch_keys: list[tuple[str, str] | None] = []
    for trajectory in trajectories or []:
        output = trajectory.get("output") if isinstance(trajectory, Mapping) else None
        if not isinstance(output, Mapping):
            mismatch_keys.append(None)
            continue
        mismatch_keys.append(
            first_tool_mismatch_pair(output.get("teacher_tool_events"), output.get("student_tool_events"))
        )
    _indices, groups = select_first_tool_mismatch_groups(mismatch_keys)
    found: list[str] = []
    for teacher_tool, student_tool, _count in groups:
        for name in (teacher_tool, student_tool):
            key = tool_description_override_key(name)
            if key in CORE_TOOL_KEYS and key not in found:
                found.append(key)
    return found


def compile_tool_description_overrides(candidate: Mapping[str, str]) -> str:
    """Encode core-tool modules as ``co.pyagents_tool_description_overrides=key:b64;...``.

    Empty string means evals keep stock descriptions for every tool. A subset is
    allowed; omitted keys keep stock text.
    """
    segments: list[str] = []
    for key in CORE_TOOLS:
        text = candidate.get(key, "")
        if not text:
            continue
        encoded = urlsafe_b64encode(text.encode("utf-8")).decode("ascii")
        segments.append(f"{key}:{encoded}")
    if not segments:
        return ""
    return TOOL_DESCRIPTION_OVERRIDES_PARAM + "=" + ";".join(segments)
