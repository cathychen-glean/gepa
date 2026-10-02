"""Prompt encoding for the Glean assistant."""

from __future__ import annotations

import keyword
import re
from base64 import urlsafe_b64encode
from collections.abc import Mapping, Sequence

from glean_gepa.prompt_constants import (
    CORE_TOOL_DESCRIPTIONS,
    CORE_TOOL_KEYS,
    CORE_TOOLS,
    DEFAULT_EXECUTION_DISCIPLINE,
    DEFAULT_FULL_PROMPT,
    DEFAULT_RULES_EXT,
    DEFAULT_WRITING_CODE,
    EXECUTION_DISCIPLINE_KEY,
    FULL_PROMPT_KEY,
    RULES_EXT_KEY,
    TOOL_DESCRIPTION_OVERRIDES_PARAM,
    WRITING_CODE_KEY,
)
from glean_gepa.waldo_prompt_constants import (
    WALDO_SYSTEM_KEY,
    WALDO_SYSTEM_OVERRIDE_PARAM,
    WALDO_TOOL_USAGE_KEY,
    compile_waldo_system_prompt,
)

_VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")
_RULES_EXT_SLOT = re.compile(r"\{RULES_EXT\}[ \t]*\n?")


def materialize_system_prompt(candidate: dict[str, str]) -> str:
    """Inject writing-code into one system prompt, leaving the other slots for compile time.

    Used to freeze the system prompt for runs that do not edit ``WRITING_CODE``:
    after this there is no ``{WRITING_CODE}`` slot left to fill at compile time.
    ``RULES_EXT`` and ``EXECUTION_DISCIPLINE`` stay placeholders so children editing
    those do not re-materialize Writing Code.
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
    if "{EXECUTION_DISCIPLINE}" in template:
        execution_discipline = candidate.get(EXECUTION_DISCIPLINE_KEY, "").strip() or DEFAULT_EXECUTION_DISCIPLINE
        template = template.replace("{EXECUTION_DISCIPLINE}", execution_discipline)
    if "{RULES_EXT}" in template:
        rules_ext = candidate.get(RULES_EXT_KEY, DEFAULT_RULES_EXT).strip()
        template = template.replace("{RULES_EXT}", rules_ext) if rules_ext else _RULES_EXT_SLOT.sub("", template)
    return template


def compile_encoded_prompt(candidate: dict[str, str]) -> str:
    """Compile candidate modules into encoded scParams fragments.

    Always includes the coding-agent system prompt. Core-tool description
    overrides are appended when the candidate has those modules. The Waldo
    system prompt override is appended when the candidate edits a Waldo module.

    The system prompt must be URL-safe base64. QE parses ``sc=`` with
    ``url.QueryUnescape``, which turns ``+`` into space; standard base64 then
    fails to decode and the override is dropped.
    """
    encoded_system_prompt = urlsafe_b64encode(compile_system_prompt(candidate).encode("utf-8")).decode("ascii")
    parts = ["llmo.per_prompt_overrides.coding_agent_loop_system=" + encoded_system_prompt]
    tool_overrides = compile_tool_description_overrides(candidate)
    if tool_overrides:
        parts.append(tool_overrides)
    waldo_override = compile_waldo_system_override(candidate)
    if waldo_override:
        parts.append(waldo_override)
    return ",".join(parts)


def compile_waldo_system_override(candidate: Mapping[str, str]) -> str:
    """Encode the compiled Waldo prompt as ``llmo.per_prompt_overrides.waldo_system=<b64>``.

    Empty string when the candidate edits neither ``WALDO_SYSTEM`` nor
    ``WALDO_TOOL_USAGE``, so evals keep the stock template. Scio renders the
    ``[[...]]`` and ``<<<[[...]]>>>`` markers in the override at request time.
    """
    if not (candidate.get(WALDO_SYSTEM_KEY) or candidate.get(WALDO_TOOL_USAGE_KEY)):
        return ""
    text = compile_waldo_system_prompt(candidate)
    encoded = urlsafe_b64encode(text.encode("utf-8")).decode("ascii")
    return WALDO_SYSTEM_OVERRIDE_PARAM + "=" + encoded


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


def with_core_tool_defaults(prompt_modules: Mapping[str, str]) -> dict[str, str]:
    """Copy ``prompt_modules`` and fill any missing core-tool descriptions from stock text."""
    merged = dict(prompt_modules)
    for key, text in CORE_TOOL_DESCRIPTIONS.items():
        merged.setdefault(key, text)
    return merged


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
