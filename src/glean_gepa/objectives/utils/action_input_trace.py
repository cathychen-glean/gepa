"""Fetch per-entry tool-call payloads from detailed eval traces.

The ``scrubbed_agentspan`` table drops ``span_info.inputs`` before logging, so the
tool-call payload (the search string or shell command the agent issued) is *not*
queryable from BigQuery — a query against that field always comes back empty. The
detailed trace served by ``evalcli analyze trace`` still carries it under each
``Execute Action`` span's ``attributes.input``, which is the same surviving source
the Shell objective already uses.

This module locates traces from scrub-safe identifiers (trace id, deployment,
span timestamps) that survive scrubbing, fetches them for a bounded set of
entries, and extracts each span's tool payload from whichever envelope it used.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from glean_gepa.objectives.utils.agentspan_query import action_input_tuple

_EXECUTE_ACTION_PREFIX = "Execute Action: "
_CALL_METADATA_KEYS = frozenset({"id", "action", "tool_id", "tool_name"})
TRACE_WINDOW_LEAD_MS = 3_600_000
TRACE_WINDOW_TRAIL_MS = 60_000
DEFAULT_MAX_TRACE_FETCHES = 60
# Customer deployments 403 on ``analyze trace``. Reflection only reads Action Inputs from scio-prod
INTERNAL_TRACE_DEPLOYMENT_ID = "scio-prod"


@dataclass(frozen=True)
class TraceActionInputLocator:
    """Scrub-safe coordinates for fetching one entry's detailed trace."""

    entry_id: str
    deployment_id: str
    trace_id: str
    start_ms: int
    end_ms: int


def _typed_str_value(value: Any) -> str | None:
    """Unwrap an OTel-style ``{'strValue': ...}`` attribute, or a bare string."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        str_value = value.get("strValue")
        return str_value if isinstance(str_value, str) else None
    return None


def _as_json_text(value: Any) -> str:
    """Render a payload as compact JSON, passing text through unchanged."""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return ""


def _glean_tool_arguments(payload: dict[str, Any]) -> str:
    """Unwrap the nested ``{"input": "<json>"}`` envelope Glean tools use."""
    inner = payload.get("input")
    if isinstance(inner, str):
        try:
            inner = json.loads(inner)
        except (TypeError, ValueError, json.JSONDecodeError):
            return ""
    if not isinstance(inner, dict):
        return ""
    arguments = {
        key: value for key, value in inner.items() if key not in _CALL_METADATA_KEYS and value not in (None, "", {}, [])
    }
    return _as_json_text(arguments) if arguments else ""


def _span_tool_payload(raw_input: str) -> str:
    """The tool's arguments from one ``Execute Action`` span's ``input`` attribute."""
    try:
        payload = json.loads(raw_input)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    if "action_input" in payload:
        return _as_json_text(payload["action_input"])
    return _glean_tool_arguments(payload)


def extract_trace_tool_inputs(
    detailed_trace: Any,
    *,
    skip_tools: frozenset[str] = frozenset(),
    limit: int | None = None,
) -> tuple[tuple[str, str], ...]:
    """Ordered, de-duplicated ``(tool_name, payload)`` pairs from a detailed trace.

    Reads every ``Execute Action:`` span's ``attributes.input`` JSON and resolves the
    tool's arguments from whichever envelope it used, preserving span order. Tools in
    ``skip_tools`` (matched on the name after the ``Execute Action:`` prefix) are
    ignored.
    """
    if not isinstance(detailed_trace, dict):
        return ()
    spans = (detailed_trace.get("trace") or {}).get("spans") or []
    ordered: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for span in spans:
        if not isinstance(span, dict):
            continue
        name = span.get("name") or ""
        if not name.startswith(_EXECUTE_ACTION_PREFIX):
            continue
        tool = name[len(_EXECUTE_ACTION_PREFIX) :]
        if tool in skip_tools:
            continue
        raw_input = _typed_str_value((span.get("attributes") or {}).get("input"))
        if not raw_input:
            continue
        payload = _span_tool_payload(raw_input)
        if not payload:
            continue
        pair = (tool, payload)
        if pair in seen:
            continue
        seen.add(pair)
        ordered.append(pair)
        if limit is not None and len(ordered) >= limit:
            break
    return tuple(ordered)


def extract_trace_action_inputs(
    detailed_trace: Any,
    *,
    skip_tools: frozenset[str] = frozenset(),
    limit: int | None = None,
) -> tuple[str, ...]:
    """Ordered, de-duplicated tool payloads from a detailed trace, without tool names."""
    pairs = extract_trace_tool_inputs(detailed_trace, skip_tools=skip_tools)
    return action_input_tuple([payload for _, payload in pairs], limit=limit)


def build_trace_locator(
    *,
    entry_id: str,
    deployment_id: Any,
    trace_id: Any,
    min_start_ms: Any,
    max_start_ms: Any,
) -> TraceActionInputLocator | None:
    """Assemble a locator from a per-entry row, or ``None`` when unfetchable."""
    if not (entry_id and deployment_id and trace_id):
        return None
    try:
        start = int(min_start_ms)
        end = int(max_start_ms)
    except (TypeError, ValueError):
        return None
    return TraceActionInputLocator(
        entry_id=str(entry_id),
        deployment_id=str(deployment_id),
        trace_id=str(trace_id),
        start_ms=start - TRACE_WINDOW_LEAD_MS,
        end_ms=end + TRACE_WINDOW_TRAIL_MS,
    )


def trace_locators_for_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    entry_ids: Collection[str] | None = None,
    role: str | None = None,
) -> list[TraceActionInputLocator]:
    """Locators for high-signal rows.

    ``role`` selects paired columns such as ``student_trace_id``. Omit it when
    the row already uses ``trace_id``, ``deployment_id``, ``min_start_ms``, and
    ``max_start_ms``.
    """
    prefix = f"{role}_" if role else ""
    allowed = None if entry_ids is None else set(entry_ids)
    locators: list[TraceActionInputLocator] = []
    for row in rows:
        entry_id = str(row.get("entry_id") or "")
        if allowed is not None and entry_id not in allowed:
            continue
        locator = build_trace_locator(
            entry_id=entry_id,
            deployment_id=row.get(f"{prefix}deployment_id"),
            trace_id=row.get(f"{prefix}trace_id"),
            min_start_ms=row.get(f"{prefix}min_start_ms"),
            max_start_ms=row.get(f"{prefix}max_start_ms"),
        )
        if locator is not None:
            locators.append(locator)
    return locators


def _iter_entry_traces(
    evalcli: Any,
    locators: Iterable[TraceActionInputLocator],
    *,
    max_fetches: int,
    role_label: str,
) -> Iterator[tuple[str, Any]]:
    """Yield ``(entry_id, detailed_trace)`` for fetchable locators.

    Skips locators that are not on ``scio-prod``: customer deployments reject
    ``analyze trace`` with 403, and validation scoring does not need payloads.
    """
    get_trace = getattr(evalcli, "get_analysis_trace", None)
    if not callable(get_trace):
        return
    fetched = 0
    skipped_external = 0
    label = f"{role_label} " if role_label else ""
    for locator in locators:
        if locator.deployment_id != INTERNAL_TRACE_DEPLOYMENT_ID:
            skipped_external += 1
            continue
        if fetched >= max_fetches:
            break
        fetched += 1
        try:
            trace = get_trace(
                deployment_id=locator.deployment_id,
                trace_id=locator.trace_id,
                start_time_millis=locator.start_ms,
                end_time_millis=locator.end_ms,
            )
        except Exception as exc:
            print(f"[Action Inputs] Failed to fetch {label}trace for entry {locator.entry_id}: {exc}")
            continue
        yield locator.entry_id, trace
    if skipped_external:
        print(
            f"[Action Inputs] Skipping {skipped_external} {label}traces on non-"
            f"{INTERNAL_TRACE_DEPLOYMENT_ID} deployments"
        )


def fetch_action_inputs_by_entry(
    evalcli: Any,
    locators: Iterable[TraceActionInputLocator],
    *,
    skip_tools: frozenset[str] = frozenset(),
    limit: int | None = None,
    max_fetches: int = DEFAULT_MAX_TRACE_FETCHES,
    role_label: str = "",
) -> dict[str, tuple[str, ...]]:
    """Fetch detailed traces and return ``{entry_id: tool payloads}``."""
    resolved: dict[str, tuple[str, ...]] = {}
    for entry_id, trace in _iter_entry_traces(evalcli, locators, max_fetches=max_fetches, role_label=role_label):
        inputs = extract_trace_action_inputs(trace, skip_tools=skip_tools, limit=limit)
        if inputs:
            resolved[entry_id] = inputs
    return resolved


def fetch_first_tool_inputs_by_entry(
    evalcli: Any,
    locators: Iterable[TraceActionInputLocator],
    *,
    skip_tools: frozenset[str] = frozenset(),
    max_fetches: int = DEFAULT_MAX_TRACE_FETCHES,
    role_label: str = "",
) -> dict[str, tuple[str, str]]:
    """Fetch detailed traces and return ``{entry_id: (tool_name, payload)}``."""
    resolved: dict[str, tuple[str, str]] = {}
    for entry_id, trace in _iter_entry_traces(evalcli, locators, max_fetches=max_fetches, role_label=role_label):
        pairs = extract_trace_tool_inputs(trace, skip_tools=skip_tools, limit=1)
        if pairs:
            resolved[entry_id] = pairs[0]
    return resolved


__all__ = [
    "DEFAULT_MAX_TRACE_FETCHES",
    "INTERNAL_TRACE_DEPLOYMENT_ID",
    "TraceActionInputLocator",
    "build_trace_locator",
    "extract_trace_action_inputs",
    "extract_trace_tool_inputs",
    "fetch_action_inputs_by_entry",
    "fetch_first_tool_inputs_by_entry",
    "trace_locators_for_rows",
]
