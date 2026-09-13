"""Fetch per-entry tool-call payloads from detailed eval traces.

The ``scrubbed_agentspan`` table drops ``span_info.inputs`` before logging, so the
tool-call payload (the search string or shell command the agent issued) is *not*
queryable from BigQuery — a query against that field always comes back empty. The
detailed trace served by ``evalcli analyze trace`` still carries it under each
``Execute Action`` span's ``attributes.input`` (a JSON blob with an ``action_input``
field), which is the same surviving source the Shell objective already uses.

This module locates those traces from scrub-safe identifiers (trace id, deployment,
span timestamps) that *do* survive scrubbing, fetches them for a bounded set of
entries, and extracts the ordered ``action_input`` payloads so reflection has real
per-entry intent evidence instead of an empty field.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from glean_gepa.objectives.utils.agentspan_query import action_input_tuple

_EXECUTE_ACTION_PREFIX = "Execute Action: "
# The analyze-trace API filters by wall-clock time rather than eval id, so widen
# the window around the entry's Execute Action span bounds to tolerate ingest skew.
TRACE_WINDOW_LEAD_MS = 3_600_000
TRACE_WINDOW_TRAIL_MS = 60_000
# Cap trace fetches per role so a large high-signal set cannot fan out into
# thousands of sequential analyze-trace calls. Entries are enriched in the
# deterministic high-signal order, covering those most likely to be surfaced.
DEFAULT_MAX_TRACE_FETCHES = 60


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


def extract_trace_action_inputs(
    detailed_trace: Any,
    *,
    skip_tools: frozenset[str] = frozenset(),
    limit: int | None = None,
) -> tuple[str, ...]:
    """Ordered, de-duplicated ``action_input`` payloads from a detailed trace.

    Reads every ``Execute Action:`` span's ``attributes.input`` JSON and pulls its
    ``action_input`` field, preserving span order. Tools in ``skip_tools`` (matched
    on the name after the ``Execute Action:`` prefix) are ignored.
    """
    if not isinstance(detailed_trace, dict):
        return ()
    spans = (detailed_trace.get("trace") or {}).get("spans") or []
    ordered: list[str] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        name = span.get("name") or ""
        if not name.startswith(_EXECUTE_ACTION_PREFIX):
            continue
        if name[len(_EXECUTE_ACTION_PREFIX) :] in skip_tools:
            continue
        raw_input = _typed_str_value((span.get("attributes") or {}).get("input"))
        if not raw_input:
            continue
        try:
            payload = json.loads(raw_input)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        action_input = payload.get("action_input") if isinstance(payload, dict) else None
        if isinstance(action_input, str) and action_input.strip():
            ordered.append(action_input)
    return action_input_tuple(ordered, limit=limit)


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


def fetch_action_inputs_by_entry(
    evalcli: Any,
    locators: Iterable[TraceActionInputLocator],
    *,
    skip_tools: frozenset[str] = frozenset(),
    limit: int | None = None,
    max_fetches: int = DEFAULT_MAX_TRACE_FETCHES,
    role_label: str = "",
) -> dict[str, tuple[str, ...]]:
    """Fetch detailed traces and return ``{entry_id: action_inputs}``.

    Silently skips entries whose trace cannot be fetched (a transient analyze-trace
    failure should degrade to "no evidence", never abort the surrounding analysis).
    """
    get_trace = getattr(evalcli, "get_analysis_trace", None)
    if not callable(get_trace):
        return {}
    resolved: dict[str, tuple[str, ...]] = {}
    fetched = 0
    for locator in locators:
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
            label = f"{role_label} " if role_label else ""
            print(f"[Action Inputs] Failed to fetch {label}trace for entry {locator.entry_id}: {exc}")
            continue
        inputs = extract_trace_action_inputs(trace, skip_tools=skip_tools, limit=limit)
        if inputs:
            resolved[locator.entry_id] = inputs
    return resolved


__all__ = [
    "DEFAULT_MAX_TRACE_FETCHES",
    "TraceActionInputLocator",
    "build_trace_locator",
    "extract_trace_action_inputs",
    "fetch_action_inputs_by_entry",
]
