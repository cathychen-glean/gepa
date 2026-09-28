"""Trace enrichment for high-signal entries, on top of ``action_input_trace``.

The scrubbed agentspan table drops tool payloads, so objectives carry scrub-safe
locators (trace id, deployment, time window) out of BigQuery and resolve the
payloads from each detailed trace through EvalCLI. Three utils had their own
copy of that loop; :func:`enrich_action_inputs` is the one that remains.

An objective passes ``apply``: a function that returns a new metrics object
with the fetched values set on whichever field it uses. Everything else --
locating traces per role, capping fetches, skipping entries with no result --
is shared.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from glean_gepa.objectives.utils.action_input_trace import (
    fetch_action_inputs_by_entry,
    fetch_first_tool_inputs_by_entry,
    trace_locators_for_rows,
)
from glean_gepa.objectives.utils.core import EVIDENCE_LIMIT

M = TypeVar("M")
# {role: {entry_id: fetched}} handed to ``apply``. Single-role objectives see
# one key, ``"student"``; paired ones see both.
FetchedByRole = Mapping[str, Mapping[str, Any]]


def enrich_action_inputs(
    evalcli: Any,
    per_entry: Mapping[str, M],
    rows: Sequence[Mapping[str, Any]],
    high_signal_entry_ids: Sequence[str],
    *,
    apply: Callable[[M, FetchedByRole, str], M],
    roles: Sequence[str] = ("student",),
    paired_columns: bool | None = None,
    first_tool_only: bool = False,
    skip_tools: frozenset[str] = frozenset(),
    limit: int | None = EVIDENCE_LIMIT,
) -> dict[str, M]:
    """Attach tool payloads from detailed traces to the high-signal entries.

    ``roles`` names the trace(s) to read per entry. ``paired_columns`` selects
    ``<role>_trace_id``-style locator columns; it defaults to ``True`` when
    more than one role is given. ``first_tool_only`` fetches
    ``(tool_name, payload)`` for the first scored tool instead of every payload.

    ``apply(metrics, fetched, entry_id)`` is called for every entry in
    ``per_entry`` (not only high-signal ones) with ``fetched[role]`` being the
    per-entry results for that role; it should read ``fetched[role].get(entry_id)``
    and return ``metrics`` unchanged when nothing was found. Returns ``per_entry``
    untouched if no trace yielded anything, so callers can compare identity.
    """
    wanted = set(high_signal_entry_ids)
    if not wanted:
        return dict(per_entry)
    use_role_columns = len(roles) > 1 if paired_columns is None else paired_columns

    fetched: dict[str, Mapping[str, Any]] = {}
    for role in roles:
        locators = trace_locators_for_rows(rows, entry_ids=wanted, role=role if use_role_columns else None)
        if not locators:
            continue
        if first_tool_only:
            result: Mapping[str, Any] = fetch_first_tool_inputs_by_entry(
                evalcli, locators, skip_tools=skip_tools, role_label=role
            )
        else:
            result = fetch_action_inputs_by_entry(
                evalcli, locators, skip_tools=skip_tools, limit=limit, role_label=role
            )
        if result:
            fetched[role] = result
    if not fetched:
        return dict(per_entry)
    return {entry_id: apply(metrics, fetched, entry_id) for entry_id, metrics in per_entry.items()}
