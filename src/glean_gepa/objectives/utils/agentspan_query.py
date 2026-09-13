"""Shared BigQuery primitives for agentspan-derived eval telemetry.

Every telemetry objective (tool match, citation match, loop count, shell
errors) scans the same partitioned ``agentspan_*`` table over the same UTC
shard window and resolves the same min/max span bounds. This module owns those
primitives so each objective only has to describe its own SQL body and params.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

DEFAULT_AGENTS_SPAN_TABLE = "scio-apps.scrubbed_agentspan.scrubbed_agentspan_*"
DEFAULT_LOOKBACK_DAYS = 1
# `_TABLE_SUFFIX` is a UTC date. Always scan tomorrow's shard so a PDT "today"
# or a just-after-midnight UTC eval is not scored as empty 0/0.
UTC_TABLE_SUFFIX_LOOKAHEAD_DAYS = 1
# Predicate matching the agent's tool-invocation spans (one per loop iteration).
EXECUTE_ACTION_FILTER = (
    "STARTS_WITH(jsonPayload.span_info.span_name, 'Execute Action:') AND jsonPayload.action.execution_mode = 'EXECUTE'"
)
# Note: the scrubber strips ``span_info.inputs`` from this table, so tool-call
# payloads are not queryable here. Objectives instead carry the scrub-safe
# ``trace_id``/``project_id``/timestamps out of BigQuery and resolve the payloads
# from the detailed trace (see ``action_input_trace``).


def action_input_tuple(values: Sequence[Any] | None, *, limit: int | None = None) -> tuple[str, ...]:
    """Non-empty, de-duplicated agentspan ``action_input`` payloads in first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values or []:
        text = str(raw).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
        if limit is not None and len(ordered) >= limit:
            break
    return tuple(ordered)


@dataclass(frozen=True)
class QueryParameter:
    name: str
    type_: str
    value: str | list[str]


def wildcard_shard_filter(start_date_param: str, end_date_param: str, *, table_alias: str = "") -> str:
    """Restrict a ``table_*`` wildcard to UTC date shards.

    Compare ``_TABLE_SUFFIX`` as a string. Wrapping it in ``PARSE_DATE`` can stop
    BigQuery from eliminating shards, which scans the full Agentspan history.
    ``FORMAT_DATE`` on query parameters is constant-folded.
    """
    suffix = f"{table_alias}._TABLE_SUFFIX" if table_alias else "_TABLE_SUFFIX"
    return f"{suffix} BETWEEN FORMAT_DATE('%Y%m%d', @{start_date_param}) AND FORMAT_DATE('%Y%m%d', @{end_date_param})"


def utc_today() -> date:
    """Calendar date of the agentspan `_TABLE_SUFFIX` shards (UTC, not host local)."""
    return datetime.now(timezone.utc).date()


def _search_window(*, lookback_days: int, end_date: date | None) -> tuple[date, date]:
    base_end = end_date or utc_today()
    search_end = base_end + timedelta(days=UTC_TABLE_SUFFIX_LOOKAHEAD_DAYS)
    return base_end - timedelta(days=lookback_days), search_end


def default_date_range(
    *, lookback_days: int = DEFAULT_LOOKBACK_DAYS, end_date: date | None = None
) -> tuple[date, date]:
    search_start, search_end = _search_window(lookback_days=lookback_days, end_date=end_date)
    return search_start, search_end


def resolve_eval_run_date_range(
    bounds_row: dict[str, Any] | None,
    *,
    lookback_days: int,
    end_date: date | None = None,
) -> tuple[date, date] | None:
    if not bounds_row:
        return None
    min_ms = bounds_row.get("min_start_ms")
    max_ms = bounds_row.get("max_start_ms")
    if min_ms is None or max_ms is None:
        return None

    # `_TABLE_SUFFIX` is a UTC date. Convert the bounds in UTC too; using the
    # host timezone can shift a just-after-midnight span into the prior day,
    # causing the aggregate query to scan a different shard and return 0/0.
    min_date = datetime.fromtimestamp(int(min_ms) / 1000, tz=timezone.utc).date()
    max_date = datetime.fromtimestamp(int(max_ms) / 1000, tz=timezone.utc).date()
    search_start, search_end = _search_window(lookback_days=lookback_days, end_date=end_date)
    start_date = max(min_date, search_start)
    end_date_resolved = min(max_date, search_end)
    if start_date > end_date_resolved:
        return None
    return start_date, end_date_resolved


def run_windowed_per_entry_query(
    client: Any,
    *,
    bounds_query: str,
    per_entry_query: str,
    bounds_params: Callable[[date, date], list[QueryParameter]],
    per_entry_params: Callable[[date, date], list[QueryParameter]],
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: date | None = None,
) -> tuple[date, date, list[dict[str, Any]]] | None:
    """Resolve the shard window from a bounds query, then run a per-entry query.

    ``bounds_params`` is called with the wide search window and ``per_entry_params``
    with the resolved ``(start, end)`` shard range. Returns ``(start, end, rows)``
    or ``None`` when the eval produced no datable spans in the window, letting the
    caller short-circuit to an empty analysis without another round trip.
    """
    search_start, search_end = default_date_range(lookback_days=lookback_days, end_date=end_date)
    bounds_rows = client.query(bounds_query, params=bounds_params(search_start, search_end))
    date_range = resolve_eval_run_date_range(
        bounds_rows[0] if bounds_rows else None,
        lookback_days=lookback_days,
        end_date=end_date,
    )
    if date_range is None:
        return None
    start_date, resolved_end = date_range
    rows = client.query(per_entry_query, params=per_entry_params(start_date, resolved_end))
    return start_date, resolved_end, list(rows)
