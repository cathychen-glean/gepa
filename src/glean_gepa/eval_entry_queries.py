"""Resolve the real user query behind each eval-set entry.

Agentspan scrubs the user query, so reflection examples have carried an
``eval_set:version`` stand-in that is identical on every example and therefore
cannot distinguish one task from another. The eval-set entries endpoint still
serves ``input.query``, joined on the same ``id`` that focused eval sets already
match against analysis entry ids.

Customer eval sets are PII-gated and refuse to list their entries, so only
training eval sets resolve; callers fall back to the stand-in for the rest.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def entry_queries_from_listing(entries: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Map ``entry_id -> user query`` over listed eval-set entries.

    Entries without an id or a non-empty query text are skipped rather than
    mapped to an empty string, so callers can tell "unresolved" from "empty".
    """
    resolved: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        entry_id = str(entry.get("id") or "")
        if not entry_id:
            continue
        entry_input = entry.get("input")
        raw = entry.get("query") or (entry_input.get("query") if isinstance(entry_input, Mapping) else None)
        if isinstance(raw, str) and raw.strip():
            resolved[entry_id] = raw.strip()
    return resolved


def fetch_entry_queries(
    evalcli: Any,
    *,
    eval_set_name: str,
    eval_set_version: str,
    deployment_ids: Sequence[str],
) -> dict[str, str]:
    """List one eval-set version and return its ``entry_id -> query`` map.

    Returns an empty map when the listing fails, which is the expected outcome
    for PII-gated customer eval sets.
    """
    if evalcli is None or not eval_set_name or not eval_set_version:
        return {}
    try:
        entries = evalcli.list_eval_set_entries(
            eval_set_name=eval_set_name,
            eval_set_version=eval_set_version,
            deployment_ids=list(deployment_ids),
        )
    except Exception as exc:
        print(f"[Entry Queries] Could not list entries for {eval_set_name}:{eval_set_version}: {exc}")
        return {}
    resolved = entry_queries_from_listing(entries or [])
    print(f"[Entry Queries] Resolved {len(resolved)} user queries for {eval_set_name}:{eval_set_version}")
    return resolved


__all__ = ["entry_queries_from_listing", "fetch_entry_queries"]
