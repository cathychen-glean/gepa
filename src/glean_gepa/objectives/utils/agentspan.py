"""Agentspan-backed analysis on top of :mod:`core`.

``agentspan_query`` owns the raw primitives (shard filters, date windows, the
two-query round trip). This module adds the three things every agentspan util
was re-typing:

* :func:`bounds_query` -- the min/max span-time query, parameterised by the
  eval-id predicate and an optional span filter.
* :func:`paired_role_query` -- the ``per_role -> student/teacher -> FULL OUTER
  JOIN`` scaffold teacher/student objectives share. The objective supplies the
  inner CTE that extracts its signal and the name of the array column.
* :func:`fetch_agentspan_analysis` -- window, query, filter, parse, high-signal,
  enrich, aggregate. The objective supplies callables for the steps that are
  about its signal and gets a :class:`~core.RunAnalysis` back.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import Any

from glean_gepa.objectives.utils.agentspan_query import (
    DEFAULT_AGENTS_SPAN_TABLE,
    DEFAULT_LOOKBACK_DAYS,
    QueryParameter,
    run_windowed_per_entry_query,
    wildcard_shard_filter,
)
from glean_gepa.objectives.utils.core import (
    A,
    E,
    RunAnalysis,
    build_analysis,
    empty_analysis,
    parse_rows,
    select_high_signal,
)

Row = Mapping[str, Any]
Rows = list[dict[str, Any]]

# Scrub-safe locator columns every per-role CTE must expose so trace enrichment
# can find the detailed trace later. ``paired_role_query`` carries them through.
LOCATOR_COLUMNS = ("trace_id", "deployment_id", "min_start_ms", "max_start_ms")


# ---------------------------------------------------------------------------
# SQL scaffolds
# ---------------------------------------------------------------------------


def bounds_query(
    *,
    eval_id_predicate: str,
    span_filter: str = "",
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> str:
    """Min/max span start time for the eval(s), used to narrow the shard window.

    ``eval_id_predicate`` is ``"= @eval_id"`` or ``"IN UNNEST(@eval_ids)"``.
    ``span_filter`` optionally restricts to one span kind (e.g. Execute Action)
    so the window tracks tool calls rather than the whole run.
    """
    extra = f"\n  AND {span_filter}" if span_filter else ""
    return f"""
SELECT
  MIN(SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64)) AS min_start_ms,
  MAX(SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64)) AS max_start_ms
FROM `{agentspan_table}`
WHERE {wildcard_shard_filter("search_start_date", "search_end_date")}
  AND jsonPayload.context.eval.eval_id {eval_id_predicate}{extra}
""".strip()


def paired_role_query(
    *,
    per_role_cte: str,
    signal_column: str,
    extra_select: str = "",
    prelude_ctes: str = "",
) -> str:
    """Pair one ARRAY<STRING> signal per entry across a teacher/student eval pair.

    ``per_role_cte`` must be a complete CTE body named ``per_role`` yielding
    ``entry_id, eval_id, <signal_column>`` plus :data:`LOCATOR_COLUMNS`. This
    function adds the ``student`` / ``teacher`` splits, the ``FULL OUTER JOIN``,
    and the ``student_*`` / ``teacher_*`` output columns the parsers read.

    ``extra_select`` appends objective-specific output columns (e.g. a
    ``run_failed`` flag); ``prelude_ctes`` are CTEs the per-role body depends on.
    """
    locators = ", ".join(LOCATOR_COLUMNS)
    role_cols = ",\n".join(
        f"  {role}.{col} AS {role}_{col}" for role in ("student", "teacher") for col in LOCATOR_COLUMNS
    )
    extra = f"\n{extra_select.rstrip(',')}," if extra_select.strip() else ""
    prelude = f"{prelude_ctes.strip()},\n" if prelude_ctes.strip() else ""
    return f"""
WITH {prelude}{per_role_cte.strip()},
student AS (
  SELECT entry_id, {signal_column}, {locators}
  FROM per_role WHERE eval_id = @student_eval_id
),
teacher AS (
  SELECT entry_id, {signal_column}, {locators}
  FROM per_role WHERE eval_id = @teacher_eval_id
)
SELECT
  COALESCE(student.entry_id, teacher.entry_id) AS entry_id,{extra}
  IFNULL(student.{signal_column}, ARRAY<STRING>[]) AS student_{signal_column},
  IFNULL(teacher.{signal_column}, ARRAY<STRING>[]) AS teacher_{signal_column},
{role_cols}
FROM student
FULL OUTER JOIN teacher
  ON student.entry_id = teacher.entry_id
ORDER BY entry_id
""".strip()


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def date_params(start_name: str, end_name: str, start: date, end: date) -> list[QueryParameter]:
    return [
        QueryParameter(start_name, "DATE", start.isoformat()),
        QueryParameter(end_name, "DATE", end.isoformat()),
    ]


def eval_id_params(eval_ids: Sequence[str], *, paired: bool) -> list[QueryParameter]:
    """``@eval_id`` for one run; ``@eval_ids`` + role ids for a pair."""
    ids = list(eval_ids)
    if not paired:
        return [QueryParameter("eval_id", "STRING", ids[-1])]
    return [
        QueryParameter("eval_ids", "STRING", ids),
        QueryParameter("teacher_eval_id", "STRING", ids[0]),
        QueryParameter("student_eval_id", "STRING", ids[-1]),
    ]


# ---------------------------------------------------------------------------
# Fetch pipeline
# ---------------------------------------------------------------------------


def fetch_agentspan_analysis(
    client: Any,
    *,
    eval_ids: Sequence[str],
    bounds_sql: str,
    per_entry_sql: str,
    parse_row: Callable[[Row], E | None],
    aggregate: Callable[[Mapping[str, E], int], A],
    aggregate_sql: str | None = None,
    parse_aggregate_row: Callable[[Row | None], A] | None = None,
    include_per_entry: bool = True,
    is_high_signal: Callable[[E], bool] | None = None,
    filter_rows: Callable[[Rows], Rows] | None = None,
    post_parse: Callable[[Mapping[str, E]], Mapping[str, E]] | None = None,
    enrich: Callable[[Mapping[str, E], Rows, tuple[str, ...]], Mapping[str, E]] | None = None,
    extra_params: Sequence[QueryParameter] = (),
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: date | None = None,
) -> RunAnalysis[A, E]:
    """Run the standard agentspan pipeline and return one analysis.

    Steps, in order: resolve the shard window from ``bounds_sql``; run
    ``per_entry_sql``; ``filter_rows`` (drop rows the objective excludes, e.g.
    failed runs); ``parse_row`` each survivor; ``post_parse`` (overlay another
    source, e.g. EvalCLI judge scores) when given; pick ``is_high_signal`` ids;
    ``enrich`` those entries from traces when given; ``aggregate``.

    ``aggregate`` receives ``(per_entry, dropped_rows)`` so
    objectives that exclude rows can report how many. All SQL strings get the
    date parameters and :func:`eval_id_params` automatically; pass anything
    else via ``extra_params``.

    **Split aggregate.** When the whole-run number must come from its own query
    (per-entry rows would miss spans that lack an entry id), pass
    ``aggregate_sql`` and ``parse_aggregate_row``. That query runs first and
    its first row (or ``None``) becomes the aggregate; ``aggregate`` is then
    unused. Set ``include_per_entry=False`` to stop after it and return an
    analysis with no entries, e.g. for a validation batch.
    """
    ids = tuple(eval_ids)
    paired = len(ids) > 1
    id_params = eval_id_params(ids, paired=paired)
    extras = list(extra_params)

    if (aggregate_sql is None) != (parse_aggregate_row is None):
        raise ValueError("aggregate_sql and parse_aggregate_row must be given together")
    split_aggregate: dict[str, A] = {}

    def agg(per_entry: Mapping[str, E], dropped: int = 0) -> A:
        if "value" in split_aggregate:
            return split_aggregate["value"]
        return aggregate(per_entry, dropped)

    def per_entry_params(s: date, e: date) -> list[QueryParameter]:
        return [*id_params, *date_params("start_date", "end_date", s, e), *extras]

    def bounds_params(s: date, e: date) -> list[QueryParameter]:
        return [*id_params, *date_params("search_start_date", "search_end_date", s, e)]

    if aggregate_sql is not None and parse_aggregate_row is not None:
        # Run the aggregate query inside the resolved window, before per-entry.
        window = run_windowed_per_entry_query(
            client,
            bounds_query=bounds_sql,
            per_entry_query=aggregate_sql,
            bounds_params=bounds_params,
            per_entry_params=per_entry_params,
            lookback_days=lookback_days,
            end_date=end_date,
        )
        if window is None:
            split_aggregate["value"] = parse_aggregate_row(None)
            return empty_analysis(eval_ids=ids, aggregate=agg, paired=paired)
        start_date, resolved_end, agg_rows = window
        split_aggregate["value"] = parse_aggregate_row(agg_rows[0] if agg_rows else None)
        if not include_per_entry:
            return build_analysis(
                eval_ids=ids,
                per_entry={},
                aggregate=agg,
                start_date=start_date,
                end_date=resolved_end,
                paired=paired,
            )
        rows = list(client.query(per_entry_sql, params=per_entry_params(start_date, resolved_end)))
    else:
        result = run_windowed_per_entry_query(
            client,
            bounds_query=bounds_sql,
            per_entry_query=per_entry_sql,
            bounds_params=bounds_params,
            per_entry_params=per_entry_params,
            lookback_days=lookback_days,
            end_date=end_date,
        )
        if result is None:
            return empty_analysis(eval_ids=ids, aggregate=agg, paired=paired)
        start_date, resolved_end, rows = result
    kept = filter_rows(rows) if filter_rows is not None else rows
    dropped = len(rows) - len(kept)
    per_entry = parse_rows(kept, parse_row)
    if post_parse is not None:
        per_entry = dict(post_parse(per_entry))
    high_signal = select_high_signal(per_entry, is_high_signal)
    if enrich is not None and high_signal:
        per_entry = dict(enrich(per_entry, kept, high_signal))
    return build_analysis(
        eval_ids=ids,
        per_entry=per_entry,
        aggregate=lambda p: agg(p, dropped),
        is_high_signal=is_high_signal,
        start_date=start_date,
        end_date=resolved_end,
        paired=paired,
    )
