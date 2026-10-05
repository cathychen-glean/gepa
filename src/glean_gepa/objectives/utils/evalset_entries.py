"""Evalset-entry lookups against the ``fact.*`` tables.

Which entries of an eval set are high-signal, and how each entry's UUID maps to
the eval runs that exercised it. Used by ``single_model_adapter`` to build a
focused replay set and by ``focused_evalset`` to track entries across
versions. Not tied to any one objective.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from typing import Any

from glean_gepa.objectives.utils.agentspan_query import (
    DEFAULT_AGENTS_SPAN_TABLE,
    DEFAULT_LOOKBACK_DAYS,
    QueryParameter,
    wildcard_shard_filter,
)

DEFAULT_EVALSET_ENTRIES_TABLE = "scio-apps.fact.evalset_entries"
DEFAULT_EVAL_WORKFLOW_RUNS_TABLE = "scio-apps.fact.eval_workflow_runs"


def _evalset_entries_deployment_filter(deployment_ids: Sequence[str] | None) -> str:
    if deployment_ids:
        return "AND project_id IN UNNEST(@deployment_ids)"
    return ""


def build_high_signal_source_entries_query(
    *,
    evalset_entries_table: str = DEFAULT_EVALSET_ENTRIES_TABLE,
    eval_workflow_runs_table: str = DEFAULT_EVAL_WORKFLOW_RUNS_TABLE,
    deployment_ids: Sequence[str] | None = None,
) -> str:
    """Resolve source evalset rows from an eval run's trace-side entry UUIDs."""
    deployment_filter = _evalset_entries_deployment_filter(deployment_ids)
    return f"""
WITH runtime_entries AS (
  SELECT
    entry_uuid,
    ARRAY_AGG(stt IGNORE NULLS ORDER BY workflow_start_timestamp DESC LIMIT 1)[SAFE_OFFSET(0)] AS stt
  FROM `{eval_workflow_runs_table}`
  WHERE eval_id = @eval_run_id
    AND entry_uuid IN UNNEST(@entry_uuids)
  GROUP BY entry_uuid
)
SELECT
  runtime_entries.entry_uuid AS id,
  evalset_entries.project_id AS deploymentId,
  evalset_entries.stt,
  evalset_entries.workflow_run_id AS runId,
  evalset_entries.query_ts,
  evalset_entries.datepartition AS source_date
FROM runtime_entries
JOIN `{evalset_entries_table}` AS evalset_entries USING (stt)
WHERE evalset_entries.eval_set_name = @eval_set_name
  AND evalset_entries.eval_set_version = @eval_set_version
  AND LENGTH(stt) > 0
  AND LENGTH(evalset_entries.workflow_run_id) > 0
  {deployment_filter}
QUALIFY ROW_NUMBER() OVER (PARTITION BY runtime_entries.entry_uuid ORDER BY evalset_entries.log_ts DESC) = 1
ORDER BY id
""".strip()


def build_eval_entry_uuid_tracking_query(
    *,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
    deployment_ids: Sequence[str] | None = None,
) -> str:
    """Resolve SESSION tracking tokens from eval spans keyed by entry_uuid.

    Cortex eval-set entry ids match Agentspan ``entry_uuid``, not
    ``fact.evalset_entries.entry_id``. Eval replays keep the original session
    tracking token, which is what SESSION uploads need.
    """
    deployment_filter = ""
    if deployment_ids:
        deployment_filter = "AND resource.labels.project_id IN UNNEST(@deployment_ids)"
    return f"""
SELECT
  COALESCE(
    jsonPayload.context.eval.entry_uuid,
    CAST(jsonPayload.context.eval.entry_id AS STRING)
  ) AS id,
  resource.labels.project_id AS deploymentId,
  jsonPayload.span_info.session_info.session_tracking_token AS stt,
  jsonPayload.context.workflow.run_id AS runId,
  jsonPayload.context.agent_trace.trace_id AS traceId
FROM `{agentspan_table}`
WHERE {wildcard_shard_filter("start_date", "end_date")}
  AND (
    jsonPayload.context.eval.entry_uuid IN UNNEST(@entry_ids)
    OR CAST(jsonPayload.context.eval.entry_id AS STRING) IN UNNEST(@entry_ids)
  )
  AND LENGTH(jsonPayload.span_info.session_info.session_tracking_token) > 0
  {deployment_filter}
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY COALESCE(
    jsonPayload.context.eval.entry_uuid,
    CAST(jsonPayload.context.eval.entry_id AS STRING)
  )
  ORDER BY SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64) DESC
) = 1
""".strip()


def build_high_signal_trace_query(
    *,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> str:
    """Join resolved source entries to their non-eval historical traces."""
    return f"""
WITH source_entries AS (
  SELECT
    @entry_ids[OFFSET(i)] AS id,
    @deployment_ids[OFFSET(i)] AS deploymentId,
    @session_tracking_tokens[OFFSET(i)] AS stt,
    @workflow_run_ids[OFFSET(i)] AS runId
  FROM UNNEST(GENERATE_ARRAY(0, ARRAY_LENGTH(@entry_ids) - 1)) AS i
)
SELECT
  e.id,
  e.deploymentId,
  e.stt,
  e.runId,
  a.jsonPayload.context.agent_trace.trace_id AS traceId
FROM source_entries e
JOIN `{agentspan_table}` a
  ON (
    a.jsonPayload.context.workflow.run_id = e.runId
    OR (
      e.runId = ''
      AND a.jsonPayload.span_info.session_info.session_tracking_token = e.stt
    )
  )
WHERE _TABLE_SUFFIX BETWEEN @start_suffix AND @end_suffix
  AND a.jsonPayload.context.eval.eval_id IS NULL
  AND a.jsonPayload.context.agent_trace.trace_id IS NOT NULL
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY e.id
  ORDER BY SAFE_CAST(a.jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64) DESC
) = 1
ORDER BY e.stt
""".strip()


def fetch_high_signal_evalset_entries(
    client: Any,
    *,
    eval_set_name: str,
    eval_set_version: str,
    eval_run_id: str,
    entry_ids: Sequence[str],
    deployment_ids: Sequence[str] | None = None,
    evalset_entries_table: str = DEFAULT_EVALSET_ENTRIES_TABLE,
    eval_workflow_runs_table: str = DEFAULT_EVAL_WORKFLOW_RUNS_TABLE,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> list[dict[str, Any]]:
    """Fetch upload-ready source entries from the parent eval's trace-side entry UUIDs."""
    entry_uuids = sorted(set(entry_ids))
    if not entry_uuids:
        return []
    params = [
        QueryParameter("eval_set_name", "STRING", eval_set_name),
        QueryParameter("eval_set_version", "STRING", eval_set_version),
        QueryParameter("eval_run_id", "STRING", eval_run_id),
        QueryParameter("entry_uuids", "STRING", entry_uuids),
    ]
    if deployment_ids:
        params.append(QueryParameter("deployment_ids", "STRING", list(deployment_ids)))
    source_rows = client.query(
        build_high_signal_source_entries_query(
            evalset_entries_table=evalset_entries_table,
            eval_workflow_runs_table=eval_workflow_runs_table,
            deployment_ids=deployment_ids,
        ),
        params=params,
    )
    source_rows = [row for row in source_rows if row.get("id") and row.get("deploymentId") and row.get("stt")]
    if not source_rows:
        return []
    trace_ids = _historical_trace_ids(client, source_rows, agentspan_table=agentspan_table)
    return [
        {
            "id": str(row["id"]),
            "deploymentId": str(row["deploymentId"]),
            "stt": str(row["stt"]),
            "runId": str(row.get("runId") or ""),
            "traceId": trace_ids[str(row["id"])],
        }
        for row in source_rows
        if str(row["id"]) in trace_ids and row.get("runId")
    ]


def _eval_set_version_search_window(eval_set_version: str) -> tuple[date, date]:
    """Search Agentspan from the eval-set version date through today (UTC)."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    if len(eval_set_version) >= 8:
        parsed = _parse_bigquery_date(f"{eval_set_version[:4]}-{eval_set_version[4:6]}-{eval_set_version[6:8]}")
        if parsed is not None:
            start = parsed
    if start > end:
        start = end
    return start, end


def fetch_evalset_entry_tracking(
    client: Any,
    *,
    eval_set_name: str,
    eval_set_version: str,
    entry_ids: Sequence[str],
    deployment_ids: Sequence[str] | None = None,
    evalset_entries_table: str = DEFAULT_EVALSET_ENTRIES_TABLE,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> dict[str, dict[str, Any]]:
    """Map Cortex eval-set entry ids to SESSION upload fields.

    Entry ids from EvalCLI / Agentspan ``entry_uuid`` do not match
    ``fact.evalset_entries.entry_id``. Resolve ``stt`` from eval spans, then
    optionally replace the eval ``runId`` with the original session's
    ``workflow_run_id`` from the fact table.
    """
    wanted = sorted({str(entry_id) for entry_id in entry_ids if entry_id})
    if not wanted:
        return {}
    start_date, end_date = _eval_set_version_search_window(eval_set_version)
    span_params = [
        QueryParameter("entry_ids", "STRING", wanted),
        QueryParameter("start_date", "DATE", start_date.isoformat()),
        QueryParameter("end_date", "DATE", end_date.isoformat()),
    ]
    if deployment_ids:
        span_params.append(QueryParameter("deployment_ids", "STRING", list(deployment_ids)))
    span_rows = [
        row
        for row in client.query(
            build_eval_entry_uuid_tracking_query(
                agentspan_table=agentspan_table,
                deployment_ids=deployment_ids,
            ),
            params=span_params,
        )
        if row.get("id") and row.get("stt")
    ]
    if not span_rows:
        return {}

    tokens = sorted({str(row["stt"]) for row in span_rows})
    fact_params = [
        QueryParameter("eval_set_name", "STRING", eval_set_name),
        QueryParameter("eval_set_version", "STRING", eval_set_version),
        QueryParameter("session_tracking_tokens", "STRING", tokens),
    ]
    if deployment_ids:
        fact_params.append(QueryParameter("deployment_ids", "STRING", list(deployment_ids)))
    fact_by_stt = {
        str(row["stt"]): row
        for row in client.query(
            build_high_signal_source_entries_query(
                evalset_entries_table=evalset_entries_table,
                deployment_ids=deployment_ids,
            ),
            params=fact_params,
        )
        if row.get("stt")
    }

    tracking: dict[str, dict[str, Any]] = {}
    for row in span_rows:
        entry_id = str(row["id"])
        source = fact_by_stt.get(str(row["stt"])) or {}
        payload: dict[str, Any] = {
            "deploymentId": str(source.get("deploymentId") or row.get("deploymentId") or ""),
            "stt": str(row["stt"]),
        }
        if source.get("runId"):
            payload["runId"] = str(source["runId"])
        if payload["deploymentId"]:
            tracking[entry_id] = payload
    return tracking


def _historical_trace_ids(
    client: Any,
    source_rows: Sequence[Mapping[str, Any]],
    *,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> dict[str, str]:
    """Return entry id -> non-eval historical trace id when Agentspan can resolve one."""
    source_dates = [
        parsed_date
        for row in source_rows
        for parsed_date in [_parse_bigquery_date(row.get("query_ts") or row.get("source_date"))]
        if parsed_date is not None
    ]
    if not source_dates:
        return {}
    start_suffix = (min(source_dates) - timedelta(days=1)).strftime("%Y%m%d")
    end_suffix = (max(source_dates) + timedelta(days=1)).strftime("%Y%m%d")
    trace_rows = client.query(
        build_high_signal_trace_query(agentspan_table=agentspan_table),
        params=[
            QueryParameter("entry_ids", "STRING", [str(row["id"]) for row in source_rows]),
            QueryParameter(
                "deployment_ids",
                "STRING",
                [str(row["deploymentId"]) for row in source_rows],
            ),
            QueryParameter(
                "session_tracking_tokens",
                "STRING",
                [str(row["stt"]) for row in source_rows],
            ),
            QueryParameter(
                "workflow_run_ids",
                "STRING",
                [str(row.get("runId") or "") for row in source_rows],
            ),
            QueryParameter("start_suffix", "STRING", start_suffix),
            QueryParameter("end_suffix", "STRING", end_suffix),
        ],
    )
    return {str(row["id"]): str(row["traceId"]) for row in trace_rows if row.get("id") and row.get("traceId")}


def _parse_bigquery_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None
