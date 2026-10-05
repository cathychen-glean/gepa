"""Shell-success objective: reward a single model for shell commands that do not error.

Shell differs from the other agentspan objectives in two deliberate ways. The
aggregate comes from its own query, not from ``per_entry``: per-entry rows need
entry attribution and would drop unattributed spans. Trace enrichment keys on
``action_run_id`` inside each error example rather than on ``entry_id``, so it
does not use ``utils.traces.enrich_action_inputs``."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta, timezone
from datetime import time as datetime_time
from typing import Any, ClassVar

from glean_gepa.adapter_types import SingleModelALRolloutOutput, SingleModelALTrajectory
from glean_gepa.al_adapter import ReflectiveExample
from glean_gepa.debug import debug_print
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.focused_evalset import SESSION_BUCKET_TYPE
from glean_gepa.objectives.base import AnalysisRequest, ScoredRow, ScoringContext, SingleModelObjective
from glean_gepa.objectives.utils.agentspan import bounds_query, fetch_agentspan_analysis
from glean_gepa.objectives.utils.agentspan_query import (
    DEFAULT_AGENTS_SPAN_TABLE,
    DEFAULT_LOOKBACK_DAYS,
    default_date_range,
    wildcard_shard_filter,
)
from glean_gepa.objectives.utils.core import RunAnalysis
from glean_gepa.prompt_constants import WRITING_CODE_KEY
from glean_gepa.reflection_prompts import CONDITIONAL_PRESERVE_RULE
from glean_gepa.reflection_sampling import strip_stdout_sections

SHELL_SUCCESS_OBJECTIVE = "shell_success_rate"
SHELL_SPAN_NAMES = ("Execute Action: Shell", "Execute Action: Shell Tool")
SHELL_ACTION_IDS = ("Shell", "Shell Tool")
FAILED_PROVIDER_STATUSES = frozenset({"failed", "error"})


# ---------------------------------------------------------------------------
# Slot 1: one error example, one entry, the run aggregate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShellToolErrorExample:
    started_at: str | None
    project_id: str | None
    entry_id: str | None
    eval_id: str | None
    run_id: str | None
    trace_id: str | None
    span_id: str | None
    span_name: str | None
    action_id: str | None
    action_status: str | None
    span_status: str | None
    provider_status: str | None
    output_status_code: str | None
    error_str: str | None
    session_tracking_token: str | None = None
    action_run_id: str | None = None
    action_input: str | None = None


@dataclass(frozen=True)
class ShellToolErrorEntryMetrics:
    entry_id: str
    shell_executions: int
    shell_errors: int
    shell_error_rate: float
    shell_error_pct: float
    recent_error_examples: tuple[ShellToolErrorExample, ...]
    trace_ids: tuple[str, ...] = ()
    session_tracking_tokens: tuple[str, ...] = ()

    @property
    def shell_success_rate(self) -> float:
        return 1.0 - self.shell_error_rate

    @property
    def has_shell_error(self) -> bool:
        return self.shell_errors > 0

    @property
    def passed(self) -> bool:
        return not self.has_shell_error

    @property
    def score(self) -> float:
        return self.shell_success_rate


@dataclass(frozen=True)
class ShellToolErrorMetrics:
    shell_executions: int
    shell_errors: int
    shell_error_rate: float
    shell_error_pct: float
    recent_error_examples: tuple[ShellToolErrorExample, ...]

    @property
    def shell_success_rate(self) -> float:
        """Higher-is-better score for GEPA objective tracking."""
        return 1.0 - self.shell_error_rate


class EvalRunShellToolErrorAnalysis(RunAnalysis[ShellToolErrorMetrics, ShellToolErrorEntryMetrics]):
    """Single-run analysis; always carries a resolved shard window.

    Unlike the other agentspan objectives the aggregate comes from its own
    query, not from ``per_entry``: per-entry rows require eval-entry
    attribution and would miss spans that lack an entry id.
    """

    start_date: date
    end_date: date


# ---------------------------------------------------------------------------
# Slot 2: rows -> entry (predicate, then parsers below the SQL)
# ---------------------------------------------------------------------------


def is_shell_tool_error(
    *,
    action_status: str | None,
    span_status: str | None,
    output_status_code: str | None,
    provider_status: str | None,
) -> bool:
    provider = (provider_status or "").lower()
    return (
        action_status == "ERROR"
        or span_status == "ERROR"
        or output_status_code == "ERROR"
        or provider in FAILED_PROVIDER_STATUSES
    )


# ---------------------------------------------------------------------------
# Slot 3: SQL (two queries; see module docstring)
# ---------------------------------------------------------------------------


def _shell_execution_id_sql() -> str:
    """Stable execution id without serializing the full ``jsonPayload`` blob."""
    return """COALESCE(
      NULLIF(jsonPayload.action.action_run_id, ''),
      NULLIF(jsonPayload.context.agent_trace.span_id, ''),
      -- Some eval spans omit both IDs. Count the observed span instead of
      -- letting COUNT(DISTINCT NULL) turn a real execution into 0/0.
      TO_JSON_STRING(STRUCT(
        jsonPayload.context.eval.entry_uuid,
        jsonPayload.context.eval.entry_id,
        jsonPayload.span_info.span_name,
        jsonPayload.span_info.start_end_timestamps.start_time_millis
      ))
    )"""


def _shell_span_filter_sql(table_alias: str = "") -> str:
    prefix = f"{table_alias}." if table_alias else ""
    span_names = ", ".join(f"'{name}'" for name in SHELL_SPAN_NAMES)
    action_ids = ", ".join(f"'{action_id}'" for action_id in SHELL_ACTION_IDS)
    return (
        f"({prefix}jsonPayload.span_info.span_name IN ({span_names}) "
        f"OR {prefix}jsonPayload.action.action_id IN ({action_ids}))"
    )


def _shell_spans_select_sql() -> str:
    shell_filter = _shell_span_filter_sql()
    return f"""
  SELECT
    jsonPayload.context.eval.eval_id AS eval_id,
    {_shell_execution_id_sql()} AS shell_execution_id,
    jsonPayload.context.eval.entry_uuid AS entry_uuid,
    CAST(jsonPayload.context.eval.entry_id AS STRING) AS entry_id,
    resource.labels.project_id AS project_id,
    jsonPayload.context.workflow.run_id AS run_id,
    jsonPayload.context.agent_trace.trace_id AS trace_id,
    jsonPayload.span_info.session_info.session_tracking_token AS session_tracking_token,
    jsonPayload.context.agent_trace.span_id AS span_id,
    NULLIF(jsonPayload.action.action_run_id, '') AS action_run_id,
    jsonPayload.span_info.span_name AS span_name,
    jsonPayload.action.action_id AS action_id,
    jsonPayload.action.execution_status AS action_status,
    COALESCE(
      NULLIF(jsonPayload.action.error_str, ''),
      NULLIF(jsonPayload.span_info.execution_status.message, ''),
      NULLIF(jsonPayload.span_info.execution_status.user_message, '')
    ) AS error_str,
    jsonPayload.span_info.execution_status.code AS span_status,
    (
      SELECT o.value
      FROM UNNEST(jsonPayload.span_info.outputs) o
      WHERE o.name = 'status'
      LIMIT 1
    ) AS provider_status,
    (
      SELECT o.value
      FROM UNNEST(jsonPayload.span_info.outputs) o
      WHERE o.name = 'status_code'
      LIMIT 1
    ) AS output_status_code,
    SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64) AS start_ms
  FROM `{{agentspan_table}}`
  WHERE {wildcard_shard_filter("start_date", "end_date")}
    AND jsonPayload.context.eval.eval_id = @eval_id
    AND {shell_filter}
    AND jsonPayload.action.execution_mode = 'EXECUTE'
"""


def build_shell_tool_error_per_entry_query(
    *,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
    entry_ids: Sequence[str] | None = None,
    include_error_examples: bool = True,
) -> str:
    """Build SQL for per-entry shell tool error metrics scoped to one eval run."""
    entry_filter = ""
    if entry_ids:
        entry_filter = "AND COALESCE(entry_id, entry_uuid) IN UNNEST(@entry_ids)"
    shell_spans = _shell_spans_select_sql().format(agentspan_table=agentspan_table)
    error_detail_columns = (
        """
    , ARRAY_AGG(
      IF(
        is_error,
        STRUCT(
          TIMESTAMP_MILLIS(start_ms) AS started_at,
          project_id,
          entry_key AS entry_id,
          eval_id,
          run_id,
          trace_id,
          session_tracking_token,
          span_id,
          span_name,
          action_id,
          action_run_id,
          action_status,
          span_status,
          provider_status,
          output_status_code,
          error_str
        ),
        NULL
      )
      IGNORE NULLS
      ORDER BY start_ms DESC
      LIMIT 10
    ) AS recent_error_examples,
    ARRAY_AGG(
      IF(is_error, trace_id, NULL)
      IGNORE NULLS
      ORDER BY start_ms DESC
      LIMIT 10
    ) AS trace_ids,
    ARRAY_AGG(
      IF(is_error, session_tracking_token, NULL)
      IGNORE NULLS
      ORDER BY start_ms DESC
      LIMIT 10
    ) AS session_tracking_tokens
"""
        if include_error_examples
        else ""
    )
    return f"""
WITH shell_spans AS (
{shell_spans}
),
classified AS (
  SELECT
    -- Prefer the explicit eval entry ID when present, otherwise use the
    -- trace-side eval entry UUID. Current eval runs populate entry_uuid.
    COALESCE(entry_id, entry_uuid) AS entry_key,
    *,
    (
      action_status = 'ERROR'
      OR span_status = 'ERROR'
      OR output_status_code = 'ERROR'
      OR LOWER(provider_status) IN ('failed', 'error')
    ) AS is_error
  FROM shell_spans
  WHERE COALESCE(entry_id, entry_uuid) IS NOT NULL
  {entry_filter}
),
per_entry AS (
  SELECT
    entry_key AS entry_id,
    COUNT(DISTINCT shell_execution_id) AS shell_executions,
    COUNT(DISTINCT IF(is_error, shell_execution_id, NULL)) AS shell_errors,
    SAFE_DIVIDE(
      COUNT(DISTINCT IF(is_error, shell_execution_id, NULL)),
      COUNT(DISTINCT shell_execution_id)
    ) AS shell_error_rate,
    ROUND(
      100 * SAFE_DIVIDE(
        COUNT(DISTINCT IF(is_error, shell_execution_id, NULL)),
        COUNT(DISTINCT shell_execution_id)
      ),
      2
    ) AS shell_error_pct{error_detail_columns}
  FROM classified
  GROUP BY entry_key
)
SELECT * FROM per_entry
ORDER BY shell_errors DESC, shell_executions DESC
""".strip()


def build_shell_tool_error_rate_query(
    *,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> str:
    """Build SQL to measure aggregate shell tool error rate for a single eval run."""
    shell_spans = _shell_spans_select_sql().format(agentspan_table=agentspan_table)
    return f"""
WITH shell_spans AS (
{shell_spans}
),
classified AS (
  SELECT
    *,
    (
      action_status = 'ERROR'
      OR span_status = 'ERROR'
      OR output_status_code = 'ERROR'
      OR LOWER(provider_status) IN ('failed', 'error')
    ) AS is_error
  FROM shell_spans
)
SELECT
  eval_id,
  COUNT(DISTINCT shell_execution_id) AS shell_executions,
  COUNT(DISTINCT IF(is_error, shell_execution_id, NULL)) AS shell_errors,
  SAFE_DIVIDE(
    COUNT(DISTINCT IF(is_error, shell_execution_id, NULL)),
    COUNT(DISTINCT shell_execution_id)
  ) AS shell_error_rate,
  ROUND(
    100 * SAFE_DIVIDE(
      COUNT(DISTINCT IF(is_error, shell_execution_id, NULL)),
      COUNT(DISTINCT shell_execution_id)
    ),
    2
  ) AS shell_error_pct,
  ARRAY_AGG(
    IF(
      is_error,
      STRUCT(
        TIMESTAMP_MILLIS(start_ms) AS started_at,
        project_id,
        COALESCE(entry_uuid, entry_id) AS entry_id,
        eval_id,
        run_id,
        trace_id,
        span_id,
        span_name,
        action_id,
        action_run_id,
        action_status,
        span_status,
        provider_status,
        output_status_code,
        error_str
      ),
      NULL
    )
    IGNORE NULLS
    ORDER BY start_ms DESC
    LIMIT 25
  ) AS recent_error_examples
FROM classified
GROUP BY eval_id
""".strip()


# ---------------------------------------------------------------------------
# Slot 2 (cont.): parsers and reduce
# ---------------------------------------------------------------------------


def parse_shell_tool_error_example(raw: dict[str, Any]) -> ShellToolErrorExample:
    return ShellToolErrorExample(
        started_at=_stringify_value(raw.get("started_at")),
        project_id=_optional_str(raw.get("project_id")),
        entry_id=_optional_str(raw.get("entry_id")),
        eval_id=_optional_str(raw.get("eval_id")),
        run_id=_optional_str(raw.get("run_id")),
        trace_id=_optional_str(raw.get("trace_id")),
        session_tracking_token=_optional_str(raw.get("session_tracking_token")),
        span_id=_optional_str(raw.get("span_id")),
        span_name=_optional_str(raw.get("span_name")),
        action_id=_optional_str(raw.get("action_id")),
        action_run_id=_optional_str(raw.get("action_run_id")),
        action_input=_optional_str(raw.get("action_input")),
        action_status=_optional_str(raw.get("action_status")),
        span_status=_optional_str(raw.get("span_status")),
        provider_status=_optional_str(raw.get("provider_status")),
        output_status_code=_optional_str(raw.get("output_status_code")),
        error_str=_optional_str(raw.get("error_str")),
    )


def parse_shell_tool_error_entry_metrics(row: dict[str, Any]) -> ShellToolErrorEntryMetrics:
    examples_raw = row.get("recent_error_examples") or []
    examples = tuple(parse_shell_tool_error_example(example) for example in examples_raw)
    shell_executions = int(row.get("shell_executions") or 0)
    shell_errors = int(row.get("shell_errors") or 0)
    shell_error_rate = float(row.get("shell_error_rate") or 0.0)
    shell_error_pct = float(row.get("shell_error_pct") or (shell_error_rate * 100))
    trace_ids = tuple(str(trace_id) for trace_id in (row.get("trace_ids") or []) if trace_id)
    session_tracking_tokens = tuple(str(token) for token in (row.get("session_tracking_tokens") or []) if token)
    return ShellToolErrorEntryMetrics(
        entry_id=str(row.get("entry_id") or ""),
        shell_executions=shell_executions,
        shell_errors=shell_errors,
        shell_error_rate=shell_error_rate,
        shell_error_pct=shell_error_pct,
        recent_error_examples=examples,
        trace_ids=trace_ids,
        session_tracking_tokens=session_tracking_tokens,
    )


def parse_shell_tool_error_metrics(row: dict[str, Any]) -> ShellToolErrorMetrics:
    examples_raw = row.get("recent_error_examples") or []
    examples = tuple(parse_shell_tool_error_example(example) for example in examples_raw)
    shell_executions = int(row.get("shell_executions") or 0)
    shell_errors = int(row.get("shell_errors") or 0)
    shell_error_rate = float(row.get("shell_error_rate") or 0.0)
    shell_error_pct = float(row.get("shell_error_pct") or (shell_error_rate * 100))
    return ShellToolErrorMetrics(
        shell_executions=shell_executions,
        shell_errors=shell_errors,
        shell_error_rate=shell_error_rate,
        shell_error_pct=shell_error_pct,
        recent_error_examples=examples,
    )


def aggregate_entry_metrics(per_entry: Mapping[str, ShellToolErrorEntryMetrics]) -> ShellToolErrorMetrics:
    if not per_entry:
        return empty_shell_tool_error_metrics()
    shell_executions = sum(entry.shell_executions for entry in per_entry.values())
    shell_errors = sum(entry.shell_errors for entry in per_entry.values())
    shell_error_rate = shell_errors / shell_executions if shell_executions else 0.0
    recent_examples: list[ShellToolErrorExample] = []
    for entry in per_entry.values():
        recent_examples.extend(entry.recent_error_examples)
    recent_examples = recent_examples[:25]
    return ShellToolErrorMetrics(
        shell_executions=shell_executions,
        shell_errors=shell_errors,
        shell_error_rate=shell_error_rate,
        shell_error_pct=shell_error_rate * 100,
        recent_error_examples=tuple(recent_examples),
    )


def shell_error_free_rate(per_entry: dict[str, ShellToolErrorEntryMetrics]) -> float:
    """Fraction of observed entries with no shell tool errors.

    Used to score a re-run of the high-signal subset. The focused eval set assigns fresh
    entry uuids, so the re-run is scored on its own entries rather than matched by id.
    """
    if not per_entry:
        return 1.0
    passing = sum(1 for metrics in per_entry.values() if not metrics.has_shell_error)
    return passing / len(per_entry)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def fetch_eval_run_shell_tool_error_analysis(
    client: Any,
    *,
    eval_id: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: date | None = None,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
    include_error_examples: bool = True,
    include_per_entry: bool = True,
) -> EvalRunShellToolErrorAnalysis:
    """Aggregate from its own query, per-entry rows optionally after it.

    Aggregate counts must cover every matching shell span. Per-entry metrics
    deliberately require eval-entry attribution and therefore omit spans that
    lack an entry id/uuid; using them as the aggregate made those omitted spans
    look like a perfect 0/0 run. Hence the split-aggregate mode.
    """
    print(f"[Shell Tool] Querying BigQuery shell metrics for eval {eval_id}")

    def parse(row: Mapping[str, Any]) -> ShellToolErrorEntryMetrics | None:
        metrics = parse_shell_tool_error_entry_metrics(dict(row))
        return metrics if metrics.entry_id else None

    def parse_aggregate(row: Mapping[str, Any] | None) -> ShellToolErrorMetrics:
        return parse_shell_tool_error_metrics(dict(row)) if row else empty_shell_tool_error_metrics()

    analysis = fetch_agentspan_analysis(
        client,
        eval_ids=(eval_id,),
        bounds_sql=bounds_query(
            eval_id_predicate="= @eval_id",
            span_filter=f"{_shell_span_filter_sql()}\n  AND jsonPayload.action.execution_mode = 'EXECUTE'",
            agentspan_table=agentspan_table,
        ),
        aggregate_sql=build_shell_tool_error_rate_query(agentspan_table=agentspan_table),
        parse_aggregate_row=parse_aggregate,
        per_entry_sql=build_shell_tool_error_per_entry_query(
            agentspan_table=agentspan_table, include_error_examples=include_error_examples
        ),
        include_per_entry=include_per_entry,
        parse_row=parse,
        aggregate=lambda per_entry, _dropped: aggregate_entry_metrics(per_entry),
        lookback_days=lookback_days,
        end_date=end_date,
    )
    if analysis.start_date is None:
        start_date, resolved_end = default_date_range(lookback_days=lookback_days, end_date=end_date)
    else:
        start_date, resolved_end = analysis.start_date, analysis.end_date or analysis.start_date
    return EvalRunShellToolErrorAnalysis(
        eval_ids=(eval_id,),
        aggregate=analysis.aggregate,
        per_entry=analysis.per_entry,
        high_signal_entry_ids=analysis.high_signal_entry_ids,
        start_date=start_date,
        end_date=resolved_end,
    )


# ---------------------------------------------------------------------------
# Slot 5: action-run-keyed trace enrichment
# ---------------------------------------------------------------------------


def _timestamp_millis(value: str | None) -> int | None:
    if not value:
        return None
    normalized = value.replace(" UTC", "+00:00")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def enrich_shell_error_action_inputs(
    evalcli: EvalCliClient,
    analysis: EvalRunShellToolErrorAnalysis,
) -> EvalRunShellToolErrorAnalysis:
    """Fetch detailed traces and attach serialized Shell inputs to failed actions."""
    from glean_gepa.al_adapter import extract_shell_action_inputs

    examples = list(analysis.aggregate.recent_error_examples)
    for metrics in analysis.per_entry.values():
        examples.extend(metrics.recent_error_examples)

    grouped: defaultdict[tuple[str, str], list[Any]] = defaultdict(list)
    for example in examples:
        if example.project_id and example.trace_id and example.action_run_id and not example.action_input:
            grouped[(example.project_id, example.trace_id)].append(example)
    if not grouped:
        return analysis

    print(f"[Shell Tool] Fetching action inputs for {len(grouped)} traces on eval {analysis.eval_id}")
    resolved: dict[tuple[str, str, str], str] = {}
    for (deployment_id, trace_id), trace_examples in grouped.items():
        timestamps = [
            timestamp
            for example in trace_examples
            for timestamp in [_timestamp_millis(example.started_at)]
            if timestamp is not None
        ]
        if timestamps:
            start_time_millis = min(timestamps) - int(timedelta(hours=1).total_seconds() * 1000)
            end_time_millis = max(timestamps) + 1000
        else:
            start_dt = datetime.combine(analysis.start_date, datetime_time.min, tzinfo=timezone.utc)
            end_dt = datetime.combine(analysis.end_date + timedelta(days=1), datetime_time.min, tzinfo=timezone.utc)
            start_time_millis = int(start_dt.timestamp() * 1000)
            end_time_millis = int(end_dt.timestamp() * 1000)
        try:
            detailed_trace = evalcli.get_analysis_trace(
                deployment_id=deployment_id,
                trace_id=trace_id,
                start_time_millis=start_time_millis,
                end_time_millis=end_time_millis,
            )
        except Exception as exc:
            print(f"[Shell Tool] Failed to fetch action inputs for trace {trace_id}: {exc}")
            continue
        for action_run_id, action_input in extract_shell_action_inputs(detailed_trace).items():
            resolved[(deployment_id, trace_id, action_run_id)] = action_input

    def enrich_example(example: Any) -> Any:
        if example.action_input or not (example.project_id and example.trace_id and example.action_run_id):
            return example
        action_input = resolved.get((example.project_id, example.trace_id, example.action_run_id))
        return replace(example, action_input=action_input) if action_input else example

    if not resolved:
        return analysis

    aggregate = replace(
        analysis.aggregate,
        recent_error_examples=tuple(enrich_example(example) for example in analysis.aggregate.recent_error_examples),
    )
    per_entry = {
        entry_id: replace(
            metrics,
            recent_error_examples=tuple(enrich_example(example) for example in metrics.recent_error_examples),
        )
        for entry_id, metrics in analysis.per_entry.items()
    }
    return EvalRunShellToolErrorAnalysis(
        eval_ids=analysis.eval_ids,
        aggregate=aggregate,
        per_entry=per_entry,
        high_signal_entry_ids=analysis.high_signal_entry_ids,
        start_date=analysis.start_date,
        end_date=analysis.end_date,
    )


def empty_shell_tool_error_metrics() -> ShellToolErrorMetrics:
    return ShellToolErrorMetrics(
        shell_executions=0,
        shell_errors=0,
        shell_error_rate=0.0,
        shell_error_pct=0.0,
        recent_error_examples=(),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _stringify_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


WRITING_CODE_RESPONSIBILITY = (
    "Focus ONLY on coding instructions that affect shell tool reliability: SDK call patterns, "
    "ToolResult handling, parallelism via asyncio.gather, sandbox rules, and when to print vs extract. "
    f"Use shell error examples as evidence. {CONDITIONAL_PRESERVE_RULE} Propose minimal deltas."
)

EVAL_ANALYSIS_CACHE_SCHEMA_VERSION = 10


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def _serialize_eval_analysis(analysis: EvalRunShellToolErrorAnalysis) -> dict[str, Any]:
    def metrics_dict(metrics: Any) -> dict[str, Any]:
        return {
            "entry_id": getattr(metrics, "entry_id", None),
            "shell_executions": metrics.shell_executions,
            "shell_errors": metrics.shell_errors,
            "shell_error_rate": metrics.shell_error_rate,
            "shell_error_pct": metrics.shell_error_pct,
            "trace_ids": list(getattr(metrics, "trace_ids", ())),
            "session_tracking_tokens": list(getattr(metrics, "session_tracking_tokens", ())),
            "recent_error_examples": [asdict(example) for example in metrics.recent_error_examples],
        }

    return {
        "schema_version": EVAL_ANALYSIS_CACHE_SCHEMA_VERSION,
        "eval_id": analysis.eval_id,
        "start_date": analysis.start_date.isoformat(),
        "end_date": analysis.end_date.isoformat(),
        "aggregate": metrics_dict(analysis.aggregate),
        "per_entry": {entry_id: metrics_dict(metrics) for entry_id, metrics in analysis.per_entry.items()},
        "high_signal_entry_ids": list(analysis.high_signal_entry_ids),
    }


def _parse_eval_analysis_cache(raw_cache: Any) -> dict[str, EvalRunShellToolErrorAnalysis]:
    parsed: dict[str, EvalRunShellToolErrorAnalysis] = {}
    if not isinstance(raw_cache, dict):
        return parsed
    for eval_id, raw in raw_cache.items():
        try:
            if not isinstance(raw, dict):
                continue
            if raw.get("schema_version") != EVAL_ANALYSIS_CACHE_SCHEMA_VERSION:
                print(f"[Cache] Refreshing legacy shell error analysis for eval_id: {eval_id}")
                continue
            aggregate = parse_shell_tool_error_metrics(raw["aggregate"])
            if aggregate.shell_executions == 0:
                print(f"[Cache] Refreshing provisional 0/0 shell analysis for eval_id: {eval_id}")
                continue
            per_entry = {
                entry_id: parse_shell_tool_error_entry_metrics(metrics)
                for entry_id, metrics in (raw.get("per_entry") or {}).items()
            }
            parsed[str(eval_id)] = EvalRunShellToolErrorAnalysis(
                eval_ids=(str(raw.get("eval_id") or eval_id),),
                aggregate=aggregate,
                per_entry=per_entry,
                high_signal_entry_ids=tuple(raw.get("high_signal_entry_ids") or ()),
                start_date=date.fromisoformat(raw["start_date"]),
                end_date=date.fromisoformat(raw["end_date"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return parsed


class ShellSuccessObjective(SingleModelObjective[EvalRunShellToolErrorAnalysis]):
    """Score student evals by shell-tool success rate from Agentspan."""

    name = SHELL_SUCCESS_OBJECTIVE
    telemetry_dimensions = (SHELL_SUCCESS_OBJECTIVE,)
    focused_bucket_type = SESSION_BUCKET_TYPE
    failure_label = "HIGH-SIGNAL FAILURES"
    pending_telemetry_label = "shell"
    pending_count = "shell_executions"
    module_responsibilities: ClassVar[Mapping[str, str]] = {WRITING_CODE_KEY: WRITING_CODE_RESPONSIBILITY}

    def __init__(self, *, bigquery_client: Any | None = None, lookback_days: int = 1):
        if bigquery_client is None:
            raise ValueError("bigquery_client is required")
        self.bigquery_client = bigquery_client
        self.lookback_days = lookback_days
        self.params: dict[str, Any] = {}
        self._eval_analysis_cache: dict[str, EvalRunShellToolErrorAnalysis] = {}

    def analyze(self, eval_id: str, *, request: AnalysisRequest) -> EvalRunShellToolErrorAnalysis:
        def fetch(req: AnalysisRequest) -> EvalRunShellToolErrorAnalysis:
            # Error examples are trace-level evidence; the adapter asks for them only on
            # full trace evals, not focused ones.
            analysis = fetch_eval_run_shell_tool_error_analysis(
                self.bigquery_client,
                eval_id=eval_id,
                lookback_days=self.lookback_days,
                include_error_examples=req.wants_traces,
                include_per_entry=req.wants_per_entry,
            )
            if req.wants_traces and req.evalcli is not None:
                analysis = enrich_shell_error_action_inputs(req.evalcli, analysis)
            return analysis

        return self.cached_eval_analysis(eval_id, request=request, fetch=fetch, label="shell error analysis")

    def cache_hit_is_sufficient(self, cached: EvalRunShellToolErrorAnalysis, request: AnalysisRequest) -> bool:
        """An aggregate-only entry cannot serve a request that needs per-entry rows."""
        return not (request.wants_per_entry and not cached.per_entry and cached.aggregate.shell_executions > 0)

    def analysis_is_cacheable(self, analysis: EvalRunShellToolErrorAnalysis, request: AnalysisRequest) -> bool:
        """Only the full trace fetch is worth keeping; it carries the error examples reflection needs."""
        return request.wants_traces and not self.is_pending(analysis)

    def focused_pass_rate(self, analysis: EvalRunShellToolErrorAnalysis, requested_entry_ids: Sequence[str]) -> float:
        passed_entries = sum(1 for entry_metrics in analysis.per_entry.values() if entry_metrics.shell_errors == 0)
        return passed_entries / len(requested_entry_ids)

    def log_analysis(self, analysis: EvalRunShellToolErrorAnalysis) -> None:
        aggregate = analysis.aggregate
        print(
            f"[Shell Tool] Fetched error rate for eval {analysis.eval_id}: "
            f"{aggregate.shell_error_pct:.2f}% "
            f"({aggregate.shell_errors}/{aggregate.shell_executions})"
        )
        for example in aggregate.recent_error_examples:
            if example.action_input:
                debug_print(f"[Shell Tool] Action input for eval {analysis.eval_id}: {example.action_input}")
            if example.error_str:
                debug_print(f"[Shell Tool] Error for eval {analysis.eval_id}: {example.error_str}")

    def aggregate_row(self, analysis: EvalRunShellToolErrorAnalysis, ctx: ScoringContext) -> ScoredRow:
        aggregate = analysis.aggregate
        output: SingleModelALRolloutOutput = {
            "deployment_id": ctx.deployment_id,
            "query": ctx.query,
            "entry_id": ctx.query,
            "student_tool_calls": aggregate.shell_executions,
            "student_tool_errors": aggregate.shell_errors,
            "shell_error_messages": [e.error_str for e in aggregate.recent_error_examples if e.error_str],
            "student_eval_run_id": ctx.student_eval_id,
        }
        if action_inputs := [e.action_input for e in aggregate.recent_error_examples if e.action_input]:
            output["shell_action_inputs"] = action_inputs
        return ScoredRow(
            entry_id=None,
            dimension_scores={SHELL_SUCCESS_OBJECTIVE: aggregate.shell_success_rate},
            output=output,
        )

    def entry_row(
        self,
        entry_id: str,
        metrics: ShellToolErrorEntryMetrics,
        analysis: EvalRunShellToolErrorAnalysis,
        ctx: ScoringContext,
    ) -> ScoredRow:
        del analysis
        # Surface evidence from the one failed eval trace, when there is one.
        failed = next((e for e in metrics.recent_error_examples if e.trace_id), None)
        eval_trace_id = failed.trace_id if failed else None
        examples = [e for e in metrics.recent_error_examples if eval_trace_id is None or e.trace_id == eval_trace_id]
        output: SingleModelALRolloutOutput = {
            "deployment_id": ctx.deployment_id,
            "query": ctx.entry_query(entry_id),
            "entry_id": entry_id,
            "student_tool_calls": metrics.shell_executions,
            "student_tool_errors": metrics.shell_errors,
            "shell_error_messages": [e.error_str for e in examples if e.error_str],
            "student_eval_run_id": ctx.student_eval_id,
        }
        if action_inputs := [e.action_input for e in examples if e.action_input]:
            output["shell_action_inputs"] = action_inputs
        data_overrides: dict[str, Any] = {"eval_entry_id": entry_id, "eval_run_id": ctx.student_eval_id}
        if eval_trace_id:
            output["eval_trace_id"] = eval_trace_id
            data_overrides["eval_trace_id"] = eval_trace_id
        # Focused evals are pass/fail per entry; full evals keep the call-level rate.
        score = float(metrics.shell_errors == 0) if ctx.is_focused else metrics.shell_success_rate
        return ScoredRow(
            entry_id=entry_id,
            dimension_scores={SHELL_SUCCESS_OBJECTIVE: score},
            output=output,
            data_overrides=data_overrides,
        )

    def failure_pattern(self, component_name: str, trajectory: SingleModelALTrajectory) -> tuple[Any, ...]:
        del component_name
        output = trajectory["output"]
        shell_success_rate = trajectory.get("objective_scores", {}).get(self.name, 1.0)
        return (
            int(shell_success_rate < float(self.experiment_param("failure_score_below", 0.9))),
            int(output.get("student_tool_errors", 0) > 0),
            len(output.get("shell_error_messages", [])),
        )

    def build_reflective_example(
        self,
        component_name: str,
        trajectory: SingleModelALTrajectory,
        candidate: dict[str, str],
    ) -> ReflectiveExample:
        del component_name, candidate
        output = trajectory["output"]
        errors = [s for e in output.get("shell_error_messages", []) if (s := strip_stdout_sections(e))]
        if errors:
            feedback = "Resolve the shell execution failures shown above."
        elif output.get("student_tool_errors", 0) > 0:
            feedback = f"Tool errors: Student encountered {output.get('student_tool_errors', 0)} shell tool errors."
        else:
            feedback = "General shell tool reliability issue."
        return self.reflective_example(
            trajectory,
            feedback=feedback,
            action_inputs=output.get("shell_action_inputs", []),
            execution_errors=errors,
        )

    def cache_payload(self) -> dict[str, Any]:
        return {eval_id: _serialize_eval_analysis(analysis) for eval_id, analysis in self._eval_analysis_cache.items()}

    def load_cache(self, raw_cache: Any) -> None:
        self._eval_analysis_cache = _parse_eval_analysis_cache(raw_cache)


__all__ = [
    "EVAL_ANALYSIS_CACHE_SCHEMA_VERSION",
    "ShellSuccessObjective",
]
