"""Template: a single-model objective whose signal comes from BigQuery agentspan rows.

Which template?

    Does the eval run's analysis view already have your number?

      yes -> _template_evalcli.py    a judge score, a count, or a yes/no the run
                                     already recorded per entry. No SQL.
      no  -> _template_agentspan.py  the signal is in span telemetry: which tools
                                     ran, how many loops, errors, citations. One
                                     SQL query.

    Not sure? Call ``evalcli.get_analysis_view(eval_id)`` on a recent run. If the
    field you would score is in an entry's ``metadata`` or a judge's ``outputs``,
    it is evalcli.

Copy to ``objectives/<signal>.py`` and fill every ``TODO``. Nothing here is
registered; add an ``ObjectiveSpec`` to ``objectives/registry.py`` and a signal to
an experiment YAML when ready.

Four decisions, top to bottom:

1. **One entry.** What one row gives you; ``passed`` / ``score``.
2. **The aggregate.** The run score, and which field is 0 while telemetry lands.
3. **The rows.** One SQL query, one row per entry, plus the locator columns if
   you want tool payloads for reflection.
4. **The feedback.** One sentence for the reflector.

Everything else is inherited from ``objectives/utils``. For a teacher/student
pair, see ``objectives/tool_match.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, ClassVar

from glean_gepa.adapter_types import SingleModelALRolloutOutput, SingleModelALTrajectory
from glean_gepa.al_adapter import ReflectiveExample
from glean_gepa.focused_evalset import QUERY_CANONICAL_BUCKET_TYPE
from glean_gepa.objectives.base import AnalysisRequest, ScoredRow, ScoringContext, SingleModelObjective
from glean_gepa.objectives.utils.agentspan import Rows, bounds_query, fetch_agentspan_analysis
from glean_gepa.objectives.utils.agentspan_query import (
    DEFAULT_AGENTS_SPAN_TABLE,
    DEFAULT_LOOKBACK_DAYS,
    EVAL_ENTRY_ID_EXPR,
    wildcard_shard_filter,
)
from glean_gepa.objectives.utils.core import RunAnalysis, log_analysis, pass_rate
from glean_gepa.objectives.utils.traces import FetchedByRole, enrich_action_inputs
from glean_gepa.prompt_constants import WRITING_CODE_KEY

# TODO: the score key. Also the signal name in the experiment YAML.
EXAMPLE_OBJECTIVE = "example_rate"


# ---------------------------------------------------------------------------
# 1. One entry, and the run aggregate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExampleEntryMetrics:
    """TODO: the fields one row gives you, plus ``passed`` and ``score``."""

    entry_id: str
    value: int
    action_inputs: tuple[str, ...] = ()  # filled by trace enrichment; drop if you never hydrate

    @property
    def passed(self) -> bool:
        return self.value == 0  # TODO

    @property
    def score(self) -> float:
        return 1.0 if self.passed else 0.0  # TODO: any value in [0.0, 1.0]


@dataclass(frozen=True)
class ExampleMetrics:
    """TODO: the whole-run aggregate. ``example_rate`` must match ``EXAMPLE_OBJECTIVE``."""

    compared_entries: int
    passed_entries: int
    example_rate: float


class ExampleAnalysis(RunAnalysis[ExampleMetrics, ExampleEntryMetrics]):
    """Agentspan analyses always carry a resolved shard window."""

    start_date: date
    end_date: date


# ---------------------------------------------------------------------------
# 2. Rows -> entry, entries -> aggregate
# ---------------------------------------------------------------------------


def parse_example_entry(row: Mapping[str, Any]) -> ExampleEntryMetrics | None:
    """Return ``None`` to skip a row (no entry id, excluded kind, ...)."""
    entry_id = str(row.get("entry_id") or "")
    if not entry_id:
        return None
    return ExampleEntryMetrics(entry_id=entry_id, value=int(row.get("value") or 0))  # TODO


def aggregate_example_metrics(per_entry: Mapping[str, ExampleEntryMetrics], dropped_rows: int = 0) -> ExampleMetrics:
    # ``dropped_rows`` is how many rows ``filter_rows`` removed. Surface it on the
    # aggregate if it matters for your signal; tool_match does.
    del dropped_rows
    return ExampleMetrics(
        compared_entries=len(per_entry),
        passed_entries=sum(1 for m in per_entry.values() if m.passed),
        example_rate=pass_rate(per_entry),  # or mean_score(per_entry) for a continuous score
    )


# ---------------------------------------------------------------------------
# 3. SQL: one row per entry
# ---------------------------------------------------------------------------


def build_example_per_entry_query(*, agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE) -> str:
    """TODO: your signal column(s). Keep ``entry_id`` and, if you hydrate tool
    payloads, the locators ``trace_id, deployment_id, min_start_ms, max_start_ms``."""
    return f"""
WITH spans AS (
  SELECT
    {EVAL_ENTRY_ID_EXPR} AS entry_id,
    -- TODO: your signal
    1 AS value,
    jsonPayload.context.agent_trace.trace_id AS trace_id,
    resource.labels.project_id AS deployment_id,
    SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64) AS start_ms
  FROM `{agentspan_table}`
  WHERE {wildcard_shard_filter("start_date", "end_date")}
    AND jsonPayload.context.eval.eval_id = @eval_id
)
SELECT
  entry_id,
  SUM(value) AS value,
  ANY_VALUE(trace_id) AS trace_id,
  ANY_VALUE(deployment_id) AS deployment_id,
  MIN(start_ms) AS min_start_ms,
  MAX(start_ms) AS max_start_ms
FROM spans
WHERE entry_id IS NOT NULL
GROUP BY entry_id
ORDER BY entry_id
""".strip()


def fetch_example_analysis(
    client: Any,
    *,
    eval_id: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    evalcli: Any | None = None,
    include_action_inputs: bool = True,
) -> ExampleAnalysis:
    def enrich(per_entry: Mapping[str, ExampleEntryMetrics], rows: Rows, high_signal: tuple[str, ...]):
        def apply(m: ExampleEntryMetrics, fetched: FetchedByRole, entry_id: str) -> ExampleEntryMetrics:
            inputs = fetched.get("student", {}).get(entry_id)
            return replace(m, action_inputs=inputs) if inputs is not None else m

        return enrich_action_inputs(evalcli, per_entry, rows, high_signal, apply=apply)

    analysis = fetch_agentspan_analysis(
        client,
        eval_ids=(eval_id,),
        bounds_sql=bounds_query(eval_id_predicate="= @eval_id"),
        per_entry_sql=build_example_per_entry_query(),
        parse_row=parse_example_entry,
        aggregate=aggregate_example_metrics,
        # is_high_signal defaults to ``not m.passed``; pass your own if reflection
        # should look at a different subset.
        enrich=enrich if evalcli is not None and include_action_inputs else None,
        lookback_days=lookback_days,
    )
    return ExampleAnalysis(
        eval_ids=analysis.eval_ids,
        aggregate=analysis.aggregate,
        per_entry=analysis.per_entry,
        high_signal_entry_ids=analysis.high_signal_entry_ids,
        start_date=analysis.start_date,
        end_date=analysis.end_date or analysis.start_date,
    )


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

WRITING_CODE_RESPONSIBILITY = "TODO: one paragraph telling the reflector what this module may change and why."


class ExampleObjective(SingleModelObjective[ExampleAnalysis]):
    """TODO: one line."""

    name = EXAMPLE_OBJECTIVE
    telemetry_dimensions = (EXAMPLE_OBJECTIVE,)
    focused_bucket_type = QUERY_CANONICAL_BUCKET_TYPE
    failure_label = "HIGH-SIGNAL FAILURES (TODO)"
    pending_telemetry_label = "example"
    # The aggregate field that is 0 until agentspan rows have landed.
    pending_count = "compared_entries"
    module_responsibilities: ClassVar[Mapping[str, str]] = {WRITING_CODE_KEY: WRITING_CODE_RESPONSIBILITY}

    def __init__(self, *, bigquery_client: Any | None = None, lookback_days: int = 1):
        if bigquery_client is None:
            raise ValueError("bigquery_client is required")
        self.bigquery_client = bigquery_client
        self.lookback_days = lookback_days
        self.params: dict[str, Any] = {}
        self._eval_analysis_cache: dict[str, ExampleAnalysis] = {}
        self._unhydrated_eval_ids: set[str] = set()

    def analyze(self, eval_id: str, *, request: AnalysisRequest) -> ExampleAnalysis:
        def fetch(req: AnalysisRequest) -> ExampleAnalysis:
            return fetch_example_analysis(
                self.bigquery_client,
                eval_id=eval_id,
                lookback_days=self.lookback_days,
                evalcli=req.evalcli,
                include_action_inputs=req.hydrate_action_inputs,
            )

        return self.cached_eval_analysis(eval_id, request=request, fetch=fetch, label="example analysis")

    def focused_pass_rate(self, analysis: ExampleAnalysis, requested_entry_ids: Sequence[str]) -> float:
        if not requested_entry_ids:
            return 0.0
        passed = sum(1 for e in requested_entry_ids if (m := analysis.per_entry.get(e)) is not None and m.passed)
        return passed / len(requested_entry_ids)

    def log_analysis(self, analysis: ExampleAnalysis) -> None:
        a = analysis.aggregate
        log_analysis(
            analysis,
            label="Example",
            headline=f"rate={a.example_rate:.2%} ({a.passed_entries}/{a.compared_entries})",
            entry_line=lambda m: f"value={m.value}",
        )

    def aggregate_row(self, analysis: ExampleAnalysis, ctx: ScoringContext) -> ScoredRow:
        output: SingleModelALRolloutOutput = {
            "deployment_id": ctx.deployment_id,
            "query": ctx.query,
            "entry_id": ctx.query,
            "student_tool_calls": 0,
            "student_tool_errors": 0,
            "shell_error_messages": [],
            "student_eval_run_id": ctx.student_eval_id,
        }
        return ScoredRow(entry_id=None, dimension_scores={self.name: analysis.aggregate.example_rate}, output=output)

    def entry_row(
        self, entry_id: str, metrics: ExampleEntryMetrics, analysis: ExampleAnalysis, ctx: ScoringContext
    ) -> ScoredRow:
        del analysis
        output: SingleModelALRolloutOutput = {
            "deployment_id": ctx.deployment_id,
            "query": ctx.entry_query(entry_id),
            "entry_id": entry_id,
            "student_tool_calls": 0,
            "student_tool_errors": int(not metrics.passed),
            "shell_error_messages": [],
            "student_eval_run_id": ctx.student_eval_id,
        }
        if metrics.action_inputs:
            output["action_inputs"] = list(metrics.action_inputs)
        # TODO: put whatever failure_pattern / build_reflective_example need on ``output``.
        return ScoredRow(
            entry_id=entry_id,
            dimension_scores={self.name: metrics.score},
            output=output,
            data_overrides={"eval_entry_id": entry_id, "eval_run_id": ctx.student_eval_id},
        )

    def failure_pattern(self, component_name: str, trajectory: SingleModelALTrajectory) -> tuple[Any, ...]:
        del component_name
        score = trajectory.get("objective_scores", {}).get(self.name, 1.0)
        return (int(score < float(self.experiment_param("failure_score_below", 1.0))),)

    def build_reflective_example(
        self, component_name: str, trajectory: SingleModelALTrajectory, candidate: dict[str, str]
    ) -> ReflectiveExample:
        del component_name, candidate
        output = trajectory["output"]
        # 4. Feedback: one sentence. State what happened on this entry, then what
        # better looks like. No query text or ids; those go in the evidence slots.
        value = int(not output.get("passed", 0))  # TODO: read what entry_row stashed on ``output``
        feedback = f"TODO: what the prompt should do differently (value={value})."
        return self.reflective_example(trajectory, feedback=feedback, action_inputs=output.get("action_inputs", []))
