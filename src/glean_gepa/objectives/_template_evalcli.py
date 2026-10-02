"""Template: a single-model objective whose signal comes from EvalCLI, not BigQuery.

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

Copy to ``objectives/<signal>.py`` and fill every ``TODO``. There is no SQL,
no shard window, and no trace hydration. The file is importable and
pyright-clean as-is. Nothing here is registered; add an ``ObjectiveSpec`` to
``objectives/registry.py`` and a signal to an experiment YAML when ready.

Three decisions, top to bottom:

1. **One entry.** What does the analysis view give you per entry, and what is
   ``passed`` / ``score``? (``ExampleEntryMetrics``)
2. **The aggregate.** What does the run score, and which field is 0 while the
   eval is still running? (``ExampleMetrics``, ``pending_count``)
3. **The feedback.** One sentence for the reflector. (inline in ``build_reflective_example``)

For a teacher/student pair backed by EvalCLI, see
``objectives/agentic_preference.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from glean_gepa.adapter_types import SingleModelALRolloutOutput, SingleModelALTrajectory
from glean_gepa.al_adapter import ReflectiveExample
from glean_gepa.focused_evalset import QUERY_CANONICAL_BUCKET_TYPE
from glean_gepa.objectives.base import AnalysisRequest, ScoredRow, ScoringContext, SingleModelObjective
from glean_gepa.objectives.utils.core import RunAnalysis, build_analysis, empty_analysis, log_analysis, pass_rate
from glean_gepa.prompt_constants import WRITING_CODE_KEY

# TODO: the score key. Also the signal name in the experiment YAML.
EXAMPLE_OBJECTIVE = "example_rate"


# ---------------------------------------------------------------------------
# 1. One entry, and the run aggregate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExampleEntryMetrics:
    """TODO: what the analysis view gives you per entry, plus ``passed`` and ``score``."""

    entry_id: str
    value: float

    @property
    def passed(self) -> bool:
        return self.value >= 0.5  # TODO

    @property
    def score(self) -> float:
        return self.value  # TODO: any value in [0.0, 1.0]


@dataclass(frozen=True)
class ExampleMetrics:
    """TODO: the whole-run aggregate. ``example_rate`` must match ``EXAMPLE_OBJECTIVE``."""

    compared_entries: int
    passed_entries: int
    example_rate: float


ExampleAnalysis = RunAnalysis[ExampleMetrics, ExampleEntryMetrics]


# ---------------------------------------------------------------------------
# 2. Analysis view -> entries -> aggregate
# ---------------------------------------------------------------------------


def parse_example_entry(run_entry: Mapping[str, Any]) -> ExampleEntryMetrics | None:
    """One EvalCLI run entry -> one metric, or ``None`` to skip it."""
    entry_id = str(run_entry.get("entryId") or run_entry.get("id") or "")
    if not entry_id:
        return None
    raw = run_entry.get("example_value")  # TODO: the field the view exposes
    if raw is None:
        return None
    return ExampleEntryMetrics(entry_id=entry_id, value=float(raw))


def aggregate_example_metrics(per_entry: Mapping[str, ExampleEntryMetrics]) -> ExampleMetrics:
    return ExampleMetrics(
        compared_entries=len(per_entry),
        passed_entries=sum(1 for m in per_entry.values() if m.passed),
        example_rate=pass_rate(per_entry),  # or mean_score(per_entry) for a continuous score
    )


def fetch_example_analysis(evalcli: Any, *, eval_id: str) -> ExampleAnalysis:
    view = evalcli.get_analysis_view(eval_id)  # TODO: the EvalCLI call that has your field
    run_entries = (view or {}).get("runEntries") or []
    if not run_entries:
        return empty_analysis(eval_ids=(eval_id,), aggregate=aggregate_example_metrics)
    per_entry = {m.entry_id: m for e in run_entries if (m := parse_example_entry(e)) is not None}
    return build_analysis(eval_ids=(eval_id,), per_entry=per_entry, aggregate=aggregate_example_metrics)


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
    pending_count = "compared_entries"
    module_responsibilities: ClassVar[Mapping[str, str]] = {WRITING_CODE_KEY: WRITING_CODE_RESPONSIBILITY}

    def __init__(self, **_: Any):
        # No BigQuery client: everything comes through ``request.evalcli``.
        self.params: dict[str, Any] = {}
        self._eval_analysis_cache: dict[str, ExampleAnalysis] = {}
        self._unhydrated_eval_ids: set[str] = set()

    def analyze(self, eval_id: str, *, request: AnalysisRequest) -> ExampleAnalysis:
        def fetch(req: AnalysisRequest) -> ExampleAnalysis:
            if req.evalcli is None:
                raise ValueError("evalcli is required on the request for an EvalCLI-backed objective")
            return fetch_example_analysis(req.evalcli, eval_id=eval_id)

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
            entry_line=lambda m: f"value={m.value:.2f}",
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
        # 3. Feedback: one sentence. State what happened on this entry, then what
        # better looks like. No query text or ids; those go in the evidence slots.
        score = float(trajectory.get("objective_scores", {}).get(self.name, 0.0))
        feedback = f"TODO: what the prompt should do differently (score={score:.2f})."
        return self.reflective_example(trajectory, feedback=feedback)
