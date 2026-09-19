"""Aggregate judge scores from Cortex eval metrics.

After ``judge create``, scores are read from ``POST /metrics/evalruns/pairwise``
(evalcli ``metrics summary``). Judge-run status is listed with
``GET /judgeruns?evalRunIds=`` (evalcli ``judge list``).
``GET /judgeruns/{id}`` and ``list-for-run`` are not used; those Cortex routes
are unimplemented.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from glean_gepa.evalcli_client import (
    AGENTIC_INPUT_MAPPINGS,
    AGENTIC_JUDGE_NAME,
    AGENTIC_JUDGE_TYPE,
    AGENTIC_PREFERENCE_RATE_METRIC,
    AGENTIC_RUN_PARAMS,
    COMPLETENESS_JUDGE_TYPE,
    COMPLETENESS_RUN_PARAMS,
    CORRECTNESS_INPUT_MAPPINGS,
    CORRECTNESS_JUDGE_TYPE,
    CORRECTNESS_RUN_PARAMS,
    EvalCliClient,
    EvalCliError,
)


@dataclass(frozen=True)
class JudgeAnalysis:
    eval_id: str
    aggregate: float
    per_entry: dict[str, float]
    judge_run_id: str | None = None
    judge_type: str | None = None


def judge_pass_rate_from_metrics(
    payload: dict[str, Any],
    *,
    judge_type: str,
    judge_run_id: str | None = None,
) -> float | None:
    """Return the judge pass rate, or None if BigQuery has no scored rows yet."""
    wanted_type = judge_type.upper()
    charts = payload.get("judgeMetrics") or {}
    if not isinstance(charts, dict):
        return None
    while isinstance(charts.get("additional_properties"), dict):
        charts = charts["additional_properties"]
    rows: list[dict[str, Any]] = []
    for key, value in charts.items():
        if key in {"totalEntries", "missingEntries", "additional_properties"}:
            continue
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            rows.append({"chart": key, **value})
    for row in rows:
        row_judge_id = row.get("judgeRunId")
        chart = str(row.get("chart") or row.get("judgeType") or "").upper()
        if judge_run_id and row_judge_id and str(row_judge_id) != judge_run_id:
            continue
        if judge_run_id and row_judge_id and str(row_judge_id) == judge_run_id:
            value = row.get("test") if row.get("test") is not None else row.get("passRate")
            return None if value is None else float(value)
        if chart != wanted_type:
            continue
        value = row.get("test") if row.get("test") is not None else row.get("passRate")
        if value is None:
            return None
        return float(value)
    return None


def wait_for_judge_metrics(
    evalcli: EvalCliClient,
    *,
    eval_id: str,
    judge_type: str,
    judge_run_id: str | None = None,
    base_eval_id: str | None = None,
    poll_interval_sec: int = 60,
    timeout_sec: int = 3600,
) -> JudgeAnalysis:
    """Poll pairwise metrics until the judge has scored rows (passRate is not null)."""
    print(f"[{judge_type}] Waiting for metrics on eval {eval_id}...")
    elapsed = 0
    last_payload: dict[str, Any] | None = None
    while elapsed <= timeout_sec:
        try:
            last_payload = evalcli.get_eval_metrics(eval_id, base_eval_id=base_eval_id)
        except EvalCliError as exc:
            print(f"[{judge_type}] Transient metrics error for {eval_id}: {exc}")
            last_payload = None
        if last_payload is not None:
            rate = judge_pass_rate_from_metrics(last_payload, judge_type=judge_type, judge_run_id=judge_run_id)
            if rate is not None:
                print(f"[{judge_type}] {eval_id}: {rate:.2f}")
                return JudgeAnalysis(
                    eval_id=eval_id,
                    aggregate=rate,
                    per_entry={},
                    judge_run_id=judge_run_id,
                    judge_type=judge_type,
                )
        if elapsed >= timeout_sec:
            break
        time.sleep(poll_interval_sec)
        elapsed += poll_interval_sec
    raise EvalCliError(
        f"{judge_type} metrics for {eval_id} were not ready after {timeout_sec}s (judge_run_id={judge_run_id})"
    )


CUSTOMER_CORRECTNESS_METRIC = "correctness"
CUSTOMER_AGENTIC_PREFERENCE_METRIC = "agentic_preference_rate"
COMPLETENESS_METRIC = "completeness"


@dataclass(frozen=True)
class JudgeSpec:
    """How to start a Cortex judge and read its floor."""

    name: str
    kind: str
    default_min: float
    judge_type: str
    run_params: str
    # True: score must be strictly above min. False: min is a passing tie.
    strict: bool
    category_aliases: tuple[str, ...]
    metrics_label: str
    score_source: str
    label: str
    input_mappings: str = ""
    judge_type_aliases: tuple[str, ...] = ()
    row_metric: str | None = None
    score_keys: tuple[str, ...] = ("test", "passRate", "pass_rate", "testValue", "test_value")

    def matches_row(self, category: str, row: Mapping[str, Any]) -> bool:
        metric = str(row.get("metric", ""))
        in_category = category in self.category_aliases or metric.upper() in self.category_aliases
        in_judge_type = str(row.get("judgeType", "")) in self.judge_type_aliases
        if not (in_category or in_judge_type):
            return False
        return self.row_metric is None or metric == self.row_metric

    def failure_message(self, score: float, floor: float) -> str | None:
        if self.strict:
            if score <= floor:
                return f"{self.label} {score:.2%} is not above {floor:.0%}"
            return None
        if score < floor:
            return f"{self.label} {score:.2%} is below {floor:.0%}"
        return None

    def report_line(self, score: float, floor: float) -> str:
        required = f">{floor:.0%}" if self.strict else f">={floor:.0%}"
        return f"{self.name}={score:.2%} (required {required})"


JUDGE_SPECS: dict[str, JudgeSpec] = {
    spec.name: spec
    for spec in (
        JudgeSpec(
            name=CUSTOMER_CORRECTNESS_METRIC,
            kind="pairwise",
            default_min=0.80,
            judge_type=CORRECTNESS_JUDGE_TYPE,
            run_params=CORRECTNESS_RUN_PARAMS,
            input_mappings=CORRECTNESS_INPUT_MAPPINGS,
            strict=True,
            category_aliases=(CORRECTNESS_JUDGE_TYPE,),
            metrics_label=CORRECTNESS_JUDGE_TYPE,
            score_source=CORRECTNESS_JUDGE_TYPE,
            label="correctness",
        ),
        JudgeSpec(
            name=CUSTOMER_AGENTIC_PREFERENCE_METRIC,
            kind="pairwise",
            default_min=0.50,
            judge_type=AGENTIC_JUDGE_TYPE,
            run_params=AGENTIC_RUN_PARAMS,
            input_mappings=AGENTIC_INPUT_MAPPINGS,
            strict=False,
            category_aliases=(AGENTIC_JUDGE_NAME.upper(),),
            judge_type_aliases=(AGENTIC_JUDGE_NAME,),
            row_metric=AGENTIC_PREFERENCE_RATE_METRIC,
            metrics_label=f"{AGENTIC_JUDGE_NAME} {AGENTIC_PREFERENCE_RATE_METRIC}",
            score_source=AGENTIC_JUDGE_NAME,
            label="agentic preference rate",
        ),
        JudgeSpec(
            name=COMPLETENESS_METRIC,
            kind="pointwise",
            default_min=0.7,
            judge_type=COMPLETENESS_JUDGE_TYPE,
            run_params=COMPLETENESS_RUN_PARAMS,
            strict=True,
            category_aliases=(COMPLETENESS_JUDGE_TYPE,),
            metrics_label=COMPLETENESS_JUDGE_TYPE,
            score_source=COMPLETENESS_JUDGE_TYPE,
            label="completeness",
        ),
    )
}
JUDGE_SPEC_NAMES = frozenset(JUDGE_SPECS)
DEFAULT_CUSTOMER_VALIDATION_GATES = {
    name: spec.default_min for name, spec in JUDGE_SPECS.items() if spec.kind == "pairwise"
}
