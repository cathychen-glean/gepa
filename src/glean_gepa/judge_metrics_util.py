"""Aggregate judge scores from Cortex eval metrics.

After ``judge create``, scores are read from ``POST /metrics/evalruns/pairwise``
(evalcli ``metrics summary``). Judge-run status is listed with
``GET /judgeruns?evalRunIds=`` (evalcli ``judge list``).
``GET /judgeruns/{id}`` and ``list-for-run`` are not used; those Cortex routes
are unimplemented.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from glean_gepa.evalcli_client import EvalCliClient, EvalCliError


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
            rate = judge_pass_rate_from_metrics(
                last_payload, judge_type=judge_type, judge_run_id=judge_run_id
            )
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
        f"{judge_type} metrics for {eval_id} were not ready after {timeout_sec}s "
        f"(judge_run_id={judge_run_id})"
    )
