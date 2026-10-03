"""Eval-level and per-entry judge scores from Cortex."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from glean_gepa.evalcli_client import (
    AGENTIC_CORRECTNESS_JUDGE_NAME,
    AGENTIC_CORRECTNESS_RUN_PARAMS,
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

PREFERENCE_TIE = 0.5
# How long a judge's sample size must hold still before a payload with no coverage
# totals counts as finished. The agentic judge scores ~10 entries/minute.
JUDGE_SAMPLE_SETTLE_SEC = 180
# The agentic judge's summary "rate" is 10x the mean of its 0-1 per-entry scores, so a
# lagging summary can be replaced by the per-entry mean.
AGENTIC_VIEW_SCORE_SCALE = 10.0
# A judge with no coverage totals is only accepted on the settle rule once it has scored
# this share of the student run's entries; it stalls for minutes right after starting.
# Completed judges cover 93-96% (the rest are failed eval tasks it cannot grade).
MIN_JUDGE_COVERAGE = 0.85

_PendingJudge = tuple[str, str, str | None, str | None]
_JudgeKey = tuple[str, str, str | None]


@dataclass(frozen=True)
class JudgeAnalysis:
    """Eval-level rate plus optional per-entry scores and judge-run labels."""

    eval_id: str
    aggregate: float
    per_entry: dict[str, float]
    judge_run_id: str | None = None
    judge_type: str | None = None
    per_entry_feedback: dict[str, str] = field(default_factory=dict)


def run_entry_id(run_entry: Mapping[str, Any]) -> str:
    """Eval-run id on an analyze-view run entry, across the payload's key names."""
    for key in ("runId", "run_id", "evalRunId", "eval_run_id"):
        value = run_entry.get(key)
        if value:
            return str(value)
    return ""


def _label_feedback(raw_labels: Any) -> str | None:
    if isinstance(raw_labels, list):
        text = "; ".join(str(label) for label in raw_labels if label)
        return text or None
    if isinstance(raw_labels, str) and raw_labels.strip():
        return raw_labels.strip()
    return None


def per_entry_from_analysis_view(
    view: Mapping[str, Any],
    *,
    eval_id: str,
    judge_run_id: str,
    judge_type: str,
) -> tuple[dict[str, float], dict[str, str]]:
    """Pull this judge-run's per-entry scores and labels off an analyze-view payload."""
    scores: dict[str, float] = {}
    feedback: dict[str, str] = {}
    for entry in view.get("entries") or []:
        if not isinstance(entry, Mapping):
            continue
        entry_id = str(entry.get("entryId") or entry.get("id") or "")
        if not entry_id:
            continue
        for run_entry in entry.get("evalRunEntries") or entry.get("runs") or []:
            if not isinstance(run_entry, Mapping) or run_entry_id(run_entry) != eval_id:
                continue
            metadata = run_entry.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                continue
            raw_scores = metadata.get("judgeScores") or {}
            if not isinstance(raw_scores, Mapping):
                continue
            # The key is present but null for entries the judge could not grade,
            # usually a failed eval run. Leave those unscored rather than zero.
            raw_score = raw_scores.get(judge_run_id)
            if raw_score is None:
                continue
            scores[entry_id] = float(raw_score) / _per_entry_scale(judge_type)
            labels = metadata.get("judgeLabels") or {}
            if isinstance(labels, Mapping):
                text = _label_feedback(labels.get(judge_run_id))
                if text:
                    feedback[entry_id] = text
            break
    return scores, feedback


def judgeable_entry_count(view: Mapping[str, Any], *, eval_id: str) -> int:
    """Entries in an analyze-view payload that carry a run entry for ``eval_id``.

    This is the population the judge is working through, so ``scored / judgeable`` is the
    judge's coverage even when Cortex's metrics summary reports no totals.
    """
    count = 0
    for entry in view.get("entries") or []:
        if not isinstance(entry, Mapping):
            continue
        runs = entry.get("evalRunEntries") or entry.get("runs") or []
        if any(isinstance(run_entry, Mapping) and run_entry_id(run_entry) == eval_id for run_entry in runs):
            count += 1
    return count


@dataclass(frozen=True)
class _ViewScores:
    per_entry: dict[str, float]
    feedback: dict[str, str]
    loaded: bool
    judgeable: int = 0


def _per_entry_from_evalcli(
    evalcli: EvalCliClient,
    *,
    eval_id: str,
    judge_run_id: str | None,
    judge_type: str,
    base_eval_id: str | None,
) -> _ViewScores:
    """Load per-entry scores from analyze view. ``loaded`` is False if the view is not ready."""
    get_view = getattr(evalcli, "get_analysis_view", None)
    if not callable(get_view) or not judge_run_id:
        return _ViewScores({}, {}, False)
    try:
        view = get_view(eval_id, base_eval_id=base_eval_id)
    except Exception as exc:
        print(f"[{eval_id}] Could not load analysis view for per-entry judge scores: {exc}")
        return _ViewScores({}, {}, False)
    if not isinstance(view, Mapping):
        return _ViewScores({}, {}, False)
    scores, feedback = per_entry_from_analysis_view(
        view, eval_id=eval_id, judge_run_id=judge_run_id, judge_type=judge_type
    )
    return _ViewScores(scores, feedback, True, judgeable_entry_count(view, eval_id=eval_id))


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    try:
        text = str(value).strip()
    except Exception:
        return None
    if text.isdigit():
        return int(text)
    return None


def _judge_metrics_charts(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split judgeMetrics into coverage totals and the innermost chart object."""
    charts = payload.get("judgeMetrics") or {}
    if not isinstance(charts, dict):
        return {}, {}
    totals: dict[str, Any] = {}
    current: dict[str, Any] = charts
    while True:
        for key in ("totalEntries", "missingEntries"):
            if current.get(key) is not None:
                totals[key] = current[key]
        inner = current.get("additional_properties")
        if not isinstance(inner, dict):
            return totals, current
        current = inner


def _matching_judge_metric_row(
    charts: Mapping[str, Any],
    *,
    judge_type: str,
    judge_run_id: str | None,
) -> dict[str, Any] | None:
    wanted_type = judge_type.upper()
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
        if judge_run_id and row_judge_id:
            if str(row_judge_id) == judge_run_id:
                return row
            continue
        if chart == wanted_type:
            return row
    return None


def _row_pass_rate(row: Mapping[str, Any]) -> float | None:
    value = row.get("test") if row.get("test") is not None else row.get("passRate")
    return None if value is None else float(value)


@dataclass(frozen=True)
class JudgeMetricsSnapshot:
    """One ``metrics summary`` payload for a judge run."""

    rate: float | None
    sample_size: int | None = None
    total_entries: int | None = None
    missing_entries: int | None = None

    @property
    def coverage_known(self) -> bool:
        """Whether Cortex reported enough to decide completeness from this payload alone."""
        return self.missing_entries is not None or (self.total_entries is not None and self.sample_size is not None)

    @property
    def coverage_complete(self) -> bool:
        """True once Cortex has finished scoring this judge, not merely published a mean.

        When Cortex reports neither ``missingEntries`` nor ``totalEntries`` (the agentic
        judge does this while it is still scoring), a partial sample is indistinguishable
        from a finished one here; the caller must confirm the sample has stopped growing.
        """
        if self.rate is None:
            return False
        if self.missing_entries == 0:
            return True
        if self.total_entries is not None and self.sample_size is not None:
            return self.sample_size >= self.total_entries
        return False

    def view_caught_up(self, per_entry_count: int) -> bool:
        """True when analyze-view rows have caught the metrics sample (or there is no sample)."""
        if self.sample_size is None:
            return True
        return per_entry_count >= self.sample_size

    def progress_label(self) -> str:
        scored = self.sample_size
        total = self.total_entries
        if scored is None and total is None:
            return "coverage unknown"
        if total is None:
            return f"{scored} scored"
        return f"{scored if scored is not None else 0}/{total} scored"


def judge_metrics_snapshot(
    payload: Mapping[str, Any],
    *,
    judge_type: str,
    judge_run_id: str | None = None,
) -> JudgeMetricsSnapshot:
    """Read pass rate and coverage counts from a metrics summary payload."""
    totals, charts = _judge_metrics_charts(payload)
    row = _matching_judge_metric_row(charts, judge_type=judge_type, judge_run_id=judge_run_id)
    if row is None:
        return JudgeMetricsSnapshot(
            rate=None,
            total_entries=_as_int(totals.get("totalEntries")),
            missing_entries=_as_int(totals.get("missingEntries")),
        )
    missing = row.get("missingEntries")
    return JudgeMetricsSnapshot(
        rate=_row_pass_rate(row),
        sample_size=_as_int(row.get("sampleSize")),
        total_entries=_as_int(row.get("totalEntries")) or _as_int(totals.get("totalEntries")),
        missing_entries=_as_int(missing if missing is not None else totals.get("missingEntries")),
    )


def _pending_judge_key(spec: _PendingJudge) -> _JudgeKey:
    eval_id, judge_type, _judge_run_id, base_eval_id = spec
    return eval_id, judge_type, base_eval_id


def _dedupe_pending(pending: Sequence[_PendingJudge]) -> list[_PendingJudge]:
    unique: list[_PendingJudge] = []
    seen: set[_JudgeKey] = set()
    for spec in pending:
        key = _pending_judge_key(spec)
        if key in seen:
            continue
        seen.add(key)
        unique.append(spec)
    return unique


def _load_complete_analysis(
    evalcli: EvalCliClient,
    *,
    eval_id: str,
    judge_type: str,
    judge_run_id: str | None,
    base_eval_id: str | None,
    snapshot: JudgeMetricsSnapshot,
) -> JudgeAnalysis | None:
    """Return an analysis once metrics coverage and analyze-view rows have caught up."""
    rate = snapshot.rate
    if rate is None:
        return None
    view = _per_entry_from_evalcli(
        evalcli,
        eval_id=eval_id,
        judge_run_id=judge_run_id,
        judge_type=judge_type,
        base_eval_id=base_eval_id,
    )
    if view.loaded and not snapshot.view_caught_up(len(view.per_entry)):
        print(
            f"[{judge_type}] {eval_id}: {rate:.2f} "
            f"({len(view.per_entry)} per-entry view, {snapshot.progress_label()}; waiting)"
        )
        return None
    print(f"[{judge_type}] {eval_id}: {rate:.2f} ({len(view.per_entry)} per-entry)")
    return JudgeAnalysis(
        eval_id=eval_id,
        aggregate=rate,
        per_entry=view.per_entry,
        judge_run_id=judge_run_id,
        judge_type=judge_type,
        per_entry_feedback=view.feedback,
    )


def _count_has_settled(
    count: int,
    key: _JudgeKey,
    stable_polls: dict[_JudgeKey, tuple[int, int]],
    *,
    poll_interval_sec: int,
    settle_sec: int,
) -> bool:
    """True once ``count`` has held still for ``settle_sec`` (at least two polls)."""
    if count <= 0:
        stable_polls.pop(key, None)
        return False
    previous = stable_polls.get(key)
    if previous is None or previous[0] != count:
        stable_polls[key] = (count, 1)
        return False
    polls = previous[1] + 1
    stable_polls[key] = (count, polls)
    interval = poll_interval_sec if poll_interval_sec > 0 else 1
    return polls >= 2 and (polls - 1) * interval >= settle_sec


def _settle_without_totals(
    evalcli: EvalCliClient,
    *,
    eval_id: str,
    judge_type: str,
    judge_run_id: str | None,
    base_eval_id: str | None,
    snapshot: JudgeMetricsSnapshot,
    key: _JudgeKey,
    stable_polls: dict[_JudgeKey, tuple[int, int]],
    poll_interval_sec: int,
    settle_sec: int,
) -> JudgeAnalysis | None:
    """Completeness fallback for judges whose metrics payload carries no coverage totals.

    The agentic judge publishes a running mean with null ``totalEntries``/``missingEntries``,
    and that summary can lag far behind the analyze view. Progress is the larger of the
    summary sample and the view's row count; the judge is finished once that holds still
    for ``settle_sec`` and covers at least ``MIN_JUDGE_COVERAGE`` of the student run's
    entries. When the summary is behind the view, the aggregate is recomputed from the
    per-entry scores.
    """
    if snapshot.rate is None:
        stable_polls.pop(key, None)
        return None
    view = _per_entry_from_evalcli(
        evalcli,
        eval_id=eval_id,
        judge_run_id=judge_run_id,
        judge_type=judge_type,
        base_eval_id=base_eval_id,
    )
    per_entry = view.per_entry
    sample = snapshot.sample_size or 0
    if not view.loaded:
        stable_polls.pop(key, None)
        print(
            f"[{judge_type}] {eval_id}: {snapshot.rate:.2f} ({snapshot.progress_label()}; no coverage totals "
            f"and the analyze view is unavailable, so coverage cannot be checked; waiting)"
        )
        return None
    count = max(sample, len(per_entry))
    settled = _count_has_settled(count, key, stable_polls, poll_interval_sec=poll_interval_sec, settle_sec=settle_sec)
    required = math.ceil(MIN_JUDGE_COVERAGE * view.judgeable)
    if view.judgeable > 0 and count < required:
        print(
            f"[{judge_type}] {eval_id}: {snapshot.rate:.2f} ({count} of {view.judgeable} entries scored, "
            f"need {required} before a stalled count can mean complete; waiting)"
        )
        return None
    if not settled:
        print(
            f"[{judge_type}] {eval_id}: {snapshot.rate:.2f} ({count} of {view.judgeable} entries scored; "
            f"no coverage totals, waiting for the count to settle)"
        )
        return None
    if len(per_entry) > sample:
        aggregate = AGENTIC_VIEW_SCORE_SCALE * sum(per_entry.values()) / len(per_entry)
        print(
            f"[{judge_type}] {eval_id}: metrics summary lagged ({snapshot.rate:.2f} over {sample} scored); "
            f"using {aggregate:.2f} from {len(per_entry)} per-entry scores, unchanged for {settle_sec}s"
        )
    else:
        aggregate = snapshot.rate
        print(
            f"[{judge_type}] {eval_id}: {aggregate:.2f} ({count} of {view.judgeable} entries scored; Cortex "
            f"reported no coverage totals, sample unchanged for {settle_sec}s, treating as complete)"
        )
    return JudgeAnalysis(
        eval_id=eval_id,
        aggregate=aggregate,
        per_entry=per_entry,
        judge_run_id=judge_run_id,
        judge_type=judge_type,
        per_entry_feedback=view.feedback,
    )


def wait_for_all_judge_metrics(
    evalcli: EvalCliClient,
    pending: Sequence[tuple[str, str, str | None, str | None]],
    *,
    poll_interval_sec: int = 60,
    timeout_sec: int = 3600,
    settle_sec: int = JUDGE_SAMPLE_SETTLE_SEC,
) -> dict[tuple[str, str, str | None], JudgeAnalysis]:
    """Poll every pending judge, then read analyses only after all have finished."""
    unique = _dedupe_pending(pending)
    if not unique:
        return {}

    print(f"Waiting for metrics on {len(unique)} judge run(s)...")
    elapsed = 0
    snapshots: dict[_JudgeKey, JudgeMetricsSnapshot] = {}
    # (last scored count, consecutive polls at that count) for judges whose payload omits coverage totals.
    stable_polls: dict[_JudgeKey, tuple[int, int]] = {}
    # Analyses already finalized through the no-totals path; they are not re-read from the summary.
    settled: dict[_JudgeKey, JudgeAnalysis] = {}
    while elapsed <= timeout_sec:
        coverage_ready = True
        for eval_id, judge_type, judge_run_id, base_eval_id in unique:
            key = _pending_judge_key((eval_id, judge_type, judge_run_id, base_eval_id))
            if key in settled:
                continue
            try:
                payload = evalcli.get_eval_metrics(eval_id, base_eval_id=base_eval_id)
            except EvalCliError as exc:
                print(f"[{judge_type}] Transient metrics error for {eval_id}: {exc}")
                coverage_ready = False
                continue
            snapshot = judge_metrics_snapshot(payload, judge_type=judge_type, judge_run_id=judge_run_id)
            snapshots[key] = snapshot
            if snapshot.coverage_complete:
                continue
            if not snapshot.coverage_known:
                analysis = _settle_without_totals(
                    evalcli,
                    eval_id=eval_id,
                    judge_type=judge_type,
                    judge_run_id=judge_run_id,
                    base_eval_id=base_eval_id,
                    snapshot=snapshot,
                    key=key,
                    stable_polls=stable_polls,
                    poll_interval_sec=poll_interval_sec,
                    settle_sec=settle_sec,
                )
                if analysis is not None:
                    settled[key] = analysis
                    continue
            coverage_ready = False
            if snapshot.rate is not None and snapshot.coverage_known:
                print(
                    f"[{judge_type}] {eval_id}: {snapshot.rate:.2f} "
                    f"({snapshot.progress_label()}; waiting for remaining entries)"
                )

        if coverage_ready and len(snapshots) == len(unique):
            analyses: dict[_JudgeKey, JudgeAnalysis] = dict(settled)
            views_ready = True
            for eval_id, judge_type, judge_run_id, base_eval_id in unique:
                key = _pending_judge_key((eval_id, judge_type, judge_run_id, base_eval_id))
                if key in analyses:
                    continue
                analysis = _load_complete_analysis(
                    evalcli,
                    eval_id=eval_id,
                    judge_type=judge_type,
                    judge_run_id=judge_run_id,
                    base_eval_id=base_eval_id,
                    snapshot=snapshots[key],
                )
                if analysis is None:
                    views_ready = False
                    continue
                analyses[key] = analysis
            if views_ready and len(analyses) == len(unique):
                return analyses

        if elapsed >= timeout_sec:
            break
        time.sleep(poll_interval_sec)
        elapsed += poll_interval_sec if poll_interval_sec > 0 else 1

    unfinished = []
    for spec in unique:
        snapshot = snapshots.get(_pending_judge_key(spec))
        label = snapshot.progress_label() if snapshot is not None else "no metrics"
        unfinished.append(f"{spec[0]} ({label})")
    raise EvalCliError(f"Judge metrics were not ready after {timeout_sec}s: {', '.join(unfinished)}")


CUSTOMER_CORRECTNESS_METRIC = "correctness"
CUSTOMER_AGENTIC_PREFERENCE_METRIC = "agentic_preference_rate"
CUSTOMER_AGENTIC_CORRECTNESS_METRIC = "agentic_correctness_rate"
# Adapter-side key for the single-dimension agentic correctness judge. It is NOT a Cortex
# judge type; Cortex sees AGENTIC_JUDGE (``JudgeSpec.cortex_judge_type``). A distinct key
# keeps its judge runs and cached analyses apart from the multi-dimension judge, which
# shares the Cortex type.
AGENTIC_CORRECTNESS_JUDGE_TYPE = "AGENTIC_CORRECTNESS_JUDGE"
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
    #: Upper bound of this judge's raw per-entry score, as set by its scoring
    #: mode in ``run_params``. Keep the two in sync.
    per_entry_scale: float = 1.0
    score_keys: tuple[str, ...] = ("test", "passRate", "pass_rate", "testValue", "test_value")
    #: Judge type sent to Cortex when it differs from ``judge_type``. ``judge_type`` is the
    #: adapter's cache / dedupe key; several specs may map onto one Cortex type.
    cortex_type_override: str | None = None
    #: ``judge_skill_name`` in ``run_params``; disambiguates runs that share a Cortex type.
    judge_skill_name: str | None = None

    @property
    def cortex_judge_type(self) -> str:
        return self.cortex_type_override or self.judge_type

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
            # scoring_mode randomized_single_0_10: 0-4 lost to the baseline,
            # 5 ties, 6-10 won. Normalizing puts the tie on PREFERENCE_TIE.
            per_entry_scale=10.0,
            judge_skill_name=AGENTIC_JUDGE_NAME,
        ),
        JudgeSpec(
            name=CUSTOMER_AGENTIC_CORRECTNESS_METRIC,
            kind="pairwise",
            default_min=0.50,
            judge_type=AGENTIC_CORRECTNESS_JUDGE_TYPE,
            cortex_type_override=AGENTIC_JUDGE_TYPE,
            run_params=AGENTIC_CORRECTNESS_RUN_PARAMS,
            input_mappings=AGENTIC_INPUT_MAPPINGS,
            strict=False,
            category_aliases=(AGENTIC_CORRECTNESS_JUDGE_NAME.upper(),),
            judge_type_aliases=(AGENTIC_CORRECTNESS_JUDGE_NAME,),
            row_metric=AGENTIC_PREFERENCE_RATE_METRIC,
            metrics_label=f"{AGENTIC_CORRECTNESS_JUDGE_NAME} {AGENTIC_PREFERENCE_RATE_METRIC}",
            score_source=AGENTIC_CORRECTNESS_JUDGE_NAME,
            label="agentic correctness rate",
            # Same randomized_single_0_10 scoring as the multi-dimension judge; this one
            # scores correctness only. 0.5 = parity with the teacher on correctness.
            per_entry_scale=10.0,
            judge_skill_name=AGENTIC_CORRECTNESS_JUDGE_NAME,
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
# Gates started when a run names none. The correctness-only agentic judge is opt-in:
# it duplicates the multi-dimension judge's correctness scorer, so running both by
# default would double the agentic judge cost for no new signal.
_OPT_IN_GATES = frozenset({CUSTOMER_AGENTIC_CORRECTNESS_METRIC})
DEFAULT_CUSTOMER_VALIDATION_GATES = {
    name: spec.default_min
    for name, spec in JUDGE_SPECS.items()
    if spec.kind == "pairwise" and name not in _OPT_IN_GATES
}


def _per_entry_scale(judge_type: str) -> float:
    """Declared raw-score upper bound. AGENTIC's 0-10 includes a literal 1.0 loss."""
    for spec in JUDGE_SPECS.values():
        if spec.judge_type == judge_type:
            return spec.per_entry_scale
    return 1.0
