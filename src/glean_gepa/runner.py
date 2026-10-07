"""CLI and low-level GEPA engine wiring for Glean prompt optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

from gepa.core.data_loader import ListDataLoader
from gepa.core.state import FrontierType
from gepa.logging.experiment_tracker import create_experiment_tracker
from gepa.logging.logger import StdOutLogger
from glean_gepa.adapter_types import ALDataInst, EvalHarness, JudgingMode
from glean_gepa.al_adapter import (
    ALRunner,
    ModuleSpec,
)
from glean_gepa.api import optimize
from glean_gepa.bigquery_client import BigQueryClient
from glean_gepa.debug import set_debug
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.evalset_policy import UnseenEvalSetPolicy
from glean_gepa.evolutionary_proposer import EvolutionaryProposer
from glean_gepa.experiment_config import (
    DEFAULT_DEPLOYMENT_IDS,
    DEFAULT_EVAL_SET_NAME,
    ExperimentConfig,
    ExperimentConfigError,
    composite_weights,
    constant_scores,
    customer_validation_gates,
    eval_harness,
    evalset_identity,
    experiment_objective_spec,
    load_experiment_config,
    pairwise_judges,
    pointwise_judges,
    resolve_config_path,
    runner_arg_defaults,
    screening_threshold,
    screening_weights,
)
from glean_gepa.fake_flow import build_fake_flow_components
from glean_gepa.harnesses import harness_for_model
from glean_gepa.judge_metrics_util import JUDGE_SPECS
from glean_gepa.objectives.registry import build_objective
from glean_gepa.objectives.utils.agentspan_query import DEFAULT_LOOKBACK_DAYS
from glean_gepa.openai_client import create_qe_openai_client, format_exception_chain, get_perfeval_secret
from glean_gepa.prompt import candidate_module_names, compile_encoded_prompt
from glean_gepa.prompt_constants import WRITING_CODE_KEY
from glean_gepa.prompt_targets import (
    PromptTargetError,
    build_seed,
    harness_requirements,
    load_seed_file,
    module_token_budget,
    parse_editable_modules,
    render_requirements,
)
from glean_gepa.run_log import capture_run_log, log_section
from glean_gepa.single_model_adapter import SingleModelAdapter
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter

CACHE_DIRECTORY_NAME = "cache"
ADAPTER_CACHE_FILENAME = "glean_adapter_cache.json"
EVAL_RUN_CACHE_FILENAME = "glean_eval_run_cache.json"
CHILDREN_CACHE_FILENAME = "glean_children_cache.json"
GEPA_STATE_FILENAME = "gepa_state.bin"
# GEPAState.i starts at -1 and the engine increments it before the first propose.
_FIRST_ITERATION_ATTEMPT = 0
_TEACHER_PROMPT_SENTINEL = "<<TEACHER_PROD_PROMPT>>"
RUN_LOG_FILENAME = "gepa_run.log"
EVALSET_SCHEDULE_FILENAME = "glean_evalset_schedule.json"
CUSTOMER_DEPLOYMENTS_FILENAME = "glean_customer_deployments.json"
# Same values the config falls back to when `data` omits them.
GLEAN_CHAT_EVAL_SET_NAME: str = DEFAULT_EVAL_SET_NAME
SCIO_PROD_DEPLOYMENT_IDS: list[str] = list(DEFAULT_DEPLOYMENT_IDS)
CUSTOMER_EVAL_DEPLOYMENT_IDS = [
    "bill",
    "guild",
    "gxs",
    "happyreturns",
    "howardhughes",
    "mccarthy",
    "motive-prod",
    "pricefx-prod",
    "seatgeek",
    "tealium",
    "televox",
    "thoughtworks",
]
# Claude-capable subset of the customer pool, plus scio-prod.
TEACHER_STUDENT_DEPLOYMENT_IDS = [
    "bill",
    "guild",
    "happyreturns",
    "howardhughes",
    "seatgeek",
    "televox",
    "thoughtworks",
]
CUSTOMER_EVAL_ALPHA = 0.05
# Cortex rejects an eval run with more than five deployments, so each run samples
# this many from the customer pool above and keeps that sample for every eval.
MAX_EVAL_RUN_DEPLOYMENTS = 1


def _default_cache_file(run_dir: Path | None, filename: str) -> Path | None:
    """Return a run-local cache path, moving a legacy root-level cache if needed."""
    if run_dir is None:
        return None

    cache_file = run_dir / CACHE_DIRECTORY_NAME / filename
    legacy_file = run_dir / filename
    if legacy_file.exists() and not cache_file.exists():
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        legacy_file.replace(cache_file)
    return cache_file


def _load_seed_candidate(path: Path | None) -> dict[str, str]:
    """Load prompt-module overrides from JSON or a marked ``.prompt``. Omitted keys use stock text."""
    if path is None:
        return {}
    try:
        return load_seed_file(path)
    except PromptTargetError as exc:
        raise SystemExit(str(exc)) from exc


def _parse_editable_modules(raw: str) -> list[str]:
    try:
        return parse_editable_modules([part.strip() for part in raw.split(",") if part.strip()])
    except PromptTargetError as exc:
        raise SystemExit(str(exc)) from exc


def _seed_for_editable_modules(raw: dict[str, str], editable_modules: list[str]) -> dict[str, str]:
    """Build the GEPA candidate dict for the requested editable modules."""
    try:
        return build_seed(raw, editable_modules)
    except PromptTargetError as exc:
        raise SystemExit(str(exc)) from exc


def _eval_harness_for(
    experiment: ExperimentConfig | None,
    editable_modules: Sequence[str],
    models: Sequence[str],
) -> EvalHarness:
    """The ``eval:`` harness plus the scParams the edited prompts need to render.

    Fails when a model runs a harness that never renders one of the edited prompts.
    """
    for target, harnesses in harness_requirements(editable_modules).items():
        for model in models:
            harness = harness_for_model(model).name
            if harness not in harnesses:
                raise SystemExit(
                    f"{target} renders under the {'/'.join(harnesses)} harness, but model {model!r} runs the "
                    f"{harness} harness. Pick a matching model or drop {target} modules from editable_modules."
                )
    extra, dropped = render_requirements(editable_modules)
    return eval_harness(experiment)._replace(extra_sc_params=extra, drop_sc_params=dropped)


def _make_reflection_lm(
    model: str,
    *,
    qe_project: str,
    qe_instance: str,
    authenticated_email: str,
    max_tokens: int = 4096,
) -> Callable[[str], str]:
    client = create_qe_openai_client(qe_instance)
    call_count = 0

    def reflection_lm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        print(f"QE reflection LLM call {call_count}: requesting model={model}, prompt_chars={len(prompt)}")
        print(f"QE reflection LLM call {call_count} prompt:\n{prompt}")
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_tokens,
                extra_body={
                    "perf_eval_secret": get_perfeval_secret(qe_project),
                    "source_info": {
                        "clientInitiator": "USER",
                        "feature": "INTEGRATION_TEST",
                    },
                    "authenticated_email": authenticated_email,
                },
            )
            response_text = response.output_text.strip()
            if not response_text:
                raise RuntimeError("QE reflection LLM returned an empty response")
            print(f"QE reflection LLM call {call_count}: received response_chars={len(response_text)}")
            print(f"QE reflection LLM call {call_count} response:\n{response_text}")
            return response_text
        except Exception as exc:
            message = format_exception_chain(exc)
            raise RuntimeError(f"QE reflection LLM call failed: {message}") from exc

    return reflection_lm


def _parse_reflection_samples(value: str) -> int | None:
    if value.lower() == "all":
        return None
    try:
        sample_count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("reflection_samples must be a positive integer or 'all'") from exc
    if sample_count <= 0:
        raise argparse.ArgumentTypeError("reflection_samples must be a positive integer or 'all'")
    return sample_count


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parse_eval_versions(value: str, *, argument_name: str) -> list[str]:
    versions = [version.strip() for version in value.split(",") if version.strip()]
    if not versions:
        raise SystemExit(f"{argument_name} must contain at least one eval version")
    return versions


def _prestart_first_iteration_training(
    *,
    runner: ALRunner,
    policy: UnseenEvalSetPolicy,
    trainset: list[ALDataInst],
    seed_candidate: dict[str, str],
    student_model: str,
    teacher_model: str,
    run_dir: Path | None,
) -> None:
    """Start the first training slice's teacher and seed-student evals without waiting.

    The seed validation blocks the first iteration, so these runs would otherwise
    sit idle until it finishes. A resumed run already has ``gepa_state.bin`` and
    starts its own slice inside propose, so this only runs for a fresh first iteration.
    ``take_unseen`` records the slice as pending at attempt 0, which is the counter
    the engine passes on that first propose, so the generation reuses this slice.
    """
    if run_dir is not None and (run_dir / GEPA_STATE_FILENAME).is_file():
        return
    loader = ListDataLoader(trainset)
    try:
        train_ids = policy.take_unseen(
            loader,
            purpose="first-iteration training prestart",
            attempt=_FIRST_ITERATION_ATTEMPT,
        )
    except RuntimeError as exc:
        print(f"[Eval set schedule] Skipping first-iteration training prestart: {exc}")
        return
    student_prompt = compile_encoded_prompt(seed_candidate)
    for item in loader.fetch(train_ids):
        eval_set_name = str(item["eval_set_name"])
        eval_set_version = str(item["eval_set_version"])
        deployment_ids = [str(deployment_id) for deployment_id in item.get("deployment_ids", [])]
        print(f"[Eval set schedule] Prestarting first-iteration training evals for {eval_set_name}:{eval_set_version}")
        for role, model, prompt in (
            ("teacher", teacher_model, _TEACHER_PROMPT_SENTINEL),
            ("student", student_model, student_prompt),
        ):
            eval_id, wait_required = runner.start(
                model,
                prompt,
                eval_set_name,
                eval_set_version,
                deployment_ids,
            )
            print(f"[Prestart] {role} {eval_set_name}:{eval_set_version} -> {eval_id} (wait={wait_required})")


def _val_eval_set_name(args: argparse.Namespace, train_eval_set_name: str) -> str:
    """Eval set for validation.

    A pinned val set otherwise reuses the training eval set. Automatic customer
    validation stays on ``Glean Chat V2 Medium`` unless this override is set.
    """
    override = getattr(args, "val_eval_set_name", None)
    if override:
        return str(override)
    if args.val_eval_versions:
        return train_eval_set_name
    return GLEAN_CHAT_EVAL_SET_NAME


def _make_evalset(
    versions: list[str],
    *,
    eval_set_name: str = GLEAN_CHAT_EVAL_SET_NAME,
    deployment_ids: list[str] | None = None,
    validation_only: bool = False,
) -> list[ALDataInst]:
    """Build eval-set items for ``versions``.

    ``validation_only`` keeps customer eval sets to eval-run metrics: their
    entries are PII-gated, so they must never back a focused high-signal set.
    """
    ids = list(deployment_ids) if deployment_ids is not None else list(SCIO_PROD_DEPLOYMENT_IDS)
    return [
        {
            "eval_set_name": eval_set_name,
            "eval_set_version": version,
            "deployment_ids": ids,
            "status": "active",
            **({"validation_only": True} if validation_only else {}),
        }
        for version in versions
    ]


def _dated_eval_versions(version_rows: list[dict[str, object]]) -> list[tuple[date, str]]:
    dated: set[tuple[date, str]] = set()
    for row in version_rows:
        raw_version = row.get("version") or row.get("evalSetVersion")
        if not isinstance(raw_version, str) or not re.fullmatch(r"\d{8}", raw_version):
            continue
        try:
            version_date = date.fromisoformat(f"{raw_version[:4]}-{raw_version[4:6]}-{raw_version[6:]}")
        except ValueError:
            continue
        dated.add((version_date, raw_version))
    return sorted(dated)


def _dated_eval_version_rows(version_rows: list[dict[str, object]]) -> list[tuple[date, str, dict[str, object]]]:
    """Return dated versions newest first, each with the row it came from."""
    rows_by_version: dict[str, tuple[date, dict[str, object]]] = {}
    for row in version_rows:
        raw_version = row.get("version") or row.get("evalSetVersion")
        if not isinstance(raw_version, str) or not re.fullmatch(r"\d{8}", raw_version):
            continue
        try:
            version_date = date.fromisoformat(f"{raw_version[:4]}-{raw_version[4:6]}-{raw_version[6:]}")
        except ValueError:
            continue
        rows_by_version.setdefault(raw_version, (version_date, row))
    ordered = sorted(
        ((version_date, version, row) for version, (version_date, row) in rows_by_version.items()),
        reverse=True,
    )
    return ordered


def _deployments_missing_from_version(row: dict[str, object], deployment_ids: list[str]) -> set[str]:
    """Return the deployments this version was not published to with entries.

    ``evalsets versions`` reports ``availableDeploymentIds`` plus a
    ``perDeploymentMetadata`` size per deployment. Most dated versions cover
    only scio-prod; the customer-wide ones land roughly weekly. This is the
    authoritative signal — the entries endpoint is PII-gated for customer
    deployments and ``fact.evalset_entries`` keys rows by a different id.
    """
    available = _unwrap_additional_properties(
        row.get("availableDeploymentIds") or row.get("available_deployment_ids") or []
    )
    published = {str(item) for item in available} if isinstance(available, list) else set()
    sizes = _unwrap_additional_properties(row.get("perDeploymentMetadata") or {})
    missing: set[str] = set()
    for deployment in deployment_ids:
        if published and deployment not in published:
            missing.add(deployment)
            continue
        if not isinstance(sizes, dict) or not sizes:
            continue
        entry = _unwrap_additional_properties(sizes.get(deployment) or {})
        size = entry.get("size") if isinstance(entry, dict) else None
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            missing.add(deployment)
    return missing


def _select_covered_dated_versions(
    version_rows: list[dict[str, object]],
    *,
    count: int,
    eval_set_name: str,
    deployment_ids: list[str],
) -> list[str]:
    """Pick the newest dated versions published to every requested deployment."""
    selected: list[str] = []
    skipped: list[str] = []
    for _version_date, version, row in _dated_eval_version_rows(version_rows):
        missing = _deployments_missing_from_version(row, deployment_ids)
        if missing:
            skipped.append(f"{version} (not published to {','.join(sorted(missing))})")
            continue
        selected.append(version)
        if len(selected) == count:
            break
    if skipped:
        print(f"[Eval set schedule] Skipped {eval_set_name} versions: {'; '.join(skipped)}")
    if not selected:
        raise SystemExit(
            f"No dated {eval_set_name} version is published to every deployment "
            f"{','.join(deployment_ids)}. Customer-wide versions land roughly weekly; "
            f"pin --val_eval_versions or lower --val_eval_version_count."
        )
    return list(reversed(selected))


def _resolve_customer_deployments(
    state_file: Path | None,
    *,
    seed: int | None = None,
    pool: list[str] | None = None,
) -> list[str]:
    """Sample the customer deployments this run evaluates on."""
    candidates = list(CUSTOMER_EVAL_DEPLOYMENT_IDS if pool is None else pool)
    if state_file is not None and state_file.is_file():
        saved = json.loads(state_file.read_text())
        if not isinstance(saved, list) or not saved:
            raise SystemExit(f"{state_file} does not hold a list of customer deployments: {saved!r}")
        if any(item not in candidates for item in saved):
            raise SystemExit(f"{state_file} does not hold a list of customer deployments: {saved!r}")
        return [str(item) for item in saved]

    sampled = sorted(random.Random(seed).sample(candidates, MAX_EVAL_RUN_DEPLOYMENTS))
    if state_file is not None:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(sampled))
    return sampled


def _select_recent_train_versions(
    version_rows: list[dict[str, object]], *, as_of: date, lookback_days: int
) -> list[str]:
    """Schedule every scio-prod version in the lookback window for training."""
    earliest = as_of - timedelta(days=lookback_days)
    ordered_versions = [
        version for version_date, version in _dated_eval_versions(version_rows) if earliest <= version_date <= as_of
    ]
    if not ordered_versions:
        raise SystemExit(
            f"Need at least one scio-prod eval version dated {earliest.isoformat()} through {as_of.isoformat()}."
        )
    return ordered_versions


def _customer_version_rows(evalcli: EvalCliClient, customer_deployments: list[str]) -> list[dict[str, Any]]:
    return evalcli.list_eval_set_versions(
        eval_set_name=GLEAN_CHAT_EVAL_SET_NAME,
        deployment_ids=customer_deployments,
    )


def _require_customer_publication(
    evalcli: EvalCliClient, val_versions: list[str], customer_deployments: list[str]
) -> None:
    """Fail before any eval run when a pinned version misses a sampled deployment."""
    rows_by_version = {
        version: row
        for _version_date, version, row in _dated_eval_version_rows(
            _customer_version_rows(evalcli, customer_deployments)
        )
    }
    for version in val_versions:
        row = rows_by_version.get(version)
        if row is None:
            print(
                f"[Eval set schedule] {GLEAN_CHAT_EVAL_SET_NAME} version {version} is not listed for these deployments"
            )
            continue
        missing = _deployments_missing_from_version(row, customer_deployments)
        if missing:
            raise SystemExit(
                f"{GLEAN_CHAT_EVAL_SET_NAME} version {version} is not published to "
                f"{','.join(sorted(missing))}; pick a version published to every sampled deployment."
            )


def _resolve_customer_val_versions(
    args: argparse.Namespace,
    evalcli: EvalCliClient,
    customer_deployments: list[str],
) -> list[str]:
    """Return the customer-deployment versions used for both GEPA val and the final gate."""
    return _select_covered_dated_versions(
        _customer_version_rows(evalcli, customer_deployments),
        count=args.val_eval_version_count,
        eval_set_name=GLEAN_CHAT_EVAL_SET_NAME,
        deployment_ids=customer_deployments,
    )


def _resolve_eval_version_split(
    args: argparse.Namespace,
    evalcli: EvalCliClient,
    customer_deployments: list[str],
) -> tuple[list[str], list[str]]:
    eval_set_name, deployment_ids = evalset_identity(args.experiment)
    # Pinned val versions run on data.deployment_ids and skip the customer
    # publication check; omitting --val_eval_versions selects customer validation.
    if args.val_eval_versions:
        val_versions = _parse_eval_versions(args.val_eval_versions, argument_name="--val_eval_versions")
        if args.train_eval_versions:
            train_versions = _parse_eval_versions(args.train_eval_versions, argument_name="--train_eval_versions")
        else:
            rows = evalcli.list_eval_set_versions(eval_set_name=eval_set_name, deployment_ids=deployment_ids)
            days_back = getattr(args, "eval_version_days_back", 0) or 0
            as_of = date.today() - timedelta(days=days_back)
            train_versions = _select_recent_train_versions(
                rows,
                as_of=as_of,
                lookback_days=args.eval_version_lookback_days,
            )
        print(
            f"[Eval set schedule] TEMP pinned val versions={','.join(val_versions)} "
            f"on {','.join(deployment_ids)} from data, not customer deployments"
        )
        if not 1 <= len(val_versions) <= 2:
            raise SystemExit("Validation must contain one or two eval versions.")
        return train_versions, val_versions
    if args.train_eval_versions:
        raise SystemExit("Set --val_eval_versions with --train_eval_versions, or neither for automatic selection.")
    rows = evalcli.list_eval_set_versions(eval_set_name=eval_set_name, deployment_ids=deployment_ids)
    days_back = getattr(args, "eval_version_days_back", 0) or 0
    as_of = date.today() - timedelta(days=days_back)
    train_versions = _select_recent_train_versions(
        rows,
        as_of=as_of,
        lookback_days=args.eval_version_lookback_days,
    )
    val_versions = _resolve_customer_val_versions(args, evalcli, customer_deployments)
    as_of_note = f" as of {as_of.isoformat()} ({days_back}d back)" if days_back else ""
    print(
        f"[Eval set schedule] Auto-selected{as_of_note} "
        f"train versions={','.join(train_versions)} (scio-prod) and "
        f"val versions={','.join(val_versions)} on {','.join(customer_deployments)}"
    )

    if not 1 <= len(val_versions) <= 2:
        raise SystemExit("Validation must contain one or two eval versions.")
    return train_versions, val_versions


def _unwrap_additional_properties(value: object) -> object:
    while isinstance(value, dict) and set(value) == {"additional_properties"}:
        value = value["additional_properties"]
    return value


# Keys that name a category themselves. Rows nested under any other key are dropped
# unless an ancestor carried an explicit "category" field.
_SYSTEM_METRIC_CATEGORIES = frozenset({"COST", "LOOP_COUNT_PERCENTILE", "TOOL_INVOCATION_RATE"})
_CATEGORY_KEYS = _SYSTEM_METRIC_CATEGORIES | {alias for spec in JUDGE_SPECS.values() for alias in spec.category_aliases}


def _metric_rows(value: object, *, category: str | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Flatten EvalCLI's map/list metric encodings while retaining category names."""
    value = _unwrap_additional_properties(value)
    rows: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, list):
        for item in value:
            rows.extend(_metric_rows(item, category=category))
        return rows
    if not isinstance(value, dict):
        return rows

    explicit_category = value.get("category")
    row_category = str(explicit_category).upper() if explicit_category else category
    if value.get("metric") is not None and row_category:
        rows.append((row_category, value))
    for key, child in value.items():
        if key in {"category", "metric"}:
            continue
        child_category = row_category
        if isinstance(child, dict | list) and str(key).upper() in _CATEGORY_KEYS:
            child_category = str(key).upper()
        rows.extend(_metric_rows(child, category=child_category))
    return rows


def _number(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return BH-adjusted p-values in the original order."""
    count = len(p_values)
    adjusted = [1.0] * count
    running = 1.0
    for rank, index in reversed(list(enumerate(sorted(range(count), key=p_values.__getitem__), start=1))):
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted


def _verify_customer_eval_metrics(
    metrics: dict[str, Any],
    *,
    gates: Mapping[str, float] | None = None,
) -> str:
    resolved_gates = dict(gates or {})
    system_rows = _metric_rows(metrics.get("systemMetrics"))
    judge_rows = _metric_rows(metrics.get("judgeMetrics"))
    guarded = [(category, row) for category, row in system_rows if category in _SYSTEM_METRIC_CATEGORIES]
    present_categories = {category for category, _row in guarded}
    missing_categories = _SYSTEM_METRIC_CATEGORIES - present_categories
    if missing_categories:
        raise SystemExit("Customer eval metrics omitted required comparisons: " + ", ".join(sorted(missing_categories)))

    p_values: list[float] = []
    for category, row in guarded:
        p_value = _number(row, "pValue", "p_value", "p-value")
        if p_value is None:
            raise SystemExit(f"Customer eval metric {category}/{row.get('metric')} has no p-value.")
        p_values.append(p_value)
    adjusted = _benjamini_hochberg(p_values)
    guardrail_lines: list[str] = []
    failures: list[str] = []
    for (category, row), raw_p, adjusted_p in zip(guarded, p_values, adjusted, strict=True):
        metric = str(row.get("metric"))
        base = _number(row, "base", "baseValue", "base_value")
        test = _number(row, "test", "testValue", "test_value")
        guardrail_lines.append(
            f"{category}/{metric}: base={base!s}, test={test!s}, p={raw_p:.4g}, p_bh={adjusted_p:.4g}"
        )
        if adjusted_p < CUSTOMER_EVAL_ALPHA:
            failures.append(f"{category}/{metric} differs significantly (BH p={adjusted_p:.4g})")

    judge_lines: list[str] = []
    for name, floor in resolved_gates.items():
        spec = JUDGE_SPECS[name]
        rows = [row for category, row in judge_rows if spec.matches_row(category, row)]
        if not rows:
            raise SystemExit(f"Customer eval metrics did not include {spec.metrics_label}.")
        score = _number(rows[0], *spec.score_keys)
        if score is None:
            raise SystemExit(f"Customer eval {spec.score_source} did not include an optimized-run score.")
        if failure := spec.failure_message(score, floor):
            failures.append(failure)
        judge_lines.append(spec.report_line(score, floor))

    status = "PASS" if not failures else "FAIL"
    report = "\n".join([f"status={status}", *judge_lines, *guardrail_lines])
    if failures:
        raise SystemExit(f"Customer eval validation failed: {'; '.join(failures)}\n{report}")
    return report


def _validate_best_candidate_on_customer_eval(
    *,
    runner: ALRunner,
    student_model: str,
    baseline_candidate: dict[str, str],
    best_candidate: dict[str, str],
    evalcli: EvalCliClient,
    valset: list[ALDataInst],
    experiment: ExperimentConfig | None = None,
) -> None:
    """Run paired customer evals on the GEPA valset and enforce metric gates."""
    if not valset:
        raise SystemExit("Customer eval requires a non-empty validation set.")
    if compile_encoded_prompt(baseline_candidate) == compile_encoded_prompt(best_candidate):
        log_section(
            "CUSTOMER EVAL",
            "Skipping customer eval: best candidate is the seed prompt, so a base/best pair would be identical.",
        )
        return
    gates = customer_validation_gates(experiment)
    for item in valset:
        version = item["eval_set_version"]
        deployment_ids = list(item["deployment_ids"])
        log_section(
            "CUSTOMER EVAL",
            "\n".join(
                [
                    f"eval_set={item['eval_set_name']}:{version}",
                    f"deployments={','.join(deployment_ids)}",
                ]
            ),
        )
        baseline_eval_id, baseline_wait = runner.start(
            student_model,
            system_prompt=compile_encoded_prompt(baseline_candidate),
            eval_set_name=item["eval_set_name"],
            eval_set_version=version,
            deployment_ids=deployment_ids,
            run_label="gepa_customer_base",
        )
        best_eval_id, best_wait = runner.start(
            student_model,
            system_prompt=compile_encoded_prompt(best_candidate),
            eval_set_name=item["eval_set_name"],
            eval_set_version=version,
            deployment_ids=deployment_ids,
            run_label="gepa_customer_best",
        )
        if baseline_wait:
            runner.wait(baseline_eval_id)
        if best_wait:
            runner.wait(best_eval_id)

        # Start every configured judge before waiting so slower agentic scoring overlaps.
        judge_run_details: list[str] = []
        wait_ids: list[str] = []
        started_types: set[str] = set()
        for name in gates:
            spec = JUDGE_SPECS[name]
            if spec.judge_type in started_types:
                continue
            started_types.add(spec.judge_type)
            judge_run_id = runner.ensure_judge_run(
                eval_run_id=best_eval_id,
                judge_type=spec.judge_type,
                run_params=spec.run_params,
                base_eval_run_id=baseline_eval_id if spec.kind == "pairwise" else None,
                input_mappings=spec.input_mappings or None,
                cortex_judge_type=spec.cortex_judge_type,
                judge_skill_name=spec.judge_skill_name,
            )
            wait_ids.append(judge_run_id)
            judge_run_details.append(f"{spec.name}_judge_run_id={judge_run_id}")
        for wait_id in wait_ids:
            evalcli.wait_for_judge_run(wait_id, eval_run_id=best_eval_id)
        metrics = evalcli.compare_eval_metrics(best_eval_id, baseline_eval_id)
        run_details = [
            f"eval_set_version={version}",
            f"baseline_eval_id={baseline_eval_id}",
            f"best_eval_id={best_eval_id}",
            *judge_run_details,
        ]
        try:
            report = _verify_customer_eval_metrics(metrics, gates=gates)
        except SystemExit as exc:
            log_section("CUSTOMER EVAL RESULT", "\n".join([*run_details, str(exc)]))
            raise
        log_section(
            "CUSTOMER EVAL RESULT",
            "\n".join([*run_details, report]),
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    # Renders defaults so `--config X --help` reports what that config supplies.
    parser = argparse.ArgumentParser(
        description="Optimize Glean prompts with GEPA's low-level engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Experiment YAML path or packaged name (teacher_student, single_model_shell). "
        "YAML supplies defaults; explicit CLI flags override it.",
    )
    parser.add_argument(
        "--seed_candidate",
        type=Path,
        help="Prompt-module overrides: a JSON object, or a .prompt file marked with {#KEY}...{/KEY}. "
        "Omitted keys (or no file) use the stock text in glean_gepa/prompts/.",
    )
    parser.add_argument("--max_metric_calls", type=int, default=10)
    parser.add_argument("--run_dir", type=Path, default=None)
    # Keep this short: argparse wraps help text, and a test asserts "(default: X)" stays on one line.
    model_help = (
        "gpt, fast, claude_sonnet, claude_opus, gpt6_luna, gpt6_sol, or gpt6_1_sol_high. "
        "waldo[:PROVIDER:MODEL[:effort]] runs the Waldo router instead."
    )
    parser.add_argument("--student_model", default="gpt", help=model_help)
    parser.add_argument("--teacher_model", default="gpt", help=model_help)
    # glean-dev only enables CUSTOM:* models; use OPEN_AI:GPT5_LATEST on a Glean-key instance.
    parser.add_argument("--reflection_lm_model", default="CUSTOM:GPT5_6_LUNA")
    # Reflection client only; evals stay on data.deployment_ids. scio-prod blocks /qe/llm
    # off-VPN, so point this at glean-dev. The secret derives from the GCP project id.
    parser.add_argument("--qe_project", default="dev-sandbox-334901")
    parser.add_argument("--qe_instance", default="glean-dev")
    parser.add_argument("--qe_authenticated_email", default="cathy.chen@glean.com")
    parser.add_argument("--global_token_cap", type=int, default=4096)
    parser.add_argument(
        "--reflection_samples",
        type=_parse_reflection_samples,
        default=8,
        help="Number of reflective examples per module, or 'all' for every available example.",
    )
    parser.add_argument(
        "--reflection_hamming_distance_k",
        type=_nonnegative_int,
        default=None,
        help="Drop later examples whose isolated execution errors are within Hamming distance k.",
    )
    parser.add_argument("--evalcli", default=None)
    parser.add_argument(
        "--agentspan_lookback_days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="UTC days of Agentspan shards to search when scoring an eval run "
        "(shell-tool errors or teacher-student tool match).",
    )
    parser.add_argument(
        "--eval_run_timeout_sec",
        type=_nonnegative_int,
        default=21600,
        help="Maximum seconds to wait for one Cortex eval run.",
    )
    parser.add_argument(
        "--eval_run_grace_period_sec",
        type=_nonnegative_int,
        default=None,
        help="Seconds to keep polling after a run is usable while entries are still unfinished. "
        "Omit to use the 1800s default. 0 returns as soon as the run is usable.",
    )
    parser.add_argument(
        "--cache_file",
        type=Path,
        default=None,
        help="Persistent adapter-analysis cache. Defaults to <run_dir>/cache/glean_adapter_cache.json.",
    )
    parser.add_argument(
        "--eval_run_cache_file",
        type=Path,
        default=None,
        help="Persistent Cortex eval-run ID cache. Defaults to <run_dir>/cache/glean_eval_run_cache.json.",
    )
    parser.add_argument(
        "--children_cache_file",
        type=Path,
        default=None,
        help="Persistent generated-child cache. Defaults to <run_dir>/cache/glean_children_cache.json.",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Append all terminal output plus reflection/child reports. "
        "Defaults to <run_dir>/gepa_run.log when --run_dir is set.",
    )
    parser.add_argument("--bigquery_project", default=None)
    parser.add_argument(
        "--train_eval_versions",
        help="Optional comma-separated override for training versions; set with --val_eval_versions.",
    )
    parser.add_argument(
        "--val_eval_versions",
        help="Optional comma-separated override for customer-deployment validation versions "
        "(same eval set as the final customer gate).",
    )
    parser.add_argument(
        "--val_eval_set_name",
        default=None,
        help="Eval set for validation. When versions are pinned, defaults to the training eval set; "
        "otherwise defaults to Glean Chat V2 Medium.",
    )
    parser.add_argument(
        "--eval_version_lookback_days",
        type=_nonnegative_int,
        default=14,
        help="Automatically select scio-prod versions from this many days ago through the as-of date.",
    )
    parser.add_argument(
        "--eval_version_days_back",
        type=_nonnegative_int,
        default=0,
        help="Shift the as-of date this many days into the past. Automatic selection otherwise tracks "
        "today, so it picks up a newly published version each day and misses the eval-run cache; "
        "setting this pins the window without hardcoding version numbers.",
    )
    parser.add_argument(
        "--val_eval_version_count",
        type=int,
        choices=[1, 2],
        default=2,
        help="Number of newest customer-deployment versions used for GEPA validation "
        "and the final customer gate (default: 2).",
    )
    parser.add_argument(
        "--customer_deployment_seed",
        type=int,
        default=None,
        help=f"Seed for sampling {MAX_EVAL_RUN_DEPLOYMENTS} customer deployments (Cortex caps an eval run at "
        f"{MAX_EVAL_RUN_DEPLOYMENTS}). Omit to sample randomly; the sample is saved under --run_dir and reused "
        "on resume so cached eval runs still hit.",
    )
    parser.add_argument(
        "--judging_mode",
        choices=["teacher_student", "single_model"],
        default="single_model",
    )
    parser.add_argument(
        "--editable_modules",
        default=WRITING_CODE_KEY,
        help=(
            "Comma-separated prompt keys to edit: any section declared under glean_gepa/prompts/ "
            "(WRITING_CODE, RULES_EXT, WALDO_ROUTING, ...), or a group such as CORE_TOOLS."
        ),
    )
    customer_eval_group = parser.add_mutually_exclusive_group()
    customer_eval_group.add_argument(
        "--customer_eval",
        dest="customer_eval",
        action="store_true",
        default=True,
        help=(
            "After the search, run the seed and best prompts on the external (customer) eval set and "
            "enforce objective.validation gates. On by default. YAML: run.customer_eval."
        ),
    )
    customer_eval_group.add_argument(
        "--no_customer_eval",
        dest="customer_eval",
        action="store_false",
        help="Skip the external eval; the search still runs and the best candidate is still written.",
    )
    parser.add_argument(
        "--fake_flow",
        action="store_true",
        help="Run deterministic fake eval data and prompt iterations without external Glean services.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show eval-set payloads and shell-tool action/error details.",
    )
    # Help-less, so `--config X --help` reaches the real parse instead of exiting here.
    config_scanner = argparse.ArgumentParser(add_help=False)
    config_scanner.add_argument("--config", default=None)
    pre, _unknown = config_scanner.parse_known_args(argv)
    experiment = None
    if pre.config:
        try:
            experiment = load_experiment_config(resolve_config_path(pre.config))
        except ExperimentConfigError as exc:
            raise SystemExit(str(exc)) from exc
        parser.set_defaults(**runner_arg_defaults(experiment))
    parser.set_defaults(experiment=experiment)
    args = parser.parse_args(argv)
    if experiment is not None and args.judging_mode != experiment.mode:
        raise SystemExit(
            f"--judging_mode {args.judging_mode} conflicts with {experiment.source_path} "
            f"(mode: {experiment.mode}); its signals are not scorable in {args.judging_mode} mode"
        )
    return args


def _format_run_config(
    args: argparse.Namespace,
    judging_mode: JudgingMode,
    editable_modules: list[str],
    seed_candidate: dict[str, str],
    experiment: ExperimentConfig | None,
) -> str:
    lines = [
        f"judging_mode={judging_mode}",
        f"student_model={args.student_model}",
        f"teacher_model={args.teacher_model}",
        f"reflection_lm={args.reflection_lm_model} via {args.qe_instance} project={args.qe_project}",
        f"editable_modules={','.join(editable_modules)}",
        f"seed_candidate={args.seed_candidate}",
        f"run_dir={args.run_dir}",
        f"max_metric_calls={args.max_metric_calls}",
        f"candidate_keys={','.join(sorted(seed_candidate))}",
    ]
    if experiment is not None:
        lines.extend(
            [
                f"config={experiment.source_path}",
                f"primary_objective={experiment.primary_objective}",
                f"frontier_type={experiment.frontier_type}",
                f"screening={experiment.screening.get('kind')} threshold={experiment.screening.get('threshold')}",
            ]
        )
    return "\n".join(lines)


def _build_adapter(
    args: argparse.Namespace,
    judging_mode: JudgingMode,
    adapter_kwargs: dict[str, Any],
    experiment: ExperimentConfig | None,
) -> TeacherStudentAdapter | SingleModelAdapter:
    kwargs = dict(adapter_kwargs)
    objective = build_objective(
        judging_mode,
        experiment.signals if experiment is not None else None,
        bigquery_client=kwargs.get("bigquery_client"),
        lookback_days=int(kwargs.get("agentspan_lookback_days") or 1),
        experiment=experiment_objective_spec(experiment) if experiment is not None else None,
    )
    kwargs["objective"] = objective
    if experiment is not None:
        if experiment.primary_objective:
            kwargs["primary_objective"] = experiment.primary_objective
        if experiment.frontier_type:
            kwargs["default_frontier_type"] = experiment.frontier_type
        kwargs["composite_weights"] = composite_weights(experiment)
        kwargs["constant_scores"] = constant_scores(experiment)
    if judging_mode == "teacher_student":
        if experiment is not None:
            kwargs["pointwise_judges"] = pointwise_judges(experiment)
            kwargs["pairwise_judges"] = pairwise_judges(experiment)
            kwargs["screening_kind"] = experiment.screening.get("kind")
            kwargs["screening_weights"] = screening_weights(experiment)
        return TeacherStudentAdapter(**kwargs, teacher_model=args.teacher_model)
    return SingleModelAdapter(**kwargs)


def _resolve_log_file(args: argparse.Namespace) -> Path | None:
    if args.log_file is not None:
        return args.log_file
    if args.run_dir is not None:
        return args.run_dir / RUN_LOG_FILENAME
    return None


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    set_debug(args.debug)
    log_file = _resolve_log_file(args)
    if log_file is None:
        _run_from_args(args)
        return
    with capture_run_log(log_file):
        print(f"[Run log] Writing all terminal output to {log_file}")
        _run_from_args(args)


def _run_from_args(args: argparse.Namespace) -> None:
    if args.fake_flow:
        _run_fake_flow(args)
        return

    experiment = args.experiment
    editable_modules = _parse_editable_modules(args.editable_modules)
    judging_mode = cast(JudgingMode, args.judging_mode)
    raw_seed = _load_seed_candidate(args.seed_candidate)
    seed_candidate = _seed_for_editable_modules(raw_seed, editable_modules)
    models = [args.student_model, *([args.teacher_model] if judging_mode == "teacher_student" else [])]
    harness = _eval_harness_for(experiment, editable_modules, models)
    log_section("RUN CONFIG", _format_run_config(args, judging_mode, editable_modules, seed_candidate, experiment))
    evalcli = EvalCliClient(binary=args.evalcli)
    bigquery_client = BigQueryClient(project_id=args.bigquery_project)
    deployment_pool = (
        TEACHER_STUDENT_DEPLOYMENT_IDS if judging_mode == "teacher_student" else CUSTOMER_EVAL_DEPLOYMENT_IDS
    )
    customer_deployments = _resolve_customer_deployments(
        _default_cache_file(args.run_dir, CUSTOMER_DEPLOYMENTS_FILENAME),
        seed=args.customer_deployment_seed,
        pool=deployment_pool,
    )
    print(
        f"[Customer eval] Sampled {len(customer_deployments)} of {len(deployment_pool)} "
        f"customer deployments allowed for {judging_mode}: {','.join(customer_deployments)}"
    )
    train_versions, val_versions = _resolve_eval_version_split(args, evalcli, customer_deployments)
    eval_set_name, deployment_ids = evalset_identity(experiment)
    trainset = _make_evalset(train_versions, eval_set_name=eval_set_name, deployment_ids=deployment_ids)
    # Pinned val versions run on data.deployment_ids against the training eval set
    # (or data.val_eval_set_name); otherwise validation is the customer eval set.
    if args.val_eval_versions:
        val_set_name = _val_eval_set_name(args, eval_set_name)
        val_deployment_ids = deployment_ids
    else:
        val_set_name = _val_eval_set_name(args, GLEAN_CHAT_EVAL_SET_NAME)
        val_deployment_ids = customer_deployments
    if val_set_name != eval_set_name:
        print(f"[Eval set schedule] train={eval_set_name}; validation={val_set_name}")
    valset = _make_evalset(
        val_versions,
        eval_set_name=val_set_name,
        deployment_ids=val_deployment_ids,
        validation_only=True,
    )
    cache_file = args.cache_file or _default_cache_file(args.run_dir, ADAPTER_CACHE_FILENAME)
    eval_run_cache_file = args.eval_run_cache_file or _default_cache_file(args.run_dir, EVAL_RUN_CACHE_FILENAME)
    children_cache_file = args.children_cache_file or _default_cache_file(args.run_dir, CHILDREN_CACHE_FILENAME)
    evalset_schedule_file = _default_cache_file(args.run_dir, EVALSET_SCHEDULE_FILENAME)
    al_runner = ALRunner(
        evalcli=evalcli,
        cache_file=str(eval_run_cache_file) if eval_run_cache_file else None,
        eval_run_timeout_sec=args.eval_run_timeout_sec,
        eval_run_grace_period_sec=args.eval_run_grace_period_sec,
        harness=harness,
    )
    adapter_kwargs = {
        "runner": al_runner,
        "student_model": args.student_model,
        "cache_file": str(cache_file) if cache_file else None,
        "bigquery_client": bigquery_client,
        "agentspan_lookback_days": args.agentspan_lookback_days,
        "editable_modules": editable_modules,
    }
    adapter = _build_adapter(args, judging_mode, adapter_kwargs, experiment)

    logger = StdOutLogger()
    tracker = create_experiment_tracker()
    spec_names = candidate_module_names(adapter.editable_modules)
    module_specs = {name: ModuleSpec(name, "free_text", module_token_budget(name) or 1024) for name in spec_names}
    global_token_cap = max(
        args.global_token_cap,
        *(module_token_budget(name) or 0 for name in adapter.editable_modules),
    )
    proposer_kwargs = {
        "logger": logger,
        "trainset": trainset,
        "al_adapter": adapter,
        "reflection_llm": _make_reflection_lm(
            args.reflection_lm_model,
            qe_project=args.qe_project,
            qe_instance=args.qe_instance,
            authenticated_email=args.qe_authenticated_email,
        ),
        "experiment_tracker": tracker,
        "model": args.student_model,
        "module_specs": module_specs,
        "global_token_cap": global_token_cap,
        "reflect_k": args.reflection_samples,
        "reflection_hamming_distance_k": args.reflection_hamming_distance_k,
        "baseline_prompt_hash": hashlib.md5(json.dumps(seed_candidate, sort_keys=True).encode()).hexdigest(),
        "children_cache_file": children_cache_file,
    }
    evalset_policy = UnseenEvalSetPolicy(state_file=evalset_schedule_file)
    if judging_mode == "teacher_student":
        _prestart_first_iteration_training(
            runner=al_runner,
            policy=evalset_policy,
            trainset=trainset,
            seed_candidate=seed_candidate,
            student_model=args.student_model,
            teacher_model=args.teacher_model,
            run_dir=args.run_dir,
        )
    proposer_kwargs["evalset_policy"] = evalset_policy
    if experiment is not None:
        offspring_count = experiment.search.get("offspring_count")
        if offspring_count is not None:
            proposer_kwargs["offspring_count"] = int(offspring_count)
        threshold = screening_threshold(experiment)
        if threshold is not None:
            proposer_kwargs["high_signal_screen_threshold"] = threshold
    proposer = EvolutionaryProposer(**proposer_kwargs)
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        proposer=proposer,
        logger=logger,
        experiment_tracker=tracker,
        max_metric_calls=args.max_metric_calls,
        run_dir=str(args.run_dir) if args.run_dir else None,
        frontier_type=cast(FrontierType, adapter.default_frontier_type),
    )
    best_candidate = result.best_candidate
    if not args.customer_eval:
        log_section(
            "CUSTOMER EVAL",
            "Skipped: customer_eval is off (run.customer_eval / --no_customer_eval). "
            "Best candidate was not validated against objective.validation gates.",
        )
        return
    if not isinstance(best_candidate, dict):
        raise SystemExit("Customer eval requires a dict prompt candidate")
    _validate_best_candidate_on_customer_eval(
        runner=al_runner,
        student_model=args.student_model,
        baseline_candidate=seed_candidate,
        best_candidate=best_candidate,
        evalcli=evalcli,
        valset=valset,
        experiment=args.experiment,
    )


def _run_fake_flow(args: argparse.Namespace) -> None:
    """Execute the real GEPA lifecycle with in-memory fake evaluations."""
    seed_candidate, trainset, valset, adapter, module_specs = build_fake_flow_components()
    print(
        "[FAKE FLOW] Starting offline Glean GEPA flow with separate train and val sets; no external services will be called."
    )
    logger = StdOutLogger()
    tracker = create_experiment_tracker()
    proposer = EvolutionaryProposer(
        logger=logger,
        trainset=trainset,
        al_adapter=adapter,
        reflection_llm=lambda _prompt: "fake reflection is supplied by FakeFlowAdapter",
        experiment_tracker=tracker,
        model="fake-model",
        module_specs=module_specs,
        global_token_cap=4096,
        reflect_k=3,
        baseline_prompt_hash=hashlib.md5(json.dumps(seed_candidate, sort_keys=True).encode()).hexdigest(),
        evalset_policy=UnseenEvalSetPolicy(state_file=_default_cache_file(args.run_dir, EVALSET_SCHEDULE_FILENAME)),
        children_cache_file=args.children_cache_file or _default_cache_file(args.run_dir, CHILDREN_CACHE_FILENAME),
    )
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,  # type: ignore[arg-type]
        proposer=proposer,
        logger=logger,
        experiment_tracker=tracker,
        max_metric_calls=args.max_metric_calls,
        run_dir=str(args.run_dir) if args.run_dir else None,
        frontier_type="objective",
    )
    print(
        f"[FAKE FLOW] Complete: iterations={result.num_candidates - 1}, "
        f"metric_calls={result.total_evals}, best_score={result.best_score:.2f}"
    )


if __name__ == "__main__":
    main()
