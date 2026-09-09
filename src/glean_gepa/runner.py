"""CLI and low-level GEPA engine wiring for Glean prompt optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

from gepa.core.state import FrontierType
from gepa.logging.experiment_tracker import create_experiment_tracker
from gepa.logging.logger import StdOutLogger
from glean_gepa.adapter_types import ALDataInst, JudgingMode
from glean_gepa.al_adapter import (
    ALRunner,
    ModuleSpec,
    Thresholds,
)
from glean_gepa.api import optimize
from glean_gepa.bigquery_client import BigQueryClient
from glean_gepa.debug import set_debug
from glean_gepa.evalcli_client import (
    CORRECTNESS_INPUT_MAPPINGS,
    CORRECTNESS_JUDGE_TYPE,
    CORRECTNESS_RUN_PARAMS,
    EvalCliClient,
)
from glean_gepa.evalset_policy import UnseenEvalSetPolicy
from glean_gepa.evolutionary_proposer import EvolutionaryProposer
from glean_gepa.experiment_config import (
    DEFAULT_DEPLOYMENT_IDS,
    DEFAULT_EVAL_SET_NAME,
    ExperimentConfig,
    ExperimentConfigError,
    composite_weights,
    constant_scores,
    evalset_identity,
    load_experiment_config,
    pointwise_judges,
    resolve_config_path,
    runner_arg_defaults,
    screening_threshold,
)
from glean_gepa.fake_flow import build_fake_flow_components
from glean_gepa.openai_client import create_qe_openai_client, format_exception_chain, get_perfeval_secret
from glean_gepa.prompt import candidate_module_names, compile_encoded_prompt, materialize_system_prompt
from glean_gepa.prompt_constants import (
    CORE_TOOLS,
    CORE_TOOLS_GROUP,
    FULL_PROMPT_KEY,
    KNOWN_PROMPT_KEYS,
    MODULE_TOKEN_BUDGETS,
    PROMPT_MODULE_DEFAULTS,
    WRITING_CODE_KEY,
)
from glean_gepa.run_log import capture_run_log, log_section
from glean_gepa.shell_tool_error_util import DEFAULT_LOOKBACK_DAYS
from glean_gepa.single_model_adapter import SingleModelAdapter
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter

CACHE_DIRECTORY_NAME = "cache"
ADAPTER_CACHE_FILENAME = "glean_adapter_cache.json"
EVAL_RUN_CACHE_FILENAME = "glean_eval_run_cache.json"
CHILDREN_CACHE_FILENAME = "glean_children_cache.json"
RUN_LOG_FILENAME = "gepa_run.log"
EVALSET_SCHEDULE_FILENAME = "glean_evalset_schedule.json"
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
CUSTOMER_EVAL_ALPHA = 0.05
CUSTOMER_CORRECTNESS_MIN = 0.80


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


def _load_seed_candidate(path: Path) -> dict[str, str]:
    """Load prompt-module overrides. Omitted keys use ``PROMPT_MODULE_DEFAULTS``."""
    if not path.is_file():
        raise SystemExit(f"seed_candidate file not found: {path}")
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise SystemExit("seed_candidate must be a JSON object")

    unknown = set(raw) - KNOWN_PROMPT_KEYS
    if unknown:
        unknown_list = ", ".join(sorted(repr(key) for key in unknown))
        raise SystemExit(f"seed_candidate has unknown keys: {unknown_list}")

    seed: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(value, str):
            raise SystemExit(f"{key} must be a string. Got type={type(value)}")
        seed[key] = value
    return seed


def _parse_editable_modules(raw: str) -> list[str]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise SystemExit("editable_modules must list at least one prompt key")
    modules: list[str] = []
    unknown: list[str] = []
    for part in parts:
        if part == CORE_TOOLS_GROUP:
            for key in CORE_TOOLS:
                if key not in modules:
                    modules.append(key)
        elif part in KNOWN_PROMPT_KEYS:
            if part not in modules:
                modules.append(part)
        else:
            unknown.append(part)
    if unknown:
        unknown_list = ", ".join(sorted(repr(key) for key in unknown))
        known_list = ", ".join(sorted([*KNOWN_PROMPT_KEYS, CORE_TOOLS_GROUP]))
        raise SystemExit(f"unknown editable_modules: {unknown_list}. Known keys: {known_list}")
    return modules


def _seed_for_editable_modules(raw: dict[str, str], editable_modules: list[str]) -> dict[str, str]:
    """Build the GEPA candidate dict for the requested editable modules.

    ``FULL_PROMPT`` is a fully stitched system prompt when that key is editable.
    When neither ``FULL_PROMPT`` nor ``WRITING_CODE`` is editable, the materialized
    seed prompt is still attached so evals keep a frozen system prompt. Core-tool
    descriptions are attached only when they are editable, so a ``WRITING_CODE``-only
    run keeps the legacy candidate and compiled-prompt hashes.
    ``RULES_EXT`` is copied from the seed when listed; compile time splices it into
    the ``{RULES_EXT}`` slot after Writing Code **Rules:**. Keys omitted from
    ``raw`` use ``PROMPT_MODULE_DEFAULTS``.
    """
    seed: dict[str, str] = {}
    for key in editable_modules:
        if key == FULL_PROMPT_KEY:
            seed[key] = materialize_system_prompt(raw)
        else:
            seed[key] = raw.get(key, PROMPT_MODULE_DEFAULTS[key])
    editing_system_prompt = any(key in {FULL_PROMPT_KEY, WRITING_CODE_KEY} for key in editable_modules)
    if not editing_system_prompt:
        seed[FULL_PROMPT_KEY] = materialize_system_prompt(raw)
    return seed


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


def _make_evalset(
    versions: list[str],
    *,
    eval_set_name: str = GLEAN_CHAT_EVAL_SET_NAME,
    deployment_ids: list[str] | None = None,
) -> list[ALDataInst]:
    ids = list(deployment_ids) if deployment_ids is not None else list(SCIO_PROD_DEPLOYMENT_IDS)
    return [
        {
            "eval_set_name": eval_set_name,
            "eval_set_version": version,
            "deployment_ids": ids,
            "status": "active",
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


def _latest_dated_eval_version(
    version_rows: list[dict[str, object]], *, required_deployment_ids: list[str] | None = None
) -> str:
    required = set(required_deployment_ids or [])
    eligible_rows = []
    for row in version_rows:
        available = row.get("availableDeploymentIds") or row.get("available_deployment_ids")
        if required and isinstance(available, list) and available and not required.issubset(map(str, available)):
            continue
        eligible_rows.append(row)
    dated = _dated_eval_versions(eligible_rows)
    if not dated:
        raise SystemExit(
            f"Need at least one dated {GLEAN_CHAT_EVAL_SET_NAME} version (YYYYMMDD) available for "
            f"customer deployments {','.join(CUSTOMER_EVAL_DEPLOYMENT_IDS)}."
        )
    return dated[-1][1]


def _select_recent_train_and_val_versions(
    version_rows: list[dict[str, object]], *, as_of: date, lookback_days: int, valset_size: int
) -> tuple[list[str], list[str]]:
    """Reserve the newest one or two versions for validation and schedule older ones for training."""
    earliest = as_of - timedelta(days=lookback_days)
    ordered_versions = [
        version for version_date, version in _dated_eval_versions(version_rows) if earliest <= version_date <= as_of
    ]
    if len(ordered_versions) < 2:
        raise SystemExit(
            f"Need at least two scio-prod eval versions dated {earliest.isoformat()} through {as_of.isoformat()}; "
            f"found {len(ordered_versions)}."
        )
    actual_valset_size = min(valset_size, len(ordered_versions) - 1)
    return ordered_versions[:-actual_valset_size], ordered_versions[-actual_valset_size:]


def _resolve_eval_version_split(args: argparse.Namespace, evalcli: EvalCliClient) -> tuple[list[str], list[str]]:
    eval_set_name, deployment_ids = evalset_identity(args.experiment)
    if bool(args.train_eval_versions) != bool(args.val_eval_versions):
        raise SystemExit("Set both --train_eval_versions and --val_eval_versions, or neither for automatic selection.")
    if args.train_eval_versions:
        train_versions = _parse_eval_versions(args.train_eval_versions, argument_name="--train_eval_versions")
        val_versions = _parse_eval_versions(args.val_eval_versions, argument_name="--val_eval_versions")
    else:
        rows = evalcli.list_eval_set_versions(eval_set_name=eval_set_name, deployment_ids=deployment_ids)
        days_back = getattr(args, "eval_version_days_back", 0) or 0
        as_of = date.today() - timedelta(days=days_back)
        train_versions, val_versions = _select_recent_train_and_val_versions(
            rows,
            as_of=as_of,
            lookback_days=args.eval_version_lookback_days,
            valset_size=args.val_eval_version_count,
        )
        as_of_note = f" as of {as_of.isoformat()} ({days_back}d back)" if days_back else ""
        print(
            f"[Eval set schedule] Auto-selected{as_of_note} "
            f"train versions={','.join(train_versions)} and val versions={','.join(val_versions)}"
        )

    overlapping_versions = set(train_versions) & set(val_versions)
    if overlapping_versions:
        overlap = ", ".join(sorted(overlapping_versions))
        raise SystemExit(f"Train and validation eval versions must not overlap: {overlap}")
    if not 1 <= len(val_versions) <= 2:
        raise SystemExit("Validation must contain one or two eval versions.")
    return train_versions, val_versions


def _unwrap_additional_properties(value: object) -> object:
    while isinstance(value, dict) and set(value) == {"additional_properties"}:
        value = value["additional_properties"]
    return value


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
        if isinstance(child, dict | list) and str(key).upper() in {
            "COST",
            "LOOP_COUNT_PERCENTILE",
            "TOOL_INVOCATION_RATE",
            CORRECTNESS_JUDGE_TYPE,
        }:
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


def _verify_customer_eval_metrics(metrics: dict[str, Any]) -> str:
    system_rows = _metric_rows(metrics.get("systemMetrics"))
    judge_rows = _metric_rows(metrics.get("judgeMetrics"))
    guarded = [
        (category, row)
        for category, row in system_rows
        if category in {"COST", "LOOP_COUNT_PERCENTILE", "TOOL_INVOCATION_RATE"}
    ]
    present_categories = {category for category, _row in guarded}
    missing_categories = {"COST", "LOOP_COUNT_PERCENTILE", "TOOL_INVOCATION_RATE"} - present_categories
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

    correctness_rows = [
        row
        for category, row in judge_rows
        if category == CORRECTNESS_JUDGE_TYPE or str(row.get("metric", "")).upper() == CORRECTNESS_JUDGE_TYPE
    ]
    if not correctness_rows:
        raise SystemExit("Customer eval metrics did not include CORRECTNESS.")
    correctness = _number(correctness_rows[0], "test", "passRate", "pass_rate", "testValue", "test_value")
    if correctness is None:
        raise SystemExit("Customer eval CORRECTNESS did not include an optimized-run score.")
    if correctness <= CUSTOMER_CORRECTNESS_MIN:
        failures.append(f"correctness {correctness:.2%} is not above {CUSTOMER_CORRECTNESS_MIN:.0%}")

    status = "PASS" if not failures else "FAIL"
    report = "\n".join(
        [
            f"status={status}",
            f"correctness={correctness:.2%} (required >{CUSTOMER_CORRECTNESS_MIN:.0%})",
            *guardrail_lines,
        ]
    )
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
) -> None:
    """Run paired customer evals and enforce correctness and system-metric gates."""
    rows = evalcli.list_eval_set_versions(
        eval_set_name=GLEAN_CHAT_EVAL_SET_NAME,
        deployment_ids=list(CUSTOMER_EVAL_DEPLOYMENT_IDS),
    )
    version = _latest_dated_eval_version(rows, required_deployment_ids=list(CUSTOMER_EVAL_DEPLOYMENT_IDS))
    customers = ",".join(CUSTOMER_EVAL_DEPLOYMENT_IDS)
    log_section(
        "CUSTOMER EVAL",
        "\n".join(
            [
                f"eval_set={GLEAN_CHAT_EVAL_SET_NAME}:{version}",
                f"deployments={customers}",
            ]
        ),
    )
    baseline_eval_id, baseline_wait = runner.start(
        student_model,
        system_prompt=compile_encoded_prompt(baseline_candidate),
        eval_set_name=GLEAN_CHAT_EVAL_SET_NAME,
        eval_set_version=version,
        deployment_ids=list(CUSTOMER_EVAL_DEPLOYMENT_IDS),
        run_label="gepa_customer_base",
    )
    best_eval_id, best_wait = runner.start(
        student_model,
        system_prompt=compile_encoded_prompt(best_candidate),
        eval_set_name=GLEAN_CHAT_EVAL_SET_NAME,
        eval_set_version=version,
        deployment_ids=list(CUSTOMER_EVAL_DEPLOYMENT_IDS),
        run_label="gepa_customer_best",
    )
    if baseline_wait:
        runner.wait(baseline_eval_id)
    if best_wait:
        runner.wait(best_eval_id)

    judge_run_id = runner.ensure_judge_run(
        eval_run_id=best_eval_id,
        judge_type=CORRECTNESS_JUDGE_TYPE,
        run_params=CORRECTNESS_RUN_PARAMS,
        base_eval_run_id=baseline_eval_id,
        input_mappings=CORRECTNESS_INPUT_MAPPINGS,
    )
    evalcli.wait_for_judge_run(judge_run_id, eval_run_id=best_eval_id)
    metrics = evalcli.compare_eval_metrics(best_eval_id, baseline_eval_id)
    run_details = [
        f"baseline_eval_id={baseline_eval_id}",
        f"best_eval_id={best_eval_id}",
        f"correctness_judge_run_id={judge_run_id}",
    ]
    try:
        report = _verify_customer_eval_metrics(metrics)
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
        help="Experiment YAML path or packaged name (teacher_student, single_model). "
        "YAML supplies defaults; explicit CLI flags override it.",
    )
    parser.add_argument(
        "--seed_candidate",
        type=Path,
        help="JSON object of prompt-module overrides. Omitted keys use the Python defaults "
        "in glean_gepa.prompt_constants (WRITING_CODE, FULL_PROMPT, RULES_EXT, core-tool descriptions).",
    )
    parser.add_argument("--max_metric_calls", type=int, default=10)
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument(
        "--student_model",
        default="gpt",
        help="gpt, fast, claude_sonnet (4.6 coding harness), or claude_opus (default: gpt).",
    )
    parser.add_argument(
        "--teacher_model",
        default="gpt",
        help="gpt, fast, claude_sonnet (4.6 coding harness), or claude_opus (default: gpt).",
    )
    parser.add_argument("--reflection_lm_model", default="OPEN_AI:GPT5_LATEST")
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
        help="Maximum time to wait for one Cortex eval run (default: 6 hours).",
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
        help="Optional comma-separated override for held-out validation versions.",
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
        help="Number of newest eligible versions reserved for validation (default: 2).",
    )
    parser.add_argument(
        "--judging_mode",
        choices=["teacher_student", "single_model"],
        default="single_model",
    )
    parser.add_argument(
        "--editable_modules",
        default=WRITING_CODE_KEY,
        help="Comma-separated prompt keys to edit (WRITING_CODE, FULL_PROMPT, CORE_TOOLS, "
        "RULES_EXT, or individual core-tool keys). CORE_TOOLS expands to all core-tool "
        "descriptions; the proposer only rewrites those involved in high-signal first-tool "
        "mismatches. RULES_EXT is at most two bullets after Writing Code **Rules:**.",
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
            f"(mode: {experiment.mode}); its {','.join(experiment.packs)} pack is not scorable in "
            f"{args.judging_mode} mode"
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
                f"packs={','.join(experiment.packs) or '(none)'}",
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
    # load_experiment_config pins each mode to the one pack and primary objective
    # its adapter can score, so no mode/objective compatibility check is needed here.
    kwargs = dict(adapter_kwargs)
    if experiment is not None:
        if experiment.primary_objective:
            kwargs["primary_objective"] = experiment.primary_objective
        if experiment.frontier_type:
            kwargs["default_frontier_type"] = experiment.frontier_type
        kwargs["composite_weights"] = composite_weights(experiment)
        kwargs["constant_scores"] = constant_scores(experiment)
    if judging_mode == "teacher_student":
        if experiment is not None:
            # Only teacher_student runs judges; the loader rejects a weighted judge
            # signal in single_model mode.
            kwargs["pointwise_judges"] = pointwise_judges(experiment)
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
    if args.seed_candidate is None:
        raise SystemExit("--seed_candidate is required unless --fake_flow or --config with run.seed_candidate is set")
    editable_modules = _parse_editable_modules(args.editable_modules)
    judging_mode = cast(JudgingMode, args.judging_mode)
    raw_seed = _load_seed_candidate(args.seed_candidate)
    seed_candidate = _seed_for_editable_modules(raw_seed, editable_modules)
    log_section("RUN CONFIG", _format_run_config(args, judging_mode, editable_modules, seed_candidate, experiment))
    evalcli = EvalCliClient(binary=args.evalcli)
    train_versions, val_versions = _resolve_eval_version_split(args, evalcli)
    eval_set_name, deployment_ids = evalset_identity(experiment)
    trainset = _make_evalset(train_versions, eval_set_name=eval_set_name, deployment_ids=deployment_ids)
    valset = _make_evalset(val_versions, eval_set_name=eval_set_name, deployment_ids=deployment_ids)
    cache_file = args.cache_file or _default_cache_file(args.run_dir, ADAPTER_CACHE_FILENAME)
    eval_run_cache_file = args.eval_run_cache_file or _default_cache_file(args.run_dir, EVAL_RUN_CACHE_FILENAME)
    children_cache_file = args.children_cache_file or _default_cache_file(args.run_dir, CHILDREN_CACHE_FILENAME)
    evalset_schedule_file = _default_cache_file(args.run_dir, EVALSET_SCHEDULE_FILENAME)
    al_runner = ALRunner(
        evalcli=evalcli,
        cache_file=str(eval_run_cache_file) if eval_run_cache_file else None,
        eval_run_timeout_sec=args.eval_run_timeout_sec,
    )
    adapter_kwargs = {
        "runner": al_runner,
        "thresholds": Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        "student_model": args.student_model,
        "cache_file": str(cache_file) if cache_file else None,
        "bigquery_client": BigQueryClient(project_id=args.bigquery_project),
        "agentspan_lookback_days": args.agentspan_lookback_days,
        "editable_modules": editable_modules,
    }
    adapter = _build_adapter(args, judging_mode, adapter_kwargs, experiment)

    logger = StdOutLogger()
    tracker = create_experiment_tracker()
    spec_names = candidate_module_names(adapter.editable_modules)
    module_specs = {name: ModuleSpec(name, "free_text", MODULE_TOKEN_BUDGETS.get(name, 1024)) for name in spec_names}
    global_token_cap = max(
        args.global_token_cap,
        *(MODULE_TOKEN_BUDGETS.get(name, 0) for name in adapter.editable_modules),
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
        "evalset_policy": UnseenEvalSetPolicy(state_file=evalset_schedule_file),
        "children_cache_file": children_cache_file,
    }
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
    if not isinstance(best_candidate, dict):
        raise SystemExit("Customer eval requires a dict prompt candidate")
    _validate_best_candidate_on_customer_eval(
        runner=al_runner,
        student_model=args.student_model,
        baseline_candidate=seed_candidate,
        best_candidate=best_candidate,
        evalcli=evalcli,
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
