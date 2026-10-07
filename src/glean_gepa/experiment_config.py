"""Load a self-contained Glean GEPA experiment YAML."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from glean_gepa.adapter_types import EvalHarness, JudgingMode, PairwiseJudge, PointwiseJudge
from glean_gepa.focused_evalset import FOCUSED_BUCKET_TYPES
from glean_gepa.judge_metrics_util import CUSTOMER_AGENTIC_PREFERENCE_METRIC, JUDGE_SPEC_NAMES, JUDGE_SPECS
from glean_gepa.objectives.registry import is_known_source, is_registered

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_EVAL_SET_NAME = "Glean Chat V2 Medium"
DEFAULT_DEPLOYMENT_IDS = ("scio-prod",)

# Mode is the eval topology. Telemetry sources are registered per mode in
# glean_gepa.objectives.registry; a signal is scorable when its source is registered for the mode.
SUPPORTED_MODES: tuple[JudgingMode, ...] = ("single_model", "teacher_student")

# Sources whose signal value is read straight from telemetry or the config, as
# opposed to a judge run that has to be started and awaited.
_CONSTANT_SOURCE = "constant"

# Only teacher_student has the judge plumbing to start, await, and cache a judge run.
_MODES_WITH_CORTEX_JUDGES = frozenset({"teacher_student"})

_JUDGE_PARAM_KEYS = {
    "llm_model": "Llm model",
    "use_cache": "Use Cache",
    "judge_type": "Judge Type",
}


class ExperimentConfigError(ValueError):
    """Raised when an experiment YAML cannot be loaded or merged."""


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    mode: JudgingMode
    source_path: Path
    run: dict[str, Any]
    models: dict[str, Any]
    data: dict[str, Any]
    #: Eval-run creation overrides: ``runner_type``, ``sc_params``.
    eval: dict[str, Any]
    signals: tuple[dict[str, Any], ...]
    objective: dict[str, Any]
    screening: dict[str, Any]
    reflection: dict[str, Any]
    search: dict[str, Any]

    @property
    def primary_objective(self) -> str | None:
        primary = self.objective.get("primary")
        return str(primary) if primary else None

    @property
    def frontier_type(self) -> str | None:
        frontier = self.objective.get("frontier_type")
        return str(frontier) if frontier else None


def resolve_config_path(value: str | Path) -> Path:
    """Resolve a filesystem path or a packaged config stem such as ``teacher_student``."""
    raw = Path(value)
    if raw.is_file():
        return raw.resolve()
    name = raw.name
    if not name.endswith((".yaml", ".yml")):
        name = f"{name}.yaml"
    packaged = CONFIGS_DIR / name
    if packaged.is_file():
        return packaged
    raise ExperimentConfigError(f"experiment config not found: {value}")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate one experiment YAML. Every section is read from this file."""
    source_path = resolve_config_path(path)
    raw = _load_yaml(source_path)
    if not isinstance(raw, dict):
        raise ExperimentConfigError(f"{source_path} must be a mapping")
    schema_version = _require_int(raw.get("schema_version"), field="schema_version", default=1)
    if schema_version != 1:
        raise ExperimentConfigError(f"unsupported schema_version {schema_version}")
    mode = raw.get("mode")
    if mode not in SUPPORTED_MODES:
        raise ExperimentConfigError(f"mode must be one of {', '.join(sorted(SUPPORTED_MODES))}, got {mode!r}")
    if "packs" in raw:
        raise ExperimentConfigError(
            f"{source_path} sets `packs`, which is no longer supported; inline the signals, objective, "
            "screening, and reflection sections into this file"
        )
    merged_signals = _parse_signals(raw.get("signals"), mode=mode)
    merged_objective = dict(raw.get("objective") or {})
    merged_screening = dict(raw.get("screening") or {})
    merged_reflection = dict(raw.get("reflection") or {})
    _require_mode_primary_objective(merged_objective.get("primary"), merged_signals, mode=mode)
    _require_screening_weights(merged_screening, merged_signals, mode=mode)
    _require_scorable_composite_signals(merged_objective, merged_signals, mode=mode)
    _require_normalized_composite_weights(merged_objective.get("composite"))
    _require_unit_valued_weighted_constants(merged_objective, merged_signals)
    _require_focused_bucket_type(merged_objective.get("focused_bucket_type"))
    _parse_customer_validation(merged_objective.get("validation"))
    return ExperimentConfig(
        schema_version=schema_version,
        mode=mode,
        source_path=source_path,
        run=dict(raw.get("run") or {}),
        models=dict(raw.get("models") or {}),
        data=dict(raw.get("data") or {}),
        eval=_parse_eval_section(raw.get("eval")),
        signals=tuple(merged_signals),
        objective=merged_objective,
        screening=merged_screening,
        reflection=merged_reflection,
        search=dict(raw.get("search") or {}),
    )


_EVAL_KEYS = frozenset({"runner_type", "sc_params"})


def _parse_eval_section(raw: Any) -> dict[str, Any]:
    """Validate ``eval:``. ``sc_params`` may be a string or a list of ``key=value``
    strings (joined with commas). Values are kept verbatim: nested scParams carry
    their own percent-encoding."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ExperimentConfigError("eval must be a mapping")
    unknown = set(raw) - _EVAL_KEYS
    if unknown:
        raise ExperimentConfigError(f"eval has unknown keys: {', '.join(sorted(unknown))}")
    out: dict[str, Any] = {}
    if raw.get("runner_type") is not None:
        out["runner_type"] = str(raw["runner_type"])
    for key in ("sc_params",):
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            joined = value.strip()
        elif isinstance(value, list):
            parts = [str(part).strip() for part in value if str(part).strip()]
            for part in parts:
                if "=" not in part:
                    raise ExperimentConfigError(f"eval.{key} entries must be key=value, got {part!r}")
            joined = ",".join(parts)
        else:
            raise ExperimentConfigError(f"eval.{key} must be a string or a list of key=value strings")
        if joined:
            out[key] = joined
    return out


def eval_harness(config: ExperimentConfig | None) -> EvalHarness:
    """The ``eval:`` section as an :class:`EvalHarness`; defaults when absent."""
    if config is None:
        return EvalHarness()
    section = config.eval
    return EvalHarness(
        runner_type=section.get("runner_type"),
        sc_params=section.get("sc_params"),
    )


def runner_arg_defaults(config: ExperimentConfig) -> dict[str, Any]:
    """Argparse defaults for fields the runner already understands. CLI flags override these."""
    defaults: dict[str, Any] = {"judging_mode": config.mode}
    run = config.run
    if run.get("dir"):
        defaults["run_dir"] = Path(str(run["dir"]))
    if run.get("max_metric_calls") is not None:
        defaults["max_metric_calls"] = int(run["max_metric_calls"])
    if run.get("eval_run_timeout_sec") is not None:
        defaults["eval_run_timeout_sec"] = int(run["eval_run_timeout_sec"])
    if run.get("eval_run_grace_period_sec") is not None:
        defaults["eval_run_grace_period_sec"] = int(run["eval_run_grace_period_sec"])
    if run.get("seed_candidate"):
        defaults["seed_candidate"] = Path(str(run["seed_candidate"]))
    if run.get("customer_eval") is not None:
        customer_eval = run["customer_eval"]
        if not isinstance(customer_eval, bool):
            raise ExperimentConfigError(f"run.customer_eval must be true or false, got {customer_eval!r}")
        defaults["customer_eval"] = customer_eval
    modules = run.get("editable_modules")
    if modules is None:
        modules = config.reflection.get("editable_modules")
    if modules is not None:
        defaults["editable_modules"] = (
            ",".join(str(module) for module in modules) if isinstance(modules, list) else str(modules)
        )
    models = config.models
    if models.get("student"):
        defaults["student_model"] = str(models["student"])
    if models.get("teacher"):
        defaults["teacher_model"] = str(models["teacher"])
    if models.get("reflection"):
        defaults["reflection_lm_model"] = str(models["reflection"])
    data = config.data
    if data.get("train_eval_versions"):
        defaults["train_eval_versions"] = ",".join(str(version) for version in data["train_eval_versions"])
    if data.get("val_eval_versions"):
        defaults["val_eval_versions"] = ",".join(str(version) for version in data["val_eval_versions"])
    if data.get("lookback_days") is not None:
        defaults["eval_version_lookback_days"] = int(data["lookback_days"])
    if data.get("days_back") is not None:
        defaults["eval_version_days_back"] = int(data["days_back"])
    if data.get("val_version_count") is not None:
        defaults["val_eval_version_count"] = int(data["val_version_count"])
    if data.get("val_eval_set_name"):
        defaults["val_eval_set_name"] = str(data["val_eval_set_name"])
    search = config.search
    if "reflection_samples" in search:
        samples = search["reflection_samples"]
        defaults["reflection_samples"] = None if samples in (None, "all") else int(samples)
    if "reflection_hamming_distance_k" in search:
        hamming = search["reflection_hamming_distance_k"]
        defaults["reflection_hamming_distance_k"] = None if hamming is None else int(hamming)
    if search.get("global_token_cap") is not None:
        defaults["global_token_cap"] = int(search["global_token_cap"])
    lookback = agentspan_lookback_days(config)
    if lookback is not None:
        defaults["agentspan_lookback_days"] = lookback
    return defaults


def evalset_identity(config: ExperimentConfig | None) -> tuple[str, list[str]]:
    if config is None:
        return DEFAULT_EVAL_SET_NAME, list(DEFAULT_DEPLOYMENT_IDS)
    name = str(config.data.get("eval_set_name") or DEFAULT_EVAL_SET_NAME)
    raw_ids = config.data.get("deployment_ids") or list(DEFAULT_DEPLOYMENT_IDS)
    return name, [str(item) for item in raw_ids]


def pointwise_judges(config: ExperimentConfig) -> tuple[PointwiseJudge, ...]:
    judges = tuple(
        PointwiseJudge(str(signal["name"]), str(signal["type"]), judge_run_params_json(signal))
        for signal in _enabled_cortex_judges(config.signals, "pointwise")
    )
    return _with_required_judges(judges, config, kind="pointwise", build=_pointwise_from_spec)


def pairwise_judges(config: ExperimentConfig) -> tuple[PairwiseJudge, ...]:
    judges = tuple(_pairwise_from_signal(signal) for signal in _enabled_cortex_judges(config.signals, "pairwise"))
    return _with_required_judges(judges, config, kind="pairwise", build=_pairwise_from_spec)


def _pairwise_from_signal(signal: Mapping[str, Any]) -> PairwiseJudge:
    """A YAML ``cortex_judge`` signal whose ``type`` is a JudgeSpec name uses that spec's Cortex wiring.

    ``type: AGENTIC_CORRECTNESS_JUDGE`` is an adapter key, not a Cortex type; the spec
    supplies the real Cortex type and skill name. Other types pass through as-is.
    """
    judge_type = str(signal["type"])
    spec = next((s for s in JUDGE_SPECS.values() if s.kind == "pairwise" and s.judge_type == judge_type), None)
    # Only adapter-only types inherit the spec's run_params; Cortex types keep the YAML's.
    adapter_only = spec is not None and spec.cortex_type_override is not None
    if signal.get("run_params") or not adapter_only:
        run_params = judge_run_params_json(signal)
    else:
        run_params = spec.run_params if spec else "{}"
    return PairwiseJudge(
        str(signal["name"]),
        judge_type,
        run_params,
        _pairwise_input_mappings(signal),
        cortex_type_override=spec.cortex_type_override if spec else None,
        judge_skill_name=spec.judge_skill_name if spec else None,
    )


def _pointwise_from_spec(spec: Any) -> PointwiseJudge:
    return PointwiseJudge(spec.name, spec.judge_type, spec.run_params)


def _pairwise_from_spec(spec: Any) -> PairwiseJudge:
    return PairwiseJudge(
        spec.name,
        spec.judge_type,
        spec.run_params,
        spec.input_mappings,
        cortex_type_override=spec.cortex_type_override,
        judge_skill_name=spec.judge_skill_name,
    )


def _required_judge_names(config: ExperimentConfig, *, kind: str) -> set[str]:
    """Judge specs the run must start even when no signal declared them.

    An ``agentic_preference_rate`` primary does this for ``AGENTIC_JUDGE``. A screening
    weight does it for whichever judge the gate reads.
    """
    names = {name for name in screening_weights(config) if name in JUDGE_SPECS}
    if kind == "pairwise" and config.primary_objective == CUSTOMER_AGENTIC_PREFERENCE_METRIC:
        names.add(CUSTOMER_AGENTIC_PREFERENCE_METRIC)
    return {name for name in names if JUDGE_SPECS[name].kind == kind}


def _with_required_judges(
    judges: tuple[Any, ...], config: ExperimentConfig, *, kind: str, build: Any
) -> tuple[Any, ...]:
    present = {judge.name for judge in judges}
    extra = tuple(
        build(JUDGE_SPECS[name]) for name in sorted(_required_judge_names(config, kind=kind)) if name not in present
    )
    return (*judges, *extra)


def composite_weights(config: ExperimentConfig) -> dict[str, float]:
    raw = config.objective.get("composite") or {}
    return {str(name): float(weight) for name, weight in raw.items()}


def constant_scores(config: ExperimentConfig) -> dict[str, float]:
    return {
        str(signal["name"]): float(signal.get("value", 0.0))
        for signal in config.signals
        if signal.get("source") == _CONSTANT_SOURCE and signal.get("name")
    }


def screening_threshold(config: ExperimentConfig) -> float | None:
    if config.screening.get("threshold") is None:
        return None
    return float(config.screening["threshold"])


def screening_weights(config: ExperimentConfig) -> dict[str, float]:
    """Child-gate blend. Empty means the gate is ``summary[primary]``."""
    raw = config.screening.get("weights") or {}
    return {str(name): float(weight) for name, weight in raw.items()}


def experiment_objective_spec(config: ExperimentConfig) -> dict[str, Any]:
    """Slice of the experiment that ``build_objective`` applies to the metric."""
    return {
        "objective": config.objective,
        "reflection": config.reflection,
        "screening": config.screening,
        "signals": list(config.signals),
    }


def customer_validation_gates(config: ExperimentConfig | None) -> dict[str, float]:
    """Judge floors the post-search customer eval enforces.

    Omitted ``objective.validation`` (or no ``--config``) starts no judge
    gates. List a metric to opt in; omit ``min`` to use that metric's default
    floor. Unknown metrics and unreadable mins fail the load.
    """
    if config is None:
        return {}
    return _parse_customer_validation(config.objective.get("validation"))


def _parse_customer_validation(raw: Any) -> dict[str, float]:
    if not isinstance(raw, list):
        return {}
    gates: dict[str, float] = {}
    for entry in raw:
        if not isinstance(entry, Mapping) or "metric" not in entry:
            raise ExperimentConfigError("objective.validation entries must be mappings with a metric")
        metric = str(entry["metric"])
        spec = JUDGE_SPECS.get(metric)
        if spec is None:
            allowed = ", ".join(sorted(JUDGE_SPEC_NAMES))
            raise ExperimentConfigError(f"objective.validation metric must be one of {allowed}, got {metric!r}")
        if "min" not in entry:
            gates[spec.name] = spec.default_min
            continue
        raw_min = entry["min"]
        if isinstance(raw_min, bool) or not isinstance(raw_min, int | float):
            raise ExperimentConfigError(f"objective.validation min for {metric} must be a number, got {raw_min!r}")
        floor = float(raw_min)
        if not 0.0 <= floor <= 1.0:
            raise ExperimentConfigError(f"objective.validation min for {metric} must be between 0 and 1, got {floor}")
        gates[spec.name] = floor
    return gates


def agentspan_lookback_days(config: ExperimentConfig) -> int | None:
    lookbacks = [
        int(signal["lookback_days"])
        for signal in config.signals
        if is_known_source(str(signal.get("source") or "")) and signal.get("lookback_days") is not None
    ]
    return max(lookbacks) if lookbacks else None


def judge_run_params_json(signal: Mapping[str, Any]) -> str:
    raw = signal.get("run_params") or {}
    if not isinstance(raw, Mapping):
        raise ExperimentConfigError("run_params must be a mapping")
    mapped: dict[str, str] = {}
    for key, value in raw.items():
        cortex_key = _JUDGE_PARAM_KEYS.get(str(key), str(key))
        if isinstance(value, bool):
            mapped[cortex_key] = "true" if value else "false"
        else:
            mapped[cortex_key] = str(value)
    return json.dumps(mapped)


def _require_int(raw: Any, *, field: str, default: int) -> int:
    """Coerce to int as ExperimentConfigError. Only None defaults, so 0 stays 0."""
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ExperimentConfigError(f"{field} must be an integer, got {raw!r}") from exc


def _parse_signals(raw: Any, *, mode: JudgingMode) -> list[dict[str, Any]]:
    """Validate ``signals``: unique names, and telemetry sources registered for ``mode``."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ExperimentConfigError(f"signals must be a list, got {raw!r}")
    signals: list[dict[str, Any]] = []
    seen: set[str] = set()
    for signal in raw:
        if not isinstance(signal, dict) or "name" not in signal:
            raise ExperimentConfigError("each signal must be a mapping with a name")
        name = str(signal["name"])
        if name in seen:
            raise ExperimentConfigError(f"signal {name!r} is declared more than once")
        seen.add(name)
        source = signal.get("source")
        if source not in {_CONSTANT_SOURCE, "cortex_judge", None} and not is_registered(mode, str(source)):
            raise ExperimentConfigError(
                f"mode {mode} cannot score signal {name!r} (source {source!r} is not registered for this mode)"
            )
        signals.append(dict(signal))
    return signals


def _enabled_cortex_judges(signals: Sequence[Mapping[str, Any]], kind: str) -> list[Mapping[str, Any]]:
    enabled: list[Mapping[str, Any]] = []
    for signal in signals:
        if signal.get("source") != "cortex_judge" or signal.get("kind") != kind:
            continue
        # Before the enabled check, so flipping enabled on cannot surface a new error.
        if not signal.get("type"):
            raise ExperimentConfigError(f"{kind} cortex_judge signal {signal.get('name')!r} requires type")
        if signal.get("enabled", True) is False:
            continue
        enabled.append(signal)
    return enabled


def _pairwise_input_mappings(signal: Mapping[str, Any]) -> str:
    raw = signal.get("input_mappings")
    if raw is None:
        judge_type = str(signal.get("type") or "")
        matches = [spec for spec in JUDGE_SPECS.values() if spec.kind == "pairwise" and spec.judge_type == judge_type]
        if len(matches) != 1:
            raise ExperimentConfigError(f"pairwise cortex_judge signal {signal.get('name')!r} requires input_mappings")
        return matches[0].input_mappings
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list | Mapping):
        return json.dumps(raw)
    raise ExperimentConfigError(
        f"pairwise cortex_judge signal {signal.get('name')!r} input_mappings must be a mapping or list"
    )


def _scorable_signal_names(signals: list[dict[str, Any]], *, mode: JudgingMode) -> set[str]:
    """Signals the mode's adapter can turn into a composite dimension.

    Everything else -- disabled signals, judges in a mode with no judge plumbing --
    is collected for reporting or future use but never produces a per-entry score.
    """
    names: set[str] = set()
    if mode in _MODES_WITH_CORTEX_JUDGES:
        for kind in ("pointwise", "pairwise"):
            names.update(str(signal["name"]) for signal in _enabled_cortex_judges(signals, kind))
    for signal in signals:
        if signal.get("enabled", True) is False:
            continue
        if signal.get("source") == _CONSTANT_SOURCE or is_registered(mode, signal.get("source")):
            names.add(str(signal["name"]))
    return names


def _require_scorable_composite_signals(
    objective: Mapping[str, Any], signals: list[dict[str, Any]], *, mode: JudgingMode
) -> None:
    """Reject composite weights that no adapter can turn into a score.

    A weight naming an unknown or unscorable signal is silently dropped at scoring
    time, so the run would report a composite the objective never actually applied.
    """
    composite = objective.get("composite")
    if composite is None:
        return
    if not isinstance(composite, Mapping):
        raise ExperimentConfigError(
            f"objective.composite must be a mapping of signal name to weight, got {composite!r}"
        )
    scorable = _scorable_signal_names(signals, mode=mode)
    declared = {str(signal["name"]) for signal in signals}
    unscorable = sorted(str(name) for name in composite if str(name) not in scorable)
    if unscorable:
        detail = ", ".join(
            f"{name} (declared but not scorable)" if name in declared else f"{name} (undeclared)" for name in unscorable
        )
        raise ExperimentConfigError(
            f"objective.composite weights signals that produce no score: {detail}; "
            f"scorable signals are {', '.join(sorted(scorable)) or '(none)'}"
        )


def _require_screening_weights(
    screening: Mapping[str, Any], signals: list[dict[str, Any]], *, mode: JudgingMode
) -> None:
    """Reject a child-gate blend the adapter cannot score.

    Parent selection stays on ``objective.primary``. ``screening.weights`` is only
    the gate a focused child must clear. A judge named here is started even when
    no signal declared it.
    """
    raw = screening.get("weights")
    if raw is None:
        return
    if screening.get("kind") == "correctness_floor":
        raise ExperimentConfigError("screening.weights cannot be combined with screening.kind correctness_floor")
    if not isinstance(raw, Mapping):
        raise ExperimentConfigError(f"screening.weights must be a mapping of signal name to weight, got {raw!r}")
    if not raw:
        raise ExperimentConfigError("screening.weights must weight at least one signal; omit it to gate on the primary")
    weights: dict[str, float] = {}
    for name, raw_weight in raw.items():
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, int | float):
            raise ExperimentConfigError(f"screening.weights weight for {name} must be a number, got {raw_weight!r}")
        weights[str(name)] = float(raw_weight)
    negative = sorted(name for name, weight in weights.items() if weight < 0)
    if negative:
        raise ExperimentConfigError(f"screening.weights must be non-negative: {', '.join(negative)}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        detail = ", ".join(f"{name}={weight:g}" for name, weight in sorted(weights.items()))
        raise ExperimentConfigError(f"screening.weights must sum to 1, got {total:g} ({detail})")
    scorable = _scorable_signal_names(signals, mode=mode)
    judge_names = set(JUDGE_SPECS) if mode in _MODES_WITH_CORTEX_JUDGES else set()
    allowed = scorable | judge_names
    unknown = sorted(name for name in weights if name not in allowed)
    if unknown:
        raise ExperimentConfigError(
            f"screening.weights names signals that produce no score: {', '.join(unknown)}; "
            f"scorable signals are {', '.join(sorted(allowed)) or '(none)'}"
        )


def _require_normalized_composite_weights(composite: Any) -> None:
    """Reject weights that cannot produce a score inside 0..1.

    High-signal selection treats ``score >= 1.0`` as a pass, so weights summing
    above 1 would mark failing entries perfect and drop them from the set the
    next generation reflects on. Rejected rather than normalized: rescaling
    would silently score a different objective than the one written down.
    """
    if composite is None:
        return
    if not isinstance(composite, Mapping):
        return
    # Omitting it falls back to the adapter default; writing it empty scores 0.0.
    if not composite:
        raise ExperimentConfigError("objective.composite must weight at least one signal; omit it to use the default")
    weights: dict[str, float] = {}
    for name, raw_weight in composite.items():
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, int | float):
            raise ExperimentConfigError(f"objective.composite weight for {name} must be a number, got {raw_weight!r}")
        weights[str(name)] = float(raw_weight)
    negative = sorted(name for name, weight in weights.items() if weight < 0)
    if negative:
        raise ExperimentConfigError(f"objective.composite weights must be non-negative: {', '.join(negative)}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        detail = ", ".join(f"{name}={weight:g}" for name, weight in sorted(weights.items()))
        raise ExperimentConfigError(
            f"objective.composite weights must sum to 1, got {total:g} ({detail}); "
            "a larger sum can push a failing entry to a passing score"
        )


def _require_unit_valued_weighted_constants(objective: Mapping[str, Any], signals: list[dict[str, Any]]) -> None:
    """Reject a weighted constant below 1.0.

    A constant adds the same amount to every candidate, so a value under 1 caps the
    composite below the ``score >= 1.0`` pass gate and nothing is ever selected.
    """
    composite = objective.get("composite")
    if not isinstance(composite, Mapping):
        return
    values = {
        str(signal["name"]): float(signal.get("value", 0.0))
        for signal in signals
        if signal.get("source") == _CONSTANT_SOURCE and signal.get("name")
    }
    offenders = sorted(
        f"{name}={values[name]:g}" for name in composite if name in values and abs(values[name] - 1.0) > 1e-6
    )
    if offenders:
        raise ExperimentConfigError(
            f"weighted constant signals must have value 1.0: {', '.join(offenders)}; "
            "a lower value caps every composite score below the 1.0 high-signal pass gate"
        )


def _require_mode_primary_objective(primary: Any, signals: list[dict[str, Any]], *, mode: JudgingMode) -> None:
    """Reject a primary the adapter cannot put in ``eval_batch.summary``.

    Screening reads ``summary[primary]``. Telemetry names always qualify; so does
    an enabled Cortex judge in a mode that starts judge runs.
    """
    if primary is None:
        return
    scorable = _scorable_signal_names(signals, mode=mode)
    if str(primary) not in scorable:
        names = ", ".join(sorted(scorable)) or "(none)"
        raise ExperimentConfigError(
            f"mode {mode} cannot score objective.primary={str(primary)!r}; scorable signals are {names}"
        )


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load glean_gepa experiment configs. Install gepa[glean].") from exc
    return yaml.safe_load(path.read_text())


def _require_focused_bucket_type(bucket: Any) -> None:
    if bucket is None:
        return
    if str(bucket) not in FOCUSED_BUCKET_TYPES:
        allowed = ", ".join(sorted(FOCUSED_BUCKET_TYPES))
        raise ExperimentConfigError(f"objective.focused_bucket_type must be one of {allowed}, got {bucket!r}")
