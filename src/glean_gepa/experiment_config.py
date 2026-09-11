"""Load packaged Glean GEPA experiment YAML and merge the mode with its signal pack."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from glean_gepa.adapter_types import JudgingMode, PointwiseJudge
from glean_gepa.objectives import (
    MODE_DEFAULT_PACK,
    is_registered_telemetry_source,
    is_telemetry_source,
)

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
PACKS_DIR = CONFIGS_DIR / "packs"
DEFAULT_EVAL_SET_NAME = "Glean Chat V2 Medium"
DEFAULT_DEPLOYMENT_IDS = ("scio-prod",)

# Mode is the eval topology. Telemetry sources are registered per mode in
# glean_gepa.objectives; a pack is valid when its telemetry sources are.
SUPPORTED_MODES: tuple[JudgingMode, ...] = ("single_model", "teacher_student")

# Sources whose signal value is read straight from telemetry or the config, as
# opposed to a judge run that has to be started and awaited.
_CONSTANT_SOURCE = "constant"

# Only teacher_student has the judge plumbing to start, await, and cache a judge run.
_MODES_WITH_POINTWISE_JUDGES = frozenset({"teacher_student"})

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
    packs: tuple[str, ...]
    source_path: Path
    run: dict[str, Any]
    models: dict[str, Any]
    data: dict[str, Any]
    signals: tuple[dict[str, Any], ...]
    objective: dict[str, Any]
    screening: dict[str, Any]
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
    """Load a mode YAML, merge its signal pack, and return the resolved experiment."""
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
    pack_names, pack = _load_mode_packs(raw.get("packs"), mode=mode, mode_path=source_path)
    merged_signals = _merge_signals(pack.get("signals") or [], raw.get("signals") or [])
    merged_objective = _overlay(pack.get("objective") or {}, raw.get("objective") or {})
    merged_screening = _overlay(pack.get("screening") or {}, raw.get("screening") or {})
    _require_mode_primary_objective(merged_objective.get("primary"), merged_signals, mode=mode)
    _require_scorable_composite_signals(merged_objective, merged_signals, mode=mode)
    _require_normalized_composite_weights(merged_objective.get("composite"))
    _require_unit_valued_weighted_constants(merged_objective, merged_signals)
    return ExperimentConfig(
        schema_version=schema_version,
        mode=mode,
        packs=pack_names,
        source_path=source_path,
        run=dict(raw.get("run") or {}),
        models=dict(raw.get("models") or {}),
        data=dict(raw.get("data") or {}),
        signals=tuple(merged_signals),
        objective=merged_objective,
        screening=merged_screening,
        search=dict(raw.get("search") or {}),
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
    if run.get("seed_candidate"):
        defaults["seed_candidate"] = Path(str(run["seed_candidate"]))
    if run.get("editable_modules") is not None:
        modules = run["editable_modules"]
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
    return tuple(_pointwise_judges(list(config.signals)))


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


def agentspan_lookback_days(config: ExperimentConfig) -> int | None:
    lookbacks = [
        int(signal["lookback_days"])
        for signal in config.signals
        if is_telemetry_source(str(signal.get("source") or "")) and signal.get("lookback_days") is not None
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


def _load_mode_packs(raw_packs: Any, *, mode: JudgingMode, mode_path: Path) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Load packs whose telemetry sources are registered for ``mode``."""
    if raw_packs is None:
        names = (MODE_DEFAULT_PACK[mode],)
    else:
        if isinstance(raw_packs, str):
            raise ExperimentConfigError(f"packs must be a list, got {raw_packs!r}")
        names = tuple(str(name) for name in raw_packs)
        if not names:
            raise ExperimentConfigError(f"mode {mode} requires at least one pack")
    merged_signals: list[Any] = []
    merged_objective: dict[str, Any] = {}
    merged_screening: dict[str, Any] = {}
    for name in names:
        pack = _load_pack(name, mode_path=mode_path)
        for signal in pack.get("signals") or []:
            if not isinstance(signal, dict):
                continue
            source = signal.get("source")
            if source in {_CONSTANT_SOURCE, "cortex_judge", None}:
                continue
            if not is_registered_telemetry_source(mode, str(source)):
                raise ExperimentConfigError(
                    f"mode {mode} cannot score pack {name!r} (source {source!r} is not registered for this mode)"
                )
        merged_signals.extend(pack.get("signals") or [])
        merged_objective = _overlay(merged_objective, pack.get("objective") or {})
        merged_screening = _overlay(merged_screening, pack.get("screening") or {})
    return names, {"signals": merged_signals, "objective": merged_objective, "screening": merged_screening}


def _pointwise_judges(signals: list[dict[str, Any]]) -> list[PointwiseJudge]:
    """Pairwise signals are skipped: nothing scores them per entry. Same for
    ``objective.validation``, which is carried on the config but never read."""
    judges: list[PointwiseJudge] = []
    for signal in signals:
        if signal.get("source") != "cortex_judge" or signal.get("kind") != "pointwise":
            continue
        # Before the enabled check, so flipping enabled on cannot surface a new error.
        judge_type = signal.get("type")
        if not judge_type:
            raise ExperimentConfigError(f"pointwise cortex_judge signal {signal.get('name')!r} requires type")
        if signal.get("enabled", True) is False:
            continue
        judges.append(PointwiseJudge(str(signal["name"]), str(judge_type), judge_run_params_json(signal)))
    return judges


def _scorable_signal_names(signals: list[dict[str, Any]], *, mode: JudgingMode) -> set[str]:
    """Signals the mode's adapter can turn into a composite dimension.

    Everything else -- pairwise judges, disabled signals, judges in a mode with no
    judge plumbing -- is collected for reporting or future use but never produces a
    per-entry score.
    """
    names: set[str] = set()
    if mode in _MODES_WITH_POINTWISE_JUDGES:
        names.update(str(judge.name) for judge in _pointwise_judges(signals))
    for signal in signals:
        if signal.get("enabled", True) is False:
            continue
        if signal.get("source") == _CONSTANT_SOURCE or is_registered_telemetry_source(mode, signal.get("source")):
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
    telemetry_names = {
        str(signal["name"])
        for signal in signals
        if signal.get("enabled", True) is not False
        and is_registered_telemetry_source(mode, signal.get("source"))
        and signal.get("name")
    }
    if primary is None:
        return
    if str(primary) not in telemetry_names:
        scorable = ", ".join(sorted(telemetry_names)) or "(none)"
        raise ExperimentConfigError(
            f"mode {mode} cannot score objective.primary={str(primary)!r}; scorable telemetry signals are {scorable}"
        )


def _load_pack(name: str, *, mode_path: Path) -> dict[str, Any]:
    candidates = [mode_path.parent / "packs" / f"{name}.yaml", PACKS_DIR / f"{name}.yaml"]
    for path in candidates:
        if path.is_file():
            loaded = _load_yaml(path)
            if not isinstance(loaded, dict):
                raise ExperimentConfigError(f"pack {name} must be a mapping")
            return loaded
    raise ExperimentConfigError(f"unknown pack {name!r}")


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load glean_gepa experiment configs. Install gepa[glean].") from exc
    return yaml.safe_load(path.read_text())


def _merge_signals(*groups: list[Any]) -> list[dict[str, Any]]:
    """Merge signals by name, field by field.

    A mode that re-declares a pack signal usually means to adjust one field, so
    replacing the whole entry would drop the rest -- losing ``source`` leaves a
    signal nothing can score, which fails the load. Nested values such as
    ``run_params`` are still replaced whole, matching ``_overlay``: a partial
    judge payload is more likely a mistake than an intended merge.
    """
    by_name: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for group in groups:
        for signal in group:
            if not isinstance(signal, dict) or "name" not in signal:
                raise ExperimentConfigError("each signal must be a mapping with a name")
            name = str(signal["name"])
            if name not in by_name:
                order.append(name)
            by_name[name] = {**by_name.get(name, {}), **signal}
    return [by_name[name] for name in order]


def _overlay(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay top-level keys, replacing container values instead of merging them.

    ``objective.composite`` must replace the pack's wholesale: merging the two would
    make a weight the pack declared impossible to remove, and would let the surviving
    weights sum past 1 without anyone noticing.
    """
    return {**base, **overlay}
