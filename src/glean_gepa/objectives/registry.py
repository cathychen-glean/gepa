"""The single catalog of objectives and the one place they are registered.

To add an objective:

1. Implement a ``TeacherStudentObjective`` or ``SingleModelObjective`` subclass.
2. Append one :class:`ObjectiveSpec` to :data:`BUILTIN_OBJECTIVES` below.
3. Declare a signal in an experiment YAML (``configs/*.yaml``) whose ``signals[].source`` is the spec's ``source``.

Nothing else needs editing. ``base.build_objective`` resolves through this
module, and :func:`register` rejects a class that does not satisfy the shared
contract in :mod:`glean_gepa.objectives.protocol` before any run starts.

Out-of-tree objectives (tests, experiments) call :func:`register` directly.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, get_args

from glean_gepa.adapter_types import JudgingMode
from glean_gepa.objectives.protocol import check_objective_contract

if TYPE_CHECKING:
    from glean_gepa.objectives.base import SingleModelObjective, TeacherStudentObjective

VALID_MODES: frozenset[str] = frozenset(get_args(JudgingMode))


class ObjectiveRegistrationError(TypeError):
    """A class cannot be registered as an objective. The message names every problem."""


@dataclass(frozen=True)
class ObjectiveSpec:
    """One row of the catalog.

    Attributes
    ----------
    mode
        Eval topology that can score this objective.
    source
        Value an experiment YAML puts in ``signals[].source`` to select it.
    class_path
        ``module:ClassName``. Imported lazily so the catalog can be read
        without importing BigQuery helpers.
    default
        Objective used when a config for ``mode`` declares no scorable signal.
        Exactly one spec per mode must set this.
    summary
        One line for humans and ``--list-objectives`` style output.
    """

    mode: JudgingMode
    source: str
    class_path: str
    summary: str
    default: bool = False

    def load(self) -> type:
        module_name, _, class_name = self.class_path.partition(":")
        if not module_name or not class_name:
            raise ObjectiveRegistrationError(f"class_path must be 'module:Class', got {self.class_path!r}")
        return getattr(importlib.import_module(module_name), class_name)


# ---------------------------------------------------------------------------
# Catalog. Edit this list to add an objective.
# ---------------------------------------------------------------------------
BUILTIN_OBJECTIVES: tuple[ObjectiveSpec, ...] = (
    ObjectiveSpec(
        mode="teacher_student",
        source="tool_match",
        class_path="glean_gepa.objectives.tool_match:FirstToolMatchObjective",
        summary="First tool the student called matches the teacher's.",
        default=True,
    ),
    ObjectiveSpec(
        mode="teacher_student",
        source="citation_match",
        class_path="glean_gepa.objectives.citation_match:CitationMatchObjective",
        summary="Student citation set matches the teacher's.",
    ),
    ObjectiveSpec(
        mode="teacher_student",
        source="agentic_preference",
        class_path="glean_gepa.objectives.agentic_preference:AgenticPreferenceObjective",
        summary="Pairwise AGENTIC_JUDGE prefers the student over the teacher.",
    ),
    ObjectiveSpec(
        mode="single_model",
        source="shell_telemetry",
        class_path="glean_gepa.objectives.shell:ShellSuccessObjective",
        summary="Shell tool calls succeed (Agentspan telemetry).",
        default=True,
    ),
    ObjectiveSpec(
        mode="single_model",
        source="loop_telemetry",
        class_path="glean_gepa.objectives.loop:LoopEfficiencyObjective",
        summary="Student uses fewer agent loops; every loop lowers the score.",
    ),
)


# ---------------------------------------------------------------------------
# Registry state.
# ---------------------------------------------------------------------------
_REGISTRY: dict[tuple[JudgingMode, str], type] = {}
_BUILTINS_LOADED = False


def _mode_base(mode: JudgingMode) -> type[TeacherStudentObjective | SingleModelObjective]:
    from glean_gepa.objectives.base import SingleModelObjective, TeacherStudentObjective

    return TeacherStudentObjective if mode == "teacher_student" else SingleModelObjective


def validate_objective_class(cls: type, *, mode: JudgingMode, source: str) -> list[str]:
    """Return every reason ``cls`` cannot be registered for ``(mode, source)``. Empty means OK."""
    problems: list[str] = []
    if mode not in VALID_MODES:
        problems.append(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    if not isinstance(source, str) or not source or not source.replace("_", "a").isalnum():
        problems.append(f"source must be a non-empty snake_case identifier, got {source!r}")
    if not inspect.isclass(cls):
        problems.append(f"expected a class, got {cls!r}")
        return problems
    if mode in VALID_MODES:
        base = _mode_base(mode)
        if not issubclass(cls, base):
            problems.append(f"must subclass {base.__name__} to be scorable by mode {mode!r}")
    problems.extend(f"missing {member}" for member in check_objective_contract(cls))
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        params = {}
    for required in ("bigquery_client", "lookback_days"):
        if required not in params and not any(p.kind is p.VAR_KEYWORD for p in params.values()):
            problems.append(f"__init__ must accept keyword {required!r}")
    return problems


def register(
    mode: JudgingMode,
    source: str,
    cls: type,
    *,
    replace: bool = False,
    validate: bool = True,
) -> None:
    """Register ``cls`` for ``(mode, source)``.

    Raises :class:`ObjectiveRegistrationError` when the class fails validation
    or when the key is already bound to a *different* class and ``replace`` is
    false. Re-registering the same class is a no-op. ``validate=False`` is for
    test doubles only.
    """
    key = (mode, source)
    existing = _REGISTRY.get(key)
    if existing is not None and existing is not cls and not replace:
        raise ObjectiveRegistrationError(
            f"{mode}/{source} is already registered to {existing.__module__}.{existing.__qualname__}; "
            f"pass replace=True to override"
        )
    if validate:
        problems = validate_objective_class(cls, mode=mode, source=source)
        if problems:
            label = f"{getattr(cls, '__module__', '?')}.{getattr(cls, '__qualname__', repr(cls))}"
            raise ObjectiveRegistrationError(
                f"cannot register {label} as {mode}/{source}:\n  - " + "\n  - ".join(problems)
            )
    _REGISTRY[key] = cls


def unregister(mode: JudgingMode, source: str) -> None:
    _REGISTRY.pop((mode, source), None)


def load_builtins() -> None:
    """Import and register every spec in :data:`BUILTIN_OBJECTIVES` once."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    defaults_per_mode: dict[str, list[str]] = {}
    for spec in BUILTIN_OBJECTIVES:
        if spec.default:
            defaults_per_mode.setdefault(spec.mode, []).append(spec.source)
    bad_defaults = {mode: sources for mode, sources in defaults_per_mode.items() if len(sources) != 1}
    if bad_defaults or set(defaults_per_mode) != VALID_MODES:
        raise ObjectiveRegistrationError(
            f"BUILTIN_OBJECTIVES must set default=True on exactly one spec per mode; got {defaults_per_mode}"
        )
    for spec in BUILTIN_OBJECTIVES:
        register(spec.mode, spec.source, spec.load())
    _BUILTINS_LOADED = True


def registered() -> dict[tuple[JudgingMode, str], type]:
    """Live view of the registry, builtins included."""
    load_builtins()
    return _REGISTRY


def resolve(mode: JudgingMode, source: str) -> type:
    """Class for ``(mode, source)``; ``KeyError`` lists the sources valid for ``mode``."""
    load_builtins()
    try:
        return _REGISTRY[(mode, source)]
    except KeyError:
        valid = ", ".join(sorted(sources_for_mode(mode))) or "<none>"
        raise KeyError(f"no objective registered for {mode}/{source}; valid sources for {mode}: {valid}") from None


def is_registered(mode: JudgingMode, source: str | None) -> bool:
    return bool(source) and (mode, source) in registered()


def is_known_source(source: str | None) -> bool:
    """True when ``source`` is registered under any mode."""
    return bool(source) and any(registered_source == source for _mode, registered_source in registered())


def sources_for_mode(mode: JudgingMode) -> list[str]:
    return [source for registered_mode, source in registered() if registered_mode == mode]


def _default_spec(mode: JudgingMode) -> ObjectiveSpec:
    for spec in BUILTIN_OBJECTIVES:
        if spec.mode == mode and spec.default:
            return spec
    raise ObjectiveRegistrationError(f"no BUILTIN_OBJECTIVES spec sets default=True for mode {mode!r}")


def default_source(mode: JudgingMode) -> str:
    """Source used when a config for ``mode`` names no scorable signal."""
    return _default_spec(mode).source


def iter_specs() -> Iterator[ObjectiveSpec]:
    yield from BUILTIN_OBJECTIVES


def describe() -> str:
    """Human-readable table of the catalog, for CLI help or docs."""
    lines = [f"{'mode':<16} {'source':<20} {'default':<8} summary"]
    for spec in BUILTIN_OBJECTIVES:
        lines.append(f"{spec.mode:<16} {spec.source:<20} {'yes' if spec.default else '-':<8} {spec.summary}")
    return "\n".join(lines)


__all__ = [
    "BUILTIN_OBJECTIVES",
    "VALID_MODES",
    "ObjectiveRegistrationError",
    "ObjectiveSpec",
    "default_source",
    "describe",
    "is_known_source",
    "is_registered",
    "iter_specs",
    "load_builtins",
    "register",
    "registered",
    "resolve",
    "sources_for_mode",
    "unregister",
    "validate_objective_class",
]
