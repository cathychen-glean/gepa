"""Prompt targets: the scio prompts GEPA overrides, split into editable sections.

Each folder under ``glean_gepa/prompts/`` is one target, named after the scio template it
overrides (``data/prompts/templates/<name>.prompt``):

- ``template.prompt`` is the scio template with ``{#KEY}`` ... ``{/KEY}`` markers around
  every section a run may edit. Deleting the marker lines gives back the scio file: a
  marker alone on its line takes that line break with it. Sections may nest, and a bare
  ``{KEY}`` slot declares a section whose stock text is empty.
- ``target.yaml`` names the template module and gives each section its token budget,
  reflection frame, fill rule, and the scio markup a rewrite must keep. It also lists the
  harnesses the prompt renders under and the scParams a run needs to render it.

Section keys are candidate keys, so they are unique across targets. The compiled text is
sent as ``llmo.per_prompt_overrides.<name>``; a ``kind: tool_descriptions`` target instead
encodes one block per tool into ``co.pyagents_tool_description_overrides``.

A seed file is either JSON (``{"KEY": "text", ...}``) or a ``.prompt`` file marked the
same way as ``template.prompt``, which is the easy way to seed from a scio PR.
"""

from __future__ import annotations

import json
import re
from base64 import urlsafe_b64encode
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal, cast

import yaml

PROMPTS_DIR = Path(__file__).parent / "prompts"
TARGET_FILE = "target.yaml"
TEMPLATE_FILE = "template.prompt"
PER_PROMPT_OVERRIDE_PREFIX = "llmo.per_prompt_overrides."

TargetKind = Literal["template", "tool_descriptions"]
ModuleRole = Literal["template", "section"]
FillRule = Literal["stripped", "verbatim"]

_MARKER = re.compile(r"\{([#/])([A-Za-z_]\w*)\}")
_PLACEHOLDER = re.compile(r"\[\[(\w+)\]\]")
_CONDITIONAL_OPEN = re.compile(r"<<<\[\[(\w+)\]\]")

_TARGET_KEYS = frozenset(
    {
        "kind",
        "override_param",
        "harnesses",
        "group",
        "always_emit",
        "template",
        "sections",
        "section_defaults",
        "render",
    }
)
_MODULE_KEYS = frozenset(
    {
        "key",
        "editable",
        "token_budget",
        "frame",
        "fill",
        "drop_empty_line",
        "required_placeholders",
        "required_conditionals",
    }
)
_RENDER_KEYS = frozenset({"sc_params", "drop_sc_params"})


class PromptTargetError(ValueError):
    """A prompt folder, seed file, or editable-module list is inconsistent."""


def placeholders(text: str) -> set[str]:
    """Scio ``[[name]]`` placeholders in ``text``."""
    return set(_PLACEHOLDER.findall(text))


def conditional_counts(text: str) -> dict[str, int]:
    """Count ``<<<[[name]] ... >>>`` conditional openers by name."""
    counts: dict[str, int] = {}
    for name in _CONDITIONAL_OPEN.findall(text):
        counts[name] = counts.get(name, 0) + 1
    return counts


def encode_b64(text: str) -> str:
    """URL-safe base64. QE parses ``sc=`` with ``url.QueryUnescape``, which turns ``+`` into
    a space, so standard base64 fails to decode and scio drops the override."""
    return urlsafe_b64encode(text.encode("utf-8")).decode("ascii")


@dataclass(frozen=True)
class MarkedPrompt:
    """A prompt split at its ``{#KEY}`` markers.

    ``template`` is the text outside every section, with each top-level section replaced by
    its ``{KEY}`` slot. ``sections`` holds each section's text in document order; a section
    that contains another holds that one's slot.
    """

    template: str
    sections: dict[str, str]


def parse_marked_prompt(text: str) -> MarkedPrompt:
    sections: dict[str, str] = {}
    stack: list[tuple[str | None, list[str]]] = [(None, [])]
    pos = 0
    for match in _MARKER.finditer(text):
        kind, key = match.groups()
        start, end = match.span()
        own_line = (start == 0 or text[start - 1] == "\n") and (end == len(text) or text[end] == "\n")
        chunk = text[pos:start]
        if kind == "#":
            if key in sections:
                raise PromptTargetError(f"section {key} is marked more than once")
            stack[-1][1].extend((chunk, "{" + key + "}"))
            sections[key] = ""
            stack.append((key, []))
            pos = end + 1 if own_line and end < len(text) else end
            continue
        open_key, parts = stack[-1]
        if open_key != key:
            expected = f"{{/{open_key}}}" if open_key else "no close marker"
            raise PromptTargetError(f"{{/{key}}} found where {expected} was expected")
        if own_line and chunk.endswith("\n"):
            chunk = chunk[:-1]
        parts.append(chunk)
        stack.pop()
        sections[key] = "".join(parts)
        pos = end
    if len(stack) > 1:
        raise PromptTargetError(f"{{#{stack[-1][0]}}} is never closed")
    stack[0][1].append(text[pos:])
    return MarkedPrompt("".join(stack[0][1]), sections)


@dataclass(frozen=True)
class PromptModule:
    """One candidate key: a target's template or one of its sections."""

    key: str
    target: str
    role: ModuleRole
    stock: str
    editable: bool
    token_budget: int | None
    frame: str
    #: ``stripped`` fills the slot with the stripped candidate text, or stock text when that
    #: is empty. ``verbatim`` fills it with the candidate text exactly, stock only when absent.
    fill: FillRule
    #: An empty fill removes the slot and the rest of its line instead of leaving a blank line.
    drop_empty_line: bool
    required_placeholders: tuple[str, ...]
    required_conditionals: tuple[str, ...]

    @property
    def slot(self) -> str:
        return "{" + self.key + "}"

    def fill_text(self, candidate: Mapping[str, str]) -> str:
        if self.fill == "verbatim":
            return candidate.get(self.key, self.stock)
        return candidate.get(self.key, "").strip() or self.stock

    def missing_markup(self, text: str) -> list[str]:
        """Required scio markup that ``text`` lacks."""
        missing = [f"[[{name}]]" for name in self.required_placeholders if f"[[{name}]]" not in text]
        counts = conditional_counts(text)
        missing += [f"<<<[[{name}]]" for name in self.required_conditionals if not counts.get(name)]
        return missing


@dataclass(frozen=True)
class PromptTarget:
    name: str
    kind: TargetKind
    override_param: str
    template_key: str | None
    #: Template module first (when there is one), then sections in fill order.
    modules: Mapping[str, PromptModule]
    harnesses: tuple[str, ...]
    group: str | None
    #: Send the override even when the candidate carries no module of this target.
    always_emit: bool
    render_sc_params: tuple[str, ...]
    drop_sc_params: tuple[str, ...]

    @property
    def section_keys(self) -> tuple[str, ...]:
        return tuple(key for key, module in self.modules.items() if module.role == "section")

    def carries(self, candidate: Mapping[str, str]) -> bool:
        return any(candidate.get(key) for key in self.modules)

    def compile_text(self, candidate: Mapping[str, str]) -> str:
        """Fill every reachable section slot, including slots that sections bring in.

        Uses replace, not ``str.format``: scio templates are full of braces.
        """
        if self.template_key is None:
            raise PromptTargetError(f"{self.name} has no template to compile")
        text = candidate.get(self.template_key, self.modules[self.template_key].stock)
        filled: set[str] = set()
        progress = True
        while progress:
            progress = False
            for key in self.section_keys:
                module = self.modules[key]
                if key in filled or module.slot not in text:
                    continue
                filled.add(key)
                progress = True
                value = module.fill_text(candidate)
                if value or not module.drop_empty_line:
                    text = text.replace(module.slot, value)
                else:
                    text = re.sub(re.escape(module.slot) + r"[ \t]*\n?", "", text)
        return text

    def override(self, candidate: Mapping[str, str]) -> str:
        """This target's scParam fragment, or ``""`` when the eval should keep scio's text."""
        if self.kind == "tool_descriptions":
            segments = [f"{key}:{encode_b64(candidate[key])}" for key in self.section_keys if candidate.get(key)]
            return f"{self.override_param}=" + ";".join(segments) if segments else ""
        if not (self.always_emit or self.carries(candidate)):
            return ""
        return f"{self.override_param}={encode_b64(self.compile_text(candidate))}"

    def reachable_sections(self, candidate: Mapping[str, str]) -> list[str]:
        """Sections whose slot appears in the template or in another reachable section."""
        if self.template_key is None:
            return list(self.section_keys)
        texts = [candidate.get(self.template_key, self.modules[self.template_key].stock)]
        reached: list[str] = []
        while texts:
            text = texts.pop()
            for key in self.section_keys:
                if key not in reached and self.modules[key].slot in text:
                    reached.append(key)
                    texts.append(self.modules[key].fill_text(candidate))
        return reached


def _str_tuple(raw: Any, where: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise PromptTargetError(f"{where} must be a list of strings")
    return tuple(raw)


def _module(
    key: str,
    raw: Mapping[str, Any],
    *,
    target: str,
    role: ModuleRole,
    stock: str,
    where: str,
) -> PromptModule:
    unknown = set(raw) - _MODULE_KEYS
    if unknown:
        raise PromptTargetError(f"{where} has unknown keys: {', '.join(sorted(unknown))}")
    fill = raw.get("fill", "stripped")
    if fill not in ("stripped", "verbatim"):
        raise PromptTargetError(f"{where}.fill must be 'stripped' or 'verbatim', got {fill!r}")
    budget = raw.get("token_budget")
    if budget is not None and (not isinstance(budget, int) or budget <= 0):
        raise PromptTargetError(f"{where}.token_budget must be a positive integer")
    return PromptModule(
        key=key,
        target=target,
        role=role,
        stock=stock,
        editable=bool(raw.get("editable", True)),
        token_budget=budget,
        frame=" ".join(str(raw.get("frame", "")).split()).replace("{name}", key),
        fill=cast(FillRule, fill),
        drop_empty_line=bool(raw.get("drop_empty_line", False)),
        required_placeholders=_str_tuple(raw.get("required_placeholders"), f"{where}.required_placeholders"),
        required_conditionals=_str_tuple(raw.get("required_conditionals"), f"{where}.required_conditionals"),
    )


def load_prompt_target(folder: Path) -> PromptTarget:
    name = folder.name
    raw = yaml.safe_load((folder / TARGET_FILE).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise PromptTargetError(f"{folder / TARGET_FILE} must be a mapping")
    unknown = set(raw) - _TARGET_KEYS
    if unknown:
        raise PromptTargetError(f"{name}/{TARGET_FILE} has unknown keys: {', '.join(sorted(unknown))}")
    kind = raw.get("kind", "template")
    if kind not in ("template", "tool_descriptions"):
        raise PromptTargetError(f"{name}.kind must be 'template' or 'tool_descriptions', got {kind!r}")
    marked = parse_marked_prompt((folder / TEMPLATE_FILE).read_text(encoding="utf-8"))

    declared = raw.get("sections") or {}
    defaults = raw.get("section_defaults") or {}
    if not isinstance(declared, Mapping) or not isinstance(defaults, Mapping):
        raise PromptTargetError(f"{name}: sections and section_defaults must be mappings")
    undeclared = [key for key in marked.sections if key not in declared]
    if undeclared and not defaults:
        raise PromptTargetError(
            f"{name}/{TEMPLATE_FILE} marks sections missing from {TARGET_FILE}: {', '.join(undeclared)}"
        )

    modules: dict[str, PromptModule] = {}
    template_key: str | None = None
    if kind == "template":
        template_raw = raw.get("template")
        key = template_raw.get("key") if isinstance(template_raw, Mapping) else None
        if not isinstance(template_raw, Mapping) or not isinstance(key, str):
            raise PromptTargetError(f"{name}.template.key is required")
        template_key = key
        modules[key] = _module(
            key,
            template_raw,
            target=name,
            role="template",
            stock=marked.template,
            where=f"{name}.template",
        )
    for key in [*declared, *undeclared]:
        section_raw = {**defaults, **(declared.get(key) or {})}
        modules[key] = _module(
            key,
            section_raw,
            target=name,
            role="section",
            stock=marked.sections.get(key, ""),
            where=f"{name}.sections.{key}",
        )

    render = raw.get("render") or {}
    if not isinstance(render, Mapping) or set(render) - _RENDER_KEYS:
        raise PromptTargetError(f"{name}.render may only set {', '.join(sorted(_RENDER_KEYS))}")
    harnesses = _str_tuple(raw.get("harnesses"), f"{name}.harnesses")
    if not harnesses:
        raise PromptTargetError(f"{name}.harnesses must list at least one harness")
    target = PromptTarget(
        name=name,
        kind=kind,
        override_param=str(raw.get("override_param") or PER_PROMPT_OVERRIDE_PREFIX + name),
        template_key=template_key,
        modules=modules,
        harnesses=harnesses,
        group=raw.get("group"),
        always_emit=bool(raw.get("always_emit", False)),
        render_sc_params=_str_tuple(render.get("sc_params"), f"{name}.render.sc_params"),
        drop_sc_params=_str_tuple(render.get("drop_sc_params"), f"{name}.render.drop_sc_params"),
    )
    for key, module in modules.items():
        if module.stock and module.missing_markup(module.stock):
            raise PromptTargetError(f"{name}: stock {key} lacks its required markup")
    return target


@cache
def prompt_targets() -> dict[str, PromptTarget]:
    """Every target under :data:`PROMPTS_DIR`, in folder-name order (the scParam order)."""
    targets: dict[str, PromptTarget] = {}
    owners: dict[str, str] = {}
    for folder in sorted(path for path in PROMPTS_DIR.iterdir() if (path / TARGET_FILE).is_file()):
        target = load_prompt_target(folder)
        names = [*target.modules, *([target.group] if target.group else [])]
        for key in names:
            if key in owners:
                raise PromptTargetError(f"{key} is declared by both {owners[key]} and {target.name}")
            owners[key] = target.name
        targets[target.name] = target
    return targets


@cache
def _modules() -> dict[str, PromptModule]:
    return {key: module for target in prompt_targets().values() for key, module in target.modules.items()}


def prompt_module(key: str) -> PromptModule | None:
    return _modules().get(key)


def owning_target(key: str) -> PromptTarget | None:
    module = prompt_module(key)
    return prompt_targets()[module.target] if module else None


def stock_text(key: str) -> str | None:
    module = prompt_module(key)
    return module.stock if module else None


def module_token_budget(key: str) -> int | None:
    module = prompt_module(key)
    return module.token_budget if module else None


def section_frame(key: str) -> str:
    """The fixed editing contract for ``key`` from its ``target.yaml``, or ``""``."""
    module = prompt_module(key)
    return module.frame if module else ""


def known_module_keys() -> frozenset[str]:
    return frozenset(_modules())


def editable_module_keys() -> frozenset[str]:
    return frozenset(key for key, module in _modules().items() if module.editable)


def module_groups() -> dict[str, tuple[str, ...]]:
    """``editable_modules`` shorthands, e.g. ``CORE_TOOLS`` for every core-tool section."""
    return {target.group: target.section_keys for target in prompt_targets().values() if target.group}


def targets_for(keys: Iterable[str]) -> list[PromptTarget]:
    """Targets that own any of ``keys``, in registry order."""
    names = {module.target for module in map(prompt_module, keys) if module}
    return [target for name, target in prompt_targets().items() if name in names]


def compile_prompt_text(target_name: str, candidate: Mapping[str, str]) -> str:
    return prompt_targets()[target_name].compile_text(candidate)


def compile_overrides(candidate: Mapping[str, str]) -> str:
    """Every target's scParam fragment for ``candidate``, comma-joined in registry order."""
    return ",".join(part for part in (target.override(candidate) for target in prompt_targets().values()) if part)


def render_requirements(keys: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(extra, dropped)`` scParams every eval needs so the targets owning ``keys`` render."""
    extra: dict[str, None] = {}
    dropped: dict[str, None] = {}
    for target in targets_for(keys):
        extra.update(dict.fromkeys(target.render_sc_params))
        dropped.update(dict.fromkeys(target.drop_sc_params))
    return tuple(extra), tuple(dropped)


def parse_editable_modules(parts: Sequence[str]) -> list[str]:
    """Resolve ``editable_modules`` entries (keys or group names) to keys, de-duplicated in order."""
    if not parts:
        raise PromptTargetError("editable_modules must list at least one prompt key")
    groups = module_groups()
    editable = editable_module_keys()
    modules: list[str] = []
    unknown: list[str] = []
    for part in parts:
        if part in groups:
            modules.extend(key for key in groups[part] if key not in modules)
        elif part in editable:
            if part not in modules:
                modules.append(part)
        elif (target := owning_target(part)) is not None:
            raise PromptTargetError(
                f"{part} is not editable: it is the {target.name} render template. Edit one of its sections "
                f"({', '.join(target.section_keys)}) instead."
            )
        else:
            unknown.append(part)
    if unknown:
        unknown_list = ", ".join(sorted(repr(key) for key in unknown))
        known_list = ", ".join(sorted([*editable, *groups]))
        raise PromptTargetError(f"unknown editable_modules: {unknown_list}. Known keys: {known_list}")
    return modules


def _check_seed_keys(raw: Mapping[str, Any]) -> dict[str, str]:
    unknown = set(raw) - known_module_keys()
    if unknown:
        unknown_list = ", ".join(sorted(repr(key) for key in unknown))
        raise PromptTargetError(f"seed_candidate has unknown keys: {unknown_list}")
    seed: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(value, str):
            raise PromptTargetError(f"{key} must be a string. Got type={type(value)}")
        seed[key] = value
    return seed


def parse_seed_prompt(text: str) -> dict[str, str]:
    """Seed modules from a marked ``.prompt`` file; its sections pick the target."""
    marked = parse_marked_prompt(text)
    if not marked.sections:
        raise PromptTargetError("a .prompt seed must mark at least one {#KEY} ... {/KEY} section")
    _check_seed_keys(marked.sections)
    targets = targets_for(marked.sections)
    if len(targets) != 1:
        names = ", ".join(target.name for target in targets)
        raise PromptTargetError(f"a .prompt seed must mark sections of one target, got {names}")
    target = targets[0]
    if target.template_key is None:
        return dict(marked.sections)
    return {target.template_key: marked.template, **marked.sections}


def load_seed_file(path: Path) -> dict[str, str]:
    """Load seed modules from JSON or a marked ``.prompt``. Omitted keys use stock text."""
    if not path.is_file():
        raise PromptTargetError(f"seed_candidate file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".prompt":
        return parse_seed_prompt(text)
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise PromptTargetError("seed_candidate must be a JSON object")
    return _check_seed_keys(raw)


def build_seed(raw: Mapping[str, str], editable_modules: Sequence[str]) -> dict[str, str]:
    """The GEPA candidate for ``editable_modules``.

    Each editable module takes its seed text, or stock text when the seed omits it. Seed
    text for the template and for every frozen section it renders is pinned too, so the
    compiled prompt matches the seed file rather than falling back to stock.
    """
    seed = {key: raw.get(key, stock_text(key) or "") for key in editable_modules}
    for target in targets_for(editable_modules):
        if target.template_key is None:
            continue
        if target.template_key in raw:
            seed.setdefault(target.template_key, raw[target.template_key])
        view = {**raw, **seed}
        reached = target.reachable_sections(view)
        for key in reached:
            if key in raw:
                seed.setdefault(key, raw[key])
        for key in editable_modules:
            module = target.modules.get(key)
            if module is None or module.role != "section":
                continue
            if key not in reached:
                raise PromptTargetError(
                    f"{key} is editable but the seed {target.template_key} has no {module.slot} slot. Add "
                    f"{module.slot} to {target.template_key} (or to a section it renders) in the seed file, or "
                    f"drop {key} from editable_modules."
                )
        for key in editable_modules:
            module = target.modules.get(key)
            missing = module.missing_markup(seed[key]) if module else []
            if missing:
                raise PromptTargetError(
                    f"seed {key} is missing required scio markup: {', '.join(missing)}. Scio fills or "
                    "branches on it at render time."
                )
    return seed


def harness_requirements(keys: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Target name -> harnesses it renders under, for the targets owning ``keys``."""
    return {target.name: target.harnesses for target in targets_for(keys)}
