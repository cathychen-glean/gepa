# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from difflib import unified_diff
from pathlib import Path
from typing import Any, cast

from gepa.core.adapter import DataInst, invoke_batch_evaluate
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.state import GEPAState
from gepa.logging.experiment_tracker import ExperimentTracker
from gepa.logging.logger import LoggerProtocol
from gepa.proposer.base import CandidateProposal
from glean_gepa.adapter_types import ALDataInst
from glean_gepa.al_adapter import (
    Candidate,
    GleanAdapterBase,
    ModuleSpec,
    within_prompt_budget,
)
from glean_gepa.batch import EvalRunIds, GleanEvaluationBatch
from glean_gepa.evalset_policy import UnseenEvalSetPolicy
from glean_gepa.prompt_constants import CORE_TOOL_KEYS, PROMPT_MODULE_DEFAULTS
from glean_gepa.reflection_prompts import parse_chosen_tool_keys, tool_description_choice_prompt
from glean_gepa.run_log import format_child_proposal_report, format_screening_report, log_section
from glean_gepa.utils import apply_single_module_edit

HIGH_SIGNAL_FIX_RATE_THRESHOLD = 0.5
SKIP_CHILD_SCREENING_KINDS = frozenset({"none", "skip"})
# How many tool descriptions one reflection pass may nominate, most useful first.
_TOOL_CHOICE_LIMIT = 3
# A tool description is only worth rewriting when the examples show the tool being invoked
# (by the student, or by the teacher where the student skipped it) in at least this many
# examples. The student asks, offers, and hedges mostly in its final message, so a rarely
# invoked tool's description cannot steer that; those rules belong in Execution Discipline.
MIN_TOOL_EVIDENCE_EXAMPLES = 3

_TOOL_NAME_CHARS = re.compile(r"[^a-z0-9]")


def _skips_child_screening(adapter: Any) -> bool:
    """True when the experiment asked not to run a focused or full-train screen."""
    return getattr(adapter, "screening_kind", None) in SKIP_CHILD_SCREENING_KINDS


@dataclass
class ChildCacheRecord:
    """Persisted state for one generated child and its focused screen."""

    eval_run_ids: list[EvalRunIds] = field(default_factory=list)
    screening_score: float | None = None
    screening_passed: bool | None = None


def pick_modules_to_edit(
    adapter: GleanAdapterBase,
    eval_batch: GleanEvaluationBatch | None = None,
) -> list[str]:
    """Return modules eligible for a rewrite this generation.

    Non-core modules listed in ``editable_modules`` are always eligible. Every
    listed core-tool description is eligible too; the reflector later names
    which of those descriptions to rewrite.
    """
    del eval_batch
    eligible = list(adapter.editable_modules)
    modules = [module for module in eligible if module not in CORE_TOOL_KEYS]
    modules.extend(module for module in eligible if module in CORE_TOOL_KEYS)
    return modules


def _choice_example_blocks(examples: Sequence[Mapping[str, Any]]) -> str:
    blocks: list[str] = []
    for example in examples:
        inputs = example["Inputs"]
        outputs = example["Generated Outputs"]
        teacher_tools = outputs.get("teacher_tools") or []
        student_tools = outputs.get("student_tools") or []
        teacher = ", ".join(str(name) for name in teacher_tools) or "(none)"
        student = ", ".join(str(name) for name in student_tools) or "(none)"
        action_lines = "".join(f"ACTION_INPUT: {line}\n" for line in example.get("Action Inputs") or [])
        blocks.append(
            f"---\nQUERY: {inputs.get('query', '')}\n"
            f"TEACHER_TOOLS: {teacher}\nSTUDENT_TOOLS: {student}\n"
            f"{action_lines}"
            f"FEEDBACK: {example.get('Feedback', '')}\n"
        )
    return "".join(blocks)


def _tool_key_matches(tool_key: str, event_name: Any) -> bool:
    """Whether a trace tool event (``"Glean Document Reader"``) is the core tool ``glean_document_reader``."""
    return _TOOL_NAME_CHARS.sub("", str(event_name).lower()) == _TOOL_NAME_CHARS.sub("", tool_key.lower())


def tool_usage_in_examples(
    tool_keys: Sequence[str], examples: Sequence[Mapping[str, Any]]
) -> dict[str, tuple[int, int]]:
    """Count, per core tool, the examples in which the student and the teacher invoked it."""
    usage: dict[str, tuple[int, int]] = {}
    for tool_key in tool_keys:
        student = teacher = 0
        for example in examples:
            outputs = example.get("Generated Outputs") or {}
            if any(_tool_key_matches(tool_key, name) for name in outputs.get("student_tools") or []):
                student += 1
            if any(_tool_key_matches(tool_key, name) for name in outputs.get("teacher_tools") or []):
                teacher += 1
        usage[tool_key] = (student, teacher)
    return usage


def tools_with_evidence(usage: Mapping[str, tuple[int, int]], example_count: int) -> list[str]:
    """Core tools invoked, by either side, in enough examples for a description edit to matter."""
    required = min(MIN_TOOL_EVIDENCE_EXAMPLES, example_count)
    return [tool for tool, (student, teacher) in usage.items() if max(student, teacher) >= required]


def modules_after_tool_choice(
    reflection_llm: Any,
    parent: Candidate,
    modules_to_edit: list[str],
    high_signal: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[str]:
    """Keep non-core modules, and only the core tools the reflector names.

    Tools that neither side invoked in enough examples are dropped before the
    reflector chooses: their descriptions cannot have caused those losses.
    Non-core modules come first: Execution Discipline and RULES_EXT govern the
    final message the judge scores, so they must not lose their offspring slot
    to a third rewording of a tool description.
    """
    non_core = [module for module in modules_to_edit if module not in CORE_TOOL_KEYS]
    core = [module for module in modules_to_edit if module in CORE_TOOL_KEYS]
    if not core:
        return non_core
    examples: Sequence[Mapping[str, Any]] = ()
    for module in core:
        if high_signal.get(module):
            examples = high_signal[module]
            break
    if not examples:
        print("Reflection skipped core-tool edits: no high-signal examples.")
        return non_core
    usage = tool_usage_in_examples(core, examples)
    evidenced = tools_with_evidence(usage, len(examples))
    skipped = [tool for tool in core if tool not in evidenced]
    if skipped:
        print(
            "Reflection skipped tool descriptions without invocation evidence: "
            + ", ".join(f"{tool} (student {usage[tool][0]}, teacher {usage[tool][1]})" for tool in skipped)
        )
    if not evidenced:
        return non_core
    descriptions = {
        module: parent.prompt_modules.get(module) or PROMPT_MODULE_DEFAULTS.get(module, "") for module in evidenced
    }
    raw = reflection_llm(
        tool_description_choice_prompt(
            tools=descriptions,
            example_blocks=_choice_example_blocks(examples),
            limit=_TOOL_CHOICE_LIMIT,
            usage={tool: usage[tool] for tool in evidenced},
            example_count=len(examples),
        )
    ).strip()
    chosen = parse_chosen_tool_keys(raw, evidenced)[:_TOOL_CHOICE_LIMIT]
    print("Reflector chose tool descriptions: " + (", ".join(chosen) if chosen else "(none)"))
    return non_core + chosen


def _format_child_delta(parent: Candidate, child: Candidate, module: str) -> str:
    """Return a unified diff for one child prompt module against its parent."""
    parent_text = parent.prompt_modules.get(module, "")
    child_text = child.prompt_modules.get(module, "")
    diff = unified_diff(
        parent_text.splitlines(keepends=True),
        child_text.splitlines(keepends=True),
        fromfile=f"parent/{parent.candidate_id}/{module}",
        tofile=f"child/{child.candidate_id}/{module}",
    )
    return "".join(diff) or "(no prompt changes)\n"


def _offspring_quotas(
    parents: Sequence[Candidate],
    *,
    scores: Mapping[str, float],
    offspring_count: int,
) -> list[tuple[Candidate, int]]:
    """Split ``offspring_count`` across parents. Extra slots go to higher scores.

    Ties break on candidate id so a resumed run assigns the same extras.
    """
    ranked = sorted(parents, key=lambda parent: (-scores[parent.candidate_id], parent.candidate_id))
    if not ranked or offspring_count <= 0:
        return [(parent, 0) for parent in ranked]
    base, remainder = divmod(offspring_count, len(ranked))
    return [(parent, base + (1 if index < remainder else 0)) for index, parent in enumerate(ranked)]


def make_children_for_generation(
    adapter: GleanAdapterBase,
    frontier_candidates: list[Candidate],
    frontier_evals: dict[str, GleanEvaluationBatch],
    reflection_llm: Any,
    offspring_count: int = 5,
    reflect_k: int | None = 8,
    max_attempts: int = 200,
    reflection_hamming_distance_k: int | None = None,
    children_by_root: dict[str, list[Candidate]] | None = None,
) -> list[Candidate]:
    """Create children by applying reflection-generated edits to one module.

    Offspring slots are split evenly across frontier parents. When the count
    does not divide evenly, the higher-scoring parents get the extra slots.
    ``children_by_root`` retains the children already reflected from a parent.
    Reusing those candidates is intentional: a root's traces and prompt are
    unchanged while it remains on the frontier, so reflecting on it again only
    spends another LLM call to rediscover mutations we already have.
    """
    children: list[Candidate] = []
    seen_child_programs: set[str] = set()

    def append_child(child: Candidate) -> bool:
        """Append a distinct child while there is room in this generation."""
        child_key = json.dumps(child.prompt_modules, sort_keys=True)
        if child_key in seen_child_programs or len(children) >= offspring_count:
            return False
        seen_child_programs.add(child_key)
        children.append(child)
        return True

    if not frontier_candidates or offspring_count <= 0 or max_attempts < 1:
        return []

    scores = {
        parent.candidate_id: adapter.get_screening_score(frontier_evals[parent.candidate_id])
        for parent in frontier_candidates
    }
    allocation = _offspring_quotas(frontier_candidates, scores=scores, offspring_count=offspring_count)
    print(f"Best quality parent: {allocation[0][0]}")
    print(
        "Offspring slots by parent: "
        + ", ".join(
            f"{parent.candidate_id}={quota} (score={scores[parent.candidate_id]:.4f})" for parent, quota in allocation
        )
    )

    def take_cached(parent: Candidate, quota: int) -> int:
        if children_by_root is None or quota <= 0:
            return 0
        taken = 0
        for child in children_by_root.get(parent.candidate_id, []):
            if taken >= quota or len(children) >= offspring_count:
                break
            if append_child(child):
                taken += 1
        return taken

    def reflect_parent(parent: Candidate, slot_budget: int) -> int:
        """Reflect once and keep up to ``slot_budget`` children. Returns how many were added."""
        if slot_budget <= 0 or len(children) >= offspring_count:
            return 0
        parent_eval = frontier_evals[parent.candidate_id]
        # Presence in the cache means this root has already had its one
        # reflection attempt for the current training slice. Record that before
        # calling the reflector so an empty/invalid response is cached too.
        cached_children = children_by_root.setdefault(parent.candidate_id, []) if children_by_root is not None else None

        modules_to_edit = pick_modules_to_edit(adapter, parent_eval)
        log_section(
            f"REFLECTION START parent={parent.candidate_id}",
            "modules_to_edit: " + (", ".join(modules_to_edit) if modules_to_edit else "(none)"),
        )

        high_signal = adapter.make_reflective_dataset(
            candidate=parent,
            eval_batch=frontier_evals[parent.candidate_id],
            components_to_update=modules_to_edit,
            k=reflect_k,
            error_hamming_distance_k=reflection_hamming_distance_k,
        )
        if any(module in CORE_TOOL_KEYS for module in modules_to_edit):
            modules_to_edit = modules_after_tool_choice(reflection_llm, parent, modules_to_edit, high_signal)
            log_section(
                f"REFLECTION TOOL CHOICE parent={parent.candidate_id}",
                "modules_to_edit: " + (", ".join(modules_to_edit) if modules_to_edit else "(none)"),
            )

        # Ask the reflection model for one to three small rewrite variants per module, then
        # fill this parent's slots round-robin across modules. Rewordings of one module's
        # edit score within judge noise of each other, so a generation spent on three
        # variants of a tool description and none of Execution Discipline learns nothing
        # about the module that governs the student's final message.
        modules_this_round = modules_to_edit[:slot_budget]
        variants_by_module: dict[str, list[str]] = {}
        diagnosis_by_module: dict[str, str] = {}
        for module in modules_this_round:
            proposed = adapter.propose_new_texts(
                reflection_llm=reflection_llm,
                candidate=parent,
                components_to_update=[module],
                reflective_examples=high_signal[module],
            )
            variants = proposed[0]
            diagnosis_by_module[module] = proposed[2] if len(proposed) > 2 else ""
            if not variants:
                print(f"Reflection produced no variants for module {module}")
                continue
            variants_by_module[module] = list(variants)

        added = 0
        for module, variant in _round_robin_variants(modules_this_round, variants_by_module):
            if added >= slot_budget or len(children) >= offspring_count:
                break
            child = apply_single_module_edit(parent, module, variant)
            if not append_child(child):
                continue
            added += 1
            if cached_children is not None and all(
                existing.prompt_modules != child.prompt_modules for existing in cached_children
            ):
                cached_children.append(child)
            log_section(
                f"CHILD PROPOSAL {child.candidate_id}",
                format_child_proposal_report(
                    parent_id=parent.candidate_id,
                    child_id=child.candidate_id,
                    module=module,
                    delta=_format_child_delta(parent, child, module),
                    justification=diagnosis_by_module.get(module, ""),
                ),
            )
        return added

    # A cached root is never reflected again. Take only that parent's quota, so
    # one parent's cached children cannot crowd the others out of the generation.
    shortfall = 0
    pending: list[tuple[Candidate, int]] = []
    for parent, quota in allocation:
        if quota <= 0:
            continue
        remaining = quota - take_cached(parent, quota)
        if remaining <= 0:
            continue
        already_reflected = children_by_root is not None and parent.candidate_id in children_by_root
        parent_eval = frontier_evals[parent.candidate_id]
        if already_reflected or not parent_eval.trajectories:
            if not already_reflected and children_by_root is not None:
                children_by_root.setdefault(parent.candidate_id, [])
            shortfall += remaining
            continue
        pending.append((parent, remaining))

    # Reflect lower-scoring parents first. Slots they do not fill, plus any
    # quota held by a parent that cannot reflect, go to the highest-scoring
    # parent that still can.
    reflections = 0
    for index, (parent, remaining) in enumerate(reversed(pending)):
        if reflections >= max_attempts:
            break
        budget = remaining + shortfall if index == len(pending) - 1 else remaining
        produced = reflect_parent(parent, budget)
        reflections += 1
        if index != len(pending) - 1:
            shortfall += max(0, remaining - produced)

    return children


def _round_robin_variants(modules: Sequence[str], variants_by_module: Mapping[str, Sequence[str]]):
    """Yield ``(module, variant)`` taking one variant per module per pass, in module order."""
    depth = max((len(variants) for variants in variants_by_module.values()), default=0)
    for index in range(depth):
        for module in modules:
            variants = variants_by_module.get(module) or ()
            if index < len(variants):
                yield module, variants[index]


def _eval_entry_ids(eval_batch: GleanEvaluationBatch) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for trajectory in getattr(eval_batch, "trajectories", None) or []:
        output = trajectory.get("output") if isinstance(trajectory, dict) else None
        entry_id = str((output or {}).get("entry_id") or "")
        if entry_id and entry_id not in seen:
            seen.add(entry_id)
            ordered.append(entry_id)
    return ordered


def _child_screen_score(
    adapter: GleanAdapterBase,
    parent_eval: GleanEvaluationBatch,
    screen_eval: GleanEvaluationBatch,
    *,
    use_high_signal_gate: bool,
) -> float:
    if not use_high_signal_gate:
        return adapter.get_screening_score(screen_eval)
    child_screen = getattr(adapter, "child_screen_score", None)
    if callable(child_screen):
        return float(child_screen(parent_eval, screen_eval))
    return adapter.high_signal_fix_rate(parent_eval, screen_eval)


def _select_screened_children(
    adapter: GleanAdapterBase,
    parent_eval: GleanEvaluationBatch,
    children: list[Candidate],
    screen_evals: list[GleanEvaluationBatch],
    *,
    use_high_signal_gate: bool,
    high_signal_screen_threshold: float = HIGH_SIGNAL_FIX_RATE_THRESHOLD,
) -> list[tuple[Candidate, GleanEvaluationBatch, float]]:
    """Keep every child eligible for GEPA's acceptance/selection stage."""
    selected: list[tuple[Candidate, GleanEvaluationBatch, float]] = []
    for child, screen_eval in zip(children, screen_evals, strict=True):
        child_score = _child_screen_score(adapter, parent_eval, screen_eval, use_high_signal_gate=use_high_signal_gate)
        if not use_high_signal_gate or child_score >= high_signal_screen_threshold:
            selected.append((child, screen_eval, child_score))
    return selected


class EvolutionaryProposer:
    """
    Proposer that generates reflection-driven mutations from Pareto-frontier
    candidates and returns every child that passes screening. For a high-signal
    screen, the proposal score is measured from a zero-fixes baseline, so a
    passing child reaches GEPA's full validation evaluation instead of being
    compared against the parent's incompatible overall score.

    Bridges between GEPA's dict[str, str] candidate format and the
    Glean AL adapter's Candidate type for reflection-driven mutation.
    """

    @staticmethod
    def get_display_iteration(state: GEPAState) -> int:
        """Number Glean iterations by completed full evaluations, not screens.

        The seed is the first full evaluation, so a fresh run's first child
        screening round is displayed as iteration 1. GEPA's ``state.i`` still
        counts proposal attempts internally for scheduling and stop conditions.
        """
        return state.num_full_ds_evals

    def __init__(
        self,
        logger: LoggerProtocol,
        trainset: list[DataInst] | DataLoader[DataId, DataInst],
        al_adapter: GleanAdapterBase,
        reflection_llm: Any,
        experiment_tracker: ExperimentTracker,
        model: str,
        module_specs: dict[str, ModuleSpec],
        global_token_cap: int,
        baseline_prompt_hash: str,
        offspring_count: int = 5,
        reflect_k: int | None = 8,
        evalset_policy: UnseenEvalSetPolicy | None = None,
        reflection_hamming_distance_k: int | None = None,
        children_cache_file: str | os.PathLike[str] | None = None,
        high_signal_screen_threshold: float = HIGH_SIGNAL_FIX_RATE_THRESHOLD,
    ):
        self.logger = logger
        self.trainset = ensure_loader(trainset)
        self.al_adapter = al_adapter
        self.reflection_llm = reflection_llm
        self.experiment_tracker = experiment_tracker
        self.evalset_policy = evalset_policy
        self.high_signal_screen_threshold = high_signal_screen_threshold

        self.model = model
        self.module_specs = module_specs
        self.global_token_cap = global_token_cap
        self.baseline_prompt_hash = baseline_prompt_hash

        self.offspring_count = offspring_count
        self.reflect_k = reflect_k
        self.reflection_hamming_distance_k = reflection_hamming_distance_k
        # A root's traces do not change while it stays on the frontier, so its children
        # are keyed by root id and reflected at most once per training slice.
        self._children_by_root: dict[str, list[Candidate]] = {}
        self._children_by_root_by_train_slice: dict[tuple[Any, ...], dict[str, list[Candidate]]] = {}
        # One record owns the generated child's eval IDs and screening result.
        # Both are scoped by training slice, root candidate, and child ID.
        self._child_cache_records_by_train_slice: dict[tuple[Any, ...], dict[str, dict[str, ChildCacheRecord]]] = {}
        self._root_screening_scores_by_train_slice: dict[tuple[Any, ...], dict[str, float]] = {}
        self.children_cache_file = Path(children_cache_file).expanduser() if children_cache_file else None
        self._load_children_cache()

        if isinstance(trainset, list):
            self._batch_data: list[dict[str, Any]] = cast(list[dict[str, Any]], trainset)
        else:
            self._batch_data = []
            try:
                for _, batch in self.trainset:  # type: ignore
                    self._batch_data = cast(list[dict[str, Any]], batch)
                    break
            except Exception:
                self._batch_data = []

    def _load_children_cache(self) -> None:
        """Restore generated children so a resumed run does not reflect them again."""
        if self.children_cache_file is None or not self.children_cache_file.exists():
            return
        try:
            data = json.loads(self.children_cache_file.read_text())
            if not isinstance(data, dict):
                raise ValueError("child cache root must be a JSON object")

            restored: dict[tuple[Any, ...], dict[str, list[Candidate]]] = {}
            restored_records: dict[tuple[Any, ...], dict[str, dict[str, ChildCacheRecord]]] = {}
            restored_root_scores: dict[tuple[Any, ...], dict[str, float]] = {}
            for entry in data.get("training_slices", []):
                train_ids = tuple(entry["train_ids"])
                roots: dict[str, list[Candidate]] = {}
                records_by_root: dict[str, dict[str, ChildCacheRecord]] = {}
                root_scores = {
                    str(root_id): float(score) for root_id, score in (entry.get("root_screening_scores") or {}).items()
                }
                for root_id, child_records in entry.get("roots", {}).items():
                    root_key = str(root_id)
                    roots[root_key] = []
                    records_by_root[root_key] = {}
                    for child_record in child_records:
                        child = self._to_candidate(child_record["prompt_modules"], parent_id=root_key)
                        roots[root_key].append(child)
                        records_by_root[root_key][child.candidate_id] = ChildCacheRecord(
                            eval_run_ids=list(child_record.get("eval_run_ids", [])),
                            screening_score=child_record.get("screening_score"),
                            screening_passed=child_record.get("screening_passed"),
                        )
                restored[train_ids] = roots
                restored_records[train_ids] = records_by_root
                restored_root_scores[train_ids] = root_scores
            self._children_by_root_by_train_slice = restored
            self._child_cache_records_by_train_slice = restored_records
            self._root_screening_scores_by_train_slice = restored_root_scores
            print(f"[Child cache] Loaded {sum(len(roots) for roots in restored.values())} root entries")
        except (OSError, TypeError, ValueError, KeyError, AttributeError) as exc:
            print(f"[Child cache] Failed to load {self.children_cache_file}: {exc}")
            self._children_by_root_by_train_slice = {}
            self._child_cache_records_by_train_slice = {}
            self._root_screening_scores_by_train_slice = {}

    def _save_children_cache(self) -> None:
        """Atomically persist generated children, including cached empty results."""
        if self.children_cache_file is None:
            return
        data = {
            "training_slices": [
                {
                    "train_ids": list(train_ids),
                    "root_screening_scores": self._root_screening_scores_by_train_slice.get(train_ids, {}),
                    "roots": {
                        root_id: [
                            {
                                "prompt_modules": child.prompt_modules,
                                "eval_run_ids": self._child_cache_records_by_train_slice.get(train_ids, {})
                                .get(root_id, {})
                                .get(child.candidate_id, ChildCacheRecord())
                                .eval_run_ids,
                                "screening_score": self._child_cache_records_by_train_slice.get(train_ids, {})
                                .get(root_id, {})
                                .get(child.candidate_id, ChildCacheRecord())
                                .screening_score,
                                "screening_passed": self._child_cache_records_by_train_slice.get(train_ids, {})
                                .get(root_id, {})
                                .get(child.candidate_id, ChildCacheRecord())
                                .screening_passed,
                            }
                            for child in children
                        ]
                        for root_id, children in roots.items()
                    },
                }
                for train_ids, roots in self._children_by_root_by_train_slice.items()
            ],
        }
        cache_dir = self.children_cache_file.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=cache_dir, delete=False) as temp_file:
                temp_path = temp_file.name
                json.dump(data, temp_file, indent=2)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, self.children_cache_file)
        except (OSError, TypeError, ValueError) as exc:
            print(f"[Child cache] Failed to save {self.children_cache_file}: {exc}")
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _cached_eval_run_ids(self, train_ids: tuple[Any, ...], child: Candidate) -> list[EvalRunIds]:
        """Return eval IDs stored alongside a child for this training slice."""
        return self._child_cache_record(train_ids, child).eval_run_ids

    def _record_eval_run_ids(
        self,
        train_ids: tuple[Any, ...],
        child: Candidate,
        eval_run_ids: list[EvalRunIds],
    ) -> None:
        """Associate screening eval IDs with the generated child that used them."""
        if not eval_run_ids:
            return
        self._child_cache_record(train_ids, child).eval_run_ids = list(eval_run_ids)

    def _child_cache_record(self, train_ids: tuple[Any, ...], child: Candidate) -> ChildCacheRecord:
        """Find or create the unified cache record for a generated child."""
        records_by_root = self._child_cache_records_by_train_slice.setdefault(train_ids, {})
        root_id = child.parent_id
        if root_id is not None:
            root_records = records_by_root.setdefault(root_id, {})
            return root_records.setdefault(child.candidate_id, ChildCacheRecord())

        # Candidate built without parent_id: child ids are unique within a slice.
        for root_records in records_by_root.values():
            record = root_records.get(child.candidate_id)
            if record is not None:
                return record
        return records_by_root.setdefault("__unknown_root__", {}).setdefault(child.candidate_id, ChildCacheRecord())

    def _cached_screening_scores(
        self,
        train_ids: tuple[Any, ...],
        children: list[Candidate],
        *,
        use_high_signal_gate: bool,
        high_signal_screen_threshold: float = HIGH_SIGNAL_FIX_RATE_THRESHOLD,
    ) -> list[tuple[float, bool]] | None:
        """Return complete cached screen results, or None when a child is missing one."""
        cached: list[tuple[float, bool]] = []
        for child in children:
            record = self._child_cache_record(train_ids, child)
            if record.screening_score is None:
                return None
            passed = not use_high_signal_gate or record.screening_score >= high_signal_screen_threshold
            cached.append((record.screening_score, passed))
        return cached

    def _slice_replay_is_fully_cached(
        self,
        train_ids: tuple[Any, ...],
        frontier_candidates: list[Candidate],
        children_by_root: dict[str, list[Candidate]],
        *,
        use_high_signal_gate: bool,
    ) -> bool:
        """Report whether this slice needs neither reflection nor screening.

        Every frontier root must already own cached children, since a root
        without them is reflected on and reflection reads the root's traces.
        The check covers every cached child rather than the subset a generation
        ends up screening, which can only err toward fetching traces.
        """
        if any(candidate.candidate_id not in children_by_root for candidate in frontier_candidates):
            return False
        children = [child for candidate in frontier_candidates for child in children_by_root[candidate.candidate_id]]
        if not children:
            return False
        return self._cached_screening_scores(train_ids, children, use_high_signal_gate=use_high_signal_gate) is not None

    def _record_screening_result(
        self,
        train_ids: tuple[Any, ...],
        child: Candidate,
        screening_score: float,
        screening_passed: bool,
    ) -> None:
        record = self._child_cache_record(train_ids, child)
        record.screening_score = screening_score
        record.screening_passed = screening_passed

    def _cached_root_screening_score(self, train_ids: tuple[Any, ...], root_id: str) -> float | None:
        """Return a cached root score only when this root has a child-cache entry."""
        if root_id not in self._children_by_root_by_train_slice.get(train_ids, {}):
            return None
        return self._root_screening_scores_by_train_slice.get(train_ids, {}).get(root_id)

    def _record_root_screening_score(self, train_ids: tuple[Any, ...], root_id: str, score: float) -> None:
        self._root_screening_scores_by_train_slice.setdefault(train_ids, {})[root_id] = score

    def _program_key(self, program: dict[str, str]) -> str:
        """Canonical prompt-module fingerprint used to detect duplicate candidates."""
        return json.dumps(self._to_candidate(program).prompt_modules, sort_keys=True)

    def _child_is_pending(
        self,
        train_ids: tuple[Any, ...],
        child: Candidate,
        existing_keys: set[str],
    ) -> bool:
        """True when this child still needs screening or has not yet entered the pool."""
        if self._program_key(child.prompt_modules) in existing_keys:
            return False
        # An over-budget child is never screened, so its cache record keeps a null
        # score. Reading that as unfinished work would pin the slice forever.
        if not within_prompt_budget(child):
            return False
        record = self._child_cache_record(train_ids, child)
        if record.screening_score is None:
            return True
        use_high_signal_gate = getattr(self.al_adapter, "supports_high_signal_eval", False)
        passed = not use_high_signal_gate or record.screening_score >= self.high_signal_screen_threshold
        return passed

    def _slice_is_exhausted(self, train_ids: tuple[Any, ...], existing_keys: set[str]) -> bool:
        """True when every cached child on this slice is failed or already in the pool."""
        roots = self._children_by_root_by_train_slice.get(train_ids)
        if roots is None:
            return False
        children = [child for group in roots.values() for child in group]
        if not children:
            return True
        return not any(self._child_is_pending(train_ids, child, existing_keys) for child in children)

    def _select_train_ids(
        self,
        existing_keys: set[str],
        *,
        iteration: int,
        attempt: int | None = None,
    ) -> list[Any] | None:
        """Pick the next training slice, retrying in-flight work instead of replaying finished slices."""
        if self.evalset_policy is None:
            return list(self.trainset.all_ids())

        for example_id in self.trainset.all_ids():
            train_ids = (example_id,)
            if train_ids in self._children_by_root_by_train_slice and not self._slice_is_exhausted(
                train_ids, existing_keys
            ):
                print(f"[Eval set schedule] reflection and offspring screening: reusing in-flight id {example_id}")
                return [example_id]

        ordered = list(self.trainset.all_ids())
        exhausted_prefix = 0
        while exhausted_prefix < len(ordered) and self._slice_is_exhausted((ordered[exhausted_prefix],), existing_keys):
            exhausted_prefix += 1
        self.evalset_policy.skip_consumed_prefix(self.trainset, exhausted_prefix)
        try:
            return self.evalset_policy.take_unseen(
                self.trainset,
                purpose="reflection and offspring screening",
                attempt=attempt,
            )
        except RuntimeError as exc:
            self.logger.log(f"Iteration {iteration}: Training eval schedule exhausted; stopping proposals ({exc})")
            return None

    def _to_candidate(self, program: dict[str, str], parent_id: str | None = None) -> Candidate:
        """Convert a GEPA program into adapter-editable Glean prompt modules."""
        prompt_modules = dict(program)
        for key in self.module_specs:
            default = PROMPT_MODULE_DEFAULTS.get(key)
            if default is not None:
                prompt_modules.setdefault(key, default)
        content = json.dumps(prompt_modules, sort_keys=True)
        cand_id = hashlib.md5(content.encode()).hexdigest()[:10]
        return Candidate(
            model=self.model,
            prompt_modules=prompt_modules,
            module_specs=self.module_specs,
            global_token_cap=self.global_token_cap,
            baseline_prompt_hash=self.baseline_prompt_hash,
            candidate_id=cand_id,
            parent_id=parent_id,
        )

    def propose(self, state: GEPAState) -> list[CandidateProposal]:
        i = self.get_display_iteration(state)

        front_mapping = state.get_pareto_front_mapping()
        frontier_idxs: set[int] = set()
        for prog_set in front_mapping.values():
            frontier_idxs.update(prog_set)
        frontier_idxs_sorted = sorted(frontier_idxs)

        if not frontier_idxs_sorted:
            self.logger.log(f"Iteration {i}: No frontier programs found")
            return []
        else:
            self.logger.log(f"Iteration {i}: Found the following frontier programs {frontier_idxs_sorted}")

        existing_keys = {self._program_key(program) for program in state.program_candidates}
        tried_slices: set[tuple[Any, ...]] = set()
        while True:
            # Reveal one training slice. Resume retries an in-flight slice; finished
            # slices whose passers are in the pool are skipped so a restart cannot
            # re-accept the same child.
            train_ids = self._select_train_ids(existing_keys, iteration=i, attempt=state.i)
            if train_ids is None:
                return []
            train_slice_key = tuple(train_ids)
            if train_slice_key in tried_slices:
                self.logger.log(
                    f"Iteration {i}: Training slice {train_slice_key} already attempted; stopping proposals"
                )
                return []
            tried_slices.add(train_slice_key)
            if self.evalset_policy is not None:
                trace_batch = self.trainset.fetch(train_ids)
            else:
                trace_batch = self._batch_data
            proposals = self._propose_for_train_slice(
                state=state,
                iteration=i,
                frontier_idxs_sorted=frontier_idxs_sorted,
                train_ids=train_ids,
                train_slice_key=train_slice_key,
                trace_batch=trace_batch,
                existing_keys=existing_keys,
            )
            if proposals is None:
                continue
            return proposals

    def _propose_for_train_slice(
        self,
        state: GEPAState,
        iteration: int,
        frontier_idxs_sorted: list[int],
        train_ids: list[Any],
        train_slice_key: tuple[Any, ...],
        trace_batch: list[Any],
        existing_keys: set[str],
    ) -> list[CandidateProposal] | None:
        i = iteration
        # Cached mutations are scoped to the training slice: a fresh slice reflects
        # again, repeat attempts within a slice do not.
        frontier_candidates: list[Candidate] = []
        prog_idx_to_cand_id: dict[int, str] = {}
        for idx in frontier_idxs_sorted:
            cand = self._to_candidate(state.program_candidates[idx])
            prog_idx_to_cand_id[idx] = cand.candidate_id
            frontier_candidates.append(cand)

        children_by_root = (
            self._children_by_root_by_train_slice.setdefault(train_slice_key, {})
            if self.evalset_policy is not None
            else self._children_by_root
        )
        use_high_signal_gate = getattr(self.al_adapter, "supports_high_signal_eval", False)
        skip_child_screening = _skips_child_screening(self.al_adapter)

        # A root's error examples exist to drive reflection and the high-signal
        # screen. When both are already cached for this slice, the generation is
        # only being replayed to reach full validation, so paying for per-entry
        # BigQuery and evalcli trace hydration would buy nothing.
        capture_root_traces = not self._slice_replay_is_fully_cached(
            train_slice_key,
            frontier_candidates,
            children_by_root,
            use_high_signal_gate=use_high_signal_gate,
        )
        if not capture_root_traces:
            self.logger.log(
                f"Iteration {i}: Replaying this slice from cached children and screens; "
                "skipping root error-example fetches"
            )

        # Score the frontier roots, reusing cached root scores when possible.
        frontier_evals: dict[str, GleanEvaluationBatch] = {}
        cached_frontier_evals: set[str] = set()
        uncached_frontier = []
        for cand in frontier_candidates:
            cached_root_score = self._cached_root_screening_score(train_slice_key, cand.candidate_id)
            if cached_root_score is None:
                uncached_frontier.append(cand)
            else:
                objective = getattr(self.al_adapter, "primary_objective", "screening_score")
                frontier_evals[cand.candidate_id] = GleanEvaluationBatch(
                    outputs=[], scores=[], trajectories=None, summary={objective: cached_root_score}
                )
                cached_frontier_evals.add(cand.candidate_id)
        if uncached_frontier:
            frontier_eval_batches = self.al_adapter.evaluate_many(
                trace_batch,
                [cand.prompt_modules for cand in uncached_frontier],
                capture_traces=capture_root_traces,
            )
            for cand, eval_batch in zip(uncached_frontier, frontier_eval_batches, strict=True):
                frontier_evals[cand.candidate_id] = eval_batch
                self._record_root_screening_score(
                    train_slice_key,
                    cand.candidate_id,
                    self.al_adapter.get_screening_score(eval_batch),
                )

        children = make_children_for_generation(
            adapter=self.al_adapter,
            frontier_candidates=frontier_candidates,
            frontier_evals=frontier_evals,
            reflection_llm=self.reflection_llm,
            offspring_count=self.offspring_count,
            reflect_k=self.reflect_k,
            reflection_hamming_distance_k=self.reflection_hamming_distance_k,
            children_by_root=children_by_root,
        )
        if self.evalset_policy is not None:
            self._save_children_cache()

        if not children:
            self.logger.log(f"Iteration {i}: Evolutionary proposer generated no children")
            return []

        # Over-budget children are recorded as failed screens so the cache does
        # not describe them as awaiting one.
        valid_children = []
        for child in children:
            if within_prompt_budget(child):
                valid_children.append(child)
            else:
                self._record_screening_result(train_slice_key, child, 0.0, False)
        if self.evalset_policy is not None and len(valid_children) != len(children):
            self._save_children_cache()
        if not valid_children:
            self.logger.log(f"Iteration {i}: No children passed budget check")
            return []

        # Screen children on the parent's high-signal failures; keep those whose
        # fix rate reaches high_signal_screen_threshold.
        best_parent_idx = max(frontier_idxs_sorted, key=lambda idx: state.program_full_scores_val_set[idx])
        best_parent_cand_id = prog_idx_to_cand_id[best_parent_idx]
        parent_eval = frontier_evals[best_parent_cand_id]
        screen_evals: list[GleanEvaluationBatch] = []
        skipped_screening = False
        cached_screening = self._cached_screening_scores(
            train_slice_key,
            valid_children,
            use_high_signal_gate=use_high_signal_gate,
            high_signal_screen_threshold=self.high_signal_screen_threshold,
        )
        if skip_child_screening and cached_screening is None:
            skipped_screening = True
            screen_scores = [1.0] * len(valid_children)
            screened_children = [(child, None, 1.0) for child in valid_children]
            for child in valid_children:
                self._record_screening_result(train_slice_key, child, 1.0, True)
            if self.evalset_policy is not None:
                self._save_children_cache()
        elif cached_screening is not None:
            screen_scores = [score for score, _passed in cached_screening]
            screened_children = [
                (child, None, score)
                for child, (score, passed) in zip(valid_children, cached_screening, strict=True)
                if passed
            ]
            print(
                f"[Child cache HIT] Reusing screening results for {len(valid_children)} children "
                f"on training slice {train_slice_key}"
            )
        else:
            if use_high_signal_gate:
                # A cached root score carries no traces, and neither does a root
                # evaluated while the screens looked cached. Either way the gate
                # needs the parent's failures, so fetch them now.
                if best_parent_cand_id in cached_frontier_evals or not parent_eval.trajectories:
                    best_parent_candidate = next(
                        candidate for candidate in frontier_candidates if candidate.candidate_id == best_parent_cand_id
                    )
                    parent_eval = self.al_adapter.evaluate(
                        trace_batch,
                        best_parent_candidate.prompt_modules,
                        capture_traces=True,
                    )
                    self._record_root_screening_score(
                        train_slice_key,
                        best_parent_cand_id,
                        self.al_adapter.get_screening_score(parent_eval),
                    )
                high_signal_batch = self.al_adapter.high_signal_batch(parent_eval)
                if not high_signal_batch:
                    self.logger.log(f"Iteration {i}: Parent has no high-signal failures; rejecting children")
                    return []
                high_signal_batch = self.al_adapter.prepare_high_signal_batch(high_signal_batch)
                if high_signal_batch is None:
                    message = f"Iteration {i}: Failed to prepare the high-signal eval set; stopping optimization"
                    self.logger.log(message)
                    raise RuntimeError(message)
                prepared_screen_batch = cast(list[ALDataInst], high_signal_batch)
                screen_evals = invoke_batch_evaluate(
                    self.al_adapter,
                    [
                        (
                            child.prompt_modules,
                            self.al_adapter.attach_cached_eval_run_ids(
                                prepared_screen_batch,
                                self._cached_eval_run_ids(train_slice_key, child),
                            ),
                        )
                        for child in valid_children
                    ],
                    capture_traces=True,
                )
            else:
                screen_evals = self.al_adapter.evaluate_many(
                    trace_batch,
                    [child.prompt_modules for child in valid_children],
                    capture_traces=False,
                )
            screen_scores = []
            screened_children = []
            for child, screen_eval in zip(valid_children, screen_evals, strict=True):
                score = _child_screen_score(
                    self.al_adapter, parent_eval, screen_eval, use_high_signal_gate=use_high_signal_gate
                )
                passed = not use_high_signal_gate or score >= self.high_signal_screen_threshold
                screen_scores.append(score)
                self._record_eval_run_ids(train_slice_key, child, getattr(screen_eval, "eval_run_ids", None) or [])
                self._record_screening_result(train_slice_key, child, score, passed)
                if passed:
                    screened_children.append((child, screen_eval, score))
            if self.evalset_policy is not None:
                self._save_children_cache()
        passed_ids = {child.candidate_id for child, _eval, _score in screened_children}
        screening_rows: list[tuple[str, float, bool, str]] = []
        if skipped_screening:
            screening_rows = [(child.candidate_id, 1.0, True, "skipped high-signal screen") for child in valid_children]
        elif cached_screening is not None:
            screening_rows = [
                (child.candidate_id, score, passed, "cached screening result")
                for child, (score, passed) in zip(valid_children, cached_screening, strict=True)
            ]
        elif screen_evals:
            for child, _screen_eval, score in zip(valid_children, screen_evals, screen_scores, strict=True):
                detail = f"score={score:.4f}"
                weighted_gate = bool(getattr(self.al_adapter, "screening_weights", None))
                if (
                    use_high_signal_gate
                    and not weighted_gate
                    and getattr(self.al_adapter, "screening_kind", None) != "correctness_floor"
                ):
                    detail = f"fix_rate={score:.3f}"
                screening_rows.append((child.candidate_id, score, child.candidate_id in passed_ids, detail))
        if skipped_screening:
            screen_mode = "skip"
        elif use_high_signal_gate:
            screen_mode = "fix-rate"
        else:
            screen_mode = "full-train"
        log_section(
            f"SCREENING iteration={i}",
            format_screening_report(
                mode=screen_mode,
                entry_ids=() if skipped_screening else _eval_entry_ids(parent_eval),
                rows=screening_rows,
            ),
        )

        if not screened_children:
            if use_high_signal_gate:
                best_score = max(screen_scores, default=0.0)
                if getattr(self.al_adapter, "screening_kind", None) == "correctness_floor":
                    self.logger.log(
                        f"Iteration {i}: No child reached the correctness floor "
                        f"{self.high_signal_screen_threshold:.0%} (best={best_score:.1%})"
                    )
                elif getattr(self.al_adapter, "screening_weights", None):
                    self.logger.log(
                        f"Iteration {i}: No child reached the screening gate "
                        f"{self.high_signal_screen_threshold:.0%} (best={best_score:.1%})"
                    )
                else:
                    self.logger.log(
                        f"Iteration {i}: No child fixed at least {self.high_signal_screen_threshold:.0%} "
                        f"of the high-signal failures (best={best_score:.1%})"
                    )
            else:
                self.logger.log(f"Iteration {i}: No children completed screening")
            return []

        pending_children = [
            (child, screen_eval, score)
            for child, screen_eval, score in screened_children
            if self._program_key(child.prompt_modules) not in existing_keys
        ]
        if not pending_children:
            self.logger.log(
                f"Iteration {i}: All {len(screened_children)} passing children are already in the "
                "candidate pool; trying the next training slice"
            )
            return None

        # A high-signal score is a fix rate over the parent's errors, not comparable
        # to the parent's overall score, so it is a zero-baseline gate: any positive
        # rate proceeds to full validation. Standard screens keep parent-vs-child.
        subsample_ids = train_ids
        parent_score = self.al_adapter.get_screening_score(parent_eval)
        best_child_score = max(score for _child, _eval, score in pending_children)
        child_score = best_child_score
        proposal_score_before = 0.0 if use_high_signal_gate or skip_child_screening else parent_score

        self.logger.log(
            f"Iteration {i}: Evolutionary proposer generated {len(children)} children, "
            f"{len(valid_children)} passed budget, {len(pending_children)} passed screening. "
            f"Best screening score={best_child_score:.3f}"
        )
        self.experiment_tracker.log_metrics(
            {
                "evolutionary_parent_eval_score": parent_score,
                "evolutionary_child_eval_score": child_score,
                "evolutionary_children_generated": len(children),
                "evolutionary_children_valid": len(valid_children),
                "evolutionary_children_screened_in": len(pending_children),
                "total_metric_calls": state.total_num_evals,
            },
            step=i,
        )

        screening_kind = getattr(self.al_adapter, "screening_kind", None)
        if skip_child_screening:
            screening_kind_meta = screening_kind or "none"
            screening_threshold_meta = None
            proposal_tag = "evolutionary"
        elif use_high_signal_gate:
            screening_kind_meta = screening_kind or "high_signal_fix_rate"
            screening_threshold_meta = self.high_signal_screen_threshold
            proposal_tag = "evolutionary_high_signal"
        else:
            screening_kind_meta = "screening_score"
            screening_threshold_meta = None
            proposal_tag = "evolutionary"
        # Children come from every frontier parent that received offspring slots, so each
        # proposal names its own root; falling back to the best parent only for a child whose
        # root is no longer in the pool.
        cand_id_to_prog_idx = {cand_id: idx for idx, cand_id in prog_idx_to_cand_id.items()}
        return [
            CandidateProposal(
                candidate=child.prompt_modules,
                parent_program_ids=[cand_id_to_prog_idx.get(child.parent_id or "", best_parent_idx)],
                subsample_indices=subsample_ids,
                subsample_scores_before=[proposal_score_before],
                subsample_scores_after=[screen_score],
                tag=proposal_tag,
                metadata={
                    "screening_kind": screening_kind_meta,
                    "screening_score": screen_score,
                    "screening_threshold": screening_threshold_meta,
                },
            )
            for child, _screen_eval, screen_score in pending_children
        ]
