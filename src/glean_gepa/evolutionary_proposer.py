# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass, field
from difflib import unified_diff
from pathlib import Path
from typing import Any, cast

from gepa.core.adapter import DataInst, invoke_batch_evaluate
from gepa.core.callbacks import GEPACallback
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
from glean_gepa.prompt import high_signal_core_tool_keys
from glean_gepa.prompt_constants import CORE_TOOL_KEYS, PROMPT_MODULE_DEFAULTS
from glean_gepa.run_log import format_child_proposal_report, format_screening_report, log_section
from glean_gepa.utils import apply_single_module_edit

CHILDREN_CACHE_SCHEMA_VERSION = 6
HIGH_SIGNAL_FIX_RATE_THRESHOLD = 1 / 3


@dataclass
class ChildCacheRecord:
    """Persisted state for one generated child and its focused screen."""

    eval_run_ids: list[EvalRunIds] = field(default_factory=list)
    screening_score: float | None = None
    screening_passed: bool | None = None


# TODO(Cathy): pick modules based on holistic performance of the eval
def pick_modules_to_edit(
    adapter: GleanAdapterBase,
    eval_batch: GleanEvaluationBatch | None = None,
) -> list[str]:
    """Return modules the proposer should rewrite this generation.

    Non-core modules listed in ``editable_modules`` (including ``RULES_EXT``) are
    always rewritten. Core-tool descriptions are eligible only when listed, and
    the proposer only rewrites those that appear in the high-signal first-tool
    mismatch set.
    """
    eligible = list(adapter.editable_modules)
    modules = [module for module in eligible if module not in CORE_TOOL_KEYS]
    extra = [
        key
        for key in high_signal_core_tool_keys(eval_batch.trajectories if eval_batch is not None else None)
        if key in eligible
    ]
    modules.extend(key for key in extra if key not in modules)
    return modules


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

    # Pick a main parent using the concrete adapter's primary objective.
    best_quality_parent = max(
        frontier_candidates,
        key=lambda c: adapter.get_screening_score(frontier_evals[c.candidate_id]),
    )
    print(f"Best quality parent: {best_quality_parent}")

    # A cached root is never reflected again. Reuse cached children first, in
    # quality order, before generating mutations for roots we have not seen.
    # This also makes the cache useful when a parent disappears and later
    # returns to the Pareto frontier.
    if children_by_root is not None:
        ordered_roots = [best_quality_parent] + [
            parent for parent in frontier_candidates if parent.candidate_id != best_quality_parent.candidate_id
        ]
        for parent in ordered_roots:
            for child in children_by_root.get(parent.candidate_id, []):
                append_child(child)
            if len(children) >= offspring_count:
                return children

    attempts = 0
    while len(children) < offspring_count and attempts < max_attempts:
        attempts += 1
        uncached_roots = [
            parent
            for parent in frontier_candidates
            if children_by_root is None or parent.candidate_id not in children_by_root
        ]
        if not uncached_roots:
            break
        parent = (
            best_quality_parent
            if best_quality_parent in uncached_roots and random.random() < 0.7
            else random.choice(uncached_roots)
        )
        parent_eval = frontier_evals[parent.candidate_id]
        if not parent_eval.trajectories:
            # Need traces to reflect; skip mutation if missing.
            if children_by_root is not None:
                children_by_root.setdefault(parent.candidate_id, [])
            continue

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

        # Ask the reflection model for one to three small rewrite variants.
        for module in modules_to_edit:
            proposed = adapter.propose_new_texts(
                reflection_llm=reflection_llm,
                candidate=parent,
                components_to_update=[module],
                reflective_examples=high_signal[module],
            )
            variants = proposed[0]
            diagnosis = proposed[2] if len(proposed) > 2 else ""
            if not variants:
                print(f"Reflection produced no variants for module {module}")
                continue

            for variant in variants[: max(1, offspring_count - len(children))]:
                child = apply_single_module_edit(parent, module, variant)
                if cached_children is not None:
                    if all(existing.prompt_modules != child.prompt_modules for existing in cached_children):
                        cached_children.append(child)
                if append_child(child):
                    log_section(
                        f"CHILD PROPOSAL {child.candidate_id}",
                        format_child_proposal_report(
                            parent_id=parent.candidate_id,
                            child_id=child.candidate_id,
                            module=module,
                            delta=_format_child_delta(parent, child, module),
                            justification=diagnosis,
                        ),
                    )
                if len(children) >= offspring_count:
                    break

    return children


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
        child_score = (
            adapter.high_signal_fix_rate(parent_eval, screen_eval)
            if use_high_signal_gate
            else adapter.get_screening_score(screen_eval)
        )
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
        # Candidate config (for converting dict[str,str] <-> Candidate)
        model: str,
        module_specs: dict[str, ModuleSpec],
        global_token_cap: int,
        baseline_prompt_hash: str,
        # Evolutionary hyperparameters
        offspring_count: int = 5,
        reflect_k: int | None = 8,
        callbacks: list[GEPACallback] | None = None,
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
        self.callbacks = callbacks
        self.evalset_policy = evalset_policy
        self.high_signal_screen_threshold = high_signal_screen_threshold

        # Candidate conversion config
        self.model = model
        self.module_specs = module_specs
        self.global_token_cap = global_token_cap
        self.baseline_prompt_hash = baseline_prompt_hash

        # Evolutionary hyperparameters
        self.offspring_count = offspring_count
        self.reflect_k = reflect_k
        self.reflection_hamming_distance_k = reflection_hamming_distance_k
        # Reflection depends on a root candidate and its fixed evaluation
        # traces. Keep its proposed children keyed by that stable root id so a
        # root revisited in a later iteration does not trigger reflection again.
        self._children_by_root: dict[str, list[Candidate]] = {}
        # Incremental training slices need fresh reflection, but each root should
        # still be reflected at most once within the same slice.
        self._children_by_root_by_train_slice: dict[tuple[Any, ...], dict[str, list[Candidate]]] = {}
        # One record owns the generated child's eval IDs and screening result.
        # Both are scoped by training slice, root candidate, and child ID.
        self._child_cache_records_by_train_slice: dict[tuple[Any, ...], dict[str, dict[str, ChildCacheRecord]]] = {}
        self._root_screening_scores_by_train_slice: dict[tuple[Any, ...], dict[str, float]] = {}
        self.children_cache_file = Path(children_cache_file).expanduser() if children_cache_file else None
        self._load_children_cache()

        # Store batch data for eval set (trainset is just metadata for eval set runs)
        # Extract first batch from trainset
        if isinstance(trainset, list):
            self._batch_data: list[dict[str, Any]] = cast(list[dict[str, Any]], trainset)
        else:
            # Get first batch from loader
            self._batch_data = []
            try:
                for _, batch in self.trainset:  # type: ignore
                    self._batch_data = cast(list[dict[str, Any]], batch)
                    break
            except Exception:
                # Fallback to empty batch
                self._batch_data = []

    def _load_children_cache(self) -> None:
        """Restore generated children so a resumed run does not reflect them again."""
        if self.children_cache_file is None or not self.children_cache_file.exists():
            return
        try:
            data = json.loads(self.children_cache_file.read_text())
            if not isinstance(data, dict):
                raise ValueError("child cache root must be a JSON object")
            schema_version = data.get("schema_version")
            # Accept every released schema up to the current one. An explicit
            # allow-list silently discarded caches whenever a version was bumped
            # without being added to it, which forces children to be re-proposed.
            # Only v1 needs a shape transform; later versions share a record shape,
            # and anything unreadable is caught below and degrades to an empty cache.
            if not isinstance(schema_version, int) or not 1 <= schema_version <= CHILDREN_CACHE_SCHEMA_VERSION:
                print(f"[Child cache] Ignoring unsupported cache schema in {self.children_cache_file}")
                return

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
                    for raw_child in child_records:
                        child_record = (
                            {"prompt_modules": raw_child, "eval_run_ids": []} if schema_version == 1 else raw_child
                        )
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
            "schema_version": CHILDREN_CACHE_SCHEMA_VERSION,
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

        # Compatibility for callers that construct Candidate objects directly
        # without parent_id (the public generation helper historically allowed
        # that). Existing child IDs are unique within a training slice in the
        # normal proposer path.
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

        # 1. Get frontier program indices from Pareto front
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
            # 2. Reveal one training slice for this generation. Resume retries an
            # in-flight slice; finished slices whose passers are already in the
            # pool are skipped so a restart cannot re-accept the same child.
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
        # 3. Convert frontier programs to Candidate objects. Cached mutations
        # are scoped to the current training slice, so a fresh slice prompts a
        # new reflection while duplicate attempts within that slice are avoided.
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

        # 4. Score the frontier roots, reusing cached root scores when possible.
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

        # 5. Generate children using evolutionary strategies.
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

        # 6. Filter children by prompt budget
        valid_children = [c for c in children if within_prompt_budget(c)]
        if not valid_children:
            self.logger.log(f"Iteration {i}: No children passed budget check")
            return []

        # 6. Screen children on the parent's high-signal failures first.
        # Keep a child whose fix rate is at least one-third
        # (or high_signal_screen_threshold).
        best_parent_idx = max(frontier_idxs_sorted, key=lambda idx: state.program_full_scores_val_set[idx])
        best_parent_cand_id = prog_idx_to_cand_id[best_parent_idx]
        parent_eval = frontier_evals[best_parent_cand_id]
        screen_evals: list[GleanEvaluationBatch] = []
        cached_screening = self._cached_screening_scores(
            train_slice_key,
            valid_children,
            use_high_signal_gate=use_high_signal_gate,
            high_signal_screen_threshold=self.high_signal_screen_threshold,
        )
        if cached_screening is not None:
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
                score = (
                    self.al_adapter.high_signal_fix_rate(parent_eval, screen_eval)
                    if use_high_signal_gate
                    else self.al_adapter.get_screening_score(screen_eval)
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
        if cached_screening is not None:
            screening_rows = [
                (child.candidate_id, score, passed, "cached screening result")
                for child, (score, passed) in zip(valid_children, cached_screening, strict=True)
            ]
        elif screen_evals:
            for child, _screen_eval, score in zip(valid_children, screen_evals, screen_scores, strict=True):
                detail = f"score={score:.4f}"
                if use_high_signal_gate:
                    detail = f"fix_rate={score:.3f}"
                screening_rows.append((child.candidate_id, score, child.candidate_id in passed_ids, detail))
        log_section(
            f"SCREENING iteration={i}",
            format_screening_report(
                mode="fix-rate" if use_high_signal_gate else "full-train",
                entry_ids=_eval_entry_ids(parent_eval),
                rows=screening_rows,
            ),
        )

        if not screened_children:
            if use_high_signal_gate:
                best_fix_rate = max(screen_scores, default=0.0)
                self.logger.log(
                    f"Iteration {i}: No child fixed at least {self.high_signal_screen_threshold:.0%} "
                    f"of the high-signal failures (best={best_fix_rate:.1%})"
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

        # 7. The engine runs selected children on the full eval set. A
        # high-signal score is a rate over the parent's errors, so it is not
        # comparable to the parent's overall screening score. Treat it as a
        # zero-baseline gate: any positive fix rate is an improvement and the
        # child can proceed to full validation. Standard screens retain their
        # parent-vs-child score comparison.
        subsample_ids = train_ids
        parent_score = self.al_adapter.get_screening_score(parent_eval)
        best_child_score = max(score for _child, _eval, score in pending_children)
        child_score = best_child_score
        proposal_score_before = 0.0 if use_high_signal_gate else parent_score

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

        return [
            CandidateProposal(
                candidate=child.prompt_modules,
                parent_program_ids=[best_parent_idx],
                subsample_indices=subsample_ids,
                subsample_scores_before=[proposal_score_before],
                subsample_scores_after=[screen_score],
                tag="evolutionary_high_signal" if use_high_signal_gate else "evolutionary",
                metadata={
                    "screening_kind": "high_signal_fix_rate" if use_high_signal_gate else "screening_score",
                    "screening_score": screen_score,
                    "screening_threshold": HIGH_SIGNAL_FIX_RATE_THRESHOLD if use_high_signal_gate else None,
                },
            )
            for child, _screen_eval, screen_score in pending_children
        ]
