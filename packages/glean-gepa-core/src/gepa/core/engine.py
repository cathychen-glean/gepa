# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from collections.abc import Sequence
import json
import os
import traceback
from typing import Any, Generic

from gepa.core.adapter import DataInst
from gepa.core.adapter import GEPAAdapter
from gepa.core.adapter import invoke_batch_evaluate
from gepa.core.adapter import RolloutOutput
from gepa.core.adapter import Trajectory
from gepa.core.data_loader import DataId
from gepa.core.data_loader import DataLoader
from gepa.core.data_loader import ensure_loader
from gepa.core.state import EvaluationCache
from gepa.core.state import FrontierType
from gepa.core.state import GEPAState
from gepa.core.state import initialize_gepa_state
from gepa.core.state import new_iteration_id
from gepa.core.state import SEED_ITERATION_ID
from gepa.core.state import TRAINSET_CACHE_SPLIT
from gepa.core.state import VALSET_CACHE_SPLIT
from gepa.core.state import ValsetEvaluation
from gepa.gepa_utils import json_default
from gepa.gepa_utils import try_json_serialize
from gepa.logging.experiment_tracker import ExperimentTracker
from gepa.logging.logger import LoggerProtocol
from gepa.logging.utils import log_detailed_metrics_after_discovering_new_program
from gepa.proposer.base import CandidateProposal
from gepa.strategies.eval_policy import EvaluationPolicy
from gepa.strategies.eval_policy import FullEvaluationPolicy
from gepa.utils import StopperProtocol

# Import tqdm for progress bar functionality
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _record_proposal_evals(trace_entry: dict, proposal: 'CandidateProposal') -> None:
    """Capture evaluation side_info from a proposal into the iteration trace.

    This records the subsample evaluation data (scores, outputs, trajectories/
    side_info) for both the parent and the child candidate so that agents
    reading ``gepa_state.json`` have the feedback needed to propose better
    candidates.
    """
    if proposal.eval_before is not None:
        eb = proposal.eval_before
        trace_entry['eval_before'] = {
            'scores': eb.scores,
            'outputs': try_json_serialize(eb.outputs),
            'objective_scores': eb.objective_scores,
            'trajectories': try_json_serialize(eb.trajectories),
        }
    if proposal.eval_after is not None:
        ea = proposal.eval_after
        trace_entry['eval_after'] = {
            'scores': ea.scores,
            'outputs': try_json_serialize(ea.outputs),
            'objective_scores': ea.objective_scores,
            'trajectories': try_json_serialize(ea.trajectories),
        }


class GEPAEngine(Generic[DataId, DataInst, Trajectory, RolloutOutput]):
    """Orchestrates the optimization loop using pluggable candidate proposers.

    ``valset_cache_split`` is derived here and is the single authority for the evaluation-cache
    namespace used on valset reads and writes. If this engine's valset loader is the same object
    as the proposer's ``trainset`` loader, minibatch and valset share ``TRAINSET_CACHE_SPLIT``;
    otherwise valset uses ``VALSET_CACHE_SPLIT``.
    """

    def __init__(
        self,
        adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput],
        run_dir: str | None,
        valset: list[DataInst] | DataLoader[DataId, DataInst] | None,
        seed_candidate: dict[str, str],
        # Controls
        perfect_score: float | None,
        seed: int,
        # Strategies and helpers
        reflective_proposer: Any,
        frontier_type: FrontierType,
        # Logging
        logger: LoggerProtocol,
        experiment_tracker: ExperimentTracker,
        # Optional parameters
        track_best_outputs: bool = False,
        display_progress_bar: bool = False,
        raise_on_exception: bool = True,
        use_cloudpickle: bool = False,
        write_agent_state: bool = False,
        # Budget and Stop Condition
        stop_callback: StopperProtocol | None = None,
        val_evaluation_policy: EvaluationPolicy[DataId, DataInst] | None = None,
        # Evaluation caching (stored in state, passed here for initialization)
        evaluation_cache: EvaluationCache[RolloutOutput, DataId] | None = None,
    ):
        self.logger = logger
        self.run_dir = run_dir

        # Graceful stopping mechanism
        self._stop_requested = False

        # Set up stopping mechanism
        self.stop_callback = stop_callback
        self.adapter = adapter

        # Store cache reference for state initialization (actual cache lives in GEPAState)
        self._initial_evaluation_cache = evaluation_cache

        def evaluator(
            batch: list[DataInst], program: dict[str, str]
        ) -> tuple[list[RolloutOutput], list[float], Sequence[dict[str, float]] | None]:
            eval_result = adapter.evaluate(batch, program, capture_traces=False)
            return eval_result.outputs, eval_result.scores, eval_result.objective_scores

        self.evaluator = evaluator

        self.valset = ensure_loader(valset) if valset is not None else None
        self.valset_cache_split = (
            TRAINSET_CACHE_SPLIT
            if self.valset is not None and self.valset is getattr(reflective_proposer, 'trainset', None)
            else VALSET_CACHE_SPLIT
        )
        self.seed_candidate = seed_candidate

        self.perfect_score = perfect_score
        self.seed = seed
        self.experiment_tracker = experiment_tracker

        self.reflective_proposer = reflective_proposer
        self.frontier_type: FrontierType = frontier_type

        self.track_best_outputs = track_best_outputs
        self.display_progress_bar = display_progress_bar
        self.use_cloudpickle = use_cloudpickle
        self.write_agent_state = write_agent_state

        self.raise_on_exception = raise_on_exception
        self.val_evaluation_policy: EvaluationPolicy[DataId, DataInst] = (
            val_evaluation_policy if val_evaluation_policy is not None else FullEvaluationPolicy()
        )

    def _sync_adapter_state_to_state(self, state: GEPAState) -> None:
        """Snapshot adapter state into GEPAState before saving.

        No-op if the adapter does not implement ``get_adapter_state``.
        Makes a shallow copy to avoid mutations between snapshot and save.
        """
        getter = getattr(self.adapter, 'get_adapter_state', None)
        if getter is not None:
            state.adapter_state = dict(getter())

    def _sync_state_to_adapter(self, state: GEPAState) -> None:
        """Restore persisted adapter state into the adapter after loading.

        No-op if the adapter does not implement ``set_adapter_state``.
        """
        setter = getattr(self.adapter, 'set_adapter_state', None)
        if setter is not None:
            setter(state.adapter_state)

    def _next_display_iteration(self, state: GEPAState[RolloutOutput, DataId]) -> int:
        """Return the user-facing number for the next proposal attempt.

        ``state.i`` is GEPA's internal attempt counter: batch samplers and
        proposal-limit stoppers rely on it advancing for every attempt. A
        proposer may instead expose ``get_display_iteration`` when its
        external notion of an iteration advances on a different event (for
        example, only after a full validation evaluation).
        """
        get_display_iteration = getattr(self.reflective_proposer, 'get_display_iteration', None)
        if get_display_iteration is not None:
            return int(get_display_iteration(state))
        return state.i + 1

    def _display_iteration(self, state: GEPAState[RolloutOutput, DataId]) -> int:
        """Return the stable user-facing iteration stamped on this attempt."""
        if state.full_program_trace:
            display_iteration = state.full_program_trace[-1].get('display_iteration')
            if display_iteration is not None:
                return int(display_iteration)
        return self._next_display_iteration(state)

    def _write_agent_iteration_files(
        self,
        iteration_id: str,
        valset_evaluation: ValsetEvaluation[RolloutOutput, DataId],
    ) -> None:
        """Write per-val_id outputs/trajectories for an iteration's full valset eval.

        Called only when ``write_agent_state`` is enabled and ``run_dir`` is
        set. Produces ``iterations/<iter_id>/outputs/<val_id>.json`` and, when
        the valset evaluation captured them,
        ``iterations/<iter_id>/trajectories/<val_id>.json``. The iteration id is
        ``SEED_ITERATION_ID`` for the seed and the proposal's random
        ``iteration_id`` for accepted loop proposals (the same anchor
        ``GEPAState._save_agent_directory`` uses).
        """
        if not self.write_agent_state or self.run_dir is None:
            return
        base = os.path.join(self.run_dir, 'iterations', iteration_id)
        outputs = valset_evaluation.outputs_by_val_id or {}
        if outputs:
            out_dir = os.path.join(base, 'outputs')
            os.makedirs(out_dir, exist_ok=True)
            for val_id, output in outputs.items():
                path = os.path.join(out_dir, f'{val_id}.json')
                with open(path, 'w') as f:
                    json.dump(try_json_serialize(output), f, indent=2, default=json_default)
        trajectories = valset_evaluation.trajectories_by_val_id or {}
        if trajectories:
            traj_dir = os.path.join(base, 'trajectories')
            os.makedirs(traj_dir, exist_ok=True)
            for val_id, traj in trajectories.items():
                path = os.path.join(traj_dir, f'{val_id}.json')
                with open(path, 'w') as f:
                    json.dump(try_json_serialize(traj), f, indent=2, default=json_default)

    def _evaluate_programs_on_valset(
        self,
        programs: list[dict[str, str]],
        state: GEPAState[RolloutOutput, DataId],
    ) -> list[tuple[ValsetEvaluation[RolloutOutput, DataId], int]]:
        """Evaluate candidates on the valset, read-only, via ``adapter.batch_evaluate``.

        Cache-miss examples across all candidates go out in one batched call, so
        parallelism (if any) is the adapter's. Honors the eval cache and
        ``val_evaluation_policy``. Returns each evaluation with its metric-call
        count (cache misses); the budget is incremented later in
        :meth:`_add_evaluated_program`, not here.
        """
        valset = self.valset
        assert valset is not None
        cache = state.evaluation_cache

        # When write_agent_state is on, bypass the cache so we can capture
        # trajectories (which the cache does not store).
        if self.write_agent_state:
            traced_results: list[tuple[ValsetEvaluation[RolloutOutput, DataId], int]] = []
            for program in programs:
                val_ids = list(self.val_evaluation_policy.get_eval_batch(valset, state))
                eval_result = self.adapter.evaluate(valset.fetch(val_ids), program, capture_traces=True)
                outputs_by_val_idx = dict(zip(val_ids, eval_result.outputs, strict=False))
                scores_by_val_idx = dict(zip(val_ids, eval_result.scores, strict=False))
                objective_by_val_idx = (
                    dict(zip(val_ids, eval_result.objective_scores, strict=False))
                    if eval_result.objective_scores is not None
                    else None
                )
                trajectories_by_val_idx = (
                    dict(zip(val_ids, eval_result.trajectories, strict=False))
                    if eval_result.trajectories is not None
                    else None
                )
                traced_results.append(
                    (
                        ValsetEvaluation(
                            outputs_by_val_id=outputs_by_val_idx,
                            scores_by_val_id=scores_by_val_idx,
                            objective_scores_by_val_id=objective_by_val_idx,
                            trajectories_by_val_id=trajectories_by_val_idx,
                        ),
                        len(val_ids),
                    )
                )
            return traced_results

        # 1) Per program: split the requested val examples into cached vs. to-evaluate.
        cached_per: list[dict[Any, Any]] = []
        todo_per: list[list[Any]] = []
        for program in programs:
            val_ids = list(self.val_evaluation_policy.get_eval_batch(valset, state))
            if cache is not None:
                cached, uncached = cache.get_batch(program, val_ids, split=self.valset_cache_split)
            else:
                cached, uncached = {}, val_ids
            cached_per.append(cached)
            todo_per.append(uncached)

        # 2) One adapter call over every (program, cache-miss examples) pair.
        eval_idxs = [i for i, todo in enumerate(todo_per) if todo]
        items = [(programs[i], valset.fetch(todo_per[i])) for i in eval_idxs]
        fresh = invoke_batch_evaluate(self.adapter, items, capture_traces=False) if items else []
        fresh_by_idx = dict(zip(eval_idxs, fresh, strict=True))

        # 3) Merge cached + fresh per program, repopulating the cache.
        results: list[tuple[ValsetEvaluation[RolloutOutput, DataId], int]] = []
        for i, program in enumerate(programs):
            outputs_by: dict[Any, Any] = {}
            scores_by: dict[Any, float] = {}
            objective_by: dict[Any, Any] | None = None
            for eid, entry in cached_per[i].items():
                outputs_by[eid] = entry.output
                scores_by[eid] = entry.score
                if entry.objective_scores is not None:
                    objective_by = objective_by or {}
                    objective_by[eid] = entry.objective_scores

            uncached = todo_per[i]
            if uncached:
                eb = fresh_by_idx[i]
                obj = list(eb.objective_scores) if eb.objective_scores else None
                for j, eid in enumerate(uncached):
                    outputs_by[eid] = eb.outputs[j]
                    scores_by[eid] = eb.scores[j]
                    if obj is not None:
                        objective_by = objective_by or {}
                        objective_by[eid] = obj[j]
                if cache is not None:
                    cache.put_batch(program, uncached, eb.outputs, eb.scores, obj, split=self.valset_cache_split)

            results.append(
                (
                    ValsetEvaluation(
                        outputs_by_val_id=outputs_by,
                        scores_by_val_id=scores_by,
                        objective_scores_by_val_id=objective_by,
                    ),
                    len(uncached),
                )
            )
        return results

    def _add_evaluated_program(
        self,
        new_program: dict[str, str],
        state: GEPAState[RolloutOutput, DataId],
        parent_program_idx: list[int],
        valset_evaluation: ValsetEvaluation[RolloutOutput, DataId],
        num_actual_evals: int,
        iteration_id: str | None = None,
    ) -> tuple[int, int]:
        """Add an already-evaluated candidate to the pool. Must run sequentially.

        Snapshots the discovery budget before incrementing so candidates,
        processed in order, record the same ``num_metric_calls_by_discovery``
        as the serial path.

        ``iteration_id`` is the on-disk anchor for the proposal being processed.
        It defaults to the current trace slot's id.
        """
        if iteration_id is None:
            iteration_id = state.current_iteration_id()
        num_metric_calls_by_discovery = state.total_num_evals
        state.increment_evals(num_actual_evals)
        state.num_full_ds_evals += 1

        # Snapshot Pareto front before update
        front_before = state.get_pareto_front_mapping()
        candidates_before: set[int] = set()
        for program_set in front_before.values():
            candidates_before.update(program_set)

        new_program_idx = state.update_state_with_new_program(
            parent_program_idx=parent_program_idx,
            new_program=new_program,
            valset_evaluation=valset_evaluation,
            run_dir=self.run_dir,
            num_metric_calls_by_discovery_of_new_program=num_metric_calls_by_discovery,
            iteration_id=iteration_id,
        )

        # ``iteration_id`` is the on-disk anchor (the same one
        # ``GEPAState._save_agent_directory`` writes the proposal dir under).
        self._write_agent_iteration_files(iteration_id, valset_evaluation)

        # Compute the best program immediately after the state update.
        # to ensure is_best_program reflects the updated Pareto front
        valset_score = self.val_evaluation_policy.get_valset_score(new_program_idx, state)
        linear_pareto_front_program_idx = self.val_evaluation_policy.get_best_program(state)
        is_best_program = new_program_idx == linear_pareto_front_program_idx
        frontier_label = f'{state.frontier_type} frontier'
        self.logger.log(
            f'Iteration {self._display_iteration(state)}: Full validation score for child candidate '
            f'{new_program_idx} (parents {parent_program_idx}) is {valset_score}; using this held-out '
            f'validation result to update the {frontier_label}.'
        )

        # Snapshot Pareto front after update.
        front_after = state.get_pareto_front_mapping()
        candidates_after: set[int] = set()
        for program_set in front_after.values():
            candidates_after.update(program_set)

        new_front = sorted(candidates_after)
        displaced_candidates = sorted(candidates_before - candidates_after)

        self.logger.log(
            f'Iteration {self._display_iteration(state)}: {frontier_label} after validating candidate '
            f'{new_program_idx}: candidates {new_front}; displaced {displaced_candidates or "none"}.'
        )

        iteration = self._display_iteration(state)
        state.full_program_trace[-1]['new_program_idx'] = new_program_idx
        state.full_program_trace[-1].setdefault('new_program_indices', []).append(new_program_idx)
        state.full_program_trace[-1]['evaluated_val_indices'] = sorted(valset_evaluation.scores_by_val_id.keys())

        if is_best_program:
            self.logger.log(f'Iteration {iteration}: Found a better program on the valset with score {valset_score}.')

        valset = self.valset
        assert valset is not None

        log_detailed_metrics_after_discovering_new_program(
            logger=self.logger,
            gepa_state=state,
            new_program_idx=new_program_idx,
            valset_evaluation=valset_evaluation,
            objective_scores=state.prog_candidate_objective_scores[new_program_idx],
            experiment_tracker=self.experiment_tracker,
            linear_pareto_front_program_idx=linear_pareto_front_program_idx,
            valset_size=len(valset),
            val_evaluation_policy=self.val_evaluation_policy,
            iteration=iteration,
        )

        # Log candidate table row with instructions and metadata
        component_names = sorted(new_program.keys())
        columns = ['iteration', 'candidate_idx', 'parent_ids', 'valset_score', 'is_best'] + [
            f'text:{name}' for name in component_names
        ]
        row = [
            iteration,
            new_program_idx,
            str(parent_program_idx),
            valset_score,
            is_best_program,
        ] + [new_program[name] for name in component_names]
        self.experiment_tracker.log_table('candidates', columns=columns, data=[row])

        return new_program_idx, linear_pareto_front_program_idx

    # ------------------------------------------------------------------
    # Reflective proposal acceptance (shared by single and parallel paths)
    # ------------------------------------------------------------------

    def _report_rejected_proposal(
        self,
        proposal: CandidateProposal,
        iteration: int,
        reason_override: str | None = None,
    ) -> None:
        """Log why a proposal did not proceed to full validation."""
        old_sum = sum(proposal.subsample_scores_before or [])
        new_sum = sum(proposal.subsample_scores_after or [])
        if reason_override is not None:
            reject_msg = f'Iteration {iteration}: {reason_override}, skipping'
        else:
            reject_msg = (
                f'Iteration {iteration}: New subsample score {new_sum} is not better than old score {old_sum}, skipping'
            )
        self.logger.log(reject_msg)
        self._log_proposal_lm_calls(iteration, proposal, candidate_idx=-1)

    def _run_reflective_batch(
        self,
        proposals: list[CandidateProposal],
        state: GEPAState[RolloutOutput, DataId],
    ) -> bool:
        """Full-evaluate and add proposals that improve their screening score."""
        iteration = self._display_iteration(state)

        # The on-disk anchor was stamped on this trace entry when its iteration
        # slot was created; fall back to the legacy sequence anchor for trace
        # entries that predate it.
        trace_entry = state.full_program_trace[-1]
        iteration_id = trace_entry.get('iteration_id') or str(trace_entry.get('i', 0) + 1)

        # Capture evaluation side_info into the trace for agent consumption.
        # All of the batch's proposals share this iteration's trace entry; the
        # first proposal keeps the single-proposal schema agents already read.
        if self.write_agent_state and proposals:
            _record_proposal_evals(trace_entry, proposals[0])
            trace_entry['proposed_candidate'] = proposals[0].candidate

        selected = [
            proposal
            for proposal in proposals
            if sum(proposal.subsample_scores_after or []) > sum(proposal.subsample_scores_before or [])
        ]

        # Deduplicate: identical child candidates in the same
        # iteration would each get a full valset eval and a pool entry.
        deduped: list[CandidateProposal] = []
        seen_candidates: set[tuple] = set()
        seen_ids: set[int] = set()
        duplicates: list[CandidateProposal] = []
        for p in selected:
            content_key = tuple(sorted(p.candidate.items()))
            if id(p) in seen_ids or content_key in seen_candidates:
                duplicates.append(p)
                continue
            seen_ids.add(id(p))
            seen_candidates.add(content_key)
            deduped.append(p)
        selected = deduped

        # Report everything not selected as rejected.
        selected_ids = {id(p) for p in selected}
        for proposal in proposals:
            if id(proposal) in selected_ids:
                continue
            if any(proposal is d for d in duplicates):
                self._report_rejected_proposal(
                    proposal,
                    iteration,
                    reason_override='Duplicate of another candidate selected this iteration',
                )
                continue
            self._report_rejected_proposal(proposal, iteration)

        if not selected:
            if self.write_agent_state:
                trace_entry['proposal_accepted'] = False
            return False

        # Full-valset eval of the selected candidates (one batched, read-only call).
        valset_evals = self._evaluate_programs_on_valset([p.candidate for p in selected], state)

        # Add each selected candidate to the pool, in order.
        any_accepted = False
        for k, (proposal, (valset_evaluation, num_actual_evals)) in enumerate(zip(selected, valset_evals, strict=True)):
            new_sum = sum(proposal.subsample_scores_after or [])
            self.logger.log(
                f'Iteration {iteration}: Accepted candidate (subsample score '
                f'{sum(proposal.subsample_scores_before or [])} -> {new_sum}); running full eval.'
            )
            # Each accepted candidate in a batch needs its own on-disk anchor.
            # The first keeps the iteration's id (back-compat: agents still find
            # the primary accepted candidate at ``iterations/<iteration_id>/``);
            # the rest get suffixed ids so their ``outputs/`` and
            # ``trajectories/`` no longer overwrite one another under the shared
            # id. ``_save_iteration_dirs`` reads these back from
            # ``iteration_ids_by_candidate_idx`` to emit one dir per candidate.
            candidate_iteration_id = iteration_id if k == 0 else f'{iteration_id}-{k}'
            new_idx, _ = self._add_evaluated_program(
                new_program=proposal.candidate,
                state=state,
                parent_program_idx=proposal.parent_program_ids,
                valset_evaluation=valset_evaluation,
                num_actual_evals=num_actual_evals,
                iteration_id=candidate_iteration_id,
            )
            self._log_proposal_lm_calls(iteration, proposal, candidate_idx=new_idx)
            any_accepted = True

        if self.write_agent_state:
            trace_entry['proposal_accepted'] = any_accepted

        return any_accepted

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------

    def run(self) -> GEPAState[RolloutOutput, DataId]:
        # Check tqdm availability if progress bar is enabled
        progress_bar = None
        if self.display_progress_bar:
            if tqdm is None:
                raise ImportError('tqdm must be installed when display_progress_bar is enabled')

            # Check if stop_callback contains MaxMetricCallsStopper
            total_calls: int | None = None
            stop_cb = self.stop_callback
            if stop_cb is not None:
                max_calls_attr = getattr(stop_cb, 'max_metric_calls', None)
                if isinstance(max_calls_attr, int):
                    # Direct MaxMetricCallsStopper
                    total_calls = max_calls_attr
                else:
                    stoppers = getattr(stop_cb, 'stoppers', None)
                    if stoppers is not None:
                        # CompositeStopper - iterate to find MaxMetricCallsStopper
                        for stopper in stoppers:
                            stopper_max = getattr(stopper, 'max_metric_calls', None)
                            if isinstance(stopper_max, int):
                                total_calls = stopper_max
                                break

            if total_calls is not None:
                progress_bar = tqdm(total=total_calls, desc='GEPA Optimization', unit='rollouts')
            else:
                progress_bar = tqdm(desc='GEPA Optimization', unit='rollouts')
            progress_bar.update(0)

        # Prepare valset
        valset = self.valset
        if valset is None:
            raise ValueError('valset must be provided to GEPAEngine.run()')

        def valset_evaluator(
            program: dict[str, str],
            val_ids: list[Any],
        ) -> ValsetEvaluation[RolloutOutput, DataId]:
            # When write_agent_state is on, evaluate with traces so the seed's
            # trajectories land under iterations/seed/trajectories/.
            if self.write_agent_state:
                eval_result = self.adapter.evaluate(valset.fetch(val_ids), program, capture_traces=True)
                return ValsetEvaluation(
                    outputs_by_val_id=dict(zip(val_ids, eval_result.outputs, strict=False)),
                    scores_by_val_id=dict(zip(val_ids, eval_result.scores, strict=False)),
                    objective_scores_by_val_id=(
                        dict(zip(val_ids, eval_result.objective_scores, strict=False))
                        if eval_result.objective_scores is not None
                        else None
                    ),
                    trajectories_by_val_id=(
                        dict(zip(val_ids, eval_result.trajectories, strict=False))
                        if eval_result.trajectories is not None
                        else None
                    ),
                )

            (eval_result,) = invoke_batch_evaluate(
                self.adapter,
                [(program, valset.fetch(val_ids))],
                capture_traces=False,
            )
            outputs_dict = dict(zip(val_ids, eval_result.outputs, strict=False))
            scores_dict = dict(zip(val_ids, eval_result.scores, strict=False))
            objective_scores_dict = (
                dict(zip(val_ids, eval_result.objective_scores, strict=False))
                if eval_result.objective_scores is not None
                else None
            )
            return ValsetEvaluation(
                outputs_by_val_id=outputs_dict,
                scores_by_val_id=scores_dict,
                objective_scores_by_val_id=objective_scores_dict,
            )

        # Evaluate seed candidate on valset.
        # Policies may narrow the seed evaluation via the optional get_seed_eval_batch
        # hook; the state does not exist yet, so get_eval_batch cannot be used here.
        seed_batch_fn = getattr(self.val_evaluation_policy, 'get_seed_eval_batch', None)
        seed_val_ids = list(seed_batch_fn(valset)) if seed_batch_fn is not None else list(valset.all_ids())
        seed_valset_evaluation = valset_evaluator(self.seed_candidate, seed_val_ids)

        # Initialize state with pre-computed seed evaluation
        resumed = self.run_dir is not None and os.path.exists(os.path.join(self.run_dir, 'gepa_state.bin'))
        state = initialize_gepa_state(
            run_dir=self.run_dir,
            logger=self.logger,
            seed_candidate=self.seed_candidate,
            seed_valset_evaluation=seed_valset_evaluation,
            track_best_outputs=self.track_best_outputs,
            frontier_type=self.frontier_type,
            evaluation_cache=self._initial_evaluation_cache,
        )
        # Fresh runs: record the seed valset eval in the cache. On resume the seed scores already
        # live in state; writing the re-computed seed eval would desynchronize cache from
        # prog_candidate_val_subscores.
        if not resumed and state.evaluation_cache is not None:
            seed_ids = list(seed_valset_evaluation.scores_by_val_id)
            seed_obj = (
                [seed_valset_evaluation.objective_scores_by_val_id[eid] for eid in seed_ids]
                if seed_valset_evaluation.objective_scores_by_val_id is not None
                else None
            )
            state.evaluation_cache.put_batch(
                self.seed_candidate,
                seed_ids,
                [seed_valset_evaluation.outputs_by_val_id[eid] for eid in seed_ids],
                [seed_valset_evaluation.scores_by_val_id[eid] for eid in seed_ids],
                seed_obj,
                split=self.valset_cache_split,
            )

        # Seed uses the reserved iteration id — outputs/trajectories go under
        # iterations/seed/ alongside subsequent loop iterations.
        self._write_agent_iteration_files(SEED_ITERATION_ID, seed_valset_evaluation)

        # Restore adapter state from persisted state (only has effect on resume)
        self._sync_state_to_adapter(state)

        # Log base program score
        # Log run configuration
        self.experiment_tracker.log_config(
            {
                'seed': self.seed,
                'perfect_score': self.perfect_score,
                'frontier_type': self.frontier_type,
                'track_best_outputs': self.track_best_outputs,
                'use_cloudpickle': self.use_cloudpickle,
                'raise_on_exception': self.raise_on_exception,
                'trainset_size': len(self.reflective_proposer.trainset),
                'valset_size': len(valset),
                'seed_candidate_components': sorted(self.seed_candidate.keys()),
                'val_evaluation_policy': type(self.val_evaluation_policy).__name__,
                'run_dir': self.run_dir,
            }
        )

        # Log base program score using the same metric names as subsequent iterations
        # so they appear on the same charts in wandb/mlflow
        base_val_avg, base_val_coverage = state.get_program_average_val_subset(0)
        pareto_scores = list(state.pareto_front_valset.values())
        base_pareto_avg = sum(pareto_scores) / len(pareto_scores) if pareto_scores else base_val_avg
        self.experiment_tracker.log_metrics(
            {
                'val_program_average': base_val_avg,
                'best_score_on_valset': base_val_avg,
                'val_evaluated_count_new_program': base_val_coverage,
                'val_total_count': len(valset),
                'total_metric_calls': state.total_num_evals,
                'valset_pareto_front_agg': base_pareto_avg,
                'new_program_idx': 0,
                'linear_pareto_front_program_idx': 0,
                'best_program_as_per_agg_score_valset': 0,
            },
            step=state.i + 1,
        )

        if resumed:
            self.logger.log(
                f'Resume checkpoint before iteration {state.i + 1}: seed program full valset score from the '
                f'checkpoint is {base_val_avg} over {base_val_coverage} / {len(valset)} examples '
                '(not a child validation or frontier update).'
            )
        else:
            self.logger.log(
                f'Seed program full valset score: {base_val_avg} over {base_val_coverage} / {len(valset)} examples'
            )

        # Main loop
        last_pbar_val = 0
        if self._should_stop(state):
            remaining = self._get_remaining_budget(state)
            budget = f', remaining budget={remaining}' if remaining is not None else ''
            self.logger.log(
                f'Stop condition already met before the optimization loop '
                f'(metric calls used={state.total_num_evals}{budget}); no new iterations will run.'
            )
        while not self._should_stop(state):
            if self.display_progress_bar and progress_bar is not None:
                delta = state.total_num_evals - last_pbar_val
                progress_bar.update(delta)
                last_pbar_val = state.total_num_evals

            assert state.is_consistent()
            evals_before_iteration = state.total_num_evals
            try:
                self._sync_adapter_state_to_state(state)
                state.save(
                    self.run_dir,
                    use_cloudpickle=self.use_cloudpickle,
                    write_agent_state=self.write_agent_state,
                )

                state.i += 1
                state.full_program_trace.append(
                    {
                        'i': state.i,
                        'display_iteration': self._next_display_iteration(state),
                        'iteration_id': new_iteration_id(),
                    }
                )

                proposals = self.reflective_proposer.propose(state)
                if not proposals:
                    self.logger.log(
                        f'Iteration {self._display_iteration(state)}: Reflective mutation did not propose a new candidate'
                    )
                    continue

                self._run_reflective_batch(proposals, state)

            except Exception as e:
                self.logger.log(f'Iteration {self._display_iteration(state)}: Exception during optimization: {e}')
                self.logger.log(traceback.format_exc())
                made_progress = state.total_num_evals > evals_before_iteration
                if self.raise_on_exception or not made_progress:
                    raise e
                else:
                    continue

        # Close progress bar if it exists
        if self.display_progress_bar and progress_bar is not None:
            progress_bar.close()

        self._sync_adapter_state_to_state(state)
        state.save(
            self.run_dir,
            use_cloudpickle=self.use_cloudpickle,
            write_agent_state=self.write_agent_state,
        )

        best_candidate_idx = self.val_evaluation_policy.get_best_program(state)

        # Log final summary: seed candidate, best candidate, and all candidates table
        best_candidate = state.program_candidates[best_candidate_idx]
        best_score = self.val_evaluation_policy.get_valset_score(best_candidate_idx, state)
        summary: dict[str, Any] = {
            'best_candidate_idx': best_candidate_idx,
            'best_valset_score': best_score,
            'total_iterations': state.i,
            'total_candidates': len(state.program_candidates),
        }
        for name in sorted(self.seed_candidate.keys()):
            summary[f'seed/{name}'] = self.seed_candidate[name]
            summary[f'best/{name}'] = best_candidate[name]
        self.experiment_tracker.log_summary(summary)

        return state

    def _log_proposal_lm_calls(
        self,
        iteration: int,
        proposal: Any,
        candidate_idx: int,
    ) -> None:
        """Log per-component LM prompt / raw-output from a proposal to the experiment tracker.

        Appends one row per component to the ``"proposals"`` table.
        ``candidate_idx`` is the assigned index for accepted proposals,
        or ``-1`` for rejected ones — making it easy to join with the
        ``"candidates"`` table in WandB / MLflow.
        """
        metadata = proposal.metadata or {}
        status = 'accepted' if candidate_idx >= 0 else 'rejected'
        subsample_before = sum(proposal.subsample_scores_before or [])
        subsample_after = sum(proposal.subsample_scores_after or [])
        parent_ids_str = str(proposal.parent_program_ids)
        components = {k.split(':', 1)[1] for k in metadata if k.startswith('prompt:') or k.startswith('raw_lm_output:')}

        # Store strategy diagnostics separately from final LM prompts and outputs.
        reflection_metadata = {
            key: value
            for key, value in metadata.items()
            if key != 'proposal_id' and not key.startswith(('prompt:', 'raw_lm_output:'))
        }
        if reflection_metadata:
            self.experiment_tracker.log_table(
                'proposal_reflection_metadata',
                columns=['iteration', 'status', 'candidate_idx', 'parent_ids', 'proposal_id', 'reflection_metadata'],
                data=[
                    [
                        iteration,
                        status,
                        candidate_idx,
                        parent_ids_str,
                        metadata.get('proposal_id', ''),
                        json.dumps(reflection_metadata, sort_keys=True, default=str),
                    ]
                ],
            )

        if not components:
            return

        rows = []
        for comp in sorted(components):
            prompt = metadata.get(f'prompt:{comp}', '')
            raw_output = metadata.get(f'raw_lm_output:{comp}', '')
            proposed_text = proposal.candidate.get(comp, '')
            rows.append(
                [
                    iteration,
                    comp,
                    status,
                    candidate_idx,
                    parent_ids_str,
                    subsample_before,
                    subsample_after,
                    prompt if isinstance(prompt, str) else str(prompt),
                    raw_output,
                    proposed_text,
                ]
            )

        self.experiment_tracker.log_table(
            'proposals',
            columns=[
                'iteration',
                'component',
                'status',
                'candidate_idx',
                'parent_ids',
                'subsample_score_before',
                'subsample_score_after',
                'prompt',
                'raw_lm_output',
                'proposed_text',
            ],
            data=rows,
        )

    def _should_stop(self, state: GEPAState[RolloutOutput, DataId]) -> bool:
        """Check if the optimization should stop."""
        if self._stop_requested:
            return True
        if self.stop_callback and self.stop_callback(state):
            return True
        return False

    def _get_remaining_budget(self, state: GEPAState[RolloutOutput, DataId]) -> int | None:
        """Get remaining metric calls budget, or None if unlimited."""
        stop_cb = self.stop_callback
        if stop_cb is None:
            return None

        max_calls = getattr(stop_cb, 'max_metric_calls', None)
        if isinstance(max_calls, int):
            return max(0, max_calls - state.total_num_evals)

        # Check for CompositeStopper
        stoppers = getattr(stop_cb, 'stoppers', None)
        if stoppers is not None:
            for stopper in stoppers:
                stopper_max = getattr(stopper, 'max_metric_calls', None)
                if isinstance(stopper_max, int):
                    return max(0, stopper_max - state.total_num_evals)

        return None

    def request_stop(self) -> None:
        """Manually request the optimization to stop gracefully."""
        self.logger.log('Stop requested manually. Initiating graceful shutdown...')
        self._stop_requested = True
