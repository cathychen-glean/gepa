import json
from typing import ClassVar
from unittest.mock import MagicMock

from gepa.core.engine import GEPAEngine
from gepa.core.state import ValsetEvaluation
from gepa.logging.utils import log_detailed_metrics_after_discovering_new_program
from gepa.strategies.acceptance import StrictImprovementAcceptance
from glean_gepa.al_adapter import Candidate, ModuleSpec
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evalset_policy import UnseenEvalSetPolicy
from glean_gepa.evolutionary_proposer import EvolutionaryProposer, make_children_for_generation


class _ReflectionAdapter:
    def __init__(self, variants: list[str] | None = None) -> None:
        self.editable_modules = ["WRITING_CODE"]
        self.reflection_calls = 0
        self.variants = variants if variants is not None else ["cached rewrite 1", "cached rewrite 2"]

    def get_screening_score(self, _eval: object) -> float:
        return 1.0

    def make_reflective_dataset(self, **_kwargs: object) -> dict[str, list[dict[str, str]]]:
        return {"WRITING_CODE": [{"feedback": "fix it"}]}

    def propose_new_texts(self, **_kwargs: object) -> tuple[list[str], object, str]:
        self.reflection_calls += 1
        return self.variants, None, ""


class _Evaluation:
    def __init__(self) -> None:
        self.trajectories = [object()]


class _ProposerAdapter(_ReflectionAdapter):
    supports_high_signal_eval = False

    def __init__(self) -> None:
        super().__init__()
        self.root_evaluation_calls = 0
        self.screen_evaluation_calls = 0

    def evaluate(self, _batch, _candidate, capture_traces=False):
        self.root_evaluation_calls += 1
        return _Evaluation()

    def evaluate_many(self, _batch, candidates, capture_traces=False):
        if capture_traces:
            self.root_evaluation_calls += len(candidates)
            return [_Evaluation() for _ in candidates]
        self.screen_evaluation_calls += 1
        return [GleanEvaluationBatch(outputs=[{}], scores=[1.0], summary={"objective": 1.0}) for _ in candidates]

    def batch_evaluate(self, items, *, capture_traces=True):
        self.screen_evaluation_calls += 1
        return [
            GleanEvaluationBatch(outputs=[{}], scores=[1.0], summary={"objective": 1.0}) for _candidate, _batch in items
        ]

    @staticmethod
    def attach_cached_eval_run_ids(batch, _eval_run_ids):
        return batch


class _HighSignalProposerAdapter(_ProposerAdapter):
    supports_high_signal_eval = True

    def get_screening_score(self, _eval: object) -> float:
        return 0.9467353951890034

    @staticmethod
    def high_signal_batch(_parent_eval: object) -> list[dict[str, object]]:
        return [{}]

    @staticmethod
    def prepare_high_signal_batch(batch: list[dict[str, object]]) -> list[dict[str, object]]:
        return batch

    @staticmethod
    def high_signal_fix_rate(_parent_eval: object, _child_eval: object) -> float:
        return 7 / 17


class _OneSliceLoader:
    def all_ids(self):
        return [0]

    def fetch(self, _ids):
        return [{}]

    def __len__(self):
        return 1


class _TwoSliceLoader:
    def all_ids(self):
        return [0, 1]

    def fetch(self, _ids):
        return [{}]

    def __len__(self):
        return 2


def _candidate(candidate_id: str, text: str = "original") -> Candidate:
    return Candidate(
        model="test",
        prompt_modules={"WRITING_CODE": text},
        module_specs={"WRITING_CODE": ModuleSpec("WRITING_CODE", "free_text", 100)},
        global_token_cap=100,
        baseline_prompt_hash="baseline",
        candidate_id=candidate_id,
    )


def _proposer(adapter: _ReflectionAdapter, cache_file: str) -> EvolutionaryProposer:
    return EvolutionaryProposer(
        logger=MagicMock(),
        trainset=[],
        al_adapter=adapter,  # type: ignore[arg-type]
        reflection_llm=object(),
        experiment_tracker=MagicMock(),
        model="test",
        module_specs={"WRITING_CODE": ModuleSpec("WRITING_CODE", "free_text", 100)},
        global_token_cap=100,
        baseline_prompt_hash="baseline",
        evalset_policy=UnseenEvalSetPolicy(),
        children_cache_file=cache_file,
    )


def test_reuses_children_cached_for_root_without_rereflecting() -> None:
    adapter = _ReflectionAdapter()
    root = _candidate("root")
    children_by_root: dict[str, list[Candidate]] = {}
    frontier_evals = {root.candidate_id: _Evaluation()}

    first = make_children_for_generation(
        adapter, [root], frontier_evals, reflection_llm=object(), offspring_count=2, children_by_root=children_by_root
    )
    second = make_children_for_generation(
        adapter, [root], frontier_evals, reflection_llm=object(), offspring_count=2, children_by_root=children_by_root
    )

    assert adapter.reflection_calls == 1
    assert [child.prompt_modules for child in second] == [child.prompt_modules for child in first]
    assert children_by_root[root.candidate_id] == first


def test_prints_child_prompt_delta_against_parent(capsys) -> None:
    adapter = _ReflectionAdapter(variants=["first line\nupdated line\n"])
    root = _candidate("root", "first line\noriginal line\n")

    make_children_for_generation(
        adapter,
        [root],
        {root.candidate_id: _Evaluation()},
        reflection_llm=object(),
        offspring_count=1,
    )

    output = capsys.readouterr().out
    assert "CHILD PROPOSAL" in output
    assert "--- parent/root/WRITING_CODE" in output
    assert "-original line" in output
    assert "+updated line" in output


def test_empty_reflection_result_marks_root_as_cached() -> None:
    adapter = _ReflectionAdapter(variants=[])
    root = _candidate("root")
    children_by_root: dict[str, list[Candidate]] = {}
    frontier_evals = {root.candidate_id: _Evaluation()}

    first = make_children_for_generation(
        adapter, [root], frontier_evals, reflection_llm=object(), children_by_root=children_by_root
    )
    second = make_children_for_generation(
        adapter, [root], frontier_evals, reflection_llm=object(), children_by_root=children_by_root
    )

    assert first == second == []
    assert adapter.reflection_calls == 1
    assert children_by_root == {root.candidate_id: []}


def test_children_cache_survives_proposer_restart(tmp_path) -> None:
    cache_file = str(tmp_path / "children.json")
    root = _candidate("root")
    frontier_evals = {root.candidate_id: _Evaluation()}
    first_adapter = _ReflectionAdapter()
    first_proposer = _proposer(first_adapter, cache_file)
    first_slice_cache = first_proposer._children_by_root_by_train_slice.setdefault((0,), {})

    first = make_children_for_generation(
        first_adapter,
        [root],
        frontier_evals,
        reflection_llm=object(),
        offspring_count=2,
        children_by_root=first_slice_cache,
    )
    first_proposer._record_eval_run_ids(
        (0,),
        first[0],
        [
            {
                "eval_set_name": "focused",
                "eval_set_version": "v1",
                "student_eval_run_id": "eval-child-1",
            }
        ],
    )
    first_proposer._save_children_cache()
    cached_child = json.loads((tmp_path / "children.json").read_text())["training_slices"][0]["roots"]["root"][0]
    assert cached_child["prompt_modules"] == first[0].prompt_modules
    assert cached_child["eval_run_ids"][0]["student_eval_run_id"] == "eval-child-1"

    second_adapter = _ReflectionAdapter()
    second_proposer = _proposer(second_adapter, cache_file)
    second = make_children_for_generation(
        second_adapter,
        [root],
        frontier_evals,
        reflection_llm=object(),
        offspring_count=2,
        children_by_root=second_proposer._children_by_root_by_train_slice[(0,)],
    )

    assert first_adapter.reflection_calls == 1
    assert second_adapter.reflection_calls == 0
    assert [child.prompt_modules for child in second] == [child.prompt_modules for child in first]
    assert second_proposer._cached_eval_run_ids((0,), second[0]) == [
        {
            "eval_set_name": "focused",
            "eval_set_version": "v1",
            "student_eval_run_id": "eval-child-1",
        }
    ]


def test_children_cache_persists_screening_result_with_eval_id(tmp_path) -> None:
    cache_file = str(tmp_path / "children.json")
    root = _candidate("root")
    frontier_evals = {root.candidate_id: _Evaluation()}
    first_adapter = _ReflectionAdapter()
    first_proposer = _proposer(first_adapter, cache_file)
    first_slice_cache = first_proposer._children_by_root_by_train_slice.setdefault((0,), {})

    children = make_children_for_generation(
        first_adapter,
        [root],
        frontier_evals,
        reflection_llm=object(),
        offspring_count=2,
        children_by_root=first_slice_cache,
    )
    first_proposer._record_eval_run_ids(
        (0,),
        children[0],
        [
            {
                "eval_set_name": "focused",
                "eval_set_version": "v1",
                "student_eval_run_id": "eval-child-1",
            }
        ],
    )
    first_proposer._record_screening_result((0,), children[0], 0.75, True)
    # Simulate a result persisted under the old 50% gate. The score must be
    # reconsidered when the threshold changes.
    first_proposer._record_screening_result((0,), children[1], 1 / 3, False)
    first_proposer._save_children_cache()

    second_proposer = _proposer(_ReflectionAdapter(), cache_file)
    cached = second_proposer._cached_screening_scores(
        (0,),
        second_proposer._children_by_root_by_train_slice[(0,)]["root"],
        use_high_signal_gate=True,
    )

    assert cached == [(0.75, True), (1 / 3, True)]


def test_same_root_and_training_slice_reuses_children_and_screen(tmp_path) -> None:
    cache_file = str(tmp_path / "children.json")
    root = _candidate("root")

    class _State:
        i = -1
        program_candidates: ClassVar[list[dict[str, str]]] = [root.prompt_modules]
        total_num_evals = 0
        num_full_ds_evals = 1
        program_full_scores_val_set: ClassVar[list[float]] = [1.0]

        @staticmethod
        def get_pareto_front_mapping():
            return {0: {0}}

    first_adapter = _ProposerAdapter()
    first_proposer = _proposer(first_adapter, cache_file)
    first_proposer.trainset = _OneSliceLoader()
    first_proposer.propose(_State())

    second_adapter = _ProposerAdapter()
    second_proposer = _proposer(second_adapter, cache_file)
    second_proposer.trainset = _OneSliceLoader()
    second_proposer.propose(_State())

    assert first_adapter.reflection_calls == 1
    assert first_adapter.root_evaluation_calls == 1
    assert first_adapter.screen_evaluation_calls == 1
    assert second_adapter.reflection_calls == 0
    assert second_adapter.root_evaluation_calls == 0
    assert second_adapter.screen_evaluation_calls == 0


def test_replaying_a_fully_cached_slice_skips_root_error_example_fetches(tmp_path) -> None:
    """Root traces feed reflection and screening; a replay needs neither."""
    root = _candidate("root")
    cache_file = str(tmp_path / "children.json")

    class _State:
        i = -1
        program_candidates: ClassVar[list[dict[str, str]]] = [root.prompt_modules]
        total_num_evals = 0
        num_full_ds_evals = 1
        program_full_scores_val_set: ClassVar[list[float]] = [1.0]

        @staticmethod
        def get_pareto_front_mapping():
            return {0: {0}}

    class _TraceRecordingAdapter(_HighSignalProposerAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.capture_traces_requests: list[bool] = []

        def evaluate(self, _batch, _candidate, capture_traces=False):
            self.capture_traces_requests.append(capture_traces)
            return super().evaluate(_batch, _candidate, capture_traces=capture_traces)

        def evaluate_many(self, _batch, candidates, capture_traces=False):
            self.capture_traces_requests.append(capture_traces)
            self.root_evaluation_calls += len(candidates)
            if capture_traces:
                return [_Evaluation() for _ in candidates]
            return [
                GleanEvaluationBatch(outputs=[{}], scores=[1.0], summary={"objective": 1.0})
                for _ in candidates
            ]

    first_adapter = _TraceRecordingAdapter()
    first_proposer = _proposer(first_adapter, cache_file)
    first_proposer.trainset = _OneSliceLoader()
    assert first_proposer.propose(_State())
    assert first_adapter.capture_traces_requests == [True]

    # The root score is what makes a replay skip evaluation entirely, so drop it
    # to force the root eval and assert it no longer asks for error examples.
    del first_proposer._root_screening_scores_by_train_slice[(0,)]
    first_proposer._save_children_cache()

    replay_adapter = _TraceRecordingAdapter()
    replay_proposer = _proposer(replay_adapter, cache_file)
    replay_proposer.trainset = _OneSliceLoader()
    assert replay_proposer.propose(_State())

    assert replay_adapter.capture_traces_requests == [False]
    assert replay_adapter.reflection_calls == 0
    assert replay_adapter.screen_evaluation_calls == 0


def test_resume_skips_slices_whose_passing_children_are_already_in_the_pool(tmp_path) -> None:
    cache_file = str(tmp_path / "children.json")
    root = _candidate("root")

    class _FreshState:
        i = -1
        program_candidates: ClassVar[list[dict[str, str]]] = [root.prompt_modules]
        total_num_evals = 0
        num_full_ds_evals = 1
        program_full_scores_val_set: ClassVar[list[float]] = [1.0]

        @staticmethod
        def get_pareto_front_mapping():
            return {0: {0}}

    first_adapter = _ProposerAdapter()
    first_adapter.variants = ["slice-0 rewrite"]
    first_proposer = _proposer(first_adapter, cache_file)
    first_proposer.trainset = _TwoSliceLoader()
    first_proposals = first_proposer.propose(_FreshState())
    assert first_proposals
    accepted_child = first_proposals[0].candidate

    class _ResumedState:
        i = 0
        program_candidates: ClassVar[list[dict[str, str]]] = [root.prompt_modules, accepted_child]
        total_num_evals = 0
        num_full_ds_evals = 2
        program_full_scores_val_set: ClassVar[list[float]] = [1.0, 0.9]

        @staticmethod
        def get_pareto_front_mapping():
            return {0: {0, 1}}

    second_adapter = _ProposerAdapter()
    second_adapter.variants = ["slice-1 rewrite"]
    second_proposer = _proposer(second_adapter, cache_file)
    second_proposer.trainset = _TwoSliceLoader()
    second_proposals = second_proposer.propose(_ResumedState())

    assert second_adapter.reflection_calls >= 1
    assert second_proposals
    assert all(proposal.candidate != accepted_child for proposal in second_proposals)
    assert second_proposals[0].candidate["WRITING_CODE"] == "slice-1 rewrite"


def test_high_signal_screen_uses_zero_baseline_before_full_validation(tmp_path) -> None:
    """A high-signal fix rate must not be compared to the parent's full score."""
    root = _candidate("root")

    class _State:
        i = -1
        program_candidates: ClassVar[list[dict[str, str]]] = [root.prompt_modules]
        total_num_evals = 0
        num_full_ds_evals = 1
        program_full_scores_val_set: ClassVar[list[float]] = [1.0]

        @staticmethod
        def get_pareto_front_mapping():
            return {0: {0}}

    proposer = _proposer(_HighSignalProposerAdapter(), str(tmp_path / "children.json"))
    proposer.trainset = _OneSliceLoader()

    proposals = proposer.propose(_State())

    assert proposals
    assert all(proposal.tag == "evolutionary_high_signal" for proposal in proposals)
    assert all(proposal.subsample_scores_before == [0.0] for proposal in proposals)
    assert all(proposal.subsample_scores_after == [7 / 17] for proposal in proposals)
    assert all(StrictImprovementAcceptance().should_accept(proposal, _State()) for proposal in proposals)


def test_display_iteration_advances_only_after_full_evaluation() -> None:
    class _State:
        i = 5
        num_full_ds_evals = 1

    assert EvolutionaryProposer.get_display_iteration(_State()) == 1
    _State.num_full_ds_evals += 1
    assert EvolutionaryProposer.get_display_iteration(_State()) == 2


def test_engine_keeps_the_stamped_full_eval_iteration_during_validation() -> None:
    class _State:
        i = 5
        num_full_ds_evals = 1
        full_program_trace: ClassVar[list[dict[str, int]]] = []

    engine = MagicMock()
    engine.reflective_proposer = EvolutionaryProposer

    assert GEPAEngine._next_display_iteration(engine, _State()) == 1
    _State.full_program_trace.append({"display_iteration": 1})
    _State.num_full_ds_evals += 1
    assert GEPAEngine._display_iteration(engine, _State()) == 1


def test_new_program_metrics_log_display_iteration_not_proposal_attempts() -> None:
    logger = MagicMock()
    experiment_tracker = MagicMock()
    val_evaluation_policy = MagicMock()
    val_evaluation_policy.get_best_program.return_value = 1
    val_evaluation_policy.get_valset_score.return_value = 0.5

    state = MagicMock()
    state.i = 8
    state.pareto_front_valset = {"a": 0.5}
    state.objective_pareto_front = {}
    state.program_at_pareto_front_valset = {"a": {1}}
    state.program_at_pareto_front_objectives = {}
    state.program_full_scores_val_set = [0.4, 0.5]
    state.prog_candidate_val_subscores = [{}, {"a": 0.5}]
    state.parent_program_for_candidate = [None, [0]]
    state.prog_candidate_objective_scores = [{}, {}]
    state.total_num_evals = 12

    log_detailed_metrics_after_discovering_new_program(
        logger=logger,
        gepa_state=state,
        new_program_idx=1,
        valset_evaluation=ValsetEvaluation(outputs_by_val_id={}, scores_by_val_id={"a": 0.5}),
        objective_scores={},
        experiment_tracker=experiment_tracker,
        linear_pareto_front_program_idx=1,
        valset_size=1,
        val_evaluation_policy=val_evaluation_policy,
        iteration=5,
    )

    logged = [call.args[0] for call in logger.log.call_args_list]
    assert logged
    assert all(message.startswith("Iteration 5:") for message in logged)
    assert not any("Iteration 9:" in message for message in logged)
    experiment_tracker.log_metrics.assert_called_once()
    metrics, kwargs = experiment_tracker.log_metrics.call_args
    assert metrics[0]["iteration"] == 5
    assert kwargs["step"] == 5
