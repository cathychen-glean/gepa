import json
from typing import ClassVar, cast
from unittest.mock import MagicMock

import pytest

from gepa.core.engine import GEPAEngine
from gepa.core.state import ValsetEvaluation
from gepa.logging.utils import log_detailed_metrics_after_discovering_new_program
from gepa.strategies.acceptance import StrictImprovementAcceptance
from glean_gepa.al_adapter import Candidate, ModuleSpec
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evalset_policy import UnseenEvalSetPolicy
from glean_gepa.evolutionary_proposer import (
    CHILDREN_CACHE_SCHEMA_VERSION,
    MAX_EMPTY_PROPOSAL_STREAK,
    EvolutionaryProposer,
    make_children_for_generation,
)
from glean_gepa.utils import apply_single_module_edit


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


class _ScoredEvaluation(_Evaluation):
    def __init__(self, score: float) -> None:
        super().__init__()
        self.score = score


class _ScoreOrderedAdapter(_ReflectionAdapter):
    """Scores each root differently and records the order they are reflected on."""

    def __init__(self) -> None:
        super().__init__()
        self.reflected_parents: list[str] = []

    def get_screening_score(self, eval_batch: object) -> float:
        return float(getattr(eval_batch, "score", 0.0))

    def propose_new_texts(self, **kwargs: object) -> tuple[list[str], object, str]:
        candidate = cast(Candidate, kwargs["candidate"])
        self.reflected_parents.append(candidate.candidate_id)
        return super().propose_new_texts(**kwargs)


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


def test_reflects_on_roots_in_descending_screening_score_order() -> None:
    adapter = _ScoreOrderedAdapter()
    roots = [_candidate("weak"), _candidate("strong"), _candidate("middle")]
    frontier_evals = {
        "weak": _ScoredEvaluation(0.72),
        "strong": _ScoredEvaluation(0.75),
        "middle": _ScoredEvaluation(0.73),
    }
    children_by_root: dict[str, list[Candidate]] = {}

    make_children_for_generation(
        adapter,
        roots,
        frontier_evals,
        reflection_llm=object(),
        offspring_count=5,
        children_by_root=children_by_root,
    )

    # Every root is reflected at most once, strongest first. Sampling a weaker
    # root ahead of a stronger one would spend a reflection call on a prompt
    # already known to score worse.
    assert adapter.reflected_parents == ["strong", "middle", "weak"]


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


@pytest.mark.parametrize("schema_version", range(2, CHILDREN_CACHE_SCHEMA_VERSION + 1))
def test_loads_every_released_children_cache_schema(tmp_path, schema_version) -> None:
    cache_file = tmp_path / "children.json"
    cache_file.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "training_slices": [
                    {
                        "train_ids": [0],
                        "root_screening_scores": {"root": 0.75},
                        "roots": {
                            "root": [
                                {
                                    "prompt_modules": {"WRITING_CODE": "cached rewrite"},
                                    "eval_run_ids": [
                                        {
                                            "eval_set_name": "focused",
                                            "eval_set_version": "v1",
                                            "student_eval_run_id": "eval-child-1",
                                        }
                                    ],
                                    "screening_score": 0.5,
                                    "screening_passed": True,
                                }
                            ]
                        },
                    }
                ],
            }
        )
    )

    proposer = _proposer(_ReflectionAdapter(), str(cache_file))
    child = proposer._children_by_root_by_train_slice[(0,)]["root"][0]

    assert child.prompt_modules["WRITING_CODE"] == "cached rewrite"
    assert proposer._cached_root_screening_score((0,), "root") == 0.75
    assert proposer._cached_eval_run_ids((0,), child)[0]["student_eval_run_id"] == "eval-child-1"
    assert proposer._cached_screening_scores((0,), [child], use_high_signal_gate=True) == [(0.5, True)]


def test_ignores_children_cache_written_by_a_newer_schema(tmp_path) -> None:
    """A cache from a future version is unreadable, so it must not be trusted."""
    cache_file = tmp_path / "children.json"
    cache_file.write_text(json.dumps({"schema_version": CHILDREN_CACHE_SCHEMA_VERSION + 1, "training_slices": []}))

    proposer = _proposer(_ReflectionAdapter(), str(cache_file))

    assert proposer._children_by_root_by_train_slice == {}


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


def _tiny_budget_candidate(candidate_id: str) -> Candidate:
    """A root whose module budget is far smaller than the variants reflection writes.

    run_ts11 hit this with RULES_EXT: a 64-token budget on an empty seed module.
    """
    return Candidate(
        model="test",
        prompt_modules={"WRITING_CODE": ""},
        module_specs={"WRITING_CODE": ModuleSpec("WRITING_CODE", "free_text", 1)},
        global_token_cap=100,
        baseline_prompt_hash="baseline",
        candidate_id=candidate_id,
    )


def test_over_budget_variants_are_excluded_from_the_generation() -> None:
    """Screening drops over-budget children, so admitting one only produces a
    candidate that can never be scored."""
    root = _tiny_budget_candidate("root")
    adapter = _ReflectionAdapter(variants=["ok", "x" * 400])

    children = make_children_for_generation(
        adapter,
        [root],
        {root.candidate_id: _Evaluation()},
        reflection_llm=object(),
        offspring_count=2,
    )

    assert [child.prompt_modules["WRITING_CODE"] for child in children] == ["ok"]


def test_cached_over_budget_children_are_not_replayed(tmp_path) -> None:
    """run_ts11 already has three of these cached; replaying them would re-fill
    the generation with candidates screening throws away."""
    root = _tiny_budget_candidate("root")
    over_budget = apply_single_module_edit(root, "WRITING_CODE", "z" * 400)
    fits = apply_single_module_edit(root, "WRITING_CODE", "ok")

    children = make_children_for_generation(
        _ReflectionAdapter(),
        [root],
        {root.candidate_id: _Evaluation()},
        reflection_llm=object(),
        offspring_count=5,
        children_by_root={root.candidate_id: [over_budget, fits]},
    )

    assert [child.prompt_modules["WRITING_CODE"] for child in children] == ["ok"]


def test_an_unscoreable_child_does_not_keep_its_slice_pending(tmp_path) -> None:
    """The run_ts11 livelock: over-budget children never get a score, so counting
    them as pending kept the proposer re-entering the same iteration forever."""
    root = _tiny_budget_candidate("root")
    proposer = _proposer(_ReflectionAdapter(), str(tmp_path / "children.json"))
    over_budget = apply_single_module_edit(root, "WRITING_CODE", "z" * 400)
    proposer._children_by_root_by_train_slice[(0,)] = {"root": [over_budget]}

    assert proposer._child_is_pending((0,), over_budget, set()) is False
    assert proposer._slice_is_exhausted((0,), set()) is True


def test_propose_raises_rather_than_spinning_on_empty_proposals(tmp_path) -> None:
    """Empty proposals consume no budget, so max_metric_calls cannot stop them."""
    proposer = _proposer(_ReflectionAdapter(), str(tmp_path / "children.json"))
    proposer._propose = lambda _state: []  # type: ignore[method-assign]

    for _ in range(MAX_EMPTY_PROPOSAL_STREAK - 1):
        assert proposer.propose(object()) == []  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="cannot advance"):
        proposer.propose(object())  # type: ignore[arg-type]


def test_a_successful_proposal_resets_the_empty_streak(tmp_path) -> None:
    proposer = _proposer(_ReflectionAdapter(), str(tmp_path / "children.json"))
    proposer._propose = lambda _state: []  # type: ignore[method-assign]
    for _ in range(MAX_EMPTY_PROPOSAL_STREAK - 1):
        proposer.propose(object())  # type: ignore[arg-type]

    proposer._propose = lambda _state: ["a proposal"]  # type: ignore[method-assign]
    assert proposer.propose(object()) == ["a proposal"]  # type: ignore[arg-type]
    assert proposer._empty_proposal_streak == 0


def test_screening_checkpoints_each_child_before_a_later_one_fails(tmp_path) -> None:
    """Scoring a screen reads BigQuery and can raise or hang partway through the
    batch. Children already scored must survive, or the restart re-screens work
    it has already paid for."""
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

    class _FailsScoringSecondChild(_HighSignalProposerAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.fix_rate_calls = 0

        def high_signal_fix_rate(self, _parent_eval, _child_eval):
            self.fix_rate_calls += 1
            if self.fix_rate_calls == 2:
                raise RuntimeError("BigQuery query failed")
            return 0.8

    adapter = _FailsScoringSecondChild()
    proposer = _proposer(adapter, cache_file)
    proposer.trainset = _OneSliceLoader()

    with pytest.raises(RuntimeError, match="BigQuery query failed"):
        proposer.propose(_State())

    assert adapter.fix_rate_calls == 2
    with open(cache_file) as handle:
        roots = json.load(handle)["training_slices"][0]["roots"]
    (records,) = roots.values()
    assert len(records) == 2
    assert records[0]["screening_score"] == 0.8
    assert records[0]["screening_passed"] is True
    # The child whose scoring raised stays unscored so it is retried.
    assert records[1]["screening_score"] is None
    assert records[1]["screening_passed"] is None


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
            return [GleanEvaluationBatch(outputs=[{}], scores=[1.0], summary={"objective": 1.0}) for _ in candidates]

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
