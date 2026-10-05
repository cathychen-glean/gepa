from __future__ import annotations
import json
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock
import pytest
from gepa.core.engine import GEPAEngine
from gepa.core.state import ValsetEvaluation
from gepa.logging.utils import log_detailed_metrics_after_discovering_new_program
from glean_gepa.al_adapter import ALRunner, Candidate, ModuleSpec
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.evalset_policy import UnseenEvalSetPolicy
from glean_gepa.evolutionary_proposer import (
    EvolutionaryProposer,
    make_children_for_generation,
    modules_after_tool_choice,
    pick_modules_to_edit,
    tool_usage_in_examples,
    tools_with_evidence,
)
from glean_gepa.prompt_constants import (
    CORE_TOOLS,
    EXECUTION_DISCIPLINE_KEY,
    RULES_EXT_KEY,
    WRITING_CODE_KEY,
)
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter
import json
import pytest
from gepa.core.data_loader import ListDataLoader
from glean_gepa.evalset_policy import TrainingScheduleExhaustedStopper, UnseenEvalSetPolicy


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
        return 9 / 17


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


class _MultiModuleAdapter(_ReflectionAdapter):
    """Reflects several modules, three variants each, recording the order asked."""

    def __init__(self) -> None:
        super().__init__()
        self.editable_modules = ["EXECUTION_DISCIPLINE", "RULES_EXT", "glean_search", "glean_document_reader"]
        self.modules_reflected: list[str] = []

    def make_reflective_dataset(self, **kwargs: object) -> dict[str, list[dict[str, str]]]:
        modules = kwargs["components_to_update"]
        assert isinstance(modules, list)
        return {module: [{"feedback": "fix it"}] for module in modules}

    def propose_new_texts(self, **kwargs: object) -> tuple[list[str], object, str]:
        self.reflection_calls += 1
        module = kwargs["components_to_update"][0]  # type: ignore[index]
        self.modules_reflected.append(module)
        return [f"{module} v1", f"{module} v2", f"{module} v3"], None, f"diagnosis for {module}"


def test_offspring_slots_are_shared_round_robin_across_modules(monkeypatch) -> None:
    """One variant per module before a second variant of any module.

    Generation 1 of the agentic run spent all five offspring on three rewordings of
    glean_document_reader and two of glean_search; Execution Discipline, which
    governs the final message the judge scores, was never edited.
    """
    from glean_gepa import evolutionary_proposer

    monkeypatch.setattr(
        evolutionary_proposer,
        "modules_after_tool_choice",
        lambda _llm, _parent, modules, _hs: [m for m in modules if not m.startswith("glean")]
        + [m for m in modules if m.startswith("glean")],
    )
    adapter = _MultiModuleAdapter()
    root = Candidate(
        model="test",
        prompt_modules=dict.fromkeys(adapter.editable_modules, "original"),
        module_specs={module: ModuleSpec(module, "free_text", 100) for module in adapter.editable_modules},
        global_token_cap=1000,
        baseline_prompt_hash="baseline",
        candidate_id="root",
    )

    children = make_children_for_generation(
        adapter, [root], {root.candidate_id: _Evaluation()}, reflection_llm=object(), offspring_count=5
    )

    edited = [next(module for module, text in child.prompt_modules.items() if text != "original") for child in children]
    assert edited == [
        "EXECUTION_DISCIPLINE",
        "RULES_EXT",
        "glean_search",
        "glean_document_reader",
        "EXECUTION_DISCIPLINE",
    ]
    assert children[0].prompt_modules["EXECUTION_DISCIPLINE"] == "EXECUTION_DISCIPLINE v1"
    assert children[4].prompt_modules["EXECUTION_DISCIPLINE"] == "EXECUTION_DISCIPLINE v2"
    assert adapter.modules_reflected == ["EXECUTION_DISCIPLINE", "RULES_EXT", "glean_search", "glean_document_reader"]


class _ScoredEvaluation(_Evaluation):
    def __init__(self, score: float) -> None:
        super().__init__()
        self.score = score


class _ScoredReflectionAdapter(_ReflectionAdapter):
    def __init__(self) -> None:
        super().__init__(variants=["v1", "v2", "v3", "v4", "v5"])
        self.reflected_parents: list[str] = []

    def get_screening_score(self, evaluation: _ScoredEvaluation) -> float:
        return evaluation.score

    def propose_new_texts(self, **kwargs: object) -> tuple[list[str], object, str]:
        self.reflection_calls += 1
        candidate = kwargs["candidate"]
        parent_id = candidate.candidate_id  # type: ignore[attr-defined]
        self.reflected_parents.append(parent_id)
        return [f"{parent_id}-{index}" for index in range(5)], None, ""


@pytest.mark.parametrize("high_is_cached", [False, True], ids=["both_reflected", "leader_cached"])
def test_offspring_slots_split_by_score_and_a_cached_parent_keeps_only_its_share(high_is_cached: bool) -> None:
    """Two frontier parents share one generation: 5 slots is 3 for the leader and 2 for the other.
    A parent whose children are already cached fills its 3 from the cache and cannot take the other's 2."""
    adapter = _ScoredReflectionAdapter()
    high, low = _candidate("high"), _candidate("low")
    cached = {"high": [_candidate(f"cached-{i}", f"cached-{i}") for i in range(5)]} if high_is_cached else {}

    children = make_children_for_generation(
        adapter,
        [low, high],
        {"high": _ScoredEvaluation(0.9), "low": _ScoredEvaluation(0.4)},
        reflection_llm=object(),
        offspring_count=5,
        children_by_root=cached,
    )

    texts = [child.prompt_modules["WRITING_CODE"] for child in children]
    leader_prefix = "cached-" if high_is_cached else "high-"
    assert sum(text.startswith(leader_prefix) for text in texts) == 3
    assert sum(text.startswith("low-") for text in texts) == 2
    assert adapter.reflected_parents == (["low"] if high_is_cached else ["low", "high"])


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
    # Simulate a result persisted as a pass under the old 1/3 gate. The score
    # must be reconsidered when the threshold changes.
    first_proposer._record_screening_result((0,), children[1], 1 / 3, True)
    first_proposer._save_children_cache()

    saved = json.loads(Path(cache_file).read_text())
    assert "training_slices" in saved and "schema_version" not in saved

    second_proposer = _proposer(_ReflectionAdapter(), cache_file)
    reloaded = second_proposer._children_by_root_by_train_slice[(0,)]["root"]
    assert second_proposer._cached_eval_run_ids((0,), reloaded[0])[0]["student_eval_run_id"] == "eval-child-1"
    assert second_proposer._cached_screening_scores((0,), reloaded, use_high_signal_gate=True) == [
        (0.75, True),
        (1 / 3, False),
    ]


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


def test_screening_kind_none_sends_every_child_to_validation(tmp_path) -> None:
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

    class _SkipAdapter(_HighSignalProposerAdapter):
        screening_kind = "none"

        def high_signal_batch(self, _parent_eval: object) -> list[dict[str, object]]:
            raise AssertionError("skip screening must not build a high-signal batch")

    adapter = _SkipAdapter()
    proposer = _proposer(adapter, cache_file)
    proposer.trainset = _OneSliceLoader()
    proposals = proposer.propose(_State())

    assert adapter.screen_evaluation_calls == 0
    assert adapter.reflection_calls == 1
    assert proposals
    assert all(proposal.metadata["screening_kind"] == "none" for proposal in proposals)
    assert all(proposal.subsample_scores_before == [0.0] for proposal in proposals)
    assert all(proposal.subsample_scores_after == [1.0] for proposal in proposals)


def test_each_proposal_names_the_frontier_parent_it_was_reflected_from(tmp_path) -> None:
    """Children of the second frontier parent must not be recorded as children of the best one.

    A run wrote every child of a two-parent generation as ``parents [2]`` even though two
    of them were candidate 4's, which corrupts the lineage in the state and the run log.
    """
    alpha = _candidate("alpha", "alpha text")
    beta = _candidate("beta", "beta text")

    class _PerParentAdapter(_ProposerAdapter):
        screening_kind = "none"

        def propose_new_texts(self, **kwargs: object) -> tuple[list[str], object, str]:
            self.reflection_calls += 1
            parent = kwargs["candidate"]
            return [f"{parent.prompt_modules['WRITING_CODE']} rewrite"], None, ""

    class _State:
        i = -1
        program_candidates: ClassVar[list[dict[str, str]]] = [alpha.prompt_modules, beta.prompt_modules]
        total_num_evals = 0
        num_full_ds_evals = 2
        program_full_scores_val_set: ClassVar[list[float]] = [1.0, 0.9]

        @staticmethod
        def get_pareto_front_mapping():
            return {0: {0, 1}}

    proposer = _proposer(_PerParentAdapter(), str(tmp_path / "children.json"))
    proposer.trainset = _OneSliceLoader()
    proposals = proposer.propose(_State())

    by_text = {proposal.candidate["WRITING_CODE"]: proposal.parent_program_ids for proposal in proposals}
    assert by_text == {"alpha text rewrite": [0], "beta text rewrite": [1]}


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


def test_over_budget_child_does_not_pin_its_training_slice(tmp_path) -> None:
    """A child dropped by the budget check is never screened, so it must not read as unfinished."""
    cache_file = str(tmp_path / "children.json")
    root = _candidate("root")
    over_budget_rewrite = "x" * 500

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
    first_adapter.variants = ["slice-0 rewrite", over_budget_rewrite]
    first_proposer = _proposer(first_adapter, cache_file)
    first_proposer.trainset = _TwoSliceLoader()
    first_proposals = first_proposer.propose(_FreshState())
    assert [proposal.candidate["WRITING_CODE"] for proposal in first_proposals] == ["slice-0 rewrite"]

    class _ResumedState:
        """The accepted child scored below its parent, so the frontier is still the root alone."""

        i = 0
        program_candidates: ClassVar[list[dict[str, str]]] = [root.prompt_modules, first_proposals[0].candidate]
        total_num_evals = 0
        num_full_ds_evals = 2
        program_full_scores_val_set: ClassVar[list[float]] = [1.0, 0.9]

        @staticmethod
        def get_pareto_front_mapping():
            return {0: {0}}

    second_adapter = _ProposerAdapter()
    second_adapter.variants = ["slice-1 rewrite"]
    second_proposer = _proposer(second_adapter, cache_file)
    second_proposer.trainset = _TwoSliceLoader()
    second_proposals = second_proposer.propose(_ResumedState())

    assert [proposal.candidate["WRITING_CODE"] for proposal in second_proposals] == ["slice-1 rewrite"]
    assert all(proposal.subsample_indices == [1] for proposal in second_proposals)


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
    assert all(proposal.subsample_scores_after == [9 / 17] for proposal in proposals)
    # Strict improvement over the zero baseline, the rule the engine applies to accept a child.
    assert all(sum(proposal.subsample_scores_after) > sum(proposal.subsample_scores_before) for proposal in proposals)


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


def _engine_with_stall_limit(*, max_stalled_proposals: int | None) -> GEPAEngine:
    return GEPAEngine(
        adapter=MagicMock(),
        run_dir=None,
        valset=None,
        seed_candidate={},
        perfect_score=1.0,
        seed=0,
        reflective_proposer=MagicMock(),
        frontier_type="instance",
        logger=MagicMock(),
        experiment_tracker=MagicMock(),
        max_stalled_proposals=max_stalled_proposals,
    )


def test_stall_limit_reached_only_when_configured() -> None:
    limited = _engine_with_stall_limit(max_stalled_proposals=3)
    assert not limited._stall_limit_reached(2)
    assert limited._stall_limit_reached(3)
    unset = _engine_with_stall_limit(max_stalled_proposals=None)
    assert not unset._stall_limit_reached(1_000_000)


def test_pick_modules_to_edit_offers_every_listed_core_tool():
    runner = ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli"))
    kwargs = {
        "runner": runner,
        "teacher_model": "gpt",
        "student_model": "fast",
    }
    eval_batch = GleanEvaluationBatch(
        outputs=[],
        scores=[],
        trajectories=[
            {
                "output": {
                    "teacher_tool_events": ["Glean Search"],
                    "student_tool_events": ["Discover"],
                },
                "score": 0.0,
            }
        ],
    )
    prompt_only = TeacherStudentAdapter(**kwargs, editable_modules=[WRITING_CODE_KEY])
    core_tools = TeacherStudentAdapter(**kwargs, editable_modules=list(CORE_TOOLS))
    both = TeacherStudentAdapter(**kwargs, editable_modules=[WRITING_CODE_KEY, *CORE_TOOLS])
    search_only = TeacherStudentAdapter(**kwargs, editable_modules=["glean_search"])

    assert pick_modules_to_edit(prompt_only) == [WRITING_CODE_KEY]
    assert pick_modules_to_edit(prompt_only, eval_batch) == [WRITING_CODE_KEY]
    assert pick_modules_to_edit(core_tools) == list(CORE_TOOLS)
    assert pick_modules_to_edit(core_tools, eval_batch) == list(CORE_TOOLS)
    assert pick_modules_to_edit(both, eval_batch) == [WRITING_CODE_KEY, *CORE_TOOLS]
    assert pick_modules_to_edit(search_only, eval_batch) == ["glean_search"]

    rules_and_core = TeacherStudentAdapter(**kwargs, editable_modules=[*CORE_TOOLS, RULES_EXT_KEY])
    rules_only = TeacherStudentAdapter(**kwargs, editable_modules=[RULES_EXT_KEY])
    assert pick_modules_to_edit(rules_and_core) == [RULES_EXT_KEY, *CORE_TOOLS]
    assert pick_modules_to_edit(rules_and_core, eval_batch) == [RULES_EXT_KEY, *CORE_TOOLS]
    assert pick_modules_to_edit(rules_only) == [RULES_EXT_KEY]
    assert pick_modules_to_edit(rules_only, eval_batch) == [RULES_EXT_KEY]


def test_modules_after_tool_choice_keeps_only_the_named_descriptions():
    parent = Candidate(
        model="gpt",
        prompt_modules={"glean_search": "search", "discover": "discover", RULES_EXT_KEY: ""},
        module_specs={},
        global_token_cap=4096,
        baseline_prompt_hash="h",
    )
    examples = {
        "glean_search": [
            {
                "Inputs": {"query": "q"},
                "Generated Outputs": {"teacher_tools": ["Glean Search"], "student_tools": ["Discover"]},
                "Action Inputs": ['teacher Glean Search: {"query": "pto"}'],
                "Feedback": "mismatch",
            }
        ],
        "discover": [],
        RULES_EXT_KEY: [],
    }

    def choose(prompt: str) -> str:
        assert "glean_search" in prompt
        assert "discover" in prompt
        assert 'ACTION_INPUT: teacher Glean Search: {"query": "pto"}' in prompt
        return "discover\nglean_search"

    chosen = modules_after_tool_choice(
        choose,
        parent,
        [RULES_EXT_KEY, "glean_search", "discover"],
        examples,
    )
    # Non-core modules keep the first offspring slots; chosen tools follow in reflector order.
    assert chosen == [RULES_EXT_KEY, "discover", "glean_search"]

    def unused(_prompt: str) -> str:
        raise AssertionError("no examples, so the reflector is not called")

    assert modules_after_tool_choice(unused, parent, ["glean_search", RULES_EXT_KEY], {"glean_search": []}) == [
        RULES_EXT_KEY
    ]


def _paired_example(student_tools: list[str], teacher_tools: list[str]) -> dict:
    return {
        "Inputs": {"query": "q"},
        "Generated Outputs": {"teacher_tools": teacher_tools, "student_tools": student_tools},
        "Action Inputs": [],
        "Feedback": "loss",
    }


def test_tool_usage_counts_examples_per_side_with_trace_event_names():
    examples = [
        _paired_example(["Personal Knowledge Vault Retrieve", "Glean Search", "Shell"], ["Glean Document Reader"]),
        _paired_example(["Ask User Questions"], ["Glean Search", "Glean Search"]),
        _paired_example([], ["Glean Document Reader", "Glean Search"]),
    ]
    usage = tool_usage_in_examples(
        ["glean_search", "glean_document_reader", "ask_user_questions", "todo_write"], examples
    )
    assert usage == {
        "glean_search": (1, 2),
        "glean_document_reader": (0, 2),
        "ask_user_questions": (1, 0),
        "todo_write": (0, 0),
    }
    # Either side's invocations count; the threshold never exceeds the example count.
    assert tools_with_evidence(usage, example_count=3) == []
    assert tools_with_evidence(usage, example_count=2) == ["glean_search", "glean_document_reader"]
    assert tools_with_evidence({"discover": (1, 0)}, example_count=1) == ["discover"]


def test_modules_after_tool_choice_drops_tools_nobody_invoked():
    """A description cannot steer a decision the student never reaches.

    In the agentic run the reflector spent three of five offspring rewriting
    ask_user_questions while the student invoked it on 4 of 189 entries and asked
    in prose instead; the val score did not move. Such tools are filtered before
    the reflector picks, and the reflector is shown the counts for the rest.
    """
    parent = Candidate(
        model="gpt",
        prompt_modules={"glean_search": "s", "glean_document_reader": "r", "ask_user_questions": "a"},
        module_specs={},
        global_token_cap=4096,
        baseline_prompt_hash="h",
    )
    examples = [
        _paired_example(["Glean Search"], ["Glean Search", "Glean Document Reader"]),
        _paired_example(["Glean Search"], ["Glean Document Reader"]),
        _paired_example([], ["Glean Search", "Glean Document Reader"]),
        _paired_example(["Ask User Questions"], ["Glean Search"]),
    ]
    high_signal = dict.fromkeys(("glean_search", "glean_document_reader", "ask_user_questions"), examples)

    def choose(prompt: str) -> str:
        assert "### ask_user_questions" not in prompt
        assert "ask_user_questions" not in prompt.split("CURRENT DESCRIPTIONS")[0]
        assert "glean_document_reader" in prompt
        return "ask_user_questions\nglean_document_reader\nglean_search"

    chosen = modules_after_tool_choice(
        choose,
        parent,
        [EXECUTION_DISCIPLINE_KEY, "glean_search", "glean_document_reader", "ask_user_questions"],
        high_signal,
    )
    assert chosen == [EXECUTION_DISCIPLINE_KEY, "glean_document_reader", "glean_search"]


def test_unseen_evalset_policy_reveals_one_training_id_at_a_time():
    loader = ListDataLoader(["v1", "v2", "v3"])
    policy = UnseenEvalSetPolicy()

    assert policy.take_unseen(loader, purpose="reflection and offspring screening") == [0]
    assert policy.take_unseen(loader, purpose="reflection and offspring screening") == [1]
    assert policy.take_unseen(loader, purpose="reflection and offspring screening") == [2]


def test_unseen_evalset_policy_fails_instead_of_reusing_seen_data():
    loader = ListDataLoader(["v1"])
    policy = UnseenEvalSetPolicy()
    policy.take_unseen(loader, purpose="reflection and offspring screening")

    with pytest.raises(RuntimeError, match="No unseen eval sets remain"):
        policy.take_unseen(loader, purpose="reflection and offspring screening")


def test_restarted_run_continues_after_the_versions_it_already_used(tmp_path):
    state_file = tmp_path / "schedule.json"
    loader = ListDataLoader(["v1", "v2", "v3"])

    first = UnseenEvalSetPolicy(state_file=state_file)
    assert first.take_unseen(loader, purpose="screening", attempt=0) == [0]
    assert first.take_unseen(loader, purpose="screening", attempt=1) == [1]

    resumed = UnseenEvalSetPolicy(state_file=state_file)
    assert resumed.take_unseen(loader, purpose="screening", attempt=2) == [2]

    skipped = UnseenEvalSetPolicy(state_file=tmp_path / "skipped.json")
    skipped.skip_consumed_prefix(loader, 2)
    assert skipped.take_unseen(loader, purpose="screening") == [2]


def test_generation_interrupted_before_it_finished_replays_its_slice(tmp_path):
    state_file = tmp_path / "schedule.json"
    loader = ListDataLoader(["v1", "v2", "v3"])

    first = UnseenEvalSetPolicy(state_file=state_file)
    first.take_unseen(loader, purpose="screening", attempt=0)
    first.take_unseen(loader, purpose="screening", attempt=1)

    # The engine checkpoints before a generation starts, so a resumed run
    # repeats the attempt counter of the generation it was killed in.
    resumed = UnseenEvalSetPolicy(state_file=state_file)
    assert resumed.take_unseen(loader, purpose="screening", attempt=1) == [1]
    assert resumed.take_unseen(loader, purpose="screening", attempt=2) == [2]


def test_schedule_tracks_eval_set_versions_not_list_positions(tmp_path):
    state_file = tmp_path / "schedule.json"
    used = [{"eval_set_name": "Medium", "eval_set_version": version} for version in ("20260820", "20260824")]

    first = UnseenEvalSetPolicy(state_file=state_file)
    first.take_unseen(ListDataLoader(used), purpose="screening", attempt=0)
    first.take_unseen(ListDataLoader(used), purpose="screening", attempt=1)

    assert json.loads(state_file.read_text())["consumed"] == ["Medium:20260820"]

    # A restart that prepends a new version must not re-run either used one.
    reordered = [{"eval_set_name": "Medium", "eval_set_version": "20260828"}, *used]
    resumed = UnseenEvalSetPolicy(state_file=state_file)
    assert resumed.take_unseen(ListDataLoader(reordered), purpose="screening", attempt=2) == [0]


def test_stopper_ends_the_run_once_every_training_version_is_used(tmp_path):
    loader = ListDataLoader(["v1", "v2"])
    policy = UnseenEvalSetPolicy(state_file=tmp_path / "schedule.json")
    stopper = TrainingScheduleExhaustedStopper(policy, loader)

    assert not stopper(None)
    policy.take_unseen(loader, purpose="screening", attempt=0)
    policy.take_unseen(loader, purpose="screening", attempt=1)
    assert not stopper(None)

    # Starting a third generation retires the last slice and exhausts the schedule.
    with pytest.raises(RuntimeError, match="No unseen eval sets remain"):
        policy.take_unseen(loader, purpose="screening", attempt=2)
    assert stopper(None)
