from __future__ import annotations

import random
from unittest.mock import MagicMock, patch

import pytest

from glean_gepa.adapter_types import PairwiseJudge
from glean_gepa.al_adapter import ALRunner, Thresholds
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evalcli_client import AGENTIC_JUDGE_TYPE, CORRECTNESS_JUDGE_TYPE
from glean_gepa.evolutionary_proposer import _select_screened_children
from glean_gepa.experiment_config import load_experiment_config, pairwise_judges
from glean_gepa.judge_metrics_util import (
    JUDGE_SPECS,
    PREFERENCE_TIE,
    JudgeAnalysis,
    per_entry_from_analysis_view,
    wait_for_all_judge_metrics,
)
from glean_gepa.objectives.agentic_preference import (
    AGENTIC_PREFERENCE_OBJECTIVE,
    KEEP_PREFERENCE_THRESHOLD,
    KEEP_SAMPLE_SIZE,
    REFLECTION_RANDOM_SAMPLE_SIZE,
    REFLECTION_SAMPLE_SEED,
    STUDENT_PREFERRED_KEY,
    TEACHER_PREFERRED_KEY,
    AgenticPreferenceObjective,
)
from glean_gepa.objectives.utils.agentic_preference_util import (
    AgenticPreferenceAnalysis,
    AgenticPreferenceEntry,
    fetch_paired_preference_traces,
    rationale_from_judge_entries,
)
from glean_gepa.objectives.utils.tool_match_util import ToolMatchEntryMetrics
from glean_gepa.prompt_constants import EXECUTION_DISCIPLINE_KEY, WRITING_CODE_KEY
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter, _StartedPair

THRESHOLDS = Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000)
EVAL_SET = {
    "eval_set_name": "Glean Chat V2 Medium",
    "eval_set_version": "20260806",
    "deployment_ids": ["scio-prod"],
    "status": "active",
}


def _agentic_judge() -> PairwiseJudge:
    spec = JUDGE_SPECS[AGENTIC_PREFERENCE_OBJECTIVE]
    return PairwiseJudge(spec.name, spec.judge_type, spec.run_params, spec.input_mappings)


def _agentic_adapter(evalcli: MagicMock | None = None, **kwargs) -> TeacherStudentAdapter:
    return TeacherStudentAdapter(
        runner=ALRunner(evalcli=evalcli or MagicMock()),
        teacher_model="gpt",
        student_model="fast",
        thresholds=THRESHOLDS,
        objective=AgenticPreferenceObjective(),
        primary_objective=AGENTIC_PREFERENCE_OBJECTIVE,
        composite_weights={AGENTIC_PREFERENCE_OBJECTIVE: 1.0},
        pairwise_judges=[_agentic_judge()],
        screening_kind="high_signal_fix_rate",
        **kwargs,
    )


def _preference_analysis(*, student_eval_id: str = "student-1", teacher_eval_id: str = "teacher-1"):
    return AgenticPreferenceAnalysis(
        teacher_eval_id=teacher_eval_id,
        student_eval_id=student_eval_id,
        per_entry={
            "lost": AgenticPreferenceEntry("lost", student_answer="student lost", teacher_answer="teacher won"),
            "tie": AgenticPreferenceEntry("tie", student_answer="same", teacher_answer="same"),
            "won": AgenticPreferenceEntry("won", student_answer="student won", teacher_answer="teacher lost"),
        },
    )


def _trajectory(
    entry_id: str,
    *,
    preference: float | None,
    eval_set: dict | None = None,
    student_tools: list[str] | None = None,
    teacher_tools: list[str] | None = None,
) -> dict:
    output: dict = {
        "entry_id": entry_id,
        "deployment_id": "scio-prod",
        "query": "q",
        "student_answer": "s",
        "teacher_answer": "t",
        "student_tool_events": list(student_tools or []),
        "teacher_tool_events": list(teacher_tools or []),
        "student_eval_run_id": "student-1",
        "teacher_eval_run_id": "teacher-1",
    }
    if preference is not None:
        output[AGENTIC_PREFERENCE_OBJECTIVE] = preference
    return {
        "data": dict(eval_set or EVAL_SET),
        "output": output,
        "score": 0.0 if preference is None else preference,
        "objective_scores": {AGENTIC_PREFERENCE_OBJECTIVE: preference} if preference is not None else {},
    }


def _judge_entry(*, orientation: str, explanation: str, label: str = "lose", name: str = "multi_dimension_overall"):
    """One ``analyze details`` judge output, shaped like the real payload."""
    return {
        "judgeRunId": "judge-1",
        "outputs": [
            {
                "name": f"judge_pairwise_agentic_{name}",
                "label": label,
                "reasoning": (
                    f"randomized_single_0_10 scoring (5=tie): score=1.00 gap=4.00 orientation={orientation}\n"
                    f'call ({orientation}): {{"explanation": "{explanation}", "preferred": "A", "gap_score": 4}}'
                ),
            }
        ],
    }


@pytest.mark.parametrize(
    ("orientation", "expected"),
    [
        ("A=base,B=test", "Teacher cited sources, Student did not"),
        ("A=test,B=base", "Student cited sources, Teacher did not"),
    ],
)
def test_rationale_resolves_the_randomized_ab_orientation(orientation, expected):
    """The judge shuffles which side is A per entry; unresolved it teaches the wrong lesson."""
    entries = [_judge_entry(orientation=orientation, explanation="Run A cited sources, Run B did not")]

    assert rationale_from_judge_entries(entries) == f"overall (teacher preferred): {expected}"


def test_rationale_skips_outputs_it_cannot_parse():
    entries = [
        {
            "judgeRunId": "judge-1",
            "outputs": [
                {"name": "judge_pairwise_agentic_correctness", "label": "lose", "reasoning": "no payload here"},
                {"name": "judge_pairwise_agentic_correctness", "label": "lose"},
            ],
        }
    ]

    assert rationale_from_judge_entries(entries) == ""


def test_judge_rationales_are_fetched_for_selected_entries_then_cached():
    trajectory = _trajectory("lost", preference=0.1)
    objective = AgenticPreferenceObjective()
    objective.hydrate_reflective_trajectories([trajectory])
    assert "agentic_preference_rate_feedback" not in trajectory["output"]

    evalcli = MagicMock()
    evalcli.get_analysis_details.return_value = [
        {
            "evalSetEntry": {"id": "lost"},
            "judgeRunEntries": [
                _judge_entry(orientation="A=base,B=test", explanation="Run B omitted the deliverable"),
                _judge_entry(
                    orientation="A=base,B=test",
                    explanation="Run B misattributed the owner",
                    name="correctness",
                ),
            ],
        }
    ]
    objective.evalcli = evalcli

    objective.hydrate_reflective_trajectories([trajectory])
    objective.hydrate_reflective_trajectories([trajectory])

    assert trajectory["output"]["agentic_preference_rate_feedback"] == (
        "overall (teacher preferred): Student omitted the deliverable\n"
        "correctness (teacher preferred): Student misattributed the owner"
    )
    evalcli.get_analysis_details.assert_called_once_with(
        entry_ids=["lost"], eval_run_ids=["student-1"], deployment_id="scio-prod"
    )
    feedback = objective.build_reflective_example(WRITING_CODE_KEY, trajectory, {})["Feedback"]
    assert "Student omitted the deliverable" in feedback


@pytest.mark.parametrize(
    ("n", "kwargs", "keys_only"),
    [
        (10, {}, False),
        (40, {"max_entries": None}, False),
        (3, {}, True),
    ],
    ids=["under_default_cap", "all", "keys_only"],
)
def test_reflection_keeps_every_loss_when_uncapped(n, kwargs, keys_only):
    objective = AgenticPreferenceObjective()
    if keys_only:
        selected, groups = objective._select_mismatch_groups([TEACHER_PREFERRED_KEY] * n, **kwargs)
        assert selected == list(range(n))
    else:
        trajectories = [_trajectory(f"loss-{i:02d}", preference=0.1) for i in range(n)]
        keys = [objective._mismatch_key(trajectory["output"]) for trajectory in trajectories]
        selected, groups = objective._select_mismatch_groups(keys, trajectories=trajectories, **kwargs)
        assert [trajectories[index]["output"]["entry_id"] for index in selected] == [
            f"loss-{i:02d}" for i in range(n)
        ]
    assert groups == [(*TEACHER_PREFERRED_KEY, n)]


def test_reflection_sample_is_stable_across_trajectory_order():
    objective = AgenticPreferenceObjective()
    trajectories = [_trajectory(f"loss-{i:02d}", preference=0.1) for i in range(40)]
    keys = [objective._mismatch_key(trajectory["output"]) for trajectory in trajectories]
    selected = objective._select_mismatch_groups(keys, trajectories=trajectories)[0]
    picked = {trajectories[index]["output"]["entry_id"] for index in selected}

    shuffled = list(reversed(trajectories))
    shuffled_keys = [objective._mismatch_key(trajectory["output"]) for trajectory in shuffled]
    shuffled_selected = objective._select_mismatch_groups(shuffled_keys, trajectories=shuffled)[0]
    shuffled_picked = {shuffled[index]["output"]["entry_id"] for index in shuffled_selected}

    assert picked == shuffled_picked
    assert len(picked) == REFLECTION_RANDOM_SAMPLE_SIZE


def test_reflection_adds_a_keep_sample_without_changing_the_loss_draw():
    objective = AgenticPreferenceObjective()
    losses = [_trajectory(f"loss-{i:02d}", preference=0.1) for i in range(40)]
    keeps = [_trajectory(f"keep-{i:02d}", preference=0.9) for i in range(20)]
    weak_win = _trajectory("weak-win", preference=0.55)
    tie = _trajectory("tie", preference=PREFERENCE_TIE)
    trajectories = losses + keeps + [weak_win, tie]
    keys = [objective._mismatch_key(trajectory["output"]) for trajectory in trajectories]

    selected, groups = objective._select_mismatch_groups(keys, trajectories=trajectories)
    picked = [trajectories[index]["output"]["entry_id"] for index in selected]
    expected_losses = sorted(
        random.Random(REFLECTION_SAMPLE_SEED).sample(
            sorted(f"loss-{i:02d}" for i in range(40)),
            REFLECTION_RANDOM_SAMPLE_SIZE,
        )
    )
    expected_keeps = sorted(
        random.Random(REFLECTION_SAMPLE_SEED).sample(
            sorted(f"keep-{i:02d}" for i in range(20)),
            KEEP_SAMPLE_SIZE,
        )
    )

    assert picked[:REFLECTION_RANDOM_SAMPLE_SIZE] == expected_losses
    assert picked[REFLECTION_RANDOM_SAMPLE_SIZE:] == expected_keeps
    assert "weak-win" not in picked
    assert "tie" not in picked
    assert groups == [
        (*TEACHER_PREFERRED_KEY, REFLECTION_RANDOM_SAMPLE_SIZE),
        (*STUDENT_PREFERRED_KEY, KEEP_SAMPLE_SIZE),
    ]


def test_keep_sample_uses_the_0_6_preference_floor():
    objective = AgenticPreferenceObjective()
    trajectories = [
        _trajectory("loss", preference=0.2),
        _trajectory("keep", preference=KEEP_PREFERENCE_THRESHOLD),
        _trajectory("below-keep", preference=KEEP_PREFERENCE_THRESHOLD - 0.05),
        _trajectory("tie", preference=PREFERENCE_TIE),
    ]
    keys = [objective._mismatch_key(trajectory["output"]) for trajectory in trajectories]

    selected, groups = objective._select_mismatch_groups(keys, trajectories=trajectories)
    picked = [trajectories[index]["output"]["entry_id"] for index in selected]

    assert picked == ["loss", "keep"]
    assert groups == [(*TEACHER_PREFERRED_KEY, 1), (*STUDENT_PREFERRED_KEY, 1)]


def test_reflection_k_caps_losses_but_not_the_keep_budget():
    objective = AgenticPreferenceObjective()
    trajectories = [_trajectory(f"loss-{i:02d}", preference=0.1) for i in range(40)]
    trajectories += [_trajectory(f"keep-{i:02d}", preference=0.9) for i in range(20)]
    keys = [objective._mismatch_key(trajectory["output"]) for trajectory in trajectories]

    selected, groups = objective._select_mismatch_groups(keys, trajectories=trajectories, max_entries=8)
    picked = [trajectories[index]["output"]["entry_id"] for index in selected]

    assert len([entry_id for entry_id in picked if entry_id.startswith("loss-")]) == 8
    assert len([entry_id for entry_id in picked if entry_id.startswith("keep-")]) == KEEP_SAMPLE_SIZE
    assert groups == [(*TEACHER_PREFERRED_KEY, 8), (*STUDENT_PREFERRED_KEY, KEEP_SAMPLE_SIZE)]


def test_tool_sequences_are_joined_onto_the_paired_answers():
    """Reflection needs the tool choices; the judge's dimensions never name a tool."""
    evalcli = MagicMock()
    evalcli.get_analysis_view.return_value = {
        "entries": [
            {
                "entryId": "lost",
                "evalRunEntries": [
                    {"evalRunId": "student-1", "output": "student text"},
                    {"evalRunId": "teacher-1", "output": "teacher text"},
                ],
            },
            {"entryId": "no-spans", "evalRunEntries": []},
        ]
    }
    tool_analysis = MagicMock(
        per_entry={
            "lost": ToolMatchEntryMetrics(
                entry_id="lost",
                student_tools=("Glean Document Reader",),
                teacher_tools=("Glean Search",),
                tools_match=False,
            )
        }
    )

    with patch(
        "glean_gepa.objectives.utils.agentic_preference_util.fetch_eval_run_tool_match_analysis",
        return_value=tool_analysis,
    ) as fetch_tools:
        analysis = fetch_paired_preference_traces(
            object(),
            teacher_eval_id="teacher-1",
            student_eval_id="student-1",
            lookback_days=7,
            evalcli=evalcli,
        )

    assert analysis.per_entry["lost"].student_tools == ("Glean Document Reader",)
    assert analysis.per_entry["lost"].teacher_tools == ("Glean Search",)
    assert analysis.per_entry["no-spans"].student_tools == ()
    assert fetch_tools.call_args.kwargs["lookback_days"] == 7
    # Passing evalcli would additionally pull one trace per mismatching entry for
    # first-call payloads this objective never shows reflection.
    assert "evalcli" not in fetch_tools.call_args.kwargs

    rows = AgenticPreferenceObjective().scored_rows(
        analysis, focused=False, capture_traces=True, query="q", deployment_id="scio-prod"
    )
    lost = next(row for row in rows if row.entry_id == "lost")
    assert lost.output["student_tool_events"] == ["Glean Document Reader"]
    assert lost.output["teacher_tool_calls"] == 1


def test_a_missing_tool_fetch_leaves_the_answers_usable():
    evalcli = MagicMock()
    evalcli.get_analysis_view.return_value = {
        "entries": [{"entryId": "lost", "evalRunEntries": [{"evalRunId": "student-1", "output": "student text"}]}]
    }

    with patch(
        "glean_gepa.objectives.utils.agentic_preference_util.fetch_eval_run_tool_match_analysis",
        side_effect=RuntimeError("agentspan is down"),
    ):
        analysis = fetch_paired_preference_traces(
            object(), teacher_eval_id="teacher-1", student_eval_id="student-1", evalcli=evalcli
        )

    assert analysis.per_entry["lost"].student_answer == "student text"
    assert analysis.per_entry["lost"].student_tools == ()


def test_core_tool_modules_are_editable_only_for_tools_a_loss_implicates():
    """The proposer drops every core-tool module this does not name."""
    objective = AgenticPreferenceObjective()
    trajectories = [
        _trajectory(
            "lost",
            preference=0.2,
            teacher_tools=["Glean Search"],
            student_tools=["Glean Document Reader"],
        ),
        _trajectory("won", preference=0.9, teacher_tools=["Delegate"], student_tools=["Todo Write"]),
    ]

    keys = objective.high_signal_core_tool_keys(trajectories)

    assert keys == ["glean_search", "glean_document_reader"]


def test_each_core_tool_module_sees_all_selected_losses():
    objective = AgenticPreferenceObjective()
    search_loss = _trajectory("search-loss", preference=0.2, teacher_tools=["Glean Search"], student_tools=["Write"])
    delegate_loss = _trajectory("delegate-loss", preference=0.1, teacher_tools=["Delegate"], student_tools=["Write"])
    selected = [search_loss, delegate_loss]
    keys = [TEACHER_PREFERRED_KEY, TEACHER_PREFERRED_KEY]

    def chosen(component: str) -> list[str]:
        picked = objective._component_trajectories(component, selected, keys, trajectories=selected, mismatch_keys=keys)
        return [trajectory["output"]["entry_id"] for trajectory in picked]

    assert chosen("glean_search") == chosen("delegate") == chosen(WRITING_CODE_KEY) == ["search-loss", "delegate-loss"]


def test_the_first_tool_divergence_reaches_the_reflector():
    objective = AgenticPreferenceObjective()
    trajectory = _trajectory("lost", preference=0.2, teacher_tools=["Glean Search"], student_tools=["Write"])

    example = objective.build_reflective_example("glean_search", trajectory, {})

    assert "opened with Glean Search" in example["Feedback"]
    assert "student opened with Write" in example["Feedback"]
    assert example["Feedback"].startswith("LOSS:")
    assert example["Generated Outputs"]["student_tools"] == ["Write"]


def test_keep_examples_are_labeled_and_skip_first_tool_copying():
    objective = AgenticPreferenceObjective()
    trajectory = _trajectory(
        "won",
        preference=0.9,
        teacher_tools=["Glean Search"],
        student_tools=["Write"],
    )

    search = objective.build_reflective_example("glean_search", trajectory, {})
    assert search["Feedback"].startswith("KEEP:")
    assert "already preferred the student" in search["Feedback"]
    assert "opened with" not in search["Feedback"]

    discipline = objective.build_reflective_example(EXECUTION_DISCIPLINE_KEY, trajectory, {})
    assert discipline["Feedback"] == search["Feedback"]


def test_agentic_pack_allows_preference_as_primary():
    config = load_experiment_config("teacher_student_agentic")

    assert config.primary_objective == AGENTIC_PREFERENCE_OBJECTIVE
    assert config.frontier_type == "objective"
    assert config.screening == {
        "kind": "high_signal_fix_rate",
        "threshold": 0.449,
        "high_signal": "teacher_preferred",
    }
    assert config.search["reflection_samples"] == 25
    judges = {judge.name: judge.judge_type for judge in pairwise_judges(config)}
    assert judges == {AGENTIC_PREFERENCE_OBJECTIVE: AGENTIC_JUDGE_TYPE}


def test_adapter_honors_reflection_samples():
    adapter = _agentic_adapter()
    trajectories = [_trajectory(f"loss-{i:02d}", preference=0.1) for i in range(40)]
    batch = GleanEvaluationBatch(outputs=[], scores=[0.0] * 40, trajectories=trajectories)

    default = adapter.objective.make_reflective_dataset(
        {}, batch, [WRITING_CODE_KEY], adapter.objective.build_reflective_example
    )
    assert len(default[WRITING_CODE_KEY]) == REFLECTION_RANDOM_SAMPLE_SIZE
    capped = adapter.make_reflective_dataset({}, batch, [WRITING_CODE_KEY], k=8)
    assert len(capped[WRITING_CODE_KEY]) == 8
    unbounded = adapter.make_reflective_dataset({}, batch, [WRITING_CODE_KEY], k=None)
    assert len(unbounded[WRITING_CODE_KEY]) == 40

    mixed = [_trajectory(f"loss-{i:02d}", preference=0.1) for i in range(40)]
    mixed += [_trajectory(f"keep-{i:02d}", preference=0.9) for i in range(20)]
    mixed_batch = GleanEvaluationBatch(outputs=[], scores=[0.0] * 60, trajectories=mixed)
    mixed_capped = adapter.make_reflective_dataset({}, mixed_batch, [WRITING_CODE_KEY], k=8)
    assert len(mixed_capped[WRITING_CODE_KEY]) == 8 + KEEP_SAMPLE_SIZE
    mixed_unbounded = adapter.make_reflective_dataset({}, mixed_batch, [WRITING_CODE_KEY], k=None)
    assert len(mixed_unbounded[WRITING_CODE_KEY]) == 60
    assert any(example["Feedback"].startswith("KEEP:") for example in mixed_capped[WRITING_CODE_KEY])


def test_analysis_view_scores_use_the_judges_own_scale():
    """A raw agentic 1.0 is one point out of ten, not an already-normalized win."""
    view = {
        "entries": [
            {
                "entryId": "lost-badly",
                "evalRunEntries": [
                    {
                        "evalRunId": "student-1",
                        "metadata": {
                            "judgeScores": {"judge-1": 1.0},
                            "judgeLabels": {"judge-1": ["lose"]},
                        },
                    }
                ],
            },
            {
                "entryId": "tie",
                "evalRunEntries": [
                    {"evalRunId": "student-1", "metadata": {"judgeScores": {"judge-1": 5.0}}},
                ],
            },
            {
                "entryId": "win",
                "evalRunEntries": [
                    {"evalRunId": "student-1", "metadata": {"judgeScores": {"judge-1": 10.0}}},
                ],
            },
            {
                "entryId": "eval-run-failed",
                "evalRunEntries": [
                    {"evalRunId": "student-1", "metadata": {"judgeScores": {"judge-1": None}}},
                ],
            },
        ]
    }

    scores, feedback = per_entry_from_analysis_view(
        view, eval_id="student-1", judge_run_id="judge-1", judge_type=AGENTIC_JUDGE_TYPE
    )

    assert scores == {"lost-badly": pytest.approx(0.1), "tie": 0.5, "win": 1.0}
    assert feedback["lost-badly"] == "lose"
    assert "eval-run-failed" not in scores

    correctness_view = {
        "entries": [
            {
                "entryId": "ok",
                "evalRunEntries": [
                    {"evalRunId": "student-1", "metadata": {"judgeScores": {"judge-1": 0.75}}},
                ],
            },
            {
                "entryId": "perfect",
                "evalRunEntries": [
                    {"evalRunId": "student-1", "metadata": {"judgeScores": {"judge-1": 1.0}}},
                ],
            },
        ]
    }
    correctness_scores, _feedback = per_entry_from_analysis_view(
        correctness_view, eval_id="student-1", judge_run_id="judge-1", judge_type=CORRECTNESS_JUDGE_TYPE
    )
    assert correctness_scores == {"ok": 0.75, "perfect": 1.0}


def test_wait_for_judge_metrics_loads_per_entry_from_the_analysis_view():
    evalcli = MagicMock()
    evalcli.get_eval_metrics.return_value = {
        "judgeMetrics": {
            "totalEntries": 1,
            "missingEntries": 0,
            "AGENTIC_JUDGE": {"passRate": 0.4, "sampleSize": 1, "judgeRunId": "judge-1"},
        }
    }
    evalcli.get_analysis_view.return_value = {
        "entries": [
            {
                "entryId": "lost",
                "evalRunEntries": [
                    {"evalRunId": "student-1", "metadata": {"judgeScores": {"judge-1": 2.0}}},
                ],
            }
        ]
    }

    analysis = wait_for_all_judge_metrics(
        evalcli,
        (("student-1", AGENTIC_JUDGE_TYPE, "judge-1", "teacher-1"),),
        poll_interval_sec=0,
    )[("student-1", AGENTIC_JUDGE_TYPE, "teacher-1")]

    assert analysis.aggregate == 0.4
    assert analysis.per_entry == {"lost": 0.2}
    evalcli.get_analysis_view.assert_called_once_with("student-1", base_eval_id="teacher-1")


def test_teacher_preferred_entries_are_high_signal_and_ties_are_not():
    objective = AgenticPreferenceObjective()
    adapter = _agentic_adapter()
    batch = GleanEvaluationBatch(
        outputs=[],
        scores=[0.2, 0.5, 0.8, 0.4],
        trajectories=[
            _trajectory("lost", preference=0.2),
            _trajectory("tie", preference=0.5),
            _trajectory("won", preference=0.8),
            _trajectory("aggregate-only", preference=None),
        ],
    )

    assert objective.is_high_signal({"agentic_preference_rate": 0.2})
    assert not objective.is_high_signal({"agentic_preference_rate": 0.5})
    assert not objective.is_high_signal({"agentic_preference_rate": 0.8})
    assert not objective.is_high_signal({})
    focused = adapter.high_signal_batch(batch)
    assert len(focused) == 1
    assert focused[0]["eval_entry_ids"] == ["lost"]


def test_aggregate_judge_score_is_not_copied_onto_entry_outputs():
    adapter = _agentic_adapter()
    adapter._analysis_cache[("teacher-1", "student-1")] = _preference_analysis()
    adapter._judge_cache[("student-1", "teacher-1", AGENTIC_JUDGE_TYPE)] = JudgeAnalysis(
        eval_id="student-1",
        aggregate=0.2,
        per_entry={},
        judge_type=AGENTIC_JUDGE_TYPE,
    )

    result = adapter._finish_batch_evals(
        [_StartedPair(al_data_inst=EVAL_SET, teacher_eval_id="teacher-1", student_eval_id="student-1")],
        capture_traces=True,
    )

    assert all(AGENTIC_PREFERENCE_OBJECTIVE not in output for output in result.outputs)
    assert adapter.high_signal_batch(result) == []


def test_high_signal_screen_uses_agentic_preference_not_correctness():
    adapter = _agentic_adapter()
    parent = GleanEvaluationBatch(
        outputs=[],
        scores=[0.0],
        trajectories=[_trajectory("lost", preference=0.2)],
        summary={AGENTIC_PREFERENCE_OBJECTIVE: 0.2, "correctness": 0.99},
    )
    fail = GleanEvaluationBatch(
        outputs=[],
        scores=[0.0],
        trajectories=[_trajectory("lost", preference=0.0)],
        summary={AGENTIC_PREFERENCE_OBJECTIVE: 0.79, "correctness": 0.99},
    )
    pass_eval = GleanEvaluationBatch(
        outputs=[],
        scores=[1.0],
        trajectories=[_trajectory("lost", preference=1.0)],
        summary={AGENTIC_PREFERENCE_OBJECTIVE: 0.80, "correctness": 0.0},
    )
    keep, reject = object(), object()

    kept = _select_screened_children(
        adapter,
        parent,
        [reject, keep],  # type: ignore[arg-type]
        [fail, pass_eval],
        use_high_signal_gate=True,
        high_signal_screen_threshold=0.80,
    )

    assert [(child, score) for child, _evaluation, score in kept] == [(keep, 0.80)]


def test_focused_and_full_evals_both_start_the_agentic_judge():
    evalcli = MagicMock()
    evalcli.find_judge_run_id.return_value = None
    created: list[str] = []

    def create_judge_run(**kwargs):
        created.append(kwargs["judge_type"])
        return f"judge-{kwargs['judge_type']}"

    evalcli.create_judge_run.side_effect = create_judge_run
    adapter = _agentic_adapter(evalcli)
    focused = _StartedPair(
        al_data_inst={**EVAL_SET, "eval_entry_ids": ["lost"]},
        teacher_eval_id="teacher-1",
        student_eval_id="student-1",
    )
    full = _StartedPair(al_data_inst=EVAL_SET, teacher_eval_id="teacher-1", student_eval_id="student-2")
    val = _StartedPair(
        al_data_inst={**EVAL_SET, "validation_only": True},
        teacher_eval_id="teacher-1",
        student_eval_id="student-3",
    )

    with patch(
        "glean_gepa.teacher_student_adapter.wait_for_all_judge_metrics",
        side_effect=lambda _evalcli, pending, **_kwargs: {
            (eval_id, judge_type, base_eval_id): JudgeAnalysis(
                eval_id=eval_id, aggregate=0.4, per_entry={}, judge_type=judge_type
            )
            for eval_id, judge_type, _judge_run_id, base_eval_id in pending
        },
    ):
        adapter._await_judge_metrics(adapter._start_judges([focused]))
        adapter._await_judge_metrics(adapter._start_judges([full]))
        adapter._await_judge_metrics(adapter._start_judges([val]))

    assert created == [AGENTIC_JUDGE_TYPE, AGENTIC_JUDGE_TYPE, AGENTIC_JUDGE_TYPE]


def test_focused_screen_keeps_the_agentic_judge_mean():
    adapter = _agentic_adapter()
    adapter._analysis_cache[("teacher-1", "student-1")] = _preference_analysis()
    adapter._judge_cache[("student-1", "teacher-1", AGENTIC_JUDGE_TYPE)] = JudgeAnalysis(
        eval_id="student-1",
        aggregate=0.5,
        per_entry={"lost": 0.2, "tie": 0.5, "won": 0.8},
        judge_type=AGENTIC_JUDGE_TYPE,
    )
    focused = {
        **EVAL_SET,
        "eval_entry_ids": ["lost", "tie", "won"],
    }

    result = adapter._finish_batch_evals(
        [_StartedPair(al_data_inst=focused, teacher_eval_id="teacher-1", student_eval_id="student-1")],
        capture_traces=True,
    )

    assert result.summary is not None
    assert result.summary[AGENTIC_PREFERENCE_OBJECTIVE] == pytest.approx((0.2 + 0.5 + 0.8) / 3)
    assert adapter.child_screen_score(
        GleanEvaluationBatch(outputs=[], scores=[0.0], trajectories=[_trajectory("lost", preference=0.2)]),
        result,
    ) == pytest.approx((0.2 + 0.5 + 0.8) / 3)
