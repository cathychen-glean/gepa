from __future__ import annotations

import random
from unittest.mock import MagicMock, patch

import pytest

from glean_gepa.adapter_types import PairwiseJudge
from glean_gepa.al_adapter import ALRunner, Thresholds
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evalcli_client import AGENTIC_JUDGE_TYPE
from glean_gepa.evolutionary_proposer import _select_screened_children
from glean_gepa.experiment_config import load_experiment_config, pairwise_judges
from glean_gepa.judge_metrics_util import JUDGE_SPECS, PREFERENCE_TIE, JudgeAnalysis, per_entry_from_analysis_view
from glean_gepa.objectives.agentic_preference import (
    AGENTIC_PREFERENCE_OBJECTIVE,
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
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter, _StartedPair

THRESHOLDS = Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000)
EVAL_SET = {
    "eval_set_name": "Glean Chat V2 Medium",
    "eval_set_version": "20260806",
    "deployment_ids": ["scio-prod"],
    "status": "active",
}


def _agentic_adapter(evalcli: MagicMock | None = None) -> TeacherStudentAdapter:
    spec = JUDGE_SPECS[AGENTIC_PREFERENCE_OBJECTIVE]
    return TeacherStudentAdapter(
        runner=ALRunner(evalcli=evalcli or MagicMock()),
        teacher_model="gpt",
        student_model="fast",
        thresholds=THRESHOLDS,
        objective=AgenticPreferenceObjective(),
        primary_objective=AGENTIC_PREFERENCE_OBJECTIVE,
        composite_weights={AGENTIC_PREFERENCE_OBJECTIVE: 1.0},
        pairwise_judges=[PairwiseJudge(spec.name, spec.judge_type, spec.run_params, spec.input_mappings)],
        screening_kind="high_signal_fix_rate",
    )


def _trajectory(
    entry_id: str,
    *,
    preference: float | None,
    student_tools: list[str] | None = None,
    teacher_tools: list[str] | None = None,
    judge_run_id: str = "judge-1",
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
        "judge_run_id": judge_run_id,
    }
    if preference is not None:
        output[AGENTIC_PREFERENCE_OBJECTIVE] = preference
    return {
        "data": dict(EVAL_SET),
        "output": output,
        "score": 0.0 if preference is None else preference,
        "objective_scores": {AGENTIC_PREFERENCE_OBJECTIVE: preference} if preference is not None else {},
    }


def _judge_entry(
    *, orientation: str, explanation: str, name: str = "multi_dimension_overall", judge_run_id: str = "judge-1"
):
    return {
        "judgeRunId": judge_run_id,
        "outputs": [
            {
                "name": f"judge_pairwise_agentic_{name}",
                "label": "lose",
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
    """The judge shuffles which side is A; unresolved it teaches the wrong lesson."""
    entries = [_judge_entry(orientation=orientation, explanation="Run A cited sources, Run B did not")]
    assert rationale_from_judge_entries(entries) == f"overall (teacher preferred): {expected}"
    assert (
        rationale_from_judge_entries(
            entries + [_judge_entry(orientation=orientation, explanation="stale", judge_run_id="judge-stale")],
            judge_run_id="judge-1",
        )
        == f"overall (teacher preferred): {expected}"
    )


def test_judge_rationales_are_fetched_for_the_selected_run_then_cached():
    trajectory = _trajectory("lost", preference=0.1)
    objective = AgenticPreferenceObjective()
    objective.hydrate_reflective_trajectories([trajectory])
    assert "agentic_preference_rate_feedback" not in trajectory["output"]

    evalcli = MagicMock()
    evalcli.get_analysis_details.side_effect = [
        [
            {
                "evalSetEntry": {"id": "lost"},
                "judgeRunEntries": [
                    _judge_entry(orientation="A=base,B=test", explanation="Run B omitted the deliverable"),
                    _judge_entry(
                        orientation="A=base,B=test", explanation="Run B misattributed the owner", name="correctness"
                    ),
                    _judge_entry(
                        orientation="A=base,B=test",
                        explanation="Run B from a stale re-judge",
                        judge_run_id="judge-stale",
                    ),
                ],
            }
        ],
        [
            {
                "evalSetEntry": {"id": "lost"},
                "judgeRunEntries": [
                    _judge_entry(
                        orientation="A=base,B=test",
                        explanation="Run B from a stale re-judge",
                        judge_run_id="judge-stale",
                    ),
                ],
            }
        ],
    ]
    objective.evalcli = evalcli
    objective.hydrate_reflective_trajectories([trajectory])
    objective.hydrate_reflective_trajectories([trajectory])
    assert "Student omitted the deliverable" in trajectory["output"]["agentic_preference_rate_feedback"]
    assert "stale re-judge" not in trajectory["output"]["agentic_preference_rate_feedback"]
    evalcli.get_analysis_details.assert_called_once()

    trajectory["output"]["judge_run_id"] = "judge-stale"
    trajectory["output"].pop("agentic_preference_rate_feedback")
    objective.hydrate_reflective_trajectories([trajectory])
    assert "stale re-judge" in trajectory["output"]["agentic_preference_rate_feedback"]
    assert evalcli.get_analysis_details.call_count == 2


def test_reflection_samples_losses_independently_of_keeps():
    objective = AgenticPreferenceObjective()
    losses = [_trajectory(f"loss-{i:02d}", preference=0.1) for i in range(40)]
    keeps = [_trajectory(f"keep-{i:02d}", preference=0.9) for i in range(20)]
    extra = [_trajectory("weak-win", preference=0.55), _trajectory("tie", preference=PREFERENCE_TIE)]
    trajectories = losses + keeps + extra
    keys = [objective._mismatch_key(trajectory["output"]) for trajectory in trajectories]
    selected, groups = objective._select_mismatch_groups(keys, trajectories=trajectories)
    picked = [trajectories[index]["output"]["entry_id"] for index in selected]
    expected_losses = sorted(
        random.Random(REFLECTION_SAMPLE_SEED).sample(
            sorted(f"loss-{i:02d}" for i in range(40)), REFLECTION_RANDOM_SAMPLE_SIZE
        )
    )
    expected_keeps = sorted(
        random.Random(REFLECTION_SAMPLE_SEED).sample(sorted(f"keep-{i:02d}" for i in range(20)), KEEP_SAMPLE_SIZE)
    )
    assert picked[:REFLECTION_RANDOM_SAMPLE_SIZE] == expected_losses
    assert picked[REFLECTION_RANDOM_SAMPLE_SIZE:] == expected_keeps
    assert "weak-win" not in picked and "tie" not in picked
    assert groups == [
        (*TEACHER_PREFERRED_KEY, REFLECTION_RANDOM_SAMPLE_SIZE),
        (*STUDENT_PREFERRED_KEY, KEEP_SAMPLE_SIZE),
    ]

    capped, capped_groups = objective._select_mismatch_groups(keys, trajectories=trajectories, max_entries=8)
    capped_ids = [trajectories[index]["output"]["entry_id"] for index in capped]
    assert len([entry_id for entry_id in capped_ids if entry_id.startswith("loss-")]) == 8
    assert len([entry_id for entry_id in capped_ids if entry_id.startswith("keep-")]) == KEEP_SAMPLE_SIZE
    assert capped_groups == [(*TEACHER_PREFERRED_KEY, 8), (*STUDENT_PREFERRED_KEY, KEEP_SAMPLE_SIZE)]


def test_paired_traces_join_tools_without_blocking_on_a_missing_fetch():
    evalcli = MagicMock()
    evalcli.get_analysis_view.return_value = {
        "entries": [
            {
                "entryId": "lost",
                "evalRunEntries": [
                    {"evalRunId": "student-1", "output": "student text"},
                    {"evalRunId": "teacher-1", "output": "teacher text"},
                ],
            }
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
            object(), teacher_eval_id="teacher-1", student_eval_id="student-1", lookback_days=7, evalcli=evalcli
        )
    assert analysis.per_entry["lost"].student_tools == ("Glean Document Reader",)
    assert "evalcli" not in fetch_tools.call_args.kwargs

    with patch(
        "glean_gepa.objectives.utils.agentic_preference_util.fetch_eval_run_tool_match_analysis",
        side_effect=RuntimeError("agentspan is down"),
    ):
        fallback = fetch_paired_preference_traces(
            object(), teacher_eval_id="teacher-1", student_eval_id="student-1", evalcli=evalcli
        )
    assert fallback.per_entry["lost"].student_answer == "student text"
    assert fallback.per_entry["lost"].student_tools == ()


def test_reflection_follows_losses_not_wins():
    objective = AgenticPreferenceObjective()
    lost = _trajectory(
        "lost",
        preference=0.2,
        teacher_tools=["Glean Search", "Glean Document Reader"],
        student_tools=["Write", "Glean Search"],
    )
    won = _trajectory("won", preference=0.9, teacher_tools=["Delegate", "Todo Write"], student_tools=["Todo Write"])
    assert objective.high_signal_core_tool_keys([lost, won]) == ["glean_search", "glean_document_reader"]
    loss = objective.build_reflective_example("glean_search", lost, {})
    keep = objective.build_reflective_example("glean_search", won, {})
    assert loss["Feedback"].startswith("LOSS:")
    assert keep["Feedback"].startswith("KEEP:")


def test_agentic_pack_wires_preference_as_primary(tmp_path):
    path = tmp_path / "mode.yaml"
    path.write_text("schema_version: 1\nmode: teacher_student\npacks: [agentic]\n")
    config = load_experiment_config(path)
    assert config.primary_objective == AGENTIC_PREFERENCE_OBJECTIVE
    judges = {judge.name: judge.judge_type for judge in pairwise_judges(config)}
    assert judges == {AGENTIC_PREFERENCE_OBJECTIVE: AGENTIC_JUDGE_TYPE}


def test_analysis_view_treats_a_raw_one_as_an_agentic_loss():
    view = {
        "entries": [
            {
                "entryId": "lost-badly",
                "evalRunEntries": [
                    {
                        "evalRunId": "student-1",
                        "metadata": {"judgeScores": {"judge-1": 1.0}, "judgeLabels": {"judge-1": ["lose"]}},
                    },
                ],
            },
            {
                "entryId": "eval-run-failed",
                "evalRunEntries": [{"evalRunId": "student-1", "metadata": {"judgeScores": {"judge-1": None}}}],
            },
        ]
    }
    scores, feedback = per_entry_from_analysis_view(
        view, eval_id="student-1", judge_run_id="judge-1", judge_type=AGENTIC_JUDGE_TYPE
    )
    assert scores == {"lost-badly": pytest.approx(0.1)}
    assert feedback["lost-badly"] == "lose"
    assert "eval-run-failed" not in scores


def test_high_signal_and_screen_use_preference_not_correctness():
    objective = AgenticPreferenceObjective()
    adapter = _agentic_adapter()
    batch = GleanEvaluationBatch(
        outputs=[],
        scores=[0.2, 0.5, 0.8],
        trajectories=[
            _trajectory("lost", preference=0.2),
            _trajectory("tie", preference=0.5),
            _trajectory("won", preference=0.8),
        ],
    )
    assert objective.is_high_signal({"agentic_preference_rate": 0.2})
    assert not objective.is_high_signal({"agentic_preference_rate": 0.5})
    focused = adapter.high_signal_batch(batch)
    assert focused[0]["eval_entry_ids"] == ["lost"]

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
        [reject, keep],
        [fail, pass_eval],  # type: ignore[arg-type]
        use_high_signal_gate=True,
        high_signal_screen_threshold=0.80,
    )
    assert [(child, score) for child, _evaluation, score in kept] == [(keep, 0.80)]


def test_focused_and_full_evals_start_agentic_and_keep_its_mean():
    evalcli = MagicMock()
    evalcli.find_judge_run_id.return_value = None
    created: list[str] = []
    evalcli.create_judge_run.side_effect = (
        lambda **kwargs: created.append(kwargs["judge_type"]) or f"judge-{kwargs['judge_type']}"
    )
    adapter = _agentic_adapter(evalcli)
    focused = _StartedPair(
        al_data_inst={**EVAL_SET, "eval_entry_ids": ["lost", "tie", "won"]},
        teacher_eval_id="teacher-1",
        student_eval_id="student-1",
    )
    full = _StartedPair(al_data_inst=EVAL_SET, teacher_eval_id="teacher-1", student_eval_id="student-2")
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
    assert created == [AGENTIC_JUDGE_TYPE, AGENTIC_JUDGE_TYPE]

    adapter._analysis_cache[("teacher-1", "student-1")] = AgenticPreferenceAnalysis(
        teacher_eval_id="teacher-1",
        student_eval_id="student-1",
        per_entry={
            "lost": AgenticPreferenceEntry("lost", student_answer="s", teacher_answer="t"),
            "tie": AgenticPreferenceEntry("tie", student_answer="s", teacher_answer="t"),
            "won": AgenticPreferenceEntry("won", student_answer="s", teacher_answer="t"),
        },
    )
    adapter._judge_cache[("student-1", "teacher-1", AGENTIC_JUDGE_TYPE)] = JudgeAnalysis(
        eval_id="student-1",
        aggregate=0.5,
        per_entry={"lost": 0.2, "tie": 0.5, "won": 0.8},
        judge_type=AGENTIC_JUDGE_TYPE,
    )
    result = adapter._finish_batch_evals([focused], capture_traces=True)
    assert result.summary is not None
    assert result.summary[AGENTIC_PREFERENCE_OBJECTIVE] == pytest.approx((0.2 + 0.5 + 0.8) / 3)
