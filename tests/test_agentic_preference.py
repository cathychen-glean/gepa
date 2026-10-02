from __future__ import annotations

import json
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
from glean_gepa.objectives.agentic_preference_traces import (
    AgenticPreferenceAnalysis,
    AgenticPreferenceEntry,
    decisive_dimensions,
    fetch_paired_preference_traces,
    rationale_from_judge_entries,
    student_behavior_flags,
)
from glean_gepa.objectives.tool_match import ToolMatchEntryMetrics
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
    assert rationale_from_judge_entries(entries) == f"overall (teacher preferred) [gap=4]: {expected}"
    assert (
        rationale_from_judge_entries(
            entries + [_judge_entry(orientation=orientation, explanation="stale", judge_run_id="judge-stale")],
            judge_run_id="judge-1",
        )
        == f"overall (teacher preferred) [gap=4]: {expected}"
    )


def _judge_output(name: str, orientation: str, payload: dict, *, label: str = "lose", score: float = 2.0) -> dict:
    return {
        "name": f"judge_pairwise_agentic_{name}",
        "label": label,
        "score": score,
        "reasoning": (
            f"randomized_single_0_10 scoring (5=tie): score={score:.2f} gap=3.00 orientation={orientation}\n"
            f"call ({orientation}): {json.dumps(payload)}"
        ),
    }


def test_rationale_keeps_fact_checks_and_coverage_evidence_in_synthesis_order():
    """The rewriter needs the judge's verified claims and requested-vs-delivered actions, not only its prose."""
    orientation = "A=base,B=test"
    entries = [
        {
            "judgeRunId": "judge-1",
            "outputs": [
                _judge_output(
                    "output_readiness",
                    orientation,
                    {"explanation": "Run B opens with a preamble.", "preferred": "A", "gap_score": 1},
                ),
                _judge_output(
                    "correctness",
                    orientation,
                    {
                        "explanation": "Run B states a title no source supports.",
                        "preferred": "A",
                        "gap_score": 3,
                        "verified_claims": [
                            {"claim": "Title is X.", "side": "B", "judgment": "supported", "evidence": "profile"},
                            {
                                "claim": "Role is Y.",
                                "side": "B",
                                "judgment": "contradicted",
                                "evidence": "lookup says Z",
                            },
                            {"claim": "Manager is M.", "side": "A", "judgment": "supported", "evidence": "lookup"},
                        ],
                    },
                ),
                _judge_output(
                    "task_completion",
                    orientation,
                    {
                        "explanation": "Run A covered more of the profile.",
                        "preferred": "A",
                        "gap_score": 1,
                        "task_completion_evidence": {
                            "actions_user_requested": ["Profile the person"],
                            "a_actions_taken": ["Looked up the employee", "Delivered a full profile"],
                            "b_actions_taken": ["Summarized snippets"],
                        },
                    },
                ),
                _judge_output(
                    "multi_dimension_overall",
                    orientation,
                    {"explanation": "Run A wins on correctness and coverage.", "preferred": "A", "gap_score": 4},
                ),
            ],
        }
    ]
    rendered = rationale_from_judge_entries(entries, judge_run_id="judge-1")
    lines = rendered.splitlines()
    assert lines[0].startswith("overall (teacher preferred) [gap=4]: Teacher wins")
    assert lines[1].startswith("task_completion (teacher preferred) [gap=1]: Teacher covered")
    assert lines[2] == "    user requested: Profile the person"
    assert lines[3] == "    Teacher actions taken: Looked up the employee | Delivered a full profile"
    assert lines[4] == "    Student actions taken: Summarized snippets"
    assert lines[5].startswith("correctness (teacher preferred) [gap=3]: Student states a title")
    # Unsupported claims come first: they are the correctness gap.
    assert lines[6] == "    claim [Student] contradicted: Role is Y. — lookup says Z"
    assert lines[7] == "    claim [Student] supported: Title is X. — profile"
    assert lines[8] == "    claim [Teacher] supported: Manager is M. — lookup"
    assert lines[9].startswith("output_readiness (teacher preferred) [gap=1]: Student opens")
    assert decisive_dimensions(rendered, side="teacher") == [
        ("correctness", 3.0),
        ("task_completion", 1.0),
        ("output_readiness", 1.0),
    ]
    assert decisive_dimensions(rendered, side="student") == []


def test_student_behavior_flags_name_prompt_steerable_gaps():
    flags = student_behavior_flags(
        student_answer=(
            "Here's a summary of what I found about the account.\n\nIt is a hosted deployment.\n\n"
            "Would you like me to dig deeper into the deployment details?"
        ),
        teacher_answer="The account is a Glean Hosted deployment under project `x`.\n" + "detail " * 200,
        student_tools=["Personal Knowledge Vault Retrieve"],
        teacher_tools=["Glean Search", "Glean Document Reader", "Glean Search"],
    )
    joined = "\n".join(flags)
    assert "ran no glean_search/glean_document_reader of its own" in joined
    assert "opened a full document with glean_document_reader" in joined
    assert "opens with a preamble" in joined
    assert "closes with an offer/question" in joined
    assert "ends with a question to the user" in joined
    assert "much shorter" in joined

    clean = student_behavior_flags(
        student_answer="The account is a Glean Hosted deployment under project `x`.",
        teacher_answer="The account is a Glean Hosted deployment under project `x`.",
        student_tools=["Glean Search"],
        teacher_tools=["Glean Search"],
    )
    assert clean == []

    asked = student_behavior_flags(
        student_answer="Which recipient should I use?",
        teacher_answer="Drafted message.",
        student_tools=["Ask User Questions"],
        teacher_tools=["Glean Search", "Write"],
    )
    assert any("asked instead of delivering" in flag for flag in asked)
    assert any("wrote or edited a file; student produced no file" in flag for flag in asked)


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
        "glean_gepa.objectives.agentic_preference_traces.fetch_eval_run_tool_match_analysis",
        return_value=tool_analysis,
    ) as fetch_tools:
        analysis = fetch_paired_preference_traces(
            object(), teacher_eval_id="teacher-1", student_eval_id="student-1", lookback_days=7, evalcli=evalcli
        )
    assert analysis.per_entry["lost"].student_tools == ("Glean Document Reader",)
    assert "evalcli" not in fetch_tools.call_args.kwargs

    with patch(
        "glean_gepa.objectives.agentic_preference_traces.fetch_eval_run_tool_match_analysis",
        side_effect=RuntimeError("agentspan is down"),
    ):
        fallback = fetch_paired_preference_traces(
            object(), teacher_eval_id="teacher-1", student_eval_id="student-1", evalcli=evalcli
        )
    assert fallback.per_entry["lost"].student_answer == "student text"
    assert fallback.per_entry["lost"].student_tools == ()


def test_validation_fetch_skips_the_analysis_view():
    evalcli = MagicMock()
    analysis = fetch_paired_preference_traces(
        object(),
        teacher_eval_id="teacher-1",
        student_eval_id="student-1",
        evalcli=evalcli,
        include_action_inputs=False,
    )
    evalcli.get_analysis_view.assert_not_called()
    assert analysis.per_entry == {}


def test_reflection_follows_losses_not_wins():
    objective = AgenticPreferenceObjective()
    lost = _trajectory("lost", preference=0.2)
    won = _trajectory("won", preference=0.9)
    loss = objective.build_reflective_example("glean_search", lost, {})
    keep = objective.build_reflective_example("glean_search", won, {})
    assert loss["Feedback"].startswith("LOSS:")
    assert keep["Feedback"].startswith("KEEP:")


def test_loss_feedback_names_the_deciding_dimension_and_behavior_flags():
    objective = AgenticPreferenceObjective()
    lost = _trajectory("lost", preference=0.1, student_tools=[], teacher_tools=["Glean Search", "Write"])
    lost["output"]["student_answer"] = "Here's what I found.\n\nSome facts.\n\nWant me to pull the full doc?"
    lost["output"]["teacher_answer"] = "Facts. " * 200
    lost["output"]["agentic_preference_rate_feedback"] = "\n".join(
        [
            "overall (teacher preferred) [gap=4]: Teacher delivered.",
            "task_completion (teacher preferred) [gap=4]: Student offered instead.",
            "correctness (tie) [gap=0]: Both fine.",
            "output_readiness (teacher preferred) [gap=1]: Student preamble.",
        ]
    )
    feedback = objective.build_reflective_example("EXECUTION_DISCIPLINE", lost, {})["Feedback"]
    assert "DECIDED BY: task_completion (gap 4), output_readiness (gap 1)." in feedback
    assert "STUDENT BEHAVIOR FLAGS:" in feedback
    assert "ran no glean_search/glean_document_reader of its own" in feedback
    assert "closes with an offer/question" in feedback
    assert "Judge verdict by dimension" in feedback
    assert feedback.index("DECIDED BY") < feedback.index("STUDENT BEHAVIOR FLAGS") < feedback.index("Judge verdict")

    won = _trajectory("won", preference=0.9, student_tools=["Glean Search"], teacher_tools=["Ask User Questions"])
    won["output"]["student_answer"] = "The policy allows 20 days."
    won["output"]["teacher_answer"] = "Which policy do you mean?"
    won["output"]["agentic_preference_rate_feedback"] = (
        "overall (student preferred) [gap=4]: Student delivered.\n"
        "task_completion (student preferred) [gap=4]: Teacher asked instead of answering."
    )
    keep = objective.build_reflective_example("EXECUTION_DISCIPLINE", won, {})["Feedback"]
    assert "WON ON: task_completion (gap 4)." in keep
    assert "student delivered while the teacher run asked a clarifying question" in keep


def test_module_responsibilities_carry_the_judge_model():
    objective = AgenticPreferenceObjective()
    for module in ("EXECUTION_DISCIPLINE", "RULES_EXT", "ask_user_questions", "glean_document_reader"):
        text = objective.reflection_prompt(module)
        assert "HOW THE JUDGE DECIDES" in text
        assert "WHAT USUALLY DECIDES A LOSS" not in text
        assert "WHAT KIND OF EDIT WORKS" in text
        assert "HOW TO READ EACH EXAMPLE" in text
        assert "never mention the teacher in the prompt text" in text
    ask = objective.reflection_prompt("ask_user_questions")
    assert "only the entries where the student actually calls the tool" in ask
    assert "belongs to Execution Discipline" in ask
    assert "stops at search snippets" in objective.reflection_prompt("glean_document_reader")
    discipline = objective.reflection_prompt("EXECUTION_DISCIPLINE")
    assert "minimizes tool loops" in discipline
    # The ask-or-deliver rule is routed to Execution Discipline, with the contradicting
    # "ask when a shaping choice is missing" wording called out for removal.
    assert "asks in prose in its final message far more often than through ask_user_questions" in discipline
    assert "shaping choice (audience, tone, depth, format) is missing, remove or invert" in discipline
    assert "core-tool description" in objective.reflection_prompt("RULES_EXT")


def test_reflective_example_includes_both_roles_tool_inputs():
    objective = AgenticPreferenceObjective()
    lost = _trajectory(
        "lost",
        preference=0.2,
        teacher_tools=["Glean Search", "Write"],
        student_tools=["Discover"],
    )
    lost["output"]["teacher_tool_inputs"] = [["Glean Search", '{"query": "pto"}'], ["Write", '{"path":"a.txt"}']]
    lost["output"]["student_tool_inputs"] = [["Discover", '{"query": "policy"}']]
    example = objective.build_reflective_example("glean_search", lost, {})
    assert example["Action Inputs"] == [
        'teacher Glean Search: {"query": "pto"}',
        'teacher Write: {"path":"a.txt"}',
        'student Discover: {"query": "policy"}',
    ]
    assert example["Generated Outputs"]["teacher_tools"] == list(lost["output"]["teacher_tool_events"])
    assert example["Generated Outputs"]["student_tools"] == list(lost["output"]["student_tool_events"])

    extra = [[f"Tool{i}", f'{{"q": "{i}"}}'] for i in range(4)]
    lost["output"]["teacher_tool_inputs"] = extra
    lost["output"]["student_tool_inputs"] = extra
    capped = objective.build_reflective_example("glean_search", lost, {})
    assert capped["Action Inputs"] == [
        'teacher Tool0: {"q": "0"}',
        'teacher Tool1: {"q": "1"}',
        'teacher Tool2: {"q": "2"}',
        'student Tool0: {"q": "0"}',
        'student Tool1: {"q": "1"}',
        'student Tool2: {"q": "2"}',
    ]


def test_agentic_preference_primary_starts_the_agentic_judge(tmp_path):
    path = tmp_path / "mode.yaml"
    path.write_text(
        "schema_version: 1\n"
        "mode: teacher_student\n"
        f"signals:\n  - name: {AGENTIC_PREFERENCE_OBJECTIVE}\n    source: agentic_preference\n"
        f"objective:\n  primary: {AGENTIC_PREFERENCE_OBJECTIVE}\n  composite:\n    {AGENTIC_PREFERENCE_OBJECTIVE}: 1.0\n"
    )
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
        eval_ids=("teacher-1", "student-1"),
        aggregate=None,
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
