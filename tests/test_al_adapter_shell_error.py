from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from glean_gepa.al_adapter import ALRunner, Candidate, ModuleSpec, Thresholds, extract_shell_action_inputs
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.debug import set_debug
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.focused_evalset import SESSION_BUCKET_TYPE, FocusedEvalSet
from glean_gepa.objectives import AnalysisRequest
from glean_gepa.objectives.shell import EVAL_ANALYSIS_CACHE_SCHEMA_VERSION, ShellSuccessObjective
from glean_gepa.objectives.shell import (
    SHELL_SUCCESS_OBJECTIVE,
    EvalRunShellToolErrorAnalysis,
    ShellToolErrorEntryMetrics,
    ShellToolErrorExample,
    ShellToolErrorMetrics,
)
from glean_gepa.single_model_adapter import SingleModelAdapter, TelemetryPendingError


def _shell_analysis(*, executions: int, per_entry: bool) -> EvalRunShellToolErrorAnalysis:
    aggregate = ShellToolErrorMetrics(
        shell_executions=executions,
        shell_errors=0,
        shell_error_rate=0.0,
        shell_error_pct=0.0,
        recent_error_examples=(),
    )
    entry = ShellToolErrorEntryMetrics(
        entry_id="e1",
        shell_executions=executions,
        shell_errors=0,
        shell_error_rate=0.0,
        shell_error_pct=0.0,
        recent_error_examples=(),
    )
    return EvalRunShellToolErrorAnalysis(
        eval_ids=("run",),
        start_date=date(2026, 8, 1),
        end_date=date(2026, 8, 2),
        aggregate=aggregate,
        per_entry={"e1": entry} if per_entry else {},
        high_signal_entry_ids=(),
    )


def test_shell_cache_hooks_drive_the_shared_helper():
    """Shell keeps only trace fetches and refetches when a hit lacks the per-entry rows asked for."""
    objective = ShellSuccessObjective(bigquery_client=MagicMock())
    aggregate_only = _shell_analysis(executions=3, per_entry=False)
    with_entries = _shell_analysis(executions=3, per_entry=True)
    pending = _shell_analysis(executions=0, per_entry=False)

    # Only trace-detail results are cacheable.
    assert not objective.analysis_is_cacheable(with_entries, AnalysisRequest(detail="aggregate"))
    assert not objective.analysis_is_cacheable(with_entries, AnalysisRequest(detail="per_entry"))
    assert objective.analysis_is_cacheable(with_entries, AnalysisRequest(detail="traces"))
    assert not objective.analysis_is_cacheable(pending, AnalysisRequest(detail="traces"))

    # An aggregate-only hit serves aggregate requests but not per-entry ones.
    assert objective.cache_hit_is_sufficient(aggregate_only, AnalysisRequest(detail="aggregate"))
    assert not objective.cache_hit_is_sufficient(aggregate_only, AnalysisRequest(detail="per_entry"))
    assert objective.cache_hit_is_sufficient(with_entries, AnalysisRequest(detail="traces"))

    # End to end through the helper: validation is never stored, trace is, and a
    # per-entry request after a trace hit reuses it.
    with patch(
        "glean_gepa.objectives.shell.fetch_eval_run_shell_tool_error_analysis",
        side_effect=[aggregate_only, with_entries],
    ) as fetch:
        assert objective.analyze("run", request=AnalysisRequest(detail="aggregate")) is aggregate_only
        assert objective.analysis_cache == {}
        assert objective.analyze("run", request=AnalysisRequest(detail="traces")) is with_entries
        assert objective.analyze("run", request=AnalysisRequest(detail="per_entry")) is with_entries
    assert fetch.call_count == 2


def test_evaluate_uses_shell_error_rate_objective(capsys: pytest.CaptureFixture[str]):
    evalcli = EvalCliClient(binary="/fake/evalcli")
    runner = ALRunner(evalcli=evalcli)
    bigquery_client = MagicMock()
    adapter = SingleModelAdapter(
        runner=runner,
        bigquery_client=bigquery_client,
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )

    batch = [
        {
            "eval_set_name": "AI Answers Small",
            "eval_set_version": "20260403",
            "deployment_ids": ["scio-prod"],
            "status": "active",
            "eval_trace_id": "trace-original",
        }
    ]
    analysis = EvalRunShellToolErrorAnalysis(
        eval_ids=("run_123",),
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
        aggregate=ShellToolErrorMetrics(
            shell_executions=8,
            shell_errors=2,
            shell_error_rate=0.25,
            shell_error_pct=25.0,
            recent_error_examples=(),
        ),
        per_entry={
            "entry-1": ShellToolErrorEntryMetrics(
                entry_id="entry-1",
                shell_executions=4,
                shell_errors=2,
                shell_error_rate=0.5,
                shell_error_pct=50.0,
                recent_error_examples=(
                    ShellToolErrorExample(
                        started_at="2026-08-11T12:00:00Z",
                        project_id="project-1",
                        entry_id="entry-1",
                        eval_id="run_123",
                        run_id="execution-1",
                        trace_id="trace-student-1",
                        span_id="span-1",
                        span_name="Execute Action: Shell",
                        action_id="Shell",
                        action_run_id="call-1",
                        action_status="error",
                        span_status="error",
                        provider_status="failed",
                        output_status_code="1",
                        error_str="command exited with status 1",
                    ),
                ),
                trace_ids=("trace-student-1",),
            )
        },
        high_signal_entry_ids=("entry-1",),
    )

    with (
        patch.object(adapter, "_get_or_run_student_eval", return_value="run_123") as run_eval,
        patch.object(
            adapter.runner.evalcli,
            "get_analysis_trace",
            return_value={
                "trace": {
                    "spans": [
                        {
                            "name": "Execute Action: Shell",
                            "attributes": {
                                "input": {
                                    "strValue": json.dumps(
                                        {"action_input": json.dumps({"command": "python3 broken.py"})}
                                    )
                                },
                                "span.gle": {"strValue": json.dumps({"action": {"action_run_id": "call-1"}})},
                            },
                        }
                    ]
                }
            },
        ) as get_trace,
        patch(
            "glean_gepa.objectives.shell.fetch_eval_run_shell_tool_error_analysis",
            return_value=analysis,
        ),
    ):
        result = adapter.evaluate(batch, {"WRITING_CODE": "test prompt"}, capture_traces=True)

    trace_log = capsys.readouterr().out
    assert "[Trace evaluation] Reading eval-set shell results for AI Answers Small 20260403: run_123" in trace_log
    run_eval.assert_called_once()
    get_trace.assert_called_once()

    assert result.summary[SHELL_SUCCESS_OBJECTIVE] == 0.75
    assert result.summary["high_signal_entry_count"] == 1.0
    assert len(result.outputs) == 1
    assert result.outputs[0]["entry_id"] == "entry-1"
    assert result.outputs[0]["student_tool_errors"] == 2
    assert result.outputs[0]["eval_trace_id"] == "trace-student-1"
    assert result.outputs[0]["shell_action_inputs"] == ['{"command": "python3 broken.py"}']
    assert result.trajectories is not None
    assert len(result.trajectories) == 1

    full_val_result = adapter.evaluate(
        [{**batch[0], "cached_student_eval_run_id": "run_123"}],
        {"WRITING_CODE": "test prompt"},
        capture_traces=False,
    )
    # Full validation has one score per eval-set item. It must use the
    # aggregate (75%), not the high-signal entry's 50% score.
    assert full_val_result.scores == [0.75]
    assert full_val_result.objective_scores == [{SHELL_SUCCESS_OBJECTIVE: 0.75}]
    assert (
        "[Validation] Reading full-validation shell results for AI Answers Small 20260403: run_123"
        in capsys.readouterr().out
    )

    assert result.trajectories[0]["data"]["eval_entry_id"] == "entry-1"
    assert result.trajectories[0]["data"]["eval_run_id"] == "run_123"
    assert result.trajectories[0]["data"]["eval_trace_id"] == "trace-student-1"
    reflective = adapter.make_reflective_dataset({"WRITING_CODE": "test prompt"}, result, ["WRITING_CODE"], k=1)[
        "WRITING_CODE"
    ][0]
    assert reflective["Inputs"]["eval_trace_id"] == "trace-student-1"
    assert reflective["Execution Errors"] == ["command exited with status 1"]
    assert reflective["Action Inputs"] == ['{"command": "python3 broken.py"}']
    assert reflective["Feedback"] == "Resolve the shell execution failures shown above."
    captured_prompts = []

    def reflection_lm(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "NOT_RELEVANT" if len(captured_prompts) == 1 else "rewritten code instructions"

    variants, not_relevant, *_ = adapter.propose_new_texts(
        reflection_lm,
        Candidate(
            model="fast",
            prompt_modules={"WRITING_CODE": "test prompt"},
            module_specs={"WRITING_CODE": ModuleSpec("WRITING_CODE", "free_text", 1024)},
            global_token_cap=4096,
            baseline_prompt_hash="seed",
        ),
        ["WRITING_CODE"],
        [reflective],
    )
    assert variants == ["rewritten code instructions"]
    assert not not_relevant
    assert "NOT_RELEVANT" not in captured_prompts[0]
    assert "TEACHER_ANSWER:" not in captured_prompts[0]
    assert "STUDENT_ANSWER:" not in captured_prompts[0]
    assert "TEACHER_TOOLS:" not in captured_prompts[0]
    assert "STUDENT_TOOLS:" not in captured_prompts[0]
    assert 'ACTION_INPUT: {"command": "python3 broken.py"}' in captured_prompts[0]
    assert "EVAL_TRACE_ID: trace-student-1" in captured_prompts[0]
    assert "METRICS:" not in captured_prompts[0]
    assert "Shell success rate issue:" not in captured_prompts[0]
    assert "FEEDBACK: Resolve the shell execution failures shown above." in captured_prompts[0]
    assert "EXECUTION_ERRORS: ['command exited with status 1']" in captured_prompts[1]


def test_proposals_that_drop_a_render_slot_are_rejected():
    """A Writing Code rewrite without ``{RULES_EXT}`` would silently orphan that module."""
    current = "- stock rule\n{RULES_EXT}\n### Sandbox\n"
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli")),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )
    candidate = Candidate(
        model="fast",
        prompt_modules={"WRITING_CODE": current},
        module_specs={"WRITING_CODE": ModuleSpec("WRITING_CODE", "free_text", 1024)},
        global_token_cap=4096,
        baseline_prompt_hash="seed",
    )
    kept = "- tightened rule\n{RULES_EXT}\n### Sandbox\n"
    prompts: list[str] = []

    def reflection_lm(prompt: str) -> str:
        prompts.append(prompt)
        return "diagnosis" if len(prompts) == 1 else f"- tightened rule\n### Sandbox\n===VARIANT==={kept}"

    variants, _, _ = adapter.propose_new_texts(reflection_lm, candidate, ["WRITING_CODE"], [])

    assert variants == [kept.strip()]
    assert "{RULES_EXT}" in prompts[0]


def test_empty_module_variants_are_bounded_by_the_patches_not_the_token_budget():
    """Two patches for an empty RULES_EXT must not come back as a seven-topic rulebook."""
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli")),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )
    candidate = Candidate(
        model="fast",
        prompt_modules={"RULES_EXT": ""},
        module_specs={"RULES_EXT": ModuleSpec("RULES_EXT", "free_text", 512)},
        global_token_cap=4096,
        baseline_prompt_hash="seed",
    )
    patch_a = "- After writing or editing a file, re-open it and quote only the changed region."
    patch_b = "- Update the existing artifact instead of creating a new file."
    diagnosis = (
        "DIAGNOSIS:\n- unverified edit claims: 6 of 25 LOSS examples\n- new-file drift: 4 of 25\n"
        f"PATCHES:\nBEFORE:\n[empty]\nAFTER:\n{patch_a}\nWHY: a\n\nBEFORE:\n[empty]\nAFTER:\n{patch_b}\nWHY: b"
    )
    tight = f"{patch_a}\n{patch_b}"
    sprawl = "\n".join(f"- {topic}: " + "x" * 150 for topic in ("Artifact", "Claims", "Calibration", "Data", "Access"))
    assert len(sprawl) < 512 * 4  # would have passed the old token-budget filter
    prompts: list[str] = []

    def reflection_lm(prompt: str) -> str:
        prompts.append(prompt)
        return diagnosis if len(prompts) == 1 else f"{sprawl}\n===VARIANT===\n{tight}"

    variants, _, _ = adapter.propose_new_texts(reflection_lm, candidate, ["RULES_EXT"], [])

    assert variants == [tight]
    assert "write 2 rules, one per AFTER snippet" in prompts[1]
    assert "each variant applies those patches and nothing else" in prompts[1]


def test_diagnosis_pass_is_reasked_when_the_reflector_returns_a_rewrite():
    """A rewrite in place of a diagnosis is re-requested once, and the tally reaches the consolidation pass."""
    current = (
        "- Resolve the request in as few tool loops as possible while ensuring accuracy. "
        "Do not follow up for minor doubts; ask only when a missing detail would change the deliverable. "
        "Deliver a best-effort answer and state assumptions when the request is answerable. "
        "Do not offer optional next steps unless the user asked for options. "
        "Keep the final message focused on the result rather than the process you followed."
    )
    rewrite = current.replace("Do not follow up for minor doubts", "Never ask a follow-up question")
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli")),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )
    candidate = Candidate(
        model="fast",
        prompt_modules={"EXECUTION_DISCIPLINE": current},
        module_specs={"EXECUTION_DISCIPLINE": ModuleSpec("EXECUTION_DISCIPLINE", "free_text", 1024)},
        global_token_cap=4096,
        baseline_prompt_hash="seed",
    )
    prompts: list[str] = []
    replies = [
        rewrite,
        "DIAGNOSIS:\n- asks in prose: 4 of 6 LOSS examples\nPATCHES:\nBEFORE: Do not follow up\nAFTER: Deliver\nWHY: w",
        rewrite,
    ]

    def reflection_lm(prompt: str) -> str:
        prompts.append(prompt)
        return replies[len(prompts) - 1]

    variants, _, diagnosis = adapter.propose_new_texts(reflection_lm, candidate, ["EXECUTION_DISCIPLINE"], [])

    assert len(prompts) == 3
    assert prompts[1].startswith("IMPORTANT: a previous attempt")
    assert prompts[1].endswith(prompts[0])
    assert "DIAGNOSIS (failure-mode tally from the first pass):\n- asks in prose: 4 of 6 LOSS examples" in prompts[2]
    assert "SUGGESTIONS:\nBEFORE: Do not follow up" in prompts[2]
    assert diagnosis.startswith("- asks in prose: 4 of 6 LOSS examples")
    assert variants == [rewrite]


def test_high_signal_evaluation_runs_the_uploaded_focused_eval_set():
    evalcli = EvalCliClient(binary="/fake/evalcli")
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=evalcli),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )
    passing_entry = MagicMock(
        shell_executions=1,
        shell_errors=0,
        shell_success_rate=1.0,
        shell_error_pct=0.0,
        recent_error_examples=(),
    )
    analysis = MagicMock(
        eval_id="focused-run",
        aggregate=passing_entry,
        per_entry={"fresh-entry": passing_entry},
        high_signal_entry_ids=(),
    )
    with (
        patch(
            "glean_gepa.focused_evalset.ensure_focused_eval_set",
            return_value=FocusedEvalSet("gepa-high-signal-source", "v1_hs_abc", 1),
        ) as ensure,
        patch.object(adapter, "_get_or_run_student_eval", return_value="focused-run") as run_eval,
        patch.object(adapter, "_get_or_fetch_analysis", return_value=analysis) as get_analysis,
    ):
        result = adapter.evaluate(
            [
                {
                    "eval_set_name": "Source",
                    "eval_set_version": "v1",
                    "deployment_ids": ["prod"],
                    "status": "active",
                    "eval_entry_ids": ["source-entry"],
                }
            ],
            {"WRITING_CODE": "prompt"},
            capture_traces=True,
        )

    ensure.assert_called_once()
    # Focused evals need per-entry scores, not trace-level error examples.
    assert get_analysis.call_args.kwargs["detail"] == "per_entry"
    assert run_eval.call_args.kwargs["eval_set_name"] == "gepa-high-signal-source"
    assert run_eval.call_args.kwargs["eval_set_version"] == "v1_hs_abc"
    assert run_eval.call_args.kwargs["run_label"] == "gepa_high_signal"
    assert result.scores == [1.0]


def test_prepare_high_signal_batch_resolves_upload_entries_from_trace_tables():
    evalcli = EvalCliClient(binary="/fake/evalcli")
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=evalcli),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )
    source_entries = [
        {
            "id": "source-entry",
            "deploymentId": "prod",
            "stt": "session-1",
            "runId": "run-1",
            "traceId": "trace-1",
        }
    ]
    with (
        patch(
            "glean_gepa.single_model_adapter.fetch_high_signal_evalset_entries",
            return_value=source_entries,
        ) as resolve_entries,
        patch(
            "glean_gepa.single_model_adapter.ensure_focused_eval_set",
            return_value=FocusedEvalSet("gepa-high-signal-source", "v1_hs_abc", 1),
        ) as ensure,
    ):
        prepared = adapter.prepare_high_signal_batch(
            [
                {
                    "eval_set_name": "Source",
                    "eval_set_version": "v1",
                    "deployment_ids": ["prod"],
                    "status": "active",
                    "eval_entry_ids": ["source-entry", "unresolved-entry"],
                    "source_eval_run_id": "parent-run",
                }
            ]
        )

    assert prepared is not None
    assert resolve_entries.call_args.kwargs["entry_ids"] == ["source-entry", "unresolved-entry"]
    assert resolve_entries.call_args.kwargs["eval_run_id"] == "parent-run"
    assert ensure.call_args.kwargs["entry_ids"] == ["source-entry"]
    assert ensure.call_args.kwargs["source_entries"] == source_entries
    assert ensure.call_args.kwargs["bucket_type"] == SESSION_BUCKET_TYPE
    assert prepared[0]["focused_eval_set_name"] == "gepa-high-signal-source"
    assert prepared[0]["eval_entry_ids"] == ["source-entry"]


def test_high_signal_batch_retains_the_parent_eval_run_id():
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli")),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )
    parent_eval = GleanEvaluationBatch(
        outputs=[],
        scores=[0.0],
        trajectories=[
            {
                "data": {
                    "eval_set_name": "Source",
                    "eval_set_version": "v1",
                    "deployment_ids": ["prod"],
                    "status": "active",
                    "eval_run_id": "parent-run",
                },
                "output": {"entry_id": "source-entry", "student_eval_run_id": "parent-run"},
                "score": 0.0,
                "objective_scores": {},
            }
        ],
        objective_scores=[{}],
        summary={"shell_success_rate": 0.0},
    )

    focused = adapter.high_signal_batch(parent_eval)

    assert focused[0]["eval_entry_ids"] == ["source-entry"]
    assert focused[0]["source_eval_run_id"] == "parent-run"


def test_high_signal_evaluation_reuses_child_cached_eval_id():
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli")),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )
    passing_entry = MagicMock(
        shell_executions=1,
        shell_errors=0,
        shell_success_rate=1.0,
        shell_error_pct=0.0,
        recent_error_examples=(),
    )
    analysis = MagicMock(
        eval_id="focused-run",
        aggregate=passing_entry,
        per_entry={"fresh-entry": passing_entry},
        high_signal_entry_ids=(),
    )

    with (
        patch.object(adapter, "_get_or_run_student_eval") as run_eval,
        patch.object(adapter, "_get_or_fetch_analysis", return_value=analysis),
    ):
        result = adapter.evaluate(
            [
                {
                    "eval_set_name": "gepa-high-signal-source",
                    "eval_set_version": "v1_hs_abc",
                    "deployment_ids": ["prod"],
                    "status": "active",
                    "eval_entry_ids": ["source-entry"],
                    "focused_eval_set_name": "gepa-high-signal-source",
                    "focused_eval_set_version": "v1_hs_abc",
                    "cached_student_eval_run_id": "focused-run",
                }
            ],
            {"WRITING_CODE": "prompt"},
            capture_traces=True,
        )

    run_eval.assert_not_called()
    assert result.eval_run_ids == [
        {
            "eval_set_name": "gepa-high-signal-source",
            "eval_set_version": "v1_hs_abc",
            "student_eval_run_id": "focused-run",
        }
    ]


def test_high_signal_evaluation_scores_entries_not_shell_calls():
    evalcli = EvalCliClient(binary="/fake/evalcli")
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=evalcli),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )
    # Telemetry for a focused run is keyed by the requested entry ids. The failing
    # entry has 100 calls with 1 error: per-entry pass/fail must score it 0.0, not
    # its 99% call-level success rate.
    passing = ShellToolErrorEntryMetrics(
        entry_id="source-1",
        shell_executions=1,
        shell_errors=0,
        shell_error_rate=0.0,
        shell_error_pct=0.0,
        recent_error_examples=(),
    )
    failing = ShellToolErrorEntryMetrics(
        entry_id="source-2",
        shell_executions=100,
        shell_errors=1,
        shell_error_rate=0.01,
        shell_error_pct=1.0,
        recent_error_examples=(),
    )
    analysis = MagicMock(
        aggregate=MagicMock(shell_success_rate=99 / 101),
        per_entry={passing.entry_id: passing, failing.entry_id: failing},
        high_signal_entry_ids=(),
    )
    with (
        patch.object(adapter, "_get_or_run_student_eval", return_value="focused-run"),
        patch.object(adapter, "_get_or_fetch_analysis", return_value=analysis),
    ):
        result = adapter.evaluate(
            [
                {
                    "eval_set_name": "Source",
                    "eval_set_version": "v1",
                    "deployment_ids": ["prod"],
                    "status": "active",
                    "eval_entry_ids": ["source-1", "source-2"],
                    "focused_eval_set_name": "focused",
                    "focused_eval_set_version": "v1_hs",
                }
            ],
            {"WRITING_CODE": "prompt"},
            capture_traces=True,
        )

    assert result.scores == [1.0, 0.0]
    assert result.summary[SHELL_SUCCESS_OBJECTIVE] == 0.5


def test_extract_shell_action_inputs_matches_action_run_id():
    action_input = json.dumps({"command": "python3 broken.py", "destructive": False})
    trace = {
        "trace": {
            "spans": [
                {
                    "name": "Execute Action: Shell",
                    "attributes": {
                        "input": {"strValue": json.dumps({"action_input": action_input})},
                        "span.gle": {"strValue": json.dumps({"action": {"action_run_id": "call-shell-1"}})},
                    },
                }
            ]
        }
    }

    assert extract_shell_action_inputs(trace) == {"call-shell-1": action_input}


def test_evaluate_logs_fetched_shell_error_rate_and_error(capsys):
    evalcli = EvalCliClient(binary="/fake/evalcli")
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=evalcli),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )
    error_example = ShellToolErrorExample(
        started_at="2026-08-11T12:00:00Z",
        project_id="project-1",
        entry_id="entry-1",
        eval_id="run_123",
        run_id="execution-1",
        trace_id="trace-1",
        span_id="span-1",
        span_name="Execute Action: Shell",
        action_id="Shell",
        action_status="error",
        span_status="error",
        provider_status="failed",
        output_status_code="1",
        error_str="command exited with status 1",
    )
    analysis = EvalRunShellToolErrorAnalysis(
        eval_ids=("run_123",),
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
        aggregate=ShellToolErrorMetrics(
            shell_executions=4,
            shell_errors=1,
            shell_error_rate=0.25,
            shell_error_pct=25.0,
            recent_error_examples=(error_example,),
        ),
        per_entry={},
        high_signal_entry_ids=(),
    )

    set_debug(True)
    try:
        with (
            patch.object(adapter, "_get_or_run_student_eval", return_value="run_123"),
            patch(
                "glean_gepa.objectives.shell.fetch_eval_run_shell_tool_error_analysis",
                return_value=analysis,
            ),
        ):
            adapter.evaluate(
                [
                    {
                        "eval_set_name": "AI Answers Small",
                        "eval_set_version": "20260403",
                        "deployment_ids": ["scio-prod"],
                        "status": "active",
                    }
                ],
                {"WRITING_CODE": "test prompt"},
            )

        output = capsys.readouterr().out
        assert "[Shell Tool] Fetched error rate for eval run_123: 25.00% (1/4)" in output
        assert "[Shell Tool] Error for eval run_123: command exited with status 1" in output
    finally:
        set_debug(False)


def test_capture_traces_reuses_persisted_minimal_error_evidence(tmp_path):
    cache_file = tmp_path / "eval-cache.json"
    runner_cache_file = tmp_path / "eval-run-cache.json"
    error = ShellToolErrorExample(
        started_at="2026-08-11T12:00:00Z",
        project_id="project-1",
        entry_id="entry-1",
        eval_id="run_123",
        run_id="execution-1",
        trace_id="trace-1",
        span_id="span-1",
        span_name="Execute Action: Shell",
        action_id="Shell",
        action_status="error",
        span_status="error",
        provider_status="failed",
        output_status_code="1",
        error_str="command exited with status 1",
    )
    entry = ShellToolErrorEntryMetrics(
        entry_id="entry-1",
        shell_executions=1,
        shell_errors=1,
        shell_error_rate=1.0,
        shell_error_pct=100.0,
        recent_error_examples=(error,),
        trace_ids=("trace-1",),
    )
    analysis = EvalRunShellToolErrorAnalysis(
        eval_ids=("run_123",),
        start_date=date(2026, 8, 11),
        end_date=date(2026, 8, 11),
        aggregate=ShellToolErrorMetrics(
            shell_executions=1,
            shell_errors=1,
            shell_error_rate=1.0,
            shell_error_pct=100.0,
            recent_error_examples=(error,),
        ),
        per_entry={"entry-1": entry},
        high_signal_entry_ids=("entry-1",),
    )
    batch = [
        {
            "eval_set_name": "AI Answers Small",
            "eval_set_version": "20260403",
            "deployment_ids": ["scio-prod"],
            "status": "active",
        }
    ]

    first_evalcli = EvalCliClient(binary="/fake/evalcli")
    first_runner = ALRunner(evalcli=first_evalcli, cache_file=str(runner_cache_file))
    first = SingleModelAdapter(
        runner=first_runner,
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        cache_file=str(cache_file),
    )
    with (
        patch.object(first_evalcli, "create_eval_run", return_value="run_123") as create_eval_run,
        patch.object(first_evalcli, "wait_for_eval_run"),
        patch("glean_gepa.objectives.shell.fetch_eval_run_shell_tool_error_analysis", return_value=analysis) as fetch,
    ):
        without_traces = first.evaluate(batch, {"WRITING_CODE": "prompt"}, capture_traces=False)
    assert without_traces.trajectories is None
    create_eval_run.assert_called_once()
    fetch.assert_called_once()
    assert fetch.call_args.kwargs["include_error_examples"] is False
    assert fetch.call_args.kwargs["include_per_entry"] is False

    second_evalcli = EvalCliClient(binary="/fake/evalcli")
    second_runner = ALRunner(evalcli=second_evalcli, cache_file=str(runner_cache_file))
    second = SingleModelAdapter(
        runner=second_runner,
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        cache_file=str(cache_file),
    )
    with (
        patch.object(second_evalcli, "create_eval_run") as create_eval_run,
        patch.object(second_evalcli, "wait_for_eval_run") as wait_for_eval_run,
        patch("glean_gepa.objectives.shell.fetch_eval_run_shell_tool_error_analysis", return_value=analysis) as fetch,
    ):
        with_traces = second.evaluate(batch, {"WRITING_CODE": "prompt"}, capture_traces=True)

    create_eval_run.assert_not_called()
    # Completed cache hits skip wait; in-flight IDs still wait on resume.
    wait_for_eval_run.assert_not_called()
    fetch.assert_called_once()
    assert fetch.call_args.kwargs["include_error_examples"] is True
    assert fetch.call_args.kwargs["include_per_entry"] is True
    assert with_traces.trajectories is not None
    trace_output = with_traces.trajectories[0]["output"]
    assert trace_output["shell_error_messages"] == ["command exited with status 1"]
    assert set(trace_output) == {
        "deployment_id",
        "query",
        "student_tool_calls",
        "student_tool_errors",
        "entry_id",
        "shell_error_messages",
        "student_eval_run_id",
        "eval_trace_id",
    }


def test_shell_error_analysis_cache_round_trip(tmp_path):
    cache_file = tmp_path / "eval-cache.json"
    analysis = EvalRunShellToolErrorAnalysis(
        eval_ids=("run_cached",),
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
        aggregate=ShellToolErrorMetrics(
            shell_executions=10,
            shell_errors=3,
            shell_error_rate=0.3,
            shell_error_pct=30.0,
            recent_error_examples=(),
        ),
        per_entry={
            "entry-1": ShellToolErrorEntryMetrics(
                entry_id="entry-1",
                shell_executions=2,
                shell_errors=1,
                shell_error_rate=0.5,
                shell_error_pct=50.0,
                recent_error_examples=(),
                trace_ids=("trace-cached",),
            )
        },
        high_signal_entry_ids=("entry-1",),
    )
    evalcli = EvalCliClient(binary="/fake/evalcli")
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=evalcli),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        cache_file=str(cache_file),
    )

    with patch("glean_gepa.objectives.shell.fetch_eval_run_shell_tool_error_analysis", return_value=analysis) as fetch:
        # Only a trace-detail fetch is cached; that is what full trace evals request.
        assert adapter._get_or_fetch_analysis("run_cached", detail="traces") is analysis
        fetch.assert_called_once()

    adapter._save_cache()
    assert "eval_cache" not in json.loads(cache_file.read_text())

    reloaded = SingleModelAdapter(
        runner=ALRunner(evalcli=evalcli),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        cache_file=str(cache_file),
    )
    with patch("glean_gepa.objectives.shell.fetch_eval_run_shell_tool_error_analysis") as fetch:
        cached = reloaded._get_or_fetch_analysis("run_cached")

    fetch.assert_not_called()
    assert cached.aggregate.shell_error_rate == 0.3
    assert cached.high_signal_entry_ids == ("entry-1",)
    assert cached.per_entry["entry-1"].trace_ids == ("trace-cached",)


def test_provisional_zero_shell_analysis_is_refetched_instead_of_cached(tmp_path):
    cache_file = tmp_path / "eval-cache.json"
    provisional = EvalRunShellToolErrorAnalysis(
        eval_ids=("run_pending_telemetry",),
        start_date=date(2026, 8, 31),
        end_date=date(2026, 9, 1),
        aggregate=ShellToolErrorMetrics(
            shell_executions=0,
            shell_errors=0,
            shell_error_rate=0.0,
            shell_error_pct=0.0,
            recent_error_examples=(),
        ),
        per_entry={},
        high_signal_entry_ids=(),
    )
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli")),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        cache_file=str(cache_file),
    )

    with patch(
        "glean_gepa.objectives.shell.fetch_eval_run_shell_tool_error_analysis", return_value=provisional
    ) as fetch:
        assert adapter._get_or_fetch_analysis("run_pending_telemetry") is provisional
        assert adapter._get_or_fetch_analysis("run_pending_telemetry") is provisional

    assert fetch.call_count == 2
    assert adapter._eval_analysis_cache == {}


def test_evaluate_refuses_to_score_provisional_zero_shell_analysis():
    provisional = EvalRunShellToolErrorAnalysis(
        eval_ids=("run_pending_telemetry",),
        start_date=date(2026, 8, 31),
        end_date=date(2026, 9, 1),
        aggregate=ShellToolErrorMetrics(
            shell_executions=0,
            shell_errors=0,
            shell_error_rate=0.0,
            shell_error_pct=0.0,
            recent_error_examples=(),
        ),
        per_entry={},
        high_signal_entry_ids=(),
    )
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli")),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )
    batch = [
        {
            "eval_set_name": "Glean Chat V2 Medium",
            "eval_set_version": "20260815",
            "deployment_ids": ["scio-prod"],
            "status": "active",
        }
    ]

    with (
        patch.object(adapter, "_get_or_run_student_eval", return_value="run_pending_telemetry"),
        patch.object(adapter, "_get_or_fetch_analysis", return_value=provisional),
        pytest.raises(TelemetryPendingError, match="refusing to score 0/0"),
    ):
        adapter.evaluate(batch, {"WRITING_CODE": "prompt"})


def test_full_validation_skips_per_entry_query_and_evalcli_trace_hydration():
    analysis = EvalRunShellToolErrorAnalysis(
        eval_ids=("gepa_gpt_5a0754e0543e49fc_1788306729",),
        start_date=date(2026, 9, 2),
        end_date=date(2026, 9, 2),
        aggregate=ShellToolErrorMetrics(
            shell_executions=560,
            shell_errors=38,
            shell_error_rate=0.0679,
            shell_error_pct=6.79,
            recent_error_examples=(
                ShellToolErrorExample(
                    eval_id="gepa_gpt_5a0754e0543e49fc_1788306729",
                    started_at="2026-09-02T05:27:00Z",
                    project_id="scio-prod",
                    entry_id="entry-1",
                    run_id="execution-1",
                    trace_id="trace-1",
                    span_id="span-1",
                    span_name="Execute Action: Shell",
                    action_id="Shell",
                    action_status="ERROR",
                    span_status="ERROR",
                    provider_status="failed",
                    output_status_code="1",
                    error_str="command exited with status 1",
                    action_run_id="call-1",
                ),
            ),
        ),
        per_entry={},
        high_signal_entry_ids=(),
    )
    evalcli = EvalCliClient(binary="/fake/evalcli")
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=evalcli),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )

    with (
        patch.object(adapter, "_get_or_run_student_eval", return_value="gepa_gpt_5a0754e0543e49fc_1788306729"),
        patch(
            "glean_gepa.objectives.shell.fetch_eval_run_shell_tool_error_analysis",
            return_value=analysis,
        ) as fetch,
        patch.object(evalcli, "get_analysis_trace") as get_trace,
    ):
        result = adapter.evaluate(
            [
                {
                    "eval_set_name": "Glean Chat V2 Medium",
                    "eval_set_version": "20260815",
                    "deployment_ids": ["scio-prod"],
                    "status": "active",
                    "cached_student_eval_run_id": "gepa_gpt_5a0754e0543e49fc_1788306729",
                }
            ],
            {"WRITING_CODE": "prompt"},
            capture_traces=False,
        )

    fetch.assert_called_once()
    assert fetch.call_args.kwargs["include_error_examples"] is False
    assert fetch.call_args.kwargs["include_per_entry"] is False
    get_trace.assert_not_called()
    assert result.scores == [pytest.approx(1 - 0.0679)]
    assert result.trajectories is None


def test_persisted_zero_shell_analysis_is_refetched(tmp_path):
    cache_file = tmp_path / "eval-cache.json"
    cache_file.write_text(
        json.dumps(
            {
                "eval_analysis_cache": {
                    "run_pending_telemetry": {
                        "schema_version": EVAL_ANALYSIS_CACHE_SCHEMA_VERSION,
                        "eval_id": "run_pending_telemetry",
                        "start_date": "2026-08-31",
                        "end_date": "2026-08-31",
                        "aggregate": {
                            "eval_id": "run_pending_telemetry",
                            "shell_executions": 0,
                            "shell_errors": 0,
                            "shell_error_rate": 0.0,
                            "shell_error_pct": 0.0,
                            "recent_error_examples": [],
                        },
                        "per_entry": {},
                        "high_signal_entry_ids": [],
                    }
                }
            }
        )
    )
    refreshed = EvalRunShellToolErrorAnalysis(
        eval_ids=("run_pending_telemetry",),
        start_date=date(2026, 8, 31),
        end_date=date(2026, 9, 1),
        aggregate=ShellToolErrorMetrics(
            shell_executions=1,
            shell_errors=0,
            shell_error_rate=0.0,
            shell_error_pct=0.0,
            recent_error_examples=(),
        ),
        per_entry={},
        high_signal_entry_ids=(),
    )
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli")),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        cache_file=str(cache_file),
    )

    with patch("glean_gepa.objectives.shell.fetch_eval_run_shell_tool_error_analysis", return_value=refreshed) as fetch:
        assert adapter._get_or_fetch_analysis("run_pending_telemetry") is refreshed

    fetch.assert_called_once()


def test_legacy_shell_error_analysis_cache_is_refetched(tmp_path):
    cache_file = tmp_path / "eval-cache.json"
    cache_file.write_text(
        json.dumps(
            {
                "eval_analysis_cache": {
                    "run_legacy": {
                        "eval_id": "run_legacy",
                        "start_date": "2026-08-08",
                        "end_date": "2026-08-11",
                        "aggregate": {
                            "eval_id": "run_legacy",
                            "shell_executions": 1,
                            "shell_errors": 1,
                            "shell_error_rate": 1.0,
                            "shell_error_pct": 100.0,
                            "recent_error_examples": [],
                        },
                        "per_entry": {},
                        "high_signal_entry_ids": [],
                    }
                }
            }
        )
    )
    refreshed = EvalRunShellToolErrorAnalysis(
        eval_ids=("run_legacy",),
        start_date=date(2026, 8, 8),
        end_date=date(2026, 8, 11),
        aggregate=ShellToolErrorMetrics(
            shell_executions=0,
            shell_errors=0,
            shell_error_rate=0.0,
            shell_error_pct=0.0,
            recent_error_examples=(),
        ),
        per_entry={},
        high_signal_entry_ids=(),
    )
    adapter = SingleModelAdapter(
        runner=ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli")),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        cache_file=str(cache_file),
    )

    with patch("glean_gepa.objectives.shell.fetch_eval_run_shell_tool_error_analysis", return_value=refreshed) as fetch:
        assert adapter._get_or_fetch_analysis("run_legacy") is refreshed

    fetch.assert_called_once()


def test_launched_student_eval_is_resumed_from_in_flight_after_timeout():
    evalcli = MagicMock()
    evalcli.create_eval_run.side_effect = lambda **kwargs: kwargs["eval_run_id"]
    runner = ALRunner(evalcli=evalcli)
    adapter = SingleModelAdapter(
        runner=runner,
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )
    eval_kwargs = {
        "eval_set_name": "set",
        "eval_set_version": "v1",
        "deployment_ids": ["prod"],
        "system_prompt": "prompt",
    }
    with patch.object(runner, "wait", side_effect=TimeoutError("terminal timed out")):
        with pytest.raises(TimeoutError):
            adapter._get_or_run_student_eval(**eval_kwargs)

    launched_id = evalcli.create_eval_run.call_args.kwargs["eval_run_id"]
    assert launched_id in runner._in_flight

    with patch.object(runner, "wait") as wait:
        eval_id = adapter._get_or_run_student_eval(**eval_kwargs)

    evalcli.create_eval_run.assert_called_once()
    wait.assert_called_once_with(launched_id)
    assert eval_id == launched_id
