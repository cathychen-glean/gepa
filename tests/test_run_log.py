from __future__ import annotations

from glean_gepa.run_log import (
    capture_run_log,
    format_child_proposal_report,
    format_eval_entry_report,
    format_high_signal_selection_report,
    format_screening_report,
    selected_entry_ids_from_examples,
)
from glean_gepa.runner import RUN_LOG_FILENAME, _parse_args, _resolve_log_file


def test_format_run_log_reports():
    trajectories = [
        {
            "data": {"eval_set_name": "Chat", "eval_set_version": "v1"},
            "output": {
                "entry_id": "e1",
                "query": "Find the Q3 plan",
                "teacher_tool_events": ["Shell", "Glean Search", "Glean Document Reader"],
                "student_tool_events": ["Discover"],
                "teacher_eval_run_id": "t1",
                "student_eval_run_id": "s1",
            },
            "score": 0.0,
            "objective_scores": {"tool_alignment": 0.0, "completeness": 0.8},
        }
    ]
    report = format_eval_entry_report(trajectories)
    assert "entry e1" in report
    assert "teacher tools: Glean Search > Glean Document Reader" in report
    assert "student tools: Discover" in report
    assert "first-tool: mismatch (teacher=Glean Search, student=Discover)" in report
    assert "tool_alignment=0.00" in report
    assert "completeness=0.80" in report

    high_signal = format_high_signal_selection_report(
        selected_groups=[(0, "Glean Search", "Discover", 12), (0, "Glean Document Reader", "todo_write", 8)],
        selected_entry_ids=["e1", "e2"],
        selected_count=20,
        total_mismatch_count=31,
        module_entry_ids={"glean_search": ["e1"], "FULL_PROMPT": ["e1", "e2"]},
    )
    assert "largest first-disagreement clusters" in high_signal
    assert "teacher=Glean Search  student=Discover  n=12" in high_signal
    assert "Selected entry_ids: e1, e2" in high_signal
    assert "glean_search: e1" in high_signal

    child = format_child_proposal_report(
        parent_id="parent",
        child_id="child",
        module="glean_search",
        delta="- old\n+ new\n",
        justification="WHY: student used Discover first.",
    )
    assert "module edited: glean_search" in child
    assert "WHY: student used Discover first." in child
    assert "+ new" in child
    screening = format_screening_report(
        mode="fix-rate",
        entry_ids=["e1", "e2"],
        rows=[("child", 0.4, True, "fix_rate=0.400")],
    )
    assert "PASS" in screening
    assert "e1, e2" in screening

    examples = [
        {"Inputs": {"entry_id": "e1"}},
        {"Inputs": {"entry_id": "e1"}},
        {"Inputs": {"entry_id": "e2"}},
    ]
    assert selected_entry_ids_from_examples(examples) == ["e1", "e2"]


def test_capture_run_log_and_default_path(tmp_path, capsys):
    log_path = tmp_path / "gepa_run.log"
    with capture_run_log(log_path):
        print("hello-run-log")
    captured = capsys.readouterr()
    assert "hello-run-log" in captured.out
    assert "hello-run-log" in log_path.read_text()

    args = _parse_args(["--seed_candidate", "seed.json", "--run_dir", "run_ts8"])
    assert args.log_file is None
    assert _resolve_log_file(args) == args.run_dir / RUN_LOG_FILENAME
    explicit = _parse_args(["--seed_candidate", "seed.json", "--log_file", "/tmp/custom.log"])
    assert _resolve_log_file(explicit).as_posix() == "/tmp/custom.log"
