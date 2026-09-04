"""Structured GEPA run logging, plus optional tee of all terminal output.

``--log_file`` (default ``<run_dir>/gepa_run.log``) captures every print, then
adds reflection / high-signal / child-proposal reports after each proposer step.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

from glean_gepa.tool_match_util import (
    REFLECTION_HIGH_SIGNAL_ENTRY_LIMIT,
    first_tool_mismatch_pair,
    first_tool_name,
    scored_tool_sequence,
)

SEPARATOR = "=" * 78
QUERY_PREVIEW_CHARS = 160


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(bool(getattr(stream, "isatty", lambda: False)()) for stream in self._streams)

    def fileno(self) -> int:
        return self._streams[0].fileno()


@contextmanager
def capture_run_log(path: Path):
    """Append stdout and stderr to ``path`` while still printing to the terminal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log_file:
        original_out, original_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(original_out, log_file)
        sys.stderr = _Tee(original_err, log_file)
        try:
            yield path
        finally:
            sys.stdout = original_out
            sys.stderr = original_err


def log_section(title: str, body: str = "") -> None:
    """Print a titled report block (captured by the run log tee)."""
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    if body:
        print(body.rstrip())
    print()


def format_eval_entry_report(trajectories: Sequence[Mapping[str, Any]] | None) -> str:
    """Teacher vs student tool sequences and metrics for every eval entry."""
    if not trajectories:
        return "(no trajectories)"
    blocks: list[str] = []
    for trajectory in trajectories:
        output = trajectory.get("output") or {}
        metrics = trajectory.get("objective_scores") or {}
        teacher_tools = output.get("teacher_tool_events")
        student_tools = output.get("student_tool_events")
        pair = first_tool_mismatch_pair(teacher_tools, student_tools)
        if pair is None:
            first_line = f"first-tool: match ({first_tool_name(teacher_tools) or '(none)'})"
        else:
            first_line = f"first-tool: mismatch (teacher={pair[0] or '(none)'}, student={pair[1] or '(none)'})"
        query = " ".join(str(output.get("query", "")).split())
        if len(query) > QUERY_PREVIEW_CHARS:
            query = query[: QUERY_PREVIEW_CHARS - 3] + "..."
        score = trajectory.get("score")
        tool_alignment = metrics.get("tool_alignment", score)
        metric_parts = [f"score={_fmt_metric(score)}"]
        if tool_alignment is not None:
            metric_parts.append(f"tool_alignment={_fmt_metric(tool_alignment)}")
        for key in ("completeness", "grounding"):
            value = metrics.get(key)
            if value is not None:
                metric_parts.append(f"{key}={_fmt_metric(value)}")
        teacher_seq = " > ".join(scored_tool_sequence(teacher_tools)) or "(none)"
        student_seq = " > ".join(scored_tool_sequence(student_tools)) or "(none)"
        blocks.append(
            "\n".join(
                [
                    f"entry {output.get('entry_id', '?')}",
                    f"  eval_set: {trajectory.get('data', {}).get('eval_set_name', '?')}:"
                    f"{trajectory.get('data', {}).get('eval_set_version', '?')}",
                    f"  query: {query}",
                    f"  metrics: {', '.join(metric_parts)}",
                    f"  {first_line}",
                    f"  teacher tools: {teacher_seq}",
                    f"  student tools: {student_seq}",
                    f"  teacher_eval={output.get('teacher_eval_run_id', '-')}  "
                    f"student_eval={output.get('student_eval_run_id', '-')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def format_high_signal_selection_report(
    *,
    selected_groups: Sequence[tuple[str, str, int]],
    selected_entry_ids: Sequence[str],
    selected_count: int,
    total_mismatch_count: int,
    module_entry_ids: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Explain which first-tool mismatch groups made the reflection set."""
    lines = [
        "Justification: most-frequent first-tool mismatch groups.",
        f"The most frequent group is always included in full, even if it exceeds "
        f"{REFLECTION_HIGH_SIGNAL_ENTRY_LIMIT} entries. Later whole groups are added only while they "
        f"still fit in that cap. Shell / Personal Knowledge Vault Retrieve are stripped before the "
        f"first-tool comparison.",
        f"Selected {selected_count} of {total_mismatch_count} first-tool mismatch entries "
        f"(cap {REFLECTION_HIGH_SIGNAL_ENTRY_LIMIT}).",
        "",
        "Groups (most frequent first):",
    ]
    if not selected_groups:
        lines.append("  (none — every compared entry already matched on first tool)")
    for index, (teacher_tool, student_tool, count) in enumerate(selected_groups, start=1):
        lines.append(f"  {index}. teacher={teacher_tool or '(none)'}  student={student_tool or '(none)'}  n={count}")
    lines.append("")
    lines.append("Selected entry_ids: " + (", ".join(selected_entry_ids) if selected_entry_ids else "(none)"))
    if module_entry_ids:
        lines.append("")
        lines.append("Per-module reflection examples (core-tool modules keep only mismatches involving that tool):")
        for module, entry_ids in module_entry_ids.items():
            lines.append(f"  {module}: {', '.join(entry_ids) if entry_ids else '(none)'}")
    return "\n".join(lines)


def format_child_proposal_report(
    *,
    parent_id: str,
    child_id: str,
    module: str,
    delta: str,
    justification: str,
) -> str:
    """Describe one proposed child: module, diff, and reflection diagnosis."""
    why = justification.strip() or "(no diagnosis returned by the reflection model)"
    return "\n".join(
        [
            f"parent={parent_id}  child={child_id}",
            f"module edited: {module}",
            "",
            "Justification (reflection diagnosis / WHY):",
            why,
            "",
            "Changes:",
            delta.rstrip() or "(no prompt changes)",
        ]
    )


def format_screening_report(
    *,
    mode: str,
    entry_ids: Sequence[str],
    rows: Sequence[tuple[str, float, bool, str]],
) -> str:
    """Child screening outcomes on the parent's high-signal failures."""
    lines = [
        f"Screen mode: {mode}",
        f"High-signal screen entries: {', '.join(entry_ids) if entry_ids else '(none)'}",
        "",
        "Children:",
    ]
    if not rows:
        lines.append("  (none)")
    for child_id, score, passed, detail in rows:
        verdict = "PASS" if passed else "FAIL"
        lines.append(f"  {child_id}  {verdict}  score={score:.4f}  {detail}")
    return "\n".join(lines)


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def selected_entry_ids_from_examples(examples: Iterable[Mapping[str, Any]]) -> list[str]:
    """Stable entry_id list from reflective examples, de-duplicated."""
    ordered: list[str] = []
    seen: set[str] = set()
    for example in examples:
        entry_id = str((example.get("Inputs") or {}).get("entry_id") or "")
        if entry_id and entry_id not in seen:
            seen.add(entry_id)
            ordered.append(entry_id)
    return ordered
