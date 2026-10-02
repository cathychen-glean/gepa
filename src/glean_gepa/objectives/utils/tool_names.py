"""Tool-name helpers shared by the tool-match objective, ``prompt``, and ``run_log``.

Pure functions on tool-name sequences. No SQL, no frame, no objective. They live
here rather than in ``objectives/tool_match.py`` so ``prompt.py`` can import
them without importing the objective (which imports ``prompt``).
"""

from __future__ import annotations

from collections.abc import Sequence

# Tools that never count toward first-tool or sequence matching.
SKIPPED_TOOL_NAMES = frozenset({"Personal Knowledge Vault Retrieve", "Shell", "Shell Tool", "find_skills_assistant"})


def scored_tool_sequence(tools: Sequence[str] | None, skip_tools: frozenset[str] | None = None) -> tuple[str, ...]:
    """Return tool names used for sequence matching, dropping Shell and other skipped tools."""
    skipped = SKIPPED_TOOL_NAMES if skip_tools is None else skip_tools
    return tuple(str(name) for name in (tools or []) if name and str(name) not in skipped)


def first_tool_name(tools: Sequence[str] | None, skip_tools: frozenset[str] | None = None) -> str:
    """Return the first scored tool name, or an empty string when none remain."""
    scored = scored_tool_sequence(tools, skip_tools=skip_tools)
    return scored[0] if scored else ""


def first_tool_mismatch_pair(
    teacher_tools: Sequence[str] | None,
    student_tools: Sequence[str] | None,
    skip_tools: frozenset[str] | None = None,
) -> tuple[str, str] | None:
    """Return ``(teacher_first, student_first)`` when they differ, else ``None``."""
    teacher = first_tool_name(teacher_tools, skip_tools=skip_tools)
    student = first_tool_name(student_tools, skip_tools=skip_tools)
    if teacher == student:
        return None
    return (teacher, student)
