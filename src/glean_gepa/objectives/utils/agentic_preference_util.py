"""Paired teacher/student traces and judge rationales for agentic preference."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from glean_gepa.judge_metrics_util import CUSTOMER_AGENTIC_PREFERENCE_METRIC
from glean_gepa.objectives.utils.tool_match_util import fetch_eval_run_tool_match_analysis

AGENTIC_PREFERENCE_OBJECTIVE = CUSTOMER_AGENTIC_PREFERENCE_METRIC

_ORIENTATION = re.compile(r"orientation=A=(\w+),B=(\w+)")
_AGENTIC_DIMENSION_PREFIX = "judge_pairwise_agentic_"
_OVERALL_DIMENSION = "multi_dimension_overall"
# ``analyze details`` names the eval runs by their judging role, not by model.
_ROLE_WORDS = {"test": "Student", "base": "Teacher"}
_LABEL_PHRASES = {"win": "student preferred", "lose": "teacher preferred", "tie": "tie"}


@dataclass(frozen=True)
class AgenticPreferenceEntry:
    entry_id: str
    student_answer: str
    teacher_answer: str
    student_tools: tuple[str, ...] = ()
    teacher_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgenticPreferenceAnalysis:
    teacher_eval_id: str
    student_eval_id: str
    per_entry: dict[str, AgenticPreferenceEntry]


def empty_agentic_preference_analysis(teacher_eval_id: str, student_eval_id: str) -> AgenticPreferenceAnalysis:
    return AgenticPreferenceAnalysis(
        teacher_eval_id=teacher_eval_id,
        student_eval_id=student_eval_id,
        per_entry={},
    )


def _run_entry_id(run_entry: Mapping[str, Any]) -> str:
    return str(
        run_entry.get("runId")
        or run_entry.get("run_id")
        or run_entry.get("evalRunId")
        or run_entry.get("eval_run_id")
        or ""
    )


def _answer_from_run_entry(run_entry: Mapping[str, Any]) -> str:
    output = run_entry.get("output")
    if isinstance(output, str) and output.strip():
        return output.strip()
    if isinstance(output, Mapping):
        chat = output.get("chatResponseInfo") or output.get("evalChatResponseInfo") or {}
        if isinstance(chat, Mapping):
            for key in ("actResponse", "ActResponse", "response"):
                value = chat.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        for key in ("actResponse", "text", "answer", "response"):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    metadata = run_entry.get("metadata") or {}
    if isinstance(metadata, Mapping):
        for key in ("actResponse", "answer", "output"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    text = run_entry.get("outputText") or run_entry.get("answer")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return ""


def _tool_sequences_by_entry(
    client: Any,
    *,
    teacher_eval_id: str,
    student_eval_id: str,
    lookback_days: int,
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Load each role's scored tool sequence per entry from agentspan."""
    if client is None:
        return {}
    try:
        analysis = fetch_eval_run_tool_match_analysis(
            client,
            teacher_eval_id=teacher_eval_id,
            student_eval_id=student_eval_id,
            lookback_days=lookback_days,
        )
    except Exception as exc:
        print(f"[agentic preference] Could not load tool sequences for {student_eval_id}: {exc}")
        return {}
    return {
        entry_id: (metrics.student_tools, metrics.teacher_tools) for entry_id, metrics in analysis.per_entry.items()
    }


def fetch_paired_preference_traces(
    client: Any,
    *,
    teacher_eval_id: str,
    student_eval_id: str,
    lookback_days: int = 1,
    evalcli: Any | None = None,
    **_: Any,
) -> AgenticPreferenceAnalysis:
    """Load student/teacher answers and tool sequences so reflection has both."""
    if evalcli is None:
        return empty_agentic_preference_analysis(teacher_eval_id, student_eval_id)
    get_view = getattr(evalcli, "get_analysis_view", None)
    if not callable(get_view):
        return empty_agentic_preference_analysis(teacher_eval_id, student_eval_id)
    try:
        view = get_view(student_eval_id, base_eval_id=teacher_eval_id)
    except TypeError:
        view = get_view(student_eval_id, teacher_eval_id)
    except Exception as exc:
        print(f"[agentic preference] Could not load paired traces for {student_eval_id}: {exc}")
        return empty_agentic_preference_analysis(teacher_eval_id, student_eval_id)
    if not isinstance(view, Mapping):
        return empty_agentic_preference_analysis(teacher_eval_id, student_eval_id)
    tools_by_entry = _tool_sequences_by_entry(
        client,
        teacher_eval_id=teacher_eval_id,
        student_eval_id=student_eval_id,
        lookback_days=lookback_days,
    )
    per_entry: dict[str, AgenticPreferenceEntry] = {}
    for entry in view.get("entries") or []:
        if not isinstance(entry, Mapping):
            continue
        entry_id = str(entry.get("entryId") or entry.get("entry_id") or "")
        if not entry_id:
            continue
        student_answer = ""
        teacher_answer = ""
        for run_entry in entry.get("evalRunEntries") or []:
            if not isinstance(run_entry, Mapping):
                continue
            run_id = _run_entry_id(run_entry)
            if run_id == student_eval_id:
                student_answer = _answer_from_run_entry(run_entry)
            elif run_id == teacher_eval_id:
                teacher_answer = _answer_from_run_entry(run_entry)
        student_tools, teacher_tools = tools_by_entry.get(entry_id, ((), ()))
        per_entry[entry_id] = AgenticPreferenceEntry(
            entry_id=entry_id,
            student_answer=student_answer,
            teacher_answer=teacher_answer,
            student_tools=student_tools,
            teacher_tools=teacher_tools,
        )
    if per_entry and tools_by_entry:
        joined = sum(1 for metrics in per_entry.values() if metrics.student_tools or metrics.teacher_tools)
        print(f"[agentic preference] Joined tool sequences onto {joined}/{len(per_entry)} paired entries")
    return AgenticPreferenceAnalysis(
        teacher_eval_id=teacher_eval_id,
        student_eval_id=student_eval_id,
        per_entry=per_entry,
    )


def log_agentic_preference_analysis(analysis: AgenticPreferenceAnalysis) -> None:
    print(
        f"[agentic preference] {analysis.student_eval_id} vs {analysis.teacher_eval_id}: "
        f"{len(analysis.per_entry)} paired traces"
    )


def _rationale_line(judge_output: Mapping[str, Any]) -> str | None:
    """Render one scored dimension of the judge's verdict."""
    reasoning = judge_output.get("reasoning")
    if not isinstance(reasoning, str):
        return None
    orientation = _ORIENTATION.search(reasoning)
    brace = reasoning.find("{")
    if orientation is None or brace == -1:
        return None
    try:
        payload = json.loads(reasoning[brace:])
    except ValueError:
        return None
    explanation = payload.get("explanation") if isinstance(payload, Mapping) else None
    if not isinstance(explanation, str) or not explanation.strip():
        return None
    for side, role in zip(("A", "B"), orientation.groups(), strict=True):
        explanation = explanation.replace(f"Run {side}", _ROLE_WORDS.get(role, f"Run {side}"))
    name = str(judge_output.get("name") or "").removeprefix(_AGENTIC_DIMENSION_PREFIX)
    if name == _OVERALL_DIMENSION or not name:
        name = "overall"
    label = str(judge_output.get("label") or "")
    return f"{name} ({_LABEL_PHRASES.get(label, label or 'unscored')}): {explanation.strip()}"


def rationale_from_judge_entries(judge_run_entries: Any, *, judge_run_id: str | None = None) -> str:
    """Join every scored dimension of one entry's agentic verdict."""
    lines: list[str] = []
    for entry in judge_run_entries or []:
        if not isinstance(entry, Mapping):
            continue
        if judge_run_id and str(entry.get("judgeRunId") or "") != judge_run_id:
            continue
        for judge_output in entry.get("outputs") or []:
            if isinstance(judge_output, Mapping) and (line := _rationale_line(judge_output)):
                lines.append(line)
    return "\n".join(lines)


def fetch_preference_rationales(
    evalcli: Any,
    *,
    entry_ids: Sequence[str],
    student_eval_id: str,
    deployment_id: str,
    judge_run_id: str | None = None,
) -> dict[str, str]:
    """Load the judge's prose verdicts for ``entry_ids`` via ``analyze details``."""
    get_details = getattr(evalcli, "get_analysis_details", None)
    if not entry_ids or not callable(get_details):
        return {}
    try:
        details: Any = get_details(
            entry_ids=list(entry_ids),
            eval_run_ids=[student_eval_id],
            deployment_id=deployment_id,
        )
    except Exception as exc:
        print(f"[agentic preference] Could not load judge rationales for {student_eval_id}: {exc}")
        return {}
    rationales: dict[str, str] = {}
    for item in details or []:
        if not isinstance(item, Mapping):
            continue
        eval_set_entry = item.get("evalSetEntry")
        entry_id = str(eval_set_entry.get("id") or "") if isinstance(eval_set_entry, Mapping) else ""
        if not entry_id:
            continue
        if rationale := rationale_from_judge_entries(item.get("judgeRunEntries"), judge_run_id=judge_run_id):
            rationales[entry_id] = rationale
    return rationales
