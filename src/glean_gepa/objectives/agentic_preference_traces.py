"""Paired traces and judge rationales for the agentic-preference objective.

The frame is ``PairedRunAnalysis[None, AgenticPreferenceEntry]``: EvalCLI supplies
the answers, the tool-match query overlays tool sequences, and the judge's
prose verdict is loaded separately for reflection. Nothing here scores an entry.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from glean_gepa.judge_metrics_util import run_entry_id
from glean_gepa.objectives.tool_match import (
    ToolMatchEntryMetrics,
    fetch_eval_run_tool_match_analysis,
)
from glean_gepa.objectives.utils.core import PairedRunAnalysis

_ORIENTATION = re.compile(r"orientation=A=(\w+),B=(\w+)")
_AGENTIC_DIMENSION_PREFIX = "judge_pairwise_agentic_"
_OVERALL_DIMENSION = "multi_dimension_overall"
# ``analyze details`` names the eval runs by their judging role, not by model.
_ROLE_WORDS = {"test": "Student", "base": "Teacher"}
_LABEL_PHRASES = {"win": "student preferred", "lose": "teacher preferred", "tie": "tie"}
# The judge synthesizes in this order: task completion decides, correctness overrides when the
# central claim is wrong, output readiness breaks ties. Render dimensions the same way.
_DIMENSION_ORDER = ("overall", "task_completion", "correctness", "output_readiness")
# Evidence caps keep one entry's verdict readable inside a 40-example reflection prompt.
_MAX_CLAIMS = 6
_MAX_ACTIONS = 5
_CLAIM_TEXT_LIMIT = 220
_EVIDENCE_TEXT_LIMIT = 200
_SUPPORTED_JUDGMENTS = {"supported", "verified", "true", "accurate"}


def _clip(text: Any, limit: int) -> str:
    value = str(text or "").strip().replace("\n", " ")
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _side_word(side: Any, roles: Mapping[str, str]) -> str:
    key = str(side or "").strip().upper()
    if key in ("A", "B"):
        return roles.get(key, key)
    if key == "BOTH":
        return "both"
    return key.lower() or "unknown"


def _verified_claim_lines(claims: Any, roles: Mapping[str, str]) -> list[str]:
    """Render the judge's fact checks, unsupported claims first: they are the correctness gap."""
    if not isinstance(claims, Sequence) or isinstance(claims, str):
        return []
    rendered: list[tuple[int, str]] = []
    for claim in claims:
        if not isinstance(claim, Mapping):
            continue
        judgment = str(claim.get("judgment") or "unverified").strip().lower()
        side = _side_word(claim.get("side"), roles)
        text = _clip(claim.get("claim"), _CLAIM_TEXT_LIMIT)
        if not text:
            continue
        evidence = _clip(claim.get("evidence"), _EVIDENCE_TEXT_LIMIT)
        line = f"    claim [{side}] {judgment}: {text}"
        if evidence:
            line += f" — {evidence}"
        rendered.append((0 if judgment not in _SUPPORTED_JUDGMENTS else 1, line))
    rendered.sort(key=lambda item: item[0])
    return [line for _, line in rendered[:_MAX_CLAIMS]]


def _action_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [_clip(value, _CLAIM_TEXT_LIMIT)] if value.strip() else []
    if isinstance(value, Sequence):
        return [_clip(item, _CLAIM_TEXT_LIMIT) for item in value if str(item or "").strip()][:_MAX_ACTIONS]
    return []


def _task_completion_lines(evidence: Any, roles: Mapping[str, str]) -> list[str]:
    """Render requested actions against what each side actually did, mapped to Student/Teacher."""
    if not isinstance(evidence, Mapping):
        return []
    lines: list[str] = []
    requested = _action_list(evidence.get("actions_user_requested"))
    if requested:
        lines.append(f"    user requested: {' | '.join(requested)}")
    for key, value in evidence.items():
        if key == "actions_user_requested":
            continue
        prefix = key[:1].upper()
        if len(key) > 2 and key[1] == "_" and prefix in roles:
            label = f"{roles[prefix]} {key[2:].replace('_', ' ')}"
        else:
            label = key.replace("_", " ")
        items = _action_list(value)
        if items:
            lines.append(f"    {label}: {' | '.join(items)}")
    return lines


@dataclass(frozen=True)
class AgenticPreferenceEntry:
    """One paired trace. Not scored here: the judge's verdict lives in the eval
    run's metrics, so ``passed`` / ``score`` are placeholders that keep the
    entry on the shared frame without pretending to a per-entry result."""

    entry_id: str
    student_answer: str
    teacher_answer: str
    student_tools: tuple[str, ...] = ()
    teacher_tools: tuple[str, ...] = ()
    student_tool_inputs: tuple[tuple[str, str], ...] = ()
    teacher_tool_inputs: tuple[tuple[str, str], ...] = ()
    student_trace_id: str = ""
    teacher_trace_id: str = ""
    student_deployment_id: str = ""
    teacher_deployment_id: str = ""
    student_min_start_ms: int = 0
    student_max_start_ms: int = 0
    teacher_min_start_ms: int = 0
    teacher_max_start_ms: int = 0

    @property
    def passed(self) -> bool:
        return True

    @property
    def score(self) -> float:
        return 1.0


class AgenticPreferenceAnalysis(PairedRunAnalysis[None, AgenticPreferenceEntry]):
    """Paired traces keyed by entry; EvalCLI-backed, so no shard window or aggregate."""


def empty_agentic_preference_analysis(teacher_eval_id: str, student_eval_id: str) -> AgenticPreferenceAnalysis:
    return AgenticPreferenceAnalysis(eval_ids=(teacher_eval_id, student_eval_id), aggregate=None)


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


def fetch_paired_preference_traces(
    client: Any,
    *,
    teacher_eval_id: str,
    student_eval_id: str,
    lookback_days: int = 1,
    evalcli: Any | None = None,
    include_action_inputs: bool = True,
    **_: Any,
) -> AgenticPreferenceAnalysis:
    """Load student/teacher answers and tool sequences so reflection has both."""
    # Validation batches score the judge aggregate, not per-entry traces.
    if not include_action_inputs:
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
    tools_by_entry: dict[str, ToolMatchEntryMetrics] = {}
    if client is not None:
        try:
            tools_by_entry = fetch_eval_run_tool_match_analysis(
                client,
                teacher_eval_id=teacher_eval_id,
                student_eval_id=student_eval_id,
                lookback_days=lookback_days,
            ).per_entry
        except Exception as exc:
            print(f"[agentic preference] Could not load tool sequences for {student_eval_id}: {exc}")
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
            run_id = run_entry_id(run_entry)
            if run_id == student_eval_id:
                student_answer = _answer_from_run_entry(run_entry)
            elif run_id == teacher_eval_id:
                teacher_answer = _answer_from_run_entry(run_entry)
        tool_metrics = tools_by_entry.get(entry_id)
        per_entry[entry_id] = AgenticPreferenceEntry(
            entry_id=entry_id,
            student_answer=student_answer,
            teacher_answer=teacher_answer,
            student_tools=tool_metrics.student_tools if tool_metrics is not None else (),
            teacher_tools=tool_metrics.teacher_tools if tool_metrics is not None else (),
            student_tool_inputs=tool_metrics.student_tool_inputs if tool_metrics is not None else (),
            teacher_tool_inputs=tool_metrics.teacher_tool_inputs if tool_metrics is not None else (),
            student_trace_id=tool_metrics.student_trace_id if tool_metrics is not None else "",
            teacher_trace_id=tool_metrics.teacher_trace_id if tool_metrics is not None else "",
            student_deployment_id=tool_metrics.student_deployment_id if tool_metrics is not None else "",
            teacher_deployment_id=tool_metrics.teacher_deployment_id if tool_metrics is not None else "",
            student_min_start_ms=tool_metrics.student_min_start_ms if tool_metrics is not None else 0,
            student_max_start_ms=tool_metrics.student_max_start_ms if tool_metrics is not None else 0,
            teacher_min_start_ms=tool_metrics.teacher_min_start_ms if tool_metrics is not None else 0,
            teacher_max_start_ms=tool_metrics.teacher_max_start_ms if tool_metrics is not None else 0,
        )
    if per_entry and tools_by_entry:
        joined = sum(1 for metrics in per_entry.values() if metrics.student_tools or metrics.teacher_tools)
        print(f"[agentic preference] Joined tool sequences onto {joined}/{len(per_entry)} paired entries")
    return AgenticPreferenceAnalysis(
        eval_ids=(teacher_eval_id, student_eval_id),
        aggregate=None,
        per_entry=per_entry,
    )


def _dimension_name(judge_output: Mapping[str, Any]) -> str:
    name = str(judge_output.get("name") or "").removeprefix(_AGENTIC_DIMENSION_PREFIX)
    if name == _OVERALL_DIMENSION or not name:
        return "overall"
    return name


def _rationale_line(judge_output: Mapping[str, Any]) -> str | None:
    """Render one scored dimension of the judge's verdict with the evidence behind it.

    Beyond the prose explanation this keeps the gap magnitude (how decisive the
    dimension was), the judge's fact checks (``verified_claims``: which specific
    claims were unsupported and on which side), and the requested-vs-delivered
    action lists (``task_completion_evidence``). Those are what a rewriter needs
    to see the behavior gap rather than only the judge's conclusion.
    """
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
    roles = {
        side: _ROLE_WORDS.get(role, f"Run {side}") for side, role in zip(("A", "B"), orientation.groups(), strict=True)
    }
    for side, word in roles.items():
        explanation = explanation.replace(f"Run {side}", word)
    name = _dimension_name(judge_output)
    label = str(judge_output.get("label") or "")
    gap = payload.get("gap_score")
    gap_text = f" [gap={gap}]" if isinstance(gap, int | float) and not isinstance(gap, bool) else ""
    lines = [f"{name} ({_LABEL_PHRASES.get(label, label or 'unscored')}){gap_text}: {explanation.strip()}"]
    lines.extend(_verified_claim_lines(payload.get("verified_claims"), roles))
    lines.extend(_task_completion_lines(payload.get("task_completion_evidence"), roles))
    for index, line in enumerate(lines[1:], start=1):
        for side, word in roles.items():
            line = line.replace(f"Run {side}", word)
        lines[index] = line
    return "\n".join(lines)


def _dimension_rank(rendered: str) -> int:
    head = rendered.split(" ", 1)[0]
    return _DIMENSION_ORDER.index(head) if head in _DIMENSION_ORDER else len(_DIMENSION_ORDER)


def rationale_from_judge_entries(judge_run_entries: Any, *, judge_run_id: str | None = None) -> str:
    """Join every scored dimension of one entry's agentic verdict, in the judge's synthesis order."""
    lines: list[str] = []
    for entry in judge_run_entries or []:
        if not isinstance(entry, Mapping):
            continue
        if judge_run_id and str(entry.get("judgeRunId") or "") != judge_run_id:
            continue
        for judge_output in entry.get("outputs") or []:
            if isinstance(judge_output, Mapping) and (line := _rationale_line(judge_output)):
                lines.append(line)
    lines.sort(key=_dimension_rank)
    return "\n".join(lines)


def decisive_dimensions(rationale: str, *, side: str) -> list[tuple[str, float]]:
    """Dimensions ``side`` ("student" or "teacher") won, largest gap first, from a rendered rationale."""
    phrase = _LABEL_PHRASES["win" if side == "student" else "lose"]
    found: list[tuple[str, float]] = []
    for line in rationale.splitlines():
        match = re.match(r"^(\w+) \(([^)]*)\)(?: \[gap=([\d.]+)\])?:", line)
        if not match or match.group(1) == "overall" or match.group(2) != phrase:
            continue
        gap = float(match.group(3)) if match.group(3) else 0.0
        found.append((match.group(1), gap))
    found.sort(key=lambda item: -item[1])
    return found


_PREAMBLE = re.compile(
    r"^(here'?s|here is|great question|good question|based on (the|my) (search|research|results)|i found|"
    r"i('ve| have) (found|looked|searched|reviewed)|sure[,!]|certainly|absolutely|let me|i'll)\b",
    re.IGNORECASE,
)
_CLOSING_OFFER = re.compile(
    r"(would you like me to|want me to|do you want me to|should i |let me know if|happy to (dig|pull|help|"
    r"expand|draft)|i can (also |)(dig|pull|look|expand|draft|check)|could you (clarify|confirm|let me know)|"
    r"which (of these|one) would you|shall i)",
    re.IGNORECASE,
)
_PLACEHOLDER = re.compile(r"\[(your name|name|date|recipient|company|insert[^\]]*|tbd|xx+)\]", re.IGNORECASE)
_MARKUP_RESIDUE = re.compile(r"</?cite>|</?parameter>|</?invoke>|citation_\d+:|\bglean-image:", re.IGNORECASE)
_ASK_TOOL_NAMES = {"ask_user_questions", "ask user questions"}
_RETRIEVAL_TOOL_NAMES = {"glean search", "glean_search", "glean document reader", "glean_document_reader"}
_DOC_READER_NAMES = {"glean document reader", "glean_document_reader"}
_ARTIFACT_TOOL_NAMES = {"write", "edit", "notebookedit", "write_file", "edit_file"}


def _norm(name: Any) -> str:
    return str(name or "").strip().lower()


def student_behavior_flags(
    *,
    student_answer: str,
    teacher_answer: str,
    student_tools: Sequence[str],
    teacher_tools: Sequence[str],
) -> list[str]:
    """Mechanical, prompt-steerable observations about how the student behaved on this entry.

    Computed from the final answer and tool sequence only, so the rewriter can turn each
    flag into a rule the student can follow at runtime.
    """
    flags: list[str] = []
    answer = (student_answer or "").strip()
    other = (teacher_answer or "").strip()
    s_tools = [_norm(name) for name in student_tools if name]
    t_tools = [_norm(name) for name in teacher_tools if name]
    s_retrieval = sum(1 for name in s_tools if name in _RETRIEVAL_TOOL_NAMES)
    t_retrieval = sum(1 for name in t_tools if name in _RETRIEVAL_TOOL_NAMES)
    if s_retrieval == 0 and t_retrieval > 0:
        flags.append(
            "student ran no glean_search/glean_document_reader of its own (answered from preloaded context or "
            f"memory) while the teacher run made {t_retrieval} retrieval call(s)"
        )
    elif t_retrieval >= s_retrieval + 2:
        flags.append(f"teacher run made {t_retrieval} retrieval calls vs student {s_retrieval}")
    if any(name in _DOC_READER_NAMES for name in t_tools) and not any(name in _DOC_READER_NAMES for name in s_tools):
        flags.append("teacher run opened a full document with glean_document_reader; student answered from snippets")
    if any(name in _ASK_TOOL_NAMES for name in s_tools) and not any(name in _ASK_TOOL_NAMES for name in t_tools):
        flags.append("student called ask_user_questions (asked instead of delivering) while the teacher run delivered")
    if any(name in _ASK_TOOL_NAMES for name in t_tools) and not any(name in _ASK_TOOL_NAMES for name in s_tools):
        flags.append("student delivered while the teacher run asked a clarifying question")
    if any(name in _ARTIFACT_TOOL_NAMES for name in t_tools) and not any(
        name in _ARTIFACT_TOOL_NAMES for name in s_tools
    ):
        flags.append("teacher run wrote or edited a file; student produced no file")
    if len(t_tools) >= len(s_tools) + 3:
        flags.append(f"teacher run took {len(t_tools)} tool calls vs student {len(s_tools)} (student stopped earlier)")
    if answer:
        first_line = answer.splitlines()[0][:160]
        if _PREAMBLE.search(first_line):
            flags.append(f"student answer opens with a preamble instead of the answer: {first_line[:80]!r}")
        tail = answer[-400:]
        if _CLOSING_OFFER.search(tail):
            match = _CLOSING_OFFER.search(tail)
            flags.append(
                "student answer closes with an offer/question instead of doing the step: "
                f"{tail[match.start() : match.start() + 90]!r}"
                if match
                else "student answer closes with an offer/question instead of doing the step"
            )
        if answer.rstrip().endswith("?") and not other.rstrip().endswith("?"):
            flags.append("student answer ends with a question to the user; the teacher run ended with a result")
        if _PLACEHOLDER.search(answer):
            flags.append("student answer contains unfilled [placeholder] text")
        if _MARKUP_RESIDUE.search(answer):
            flags.append("student answer leaks raw markup (cite/parameter tags or citation tokens)")
        if other and len(answer) * 2 < len(other) and len(other) > 600:
            flags.append(f"student answer is much shorter ({len(answer)} vs {len(other)} chars): check coverage")
        if other and len(other) * 2 < len(answer) and len(answer) > 600:
            flags.append(f"student answer is much longer ({len(answer)} vs {len(other)} chars): check for extras")
    return flags


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
