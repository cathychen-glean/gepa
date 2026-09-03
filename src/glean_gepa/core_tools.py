"""Core-tool description modules for GEPA and ``co.pyagents_tool_description_overrides``.

Override keys match ``sanitize_identifier(name.lower())``: snake_case names such as
``glean_search``, not display names such as ``Glean Search``. An override replaces
only ``schema.description`` (for ``glean_document_reader``, the whole prompt-visible
text including the raw-bytes suffix). Shell is not in this set.
"""

from __future__ import annotations

import keyword
import re
from base64 import urlsafe_b64encode
from collections.abc import Mapping, Sequence
from typing import Any

from glean_gepa.tool_match_util import first_tool_mismatch_pair, select_first_tool_mismatch_groups

CORE_TOOLS = (
    "glean_search",
    "glean_document_reader",
    "glean_container_lister",
    "tool_search",
    "todo_write",
    "delegate",
    "discover",
    "ask_user_questions",
)
CORE_TOOL_KEYS = frozenset(CORE_TOOLS)
CORE_TOOLS_GROUP = "CORE_TOOLS"
CORE_TOOL_TOKEN_BUDGET = 2048
TOOL_DESCRIPTION_OVERRIDES_PARAM = "co.pyagents_tool_description_overrides"

_VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")

CORE_TOOL_DESCRIPTIONS = {
    "glean_search": (
        "Search company knowledge across documents, messages, and other indexed enterprise data. "
        "Use this for internal project context, document lookup, acronym resolution, and company-specific facts. "
        "Keep queries short and targeted; do not batch synonyms, boolean logic, or overlapping query variations. "
        '`query` searches content, filters constrain metadata — use filters only if needed and use `query="*"` '
        "when the intent is purely metadata. Glean search is connected to all datasources."
    ),
    "glean_document_reader": (
        "Retrieve full content for one or more tagged URLs or uploaded file URLs, including images."
        "Always use glean_document_reader to read tagged or uploaded documents and/or images."
        "Use one call for multiple URLs; do not create separate calls per URL. "
        "Use page_selections only when snippets show <page N> or <slide N> markers. "
        "Set should_fetch_raw_bytes=true when: ALWAYS for tabular/spreadsheet files "
        "(.csv, .tsv, .xls, .xlsx, Google Sheets, or any spreadsheet URL) — snippets drop rows/columns/cells "
        "so raw is required even for summarization. For other structured files "
        "(.ppt, .pptx, Google Docs, SharePoint/OneDrive .docx, slides/presentations): set true when editing, "
        "filling in, referencing specific sections/rows/columns/slides, reviewing or addressing comments, "
        "or extracting precise structured information; leave false for pure summarization, factual Q&A, "
        "drafting messages, or general information-seeking. For GitHub: set true only when reading PR/issue "
        "comments or review threads — leave false for code, diffs, or file contents. When uncertain about a "
        "structured document, prefer true. The flag applies to all URLs in the call; put URLs needing different "
        "handling in a separate call."
    ),
    "glean_container_lister": (
        "List immediate children (title, url metadata only) of Spaces Folder, Project, or Space URLs "
        "(`/knowledge/collections/`, `/library/projects/`, `/library/folders/`, `/chat/spaces/`). "
        "Re-list nested `/library/projects/` URLs before reading inside.\n"
        "- Use this to enumerate items in containers; it does NOT return document content and does NOT recurse.\n"
        "- Results include normalized child metadata (e.g., title, url, mimeType, owner, update_time).\n"
        "- Use Glean Document Reader to get the full content of documents returned by this tool."
    ),
    "tool_search": "Discover available integrations and tools (Jira, Slack, Salesforce, etc.).",
    "todo_write": (
        'Update the task progress checklist shown to the user. Pass plan={"todos": '
        '[{"content": "task description", "status": "pending|in_progress|completed"}, ...]}. '
        "Use for 3+ step tasks. Never use a step just to update the list."
    ),
    "delegate": (
        "Spawn a subagent with isolated task. task: detailed instructions for the subagent, including "
        "relevant context and file paths to pass along.The subagent shares the filesystem — save large "
        "data to files and pass paths."
    ),
    "discover": (
        'Searches available skills, tools, and actions to discover ones relevant for a task. Pass query="..." '
        "as one concise, atomic search request. For distinct tasks, issue separate calls instead of combining them. "
        "Results are not exhaustive; only the top few most relevant matches are returned."
    ),
    "ask_user_questions": (
        "Use ask_user_questions for any sort of artifact generation or task-execution requests when the user "
        "has not specified any important shaping choice that is not inferable from any context (e.g. drafting emails, "
        "updating docs, creating presentations, changing code, booking meetings, sending messages). Ask when a user "
        "choice would meaningfully affect the output, such as who it is for, what outcome to optimize for, how "
        "detailed or polished it should be, or which discovered target to use. Use available context first to make "
        "the options concrete; if search finds plausible candidates, ask the user to choose among them instead of "
        "asking an open-ended question. For artifact creation or long-running tasks, invest in clarity up front: "
        "resolve every output-shaping unknown before you start so the first result lands as close to final as "
        "possible — a quick question now saves the user a round of rework later. Call await ask_user_questions("
        '[{"question": "...", "options": [{"label": "..."}, ...], "multiSelect": False}, ...]). Ask 1-3 questions '
        "with 2-3 concrete options each. Do not restate the request or re-ask questions the user already answered "
        "or skipped; approval-required tools (request_*) produce their own confirmation card. Ask everything needed "
        "in one shot, then continue after the user answers. The loop pauses after calling it; user answers arrive "
        "on the next turn as a chat message."
    ),
}


def tool_description_override_key(span_name: str) -> str:
    """Map an Execute Action span name to a ``pyagents_tool_description_overrides`` key."""
    value = span_name.lower()
    if _VALID_IDENTIFIER.fullmatch(value) and not keyword.iskeyword(value):
        return value
    sanitized = _NON_ALNUM.sub("_", value).strip("_")
    if not sanitized:
        return value
    if sanitized[0].isdigit():
        sanitized = "_" + sanitized
    if keyword.iskeyword(sanitized):
        sanitized = sanitized + "_"
    return sanitized


def with_core_tool_defaults(prompt_modules: Mapping[str, str]) -> dict[str, str]:
    """Copy ``prompt_modules`` and fill any missing core-tool descriptions from stock text."""
    merged = dict(prompt_modules)
    for key, text in CORE_TOOL_DESCRIPTIONS.items():
        merged.setdefault(key, text)
    return merged


def high_signal_core_tool_keys(trajectories: Sequence[Any] | None) -> list[str]:
    """Core-tool keys that appear in the reflection high-signal first-tool mismatch groups."""
    mismatch_keys: list[tuple[str, str] | None] = []
    for trajectory in trajectories or []:
        output = trajectory.get("output") if isinstance(trajectory, Mapping) else None
        if not isinstance(output, Mapping):
            mismatch_keys.append(None)
            continue
        mismatch_keys.append(
            first_tool_mismatch_pair(output.get("teacher_tool_events"), output.get("student_tool_events"))
        )
    _indices, groups = select_first_tool_mismatch_groups(mismatch_keys)
    found: list[str] = []
    for teacher_tool, student_tool, _count in groups:
        for name in (teacher_tool, student_tool):
            key = tool_description_override_key(name)
            if key in CORE_TOOL_KEYS and key not in found:
                found.append(key)
    return found


def compile_tool_description_overrides(candidate: Mapping[str, str]) -> str:
    """Encode core-tool modules as ``co.pyagents_tool_description_overrides=key:b64;...``.

    Empty string means evals keep stock descriptions for every tool. A subset is
    allowed; omitted keys keep stock text.
    """
    segments: list[str] = []
    for key in CORE_TOOLS:
        text = candidate.get(key, "")
        if not text:
            continue
        encoded = urlsafe_b64encode(text.encode("utf-8")).decode("ascii")
        segments.append(f"{key}:{encoded}")
    if not segments:
        return ""
    return TOOL_DESCRIPTION_OVERRIDES_PARAM + "=" + ";".join(segments)


def core_tool_reflection_prompt(module_name: str) -> str:
    """Reflection instructions for editing one core-tool ``schema.description``."""
    return (
        f"You are editing only the prompt-visible schema.description for the core tool `{module_name}`. "
        "The override replaces description text only — not the tool signature, parameters, or Returns. "
        "Rewrite the description so the student uses this tool as the first action when the teacher does, "
        "and does not use it first when the teacher chooses a different tool. Keep the text operational and concise."
    )
