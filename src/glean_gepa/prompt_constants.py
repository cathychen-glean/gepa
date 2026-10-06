"""Candidate keys that code refers to by name.

Stock text, budgets, and reflection frames live with each prompt under
``glean_gepa/prompts/``; see :mod:`glean_gepa.prompt_targets`.
"""

from glean_gepa.prompt_targets import prompt_targets

# --- coding_agent_loop_system ---
FULL_PROMPT_KEY = "FULL_PROMPT"
WRITING_CODE_KEY = "WRITING_CODE"
RULES_EXT_KEY = "RULES_EXT"
EXECUTION_DISCIPLINE_KEY = "EXECUTION_DISCIPLINE"

# --- waldo_system ---
WALDO_SYSTEM_KEY = "WALDO_SYSTEM"
WALDO_ROUTING_KEY = "WALDO_ROUTING"
WALDO_TOOL_USAGE_KEY = "WALDO_TOOL_USAGE"

# --- core_tool_descriptions ---
_CORE_TOOL_TARGET = prompt_targets()["core_tool_descriptions"]
CORE_TOOLS = _CORE_TOOL_TARGET.section_keys
CORE_TOOL_KEYS = frozenset(CORE_TOOLS)
CORE_TOOLS_GROUP = _CORE_TOOL_TARGET.group or "CORE_TOOLS"
TOOL_DESCRIPTION_OVERRIDES_PARAM = _CORE_TOOL_TARGET.override_param
