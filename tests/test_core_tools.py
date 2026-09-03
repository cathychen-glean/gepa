from __future__ import annotations

from base64 import urlsafe_b64decode

from glean_gepa.al_adapter import ALRunner, Candidate, ModuleSpec, Thresholds, approx_token_len, total_prompt_tokens
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.core_tools import (
    CORE_TOOL_DESCRIPTIONS,
    CORE_TOOLS,
    TOOL_DESCRIPTION_OVERRIDES_PARAM,
    compile_tool_description_overrides,
    high_signal_core_tool_keys,
    tool_description_override_key,
)
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.evolutionary_proposer import pick_modules_to_edit
from glean_gepa.prompt import FULL_PROMPT_KEY, compile_encoded_prompt
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter


def test_tool_description_override_key_matches_sanitize_identifier_lower():
    assert tool_description_override_key("glean_search") == "glean_search"
    assert tool_description_override_key("Glean Search") == "glean_search"
    assert tool_description_override_key("Glean Document Reader") == "glean_document_reader"
    assert tool_description_override_key("Ask User Questions") == "ask_user_questions"
    assert tool_description_override_key("Shell") == "shell"
    assert tool_description_override_key("Subagent") == "subagent"
    assert tool_description_override_key("Subagent") != "delegate"


def test_compile_tool_description_overrides():
    assert compile_tool_description_overrides({}) == ""
    candidate = {"glean_search": CORE_TOOL_DESCRIPTIONS["glean_search"], "FULL_PROMPT": "unused"}
    encoded = compile_tool_description_overrides(candidate)
    assert encoded.startswith(TOOL_DESCRIPTION_OVERRIDES_PARAM + "=")
    payload = encoded.split("=", 1)[1]
    assert ";" not in payload
    key, b64 = payload.split(":", 1)
    assert key == "glean_search"
    assert urlsafe_b64decode(b64.encode("ascii")).decode("utf-8") == CORE_TOOL_DESCRIPTIONS["glean_search"]

    encoded_all = compile_tool_description_overrides(CORE_TOOL_DESCRIPTIONS)
    payload_all = encoded_all.split("=", 1)[1]
    keys = [segment.split(":", 1)[0] for segment in payload_all.split(";")]
    assert keys == list(CORE_TOOLS)
    for segment in payload_all.split(";"):
        key, b64 = segment.split(":", 1)
        assert urlsafe_b64decode(b64.encode("ascii")).decode("utf-8") == CORE_TOOL_DESCRIPTIONS[key]

    compiled = compile_encoded_prompt({"FULL_PROMPT": "hello", "glean_search": "Search less."})
    assert compiled.startswith("llmo.per_prompt_overrides.coding_agent_loop_system=")
    _system_fragment, tool_fragment = compiled.split(",", 1)
    assert tool_fragment == compile_tool_description_overrides({"FULL_PROMPT": "hello", "glean_search": "Search less."})


def test_high_signal_core_tool_keys():
    trajectories = [
        {
            "output": {
                "teacher_tool_events": ["Glean Search"],
                "student_tool_events": ["Discover"],
            }
        },
        {
            "output": {
                "teacher_tool_events": ["Jira"],
                "student_tool_events": ["Glean Document Reader"],
            }
        },
        {
            "output": {
                "teacher_tool_events": ["Glean Search"],
                "student_tool_events": ["Discover"],
            }
        },
    ]
    assert high_signal_core_tool_keys(trajectories) == ["glean_search", "discover", "glean_document_reader"]

    shell_only = [
        {
            "output": {
                "teacher_tool_events": ["Shell", "Glean Search"],
                "student_tool_events": ["Subagent"],
            }
        }
    ]
    assert high_signal_core_tool_keys(shell_only) == ["glean_search"]
    assert high_signal_core_tool_keys([object()]) == []


def test_pick_modules_to_edit_rewrites_only_eligible_high_signal_core_tools():
    runner = ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli"))
    kwargs = {
        "runner": runner,
        "teacher_model": "gpt",
        "student_model": "fast",
        "thresholds": Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    }
    eval_batch = GleanEvaluationBatch(
        outputs=[],
        scores=[],
        trajectories=[
            {
                "output": {
                    "teacher_tool_events": ["Glean Search"],
                    "student_tool_events": ["Discover"],
                },
                "score": 0.0,
            }
        ],
    )
    prompt_only = TeacherStudentAdapter(**kwargs, editable_modules=[FULL_PROMPT_KEY])
    core_tools = TeacherStudentAdapter(**kwargs, editable_modules=list(CORE_TOOLS))
    both = TeacherStudentAdapter(**kwargs, editable_modules=[FULL_PROMPT_KEY, *CORE_TOOLS])
    search_only = TeacherStudentAdapter(**kwargs, editable_modules=["glean_search"])

    assert pick_modules_to_edit(prompt_only) == [FULL_PROMPT_KEY]
    assert pick_modules_to_edit(prompt_only, eval_batch) == [FULL_PROMPT_KEY]
    assert pick_modules_to_edit(core_tools) == []
    assert pick_modules_to_edit(core_tools, eval_batch) == ["glean_search", "discover"]
    assert pick_modules_to_edit(both, eval_batch) == [FULL_PROMPT_KEY, "glean_search", "discover"]
    assert pick_modules_to_edit(search_only, eval_batch) == ["glean_search"]


def test_total_prompt_tokens_excludes_core_tool_descriptions():
    candidate = Candidate(
        model="gpt",
        prompt_modules={"FULL_PROMPT": "abcd" * 10, "glean_search": "x" * 400},
        module_specs={"FULL_PROMPT": ModuleSpec("FULL_PROMPT", "free_text", 8192)},
        global_token_cap=4096,
        baseline_prompt_hash="h",
    )
    assert total_prompt_tokens(candidate) == approx_token_len("abcd" * 10)
