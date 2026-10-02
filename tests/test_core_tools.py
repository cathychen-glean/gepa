from __future__ import annotations

from base64 import urlsafe_b64decode
from urllib.parse import unquote_plus

from glean_gepa.al_adapter import ALRunner, Candidate, ModuleSpec, Thresholds, approx_token_len, total_prompt_tokens
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.evolutionary_proposer import (
    modules_after_tool_choice,
    pick_modules_to_edit,
    tool_usage_in_examples,
    tools_with_evidence,
)
from glean_gepa.prompt import (
    compile_encoded_prompt,
    compile_tool_description_overrides,
    is_core_tool_span,
    tool_description_override_key,
)
from glean_gepa.prompt_constants import (
    CORE_TOOL_DESCRIPTIONS,
    CORE_TOOLS,
    EXECUTION_DISCIPLINE_KEY,
    RULES_EXT_KEY,
    TOOL_DESCRIPTION_OVERRIDES_PARAM,
    WRITING_CODE_KEY,
    WRITING_CODE_TOKEN_BUDGET,
)
from glean_gepa.reflection_prompts import parse_chosen_tool_keys
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter


def test_tool_description_override_key_matches_sanitize_identifier_lower():
    assert tool_description_override_key("glean_search") == "glean_search"
    assert tool_description_override_key("Glean Search") == "glean_search"
    assert tool_description_override_key("Glean Document Reader") == "glean_document_reader"
    assert tool_description_override_key("Ask User Questions") == "ask_user_questions"
    assert tool_description_override_key("Shell") == "shell"
    assert tool_description_override_key("Subagent") == "subagent"
    assert tool_description_override_key("Subagent") != "delegate"
    assert is_core_tool_span("Glean Search")
    assert is_core_tool_span("todo_write")
    assert not is_core_tool_span("Write")
    assert not is_core_tool_span("")


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


def test_compile_encoded_prompt_survives_qe_query_unescape():
    # Standard base64 of this payload contains both '+' and '/'; QE QueryUnescape
    # would turn '+' into space and drop the override if we used b64encode.
    payload = "a+b/c?x=1"
    compiled = compile_encoded_prompt({"FULL_PROMPT": payload})
    fragment = compiled.split(",", 1)[0]
    key, b64 = fragment.split("=", 1)
    assert key == "llmo.per_prompt_overrides.coding_agent_loop_system"
    assert "+" not in b64
    assert "/" not in b64
    assert urlsafe_b64decode(unquote_plus(b64).encode("ascii")).decode("utf-8") == payload


def test_parse_chosen_tool_keys_keeps_reflector_order():
    eligible = ["glean_search", "discover", "todo_write"]
    assert parse_chosen_tool_keys("NONE", eligible) == []
    assert parse_chosen_tool_keys("", eligible) == []
    assert parse_chosen_tool_keys("todo_write\ndiscover\nnot_a_tool\ntodo_write", eligible) == [
        "todo_write",
        "discover",
    ]


def test_pick_modules_to_edit_offers_every_listed_core_tool():
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
    prompt_only = TeacherStudentAdapter(**kwargs, editable_modules=[WRITING_CODE_KEY])
    core_tools = TeacherStudentAdapter(**kwargs, editable_modules=list(CORE_TOOLS))
    both = TeacherStudentAdapter(**kwargs, editable_modules=[WRITING_CODE_KEY, *CORE_TOOLS])
    search_only = TeacherStudentAdapter(**kwargs, editable_modules=["glean_search"])

    assert pick_modules_to_edit(prompt_only) == [WRITING_CODE_KEY]
    assert pick_modules_to_edit(prompt_only, eval_batch) == [WRITING_CODE_KEY]
    assert pick_modules_to_edit(core_tools) == list(CORE_TOOLS)
    assert pick_modules_to_edit(core_tools, eval_batch) == list(CORE_TOOLS)
    assert pick_modules_to_edit(both, eval_batch) == [WRITING_CODE_KEY, *CORE_TOOLS]
    assert pick_modules_to_edit(search_only, eval_batch) == ["glean_search"]

    rules_and_core = TeacherStudentAdapter(**kwargs, editable_modules=[*CORE_TOOLS, RULES_EXT_KEY])
    rules_only = TeacherStudentAdapter(**kwargs, editable_modules=[RULES_EXT_KEY])
    assert pick_modules_to_edit(rules_and_core) == [RULES_EXT_KEY, *CORE_TOOLS]
    assert pick_modules_to_edit(rules_and_core, eval_batch) == [RULES_EXT_KEY, *CORE_TOOLS]
    assert pick_modules_to_edit(rules_only) == [RULES_EXT_KEY]
    assert pick_modules_to_edit(rules_only, eval_batch) == [RULES_EXT_KEY]


def test_modules_after_tool_choice_keeps_only_the_named_descriptions():
    parent = Candidate(
        model="gpt",
        prompt_modules={"glean_search": "search", "discover": "discover", RULES_EXT_KEY: ""},
        module_specs={},
        global_token_cap=4096,
        baseline_prompt_hash="h",
    )
    examples = {
        "glean_search": [
            {
                "Inputs": {"query": "q"},
                "Generated Outputs": {"teacher_tools": ["Glean Search"], "student_tools": ["Discover"]},
                "Action Inputs": ['teacher Glean Search: {"query": "pto"}'],
                "Feedback": "mismatch",
            }
        ],
        "discover": [],
        RULES_EXT_KEY: [],
    }

    def choose(prompt: str) -> str:
        assert "glean_search" in prompt
        assert "discover" in prompt
        assert 'ACTION_INPUT: teacher Glean Search: {"query": "pto"}' in prompt
        assert "glean_search: student invoked it in 0 of 1 examples, teacher in 1 of 1" in prompt
        assert "discover: student invoked it in 1 of 1 examples, teacher in 0 of 1" in prompt
        return "discover\nglean_search"

    chosen = modules_after_tool_choice(
        choose,
        parent,
        [RULES_EXT_KEY, "glean_search", "discover"],
        examples,
    )
    # Non-core modules keep the first offspring slots; chosen tools follow in reflector order.
    assert chosen == [RULES_EXT_KEY, "discover", "glean_search"]

    def unused(_prompt: str) -> str:
        raise AssertionError("no examples, so the reflector is not called")

    assert modules_after_tool_choice(unused, parent, ["glean_search", RULES_EXT_KEY], {"glean_search": []}) == [
        RULES_EXT_KEY
    ]


def _paired_example(student_tools: list[str], teacher_tools: list[str]) -> dict:
    return {
        "Inputs": {"query": "q"},
        "Generated Outputs": {"teacher_tools": teacher_tools, "student_tools": student_tools},
        "Action Inputs": [],
        "Feedback": "loss",
    }


def test_tool_usage_counts_examples_per_side_with_trace_event_names():
    examples = [
        _paired_example(["Personal Knowledge Vault Retrieve", "Glean Search", "Shell"], ["Glean Document Reader"]),
        _paired_example(["Ask User Questions"], ["Glean Search", "Glean Search"]),
        _paired_example([], ["Glean Document Reader", "Glean Search"]),
    ]
    usage = tool_usage_in_examples(
        ["glean_search", "glean_document_reader", "ask_user_questions", "todo_write"], examples
    )
    assert usage == {
        "glean_search": (1, 2),
        "glean_document_reader": (0, 2),
        "ask_user_questions": (1, 0),
        "todo_write": (0, 0),
    }
    # Either side's invocations count; the threshold never exceeds the example count.
    assert tools_with_evidence(usage, example_count=3) == []
    assert tools_with_evidence(usage, example_count=2) == ["glean_search", "glean_document_reader"]
    assert tools_with_evidence({"discover": (1, 0)}, example_count=1) == ["discover"]


def test_modules_after_tool_choice_drops_tools_nobody_invoked():
    """A description cannot steer a decision the student never reaches.

    In the agentic run the reflector spent three of five offspring rewriting
    ask_user_questions while the student invoked it on 4 of 189 entries and asked
    in prose instead; the val score did not move. Such tools are filtered before
    the reflector picks, and the reflector is shown the counts for the rest.
    """
    parent = Candidate(
        model="gpt",
        prompt_modules={"glean_search": "s", "glean_document_reader": "r", "ask_user_questions": "a"},
        module_specs={},
        global_token_cap=4096,
        baseline_prompt_hash="h",
    )
    examples = [
        _paired_example(["Glean Search"], ["Glean Search", "Glean Document Reader"]),
        _paired_example(["Glean Search"], ["Glean Document Reader"]),
        _paired_example([], ["Glean Search", "Glean Document Reader"]),
        _paired_example(["Ask User Questions"], ["Glean Search"]),
    ]
    high_signal = dict.fromkeys(("glean_search", "glean_document_reader", "ask_user_questions"), examples)

    def choose(prompt: str) -> str:
        assert "### ask_user_questions" not in prompt
        assert "ask_user_questions" not in prompt.split("CURRENT DESCRIPTIONS")[0]
        assert "glean_document_reader: student invoked it in 0 of 4 examples, teacher in 3 of 4" in prompt
        return "ask_user_questions\nglean_document_reader\nglean_search"

    chosen = modules_after_tool_choice(
        choose,
        parent,
        [EXECUTION_DISCIPLINE_KEY, "glean_search", "glean_document_reader", "ask_user_questions"],
        high_signal,
    )
    assert chosen == [EXECUTION_DISCIPLINE_KEY, "glean_document_reader", "glean_search"]


def test_total_prompt_tokens_excludes_core_tool_descriptions():
    candidate = Candidate(
        model="gpt",
        prompt_modules={WRITING_CODE_KEY: "abcd" * 10, "glean_search": "x" * 400},
        module_specs={WRITING_CODE_KEY: ModuleSpec(WRITING_CODE_KEY, "free_text", WRITING_CODE_TOKEN_BUDGET)},
        global_token_cap=4096,
        baseline_prompt_hash="h",
    )
    assert total_prompt_tokens(candidate) == approx_token_len("abcd" * 10)
