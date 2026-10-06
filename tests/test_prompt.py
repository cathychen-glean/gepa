"""``prompt.py``: core-tool description overrides and their encoding into scParams."""

from __future__ import annotations
from base64 import urlsafe_b64decode
from urllib.parse import unquote_plus
from glean_gepa.al_adapter import ALRunner, Candidate, ModuleSpec, approx_token_len, total_prompt_tokens
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
    CORE_TOOLS,
    EXECUTION_DISCIPLINE_KEY,
    RULES_EXT_KEY,
    TOOL_DESCRIPTION_OVERRIDES_PARAM,
    WRITING_CODE_KEY,
)
from glean_gepa.prompt_targets import module_token_budget, stock_text
from glean_gepa.reflection_prompts import parse_chosen_tool_keys
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter
from glean_gepa.objectives.registry import BUILTIN_OBJECTIVES
from glean_gepa.prompt_constants import FULL_PROMPT_KEY
from glean_gepa.reflection_prompts import (
    MODULE_TEXT_BEGIN,
    MODULE_TEXT_END,
    consolidate_prompt,
    diagnosis_prompt,
    drops_render_slot,
    length_rule_for,
    markup_rule_for,
    module_char_budget,
    new_module_length_rule,
    parse_diagnosis_reply,
    patch_after_snippets,
    sanitize_proposed_module,
)
from base64 import urlsafe_b64decode
from urllib.parse import unquote_plus
from glean_gepa.al_adapter import ALRunner, Candidate, ModuleSpec, approx_token_len, total_prompt_tokens
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
    CORE_TOOLS,
    EXECUTION_DISCIPLINE_KEY,
    RULES_EXT_KEY,
    TOOL_DESCRIPTION_OVERRIDES_PARAM,
    WRITING_CODE_KEY,
)
from glean_gepa.reflection_prompts import parse_chosen_tool_keys
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter

CORE_TOOL_DESCRIPTIONS = {key: stock_text(key) or "" for key in CORE_TOOLS}
WRITING_CODE_TOKEN_BUDGET = module_token_budget(WRITING_CODE_KEY) or 0



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


# Inline fixtures: a plain module and one carrying renderer markup (conditional + slot).





PLAIN = (
    "Use the search tool when the user asks where something is. Prefer one call over several. "
    "Do not ask the user to clarify when the request is answerable; deliver and state the assumption. "
    "Never ask about audience, tone, depth or format when a reasonable default exists. "
    "When a result is truncated, read the saved file rather than re-running the same call. "
    "Cite every source you rely on and keep the citation identifiers you were given."
)
WITH_MARKUP = "intro\n**Rules:**\n- stock rule\n{RULES_EXT}\n<<<[[hitl_approval_instructions]]>>>\n### Sandbox\n"


def _diagnosis(current: str) -> str:
    return diagnosis_prompt(
        module_name="m", responsibility="r", current=current, failure_label="F", example_blocks="e", length_rule="l"
    )


def _consolidate(current: str, **kw) -> str:
    return consolidate_prompt(
        module_name="m",
        max_variants=2,
        consolidate_length="l",
        current=current,
        example_blocks="e",
        suggestions="ZZ_SUGGESTION_BODY",
        **kw,
    )


def test_no_objective_claims_the_render_template_as_a_module():
    for spec in BUILTIN_OBJECTIVES:
        assert FULL_PROMPT_KEY not in spec.load().module_responsibilities


# --- Markup safety: the renderer treats <<<[[name]] ... >>> as a live conditional and {SLOT}
# as a splice point, so prompts for plain modules must not introduce either and the reflector's
# output must not keep or drop them by accident.


def test_prompts_for_a_plain_module_carry_no_renderer_markup():
    for prompt in (_diagnosis(PLAIN), _consolidate(PLAIN)):
        assert MODULE_TEXT_BEGIN in prompt and MODULE_TEXT_END in prompt
        assert "<<<" not in prompt and "[[" not in prompt and "{" not in prompt
    assert "<<<" not in MODULE_TEXT_BEGIN + MODULE_TEXT_END


def test_markup_rule_tracks_the_slots_and_conditionals_in_the_module():
    assert "{RULES_EXT}" in markup_rule_for(WITH_MARKUP)
    assert "hitl_approval_instructions" in markup_rule_for(WITH_MARKUP)
    assert markup_rule_for(PLAIN) == markup_rule_for("other plain text")
    assert drops_render_slot("- rule\n### Sandbox", current=WITH_MARKUP)
    assert not drops_render_slot("- rule\n{RULES_EXT}\n### Sandbox", current=WITH_MARKUP)
    assert not drops_render_slot("- rule", current=PLAIN)


def test_sanitize_strips_reflector_artifacts_but_keeps_real_markup():
    invented = "<<<\nSearch company docs. <<<[[hitl_approval_instructions]]>>> Prefer this first. >>>"
    cleaned = sanitize_proposed_module(invented, current=PLAIN)
    assert "<<<" not in cleaned and "[[" not in cleaned and "Prefer this first." in cleaned

    labeled = "Variant 1 (351 chars)\nRULES_EXT\n- Use Edit on the named file."
    assert sanitize_proposed_module(labeled, current="", module_name="RULES_EXT") == "- Use Edit on the named file."

    wrapped = f"<<<\n{WITH_MARKUP}\n>>>"
    assert sanitize_proposed_module(wrapped, current=WITH_MARKUP).count("<<<") == WITH_MARKUP.count("<<<")


# --- Diagnosis/consolidate plumbing.


def test_parse_diagnosis_reply_splits_the_tally_from_the_patches():
    reply = parse_diagnosis_reply(
        "DIAGNOSIS:\n- Asks in prose: 5 of 8 LOSS examples.\nPATCHES:\nBEFORE: a\nAFTER: b\nWHY: c", current=PLAIN
    )
    assert not reply.is_module_rewrite
    assert reply.diagnosis == "- Asks in prose: 5 of 8 LOSS examples."
    assert reply.patches == "BEFORE: a\nAFTER: b\nWHY: c"
    # A near-verbatim module rewrite is not a diagnosis; headerless patches and short prose are.
    assert parse_diagnosis_reply(PLAIN.replace("one call", "a single call"), current=PLAIN).is_module_rewrite
    assert not parse_diagnosis_reply("BEFORE: x\nAFTER: y\nWHY: z", current=PLAIN).is_module_rewrite
    assert not parse_diagnosis_reply("diagnosis", current=PLAIN).is_module_rewrite


def test_consolidate_threads_the_diagnosis_before_the_suggestions():
    with_diag = _consolidate(PLAIN, diagnosis="- hedging: 4 of 10")
    assert "- hedging: 4 of 10" in with_diag
    assert with_diag.index("- hedging: 4 of 10") < with_diag.index("ZZ_SUGGESTION_BODY")
    assert "- hedging: 4 of 10" not in _consolidate(PLAIN)


_TWO_PATCHES = "BEFORE:\n[empty]\nAFTER:\n- first rule.\nWHY: a\n\nBEFORE:\n[empty]\nAFTER:\n- second rule.\nWHY: b"


def test_patch_after_snippets_and_new_module_budget():
    afters = patch_after_snippets(_TWO_PATCHES)
    assert afters == ["- first rule.", "- second rule."]
    assert patch_after_snippets("no patches here") == []

    rule, budget = new_module_length_rule(_TWO_PATCHES, 2048)
    assert budget == max(200, int(1.5 * sum(map(len, afters)))) and str(budget) in rule
    assert new_module_length_rule(_TWO_PATCHES, 100)[1] == 100  # token cap wins
    assert new_module_length_rule("just prose", 2048)[1] == 2048  # no patches: full budget


def test_module_char_budget_and_length_rule():
    assert module_char_budget("x" * 8000) == 10400  # MODULE_GROWTH
    assert module_char_budget("x" * 8000, token_budget=2000) == 8000  # token cap wins
    assert module_char_budget("short") == 200  # floor
    assert module_char_budget("", 128) == 512  # empty module: token cap only
    rule = length_rule_for(PLAIN)
    assert str(len(PLAIN)) in rule and str(module_char_budget(PLAIN)) in rule


def test_parse_chosen_tool_keys_keeps_reflector_order():
    eligible = ["glean_search", "discover", "todo_write"]
    assert parse_chosen_tool_keys("NONE", eligible) == []
    assert parse_chosen_tool_keys("", eligible) == []
    assert parse_chosen_tool_keys("todo_write\ndiscover\nnot_a_tool\ntodo_write", eligible) == [
        "todo_write",
        "discover",
    ]
