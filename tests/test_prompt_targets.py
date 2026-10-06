"""``prompt_targets``: marked prompt files, the target registry, compile, and seeds."""

from __future__ import annotations

import re
from base64 import urlsafe_b64decode
from pathlib import Path

import pytest

from glean_gepa import prompt_constants
from glean_gepa.al_adapter import GleanAdapterBase
from glean_gepa.prompt_targets import (
    PROMPTS_DIR,
    TEMPLATE_FILE,
    PromptTargetError,
    build_seed,
    compile_overrides,
    known_module_keys,
    load_prompt_target,
    load_seed_file,
    parse_marked_prompt,
    prompt_targets,
    render_requirements,
)
from glean_gepa.reflection_prompts import markup_rule_for
from glean_gepa.runner import _eval_harness_for

MEMORY = "stripped_engram_memory_instructions"
MEMORY_SEED = Path(__file__).resolve().parents[1] / "data" / "memory_personalization_seed.prompt"
_MARKER_LINE = re.compile(r"^\{[#/]\w+\}\n", re.MULTILINE)


def _unmarked(text: str) -> str:
    """The prompt as scio stores it: marker lines deleted."""
    return _MARKER_LINE.sub("", text)


def test_markers_alone_on_a_line_take_their_line_break():
    marked = parse_marked_prompt("A\n{#X}\nbody\n{/X}\nB\n")
    assert marked.template == "A\n{X}\nB\n"
    assert marked.sections == {"X": "body"}

    inline = parse_marked_prompt("Say {#Y}hello{/Y} twice.")
    assert inline.template == "Say {Y} twice."
    assert inline.sections == {"Y": "hello"}

    empty = parse_marked_prompt("{#Z}\n{/Z}\n")
    assert empty.sections == {"Z": ""}


def test_sections_nest_and_keep_document_order():
    marked = parse_marked_prompt("{#OUTER}\nstart\n{#INNER}\nx\n{/INNER}\nend\n{/OUTER}\n")
    assert list(marked.sections) == ["OUTER", "INNER"]
    assert marked.sections == {"OUTER": "start\n{INNER}\nend", "INNER": "x"}
    assert marked.template == "{OUTER}\n"


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("{#A}\nx\n", "never closed"),
        ("{#A}\nx\n{/B}\n", r"\{/B\} found where \{/A\} was expected"),
        ("{#A}\n{/A}\n{#A}\n{/A}\n", "marked more than once"),
    ],
)
def test_malformed_markers_are_rejected(text, match):
    with pytest.raises(PromptTargetError, match=match):
        parse_marked_prompt(text)


@pytest.mark.parametrize("name", [name for name, t in prompt_targets().items() if t.kind == "template"])
def test_stock_compile_is_the_template_file_without_markers(name):
    """Deleting the marker lines from template.prompt gives the prompt scio would render
    with stock text; an empty droppable slot also takes its line."""
    target = prompt_targets()[name]
    expected = _unmarked((PROMPTS_DIR / name / TEMPLATE_FILE).read_text(encoding="utf-8"))
    for key in target.section_keys:
        module = target.modules[key]
        if not module.stock and module.drop_empty_line:
            expected = expected.replace(module.slot + "\n", "")
    assert target.compile_text({}) == expected


def test_memory_seed_compiles_to_the_scio_pr_prompt():
    seed = load_seed_file(MEMORY_SEED)
    assert set(seed) == {"MEMORY_INSTRUCTIONS", "MEMORY_APPLY", "MEMORY_TRIGGERS", "MEMORY_TOOL_FIX", "PERSONALIZATION"}
    expected = _unmarked(MEMORY_SEED.read_text(encoding="utf-8"))
    assert prompt_targets()[MEMORY].compile_text(seed) == expected

    candidate = build_seed(seed, ["PERSONALIZATION"])
    assert candidate == seed
    fragment = next(p for p in compile_overrides(candidate).split(",") if p.startswith("llmo.per_prompt_overrides.stripped"))
    assert urlsafe_b64decode(fragment.split("=", 1)[1]).decode("utf-8") == expected


def test_stock_memory_prompt_has_no_personalization_slot():
    with pytest.raises(PromptTargetError, match=r"PERSONALIZATION is editable but the seed MEMORY_INSTRUCTIONS has no"):
        build_seed({}, ["PERSONALIZATION"])


def test_prompt_constants_name_registered_modules():
    names = [value for key, value in vars(prompt_constants).items() if key.endswith("_KEY")]
    assert names and set(names) <= known_module_keys()
    assert prompt_constants.CORE_TOOLS[0] == "glean_search"


def test_a_new_prompt_is_a_folder(tmp_path):
    folder = tmp_path / "my_prompt"
    folder.mkdir()
    (folder / "template.prompt").write_text(
        "# Title\n{#INTRO}\nHello [[user_name]].\n{/INTRO}\n\n<<<[[extra]]{EXTRA}>>>\n", encoding="utf-8"
    )
    (folder / "target.yaml").write_text(
        "harnesses: [coding]\n"
        "template: {key: MY_PROMPT}\n"
        "sections:\n"
        "  INTRO: {token_budget: 256, required_placeholders: [user_name], frame: Rewrite the greeting.}\n"
        "  EXTRA: {token_budget: 128}\n",
        encoding="utf-8",
    )
    target = load_prompt_target(folder)
    assert target.override_param == "llmo.per_prompt_overrides.my_prompt"
    assert target.modules["INTRO"].frame == "Rewrite the greeting."
    assert target.compile_text({}) == "# Title\nHello [[user_name]].\n\n<<<[[extra]]>>>\n"
    assert target.compile_text({"EXTRA": " more "}) == "# Title\nHello [[user_name]].\n\n<<<[[extra]]more>>>\n"
    assert target.override({}) == ""

    (folder / "target.yaml").write_text("harnesses: [coding]\ntemplate: {key: MY_PROMPT}\n", encoding="utf-8")
    with pytest.raises(PromptTargetError, match="marks sections missing from target.yaml: INTRO"):
        load_prompt_target(folder)


def test_required_markup_guards_rewrites_and_shows_in_the_reflection_rule():
    seed = load_seed_file(MEMORY_SEED)
    template = seed["MEMORY_INSTRUCTIONS"]
    dropped = template.replace("[[memory_context]]", "")
    assert not GleanAdapterBase._keeps_structure(dropped, current=template, module_name="MEMORY_INSTRUCTIONS")
    assert GleanAdapterBase._keeps_structure(template + "\nMore.", current=template, module_name="MEMORY_INSTRUCTIONS")
    assert "[[memory_dir]], [[memory_context]]" in markup_rule_for(template, "MEMORY_INSTRUCTIONS")

    writing = "plain text\n{RULES_EXT}"
    assert markup_rule_for(writing, "WRITING_CODE") == markup_rule_for(writing)


def test_memory_runs_turn_memory_on_for_every_eval():
    extra, dropped = render_requirements(["PERSONALIZATION"])
    assert "co.lo.mo.slwo.enabled=1" in extra and "co.lo.cao.use_stripped_prompts=1" in extra
    assert dropped == ("co.lo.mo.slwo.disabled=1",)
    assert render_requirements(["WRITING_CODE", "glean_search"]) == ((), ())

    harness = _eval_harness_for(None, ["PERSONALIZATION"], ["gpt6_luna", "gpt6_1_sol_high"])
    assert harness.extra_sc_params == extra and harness.drop_sc_params == dropped


def test_a_prompt_cannot_run_under_a_harness_that_never_renders_it():
    with pytest.raises(SystemExit, match="waldo_system renders under the waldo harness, but model 'gpt6_luna'"):
        _eval_harness_for(None, ["WALDO_ROUTING"], ["gpt6_luna"])
    with pytest.raises(SystemExit, match="coding_agent_loop_system renders under the coding harness"):
        _eval_harness_for(None, ["WRITING_CODE"], ["waldo"])
