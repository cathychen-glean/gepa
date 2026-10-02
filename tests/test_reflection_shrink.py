"""Over-budget reflection variants get one cut-only shrink pass before discard."""

from __future__ import annotations

from unittest.mock import MagicMock

from glean_gepa.al_adapter import ALRunner, Candidate, ModuleSpec, Thresholds
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.reflection_prompts import module_char_budget, shrink_prompt
from glean_gepa.single_model_adapter import SingleModelAdapter

MODULE = "WALDO_TOOL_USAGE"
# 400 chars so the 10% growth budget (440) is the binding limit, not the token cap.
CURRENT = ("- rule one about search arguments.\n" * 11) + "<<<[[has_search_tools]]>>>\nkeep\n<<<[[/has_search_tools]]>>>\n"
BUDGET = module_char_budget(CURRENT, 1024)
assert BUDGET is not None and len(CURRENT) < BUDGET


def _adapter() -> SingleModelAdapter:
    return SingleModelAdapter(
        runner=ALRunner(evalcli=EvalCliClient(binary="/fake/evalcli")),
        bigquery_client=MagicMock(),
        student_model="fast",
        thresholds=Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
    )


def _candidate() -> Candidate:
    return Candidate(
        model="fast",
        prompt_modules={MODULE: CURRENT},
        module_specs={MODULE: ModuleSpec(MODULE, "free_text", 1024)},
        global_token_cap=4096,
        baseline_prompt_hash="seed",
    )


def _oversize() -> str:
    return CURRENT + ("- new rule that is long enough to overshoot the growth budget.\n" * 4)


def _fits() -> str:
    return CURRENT + "- new rule.\n"


def _run(responses: list[str]) -> tuple[list[str], list[str]]:
    prompts: list[str] = []

    def lm(prompt: str) -> str:
        prompts.append(prompt)
        return responses[len(prompts) - 1]

    variants, _, _ = _adapter().propose_new_texts(lm, _candidate(), [MODULE], [])
    return variants, prompts


def test_oversize_variant_is_shrunk_and_kept(capsys):
    oversize, fits = _oversize(), _fits()
    assert len(oversize) > BUDGET >= len(fits)
    variants, prompts = _run(["diagnosis", oversize, fits])
    assert variants == [fits.strip()]
    assert len(prompts) == 3  # diagnosis, consolidate, one shrink
    shrink = prompts[2]
    assert f"over the hard limit of {BUDGET}" in shrink
    assert "Only remove or tighten text" in shrink
    assert oversize.strip() in shrink and CURRENT.strip() in shrink
    out = capsys.readouterr().out
    assert f"Shrinking {MODULE} variant of {len(oversize.strip())} chars to budget {BUDGET}" in out
    assert "after shrink" in out


def test_variant_still_over_after_shrink_is_discarded(capsys):
    oversize = _oversize()
    variants, prompts = _run(["diagnosis", oversize, oversize + "- still too long.\n"])
    assert variants == []
    assert len(prompts) == 3
    assert "after shrink (budget" in capsys.readouterr().out


def test_shrink_that_drops_a_conditional_is_discarded():
    oversize = _oversize()
    broken = _fits().replace("<<<[[has_search_tools]]>>>\nkeep\n<<<[[/has_search_tools]]>>>\n", "keep\n")
    variants, prompts = _run(["diagnosis", oversize, broken])
    assert variants == []
    assert len(prompts) == 3


def test_in_budget_variant_is_not_shrunk():
    fits = _fits()
    variants, prompts = _run(["diagnosis", fits])
    assert variants == [fits.strip()]
    assert len(prompts) == 2


def test_only_oversize_variants_trigger_shrink_calls():
    oversize, fits = _oversize(), _fits()
    fits2 = CURRENT + "- other new rule.\n"
    variants, prompts = _run(["diagnosis", f"{fits}===VARIANT==={oversize}===VARIANT==={fits2}", fits2 + "- x.\n"])
    assert len(prompts) == 3  # one shrink for the one oversize variant
    assert variants == [fits.strip(), (fits2 + "- x.\n").strip(), fits2.strip()]


def test_shrink_prompt_states_sizes_and_forbids_additions():
    text = shrink_prompt(module_name="M", variant="x" * 120, budget=100, current="y" * 90)
    assert "is 120 characters, which is 20 characters over the hard limit of 100" in text
    assert "Shorten it to at most 100 characters" in text
    assert "Do not add any rule" in text
    assert "ORIGINAL module (for reference, 90 characters)" in text
