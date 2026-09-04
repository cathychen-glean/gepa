import json
from datetime import date
from pathlib import Path

import pytest

from glean_gepa.prompt import compile_system_prompt, materialize_system_prompt, with_core_tool_defaults
from glean_gepa.prompt_constants import (
    CORE_TOOLS,
    CORE_TOOLS_GROUP,
    DEFAULT_FULL_PROMPT,
    DEFAULT_WRITING_CODE,
    FULL_PROMPT_KEY,
    PROMPT_MODULE_DEFAULTS,
    RULES_EXT_KEY,
    WRITING_CODE_KEY,
)
from glean_gepa.runner import (
    ADAPTER_CACHE_FILENAME,
    CACHE_DIRECTORY_NAME,
    _default_cache_file,
    _load_seed_candidate,
    _parse_args,
    _parse_editable_modules,
    _seed_for_editable_modules,
    _select_recent_train_and_val_versions,
)

SEED_BOTH = {"WRITING_CODE": "patterns", "FULL_PROMPT": "PREFIX\n{WRITING_CODE}\nSUFFIX"}


def test_compile_and_materialize_splice_writing_code_when_present():
    assert compile_system_prompt(SEED_BOTH) == "PREFIX\npatterns\nSUFFIX"
    assert materialize_system_prompt(SEED_BOTH) == "PREFIX\npatterns\nSUFFIX"

    stock = compile_system_prompt({WRITING_CODE_KEY: "CUSTOM_PATTERNS"})
    assert "CUSTOM_PATTERNS" in stock
    assert DEFAULT_WRITING_CODE not in stock
    assert "{WRITING_CODE}" not in stock
    assert "## Writing Code" in stock


def test_compile_system_prompt_leaves_writing_code_slot_when_key_absent():
    assert compile_system_prompt({}) == DEFAULT_FULL_PROMPT
    assert (
        compile_system_prompt({FULL_PROMPT_KEY: "PREFIX\n{WRITING_CODE}\nSUFFIX"}) == "PREFIX\n{WRITING_CODE}\nSUFFIX"
    )


def test_materialize_system_prompt_fills_defaults_when_modules_missing():
    prompt = materialize_system_prompt({})

    assert prompt == DEFAULT_FULL_PROMPT.replace("{WRITING_CODE}", DEFAULT_WRITING_CODE)
    assert "{WRITING_CODE}" not in prompt
    assert "{RULES_EXT}" in prompt
    assert "**Rules:**" in prompt


def test_compile_system_prompt_splices_rules_ext_after_rules():
    writing = "intro\n**Rules:**\n- stock rule\n{RULES_EXT}\n### Sandbox\n"
    compiled = compile_system_prompt(
        {
            WRITING_CODE_KEY: writing,
            FULL_PROMPT_KEY: "PREFIX\n{WRITING_CODE}\nSUFFIX",
            RULES_EXT_KEY: "- Prefer Write after retrieving sources.\n- Do not skip Write when the teacher writes.",
        }
    )
    assert "{RULES_EXT}" not in compiled
    assert compiled == (
        "PREFIX\nintro\n**Rules:**\n- stock rule\n"
        "- Prefer Write after retrieving sources.\n- Do not skip Write when the teacher writes."
        "\n### Sandbox\n\nSUFFIX"
    )
    empty = compile_system_prompt(
        {WRITING_CODE_KEY: writing, FULL_PROMPT_KEY: "PREFIX\n{WRITING_CODE}\nSUFFIX", RULES_EXT_KEY: ""}
    )
    assert empty == "PREFIX\nintro\n**Rules:**\n- stock rule\n\n### Sandbox\n\nSUFFIX"


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"WRITING_CODE": "code instructions"},
        {"FULL_PROMPT": "full template"},
        {"WRITING_CODE": "patterns", "FULL_PROMPT": "template"},
        {"glean_search": "Search less."},
        {RULES_EXT_KEY: "- Prefer Write after retrieving sources."},
    ],
)
def test_load_seed_candidate_accepts_known_keys(tmp_path, raw):
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(raw))

    assert _load_seed_candidate(path) == raw


def test_seed_for_editable_modules():
    assert _seed_for_editable_modules(SEED_BOTH, [FULL_PROMPT_KEY]) == with_core_tool_defaults(
        {FULL_PROMPT_KEY: "PREFIX\npatterns\nSUFFIX"}
    )
    assert _seed_for_editable_modules(SEED_BOTH, [WRITING_CODE_KEY]) == with_core_tool_defaults(
        {WRITING_CODE_KEY: "patterns"}
    )

    raw = {**SEED_BOTH, "glean_search": "Search less."}
    seed = _seed_for_editable_modules(raw, [FULL_PROMPT_KEY])
    assert seed["glean_search"] == "Search less."
    assert seed["discover"] == with_core_tool_defaults({})["discover"]

    frozen = _seed_for_editable_modules(SEED_BOTH, [])
    assert frozen[FULL_PROMPT_KEY] == "PREFIX\npatterns\nSUFFIX"
    assert WRITING_CODE_KEY not in frozen
    assert frozen["glean_search"] == with_core_tool_defaults({})["glean_search"]

    rules_seed = _seed_for_editable_modules({**SEED_BOTH, RULES_EXT_KEY: ""}, [RULES_EXT_KEY])
    assert rules_seed[RULES_EXT_KEY] == ""
    assert rules_seed[FULL_PROMPT_KEY] == "PREFIX\npatterns\nSUFFIX"

    defaulted = _seed_for_editable_modules({}, [WRITING_CODE_KEY, RULES_EXT_KEY])
    assert defaulted[WRITING_CODE_KEY] == DEFAULT_WRITING_CODE
    assert defaulted[RULES_EXT_KEY] == ""
    assert FULL_PROMPT_KEY not in defaulted

    overridden = _seed_for_editable_modules({RULES_EXT_KEY: "- Prefer Write."}, [RULES_EXT_KEY])
    assert overridden[RULES_EXT_KEY] == "- Prefer Write."
    assert overridden[FULL_PROMPT_KEY] == materialize_system_prompt({})


def test_parse_editable_modules():
    assert _parse_editable_modules("FULL_PROMPT") == [FULL_PROMPT_KEY]
    assert _parse_editable_modules("WRITING_CODE,FULL_PROMPT") == [WRITING_CODE_KEY, FULL_PROMPT_KEY]
    assert _parse_editable_modules("glean_search") == ["glean_search"]
    assert _parse_editable_modules(RULES_EXT_KEY) == [RULES_EXT_KEY]
    assert _parse_editable_modules(f"{CORE_TOOLS_GROUP},{RULES_EXT_KEY}") == [*CORE_TOOLS, RULES_EXT_KEY]
    assert _parse_editable_modules(CORE_TOOLS_GROUP) == list(CORE_TOOLS)
    assert _parse_editable_modules(f"FULL_PROMPT,{CORE_TOOLS_GROUP}") == [FULL_PROMPT_KEY, *CORE_TOOLS]
    assert _parse_editable_modules(f"{CORE_TOOLS_GROUP},glean_search") == list(CORE_TOOLS)
    with pytest.raises(SystemExit, match="unknown editable_modules"):
        _parse_editable_modules("GLOBAL_ROLE")


@pytest.mark.parametrize(
    "raw, match",
    [
        ({"GLOBAL_ROLE": "role", "WRITING_CODE": "code instructions"}, "unknown keys"),
        ({"WRITING_CODE": ["code instructions"]}, "WRITING_CODE must be a string"),
        (["WRITING_CODE"], "seed_candidate must be a JSON object"),
    ],
)
def test_load_seed_candidate_rejects_invalid(tmp_path, raw, match):
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(SystemExit, match=match):
        _load_seed_candidate(path)


def test_committed_seed_candidate_is_overrides_only():
    path = Path(__file__).resolve().parents[1] / "data" / "seed_candidate.json"
    raw = _load_seed_candidate(path)

    assert WRITING_CODE_KEY not in raw
    assert FULL_PROMPT_KEY not in raw
    seed = _seed_for_editable_modules(raw, [WRITING_CODE_KEY, RULES_EXT_KEY])
    assert seed[WRITING_CODE_KEY] == PROMPT_MODULE_DEFAULTS[WRITING_CODE_KEY]
    assert seed[RULES_EXT_KEY] == PROMPT_MODULE_DEFAULTS[RULES_EXT_KEY]


def test_parse_args_defaults_editable_modules_to_writing_code():
    args = _parse_args(["--seed_candidate", "seed.json"])

    assert args.editable_modules == WRITING_CODE_KEY


def test_parse_args_accepts_all_reflection_samples_and_hamming_k():
    args = _parse_args(
        [
            "--seed_candidate",
            "seed.json",
            "--reflection_samples",
            "all",
            "--reflection_hamming_distance_k",
            "10",
        ]
    )

    assert args.reflection_samples is None
    assert args.reflection_hamming_distance_k == 10


def test_default_cache_files_live_in_run_cache_directory(tmp_path):
    assert _default_cache_file(tmp_path, ADAPTER_CACHE_FILENAME) == (
        tmp_path / CACHE_DIRECTORY_NAME / ADAPTER_CACHE_FILENAME
    )


def test_default_cache_file_moves_legacy_root_cache_on_resume(tmp_path):
    legacy = tmp_path / ADAPTER_CACHE_FILENAME
    legacy.write_text('{"cached": true}')

    cache_file = _default_cache_file(tmp_path, ADAPTER_CACHE_FILENAME)

    assert cache_file == tmp_path / CACHE_DIRECTORY_NAME / ADAPTER_CACHE_FILENAME
    assert cache_file.read_text() == '{"cached": true}'
    assert not legacy.exists()


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_parse_args_rejects_invalid_reflection_sample_count(value):
    with pytest.raises(SystemExit):
        _parse_args(["--seed_candidate", "seed.json", "--reflection_samples", value])


def test_recent_versions_are_split_into_incremental_train_and_held_out_val():
    train_versions, val_versions = _select_recent_train_and_val_versions(
        [{"version": "20260813"}, {"version": "20260820"}, {"version": "20260827"}],
        today=date(2026, 8, 27),
        lookback_days=14,
        valset_size=2,
    )

    assert train_versions == ["20260813"]
    assert val_versions == ["20260820", "20260827"]


def test_recent_versions_fall_back_to_one_val_version_when_only_two_are_available():
    train_versions, val_versions = _select_recent_train_and_val_versions(
        [{"version": "20260820"}, {"version": "20260827"}],
        today=date(2026, 8, 27),
        lookback_days=14,
        valset_size=2,
    )

    assert train_versions == ["20260820"]
    assert val_versions == ["20260827"]
