import json
from datetime import date

import pytest

from glean_gepa.prompt import (
    FULL_PROMPT_KEY,
    WRITING_CODE_KEY,
    compile_system_prompt,
    default_writing_code,
    materialize_system_prompt,
    prompt_format,
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
    assert default_writing_code not in stock
    assert "{WRITING_CODE}" not in stock
    assert "## Writing Code" in stock


def test_compile_system_prompt_leaves_writing_code_slot_when_key_absent():
    assert compile_system_prompt({}) == prompt_format
    assert compile_system_prompt({FULL_PROMPT_KEY: "PREFIX\n{WRITING_CODE}\nSUFFIX"}) == "PREFIX\n{WRITING_CODE}\nSUFFIX"


def test_materialize_system_prompt_fills_defaults_when_modules_missing():
    prompt = materialize_system_prompt({})

    assert prompt == prompt_format.replace("{WRITING_CODE}", default_writing_code)
    assert "{WRITING_CODE}" not in prompt
    assert "**Rules:**" in prompt


@pytest.mark.parametrize(
    "raw, required_keys",
    [
        ({"WRITING_CODE": "code instructions"}, {WRITING_CODE_KEY}),
        ({"FULL_PROMPT": "full template"}, {FULL_PROMPT_KEY}),
        ({"WRITING_CODE": "patterns", "FULL_PROMPT": "template"}, {FULL_PROMPT_KEY}),
    ],
)
def test_load_seed_candidate_accepts_known_keys(tmp_path, raw, required_keys):
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(raw))

    assert _load_seed_candidate(path, required_keys=required_keys) == raw


@pytest.mark.parametrize(
    "editable, expected",
    [
        ([FULL_PROMPT_KEY], {FULL_PROMPT_KEY: "PREFIX\npatterns\nSUFFIX"}),
        ([WRITING_CODE_KEY], {WRITING_CODE_KEY: "patterns"}),
    ],
)
def test_seed_for_editable_modules(editable, expected):
    assert _seed_for_editable_modules(SEED_BOTH, editable) == expected


def test_parse_editable_modules():
    assert _parse_editable_modules("FULL_PROMPT") == [FULL_PROMPT_KEY]
    assert _parse_editable_modules("WRITING_CODE,FULL_PROMPT") == [WRITING_CODE_KEY, FULL_PROMPT_KEY]
    with pytest.raises(SystemExit, match="unknown editable_modules"):
        _parse_editable_modules("GLOBAL_ROLE")


@pytest.mark.parametrize(
    "raw, required_keys, match",
    [
        ({"GLOBAL_ROLE": "role", "WRITING_CODE": "code instructions"}, {WRITING_CODE_KEY}, "unknown keys"),
        ({"WRITING_CODE": "code instructions"}, {FULL_PROMPT_KEY}, "missing required keys"),
        ({"WRITING_CODE": ["code instructions"]}, {WRITING_CODE_KEY}, "WRITING_CODE must be a string"),
    ],
)
def test_load_seed_candidate_rejects_invalid(tmp_path, raw, required_keys, match):
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(SystemExit, match=match):
        _load_seed_candidate(path, required_keys=required_keys)


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
