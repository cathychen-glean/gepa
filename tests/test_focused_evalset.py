from unittest.mock import MagicMock, patch

import pytest

from glean_gepa.focused_evalset import (
    DEFAULT_RUN_LABEL,
    HIGH_SIGNAL_RUN_LABEL,
    EvalRunTarget,
    FocusedEvalSet,
    build_upload_eval_set_request,
    ensure_focused_eval_set,
    focused_eval_set_version,
    prepare_high_signal_eval_batch,
    resolve_eval_run_target,
)

SOURCE = {
    "eval_set_name": "Example",
    "eval_set_version": "v1",
    "deployment_ids": ["prod"],
    "status": "active",
}
FOCUSED = FocusedEvalSet("gepa-high-signal-example", "v1_hs_abc", 1)


def test_focused_eval_set_version_is_stable_for_the_same_entries():
    assert focused_eval_set_version("v1", ["b", "a"]) == focused_eval_set_version("v1", ["a", "b"])


def test_build_upload_request_keeps_source_metadata():
    request = build_upload_eval_set_request(
        name="gepa-high-signal-example",
        version="v1_hs_123",
        entries=[{"deploymentId": "prod", "stt": "session"}],
        base_eval_set_name="Example",
        base_eval_set_version="v1",
    )

    assert request["useUploadJob"] is True
    assert request["metadata"] == {
        "gepaSourceEvalSetName": "Example",
        "gepaSourceEvalSetVersion": "v1",
    }


def test_ensure_focused_eval_set_uploads_only_requested_entries():
    evalcli = MagicMock()
    evalcli.get_eval_set_version.return_value = None
    evalcli.list_eval_set_entries.return_value = [
        {"id": "keep", "deploymentId": "prod", "stt": "session-1"},
        {"id": "drop", "deploymentId": "prod", "stt": "session-2"},
    ]
    evalcli.wait_for_eval_set_entries.return_value = [{"id": "new-entry"}]

    focused = ensure_focused_eval_set(
        evalcli,
        base_eval_set_name="Example",
        base_eval_set_version="v1",
        deployment_ids=["prod"],
        entry_ids=["keep"],
    )

    assert focused is not None
    request = evalcli.upload_eval_set.call_args.args[0]
    assert request["entries"] == [{"deploymentId": "prod", "stt": "session-1"}]
    assert focused.entry_count == 1


def test_ensure_focused_eval_set_reuses_an_existing_version():
    evalcli = MagicMock()
    evalcli.get_eval_set_version.return_value = {"name": "existing"}
    evalcli.list_eval_set_entries.return_value = [{"id": "existing-entry"}]

    focused = ensure_focused_eval_set(
        evalcli,
        base_eval_set_name="Example",
        base_eval_set_version="v1",
        deployment_ids=["prod"],
        entry_ids=["keep"],
    )

    assert focused is not None
    evalcli.upload_eval_set.assert_not_called()


def test_prepare_high_signal_eval_batch_attaches_focused_set_or_fails():
    focused = FOCUSED
    batch = [{**SOURCE, "eval_entry_ids": ["keep"]}]
    with patch("glean_gepa.focused_evalset.ensure_focused_eval_set", return_value=focused) as ensure:
        prepared = prepare_high_signal_eval_batch(MagicMock(), batch)

    ensure.assert_called_once()
    assert prepared is not None
    assert prepared[0]["eval_set_name"] == focused.name
    assert prepared[0]["eval_set_version"] == focused.version
    assert prepared[0]["focused_eval_set_name"] == focused.name
    assert prepared[0]["focused_eval_set_version"] == focused.version
    assert prepared[0]["eval_entry_ids"] == ["keep"]

    with patch("glean_gepa.focused_evalset.ensure_focused_eval_set", return_value=None):
        assert prepare_high_signal_eval_batch(MagicMock(), batch) is None


@pytest.mark.parametrize(
    "data, ensure_return, expected, ensure_called",
    [
        (SOURCE, None, EvalRunTarget("Example", "v1", DEFAULT_RUN_LABEL, is_focused=False), False),
        (
            {**SOURCE, "eval_entry_ids": ["keep"], "focused_eval_set_name": "focused", "focused_eval_set_version": "v1_hs"},
            None,
            EvalRunTarget("focused", "v1_hs", HIGH_SIGNAL_RUN_LABEL, is_focused=True),
            False,
        ),
        (
            {**SOURCE, "eval_entry_ids": ["keep"]},
            FOCUSED,
            EvalRunTarget(FOCUSED.name, FOCUSED.version, HIGH_SIGNAL_RUN_LABEL, is_focused=True),
            True,
        ),
        ({**SOURCE, "eval_entry_ids": ["keep"]}, None, None, True),
    ],
)
def test_resolve_eval_run_target(data, ensure_return, expected, ensure_called):
    with patch("glean_gepa.focused_evalset.ensure_focused_eval_set", return_value=ensure_return) as ensure:
        assert resolve_eval_run_target(MagicMock(), data) == expected
    assert ensure.called is ensure_called
