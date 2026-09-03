from unittest.mock import MagicMock

from glean_gepa.evalcli_client import EvalCliError
from glean_gepa.focused_evalset import (
    build_upload_eval_set_request,
    ensure_focused_eval_set,
    focused_eval_set_version,
)


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


def test_ensure_focused_eval_set_uploads_resolved_trace_identifiers():
    evalcli = MagicMock()
    evalcli.get_eval_set_version.return_value = None
    evalcli.wait_for_eval_set_entries.return_value = [{"id": "new-entry"}]
    source_entries = [
        {
            "id": "keep",
            "deploymentId": "prod",
            "stt": "session-1",
            "runId": "run-1",
            "traceId": "trace-1",
        }
    ]

    focused = ensure_focused_eval_set(
        evalcli,
        base_eval_set_name="Example",
        base_eval_set_version="v1",
        deployment_ids=["prod"],
        entry_ids=["keep"],
        source_entries=source_entries,
    )

    assert focused is not None
    assert evalcli.upload_eval_set.call_args.args[0]["entries"] == [
        {
            "deploymentId": "prod",
            "stt": "session-1",
            "runId": "run-1",
            "traceId": "trace-1",
        }
    ]


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


def test_ensure_focused_eval_set_retries_when_existing_version_has_no_entries():
    evalcli = MagicMock()
    evalcli.get_eval_set_version.return_value = {"name": "existing"}
    evalcli.list_eval_set_entries.return_value = []
    evalcli.wait_for_eval_set_entries.return_value = [{"id": "new-entry"}]
    source_entries = [{"id": "keep", "deploymentId": "prod", "stt": "session-1"}]

    focused = ensure_focused_eval_set(
        evalcli,
        base_eval_set_name="Example",
        base_eval_set_version="v1",
        deployment_ids=["prod"],
        entry_ids=["keep"],
        source_entries=source_entries,
    )

    assert focused is not None
    uploaded_version = evalcli.upload_eval_set.call_args.args[0]["version"]
    assert "_retry_" in uploaded_version
    assert evalcli.wait_for_eval_set_entries.call_args.kwargs["eval_set_version"] == uploaded_version


def test_ensure_focused_eval_set_retries_unreadable_already_exists_version():
    evalcli = MagicMock()
    evalcli.get_eval_set_version.return_value = None
    evalcli.upload_eval_set.side_effect = [
        EvalCliError('Eval set with name "gepa-high-signal-example" and version "v1_hs_abc" already exists'),
        None,
    ]
    evalcli.wait_for_eval_set_entries.return_value = [{"id": "new-entry"}]
    source_entries = [{"id": "keep", "deploymentId": "prod", "stt": "session-1"}]

    focused = ensure_focused_eval_set(
        evalcli,
        base_eval_set_name="Example",
        base_eval_set_version="v1",
        deployment_ids=["prod"],
        entry_ids=["keep"],
        source_entries=source_entries,
    )

    assert focused is not None
    assert evalcli.upload_eval_set.call_count == 2
    original_version = evalcli.upload_eval_set.call_args_list[0].args[0]["version"]
    retry_version = evalcli.upload_eval_set.call_args_list[1].args[0]["version"]
    assert original_version == focused_eval_set_version("v1", ["keep"])
    assert retry_version.startswith(f"{original_version}_retry_")
    assert focused.version == retry_version
    assert evalcli.wait_for_eval_set_entries.call_args.kwargs["eval_set_version"] == retry_version


def test_ensure_focused_eval_set_reuses_readable_concurrent_create():
    evalcli = MagicMock()
    original_version = focused_eval_set_version("v1", ["keep"])
    evalcli.get_eval_set_version.side_effect = [None, {"name": "existing", "version": original_version}]
    evalcli.upload_eval_set.side_effect = EvalCliError(
        f'Eval set with name "gepa-high-signal-example" and version "{original_version}" already exists'
    )
    evalcli.wait_for_eval_set_entries.return_value = [{"id": "new-entry"}]
    source_entries = [{"id": "keep", "deploymentId": "prod", "stt": "session-1"}]

    focused = ensure_focused_eval_set(
        evalcli,
        base_eval_set_name="Example",
        base_eval_set_version="v1",
        deployment_ids=["prod"],
        entry_ids=["keep"],
        source_entries=source_entries,
    )

    assert focused is not None
    assert evalcli.upload_eval_set.call_count == 1
    assert focused.version == original_version
    assert evalcli.wait_for_eval_set_entries.call_args.kwargs["eval_set_version"] == original_version
