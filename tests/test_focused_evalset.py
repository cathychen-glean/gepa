from unittest.mock import MagicMock, patch

import pytest

from glean_gepa.evalcli_client import EvalCliError
from glean_gepa.focused_evalset import (
    DEFAULT_RUN_LABEL,
    HIGH_SIGNAL_RUN_LABEL,
    QUERY_CANONICAL_BUCKET_TYPE,
    SESSION_BUCKET_TYPE,
    EvalRunTarget,
    FocusedEvalSet,
    build_upload_entry,
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


def test_focused_eval_set_version_changes_with_bucket_type():
    ids = ["a", "b"]
    session = focused_eval_set_version("v1", ids, bucket_type=SESSION_BUCKET_TYPE)
    query = focused_eval_set_version("v1", ids, bucket_type=QUERY_CANONICAL_BUCKET_TYPE)
    assert session != query


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
        {"id": "keep", "deploymentId": "prod", "user": "spark", "input": {"query": "hello"}, "stt": "session-1"},
        {"id": "drop", "deploymentId": "prod", "user": "spark", "input": {"query": "other"}, "stt": "session-2"},
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
    assert request["bucketType"] == QUERY_CANONICAL_BUCKET_TYPE
    assert request["entries"] == [{"deploymentId": "prod", "user": "spark", "query": "hello"}]
    assert focused.entry_count == 1


def test_build_upload_entry_rejects_session_rows_without_stt():
    assert (
        build_upload_entry(
            {
                "id": "keep",
                "deploymentId": "prod",
                "input": {"query": "hello"},
                "sourceTrackingInfo": {"sessionTrackingToken": None, "traceId": None},
            },
            bucket_type=SESSION_BUCKET_TYPE,
        )
        is None
    )


def test_build_upload_entry_session_keeps_only_stt_identity():
    entry = build_upload_entry(
        {
            "deploymentId": "prod",
            "user": "spark",
            "stt": "session-1",
            "traceId": "eval-trace",
            "runId": "orig-run",
            "qtt": "qtt-1",
            "input": {"query": "hello"},
        },
        bucket_type=SESSION_BUCKET_TYPE,
    )

    assert entry == {
        "deploymentId": "prod",
        "user": "spark",
        "stt": "session-1",
        "query": "hello",
    }


def test_build_upload_entry_query_canonical_is_fresh_query():
    entry = build_upload_entry(
        {
            "deploymentId": "prod",
            "user": "spark",
            "stt": "session-1",
            "traceId": "eval-trace",
            "runId": "orig-run",
            "qtt": "qtt-1",
            "input": {"query": "hello"},
        }
    )

    assert entry == {"deploymentId": "prod", "user": "spark", "query": "hello"}


def test_build_upload_entry_query_canonical_rejects_rows_without_query():
    assert build_upload_entry({"deploymentId": "prod", "stt": "session-1"}) is None


def test_ensure_focused_eval_set_does_not_upload_session_rows_without_stt():
    evalcli = MagicMock()
    evalcli.get_eval_set_version.return_value = None
    evalcli.list_eval_set_entries.return_value = [
        {"id": "keep", "deploymentId": "prod", "input": {"query": "hello"}},
    ]

    focused = ensure_focused_eval_set(
        evalcli,
        base_eval_set_name="Example",
        base_eval_set_version="v1",
        deployment_ids=["prod"],
        entry_ids=["keep"],
        bucket_type=SESSION_BUCKET_TYPE,
    )

    assert focused is None
    evalcli.upload_eval_set.assert_not_called()


def test_ensure_focused_eval_set_does_not_resolve_stt_for_query_canonical():
    evalcli = MagicMock()
    evalcli.get_eval_set_version.return_value = None
    evalcli.list_eval_set_entries.return_value = [
        {
            "id": "keep",
            "deploymentId": "prod",
            "user": "spark",
            "input": {"query": "hello"},
            "sourceTrackingInfo": {
                "traceId": None,
                "sessionTrackingToken": None,
                "queryTrackingToken": None,
                "runId": None,
            },
        },
    ]
    evalcli.wait_for_eval_set_entries.return_value = [{"id": "new-entry"}]

    with patch("glean_gepa.focused_evalset.fetch_evalset_entry_tracking") as fetch:
        focused = ensure_focused_eval_set(
            evalcli,
            base_eval_set_name="Example",
            base_eval_set_version="v1",
            deployment_ids=["prod"],
            entry_ids=["keep"],
            bigquery_client=MagicMock(),
        )

    fetch.assert_not_called()
    request = evalcli.upload_eval_set.call_args.args[0]
    assert request["bucketType"] == QUERY_CANONICAL_BUCKET_TYPE
    assert request["entries"] == [
        {
            "deploymentId": "prod",
            "user": "spark",
            "query": "hello",
        }
    ]
    assert focused is not None
    assert focused.entry_count == 1


def test_ensure_focused_eval_set_fills_stt_from_bigquery_for_session_bucket():
    evalcli = MagicMock()
    evalcli.get_eval_set_version.return_value = None
    evalcli.list_eval_set_entries.return_value = [
        {
            "id": "keep",
            "deploymentId": "prod",
            "user": "spark",
            "input": {"query": "hello"},
            "sourceTrackingInfo": {
                "traceId": None,
                "sessionTrackingToken": None,
                "queryTrackingToken": None,
                "runId": None,
            },
        },
    ]
    evalcli.wait_for_eval_set_entries.return_value = [{"id": "new-entry"}]

    with patch(
        "glean_gepa.focused_evalset.fetch_evalset_entry_tracking",
        return_value={
            "keep": {
                "deploymentId": "prod",
                "stt": "session-1",
                "runId": "run-1",
            }
        },
    ) as fetch:
        focused = ensure_focused_eval_set(
            evalcli,
            base_eval_set_name="Example",
            base_eval_set_version="v1",
            deployment_ids=["prod"],
            entry_ids=["keep"],
            bigquery_client=MagicMock(),
            bucket_type=SESSION_BUCKET_TYPE,
        )

    fetch.assert_called_once()
    request = evalcli.upload_eval_set.call_args.args[0]
    assert request["entries"] == [
        {
            "deploymentId": "prod",
            "user": "spark",
            "stt": "session-1",
            "query": "hello",
        }
    ]
    assert focused is not None
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
        bucket_type=SESSION_BUCKET_TYPE,
    )

    assert focused is not None
    assert evalcli.upload_eval_set.call_args.args[0]["entries"] == [
        {
            "deploymentId": "prod",
            "stt": "session-1",
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


def test_ensure_focused_eval_set_reuses_a_nonempty_retry_version():
    evalcli = MagicMock()
    evalcli.get_eval_set_version.return_value = {"name": "empty-deterministic"}
    prefix = focused_eval_set_version("v1", ["keep"])
    retry_version = f"{prefix}_retry_c8ad"
    evalcli.list_eval_set_versions.return_value = [{"version": retry_version}]
    evalcli.list_eval_set_entries.side_effect = [
        [],
        [{"id": "ingested-1"}, {"id": "ingested-2"}],
    ]

    focused = ensure_focused_eval_set(
        evalcli,
        base_eval_set_name="Example",
        base_eval_set_version="v1",
        deployment_ids=["prod"],
        entry_ids=["keep"],
    )

    assert focused is not None
    assert focused.version == retry_version
    assert focused.entry_count == 2
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


def test_ensure_focused_eval_set_retries_when_existing_version_has_no_entries():
    evalcli = MagicMock()
    evalcli.get_eval_set_version.return_value = {"name": "existing"}
    evalcli.list_eval_set_entries.return_value = []
    evalcli.wait_for_eval_set_entries.return_value = [{"id": "new-entry"}]
    source_entries = [{"id": "keep", "deploymentId": "prod", "query": "how do I list shell files"}]

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
    source_entries = [{"id": "keep", "deploymentId": "prod", "query": "how do I list shell files"}]

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
    source_entries = [{"id": "keep", "deploymentId": "prod", "query": "how do I list shell files"}]

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
