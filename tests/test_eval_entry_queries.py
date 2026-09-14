from __future__ import annotations

from unittest.mock import MagicMock

from glean_gepa.eval_entry_queries import entry_queries_from_listing, fetch_entry_queries


def test_entry_queries_reads_both_listing_shapes_and_skips_unusable_rows():
    resolved = entry_queries_from_listing(
        [
            {"id": "entry-1", "input": {"query": "how many employees in Belgium"}},
            # Some listings carry the query at the top level rather than under input.
            {"id": "entry-2", "query": "  draft the BNZ follow-up  "},
            # Unusable: no id to join on, no query text, or a non-string payload.
            {"input": {"query": "orphaned"}},
            {"id": "entry-3", "input": {"query": "   "}},
            {"id": "entry-4", "input": {}},
            {"id": "entry-5", "input": {"query": {"text": "nested"}}},
            "not-a-mapping",
        ]
    )

    assert resolved == {
        "entry-1": "how many employees in Belgium",
        "entry-2": "draft the BNZ follow-up",
    }


def test_fetch_entry_queries_degrades_to_empty_when_the_listing_is_refused():
    """Customer eval sets are PII-gated: listing raises, and reflection must keep
    running on the eval_set:version stand-in rather than failing the batch."""
    evalcli = MagicMock()
    evalcli.list_eval_set_entries.side_effect = RuntimeError("403 PII-gated")

    resolved = fetch_entry_queries(
        evalcli,
        eval_set_name="Glean Chat V2 Medium",
        eval_set_version="20260907",
        deployment_ids=["glean-televox"],
    )

    assert resolved == {}


def test_fetch_entry_queries_skips_the_call_without_an_eval_set():
    evalcli = MagicMock()

    assert fetch_entry_queries(evalcli, eval_set_name="", eval_set_version="20260907", deployment_ids=[]) == {}
    assert fetch_entry_queries(None, eval_set_name="set", eval_set_version="20260907", deployment_ids=[]) == {}
    evalcli.list_eval_set_entries.assert_not_called()


def test_fetch_entry_queries_passes_the_requested_version_and_deployments():
    evalcli = MagicMock()
    evalcli.list_eval_set_entries.return_value = [{"id": "entry-1", "input": {"query": "case 007"}}]

    resolved = fetch_entry_queries(
        evalcli,
        eval_set_name="Glean Chat V2 Medium",
        eval_set_version="20260907",
        deployment_ids=["scio-prod"],
    )

    assert resolved == {"entry-1": "case 007"}
    evalcli.list_eval_set_entries.assert_called_once_with(
        eval_set_name="Glean Chat V2 Medium",
        eval_set_version="20260907",
        deployment_ids=["scio-prod"],
    )
