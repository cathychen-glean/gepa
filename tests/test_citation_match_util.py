from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock

import pytest

from glean_gepa.experiment_config import load_experiment_config
from glean_gepa.objectives import build_objective
from glean_gepa.objectives.citation_match import CitationMatchObjective
from glean_gepa.objectives.utils.citation_match_util import (
    CitationMatchEntryMetrics,
    NoComparedCitationEntriesError,
    aggregate_citation_match_metrics,
    build_citation_match_per_entry_query,
    build_citation_match_time_bounds_query,
    citation_mismatch_pair,
    empty_citation_match_analysis,
    fetch_eval_run_citation_match_analysis,
    parse_citation_match_entry_metrics,
    require_compared_citation_entries,
    scored_citation_ids,
)
from glean_gepa.objectives.utils.mismatch import select_mismatch_groups


def test_citation_set_scoring_dedupes_and_ignores_order():
    assert scored_citation_ids(["b", "a", "b", "  "]) == ("b", "a")
    assert citation_mismatch_pair(("a", "b"), ("b", "a")) is None
    assert citation_mismatch_pair(("a", "b"), ("a",)) == ("missing:b", "extra:(none)")
    assert citation_mismatch_pair(("a",), ("a", "c")) == ("missing:(none)", "extra:c")
    assert citation_mismatch_pair((), ("x",)) == ("missing:(none)", "extra:x")

    match = parse_citation_match_entry_metrics(
        {"entry_id": "entry-1", "student_citations": ["b", "a"], "teacher_citations": ["a", "b"]}
    )
    assert match.citations_match
    assert match.missing == ()
    assert match.extra == ()

    mismatch = parse_citation_match_entry_metrics(
        {"entry_id": "entry-2", "student_citations": ["a"], "teacher_citations": ["a", "b"]}
    )
    assert not mismatch.citations_match
    assert mismatch.missing == ("b",)
    assert mismatch.extra == ()


def _trace_with_action_inputs(*action_inputs: str) -> dict:
    """A detailed-trace payload whose Execute Action spans carry ``action_input``."""
    return {
        "trace": {
            "spans": [
                {
                    "name": "Execute Action: Search",
                    "attributes": {"input": {"strValue": json.dumps({"action_input": action_input})}},
                }
                for action_input in action_inputs
            ]
        }
    }


def test_citation_match_queries_and_fetch():
    bounds_sql = build_citation_match_time_bounds_query()
    sql = build_citation_match_per_entry_query()
    assert "PARSE_DATE" not in bounds_sql
    assert "PARSE_DATE" not in sql
    assert "_TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', @search_start_date)" in bounds_sql
    assert "_TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', @start_date)" in sql
    assert "@student_eval_id" in sql and "@teacher_eval_id" in sql
    assert "citationId" in sql
    assert "FULL OUTER JOIN" in sql
    # Tool payloads are scrubbed from this table, so the query must NOT read them here
    # and must instead carry the scrub-safe locators used to fetch the detailed trace.
    assert "span_info.inputs" not in sql
    assert "agent_trace.trace_id" in sql
    assert "student_trace_id" in sql and "teacher_trace_id" in sql

    client = MagicMock()
    client.query.side_effect = [
        [{"min_start_ms": 1_786_363_200_000, "max_start_ms": 1_786_449_600_000}],
        [
            {
                "entry_id": "entry-1",
                "student_citations": ["a"],
                "teacher_citations": ["b"],
                "teacher_trace_id": "t-trace-1",
                "teacher_deployment_id": "scio-prod",
                "teacher_min_start_ms": 1_786_400_000_000,
                "teacher_max_start_ms": 1_786_400_050_000,
            },
            {"entry_id": "entry-2", "student_citations": ["a"], "teacher_citations": ["a"]},
        ],
    ]

    evalcli = MagicMock()
    evalcli.get_analysis_trace.return_value = _trace_with_action_inputs('{"query":"does BILL help"}')

    analysis = fetch_eval_run_citation_match_analysis(
        client,
        teacher_eval_id="teacher",
        student_eval_id="student",
        lookback_days=7,
        end_date=date(2026, 8, 11),
        evalcli=evalcli,
    )
    assert analysis.per_entry["entry-1"].citations_match is False
    # The teacher's tool payload is resolved from the detailed trace, not the scrubbed table.
    assert analysis.per_entry["entry-1"].teacher_action_inputs == ('{"query":"does BILL help"}',)
    assert analysis.per_entry["entry-2"].citations_match is True
    assert analysis.high_signal_entry_ids == ("entry-1",)
    assert analysis.aggregate.citation_match_rate == 0.5
    assert client.query.call_count == 2


def test_aggregate_and_empty_analysis():
    per_entry = {
        "a": CitationMatchEntryMetrics("a", ("doc-1",), ("doc-1",), True),
        "b": CitationMatchEntryMetrics("b", (), ("doc-1",), False),
    }
    aggregate = aggregate_citation_match_metrics("teacher", "student", per_entry)
    assert aggregate.compared_entries == 2
    assert aggregate.matching_entries == 1
    assert aggregate.citation_match_rate == 0.5

    empty = empty_citation_match_analysis("teacher", "student")
    with pytest.raises(NoComparedCitationEntriesError):
        require_compared_citation_entries(empty)


def test_select_citation_mismatch_groups():
    keys = [("missing:a", "extra:(none)")] * 3 + [("missing:(none)", "extra:b")] * 2 + [None]
    selected, groups = select_mismatch_groups(keys)
    assert selected == [0, 1, 2, 3, 4]
    assert groups[0] == ("missing:a", "extra:(none)", 3)


def test_citations_pack_constructs_the_citation_match_objective(tmp_path):
    mode = tmp_path / "mode.yaml"
    mode.write_text("schema_version: 1\nmode: teacher_student\npacks: [citations]\n")
    config = load_experiment_config(mode)
    objective = build_objective("teacher_student", config.signals, bigquery_client=MagicMock())
    assert config.primary_objective == "citation_match"
    assert isinstance(objective, CitationMatchObjective)


def test_citation_match_objective_scores_and_flags_mismatches():
    objective = CitationMatchObjective()
    analysis = empty_citation_match_analysis("teacher", "student")
    analysis.per_entry["e1"] = CitationMatchEntryMetrics(
        "e1", ("a",), ("a", "b"), False, teacher_action_inputs=tuple(f"q{i}" for i in range(6))
    )
    analysis.per_entry["e2"] = CitationMatchEntryMetrics("e2", ("a",), ("a",), True)
    rows = objective.scored_rows(
        analysis,
        focused=True,
        capture_traces=True,
        query="q",
        deployment_id="dep",
    )
    by_entry = {row.entry_id: row for row in rows}
    assert by_entry["e1"].dimension_scores["citation_match"] == 0.0
    assert by_entry["e2"].dimension_scores["citation_match"] == 1.0
    assert objective.is_high_signal(by_entry["e1"].output)
    assert not objective.is_high_signal(by_entry["e2"].output)
    example = objective.build_reflective_example(
        "MODULE",
        {
            "data": {"eval_set_name": "set"},
            "score": 0.0,
            "objective_scores": {"citation_match": 0.0, "completeness": 0.5},
            "output": by_entry["e1"].output,
        },
        {},
    )
    assert example["Action Inputs"] == [f"q{i}" for i in range(5)]
