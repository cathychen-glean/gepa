# Glean GEPA architecture guidelines

This is a short handoff guide for the `glean_gepa` integration.

## Target architecture

```text
CLI / experiment setup (`runner.py`)
        |
        v
GEPA engine wiring (`api.py`)
        |
        +--> EvolutionaryProposer
        |      selects parents, requests reflection, creates and screens children
        |
        +--> TeacherStudentAdapter
        |      evaluates student vs. teacher through EvalCLI + Judge
        |      objectives (objectives/registry.py): tool_match, citation_match, agentic_preference
        |
        +--> SingleModelAdapter
               evaluates one student through EvalCLI + BigQuery
               objectives (objectives/registry.py): shell_telemetry, loop_telemetry
               optionally creates a fresh replay eval set per candidate

Shared infrastructure (not adapters)
        - prompt.py: compile a candidate into an override
        - ALRunner: start/reuse EvalCLI runs
        - EvalCLI / BigQuery clients
        - cache records and cache persistence
        - common candidate, output, trajectory, and reflective-example types
        - shared reflection-dataset helpers only where their behavior is truly identical
```

Both adapters satisfy the same GEPA-facing contract: candidates are `dict[str, str]`; evaluations return `GleanEvaluationBatch`; and reflection is based on captured trajectories. The GEPA engine and `EvolutionaryProposer` must not branch on evaluation mode.

## Responsibility boundaries

### `TeacherStudentAdapter`

- Requires `ALRunner`, `Judge`, teacher/student model settings, and teacher/student cache state.
- Runs or reuses teacher and student evaluations, triggers the judge, and turns judged traces into per-entry trajectories.
- Owns correctness, tool-alignment, grounding, token, loop, and tool-error objective construction.
- Does not import BigQuery or know how shell-error analyses are queried.

### `SingleModelAdapter`

- Requires `ALRunner`, `BigQueryClient`, student model settings, and shell-analysis cache state.
- Runs or reuses a student evaluation, fetches per-entry shell-error metrics, and creates trajectories from high-signal failing entries.
- Owns the `shell_success_rate` objective, fresh-eval-set creation policy, and shell-error diagnostics passed to reflection.
- Does not create teacher runs, trigger judges, or emit teacher comparison fields.

### Shared code

- `ALRunner` is the only place that knows how to invoke and wait for EvalCLI evaluation runs.
- `prompt.py` is the only place that knows the prompt-override encoding.
- `fresh_evalset.py` is a service helper used by `SingleModelAdapter`, not an adapter itself.
- Cache serialization should use separate namespaces/records for run IDs, judge state, and shell analysis. Do not make one adapter deserialize fields owned by the other.
- Keep `Candidate`, `ModuleSpec`, prompt-budget helpers, and generic reflective-example formatting in neutral modules, not in either adapter file.

## What is already in place

- `src/glean_gepa/api.py` wires a Glean-specific proposer into the low-level `GEPAEngine`.
- `src/glean_gepa/evolutionary_proposer.py` handles frontier-parent selection, reflection-driven mutations, prompt-budget filtering, and child screening.
- `TeacherStudentAdapter` and `SingleModelAdapter` provide the two evaluation paths: teacher/student judging and single-model shell reliability.
- `ALRunner`, `Judge`, `shell_tool_error_util.py`, `fresh_evalset.py`, prompt compilation, and versioned cache serialization already provide most of the supporting pieces.

Keep evaluation behavior stable while changing the surrounding code. Do not simultaneously change scoring, candidate selection, or remote-evaluation semantics.

## Customer eval after optimization

When a real (non-`--fake_flow`) run finishes, `runner.py` runs the seed baseline
and best candidate on the same **Glean Chat V2 Medium** customer entries. The
eval-set **version is not configured**: EvalCLI is queried at runtime, and the
newest `YYYYMMDD` version available to every configured customer is used.

The best run is judged for pairwise correctness against the baseline. The
held-out check passes only when correctness is above 80% and the EvalCLI paired
comparison finds no statistically significant change (Benjamini-Hochberg
adjusted `p >= 0.05`) in average cost, average loops, or any tool invocation
rate. A failed gate exits the run unsuccessfully. These evals do not feed the
search.

## Reflection sampling CLI

Use all available reflective examples when each iteration's example set is
small enough to fit in one reflection prompt. Glean-specific near-duplicate
filtering isolates each example's `Execution Errors` and drops later examples
whose errors are within the configured Hamming distance of an earlier error:

```bash
uv run python -m glean_gepa.runner \
  --seed_candidate data/seed_candidate.json \
  --run_dir gepa_runs/run-002 \
  --max_metric_calls 10 \
  --judging_mode single_model \
  --eval_versions 20260806,20260824,20260820,20260815 \
  --reflection_samples all \
  --reflection_hamming_distance_k 10
```

## Implementation rules

1. Select the concrete adapter explicitly in `runner.py`; keep each adapter free of branches for the other evaluation path.
2. Use a small protocol/base type only for the methods the proposer actually calls: `evaluate`, `make_reflective_dataset`, and `propose_new_texts`. Prefer duplicated small methods over a large inheritance hierarchy.
3. Each adapter's constructor must require only its own dependencies. Invalid combinations should be impossible to construct.
4. Keep objective names and score direction stable: all GEPA scores are higher-is-better; shell error rate enters GEPA as `shell_success_rate`.
5. Preserve the distinction between a selection score and reflection diagnostics. Scores choose candidates; traces, error strings, and per-entry data explain what to edit.
6. Cache keys must include every result-changing input: eval-set identity/version, model, prompt hash, and run label. Keep fresh eval-set cache behavior explicit.
7. Every extraction step gets characterization tests before cleanup. Remote EvalCLI/BigQuery runs are smoke tests, not unit tests.

## Adding an objective

An objective turns one eval run (or a teacher/student pair) into per-entry
scores and reflection evidence. Adapters, the proposer, caching, and the
reflection frame are shared; you write the parts that know your signal.

### Files you touch

| File | What goes there |
|---|---|
| `objectives/utils/<signal>_util.py` | Fetch and parse: the BigQuery/EvalCLI query, an `EntryMetrics` dataclass, an `Analysis` dataclass, `empty_*_analysis`, `log_*_analysis`. Nothing here imports from `objectives/*.py`. |
| `objectives/<signal>.py` | The objective class. Subclass `TeacherStudentObjective` or `SingleModelObjective` and fill in the hooks below. |
| `objectives/registry.py` | One `ObjectiveSpec` in `BUILTIN_OBJECTIVES`. `source` is the string a pack YAML uses to select you. |
| `configs/packs/<pack>.yaml` | A pack that names your `source` and sets `objective.primary`, `composite`, `screening`, and `reflection`. Copy `loops.yaml`. |
| `tests/test_<signal>_objective.py` | At least the contract test (see below), a scoring test, and a reflective-example test. |

Do not touch the adapters, `base.py`, or `protocol.py` unless every existing
objective needs the change.

### Class attributes

```python
name = "loop_efficiency"                 # objective score key; also the pack signal name
telemetry_dimensions = ("loop_efficiency",)
focused_bucket_type = QUERY_CANONICAL_BUCKET_TYPE
failure_label = "HIGH-SIGNAL FAILURES (extra loops)"
module_responsibilities = {WRITING_CODE_KEY: "..."}   # optional; seeds the reflection prompt
```

`SingleModelObjective` also needs `pending_telemetry_label` and `pending_count`,
which name the aggregate field that is `0` while BigQuery is still ingesting.

### Hooks (all abstract)

| Hook | Returns | Notes |
|---|---|---|
| `analyze(eval_id, *, request)` / `analyze(teacher_eval_id, student_eval_id, *, request)` | your `Analysis` | Wrap the fetch in `self.cached_eval_analysis(...)` (single model) or `self.cached_paired_analysis(...)` (teacher/student). Read `request.wants_per_entry`, `request.wants_traces`, `request.hydrate_action_inputs` to size the query. |
| `focused_pass_rate(analysis, requested_entry_ids)` | `float` | Share of the requested entries that pass. |
| `entry_row(entry_id, metrics, analysis, ctx)` | `ScoredRow` | One entry. `ctx` carries `eval_set_name`, `deployment_id`, and `ctx.entry_query(entry_id)`. Put every field `build_reflective_example` will read into `output`. |
| `aggregate_row(analysis, ctx)` | `ScoredRow` | The whole-run row used for validation batches. |
| `failure_pattern(component_name, trajectory)` | `tuple` | Grouping key for near-duplicate failures; return `()` to disable. |
| `build_reflective_example(component_name, candidate, trajectory)` | `ReflectiveExample` | Compute `feedback` (and `generated` for teacher/student), then `return self.reflective_example(trajectory, feedback=..., action_inputs=..., execution_errors=...)`. The helper owns `Inputs`, `Metrics`, and the evidence caps. |
| `log_analysis(analysis)` | `None` | Single model only. Usually delegates to `log_*_analysis` in your util. |
| `validate_full_eval(analysis)` | `None` | Teacher/student only. Raise when a full eval has nothing to compare. |

Optional overrides with sensible defaults: `is_pending`, `aggregate_score`,
`cache_hit_is_sufficient`, `analysis_is_cacheable`, `cache_payload` /
`load_cache` (implement both if your analysis should survive a resume).

### Scores

- Every score is higher-is-better in `[0.0, 1.0]`. `scored_rows_are_normalized`
  in `protocol.py` rejects anything else.
- `objective_scores` in each row must contain `self.name`. Extra keys are fine;
  the pack decides which ones enter the composite.
- Provisional results (telemetry not yet ingested) return an analysis whose
  aggregate reports `0` entries. `is_pending` sees that and the adapter retries.

### Register and test

```python
# objectives/registry.py
ObjectiveSpec(
    mode="single_model",
    source="downvote_judge",
    class_path="glean_gepa.objectives.downvote_judge:DownvoteJudgeObjective",
    summary="Judge did not downvote the answer.",
),
```

```python
# tests/test_downvote_judge_objective.py  (mirrors tests/test_loop_objective.py)
from glean_gepa.objectives import registry
from glean_gepa.objectives.protocol import ObjectiveProtocol, check_objective_contract

def test_satisfies_contract_and_is_registered() -> None:
    assert check_objective_contract(DownvoteJudgeObjective) == []
    assert isinstance(DownvoteJudgeObjective(bigquery_client=MagicMock()), ObjectiveProtocol)
    assert registry.resolve("single_model", "downvote_judge") is DownvoteJudgeObjective
```

`tests/test_objectives_catalog.py` already asserts that every spec in
`BUILTIN_OBJECTIVES` loads and satisfies the protocol, so a missing hook fails
CI before you write a scoring test.

### Checklist

- [ ] Util module fetches, parses, logs. No adapter or objective imports.
- [ ] Objective class sets the class attributes and implements every hook in the table.
- [ ] `analyze` goes through the shared cache helper.
- [ ] `build_reflective_example` goes through `self.reflective_example`.
- [ ] `ObjectiveSpec` added; `uv run pytest tests/test_objectives_catalog.py` passes.
- [ ] Pack YAML added; `uv run pytest tests/test_experiment_config.py` passes.
- [ ] `uv run ruff check src/ && uv run pyright src/` clean.

## Children cache (`glean_children_cache.json`)

`EvolutionaryProposer` persists generated children so a resumed run does not
re-reflect the same root on the same training slice. The file lives at
`<run_dir>/cache/glean_children_cache.json` unless `--children_cache_file` is set.

There is no schema version field. The file is the current record shape only;
unreadable files degrade to an empty cache and those children are proposed
again. Do not reintroduce version numbers or compatibility shims.

```json
{
  "training_slices": [
    {
      "train_ids": [0],
      "root_screening_scores": {"<root_id>": 0.75},
      "roots": {
        "<root_id>": [
          {
            "prompt_modules": {"WRITING_CODE": "..."},
            "eval_run_ids": [
              {
                "eval_set_name": "focused",
                "eval_set_version": "v1",
                "student_eval_run_id": "eval-child-1"
              }
            ],
            "screening_score": 0.5,
            "screening_passed": true
          }
        ]
      }
    }
  ]
}
```

`screening_passed` is stored, but a high-signal resume recomputes pass/fail
from `screening_score` and the current threshold.

## Suggested ownership split

The junior collaborator can own the **SingleModelAdapter vertical slice**—about one-third of the overall work—because it has a clear product boundary and can be tested with fakes. You retain the more coupled optimization and judge-comparison behavior.

### Her ownership: shell reliability (~1/3)

- Move shell-analysis cache read/write and cache-migration tests into its ownership boundary.
- Own `shell_tool_error_util.py`: classification edge cases, per-entry aggregation, query-builder tests, and diagnostic summaries.
- Own fresh eval-set behavior: replayability filtering, metadata, idempotency/cleanup policy, and EvalCLI-client fakes.
- Add local fixtures and a documented small shell-reliability smoke command.

### Your ownership: search and judged quality

- Extract/own `TeacherStudentAdapter`, including judge orchestration and judged-trace scoring.
- Own `EvolutionaryProposer`, module-selection strategy, parent selection, child screening, and evaluation-budget accounting.
- Own the candidate/module contract, prompt compilation contract, and any changes to GEPA wiring.
- Decide cross-cutting experiment policy: model defaults, eval-set versions, objective weighting, and which results are comparable.

### Pair-review changes

- The small adapter protocol and shared typed output/trajectory models.
- Objective names or score-direction changes.
- Cache-key changes that affect experiment reuse.
- The first end-to-end smoke test for each adapter.

## First collaboration milestone

1. Add characterization tests for the two adapter paths.
2. Move common types/utilities out of adapter modules; do not change behavior.
3. Verify `SingleModelAdapter` with its unit suite and one smoke evaluation.
4. Verify `TeacherStudentAdapter` with the same level of coverage.
5. Keep adapter selection explicit in the CLI and GEPA wiring.

This sequencing keeps each PR small and makes behavior changes obvious instead of hiding them inside a large refactor.
