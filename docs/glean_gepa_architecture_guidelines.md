# Glean GEPA architecture guidelines

This is a short handoff guide for the `glean_gepa` integration.

## Target architecture

```text
CLI / experiment setup (`runner.py`, `--config configs/<experiment>.yaml`)
        |
        v
GEPA engine wiring (`api.py`)
        |
        +--> EvolutionaryProposer
        |      selects parents, requests reflection, creates and screens children
        |
        +--> TeacherStudentAdapter
        |      evaluates student vs. teacher through EvalCLI + Judge
        |      objectives (objectives/registry.py): tool_match, citation_match, agentic_preference,
        |      escalation_match
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
and best candidate on the validation set and enforces the `objective.validation`
gates from the experiment YAML (judge metrics with a `min`). A failed gate exits
the run unsuccessfully. These evals do not feed the search.

**Toggle.** `run.customer_eval: false` in the YAML, or `--no_customer_eval` on
the CLI, skips this step. The search still runs and the best candidate is still
written; the log records that the gates were not enforced. The CLI flag wins over
the YAML (`--customer_eval` re-enables it). Default is on. Use this while external
(customer) deployments are unavailable.

**Where validation runs.** Two paths, chosen by whether `data.val_eval_versions`
is set:

- *Unpinned* (no `val_eval_versions`): the runner samples customer deployments
  from the pool, persists the sample in `<run_dir>/cache/glean_customer_deployments.json`,
  and picks the `val_version_count` newest **Glean Chat V2 Medium** versions fully
  published to that sample. In-loop validation evals then run on those customer
  deployments. This path needs external evals.
- *Pinned* (`val_eval_versions: [YYYYMMDD]`): validation is scored on
  `data.deployment_ids` (normally `scio-prod`) with `data.val_eval_set_name`
  (defaults to the training set). Set `train_eval_versions` too so train and val
  are disjoint; otherwise training is every `scio-prod` version in the
  `lookback_days` window. One or two val versions are allowed. Use this path when
  customer deployments are off.

`customer_eval: false` only skips the post-search gate. It does not change which
path the in-loop validation set uses; pin `val_eval_versions` for that.

The post-search gate runs on the same valset the search used. So with
`val_eval_versions` pinned and `customer_eval: true`, the seed-vs-best pair runs on
the internal `scio-prod` set and the `objective.validation` gates are enforced
there, with no customer deployment involved. "Customer" in the name refers to the
unpinned default, not a requirement. `teacher_student_waldo` uses this combination.

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

## Experiment YAML (`configs/*.yaml`)

Each file under `src/glean_gepa/configs/` is one complete experiment. There is
no inheritance or pack merging: the `signals`, `objective`, `screening`, and
`reflection` sections in the file are the whole experiment. A `packs:` key fails
the load. Run one with `--config <stem>` (or a path); CLI flags override YAML.

```bash
uv run python -m glean_gepa.runner --config teacher_student_waldo
uv run python -m glean_gepa.runner --config single_model_shell --max_metric_calls 4
```

Shipped experiments:

| File | Mode | Primary | Notes |
|---|---|---|---|
| `single_model_shell.yaml` | single_model | `shell_success_rate` | Shell-tool success from Agentspan telemetry. |
| `teacher_student.yaml` | teacher_student | `agentic_preference_rate` | Pairwise AGENTIC_JUDGE vs the teacher; screens on the same judge. |
| `teacher_student_tool.yaml` | teacher_student | `tool_alignment` | First-tool match. Weighted screen. |
| `teacher_student_agentic_1/2.yaml` | teacher_student | `agentic_preference_rate` | Pinned train/val slices, `screening.kind: none`. |
| `teacher_student_waldo.yaml` | teacher_student | `tool_alignment` | Waldo router prompt; student and teacher both run Waldo (`waldo:PROVIDER:MODEL[:effort]`). |
Sections:

- `signals` -- every metric the run scores or reports. `source` is a key of
  `OBJECTIVES[mode]` in `objectives/registry.py` (`tool_match`,
  `citation_match`, `agentic_preference`, `escalation_match`; `shell_telemetry`, `loop_telemetry`),
  `cortex_judge` (with `type` and `kind`), or `constant`. Names must be unique.
  A telemetry source not registered for the file's `mode` fails the load.
- `objective` -- `primary` (parent selection; must be scorable), `composite`
  (weights summing to 1; every name must be scorable), `frontier_type`,
  `focused_bucket_type`, `params` (objective knobs such as `skipped_tools`,
  `failure_score_below`), and `validation` (judge-metric floors for the
  post-search customer eval).
- `screening` -- the focused child gate. `kind: high_signal_fix_rate` with
  `threshold` and `high_signal`; `kind: correctness_floor`; `kind: none`; or
  `weights` for a blend of summary metrics. A judge named only in `weights`
  is still started.
- `reflection` -- `editable_modules`, `failure_label`, `report_title`, and
  per-module prompt overrides under `modules.<KEY>`.
- `run`, `models`, `data`, `search` -- runner defaults (`run.customer_eval`,
  `data.train_eval_versions`, `data.val_eval_versions`, ...) that CLI flags override.
- `eval` -- eval-run creation overrides: `runner_type` and `sc_params`
  (string or list of `key=value`; the Waldo config uses `GLEAN_CHAT` and the
  production Waldo harness params).

In teacher_student mode, validation evals start every configured pairwise judge.
Full-train evals and focused screen slices start only the judges the search
reads: the primary, anything in `objective.composite`, and anything in
`screening.weights`. A judge listed only under `objective.validation` (the Waldo
correctness judge) therefore runs once per candidate, on the val eval.

## Adding a prompt to optimize (`prompts/<name>/`)

Each scio prompt GEPA can override is one folder under `src/glean_gepa/prompts/`, named
after the scio template (`data/prompts/templates/<name>.prompt`). The compiled text is sent
as `llmo.per_prompt_overrides.<name>`. `prompt_targets.py` loads every folder; nothing else
needs a code change.

1. Copy the scio template to `prompts/<name>/template.prompt` and wrap each editable span in
   `{#KEY}` ... `{/KEY}` marker lines. Deleting the marker lines must give back the scio
   file. Sections may nest; a bare `{KEY}` declares a section with empty stock text.
2. Write `prompts/<name>/target.yaml`:

   ```yaml
   harnesses: [coding]          # harnesses that render this prompt: coding, waldo
   template:
     key: MY_PROMPT             # candidate key for the frame around the sections
     editable: true             # false keeps the frame fixed
   sections:
     MY_SECTION:
       token_budget: 512
       frame: You are rewriting ...   # fixed editing contract shown to the reflector
       required_placeholders: [user_name]   # [[...]] a rewrite must keep
       required_conditionals: [has_search_tools]   # <<<[[...]] a rewrite must keep
       # fill: verbatim         # default strips the text and falls back to stock when empty
       # drop_empty_line: true  # an empty fill removes the slot's line
   render:                      # scParams every eval in the run needs to render this prompt
     sc_params: [co.some.flag=1]
     drop_sc_params: []         # preset entries to remove
   ```

3. Seed: a JSON object of keys, or a `.prompt` file marked like `template.prompt` (copy a
   scio PR's version of the file and add the same markers). Without a seed, stock text is
   the seed.
4. Point a config's `reflection.editable_modules` at the section keys. An objective's
   `module_responsibilities` or `reflection.modules.<KEY>` overrides the YAML frame.

The runner refuses a section with no slot in the seed, a seed missing required markup, and
a model whose harness never renders the prompt. Reflection rejects a variant that drops a
slot, a conditional, or required markup. `coding_agent_loop_system` (Coding Harness),
`core_tool_descriptions` (tool `schema.description` overrides), and `waldo_system` are the
worked examples. The `new-prompt-target` skill (`.cursor/skills/new-prompt-target/`) walks
through the whole wiring.

## Implementation rules

1. Select the concrete adapter explicitly in `runner.py`; keep each adapter free of branches for the other evaluation path.
2. Use a small protocol/base type only for the methods the proposer actually calls: `evaluate`, `make_reflective_dataset`, and `propose_new_texts`. Prefer duplicated small methods over a large inheritance hierarchy.
3. Each adapter's constructor must require only its own dependencies. Invalid combinations should be impossible to construct.
4. Keep objective names and score direction stable: all GEPA scores are higher-is-better; shell error rate enters GEPA as `shell_success_rate`.
5. Preserve the distinction between a selection score and reflection diagnostics. Scores choose candidates; traces, error strings, and per-entry data explain what to edit.
6. Cache keys must include every result-changing input: eval-set identity/version, model, prompt hash, and run label. Keep fresh eval-set cache behavior explicit.
7. Every extraction step gets characterization tests before cleanup. Remote EvalCLI/BigQuery runs are smoke tests, not unit tests.
8. A cached eval run is reused only when Cortex reports it `usable`: every task terminal, or succeeded+failed more than 9x the unfinished remainder. A run whose cancelled tasks are at least its finished tasks (someone killed it) is `missing` and is dropped from `glean_eval_run_cache.json` and relaunched. Cancelling an eval therefore needs no manual cache edit; restarting the runner recreates it.

## Adding an objective

An objective turns one eval run (or a teacher/student pair) into per-entry
scores and reflection evidence. Adapters, the proposer, caching, and the
reflection frame are shared; you write the parts that know your signal.

Adding one is one class file, one entry in `OBJECTIVES`, and a test.

### Start from a template

Two complete, importable objectives live beside the real ones. Copy the one
that matches where your signal comes from, rename, and fill the `TODO`s. The
question is: does the eval run's analysis view already have your number?

| Answer | Template | Lines | Examples |
|---|---|---|---|
| Yes: a judge score, a count, or a yes/no the run already recorded per entry | `objectives/_template_evalcli.py` | ~210 | agentic_preference; a downvote rate; any judge dimension |
| No: the signal is in span telemetry | `objectives/_template_agentspan.py` | ~300 | tool_match, citation_match, loop_efficiency, shell |

Not sure? Call `evalcli.get_analysis_view(eval_id)` on a recent run. If the field
you would score is in an entry's `metadata` or a judge's `outputs`, it is evalcli.
Both templates open with this same question.

Both instantiate and pass pyright as-is. Each is one file in the same layout
as the shipped objectives, top to bottom:

1. **Types.** One `EntryMetrics` dataclass with `entry_id`, `passed`, `score`;
   one aggregate dataclass; the `Analysis` frame alias.
2. **Parse and reduce.** `parse_row(row) -> EntryMetrics | None` and
   `aggregate(per_entry) -> Aggregate`; `pass_rate` and `mean_score` in `utils.core`
   cover the two common reductions.
3. **Source.** Agentspan: one SQL query returning one row per entry, passed to
   `fetch_agentspan_analysis`. EvalCLI: one call, then `build_analysis`.
4. **Feedback.** The sentence the reflector reads for a failing entry, built
   inline in `build_reflective_example`.
5. **Objective class.** `SingleModelObjective[Analysis]` or
   `TeacherStudentObjective[Analysis]`, with the hooks below.

### The four decisions

Everything specific to your objective is an answer to one of these. The
template marks where each goes.

| Decision | Where it lands |
|---|---|
| What is one entry, and when has it passed? | `EntryMetrics.passed` / `.score` |
| What does the run score, and which field is `0` while telemetry is still landing? | the aggregate dataclass; `pending_count` on the class |
| Where do the rows come from? | `build_*_per_entry_query` + `fetch_agentspan_analysis`, or one EvalCLI call + `build_analysis` |
| What should the prompt do differently for a failing entry? | the `feedback` string built inline in `build_reflective_example` |

### Files you touch

| File | What goes there |
|---|---|
| `objectives/<signal>.py` | The copied template. One file: types, parsing, source, feedback, class. |
| `objectives/registry.py` | One import and one entry in `OBJECTIVES[mode]`, keyed by `source`: the string an experiment YAML's `signals[].source` uses to select you. |
| `configs/<experiment>.yaml` | A self-contained experiment that declares a signal with your `source` and sets `objective.primary`, `composite`, `screening`, and `reflection`. Copy `single_model_shell.yaml` or `teacher_student.yaml`. |
| `tests/test_<signal>_objective.py` | At least a selection test (below), a scoring test, and a reflective-example test. |

Do not touch the adapters, `base.py`, or anything under `objectives/utils/`
unless every existing objective needs the change. Never import `registry.py`
from inside `glean_gepa.objectives`: the objective classes import `al_adapter`,
which imports the package, so the catalog must stay a leaf module.

### Class attributes

```python
name = "loop_efficiency"                 # objective score key; also the signal name in the experiment YAML
telemetry_dimensions = ("loop_efficiency",)
focused_bucket_type = QUERY_CANONICAL_BUCKET_TYPE
failure_label = "HIGH-SIGNAL FAILURES (extra loops)"
module_responsibilities = {WRITING_CODE_KEY: "..."}   # optional; seeds the reflection prompt
```

`SingleModelObjective` also needs `pending_telemetry_label` and `pending_count`,
which name the aggregate field that is `0` while telemetry is still ingesting.

### Hooks

The base classes are generic in the frame your `analyze()` returns:
`class LoopEfficiencyObjective(SingleModelObjective[EvalRunLoopCountAnalysis])`.
Every hook then receives that type, and pyright flags a hook typed against the
wrong frame.

| Hook | Returns | Notes |
|---|---|---|
| `analyze(eval_id, *, request)` / `analyze(teacher_eval_id, student_eval_id, *, request)` | your `Analysis` | Wrap the fetch in `self.cached_eval_analysis(...)` (single model) or `self.cached_paired_analysis(...)` (teacher/student). Read `request.wants_per_entry`, `request.wants_traces`, `request.hydrate_action_inputs` to size the query. |
| `focused_pass_rate(analysis, requested_entry_ids)` | `float` | Share of the requested entries that pass. |
| `entry_row(entry_id, metrics, analysis, ctx)` | `ScoredRow` | One entry. `ctx` carries `eval_set_name`, `deployment_id`, and `ctx.entry_query(entry_id)`. Put every field `build_reflective_example` will read into `output`. |
| `aggregate_row(analysis, ctx)` | `ScoredRow` | The whole-run row used for validation batches. |
| `failure_pattern(component_name, trajectory)` | `tuple` | Grouping key for near-duplicate failures; return `()` to disable. |
| `build_reflective_example(component_name, trajectory, candidate)` | `ReflectiveExample` | Compute `feedback` (and `generated` for teacher/student), then `return self.reflective_example(trajectory, feedback=..., action_inputs=..., execution_errors=...)`. The helper owns `Inputs`, `Metrics`, and the evidence caps. |
| `log_analysis(analysis)` | `None` | Single model only. Call `core.log_analysis(analysis, label=, headline=, entry_line=)`. |
| `validate_full_eval(analysis)` | `None` | Teacher/student only. Log the comparison; raise when a full eval has nothing to compare. |
| `require_compared_entries(analysis)` | `None` | Teacher/student only, optional. Raise `core.NoComparedEntriesError` with a hint when `compared_entries == 0`; default does nothing. |

Optional overrides with sensible defaults: `is_pending`, `aggregate_score`,
`cache_hit_is_sufficient`, `analysis_is_cacheable`, `cache_payload` /
`load_cache` (implement both if your analysis should survive a resume).

### Scores

- Every score is higher-is-better in `[0.0, 1.0]`.
- `objective_scores` in each row must contain `self.name`. Extra keys are fine;
  the experiment YAML's `objective.composite` decides which ones enter the composite.
- Provisional results (telemetry not yet ingested) return an analysis whose
  aggregate reports `0` entries. `is_pending` sees that and the adapter retries.

### Register and test

```python
# objectives/registry.py
from glean_gepa.objectives.downvote_judge import DownvoteJudgeObjective

OBJECTIVES = {
    ...
    "single_model": {
        ...
        "downvote_judge": DownvoteJudgeObjective,
    },
}
```

```python
# tests/test_downvote_judge_objective.py  (mirrors tests/test_loop_objective.py)
from glean_gepa.objectives.registry import build_objective

def test_selected_from_its_signal() -> None:
    signals = [{"name": "downvote_judge", "source": "downvote_judge"}]
    objective = build_objective("single_model", signals, bigquery_client=MagicMock())
    assert isinstance(objective, DownvoteJudgeObjective)
```

`tests/test_objectives.py` already instantiates every entry in `OBJECTIVES`
and checks its mode base class and required class attributes, so a missing
abstract hook or attribute fails CI before you write a scoring test. Never set
a new entry as `DEFAULT_SOURCE`; each mode already has one.

### What `objectives/utils/` gives you

You call these; you do not add to them for one objective.

| Module | What it is |
|---|---|
| `core` | The frame: `RunAnalysis[A, E]` / `PairedRunAnalysis`, `EntryMetricsLike`, `build_analysis`, `empty_analysis`, `select_high_signal`, `require_compared_entries`, `log_analysis`, `EVIDENCE_LIMIT`. |
| `agentspan` | BigQuery: `bounds_query` (`"= @eval_id"` or `"IN UNNEST(@eval_ids)"`), `paired_role_query` (the teacher/student FULL OUTER JOIN scaffold), `fetch_agentspan_analysis` (window → query → `filter_rows` → `parse_row` → `post_parse` → high-signal → `enrich` → `aggregate`). |
| `traces` | `enrich_action_inputs`: tool payloads from detailed EvalCLI traces for high-signal entries. You supply one `apply(metrics, fetched, entry_id)`. |
| `agentspan_query` | Shard-window and table constants, `default_date_range`, `wildcard_shard_filter`, `EVAL_ENTRY_ID_EXPR`. |
| `evalset_entries` | `fact.*` lookups the adapters use to build focused replay sets. |
| `tool_names` | `SKIPPED_TOOL_NAMES`, `scored_tool_sequence`, `first_tool_name`, `first_tool_mismatch_pair`. Shared with `prompt` and `run_log`. |
| `mismatch` | Grouping near-duplicate failures for reflection. |

Two shipped objectives depart from the template and say why in their module
docstrings: `shell.py` (own aggregate query; `action_run_id`-keyed enrichment)
and `agentic_preference.py` (no SQL, no aggregate on the frame).

### Checklist

- [ ] One file under `objectives/`, in the template layout. Imports from `utils/`, never from another objective.
- [ ] Class binds its frame: `SingleModelObjective[YourAnalysis]`.
- [ ] `analyze` goes through the shared cache helper.
- [ ] `build_reflective_example` goes through `self.reflective_example`.
- [ ] Entry added to `OBJECTIVES`; `uv run pytest tests/test_objectives.py` passes.
- [ ] Experiment YAML added or extended; `uv run pytest tests/test_experiment_config.py` passes.
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
