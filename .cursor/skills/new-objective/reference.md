# New objective: reference

## Contents

- Where to start
- YAML extension points
- Scores and the frontier
- Data sources
- The `bq` CLI shim
- Splitting a prompt module
- Tests
- Gotchas

## Where to start

Copy the closest objective, rename, and delete what you don't need. All paths are under
`src/glean_gepa/objectives/`.

| Mode | Signal | Start from | What it shows |
|---|---|---|---|
| teacher_student | agentspan; compare one value per entry | `tool_match.py` (first tool), `citation_match.py` (sets) | `paired_role_query` FULL OUTER JOIN scaffold, `mismatch_pair`, trace hydration of tool payloads, `NoComparedEntriesError` with a breakdown, `compose_responsibility` briefs |
| teacher_student | EvalCLI; pairwise judge per entry | `agentic_preference.py` | no SQL, no aggregate on the frame; loss/keep reflection selection |
| teacher_student | a Cortex judge score only | none | YAML-only `cortex_judge` signal |
| single_model | agentspan; per-entry count or rate | `_template_agentspan.py`, `loop.py` | `fetch_agentspan_analysis` pipeline, `pending_count` |
| single_model | agentspan; needs its own aggregate query | `shell.py` | `action_run_id`-keyed enrichment |
| single_model | EvalCLI analysis view | `_template_evalcli.py` | `build_analysis` / `empty_analysis`, no BigQuery client |

Cortex judges are scorable only in teacher_student (`_MODES_WITH_CORTEX_JUDGES` in
`experiment_config.py`).

Required class members, beyond the abstract methods and the `name`, `telemetry_dimensions`,
`focused_bucket_type`, and `failure_label` attributes on the two bases in `objectives/base.py`:

- teacher_student: `teacher_compared_key`, `student_compared_key`, `mismatch_pair`
  (returns `(teacher, student)` or `None`), `validate_full_eval`.
- single_model: `pending_telemetry_label`, `pending_count`, `log_analysis`.
- Both: `__init__` must accept `bigquery_client` and `lookback_days` (or `**kwargs`).
  `build_objective` passes both as keywords.

## YAML extension points

`configure_objective` in `objectives/base.py` applies these after construction. Class
attributes stay the unconfigured defaults.

| YAML key | Lands on | Notes |
|---|---|---|
| `objective.params.<key>` | `self.params`, read via `self.experiment_param(key, default)` | The way to add a knob to an existing objective without changing it. Existing: `failure_score_below` (all), `skipped_tools` (`tool_match`, `agentic_preference`). |
| `objective.focused_bucket_type` | `focused_bucket_type` | `QUERY_CANONICAL` or `SESSION` |
| `reflection.failure_label` | `failure_label` | Shipped configs set this, so a class-default change does not reach them. |
| `reflection.report_title` | `reflection_report_title` | |
| `reflection.modules.<KEY>` | `module_responsibilities[KEY]` | Replaces the class brief for that module wholesale. |
| `screening.high_signal` | `high_signal` | Score key the focused screen measures. |
| `signals[].name` | `signal_names` | |

Adding a param safely:

1. Read it with `self.experiment_param("my_knob", <current behavior>)`.
2. Test the default path (shipped configs unchanged) and the set path.
3. Document it in the new config's comment and in the guide's `objective.params` line.

## Scores and the frontier

- Every score is a float in [0, 1] (not a bool) and higher is better.
- A teacher_student row's `objective_score` is
  `{**constant_scores, **row.dimension_scores, **judge_scores}`.
- With `frontier_type: hybrid`, the frontier keys are one per val instance (the composite)
  plus one per `objective_score` key, each averaged across val instances. Parent selection
  uses the union. So every telemetry dimension and every judge signal is a frontier key,
  even at weight 0 in `composite`. Keep diagnostics out of `dimension_scores`; log them.
- `objective.validation` floors gate only the post-search customer eval, never the frontier.
- Validation evals start every configured pairwise judge. Train and screen evals start only
  the judges the search reads (primary, `composite`, `screening.weights`).
- `cached_paired_analysis` does not cache a provisional analysis with
  `compared_entries == 0`, so a still-ingesting run is retried, not frozen.

## Data sources

**BigQuery agentspan**

- Table: `scio-apps.scrubbed_agentspan.scrubbed_agentspan_*`
  (`DEFAULT_AGENTS_SPAN_TABLE`). Restrict shards with `wildcard_shard_filter` and a bounds
  query (`bounds_query(eval_id_predicate=...)`).
- Entry id: `EVAL_ENTRY_ID_EXPR`. Join teacher and student on eval id plus entry id.
- Retries create several traces per entry. When the spec says "latest", pick the trace with
  `ARRAY_AGG(trace_id ORDER BY start_ms DESC LIMIT 1)[OFFSET(0)]`, then read fields only
  from that trace.
- Pass extra filtering to `fetch_agentspan_analysis` through `filter_rows`. Default
  high-signal is `not m.passed`; override `is_high_signal` when reflection should see a
  different subset.

**EvalCLI**

- Check `evalcli.get_analysis_view(eval_id)` on a recent run. If the field is in an entry's
  `metadata` or a judge's `outputs`, the objective needs no SQL.
- The objective receives the client as `request.evalcli` inside `analyze`.

## The `bq` CLI shim

Use this when Python BigQuery ADC fails and the user's `bq` CLI is authenticated. It runs
the objective's own parameterized SQL, so reference checks exercise the real code path.

```python
import json
import subprocess


class BqCliClient:
    def query(self, sql, params=()):
        args = ["bq", "--project_id=scio-apps", "query", "--use_legacy_sql=false",
                "--format=json", "--max_rows=100000"]
        for p in params:
            if isinstance(p.value, list):
                args.append(f"--parameter={p.name}:ARRAY<{p.type_}>:{json.dumps(p.value)}")
            else:
                args.append(f"--parameter={p.name}:{p.type_}:{p.value}")
        out = subprocess.run([*args, sql], capture_output=True, text=True, check=True).stdout
        rows = json.loads(out[out.index("["):])
        return [{k: {"true": True, "false": False}.get(v, v) for k, v in row.items()} for row in rows]
```

Pass it as the `client`/`bigquery_client`, and pin `end_date` and `lookback_days` to cover
the reference evals' dates. Run it with `full_network`.

## Splitting a prompt module

Use this when the rules that drive the signal sit in a frozen part of the prompt. Prompts
and their sections are data under `src/glean_gepa/prompts/<scio template>/`; see "Adding a
prompt to optimize" in `docs/glean_gepa_architecture_guidelines.md`. The worked example is
`WALDO_ROUTING`, the `{WALDO_ROUTING}` slot in `WALDO_SYSTEM`.

1. Pick one contiguous span of the live seed text. A contiguous span is what lets the
   compiled seed stay byte-identical.
2. In `prompts/<name>/template.prompt`, wrap the stock span in `{#KEY}` ... `{/KEY}`
   marker lines. In `target.yaml`, declare the section with a token budget that has
   headroom over the seed, a frame, and any `required_placeholders` /
   `required_conditionals`. Stripped fill is the default, so separators around the slot
   belong in the template, not in the module.
3. Write a new seed file with the span cut out of the template into the new key (or a
   marked `.prompt` seed). Leave the old one in place, since shipped configs use it. Assert
   that the new seed compiles to the same prompt as the old one.
4. Point the new config's `editable_modules` and the objective's `module_responsibilities`
   at the new key. Raise `search.global_token_cap` above the editable target's modules plus
   the new budget.

To audit the split, run `uv run python .cursor/skills/new-prompt-target/scripts/prompt_target_check.py
check <name> --source <scio file> --seed <new seed> --editable <KEY>`. It checks byte
identity, budgets, frames, and placeholders.

The seed builder pins the seed's template and every frozen section it renders, and refuses
an editable section with no slot. Slots, `<<<[[...]]>>>` conditionals, and declared
required markup are protected during reflection: `drops_render_slot`, `drops_conditional`,
and `PromptModule.missing_markup` reject variants that lose them.

## Tests

`tests/test_<signal>_objective.py` should cover:

- [ ] Catalog: `OBJECTIVES[mode][source] is cls`; `tests/test_objectives.py` instantiates it
      and checks its base class and required attributes.
- [ ] Parse and classify: each status, precedence, and exclusion rule.
- [ ] Aggregate on a fixture shaped like the reference (counts and rates).
- [ ] SQL guards: required filters and exclusions are in the query; forbidden fields are not.
- [ ] Fetch with a mocked client: filtering, coverage counts, and warnings. Pin `end_date`
      in the test, or mocked bounds fall outside the default window that ends today.
- [ ] Zero compared entries raises with the breakdown (teacher_student).
- [ ] Rows emit exactly the declared `dimension_scores` keys.
- [ ] Reflective example: the feedback text for each failure direction.
- [ ] Custom mismatch selection, if you overrode `_select_mismatch_groups`.
- [ ] The new config loads (`load_experiment_config`) and `build_objective` returns the class.

`tests/test_objectives.py` already checks that every catalog spec loads, validates, and
instantiates. The guide calls it `test_objectives_catalog.py`, but the file is
`test_objectives.py`.

## Gotchas

- `uv` goes through a socket-firewall wrapper. Without `full_network` it fails with
  "Unable to reach Socket API".
- pyright has pre-existing errors in this repo. Judge your change by the verify script's
  "new errors", not by the total.
- `ruff format` is enforced: run `uv run ruff format src/ <your test file>` before verify.
- Python BigQuery ADC expiry ("Reauthentication is needed"): use the `bq` shim, or ask the
  user to run `gcloud auth application-default login`. Never ask for credentials.
- An existing run directory caches eval-run ids keyed by candidate. A config whose seed
  changed should use a fresh `run.dir`, or tell the user about the stale entries.
