---
name: new-objective
description: >-
  Creates or extends a glean_gepa optimization objective (teacher-student or single-model;
  BigQuery agentspan, EvalCLI, or Cortex-judge signal), validates it against a reference
  result, wires it into a new experiment YAML, and verifies every shipped config still
  builds the same objective and sends the same prompt. Use when the user asks to add,
  create, define, or change an objective, metric, score, signal, telemetry source, or match
  rate for GEPA prompt optimization, or to build an experiment config around one.
---

# New objective

An objective turns one eval run (single_model) or a teacher/student pair (teacher_student)
into per-entry scores in [0, 1] plus reflection evidence. Adapters, the proposer, caching,
and the reflection frame are shared; an objective supplies only what knows its signal.

The authoritative contract is `docs/glean_gepa_architecture_guidelines.md`, section
"Adding an objective". Read it before writing code. [reference.md](reference.md) holds the
repo-specific facts this skill relies on: which objective to start from, YAML extension
points, frontier behavior, data-source notes, test list, and gotchas.

## Ground rules

1. **Shipped configs keep working unchanged.** Every `src/glean_gepa/configs/*.yaml` must build
   the same objective (class, score keys, params, reflection briefs) and send the same seed
   prompt after your change. The verify script enforces this. New behavior goes in a new
   YAML file, or behind a new param whose default reproduces today's behavior.
2. **Add, don't alter.** Never change an existing objective's `name`, `telemetry_dimensions`,
   score semantics, or param defaults. Never set `default=True` on a new `ObjectiveSpec`:
   each mode already has one, and a second fails `load_builtins`. Don't touch adapters,
   `objectives/base.py`, `objectives/protocol.py`, or `objectives/utils/` unless every
   objective needs the change (ask the user first).
3. **Reuse before writing.** Prefer, in order: a YAML-only signal, a new param on an existing
   objective, then a new objective file copied from the closest existing one. Objectives
   never import from each other; shared code comes from `objectives/utils/`.
4. **uv needs network.** Run every `uv run ...` with `required_permissions: ["full_network"]`.

## Workflow

Copy this checklist and keep it updated:

```
- [ ] 1. Baseline snapshot (before any edit)
- [ ] 2. Requirements gathered and confirmed
- [ ] 3. Path chosen: YAML-only / param on existing / new objective
- [ ] 4. Metric reproduced against the reference
- [ ] 5. Objective implemented and registered
- [ ] 6. New experiment YAML wired
- [ ] 7. Editable prompt module confirmed to control the signal
- [ ] 8. Tests written
- [ ] 9. verify passes
- [ ] 10. Reported to the user
```

### 1. Baseline snapshot

Before editing anything:

```bash
bash .cursor/skills/new-objective/scripts/check_objective.sh baseline
```

This records what every shipped config builds and the current pyright errors, so step 9
can tell your regressions from pre-existing ones.

### 2. Requirements

Infer what you can from the request, any handoff spec, and the code. Then use AskQuestion
for anything still open. When a definitional choice has trade-offs, offer an
"explain more" option and explain the options with concrete numbers before deciding. Pin down:

- **Entry and score.** What one entry is, when it passes, and the exact classification rules
  (status values, precedence, what counts as "other"). Is the score symmetric agreement,
  one-sided, or continuous?
- **Exclusions and coverage.** Which entries are unscored: skipped, missing on one side,
  incomplete, retried traces (keep latest?). Any fields or spans that must *not* be used.
- **Mode.** teacher_student (match or beat a teacher run) or single_model (an absolute
  property of one run).
- **Source.** Where the number lives: span telemetry (BigQuery agentspan), the eval run's
  analysis view (EvalCLI), or a Cortex judge (YAML-only, teacher_student only).
- **Primary vs secondary.** One score. Secondary metrics are logged, not score keys; every
  score key becomes a frontier key (see reference.md, "Scores and the frontier").
- **Reference.** A known eval id (or pair) with expected numbers and, ideally, the SQL that
  produced them.
- **Experiment.** Which shipped config to copy, models, eval sets, validation gates, and which
  prompt modules reflection may edit.

Restate the final definition in one short paragraph and get a yes before step 4.

### 3. Choose the path

| Need | Path |
|---|---|
| The score is a Cortex judge's output (teacher_student) | YAML-only: a `source: cortex_judge` signal with `type` and `kind`. No code. |
| A fixed-value term in the composite | YAML-only: `source: constant`. |
| An existing objective with a different threshold, filter, or skip list | Add an `objective.params` key read with `self.experiment_param(key, default)`; the default must equal current behavior. |
| An existing objective with a different reflection brief or label | YAML-only: `reflection.modules.<KEY>` (replaces that module's brief wholesale), `reflection.failure_label`, `reflection.report_title`. |
| A new signal | New `objectives/<signal>.py`, copied from the closest objective in reference.md, "Where to start". |

### 4. Reproduce the reference (required)

Before wiring a YAML, run the objective's own fetch, parse, and aggregate code against the
reference eval ids. Every reference number must match exactly; explain any difference to
the user and resolve it. Write the script under `.tmp_debug/`, not `src/`.

- If Python BigQuery credentials fail ("Reauthentication is needed"), route the objective's
  SQL through the `bq` CLI with the shim in reference.md rather than blocking.
- If no reference exists, ask the user for one. If there truly is none, run on one recent
  real eval, show the user the distribution and three or four per-entry examples, and state
  that no reference was available.

### 5. Implement

Follow the guide's "Adding an objective" layout (types, parse and aggregate, source,
feedback, class) and its checklist. Repo-specific musts:

- Register one `ObjectiveSpec` in `objectives/registry.py`. Never set `default=True`.
- New fields on rollout output go into `adapter_types.py` as `NotRequired` keys.
- Log coverage (compared, unpaired, skipped by reason, incomplete) in `validate_full_eval`
  or `log_analysis`. For teacher_student, override `require_compared_entries` and raise
  `NoComparedEntriesError` with that breakdown. The base method does nothing.
  `utils.core.require_compared_entries` only checks that `per_entry` is non-empty; it does
  not carry skip reasons, so paired objectives raise the error themselves. See `tool_match.py`.
- `telemetry_dimensions` and `ScoredRow.dimension_scores` hold the score only.
- Write the reflection brief with `compose_responsibility`: what the score rewards, both
  failure directions, which edits move it, and how to read each example. See the briefs in
  `objectives/tool_match.py`.

### 6. Wire the experiment YAML

- Create `configs/<mode>_<signal>.yaml` by copying the closest shipped config. Never repoint
  a shipped config at the new objective.
- Only one telemetry objective runs per experiment. `build_objective` walks `signals` and
  takes the first source registered for the mode. `cortex_judge` and `constant` are not
  registered, so they do not count. Put the telemetry signal before any other telemetry source.
- Set `objective.primary`, `composite` (weights sum to 1), `frontier_type`,
  `screening` (usually `high_signal_fix_rate` with `high_signal: <name>`),
  `reflection.editable_modules`, and `run.dir`.
- Open the file with a comment: what the score is, any base-rate caveat, and the run command.
- Add the new source and config to the catalog lines in
  `docs/glean_gepa_architecture_guidelines.md` ("Shipped experiments" table and the
  `signals` source list).

### 7. Check the editable surface

Confirm the editable modules' text actually decides the behavior the signal measures.
Read the seed prompt and locate the rules that drive it. If they sit outside the editable
modules, tell the user and propose splitting them into a new slot module (reference.md,
"Splitting a prompt module"). Do the split only with the user's approval. The new seed
must compile byte-identical to the old one.

### 8. Tests

Create `tests/test_<signal>_objective.py` covering the list in reference.md, "Tests".
Model it on `tests/test_tool_match_objective.py` (teacher_student, agentspan) or
`tests/test_shell_objective.py` (single_model).

### 9. Verify

```bash
bash .cursor/skills/new-objective/scripts/check_objective.sh verify
```

This runs the full test suite, ruff on `src/` and changed tests, the shipped-config
fingerprint comparison, and pyright compared against the baseline. Fix until it prints
`PASS`. A `CHANGED <config>` line is a regression unless the user asked for that change;
`NEW <config>` lines are your new files.

### 10. Report

Lead with what was built and whether it matched the reference, with the numbers. Then list
the files changed, the run command, and caveats (base rates, coverage, judge gates).
Don't commit unless asked.

## Additional resources

- [reference.md](reference.md): where to start, YAML extension points, frontier facts,
  data sources and the `bq` shim, splitting a prompt module, the test list, gotchas.
- `docs/glean_gepa_architecture_guidelines.md`: the objective contract, hooks, and checklist.
- `src/glean_gepa/objectives/_template_agentspan.py`, `_template_evalcli.py`: copyable
  single-model starting points.
