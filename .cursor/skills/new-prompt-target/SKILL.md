---
name: new-prompt-target
description: >-
  Wires a scio prompt template into glean_gepa as an optimizable prompt target: fetches the
  template from scio master or a scio PR, confirms scio can override it and under which
  harness and flags it renders, splits it into editable sections with {#KEY} markers, writes
  target.yaml (budgets, frames, required markup, render scParams), seeds it, wires an
  experiment YAML, and verifies every shipped config still sends the same prompt. Use when the
  user asks to optimize, add, wire, or modularize a new prompt, template, or system prompt
  (often given as a scio PR or file link), or to split a prompt into editable sections.
---

# New prompt target

A prompt target is one folder, `src/glean_gepa/prompts/<scio template name>/`, holding
`template.prompt` (the scio file with `{#KEY}` ... `{/KEY}` markers around editable spans)
and `target.yaml`. `prompt_targets.py` loads every folder, and the runner, the reflector,
budgets, seeds, and the scParam override all read from it. No Python changes are needed.

Read "Adding a prompt to optimize" in `docs/glean_gepa_architecture_guidelines.md` for the
`target.yaml` schema. [reference.md](reference.md) has the scio facts this skill relies on
(override plumbing, render rules, gating, harnesses), the splitting guidance, and gotchas.
The worked examples are `prompts/waldo_system/` (editable frame with conditional-guarded
sections) and `prompts/coding_agent_loop_system/` (fixed frame, nested sections).

## Chained skills

- **new-objective** (`.cursor/skills/new-objective/`): its `check_objective.sh` provides
  the baseline and verify steps here. Follow that skill from its step 2 when no existing
  objective measures what the new sections control.
- Commit or open a PR only when the user asks. If they ask to babysit the PR, follow the
  `greenify` rule.

## Ground rules

1. **Shipped configs keep sending the same prompt.** A new target never changes another
   target's text or an existing config; verify enforces this.
2. **Stock compile is byte-identical to the scio file.** Markers only delimit sections.
   The checker's `--source` comparison must say `byte-identical` before you go further.
3. **Only overridable prompts.** The template name must be a field of
   `message PromptOverrides` in scio's `llm_options.proto`; otherwise scio silently ignores
   the override. Stop and tell the user if it is missing.
4. **Permissions.** `uv run` needs `full_network`. `gh api` output written into the
   workspace needs `all`.

## Workflow

Copy this checklist and keep it updated:

```
- [ ] 1. Baseline snapshot (before any edit)
- [ ] 2. Scio text fetched (master and/or PR) into .tmp_debug/<name>/
- [ ] 3. Override, render path, gating flags, and harness confirmed in scio
- [ ] 4. Sections, seed, objective, and models agreed with the user
- [ ] 5. Folder scaffolded
- [ ] 6. Sections marked and declared
- [ ] 7. Checker passes against the scio file
- [ ] 8. Seed written and checked
- [ ] 9. Experiment YAML wired (objective chosen or built)
- [ ] 10. Tests written
- [ ] 11. verify passes
- [ ] 12. Reported to the user
```

### 1. Baseline

```bash
bash .cursor/skills/new-objective/scripts/check_objective.sh baseline
```

### 2. Fetch the scio text

Save each version under `.tmp_debug/<name>/`, for example `master.prompt` and `pr<N>.prompt`.
The template lives at `data/prompts/templates/<name>.prompt`. For a PR link, see
reference.md, "Fetching scio text", which covers mapping a `#diff-<hash>` anchor to a path.

### 3. Confirm scio renders and overrides it

Answer these from the scio source; reference.md, "Scio facts", has the commands:

- Is `<name>` a `PromptOverrides` field? (The checker also tests this.)
- Which code path renders it (`Prompt(name='<name>', ...)`), which `[[args]]` it passes,
  and what gates it (search-config flags, feature checks). Which sibling template renders
  instead when a flag is off (for example `stripped_<name>` vs `<name>`)?
- Which harness reaches that path (`coding` or `waldo`), and does that harness's preset
  turn a gate off? If so, the target needs `render.sc_params` / `render.drop_sc_params`.
- Does another copy exist (TypeScript, Go, a different template)? This override does not
  reach those copies; tell the user.

### 4. Agree on the shape

Infer what you can, then use AskQuestion for what remains:

- **Sections.** Which spans are editable. Propose the split first (reference.md,
  "Splitting into sections").
- **Seed.** Stock text, or a PR's version of the file.
- **Signal.** An existing objective whose score the sections control, or a new one
  (new-objective).
- **Models.** Student and teacher on the target's harness.

### 5. Scaffold

```bash
uv run python .cursor/skills/new-prompt-target/scripts/prompt_target_check.py \
  scaffold <name> --source .tmp_debug/<name>/master.prompt --harness coding
```

This copies the file verbatim to `template.prompt`. It writes a `target.yaml` skeleton and
prints the placeholders and conditionals the file uses.

### 6. Mark and declare

- Wrap each editable span in `{#KEY}` / `{/KEY}` marker lines.
- Declare every marked key under `sections:` with `token_budget`, `frame`, and
  `required_placeholders` (every `[[fill]]` in that section's stock text), plus
  `required_conditionals` where a conditional must survive.
- Give the template module the placeholders left in the frame.
- Add `render:` when step 3 found a gate.

A section that only the seed adds (new PR content with no stock text) is declared in
`target.yaml` but not marked in `template.prompt`. Its stock text is empty, and the seed
file supplies the slot and the text.

A marked-but-undeclared key makes `import glean_gepa` fail for every test and run. Run
step 7 after each edit.

### 7. Check against the scio file

```bash
uv run python .cursor/skills/new-prompt-target/scripts/prompt_target_check.py \
  check <name> --source .tmp_debug/<name>/master.prompt
```

Fix until it prints `PASS`, then resolve each `WARN`: undeclared placeholders, missing
frames, tight budgets. Leave a warning only if you can explain it to the user.

### 8. Seed

- **Stock seed:** omit `run.seed_candidate`.
- **PR seed:** copy `pr<N>.prompt` to `data/<name>_<purpose>_seed.prompt` and add the
  same marker lines as `template.prompt`. Wrap PR-only content in a marker for its
  declared section.
- **Small override:** a JSON object of keys.

Then check it the way the run will build it:

```bash
uv run python .cursor/skills/new-prompt-target/scripts/prompt_target_check.py \
  check <name> --source .tmp_debug/<name>/master.prompt \
  --seed data/<seed file> --editable KEY_A,KEY_B
```

For a `.prompt` seed, it must say the seed compiles to the seed file without its markers.
Note the suggested `search.global_token_cap`.

### 9. Experiment YAML

- Copy the closest shipped config: `teacher_student.yaml` for the coding harness,
  `teacher_student_waldo.yaml` for Waldo. Save it as `configs/<mode>_<short name>.yaml`
  with a fresh `run.dir`. Never repoint a shipped config.
- Set `reflection.editable_modules`, `run.seed_candidate`, and models on the target's
  harness. Set `search.global_token_cap` at or above the checker's suggestion.
- Choose the objective:
  - If an existing one already measures the behavior, keep it. Per-section briefs go
    under `reflection.modules.<KEY>`.
  - Otherwise follow new-objective from its step 2. Step 1 here already took its baseline.
- Open the file with a comment covering what is optimized, the seed's source, the gate,
  whether it has run yet, and the run command.
- Add a row to the "Shipped experiments" table in the architecture guide.

### 10. Tests

`tests/test_prompt_targets.py::test_stock_compile_is_the_template_file_without_markers`
already covers every registered target. Add tests to that file for:

- [ ] A `.prompt` seed compiles to its text without markers, and its override decodes to it.
- [ ] `render_requirements(editable)` and `_eval_harness_for(None, editable, models)` carry
      the render scParams (only when the target has `render:`).
- [ ] The new config loads, and its editable modules parse.

Never read the scio checkout from a test.

### 11. Verify

```bash
bash .cursor/skills/new-objective/scripts/check_objective.sh verify
```

It must print `PASS`, with the new config as a `NEW` line. Rerun the step 7 check too.

### 12. Report

- **Lead with the outcome:** the target, its sections, and that stock compiles
  byte-identical to scio (name the commit). If seeded from a PR, say the seed compiles to
  the PR text.
- **Then:** the files added, the run command, and caveats. Caveats include any gate or
  render scParams, copies this override doesn't reach, and anything unverified.
- **Before a long run,** recommend checking that one student trajectory's system prompt
  contains a sentence unique to the seed.
- Don't commit unless asked.

## Additional resources

- [reference.md](reference.md): fetching scio text, scio override and render facts,
  gating, harnesses, splitting guidance, and gotchas.
- `scripts/prompt_target_check.py`: `scaffold` and `check`. Run it; don't read it.
