# New prompt target: reference

## Contents

- Fetching scio text
- Scio facts
- Harnesses
- Splitting into sections
- Gotchas

## Fetching scio text

The local checkout at `~/workspace/scio` may be stale. Compare it with GitHub first:

```bash
git -C ~/workspace/scio log -1 --format='%h %cd'
gh api repos/askscio/scio/commits/master --jq '.sha[:11] + " " + .commit.committer.date'
```

Run every `gh api` call that writes into the workspace with `required_permissions: ["all"]`;
under `full_network` the sandbox drops the output.

```bash
mkdir -p .tmp_debug/<name>
# master
gh api "repos/askscio/scio/contents/data/prompts/templates/<name>.prompt" \
  -H "Accept: application/vnd.github.raw" > .tmp_debug/<name>/master.prompt
# a PR's version
head=$(gh api repos/askscio/scio/pulls/<N> --jq .head.sha)
gh api "repos/askscio/scio/contents/<path>?ref=$head" \
  -H "Accept: application/vnd.github.raw" > .tmp_debug/<name>/pr<N>.prompt
# which files the PR touches
gh api repos/askscio/scio/pulls/<N>/files --paginate --jq '.[] | "\(.status) \(.filename)"'
```

A PR link's `#diff-<hex>` anchor is the SHA-256 of the file path. To find the file it
points at:

```bash
gh api repos/askscio/scio/pulls/<N>/files --paginate --jq '.[].filename' |
  while read -r f; do printf '%s %s\n' "$(printf '%s' "$f" | shasum -a 256 | cut -c1-64)" "$f"; done |
  rg '^<hex>'
```

## Scio facts

**Override plumbing.**

- `llmo.per_prompt_overrides.<name>=<url-safe base64>` sets the field `<name>` of
  `message PromptOverrides` in `com/askscio/proto/sc/llm_options.proto`.
- `get_structured_prompt_overrides` (`python_scio/agents/core/core.py`) maps each field
  that is set to its decoded text.
- `Prompt._maybe_get_prompt_override` then swaps that text in for the template named
  `<name>`.
- A template with no proto field cannot be overridden this way; it needs a scio change first.
- Tool descriptions go through `co.pyagents_tool_description_overrides` instead (see
  `prompts/core_tool_descriptions/`).

**Render rules** (`Prompt._render` in `core.py`). The override replaces the whole template
text, and then:

- Each `<<<...>>>` block is dropped when any `[[placeholder]]` inside it is not passed by
  the caller. That is how `<<<[[has_search_tools]] ...>>>` branches work.
- A `[[placeholder]]` that the caller does not pass raises `MissingArgumentsError`, so a
  rewrite must never invent placeholders.
- An argument the caller passes that no longer appears in the text is silently dropped.
  Scio's default search config sets `disable_prompt_extra_args_validation = true`; only
  with it off does this raise `TooManyArgumentsError`. Either way the rendered prompt loses
  what the placeholder carried, such as the user's name or the tool names. That is why every
  fill belongs in its module's `required_placeholders`.
- Some callers adapt to an override. For example, Waldo passes `[[waldo_discover_tool_name]]`
  only when the override contains it, and otherwise appends a note naming the handoff tool.
  Read the call site before declaring a placeholder required.

**Finding the render path and its gates.**

```bash
cd ~/workspace/scio
rg -n "string <name> = " com/askscio/proto/sc/llm_options.proto
rg -n "['\"]<name>['\"]" python_scio go --glob '!**/*_test.*'
rg -n "<flag name>" python_scio/agents data/sc/search_configuration.ini
```

Read the call site to learn three things:

- **The arguments it passes.** These are the placeholders the target must keep.
- **The conditions around it.** These are search-config flags (`co.*`) and feature checks.
  For example, a stripped variant renders only with `co.lo.cao.use_stripped_prompts`.
- **Whether a sibling template renders instead.** For example, `stripped_<name>` vs `<name>`.

Then grep the harness presets for those flags: `src/glean_gepa/coding_harness_params.py` and
`waldo_harness_params.py`.

- **Off by default in the scio defaults:** a run must turn the flag on through `render.sc_params`.
- **Turned off by the preset:** list the preset entry in `render.drop_sc_params` and the
  replacement in `render.sc_params`.
- **Where the params land:** they apply to every eval in the run, teacher included, and
  key the eval cache.
- **Top-level vs nested:** top-level scParams reach the agent loop. Entries nested inside
  `agentic_loop_sc_params` are forced extras, which win over the loop's own config.

**Copies.** Grep a distinctive sentence across the repo to find duplicates:
`rg -F "<sentence>" ~/workspace/scio --glob '!**/node_modules/**'`. A TypeScript or Go copy
renders on its own path, and this override does not reach it.

## Harnesses

`src/glean_gepa/harnesses.py`:

- **`coding`:** the Coding Harness agent loop, using `CODING_HARNESS_SC_PARAMS`. Models are
  plain aliases (`gpt6_luna`, `claude_opus`, ...), and `gleanchat_agent` comes from the model.
- **`waldo`:** the Waldo router. The runner type is `GLEAN_CHAT`, `gleanchat_agent=AUTO`, and
  models are written `waldo:PROVIDER:MODEL[:effort]`.

`target.yaml` `harnesses` lists every harness whose flow renders the prompt. The runner
refuses a student or teacher model on any other harness.

## Splitting into sections

- **One responsibility per section.** A child's primary edit should have one goal. The
  reconcile step only fixes conflicting lines in sibling sections, so a section that mixes
  rule families produces muddled children.
- **Contiguous spans of the scio text.** Headings stay in the template, and each frame says
  "Do not add a heading."
- **Keys.** UPPER_SNAKE, unique across every target, because they are candidate keys.
  Renaming one later orphans cached candidates and eval caches.
- **Template module.**
  - `editable: false` (as in coding) keeps the frame fixed.
  - `editable: true` (as in Waldo) lets reflection reorder or retitle the frame. It then
    needs a `token_budget` and a frame telling it to keep every `{SLOT}` line.
- **Budgets** use `len(text) // 4` tokens.
  - Use at least 1.5x stock for a section expected to grow; the checker warns under 1.25x.
    Shipped sections use 512–1024 for coding paragraphs and 2048 per tool description.
    Waldo's budgets (2560–8192) are 7–12x stock, far looser than this rule.
  - `search.global_token_cap` must cover the pinned template plus the editable budgets.
- **Frames.**
  - Write them as "You are rewriting <what the section owns>." plus format constraints
    (for example "Each line must start with '- '.") and "Do not add a heading."
  - Name sibling sections when the boundary is ambiguous.
  - An objective's `module_responsibilities` or `reflection.modules.<KEY>` replaces the frame.
- **`fill` and `drop_empty_line`.**
  - `fill: verbatim` keeps whitespace around a slot exactly (`WRITING_CODE`); the default
    strips the text and falls back to stock when it is empty.
  - `drop_empty_line: true` removes a slot's line when its fill is empty (`RULES_EXT`).
    An empty-stock slot alone on its line needs it, or the stock compile gains a blank line.
- **`required_conditionals`.** Use it for conditional names that a section must keep even if
  reflection merges blocks (`WALDO_ROUTING`, `WALDO_TOOL_USAGE`). Conditionals that
  `current` already has are guarded anyway.
- **Inline markers.** `Say {#GREETING}hello{/GREETING} twice.` works for a span inside a line.

## Gotchas

- **The registry loads at import.** One malformed folder makes `import glean_gepa` fail
  everywhere: tests, the runner, and the checker. This includes an undeclared marker,
  unknown YAML keys, or stock text missing its declared markup. Never leave a half-edited
  folder in place while someone may start a run.
- **Running processes keep the registry they loaded.** A run started before your edit keeps
  the old text. A restarted run picks up the new text, so any candidate that carries the
  edited target gets new prompt hashes.
- **Section text is in candidates.** Changing a section's stock text changes the seed of
  every config that edits it. Verify shows that as `CHANGED`.
- **The registry order is folder-name order, and so is the scParam order.** A new target
  emits nothing unless a candidate carries one of its keys (or it sets `always_emit`), so
  adding a folder leaves other configs' scParams unchanged.
- **Undeclared fills in shipped targets.** The checker warns that `waldo_system` has
  undeclared `[[...]]` fills. That gap predates the framework. Don't copy it into new
  targets, and don't fix it in a shipped target without the user's approval, because it
  changes that config's reflection prompts.
- **The coding prompt is a pinned snapshot.** `coding_agent_loop_system` stock text is the
  snapshot glean_gepa has always sent, not scio master. `check --source` against master
  shows the drift (newer `<<<[[...]]>>>` blocks, a leading blank line). Re-syncing it changes
  every coding config's prompt, so do it only as its own change, with the user's approval.
- **Tool descriptions.** Use a `kind: tool_descriptions` target; there is no template
  compile, and each section is one tool's description.
