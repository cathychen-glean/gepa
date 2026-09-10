# Glean GEPA

`glean_gepa` optimizes Glean agent prompts using LLM-based reflection and Pareto-efficient
evolutionary search. The search engine itself is not in this repository: it is consumed as the
built **`glean-gepa-core`** distribution, which owns the `gepa` import namespace.

## Setup

We use **uv** for dependency management. The project uses setuptools as the build backend. All
python executions must be done through uv.

```bash
uv sync --extra dev
```

## Project Structure

- `src/glean_gepa/` — the package source
  - `runner.py` — CLI and experiment setup
  - `api.py` — wiring into the `gepa` engine
  - `evolutionary_proposer.py` — parent selection, reflection, child screening
  - `single_model_adapter.py`, `teacher_student_adapter.py` — the two evaluation paths
  - `configs/` — shipped experiment configs
- `packages/glean-gepa-core/` — the pinned GEPA engine subset, imported as `gepa`
- `tests/` — pytest test suite
- `docs/` — `glean_gepa` architecture and deployment guides

## The `gepa` namespace

`from gepa.core.engine import GEPAEngine` resolves to `glean-gepa-core`, a deliberately pruned
snapshot of GEPA: it has no adapters, no `optimize`/`optimize_anything` entry points, no merge
proposer, and no callback or acceptance-criterion plumbing. Only the modules under
`packages/glean-gepa-core/src/gepa/` exist. Never install the public `gepa` package or add a
second `gepa` source tree alongside it; both claim the same import name and would shadow each
other.

## Build & Test

```bash
uv run pytest
uv run ruff check src/
uv run ruff format src/
uv run pyright src/
```

## Code Style

- Linter/formatter: ruff (line length 120, double quotes, space indent)
- Type checking: pyright
- Python target: 3.10+
- No relative imports (enforced by ruff)
