#!/usr/bin/env bash
# Usage (from anywhere in the repo; uv needs network, so request full_network in the sandbox):
#   check_objective.sh baseline   before editing: snapshot config fingerprints + pyright errors
#   check_objective.sh verify     after editing: tests, ruff, configs unchanged, no new pyright errors
#
# verify without a baseline falls back to a clean HEAD worktree, which cannot see configs
# that are untracked at HEAD; take the baseline first whenever possible.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="$SCRIPT_DIR/objective_check.py"
ROOT="$(git rev-parse --show-toplevel)"
STATE="$ROOT/.tmp_debug/objective_check"
cd "$ROOT" || exit 2
mkdir -p "$STATE"

snapshot() {  # $1 = output dir; runs in the current directory's checkout
  mkdir -p "$1"
  uv run --project "$ROOT" python "$HELPER" fingerprint --out "$1/configs.json"
  uv run --project "$ROOT" pyright --outputjson src/ >"$1/pyright.json" 2>/dev/null || true
}

head_baseline() {
  local wt="$STATE/head_worktree"
  echo "No baseline snapshot; building one from a clean HEAD worktree."
  git worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"
  git worktree add -q --detach "$wt" HEAD || return 1
  (cd "$wt" && export PYTHONPATH="$wt/src" && snapshot "$STATE/baseline")
  git worktree remove --force "$wt"
}

case "${1:-}" in
  baseline)
    snapshot "$STATE/baseline"
    echo "Baseline saved to $STATE/baseline"
    ;;
  verify)
    [[ -f "$STATE/baseline/configs.json" ]] || head_baseline || { echo "FAIL: could not build a baseline"; exit 2; }
    failed=()

    echo "== pytest (full suite) =="
    uv run pytest -q || failed+=("pytest")

    echo "== ruff =="
    # Staged and unstaged edits, plus untracked tests. `git ls-files -m` misses a staged add.
    changed_tests=$(
      {
        git diff --name-only --diff-filter=ACMR HEAD -- 'tests/*.py'
        git ls-files -o --exclude-standard -- 'tests/*.py'
      } | sort -u
    )
    ruff_args=(src/)
    if [[ -n "$changed_tests" ]]; then
      while IFS= read -r test_file; do
        [[ -n "$test_file" ]] && ruff_args+=("$test_file")
      done <<<"$changed_tests"
    fi
    { uv run ruff check "${ruff_args[@]}" && uv run ruff format --check "${ruff_args[@]}"; } || failed+=("ruff")

    echo "== shipped configs: same objective, same seed prompt =="
    snapshot "$STATE/current"
    uv run python "$HELPER" compare-configs "$STATE/baseline/configs.json" "$STATE/current/configs.json" \
      || failed+=("configs")

    echo "== pyright: no new errors =="
    uv run python "$HELPER" compare-pyright "$STATE/baseline/pyright.json" "$STATE/current/pyright.json" \
      || failed+=("pyright")

    if ((${#failed[@]})); then
      echo "FAIL: ${failed[*]}"
      exit 1
    fi
    echo "PASS"
    ;;
  *)
    echo "usage: $0 baseline|verify" >&2
    exit 2
    ;;
esac
