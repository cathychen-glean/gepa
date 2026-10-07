"""Helpers for check_objective.sh. Run through `uv run python` from the repo root.

fingerprint --out FILE       What every configs/*.yaml builds: objective class, score keys,
                             params, reflection briefs, and a hash of the seed prompt it sends.
compare-configs BASE CUR     Exit 1 if a config present in BASE now builds or sends something else.
compare-pyright BASE CUR     Exit 1 if CUR (pyright --outputjson) has errors BASE does not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _fingerprint_config(path: Path) -> dict[str, Any]:
    from glean_gepa.experiment_config import (
        agentspan_lookback_days,
        experiment_objective_spec,
        load_experiment_config,
        runner_arg_defaults,
    )
    from glean_gepa.objectives.registry import build_objective
    from glean_gepa.prompt import compile_encoded_prompt
    from glean_gepa.prompt_constants import WRITING_CODE_KEY
    from glean_gepa.runner import _load_seed_candidate, _parse_editable_modules, _seed_for_editable_modules

    config = load_experiment_config(path)
    objective = build_objective(
        config.mode,
        config.signals,
        bigquery_client=MagicMock(),
        lookback_days=agentspan_lookback_days(config) or 1,
        experiment=experiment_objective_spec(config),
    )
    defaults = runner_arg_defaults(config)
    modules = _parse_editable_modules(defaults.get("editable_modules") or WRITING_CODE_KEY)
    seed_path = defaults.get("seed_candidate")
    seed_prompt = None
    if seed_path:
        seed = _seed_for_editable_modules(_load_seed_candidate(Path(seed_path)), modules)
        seed_prompt = _sha(compile_encoded_prompt(seed))
    cls = type(objective)
    return {
        "mode": config.mode,
        "objective_class": f"{cls.__module__}:{cls.__qualname__}",
        "name": objective.name,
        "telemetry_dimensions": list(objective.telemetry_dimensions),
        "params": json.loads(json.dumps(getattr(objective, "params", {}), default=str)),
        "focused_bucket_type": objective.focused_bucket_type,
        "failure_label": objective.failure_label,
        "primary": config.primary_objective,
        "composite": config.objective.get("composite"),
        "frontier_type": config.frontier_type,
        "screening": config.screening,
        "signals": [{k: v for k, v in s.items() if k in ("name", "source", "type", "kind")} for s in config.signals],
        "editable_modules": modules,
        "reflection_briefs": {m: _sha(objective.reflection_prompt(m)) for m in modules},
        "seed_prompt": seed_prompt,
    }


def fingerprint(out: Path) -> int:
    from glean_gepa.experiment_config import CONFIGS_DIR

    result: dict[str, Any] = {}
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        try:
            result[path.stem] = _fingerprint_config(path)
        # SystemExit: runner helpers abort the process on a bad seed; that is a finding here.
        except (Exception, SystemExit) as exc:
            result[path.stem] = {"error": f"{type(exc).__name__}: {exc}"}
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    errors = [name for name, fp in result.items() if "error" in fp]
    print(f"fingerprinted {len(result)} configs -> {out}" + (f" (errors: {', '.join(errors)})" if errors else ""))
    return 0


def compare_configs(base_path: Path, current_path: Path) -> int:
    base = json.loads(base_path.read_text())
    current = json.loads(current_path.read_text())
    changed = 0
    for name in sorted(base):
        if name not in current:
            print(f"REMOVED  {name}")
            changed += 1
            continue
        diffs = sorted(k for k in set(base[name]) | set(current[name]) if base[name].get(k) != current[name].get(k))
        if diffs:
            changed += 1
            print(f"CHANGED  {name}")
            for key in diffs:
                print(f"    {key}: {json.dumps(base[name].get(key))} -> {json.dumps(current[name].get(key))}")
    for name in sorted(set(current) - set(base)):
        status = "ERROR" if "error" in current[name] else "new"
        detail = current[name].get("error") or current[name].get("objective_class")
        print(f"NEW      {name} ({status}: {detail})")
        if status == "ERROR":
            changed += 1
    if not changed:
        print(f"all {len(base)} baseline configs build the same objective and send the same seed prompt")
    return 1 if changed else 0


_SRC_PATH = re.compile(r"((?:src|tests)/.*)$")


def _pyright_errors(path: Path) -> set[str]:
    raw = path.read_text().strip()
    if not raw or "{" not in raw:
        raise SystemExit(f"pyright output is not JSON: {path}")
    data = json.loads(raw[raw.index("{") :])
    errors = set()
    for diag in data.get("generalDiagnostics", []):
        if diag.get("severity") != "error":
            continue
        match = _SRC_PATH.search(diag.get("file", ""))
        where = match.group(1) if match else diag.get("file", "")
        errors.add(f"{where} [{diag.get('rule', '')}] {diag.get('message', '').splitlines()[0]}")
    return errors


def compare_pyright(base_path: Path, current_path: Path) -> int:
    base = _pyright_errors(base_path)
    current = _pyright_errors(current_path)
    new = sorted(current - base)
    for line in new:
        print(f"NEW ERROR  {line}")
    print(f"pyright: {len(current)} errors now, {len(base)} in baseline, {len(new)} new")
    return 1 if new else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    fp = sub.add_parser("fingerprint")
    fp.add_argument("--out", type=Path, required=True)
    for cmd in ("compare-configs", "compare-pyright"):
        p = sub.add_parser(cmd)
        p.add_argument("base", type=Path)
        p.add_argument("current", type=Path)
    args = parser.parse_args()
    if args.cmd == "fingerprint":
        return fingerprint(args.out)
    if args.cmd == "compare-configs":
        return compare_configs(args.base, args.current)
    return compare_pyright(args.base, args.current)


if __name__ == "__main__":
    sys.exit(main())
