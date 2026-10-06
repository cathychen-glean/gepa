"""Helpers for wiring a scio prompt into src/glean_gepa/prompts/. Run with `uv run python` from the repo root.

scaffold NAME --source FILE [--harness coding|waldo] [--key KEY]
    Create prompts/NAME/ from a scio template: template.prompt is the file verbatim and
    target.yaml is a skeleton. Prints the scio markup the file uses.

check NAME [--source FILE] [--seed FILE] [--editable K1,K2] [--scio DIR]
    Load the target through the registry, prove stock text compiles back to the scio file,
    audit each module (budget, frame, placeholders), confirm scio can override the template,
    and compile a seed the way a run would. Exits 1 on any FAIL.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from base64 import urlsafe_b64decode
from pathlib import Path

try:
    from glean_gepa.al_adapter import approx_token_len
    from glean_gepa.prompt_targets import (
        PER_PROMPT_OVERRIDE_PREFIX,
        PROMPTS_DIR,
        TARGET_FILE,
        TEMPLATE_FILE,
        PromptTarget,
        PromptTargetError,
        build_seed,
        conditional_counts,
        load_seed_file,
        parse_editable_modules,
        placeholders,
        prompt_targets,
        render_requirements,
    )
except Exception as exc:  # the registry loads at import, so one malformed prompts/ folder fails here
    print(f"FAIL  glean_gepa does not import: {type(exc).__name__}: {exc}")
    sys.exit(1)

_MARKER_LINE = re.compile(r"^\{[#/]\w+\}\n", re.MULTILINE)
_MARKER = re.compile(r"\{[#/]\w+\}")
_CONDITIONAL_BLOCK = re.compile(r"<<<.*?>>>", re.DOTALL)
_DEFAULT_SCIO = Path.home() / "workspace" / "scio"
_PROTO = Path("com/askscio/proto/sc/llm_options.proto")
_TIGHT_BUDGET = 1.25


class Report:
    def __init__(self) -> None:
        self.failed = False

    def ok(self, msg: str) -> None:
        print(f"OK    {msg}")

    def info(self, msg: str) -> None:
        print(f"INFO  {msg}")

    def warn(self, msg: str) -> None:
        print(f"WARN  {msg}")

    def fail(self, msg: str) -> None:
        self.failed = True
        print(f"FAIL  {msg}")


def unmarked(text: str) -> str:
    """The prompt as scio stores it: marker lines and inline markers deleted."""
    return _MARKER.sub("", _MARKER_LINE.sub("", text))


def _first_diff(expected: str, actual: str, *, expected_name: str, actual_name: str) -> str:
    diff = difflib.unified_diff(
        expected.splitlines(keepends=True),
        actual.splitlines(keepends=True),
        fromfile=expected_name,
        tofile=actual_name,
        n=2,
    )
    return "".join(list(diff)[:30])


def _unconditional_placeholders(text: str) -> set[str]:
    return placeholders(_CONDITIONAL_BLOCK.sub("", text))


def scaffold(name: str, source: Path, harness: str, key: str | None) -> int:
    folder = PROMPTS_DIR / name
    if folder.exists():
        print(f"FAIL  {folder} already exists")
        return 1
    text = source.read_text(encoding="utf-8")
    template_key = key or name.upper()
    folder.mkdir(parents=True)
    (folder / TEMPLATE_FILE).write_text(text, encoding="utf-8")
    conditionals = sorted(conditional_counts(text))
    always = sorted(_unconditional_placeholders(text))
    conditional = sorted(placeholders(text) - set(always) - set(conditionals))
    (folder / TARGET_FILE).write_text(
        f"# scio data/prompts/templates/{name}.prompt (copied from {source}).\n"
        "# Wrap each editable span of template.prompt in {#KEY} ... {/KEY} marker lines, then\n"
        "# declare every marked KEY under sections. Each module lists the [[placeholders]] its\n"
        "# stock text contains: scio silently drops a fill whose placeholder goes missing.\n"
        f"#   placeholders outside conditionals: {', '.join(always) or 'none'}\n"
        f"#   placeholders inside <<< >>> blocks: {', '.join(conditional) or 'none'}\n"
        f"#   conditionals: {', '.join(conditionals) or 'none'}\n"
        f"harnesses: [{harness}]\n"
        "\n"
        "template:\n"
        f"  key: {template_key}\n"
        "  editable: false\n"
        "\n"
        "sections: {}\n",
        encoding="utf-8",
    )
    print(f"OK    created {folder}/{TEMPLATE_FILE} ({approx_token_len(text)} tokens) and {TARGET_FILE}")
    print(f"INFO  placeholders outside conditionals: {', '.join(always) or 'none'}")
    print(f"INFO  placeholders inside <<< >>> blocks: {', '.join(conditional) or 'none'}")
    print(f"INFO  conditionals: {', '.join(conditionals) or 'none'}")
    print("NEXT  add markers, declare sections, then run: check", name, f"--source {source}")
    return 0


def _check_scio(name: str, target: PromptTarget, scio: Path, report: Report) -> None:
    if not target.override_param.startswith(PER_PROMPT_OVERRIDE_PREFIX):
        report.info(f"override param {target.override_param}: not a per-prompt override, scio check skipped")
        return
    proto = scio / _PROTO
    if not proto.is_file():
        report.warn(f"no scio checkout at {scio}; confirm `string {name} = N;` is in message PromptOverrides")
        return
    body = proto.read_text(encoding="utf-8")
    block = re.search(r"message PromptOverrides \{(.*?)\n\}", body, re.DOTALL)
    if block and re.search(rf"\bstring {re.escape(name)} = \d+;", block.group(1)):
        report.ok(f"{name} is a field of PromptOverrides ({_PROTO})")
    else:
        report.fail(
            f"{name} is not a field of message PromptOverrides in {_PROTO}: scio drops "
            f"{target.override_param}. It needs a scio change first."
        )
    template_file = scio / "data" / "prompts" / "templates" / f"{name}.prompt"
    if template_file.is_file():
        report.ok(f"scio template exists: {template_file.relative_to(scio)}")
    else:
        report.warn(f"no {template_file.relative_to(scio)} in the scio checkout (stale checkout or a PR-only prompt?)")


def _audit_modules(target: PromptTarget, report: Report) -> None:
    findings: list[tuple[str, str]] = []
    print(f"\n{'module':28} {'role':8} {'edit':5} {'stock':>6} {'budget':>7}")
    for key, module in target.modules.items():
        tokens = approx_token_len(module.stock) if module.stock else 0
        budget = module.token_budget
        if module.editable and budget is None and module.role == "section":
            findings.append(("warn", f"{key}: editable section without token_budget"))
        if budget is not None and tokens > budget:
            findings.append(("fail", f"{key}: stock is {tokens} tokens, over its {budget} budget"))
        elif budget is not None and module.stock and tokens * _TIGHT_BUDGET > budget:
            findings.append(("warn", f"{key}: budget {budget} leaves under 25% headroom over {tokens} stock tokens"))
        if module.editable and module.role == "section" and not module.frame:
            findings.append(
                ("warn", f"{key}: editable section has no frame; the reflector sees only the generic brief")
            )
        # Conditional names (<<<[[name]]) are guarded by drops_conditional; plain fills are not.
        fills = placeholders(module.stock) - set(conditional_counts(module.stock))
        undeclared = sorted(fills - set(module.required_placeholders))
        if undeclared and module.editable:
            listed = ", ".join(f"[[{name}]]" for name in undeclared)
            findings.append(
                (
                    "warn",
                    f"{key}: {listed} not in required_placeholders. Nothing stops a rewrite dropping one, and "
                    "scio then silently renders without that value",
                )
            )
        print(
            f"{key:28} {module.role:8} {'yes' if module.editable else 'no':5} {tokens:>6} "
            f"{budget if budget is not None else '-':>7}"
        )
    print()
    for level, message in findings:
        (report.fail if level == "fail" else report.warn)(message)


def _check_seed(target: PromptTarget, seed_path: Path, editable_arg: str | None, report: Report) -> None:
    try:
        raw = load_seed_file(seed_path)
        editable = (
            parse_editable_modules([part.strip() for part in editable_arg.split(",") if part.strip()])
            if editable_arg
            else [key for key in target.section_keys if target.modules[key].editable]
        )
        candidate = build_seed(raw, editable)
    except PromptTargetError as exc:
        report.fail(f"seed: {exc}")
        return
    report.ok(f"seed builds for editable_modules {editable}")
    if target.kind != "template":
        return
    compiled = target.compile_text(candidate)
    if seed_path.suffix == ".prompt":
        expected = unmarked(seed_path.read_text(encoding="utf-8"))
        if compiled == expected:
            report.ok("seed compiles to the seed file without its markers")
        else:
            report.fail("seed compiles to different text than the seed file without markers:")
            print(_first_diff(expected, compiled, expected_name="seed file", actual_name="compiled"))
    fragment = target.override(candidate)
    decoded = urlsafe_b64decode(fragment.split("=", 1)[1]).decode("utf-8") if fragment else ""
    if decoded == compiled:
        report.ok(f"{target.override_param} decodes to the compiled seed ({approx_token_len(compiled)} tokens)")
    else:
        report.fail(f"{target.override_param} does not decode to the compiled seed")
    owned = [key for key in candidate if key in target.modules]
    frozen = sum(approx_token_len(candidate[key]) for key in owned if key not in editable)
    budgets = sum(target.modules[key].token_budget or approx_token_len(candidate[key]) for key in editable)
    report.info(
        f"seed modules {owned}: {sum(approx_token_len(candidate[k]) for k in owned)} tokens now; "
        f"set search.global_token_cap >= {frozen + budgets} (pinned text + editable budgets)"
    )
    extra, dropped = render_requirements(editable)
    if extra or dropped:
        report.info(f"every eval gets render sc_params {list(extra)} and drops {list(dropped)}")


def check(name: str, source: Path | None, seed: Path | None, editable: str | None, scio: Path) -> int:
    report = Report()
    try:
        target = prompt_targets().get(name)
    except PromptTargetError as exc:
        print(f"FAIL  registry: {exc}")
        return 1
    if target is None:
        print(f"FAIL  no target {name!r}; known: {', '.join(prompt_targets())}")
        return 1
    report.ok(
        f"{name} loads: kind={target.kind}, harnesses={list(target.harnesses)}, "
        f"override={target.override_param}, sections={list(target.section_keys)}"
    )
    if target.render_sc_params or target.drop_sc_params:
        report.info(f"render sc_params {list(target.render_sc_params)}, drop {list(target.drop_sc_params)}")
    _check_scio(name, target, scio, report)

    if target.kind == "template":
        stock = target.compile_text({})
        reference = source.read_text(encoding="utf-8") if source else None
        file_text = unmarked((PROMPTS_DIR / name / TEMPLATE_FILE).read_text(encoding="utf-8"))
        expected, label = (reference, str(source)) if reference is not None else (file_text, "template unmarked")
        if stock == expected:
            report.ok(f"stock compile is byte-identical to {label}")
        else:
            report.fail(f"stock compile differs from {label}:")
            print(_first_diff(expected, stock, expected_name=label, actual_name="stock compile"))
            report.info("an empty-stock {KEY} slot alone on its line needs drop_empty_line: true")
        if reference is None:
            report.warn("no --source given; compared against template.prompt, not the scio file")

    _audit_modules(target, report)
    if seed is not None:
        _check_seed(target, seed, editable, report)
    print("FAIL" if report.failed else "PASS")
    return 1 if report.failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("scaffold")
    sc.add_argument("name")
    sc.add_argument("--source", type=Path, required=True)
    sc.add_argument("--harness", choices=("coding", "waldo"), default="coding")
    sc.add_argument("--key")
    ck = sub.add_parser("check")
    ck.add_argument("name")
    ck.add_argument("--source", type=Path)
    ck.add_argument("--seed", type=Path)
    ck.add_argument("--editable")
    ck.add_argument("--scio", type=Path, default=_DEFAULT_SCIO)
    args = parser.parse_args()
    if args.cmd == "scaffold":
        return scaffold(args.name, args.source, args.harness, args.key)
    return check(args.name, args.source, args.seed, args.editable, args.scio)


if __name__ == "__main__":
    sys.exit(main())
