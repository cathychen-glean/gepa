"""Subprocess wrapper for Glean's evalcli."""

from __future__ import annotations

import json
import math
import os
import ssl
import subprocess
import tempfile
import time
from typing import Any

from glean_gepa.debug import debug_print

CODING_HARNESS_PRESET = "Coding Harness"
DEFAULT_PRESET = CODING_HARNESS_PRESET

CORRECTNESS_INPUT_MAPPINGS = json.dumps(
    [
        {"entryType": "TEST", "name": "Query", "path": "Query", "sourceType": "EVAL_SET_PROTO"},
        {
            "entryType": "TEST",
            "name": "Response",
            "path": "EvalChatResponseInfo.ActResponse",
            "sourceType": "EVAL_RUN_OUTPUT",
        },
        {
            "entryType": "BASE",
            "name": "CanonicalAnswer",
            "path": "EvalChatResponseInfo.ActResponse",
            "sourceType": "EVAL_RUN_OUTPUT",
        },
    ]
)

CORRECTNESS_JUDGE_TYPE = "CORRECTNESS"
CORRECTNESS_RUN_PARAMS = json.dumps({"Judge Type": "DIRECT_CORRECTNESS", "Llm model": "default", "Use Cache": "true"})
COMPLETENESS_JUDGE_TYPE = "COMPLETENESS"
COMPLETENESS_RUN_PARAMS = json.dumps({"Llm model": "default", "Use Cache": "true"})

TERMINAL_JUDGE_STATUSES = {"SUCCEEDED", "FAILED", "CANCELLED"}

TERMINAL_TASK_STATUSES = {
    "TASK_SUCCEEDED",
    "TASK_FAILED",
    "TASK_DEPENDENCY_FAILED",
    "TASK_CANCELLED",
    "TASK_TIMED_OUT",
    "TASK_ABORTED",
}

TRANSIENT_EVALCLI_PATTERNS = (
    "API request failed: 502",
    "API request failed: 503",
    "API request failed: 504",
    "Connection refused",
    "Connection reset",
    "__Host-GCP_IAP_AUTH_TOKEN_",
)
# evalcli sometimes exits 1 with a bare "Error:" and empty stdout.
OPAQUE_EVALCLI_ERROR_MARKER = "stderr: Error:\nstdout:"
JUDGE_CREATE_ATTEMPTS = 4
JUDGE_CREATE_RETRY_SEC = 30

MIN_INGESTED_ENTRY_FRACTION = 0.5


class EvalCliError(RuntimeError):
    pass


def min_ingested_eval_set_entries(expected_count: int, *, min_fraction: float = MIN_INGESTED_ENTRY_FRACTION) -> int:
    """Return the fewest ingested rows that still count as a usable eval set."""
    if expected_count <= 0:
        return 1
    return max(1, math.ceil(expected_count * min_fraction))


def _is_unreliable_ca_bundle(path: str) -> bool:
    # Cursor sandbox and some proxies inject ephemeral temp CA files that break child Python tools.
    normalized = path.lower()
    return "/var/folders/" in path or normalized.startswith("/tmp/") or "socketfirewallca.crt" in normalized


def _resolve_ca_bundle() -> str | None:
    candidates = [
        "/etc/ssl/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
    ]

    defaults = ssl.get_default_verify_paths()
    if defaults.openssl_cafile and os.path.isfile(defaults.openssl_cafile):
        candidates.append(defaults.openssl_cafile)
    if defaults.cafile and os.path.isfile(defaults.cafile) and not _is_unreliable_ca_bundle(defaults.cafile):
        candidates.append(defaults.cafile)

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate

    try:
        import certifi

        return certifi.where()
    except ImportError:
        return None


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    ca_bundle = _resolve_ca_bundle()
    current = env.get("SSL_CERT_FILE")
    if ca_bundle and (not current or _is_unreliable_ca_bundle(current)):
        env["SSL_CERT_FILE"] = ca_bundle
        env["REQUESTS_CA_BUNDLE"] = ca_bundle
        env.pop("SSL_CERT_DIR", None)
    return env


def _is_transient_evalcli_error(exc: EvalCliError) -> bool:
    message = str(exc)
    if any(pattern in message for pattern in TRANSIENT_EVALCLI_PATTERNS):
        return True
    return OPAQUE_EVALCLI_ERROR_MARKER in message


def classify_eval_run_status(status: Any) -> str:
    """Classify a Cortex run-status payload as ongoing, usable, or missing."""
    if not isinstance(status, dict):
        return "missing"
    task_counts = status.get("taskCountsByStatus") or []
    if not isinstance(task_counts, list):
        return "missing"
    active_counts = [entry for entry in task_counts if isinstance(entry, dict) and (entry.get("count") or 0) > 0]
    if not active_counts:
        return "missing"
    if any(entry.get("status") not in TERMINAL_TASK_STATUSES for entry in active_counts):
        return "ongoing"
    return "usable"


def is_missing_eval_job(exc: EvalCliError) -> bool:
    message = str(exc).lower()
    return "no job found" in message or "not found" in message


class EvalCliClient:
    """Thin wrapper around the evalcli binary for eval runs, judges, and analysis."""

    def __init__(self, binary: str | None = None):
        self.binary = binary or os.environ.get("EVALCLI_BIN", "evalcli")

    def _invoke(self, *args: str, expect_json: bool = False) -> Any:
        cmd = [self.binary, *args]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=_subprocess_env())
        if proc.returncode != 0:
            raise EvalCliError(
                f"evalcli failed (exit {proc.returncode}): {' '.join(cmd)}\n"
                f"stderr: {proc.stderr.strip()}\n"
                f"stdout: {proc.stdout.strip()}"
            )
        stdout = proc.stdout.strip()
        if expect_json:
            return json.loads(stdout) if stdout else None
        return stdout

    def _invoke_json(self, *args: str) -> Any:
        if "--json" not in args:
            args = (*args, "--json")
        return self._invoke(*args, expect_json=True)

    def create_eval_run(
        self,
        *,
        eval_run_id: str,
        eval_set_name: str,
        eval_set_version: str,
        deployment_ids: list[str],
        description: str,
        sc_params: str | None = None,
        eval_params: str | None = None,
        preset: str = DEFAULT_PRESET,
    ) -> str:
        eval_set = f"{eval_set_name}:{eval_set_version}"
        cmd = [
            "run",
            "create",
            "--eval-set",
            eval_set,
            "--preset",
            preset,
            "--deployment-ids",
            *deployment_ids,
            "--id",
            eval_run_id,
            "--description",
            description,
        ]
        if sc_params:
            cmd.extend(["--sc-params", sc_params])
        if eval_params:
            cmd.extend(["--eval-params", eval_params])

        result = self._invoke_json(*cmd)
        if not isinstance(result, dict):
            raise EvalCliError(f"Unexpected eval run create response: {result!r}")
        return str(result.get("id") or eval_run_id)

    def wait_for_eval_run(
        self,
        eval_run_id: str,
        *,
        poll_interval_sec: int = 60,
        timeout_sec: int | None = None,
    ) -> list[Any] | None:
        print(f"Waiting for eval run {eval_run_id} to complete...")
        started_at = time.monotonic()
        while True:
            if timeout_sec is not None and time.monotonic() - started_at >= timeout_sec:
                raise EvalCliError(f"Eval run {eval_run_id} timed out after {timeout_sec}s")
            try:
                statuses = self._invoke_json("run", "status", "--id", eval_run_id)
            except EvalCliError as exc:
                if not _is_transient_evalcli_error(exc):
                    raise
                if "__Host-GCP_IAP_AUTH_TOKEN_" in str(exc):
                    print(
                        f"IAP cookie missing while polling {eval_run_id}. "
                        "Open https://dev.glean.com/internal/evaltool/runs in Chrome, then waiting to retry..."
                    )
                else:
                    print(f"Transient Cortex error while polling {eval_run_id}; retrying in {poll_interval_sec}s...")
                time.sleep(poll_interval_sec)
                continue

            print(f"Eval run {eval_run_id} status: {json.dumps(statuses, sort_keys=True, default=str)}")
            if isinstance(statuses, list) and statuses and classify_eval_run_status(statuses[0]) == "usable":
                print(f"Eval run {eval_run_id} completed successfully")
                return statuses
            if isinstance(statuses, list) and statuses:
                counts = statuses[0].get("taskCountsByStatus") or []
                summary = ", ".join(
                    f"{entry.get('status')}={entry.get('count')}" for entry in counts if (entry.get("count") or 0) > 0
                )
                elapsed = int(time.monotonic() - started_at)
                print(f"Eval run {eval_run_id} still running after {elapsed}s" + (f" ({summary})" if summary else ""))

            time.sleep(poll_interval_sec)

    def get_eval_run_status(self, eval_run_id: str) -> list[Any] | None:
        """Return Cortex task status for an eval run, or None if the job is gone."""
        try:
            statuses = self._invoke_json("run", "status", "--id", eval_run_id)
        except EvalCliError as exc:
            if is_missing_eval_job(exc):
                return None
            raise
        if isinstance(statuses, list):
            return statuses
        if isinstance(statuses, dict):
            return [statuses]
        return None

    def get_eval_set_version(self, *, eval_set_name: str, eval_set_version: str) -> dict[str, Any] | None:
        """Return the eval set version, or None when it does not exist yet."""
        try:
            result = self._invoke_json(
                "evalsets",
                "get",
                "--name",
                eval_set_name,
                "--version",
                eval_set_version,
            )
        except EvalCliError:
            return None
        return result if isinstance(result, dict) else None

    def list_eval_set_versions(
        self,
        *,
        eval_set_name: str,
        deployment_ids: list[str],
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        """List versions of an eval set, optionally filtered to the given deployments."""
        rows: list[dict[str, Any]] = []
        page = 1
        while True:
            result = self._invoke_json(
                "evalsets",
                "versions",
                "--name",
                eval_set_name,
                "--page",
                str(page),
                "--page-size",
                str(page_size),
            )
            if isinstance(result, list):
                batch = result
            elif isinstance(result, dict):
                batch = (
                    result.get("evalSetVersions")
                    or result.get("versions")
                    or result.get("evalSets")
                    or result.get("items")
                    or []
                )
            else:
                raise EvalCliError(f"Unexpected eval set versions response: {result!r}")
            if not isinstance(batch, list) or not all(isinstance(row, dict) for row in batch):
                raise EvalCliError(f"Unexpected eval set versions response: {result!r}")
            rows.extend(batch)
            total_pages = 1
            if isinstance(result, dict):
                total_pages = (result.get("pageInfo") or {}).get("totalPages") or 1
            if not batch or page >= total_pages:
                break
            page += 1
        if not deployment_ids:
            return rows
        wanted = set(deployment_ids)
        filtered: list[dict[str, Any]] = []
        for row in rows:
            available = row.get("availableDeploymentIds") or row.get("available_deployment_ids")
            if not isinstance(available, list) or not available:
                filtered.append(row)
                continue
            if wanted.intersection(available):
                filtered.append(row)
        return filtered

    def list_eval_set_entries(
        self,
        *,
        eval_set_name: str,
        eval_set_version: str,
        deployment_ids: list[str],
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch every entry of an eval set version, paging until exhausted."""
        entries: list[dict[str, Any]] = []
        page = 1
        while True:
            result = self._invoke_json(
                "evalsets",
                "entries",
                "--name",
                eval_set_name,
                "--version",
                eval_set_version,
                "--deployment-ids",
                *deployment_ids,
                "--page",
                str(page),
                "--page-size",
                str(page_size),
            )
            if not isinstance(result, dict):
                raise EvalCliError(f"Unexpected evalsets entries response: {result!r}")

            batch = result.get("evalSetEntries") or []
            entries.extend(entry for entry in batch if isinstance(entry, dict))

            total_pages = (result.get("pageInfo") or {}).get("totalPages") or 1
            if not batch or page >= total_pages:
                return entries
            page += 1

    def upload_eval_set(self, request: dict[str, Any]) -> None:
        """Upload a new eval set version. Entries are ingested asynchronously."""
        debug_print(f"Uploading eval set payload:\n{json.dumps(request, indent=2)}")
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        try:
            json.dump(request, handle)
            handle.close()
            self._invoke("evalsets", "upload", "--file", handle.name)
        finally:
            os.unlink(handle.name)

    def wait_for_eval_set_entries(
        self,
        *,
        eval_set_name: str,
        eval_set_version: str,
        deployment_ids: list[str],
        expected_count: int,
        poll_interval_sec: int = 15,
        timeout_sec: int = 900,
    ) -> list[dict[str, Any]]:
        """Poll until the uploaded eval set version has ingested enough entries.

        SESSION ingest can skip tokens with no session log. Once the published
        size stops growing, succeed if at least ``min_fraction`` of the uploaded
        rows landed (default 50%).
        """
        min_count = min_ingested_eval_set_entries(expected_count)
        print(
            f"Waiting for eval set {eval_set_name}:{eval_set_version} to ingest "
            f"{expected_count} entries (min {min_count})..."
        )
        elapsed = 0
        entries: list[dict[str, Any]] = []
        previous_count = -1
        while elapsed < timeout_sec:
            try:
                entries = self.list_eval_set_entries(
                    eval_set_name=eval_set_name,
                    eval_set_version=eval_set_version,
                    deployment_ids=deployment_ids,
                )
            except EvalCliError:
                entries = []

            count = len(entries)
            if count >= expected_count:
                print(f"Eval set {eval_set_name}:{eval_set_version} ready with {count} entries")
                return entries
            if count >= min_count and count == previous_count:
                print(
                    f"Eval set {eval_set_name}:{eval_set_version} ready with {count}/{expected_count} "
                    "entries (SESSION ingest skipped the rest)"
                )
                return entries
            previous_count = count

            time.sleep(poll_interval_sec)
            elapsed += poll_interval_sec

        raise EvalCliError(
            f"Eval set {eval_set_name}:{eval_set_version} only ingested {len(entries)}/{expected_count} "
            f"entries after {timeout_sec}s (need at least {min_count})"
        )

    def _parse_judge_create_response(self, results: Any) -> str:
        if not results:
            raise EvalCliError("judge create returned empty response")
        judge_run = results[0] if isinstance(results, list) else results
        if not isinstance(judge_run, dict):
            raise EvalCliError(f"Unexpected judge create response: {judge_run!r}")
        judge_run_id = judge_run.get("id")
        if not judge_run_id:
            raise EvalCliError(f"judge create response missing id: {judge_run}")
        return str(judge_run_id)

    def create_judge_run(
        self,
        *,
        eval_run_id: str,
        judge_type: str,
        run_params: str,
        base_eval_run_id: str | None = None,
        input_mappings: str | None = None,
    ) -> str:
        cmd = [
            "judge",
            "create",
            "--eval-run-id",
            eval_run_id,
            "--judge-type",
            judge_type,
            "--run-params",
            run_params,
        ]
        if base_eval_run_id:
            cmd.extend(["--base-eval-run-id", base_eval_run_id])
        if input_mappings:
            cmd.extend(["--input-mappings", input_mappings])
        last_exc: EvalCliError | None = None
        for attempt in range(JUDGE_CREATE_ATTEMPTS):
            try:
                return self._parse_judge_create_response(self._invoke_json(*cmd))
            except EvalCliError as exc:
                last_exc = exc
                if not _is_transient_evalcli_error(exc):
                    raise
                existing = self._find_judge_run_id_after_create_error(eval_run_id, judge_type)
                if existing:
                    print(
                        f"[{judge_type}] Reusing judge run {existing} after create error for {eval_run_id}"
                    )
                    return existing
                if attempt + 1 >= JUDGE_CREATE_ATTEMPTS:
                    break
                print(
                    f"Transient evalcli error creating {judge_type} judge for {eval_run_id}; "
                    f"retrying in {JUDGE_CREATE_RETRY_SEC}s..."
                )
                time.sleep(JUDGE_CREATE_RETRY_SEC)
        assert last_exc is not None
        raise last_exc

    def _find_judge_run_id_after_create_error(self, eval_run_id: str, judge_type: str) -> str | None:
        try:
            return self.find_judge_run_id(eval_run_id, judge_type=judge_type)
        except EvalCliError:
            return None

    def find_judge_run_id(self, eval_run_id: str, *, judge_type: str) -> str | None:
        """Find a judge run via GET /judgeruns?evalRunIds= (evalcli judge list)."""
        wanted_type = judge_type.upper()
        result = self._invoke_json(
            "judge",
            "list",
            "--eval-run-ids",
            eval_run_id,
            "--judge-types",
            wanted_type,
        )
        if isinstance(result, list):
            rows = [row for row in result if isinstance(row, dict)]
        elif isinstance(result, dict):
            raw = result.get("judgeRuns") or result.get("runs") or result.get("items") or []
            rows = [row for row in raw if isinstance(row, dict)] if isinstance(raw, list) else []
        else:
            rows = []
        for row in rows:
            config = row.get("config") if isinstance(row.get("config"), dict) else {}
            row_type = config.get("judgeType") or row.get("judgeType") or row.get("judge_type")
            if str(row_type or wanted_type).upper() != wanted_type:
                continue
            judge_run_id = row.get("id")
            if judge_run_id:
                return str(judge_run_id)
        return None

    def get_eval_metrics(self, eval_run_id: str, *, base_eval_id: str | None = None) -> dict[str, Any]:
        """Aggregated judge metrics via POST /metrics/evalruns/pairwise."""
        cmd = ["metrics", "summary", "--test-eval-id", eval_run_id]
        if base_eval_id:
            cmd.extend(["--base-eval-id", base_eval_id])
        result = self._invoke_json(*cmd)
        if not isinstance(result, dict):
            raise EvalCliError(f"Unexpected metrics summary response: {result!r}")
        return result

    def wait_for_judge_run(
        self,
        judge_run_id: str,
        *,
        poll_interval_sec: int = 60,
        timeout_sec: int = 3600,
    ) -> None:
        print(f"Waiting for judge run {judge_run_id} to complete...")
        elapsed = 0
        while elapsed < timeout_sec:
            run = self._invoke_json("judge", "get", "--id", judge_run_id)
            status = run.get("status") if isinstance(run, dict) else None
            if status in TERMINAL_JUDGE_STATUSES:
                if status != "SUCCEEDED":
                    raise EvalCliError(f"Judge run {judge_run_id} ended with status {status}")
                print(f"Judge run {judge_run_id} completed successfully")
                return
            time.sleep(poll_interval_sec)
            elapsed += poll_interval_sec
        raise EvalCliError(f"Judge run {judge_run_id} timed out after {timeout_sec}s")

    def get_analysis_view(self, test_eval_id: str, base_eval_id: str | None = None) -> dict[str, Any]:
        cmd = ["analyze", "view", "--test-run-ids", test_eval_id]
        if base_eval_id:
            cmd.extend(["--base-run-id", base_eval_id])
        result = self._invoke_json(*cmd)
        if not isinstance(result, dict):
            raise EvalCliError(f"Unexpected analysis view response: {result!r}")
        return result

    def get_analysis_details(
        self,
        *,
        entry_ids: list[str],
        eval_run_ids: list[str],
        deployment_id: str,
    ) -> list[dict[str, Any]]:
        if not entry_ids:
            return []
        result = self._invoke_json(
            "analyze",
            "details",
            "--entry-ids",
            *entry_ids,
            "--eval-run-ids",
            *eval_run_ids,
            "--deployment-id",
            deployment_id,
        )
        if not isinstance(result, list):
            raise EvalCliError(f"Unexpected analysis details response: {result!r}")
        return result

    def get_analysis_trace(
        self,
        *,
        deployment_id: str,
        trace_id: str,
        start_time_millis: int,
        end_time_millis: int,
    ) -> dict[str, Any]:
        result = self._invoke_json(
            "analyze",
            "trace",
            "--deployment-id",
            deployment_id,
            "--trace-id",
            trace_id,
            "--start-time-millis",
            str(start_time_millis),
            "--end-time-millis",
            str(end_time_millis),
        )
        if not isinstance(result, dict):
            raise EvalCliError(f"Unexpected analysis trace response: {result!r}")
        return result
