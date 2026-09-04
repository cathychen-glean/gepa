"""Create small eval sets containing only the high-signal entries of a larger eval set.

Cortex has no way to run a subset of an existing eval set, so reproducing a handful of
failing entries requires uploading them as their own eval set version.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from glean_gepa.adapter_types import ALDataInst
from glean_gepa.evalcli_client import EvalCliClient, EvalCliError, min_ingested_eval_set_entries
from glean_gepa.shell_tool_error_util import fetch_evalset_entry_tracking

SESSION_BUCKET_TYPE = "SESSION"
QUERY_CANONICAL_BUCKET_TYPE = "QUERY_CANONICAL"
DEFAULT_BUCKET_TYPE = QUERY_CANONICAL_BUCKET_TYPE
DEFAULT_RUN_LABEL = "gepa"
FOCUSED_EVAL_SET_NAME_PREFIX = "gepa-high-signal"
HIGH_SIGNAL_EVAL_SET_SOURCE_SCHEMA = "fact-agentspan-trace-v5"
HIGH_SIGNAL_RUN_LABEL = "gepa_high_signal"


@dataclass(frozen=True)
class FocusedEvalSet:
    name: str
    version: str
    entry_count: int


@dataclass(frozen=True)
class EvalRunTarget:
    """Eval-set name/version and run label to use for one data inst."""

    eval_set_name: str
    eval_set_version: str
    run_label: str
    is_focused: bool = False


def prepare_high_signal_eval_batch(
    evalcli: EvalCliClient,
    batch: Sequence[ALDataInst],
    *,
    bigquery_client: Any | None = None,
    bucket_type: str = QUERY_CANONICAL_BUCKET_TYPE,
) -> list[ALDataInst] | None:
    """Upload/reuse focused eval sets once, before concurrent child screening.

    Teacher-student uses ``QUERY_CANONICAL`` (fresh query, no session restore).
    Returns None if any focused set cannot be created so a candidate cannot
    silently fall back to the full eval set.
    """
    prepared: list[ALDataInst] = []
    for data in batch:
        entry_ids = data.get("eval_entry_ids") or []
        if not entry_ids:
            prepared.append(data)
            continue
        focused = ensure_focused_eval_set(
            evalcli,
            base_eval_set_name=data["eval_set_name"],
            base_eval_set_version=data["eval_set_version"],
            deployment_ids=list(data.get("deployment_ids") or []),
            entry_ids=entry_ids,
            bucket_type=bucket_type,
            bigquery_client=bigquery_client,
        )
        if focused is None:
            return None
        prepared.append(
            {
                **data,
                "eval_set_name": focused.name,
                "eval_set_version": focused.version,
                "focused_eval_set_name": focused.name,
                "focused_eval_set_version": focused.version,
            }
        )
    return prepared


def resolve_eval_run_target(
    evalcli: EvalCliClient,
    data: Mapping[str, Any],
    *,
    bigquery_client: Any | None = None,
    bucket_type: str = QUERY_CANONICAL_BUCKET_TYPE,
) -> EvalRunTarget | None:
    """Return where to run this data inst. None if focused setup failed."""
    entry_ids = data.get("eval_entry_ids") or []
    eval_set_name = str(data.get("eval_set_name", ""))
    eval_set_version = str(data.get("eval_set_version", ""))
    if not entry_ids:
        return EvalRunTarget(
            eval_set_name=eval_set_name,
            eval_set_version=eval_set_version,
            run_label=DEFAULT_RUN_LABEL,
            is_focused=False,
        )

    focused_name = data.get("focused_eval_set_name")
    focused_version = data.get("focused_eval_set_version")
    if focused_name and focused_version:
        return EvalRunTarget(
            eval_set_name=str(focused_name),
            eval_set_version=str(focused_version),
            run_label=HIGH_SIGNAL_RUN_LABEL,
            is_focused=True,
        )

    focused = ensure_focused_eval_set(
        evalcli,
        base_eval_set_name=eval_set_name,
        base_eval_set_version=eval_set_version,
        deployment_ids=list(data.get("deployment_ids") or []),
        entry_ids=entry_ids,
        bucket_type=bucket_type,
        bigquery_client=bigquery_client,
    )
    if focused is None:
        return None
    return EvalRunTarget(
        eval_set_name=focused.name,
        eval_set_version=focused.version,
        run_label=HIGH_SIGNAL_RUN_LABEL,
        is_focused=True,
    )


def focused_eval_set_name(base_eval_set_name: str) -> str:
    """Return a DSE-compatible lowercase slug for the focused eval set."""
    base_slug = re.sub(r"[^a-z0-9_-]+", "-", base_eval_set_name.lower()).strip("-_")
    return f"{FOCUSED_EVAL_SET_NAME_PREFIX}-{base_slug or 'eval-set'}"


def focused_eval_set_version(
    base_eval_set_version: str,
    entry_ids: Sequence[str],
    *,
    bucket_type: str = DEFAULT_BUCKET_TYPE,
) -> str:
    """Deterministic version so the same entry set is reused instead of re-uploaded.

    SESSION keeps the original fingerprint so single-model reuses focused sets
    uploaded before bucket type was part of the key. QUERY_CANONICAL is prefixed
    so it cannot collide with those SESSION versions.
    """
    if bucket_type == SESSION_BUCKET_TYPE:
        fingerprint = f"{HIGH_SIGNAL_EVAL_SET_SOURCE_SCHEMA}:{','.join(sorted(entry_ids))}"
    else:
        fingerprint = f"{bucket_type}:{HIGH_SIGNAL_EVAL_SET_SOURCE_SCHEMA}:{','.join(sorted(entry_ids))}"
    digest = hashlib.md5(fingerprint.encode()).hexdigest()[:12]
    return f"{base_eval_set_version}_hs_{digest}"


def focused_eval_set_retry_version(version: str) -> str:
    """Return a fresh version when the deterministic focused version is empty."""
    return f"{version}_retry_{uuid.uuid4().hex[:12]}"


def _source_session_token(source_entry: Mapping[str, Any]) -> str | None:
    tracking = source_entry.get("sourceTrackingInfo") or {}
    token = (
        source_entry.get("stt")
        or source_entry.get("session_tracking_token")
        or tracking.get("sessionTrackingToken")
    )
    return str(token) if token else None


def _is_eval_set_already_exists_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "already exists" in message and "eval set" in message


def _upload_focused_eval_set(
    evalcli: EvalCliClient,
    *,
    name: str,
    version: str,
    entries: Sequence[Mapping[str, Any]],
    bucket_type: str,
    base_eval_set_name: str,
    base_eval_set_version: str,
) -> None:
    request = build_upload_eval_set_request(
        name=name,
        version=version,
        entries=entries,
        bucket_type=bucket_type,
        base_eval_set_name=base_eval_set_name,
        base_eval_set_version=base_eval_set_version,
    )
    print(f"[Focused eval set] Uploading {name}:{version} with {len(entries)} entries")
    evalcli.upload_eval_set(request)


def build_upload_entry(
    source_entry: Mapping[str, Any],
    *,
    bucket_type: str = DEFAULT_BUCKET_TYPE,
) -> dict[str, Any] | None:
    """Map a listed EvalSetEntry onto the UploadEvalSetEntry shape."""
    deployment_id = source_entry.get("deploymentId")
    if not deployment_id:
        return None

    tracking = source_entry.get("sourceTrackingInfo") or {}
    source_trace = source_entry.get("sourceTrace") or {}
    entry_input = source_entry.get("input") or {}
    query = source_entry.get("query") or entry_input.get("query")
    user = source_entry.get("user") or source_entry.get("user_id")

    if bucket_type == QUERY_CANONICAL_BUCKET_TYPE:
        # Fresh chat: query text only. Copying stt restores the source session.
        # Restore output is valid as stored. The Anthropic Messages-API adapter
        # then hoists trailing system messages into the top-level system
        # parameter and strips OpenAI-shaped reasoning items, leaving messages
        # ending on an assistant turn (Claude 400). The Responses API keeps
        # system/reasoning/function-call items inline, so the same restore never
        # ends on an assistant turn. Restored assistant text can also leak
        # expectedOutput.
        if not query:
            return None
        entry: dict[str, Any] = {"deploymentId": deployment_id, "query": str(query)}
        if user:
            entry["user"] = user
        return entry

    entry = {"deploymentId": deployment_id}
    optional_fields = {
        "user": user,
        "traceId": (
            source_entry.get("traceId")
            or source_entry.get("trace_id")
            or tracking.get("traceId")
            or source_trace.get("id")
        ),
        "stt": _source_session_token(source_entry),
        "qtt": (
            source_entry.get("qtt")
            or source_entry.get("query_tracking_token")
            or tracking.get("queryTrackingToken")
        ),
        "runId": source_entry.get("runId") or source_entry.get("workflow_run_id") or tracking.get("runId"),
        "query": query,
    }
    entry.update({key: value for key, value in optional_fields.items() if value})
    return entry if any(entry.get(key) for key in ("traceId", "stt", "qtt", "query")) else None


def _enrich_source_entries_with_tracking(
    selected: Sequence[Mapping[str, Any]],
    *,
    bigquery_client: Any | None,
    base_eval_set_name: str,
    base_eval_set_version: str,
    deployment_ids: list[str],
) -> list[dict[str, Any]]:
    enriched = [dict(entry) for entry in selected]
    missing_ids = [
        str(entry.get("id") or "") for entry in enriched if entry.get("id") and not _source_session_token(entry)
    ]
    if not missing_ids:
        return enriched
    if bigquery_client is None:
        print(
            "[Focused eval set] EvalCLI entries are missing session tracking tokens; "
            "pass a BigQuery client to resolve stt from fact.evalset_entries"
        )
        return enriched
    tracking_by_id = fetch_evalset_entry_tracking(
        bigquery_client,
        eval_set_name=base_eval_set_name,
        eval_set_version=base_eval_set_version,
        entry_ids=missing_ids,
        deployment_ids=deployment_ids or None,
    )
    print(
        f"[Focused eval set] Resolved stt for {len(tracking_by_id)}/{len(missing_ids)} entries from BigQuery"
    )
    for entry in enriched:
        extra = tracking_by_id.get(str(entry.get("id") or ""))
        if extra:
            entry.update({key: value for key, value in extra.items() if value})
    return enriched


def build_upload_eval_set_request(
    *,
    name: str,
    version: str,
    entries: Sequence[Mapping[str, Any]],
    bucket_type: str = DEFAULT_BUCKET_TYPE,
    base_eval_set_name: str | None = None,
    base_eval_set_version: str | None = None,
) -> dict[str, Any]:
    request = {
        "name": name,
        "version": version,
        "bucketType": bucket_type,
        "entries": [dict(entry) for entry in entries],
        "useUploadJob": True,
    }
    if base_eval_set_name or base_eval_set_version:
        request["metadata"] = {
            "gepaSourceEvalSetName": base_eval_set_name,
            "gepaSourceEvalSetVersion": base_eval_set_version,
        }
    return request


def _find_ingested_focused_version(
    evalcli: EvalCliClient,
    *,
    name: str,
    version_prefix: str,
    deployment_ids: list[str],
    min_count: int,
) -> FocusedEvalSet | None:
    """Reuse a retry version that already ingested enough entries for this fingerprint."""
    for row in evalcli.list_eval_set_versions(eval_set_name=name, deployment_ids=deployment_ids):
        version = str(row.get("version") or "")
        if version == version_prefix or not version.startswith(version_prefix):
            continue
        entries = evalcli.list_eval_set_entries(
            eval_set_name=name, eval_set_version=version, deployment_ids=deployment_ids
        )
        if len(entries) >= min_count:
            print(f"[Focused eval set] Reusing {name}:{version} with {len(entries)} entries")
            return FocusedEvalSet(name, version, len(entries))
    return None


def ensure_focused_eval_set(
    evalcli: EvalCliClient,
    *,
    base_eval_set_name: str,
    base_eval_set_version: str,
    deployment_ids: list[str],
    entry_ids: Sequence[str],
    source_entries: Sequence[Mapping[str, Any]] | None = None,
    bucket_type: str = DEFAULT_BUCKET_TYPE,
    bigquery_client: Any | None = None,
) -> FocusedEvalSet | None:
    """Create or reuse an eval-set version containing only ``entry_ids``."""
    if not entry_ids:
        return None

    name = focused_eval_set_name(base_eval_set_name)
    version = focused_eval_set_version(base_eval_set_version, entry_ids, bucket_type=bucket_type)
    min_count = min_ingested_eval_set_entries(len(entry_ids))
    existing = evalcli.get_eval_set_version(eval_set_name=name, eval_set_version=version)
    if existing is not None:
        existing_entries = evalcli.list_eval_set_entries(
            eval_set_name=name, eval_set_version=version, deployment_ids=deployment_ids
        )
        if len(existing_entries) >= min_count:
            print(f"[Focused eval set] Reusing {name}:{version} with {len(existing_entries)} entries")
            return FocusedEvalSet(name, version, len(existing_entries))
        reused = _find_ingested_focused_version(
            evalcli,
            name=name,
            version_prefix=version,
            deployment_ids=deployment_ids,
            min_count=min_count,
        )
        if reused is not None:
            return reused
        version = focused_eval_set_retry_version(version)

    if source_entries is None:
        source_entries = evalcli.list_eval_set_entries(
            eval_set_name=base_eval_set_name,
            eval_set_version=base_eval_set_version,
            deployment_ids=deployment_ids,
        )
    wanted = set(entry_ids)
    selected = [entry for entry in source_entries if str(entry.get("id") or "") in wanted]
    if bucket_type == SESSION_BUCKET_TYPE:
        selected = _enrich_source_entries_with_tracking(
            selected,
            bigquery_client=bigquery_client,
            base_eval_set_name=base_eval_set_name,
            base_eval_set_version=base_eval_set_version,
            deployment_ids=deployment_ids,
        )
    upload_entries = [
        entry
        for source in selected
        if (entry := build_upload_entry(source, bucket_type=bucket_type)) is not None
    ]
    if not upload_entries:
        missing = "queries" if bucket_type == QUERY_CANONICAL_BUCKET_TYPE else "session tracking tokens"
        print(
            f"[Focused eval set] None of the {len(entry_ids)} high-signal entries had {missing} "
            f"for {base_eval_set_name}:{base_eval_set_version}"
        )
        return None

    print(
        f"[Focused eval set] {bucket_type} payload "
        f"stt={sum(1 for entry in upload_entries if entry.get('stt'))} "
        f"runId={sum(1 for entry in upload_entries if entry.get('runId'))} "
        f"traceId={sum(1 for entry in upload_entries if entry.get('traceId'))} "
        f"query={sum(1 for entry in upload_entries if entry.get('query'))}"
    )
    try:
        _upload_focused_eval_set(
            evalcli,
            name=name,
            version=version,
            entries=upload_entries,
            bucket_type=bucket_type,
            base_eval_set_name=base_eval_set_name,
            base_eval_set_version=base_eval_set_version,
        )
    except EvalCliError as exc:
        if not _is_eval_set_already_exists_error(exc):
            raise
        # Cortex upload uniqueness is a SQL name:version row. GET/entries can still
        # 404 when that row was reserved by a failed or unpublished prior upload.
        if evalcli.get_eval_set_version(eval_set_name=name, eval_set_version=version) is not None:
            print(f"[Focused eval set] {name}:{version} was created concurrently; reusing it")
        else:
            retry_version = focused_eval_set_retry_version(version)
            print(
                f"[Focused eval set] {name}:{version} already exists in Cortex SQL but is not readable; "
                f"uploading {retry_version} instead"
            )
            version = retry_version
            _upload_focused_eval_set(
                evalcli,
                name=name,
                version=version,
                entries=upload_entries,
                bucket_type=bucket_type,
                base_eval_set_name=base_eval_set_name,
                base_eval_set_version=base_eval_set_version,
            )

    try:
        ingested = evalcli.wait_for_eval_set_entries(
            eval_set_name=name,
            eval_set_version=version,
            deployment_ids=deployment_ids,
            expected_count=len(upload_entries),
        )
    except EvalCliError as exc:
        print(f"[Focused eval set] {exc}")
        return None
    return FocusedEvalSet(name, version, len(ingested))


__all__ = [
    "DEFAULT_BUCKET_TYPE",
    "DEFAULT_RUN_LABEL",
    "EvalRunTarget",
    "FocusedEvalSet",
    "HIGH_SIGNAL_EVAL_SET_SOURCE_SCHEMA",
    "HIGH_SIGNAL_RUN_LABEL",
    "QUERY_CANONICAL_BUCKET_TYPE",
    "SESSION_BUCKET_TYPE",
    "build_upload_entry",
    "build_upload_eval_set_request",
    "ensure_focused_eval_set",
    "focused_eval_set_name",
    "focused_eval_set_version",
    "prepare_high_signal_eval_batch",
    "resolve_eval_run_target",
]
