"""BigQuery access used by the Glean evaluator."""

from __future__ import annotations

import os
import time
from typing import Any, Protocol, Sequence

DEFAULT_BIGQUERY_PROJECT = "scio-apps"
# Bounds each HTTP request, not the query. Without it a TLS connection that dies
# silently — a laptop sleeping mid-run is enough — leaves the reader blocked in
# recv() forever instead of raising.
DEFAULT_REQUEST_TIMEOUT_SEC = 120.0
DEFAULT_QUERY_ATTEMPTS = 3
RETRY_BACKOFF_SEC = 5.0

# Matched by class name so this module does not import google.api_core eagerly.
_RETRYABLE_EXCEPTION_NAMES = frozenset(
    {
        "TransportError",  # google.auth, connection broken mid-refresh
        "ServiceUnavailable",  # 503
        "DeadlineExceeded",  # 504
        "InternalServerError",  # 500
        "TooManyRequests",  # 429
        "RetryError",
    }
)


def _is_retryable(exc: BaseException) -> bool:
    """True for stalled or transient transport failures.

    Reauthentication, permission, and malformed-query errors are permanent, and
    retrying them only delays a message the caller needs to act on.
    """
    # OSError covers TimeoutError, ConnectionError, and requests' RequestException.
    return isinstance(exc, OSError) or type(exc).__name__ in _RETRYABLE_EXCEPTION_NAMES


class BigQueryError(RuntimeError):
    pass


class BigQueryQueryClient(Protocol):
    def query(self, sql: str, job_config: Any | None = None, timeout: float | None = None) -> Any: ...


class BigQueryClient:
    """Thin wrapper around google-cloud-bigquery for parameterized SQL execution."""

    def __init__(
        self,
        *,
        project_id: str | None = None,
        client: BigQueryQueryClient | None = None,
        request_timeout_sec: float = DEFAULT_REQUEST_TIMEOUT_SEC,
        attempts: int = DEFAULT_QUERY_ATTEMPTS,
    ):
        self.project_id = project_id or os.environ.get("BIGQUERY_PROJECT", DEFAULT_BIGQUERY_PROJECT)
        self._client = client
        # An injected client belongs to the caller, so retries must not discard it.
        self._owns_client = client is None
        self.request_timeout_sec = request_timeout_sec
        self.attempts = max(1, attempts)

    def _get_client(self) -> BigQueryQueryClient:
        if self._client is not None:
            return self._client
        try:
            from google.cloud import bigquery
        except ImportError as exc:
            raise BigQueryError(
                "google-cloud-bigquery is required for BigQueryClient. Install with: uv sync --extra glean"
            ) from exc

        self._client = bigquery.Client(project=self.project_id)
        return self._client

    def _discard_client(self) -> None:
        """Drop a client whose connection pool may hold dead sockets."""
        if self._owns_client:
            self._client = None

    def _build_job_config(self, params: Sequence[Any] | None) -> Any | None:
        from google.cloud import bigquery

        if not params:
            return None
        bq_params = []
        for param in params:
            if hasattr(param, "name") and hasattr(param, "type_"):
                if isinstance(param.value, list):
                    bq_params.append(bigquery.ArrayQueryParameter(param.name, param.type_, param.value))
                else:
                    bq_params.append(bigquery.ScalarQueryParameter(param.name, param.type_, param.value))
            else:
                bq_params.append(param)
        return bigquery.QueryJobConfig(query_parameters=bq_params)

    def _run_query(self, sql: str, job_config: Any | None) -> list[dict[str, Any]]:
        query_job = self._get_client().query(sql, job_config=job_config, timeout=self.request_timeout_sec)
        rows = query_job.result(timeout=self.request_timeout_sec)
        return [dict(row.items()) for row in rows]

    def query(
        self,
        sql: str,
        *,
        params: Sequence[Any] | None = None,
    ) -> list[dict[str, Any]]:
        job_config = self._build_job_config(params)

        last_exc: Exception | None = None
        attempts_made = 0
        for attempt in range(1, self.attempts + 1):
            attempts_made = attempt
            try:
                return self._run_query(sql, job_config)
            except Exception as exc:
                last_exc = exc
                if attempt == self.attempts or not _is_retryable(exc):
                    break
                print(
                    f"[BigQuery] Attempt {attempt}/{self.attempts} failed ({exc}); retrying in {RETRY_BACKOFF_SEC:.0f}s"
                )
                self._discard_client()
                time.sleep(RETRY_BACKOFF_SEC)

        raise BigQueryError(f"BigQuery query failed after {attempts_made} attempt(s): {last_exc}") from last_exc
