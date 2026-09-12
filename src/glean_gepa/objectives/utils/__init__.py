"""Telemetry helpers for the objective plugins.

Each module owns the BigQuery SQL, parsing, and score aggregation for one
metric family; :mod:`glean_gepa.objectives.utils.agentspan_query` holds the
shared shard-window and per-entry-query primitives they build on.
"""
