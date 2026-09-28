"""Shared plumbing the objective modules call. Nothing here is objective-specific.

An objective is one file under ``objectives/`` (see ``_template_agentspan.py``
and ``_template_evalcli.py``). It imports from here and never from another
objective.

``core``
    The frame. ``RunAnalysis`` / ``PairedRunAnalysis``, the ``EntryMetricsLike``
    protocol (``entry_id``, ``passed``, ``score``), ``build_analysis``,
    ``empty_analysis``, ``select_high_signal``, ``require_compared_entries``,
    ``log_analysis``, and the one ``EVIDENCE_LIMIT``.
``agentspan``
    BigQuery on top of ``agentspan_query``. ``bounds_query``,
    ``paired_role_query`` (the teacher/student FULL OUTER JOIN scaffold), and
    ``fetch_agentspan_analysis`` (window -> query -> filter -> parse ->
    post_parse -> high-signal -> enrich -> aggregate).
``traces``
    ``enrich_action_inputs``: resolve tool payloads from detailed traces for
    high-signal entries, single or paired role, all payloads or first tool.
``agentspan_query``
    Table and shard-window constants, ``default_date_range``,
    ``wildcard_shard_filter``, ``EVAL_ENTRY_ID_EXPR``.
``evalset_entries``
    ``fact.*`` lookups used by the adapters to build focused replay sets.
``tool_names``
    Pure tool-name helpers shared with ``prompt`` and ``run_log``.
``mismatch``
    Grouping near-duplicate failures for reflection.
"""
