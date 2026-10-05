"""Shared plumbing the objective modules call. Nothing here is objective-specific.

An objective is one file under ``objectives/`` (see the ``_template_*.py`` files).
It imports from here and never from another objective. ``core`` owns the analysis
frame, ``agentspan`` the BigQuery pipeline, ``traces`` the trace enrichment; each
module's docstring says what it adds.
"""
