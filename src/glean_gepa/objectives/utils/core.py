"""Source-agnostic pieces every objective's util module builds on.

An objective util turns rows from *some* source (BigQuery Agentspan, EvalCLI
judge runs, ...) into one :class:`RunAnalysis`: an aggregate, per-entry
metrics, and the ids reflection should look at. This module owns the frame and
the steps that do not depend on where the rows came from. Source-specific
helpers live in ``agentspan.py`` (SQL + BigQuery) and ``traces.py`` (EvalCLI
trace enrichment).

A new util module supplies:

* an ``EntryMetrics`` dataclass satisfying :class:`EntryMetricsLike`
* ``parse_row(row) -> EntryMetrics | None``
* ``aggregate(per_entry) -> A`` for its own aggregate dataclass
* ``is_high_signal(metrics) -> bool``

and calls :func:`build_analysis`. Everything else here is reused as-is.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

# Reflection and console logging surface at most this many items per example:
# tool payloads, error strings, high-signal entries. One number, one name.
EVIDENCE_LIMIT = 5


# ---------------------------------------------------------------------------
# Entry and analysis frame
# ---------------------------------------------------------------------------


@runtime_checkable
class EntryMetricsLike(Protocol):
    """What ``core`` needs from one entry's metrics.

    ``passed`` decides high-signal defaults and pass rates; ``score`` is the
    per-entry value the objective reports in ``[0.0, 1.0]``. Objectives add
    their own fields (loop counts, citations, error examples) beside these.
    """

    @property
    def entry_id(self) -> str: ...

    @property
    def passed(self) -> bool: ...

    @property
    def score(self) -> float: ...


E = TypeVar("E", bound=EntryMetricsLike)
A = TypeVar("A")


@dataclass(frozen=True)
class RunAnalysis(Generic[A, E]):
    """One eval run (or a teacher/student pair) reduced to scores and evidence.

    ``eval_ids`` is ``(student,)`` for single-model sources and
    ``(teacher, student)`` for paired ones. ``start_date`` / ``end_date`` are
    ``None`` when the source is not date-sharded (EvalCLI views).
    """

    eval_ids: tuple[str, ...]
    aggregate: A
    per_entry: dict[str, E] = field(default_factory=dict)
    high_signal_entry_ids: tuple[str, ...] = ()
    start_date: date | None = None
    end_date: date | None = None

    @property
    def eval_id(self) -> str:
        """The scored (student) run. Last id so paired analyses read naturally."""
        return self.eval_ids[-1]

    @property
    def compared_entries(self) -> int:
        return len(self.per_entry)

    @property
    def passed_entries(self) -> int:
        return sum(1 for m in self.per_entry.values() if m.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed_entries / self.compared_entries if self.per_entry else 0.0

    @classmethod
    def of(
        cls,
        *,
        eval_id: str,
        aggregate: A,
        per_entry: Mapping[str, E] | None = None,
        high_signal_entry_ids: Sequence[str] = (),
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> RunAnalysis[A, E]:
        """Keyword constructor with the single-run field names callers already use."""
        return cls(
            eval_ids=(eval_id,),
            aggregate=aggregate,
            per_entry=dict(per_entry or {}),
            high_signal_entry_ids=tuple(high_signal_entry_ids),
            start_date=start_date,
            end_date=end_date,
        )


@dataclass(frozen=True)
class PairedRunAnalysis(RunAnalysis[A, E]):
    """Teacher/student analysis. Adds the role-named ids the adapters expect."""

    @property
    def teacher_eval_id(self) -> str:
        return self.eval_ids[0]

    @property
    def student_eval_id(self) -> str:
        return self.eval_ids[-1]

    @classmethod
    def of_pair(
        cls,
        *,
        teacher_eval_id: str,
        student_eval_id: str,
        aggregate: A,
        per_entry: Mapping[str, E] | None = None,
        high_signal_entry_ids: Sequence[str] = (),
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> PairedRunAnalysis[A, E]:
        """Keyword constructor with the paired field names callers already use."""
        return cls(
            eval_ids=(teacher_eval_id, student_eval_id),
            aggregate=aggregate,
            per_entry=dict(per_entry or {}),
            high_signal_entry_ids=tuple(high_signal_entry_ids),
            start_date=start_date,
            end_date=end_date,
        )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def parse_rows(rows: Iterable[Mapping[str, Any]], parse_row: Callable[[Mapping[str, Any]], E | None]) -> dict[str, E]:
    """Apply ``parse_row`` to each row, keeping the last metrics per ``entry_id``.

    ``parse_row`` returns ``None`` for rows the objective wants to skip (no
    entry id, excluded tool, ...). Later rows for the same entry win, which
    matches BigQuery's unordered result semantics closely enough for per-entry
    aggregates that are already grouped.
    """
    per_entry: dict[str, E] = {}
    for row in rows:
        metrics = parse_row(row)
        if metrics is not None:
            per_entry[metrics.entry_id] = metrics
    return per_entry


def pass_rate(per_entry: Mapping[str, E]) -> float:
    """Fraction of entries that ``passed``; ``0.0`` for an empty run."""
    return (sum(1 for m in per_entry.values() if m.passed) / len(per_entry)) if per_entry else 0.0


def mean_score(per_entry: Mapping[str, E]) -> float:
    """Mean of per-entry ``score``; ``0.0`` for an empty run."""
    return (sum(m.score for m in per_entry.values()) / len(per_entry)) if per_entry else 0.0


def select_high_signal(
    per_entry: Mapping[str, E], is_high_signal: Callable[[E], bool] | None = None
) -> tuple[str, ...]:
    """Sorted ids reflection should look at. Defaults to every failing entry."""
    predicate = is_high_signal or (lambda m: not m.passed)
    return tuple(sorted(entry_id for entry_id, m in per_entry.items() if predicate(m)))


def build_analysis(
    *,
    eval_ids: Sequence[str],
    per_entry: Mapping[str, E],
    aggregate: Callable[[Mapping[str, E]], A],
    is_high_signal: Callable[[E], bool] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    paired: bool = False,
) -> RunAnalysis[A, E]:
    """Assemble the frame from parsed entries.

    ``aggregate`` is the objective's own reducer so its aggregate dataclass
    keeps the field names its pack YAML and adapters already read.
    """
    ids = tuple(eval_ids)
    entries = dict(per_entry)
    cls: type[RunAnalysis[A, E]] = PairedRunAnalysis if paired else RunAnalysis
    return cls(
        eval_ids=ids,
        aggregate=aggregate(entries),
        per_entry=entries,
        high_signal_entry_ids=select_high_signal(entries, is_high_signal),
        start_date=start_date,
        end_date=end_date,
    )


def empty_analysis(
    *,
    eval_ids: Sequence[str],
    aggregate: Callable[[Mapping[str, E]], A],
    start_date: date | None = None,
    end_date: date | None = None,
    paired: bool = False,
) -> RunAnalysis[A, E]:
    """The 0-entry frame returned when the source has nothing yet.

    Objectives treat ``compared_entries == 0`` as *pending* (telemetry still
    ingesting), so this must not be cached.
    """
    return build_analysis(
        eval_ids=eval_ids,
        per_entry={},
        aggregate=aggregate,
        start_date=start_date,
        end_date=end_date,
        paired=paired,
    )


# ---------------------------------------------------------------------------
# Guards and logging
# ---------------------------------------------------------------------------


class NoComparedEntriesError(RuntimeError):
    """Raised when a *full* eval produced no comparable entries.

    A focused eval legitimately scores a subset; a full one with nothing to
    compare means the run failed or telemetry never landed, and continuing
    would silently score the candidate 0.
    """


def require_compared_entries(analysis: RunAnalysis[Any, Any], *, hint: str) -> None:
    if analysis.compared_entries > 0:
        return
    ids = ", ".join(analysis.eval_ids)
    raise NoComparedEntriesError(f"No entries compared for eval(s) {ids}. {hint}")


def log_analysis(
    analysis: RunAnalysis[Any, E],
    *,
    label: str,
    headline: str,
    entry_line: Callable[[E], str],
    limit: int = EVIDENCE_LIMIT,
) -> None:
    """Print one aggregate line then the first ``limit`` high-signal entries.

    ``headline`` is the objective's own summary (rates, means); ``entry_line``
    renders one entry's metrics. The banner, id prefix, and cap are shared.
    """
    print(f"[{label}] {analysis.eval_id}: {headline}")
    for entry_id in analysis.high_signal_entry_ids[:limit]:
        print(f"[{label}] High-signal entry={entry_id}: {entry_line(analysis.per_entry[entry_id])}")
