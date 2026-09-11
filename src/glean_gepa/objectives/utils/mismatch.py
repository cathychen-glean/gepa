"""Frequency-ranked selection of high-signal teacher/student mismatch groups.

Every paired objective turns a per-entry teacher/student difference into a
``(a, b)`` signature (first-tool pair, citation missing/extra summary, ...) and
then wants the most frequent divergences for reflection. That selection logic is
identical across objectives, so it lives here once.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

REFLECTION_HIGH_SIGNAL_ENTRY_LIMIT = 20


def select_mismatch_groups(
    mismatch_keys: Sequence[tuple[str, str] | None],
    *,
    max_entries: int = REFLECTION_HIGH_SIGNAL_ENTRY_LIMIT,
) -> tuple[list[int], list[tuple[str, str, int]]]:
    """Select mismatch indices by descending ``(a, b)`` group frequency.

    The most frequent group is always included in full, even when it exceeds
    ``max_entries``. Later whole groups are added while they still fit in the
    cap; groups that would overflow are skipped so later smaller groups can
    still be included.

    Returns ``(selected_indices, selected_groups)`` where each group is
    ``(key_a, key_b, taken_count)``.
    """
    if max_entries < 0:
        raise ValueError("max_entries must be non-negative")
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, key in enumerate(mismatch_keys):
        if key is None:
            continue
        groups[key].append(index)
    ranked = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1]))
    selected: list[int] = []
    selected_groups: list[tuple[str, str, int]] = []
    for (key_a, key_b), indices in ranked:
        if selected and len(selected) + len(indices) > max_entries:
            continue
        selected.extend(indices)
        selected_groups.append((key_a, key_b, len(indices)))
        if len(selected) >= max_entries:
            break
    return selected, selected_groups
