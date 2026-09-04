"""Training-eval scheduling for Glean's evolutionary proposer."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from gepa.core.data_loader import DataLoader
from gepa.core.state import GEPAState

SCHEDULE_SCHEMA_VERSION = 1


def _slice_label(loader: DataLoader, data_id: Any) -> str:
    """Name a training slice by its eval set rather than its position.

    ``DataLoader`` ids are positions in the configured list, so a restart that
    adds or reorders ``--train_eval_versions`` would otherwise re-run a version
    that a previous process already used.
    """
    item = loader.fetch([data_id])[0]
    if isinstance(item, dict):
        eval_set_name = item.get("eval_set_name")
        eval_set_version = item.get("eval_set_version")
        if eval_set_name and eval_set_version:
            return f"{eval_set_name}:{eval_set_version}"
    elif isinstance(item, str):
        return item
    return f"index:{data_id}"


class UnseenEvalSetPolicy:
    """Select one previously unseen training eval-set item per generation.

    The schedule is persisted, so a restarted run continues with the next
    unused training version instead of replaying the first one. The slice a
    generation takes stays pending until a later generation starts: a run that
    is killed mid-generation therefore replays the same slice on resume and
    reuses the eval runs it already created for it.

    Validation remains the engine's full-evaluation policy: every configured
    validation version is evaluated for each accepted candidate.
    """

    def __init__(self, state_file: str | os.PathLike[str] | None = None) -> None:
        self._ordered_ids: list[Any] | None = None
        self._consumed: list[str] = []
        self._pending_label: str | None = None
        self._pending_attempt: int | None = None
        self.state_file = Path(state_file).expanduser() if state_file else None
        self._load()

    def _load(self) -> None:
        if self.state_file is None or not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text())
            if not isinstance(data, dict):
                raise ValueError("eval-set schedule must be a JSON object")
            if data.get("schema_version") != SCHEDULE_SCHEMA_VERSION:
                print(f"[Eval set schedule] Ignoring unsupported schedule schema in {self.state_file}")
                return
            self._consumed = [str(label) for label in data.get("consumed", [])]
            pending = data.get("pending")
            if isinstance(pending, dict):
                self._pending_label = str(pending["label"])
                self._pending_attempt = int(pending["attempt"])
            print(f"[Eval set schedule] Loaded {len(self._consumed)} used training eval sets from {self.state_file}")
        except (OSError, TypeError, ValueError, KeyError) as exc:
            print(f"[Eval set schedule] Failed to load {self.state_file}: {exc}")
            self._consumed = []
            self._pending_label = None
            self._pending_attempt = None

    def _save(self) -> None:
        if self.state_file is None:
            return
        data: dict[str, Any] = {
            "schema_version": SCHEDULE_SCHEMA_VERSION,
            "consumed": list(self._consumed),
            "pending": (
                None
                if self._pending_label is None
                else {"label": self._pending_label, "attempt": self._pending_attempt}
            ),
        }
        state_dir = self.state_file.parent
        state_dir.mkdir(parents=True, exist_ok=True)
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=state_dir, delete=False) as temp_file:
                temp_path = temp_file.name
                json.dump(data, temp_file, indent=2)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, self.state_file)
        except (OSError, TypeError, ValueError) as exc:
            print(f"[Eval set schedule] Failed to save {self.state_file}: {exc}")
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _ids(self, loader: DataLoader) -> list[Any]:
        ids = list(loader.all_ids())
        if self._ordered_ids is None:
            self._ordered_ids = ids
        elif ids != self._ordered_ids:
            raise ValueError("UnseenEvalSetPolicy must be shared with loaders containing the same eval sets")
        return self._ordered_ids

    def _consume_pending(self) -> None:
        """Retire the slice of the generation that has now been left behind."""
        if self._pending_label is not None and self._pending_label not in self._consumed:
            self._consumed.append(self._pending_label)
        self._pending_label = None
        self._pending_attempt = None
        self._save()

    def take_unseen(self, loader: DataLoader, *, purpose: str, attempt: int | None = None) -> list[Any]:
        """Return the slice for this generation.

        ``attempt`` is the engine's proposal counter. A resumed run repeats the
        counter value of the generation it was interrupted in, which is how a
        pending slice is recognised as unfinished rather than used.
        """
        ids = self._ids(loader)
        labels = [_slice_label(loader, data_id) for data_id in ids]

        if self._pending_label is not None:
            if attempt is not None and attempt == self._pending_attempt and self._pending_label in labels:
                selected = ids[labels.index(self._pending_label)]
                print(
                    f"[Eval set schedule] {purpose}: replaying id {selected} ({self._pending_label}) "
                    "from the generation that was interrupted before it finished"
                )
                return [selected]
            self._consume_pending()

        for data_id, label in zip(ids, labels, strict=True):
            if label in self._consumed:
                continue
            self._pending_label = label
            self._pending_attempt = attempt
            self._save()
            used = sum(1 for candidate_label in labels if candidate_label in self._consumed) + 1
            print(f"[Eval set schedule] {purpose}: selected id {data_id} ({label}) ({used}/{len(ids)})")
            return [data_id]

        raise RuntimeError(f"No unseen eval sets remain for {purpose}; configured {len(ids)} eval-set versions")

    def is_exhausted(self, loader: DataLoader) -> bool:
        """Report whether every configured training eval set has been used."""
        if self._pending_label is not None:
            return False
        return all(_slice_label(loader, data_id) in self._consumed for data_id in self._ids(loader))


class TrainingScheduleExhaustedStopper:
    """Stop the run once no unseen training eval set is left to learn from.

    Without it the engine keeps looping over a proposer that can only return
    empty proposals, which consumes no budget and therefore never stops.
    """

    def __init__(self, policy: UnseenEvalSetPolicy, loader: DataLoader) -> None:
        self.policy = policy
        self.loader = loader

    def __call__(self, gepa_state: GEPAState) -> bool:
        return self.policy.is_exhausted(self.loader)
