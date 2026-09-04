import json

import pytest

from gepa.core.data_loader import ListDataLoader
from glean_gepa.evalset_policy import TrainingScheduleExhaustedStopper, UnseenEvalSetPolicy


def test_unseen_evalset_policy_reveals_one_training_id_at_a_time():
    loader = ListDataLoader(["v1", "v2", "v3"])
    policy = UnseenEvalSetPolicy()

    assert policy.take_unseen(loader, purpose="reflection and offspring screening") == [0]
    assert policy.take_unseen(loader, purpose="reflection and offspring screening") == [1]
    assert policy.take_unseen(loader, purpose="reflection and offspring screening") == [2]


def test_unseen_evalset_policy_fails_instead_of_reusing_seen_data():
    loader = ListDataLoader(["v1"])
    policy = UnseenEvalSetPolicy()
    policy.take_unseen(loader, purpose="reflection and offspring screening")

    with pytest.raises(RuntimeError, match="No unseen eval sets remain"):
        policy.take_unseen(loader, purpose="reflection and offspring screening")


def test_restarted_run_continues_after_the_versions_it_already_used(tmp_path):
    state_file = tmp_path / "schedule.json"
    loader = ListDataLoader(["v1", "v2", "v3"])

    first = UnseenEvalSetPolicy(state_file=state_file)
    assert first.take_unseen(loader, purpose="screening", attempt=0) == [0]
    assert first.take_unseen(loader, purpose="screening", attempt=1) == [1]

    resumed = UnseenEvalSetPolicy(state_file=state_file)
    assert resumed.take_unseen(loader, purpose="screening", attempt=2) == [2]


def test_generation_interrupted_before_it_finished_replays_its_slice(tmp_path):
    state_file = tmp_path / "schedule.json"
    loader = ListDataLoader(["v1", "v2", "v3"])

    first = UnseenEvalSetPolicy(state_file=state_file)
    first.take_unseen(loader, purpose="screening", attempt=0)
    first.take_unseen(loader, purpose="screening", attempt=1)

    # The engine checkpoints before a generation starts, so a resumed run
    # repeats the attempt counter of the generation it was killed in.
    resumed = UnseenEvalSetPolicy(state_file=state_file)
    assert resumed.take_unseen(loader, purpose="screening", attempt=1) == [1]
    assert resumed.take_unseen(loader, purpose="screening", attempt=2) == [2]


def test_schedule_tracks_eval_set_versions_not_list_positions(tmp_path):
    state_file = tmp_path / "schedule.json"
    used = [{"eval_set_name": "Medium", "eval_set_version": version} for version in ("20260820", "20260824")]

    first = UnseenEvalSetPolicy(state_file=state_file)
    first.take_unseen(ListDataLoader(used), purpose="screening", attempt=0)
    first.take_unseen(ListDataLoader(used), purpose="screening", attempt=1)

    assert json.loads(state_file.read_text())["consumed"] == ["Medium:20260820"]

    # A restart that prepends a new version must not re-run either used one.
    reordered = [{"eval_set_name": "Medium", "eval_set_version": "20260828"}, *used]
    resumed = UnseenEvalSetPolicy(state_file=state_file)
    assert resumed.take_unseen(ListDataLoader(reordered), purpose="screening", attempt=2) == [0]


def test_stopper_ends_the_run_once_every_training_version_is_used(tmp_path):
    loader = ListDataLoader(["v1", "v2"])
    policy = UnseenEvalSetPolicy(state_file=tmp_path / "schedule.json")
    stopper = TrainingScheduleExhaustedStopper(policy, loader)

    assert not stopper(None)
    policy.take_unseen(loader, purpose="screening", attempt=0)
    policy.take_unseen(loader, purpose="screening", attempt=1)
    assert not stopper(None)

    # Starting a third generation retires the last slice and exhausts the schedule.
    with pytest.raises(RuntimeError, match="No unseen eval sets remain"):
        policy.take_unseen(loader, purpose="screening", attempt=2)
    assert stopper(None)
