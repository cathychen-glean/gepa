import threading

from glean_gepa.al_adapter import GleanAdapterBase
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evolutionary_proposer import _select_screened_children


def _batch(scores: list[float]) -> GleanEvaluationBatch:
    trajectories = [
        {
            "data": {
                "eval_set_name": "set",
                "eval_set_version": "v1",
                "deployment_ids": ["prod"],
                "status": "active",
            },
            "output": {"entry_id": f"entry-{index}"},
            "score": score,
            "objective_scores": {},
        }
        for index, score in enumerate(scores)
    ]
    return GleanEvaluationBatch(
        outputs=[],
        scores=scores,
        trajectories=trajectories,
        objective_scores=[{} for _ in scores],
        summary={"objective": sum(scores) / len(scores) if scores else 0.0},
    )


def test_high_signal_batch_contains_only_parent_failures():
    adapter = GleanAdapterBase.__new__(GleanAdapterBase)

    focused = adapter.high_signal_batch(_batch([0.0, 0.5, 1.0]))

    assert len(focused) == 1
    assert focused[0]["eval_entry_ids"] == ["entry-0", "entry-1"]


def test_cached_eval_run_id_is_attached_to_matching_eval_set():
    adapter = GleanAdapterBase.__new__(GleanAdapterBase)
    batch = [
        {
            "eval_set_name": "focused",
            "eval_set_version": "v1",
            "deployment_ids": ["prod"],
            "status": "active",
        }
    ]

    attached = adapter.attach_cached_eval_run_ids(
        batch,
        [
            {
                "eval_set_name": "focused",
                "eval_set_version": "v1",
                "student_eval_run_id": "run-cached",
            }
        ],
    )

    assert attached[0]["cached_student_eval_run_id"] == "run-cached"


def test_high_signal_fix_rate():
    adapter = GleanAdapterBase.__new__(GleanAdapterBase)
    parent = _batch([0.0, 0.0, 0.0, 1.0])

    assert adapter.high_signal_fix_rate(parent, _batch([1.0, 1.0, 0.0])) == 2 / 3
    assert adapter.high_signal_fix_rate(_batch([1.0]), _batch([1.0])) == 0.0


def test_high_signal_screen_threshold():
    adapter = GleanAdapterBase.__new__(GleanAdapterBase)
    parent = _batch([0.0, 0.0, 0.0])
    keep, reject, exact = object(), object(), object()

    kept = _select_screened_children(
        adapter,
        parent,
        [keep, reject, exact],  # type: ignore[arg-type]
        [_batch([1.0, 1.0, 0.0]), _batch([0.0, 0.0, 0.0]), _batch([1.0, 0.0, 0.0])],
        use_high_signal_gate=True,
    )
    assert [(child, score) for child, _evaluation, score in kept] == [(keep, 2 / 3), (exact, 1 / 3)]

    below_custom = _select_screened_children(
        adapter,
        parent,
        [keep],  # type: ignore[arg-type]
        [_batch([1.0, 1.0, 0.0])],
        use_high_signal_gate=True,
        high_signal_screen_threshold=0.8,
    )
    assert below_custom == []


def test_high_signal_batch_evaluation_dispatches_children_concurrently():
    adapter = GleanAdapterBase.__new__(GleanAdapterBase)
    barrier = threading.Barrier(2)

    def evaluate_fn(_batch_data, _candidate, _capture_traces):
        barrier.wait(timeout=1)
        return _batch([1.0])

    adapter._evaluate_fn = evaluate_fn
    items = [
        (
            {"WRITING_CODE": f"child-{index}"},
            [
                {
                    "eval_set_name": "set",
                    "eval_set_version": "v1",
                    "deployment_ids": ["prod"],
                    "status": "active",
                    "eval_entry_ids": ["entry"],
                }
            ],
        )
        for index in range(2)
    ]

    results = adapter.batch_evaluate(items)

    assert len(results) == 2


def test_full_validation_batch_evaluation_runs_non_focused_items():
    adapter = GleanAdapterBase.__new__(GleanAdapterBase)

    def evaluate_fn(_batch_data, _candidate, _capture_traces):
        return _batch([1.0])

    adapter._evaluate_fn = evaluate_fn
    items = [
        (
            {"WRITING_CODE": f"child-{index}"},
            [
                {
                    "eval_set_name": "set",
                    "eval_set_version": "v1",
                    "deployment_ids": ["prod"],
                    "status": "active",
                }
            ],
        )
        for index in range(2)
    ]

    results = adapter.batch_evaluate(items, capture_traces=False)

    assert len(results) == 2
