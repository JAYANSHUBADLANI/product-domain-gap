import json

import pytest

from studio_shift.recovery import (
    load_completed_cycles,
    nested_samples_per_run,
    summarize_recovery_curve,
)


def _pool(category: str, n: int) -> list[dict]:
    return [{"path": f"{category}_{i}.jpg", "category": category} for i in range(n)]


def test_samples_are_nested_within_a_run():
    # The real property this guards: budget 4's sample must be a strict
    # superset of budget 2's, for the same run and category, or the
    # recovery curve's shape could be an artifact of which images got
    # swapped in rather than genuinely more data.
    pool = _pool("widget", 60)
    result = nested_samples_per_run(pool, ["widget"], budgets=[1, 2, 4, 8], run_index=0, root_seed=1)

    paths_1 = {r["path"] for r in result[1]}
    paths_2 = {r["path"] for r in result[2]}
    paths_4 = {r["path"] for r in result[4]}
    paths_8 = {r["path"] for r in result[8]}

    assert paths_1 <= paths_2 <= paths_4 <= paths_8


def test_sample_sizes_match_budget_times_category_count():
    pool = _pool("widget", 60) + _pool("gadget", 60)
    result = nested_samples_per_run(pool, ["widget", "gadget"], budgets=[2, 8], run_index=0, root_seed=1)
    assert len(result[2]) == 4  # 2 per category, 2 categories
    assert len(result[8]) == 16


def test_different_runs_get_different_samples():
    pool = _pool("widget", 60)
    run0 = nested_samples_per_run(pool, ["widget"], budgets=[8], run_index=0, root_seed=1)
    run1 = nested_samples_per_run(pool, ["widget"], budgets=[8], run_index=1, root_seed=1)
    assert {r["path"] for r in run0[8]} != {r["path"] for r in run1[8]}


def test_same_run_same_seed_is_deterministic():
    pool = _pool("widget", 60)
    a = nested_samples_per_run(pool, ["widget"], budgets=[8], run_index=0, root_seed=1)
    b = nested_samples_per_run(pool, ["widget"], budgets=[8], run_index=0, root_seed=1)
    assert [r["path"] for r in a[8]] == [r["path"] for r in b[8]]


def test_budget_zero_produces_an_empty_sample():
    pool = _pool("widget", 60)
    result = nested_samples_per_run(pool, ["widget"], budgets=[0, 4], run_index=0, root_seed=1)
    assert result[0] == []


def test_raises_when_pool_smaller_than_largest_budget():
    pool = _pool("widget", 5)
    with pytest.raises(ValueError, match="fewer than the largest recovery budget"):
        nested_samples_per_run(pool, ["widget"], budgets=[1, 8], run_index=0, root_seed=1)


def test_summarize_recovery_curve_averages_across_runs():
    results = [
        {"run": 0, "labels_per_class": 4, "class_balanced_accuracy": 0.5},
        {"run": 1, "labels_per_class": 4, "class_balanced_accuracy": 0.7},
        {"run": 2, "labels_per_class": 4, "class_balanced_accuracy": 0.6},
        {"run": 0, "labels_per_class": 8, "class_balanced_accuracy": 0.8},
    ]
    summary = summarize_recovery_curve(results)
    row_4 = next(r for r in summary if r["labels_per_class"] == 4)
    assert row_4["mean_accuracy"] == pytest.approx(0.6)
    assert row_4["n_runs"] == 3
    assert row_4["min_accuracy"] == 0.5
    assert row_4["max_accuracy"] == 0.7


def test_load_completed_cycles_returns_empty_for_missing_file(tmp_path):
    assert load_completed_cycles(tmp_path / "does_not_exist.jsonl") == {}


def test_load_completed_cycles_reads_jsonl_into_a_run_budget_map(tmp_path):
    progress_path = tmp_path / "progress.jsonl"
    progress_path.write_text(
        json.dumps({"run": 0, "labels_per_class": 1, "class_balanced_accuracy": 0.5}) + "\n"
        + json.dumps({"run": 0, "labels_per_class": 2, "class_balanced_accuracy": 0.6}) + "\n"
    )
    completed = load_completed_cycles(progress_path)
    assert set(completed) == {(0, 1), (0, 2)}
    assert completed[(0, 1)]["class_balanced_accuracy"] == 0.5


def test_load_completed_cycles_skips_blank_lines(tmp_path):
    # A real killed-mid-write process could leave a trailing blank line
    # or an incomplete flush; blank lines specifically must not crash
    # the resume path on the next run.
    progress_path = tmp_path / "progress.jsonl"
    progress_path.write_text(
        json.dumps({"run": 0, "labels_per_class": 1, "class_balanced_accuracy": 0.5}) + "\n\n"
    )
    completed = load_completed_cycles(progress_path)
    assert set(completed) == {(0, 1)}


def test_summarize_recovery_curve_is_sorted_by_budget():
    results = [
        {"run": 0, "labels_per_class": 8, "class_balanced_accuracy": 0.8},
        {"run": 0, "labels_per_class": 1, "class_balanced_accuracy": 0.3},
    ]
    summary = summarize_recovery_curve(results)
    assert [r["labels_per_class"] for r in summary] == [1, 8]
