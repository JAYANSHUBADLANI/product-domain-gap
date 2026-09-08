import pytest

from studio_shift.evaluate import (
    Prediction,
    bootstrap_class_balanced_ci,
    class_balanced_accuracy,
    confidence_gap_analysis,
    per_category_accuracy,
)


def _pred(true_cat, correct, confidence=0.9):
    return Prediction(
        true_category=true_cat,
        predicted_category=true_cat if correct else "other",
        confidence=confidence,
        correct=correct,
    )


def test_class_balanced_accuracy_averages_per_category_not_overall():
    # 9 correct out of 10 for "widget", 0 correct out of 1 for "rare":
    # overall micro accuracy would be 9/11 = 81.8%, masking that "rare"
    # is at 0%. Class-balanced accuracy must average the two categories'
    # own accuracies (90% and 0%) instead: 45%.
    predictions = [_pred("widget", True) for _ in range(9)] + [_pred("widget", False)] + [_pred("rare", False)]
    assert class_balanced_accuracy(predictions) == 0.45


def test_class_balanced_accuracy_all_correct():
    predictions = [_pred("a", True), _pred("b", True)]
    assert class_balanced_accuracy(predictions) == 1.0


def test_per_category_accuracy_reports_each_category_separately():
    predictions = [_pred("a", True), _pred("a", False), _pred("b", True)]
    result = per_category_accuracy(predictions)
    assert result == {"a": 0.5, "b": 1.0}


def test_bootstrap_ci_contains_the_point_estimate():
    predictions = [_pred("a", True) for _ in range(15)] + [_pred("a", False) for _ in range(5)]
    result = bootstrap_class_balanced_ci(predictions, n_boot=500, root_seed=1, namespace="test")
    assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]


def test_bootstrap_ci_is_deterministic_for_a_fixed_seed():
    predictions = [_pred("a", i % 3 != 0) for i in range(20)]
    a = bootstrap_class_balanced_ci(predictions, n_boot=200, root_seed=7, namespace="ns")
    b = bootstrap_class_balanced_ci(predictions, n_boot=200, root_seed=7, namespace="ns")
    assert a == b


def test_bootstrap_ci_is_narrow_for_a_large_unanimous_sample():
    # All correct, many examples: the interval should sit tight around 1.0,
    # not span a wide range, since there is no real uncertainty here.
    predictions = [_pred("a", True) for _ in range(200)]
    result = bootstrap_class_balanced_ci(predictions, n_boot=500, root_seed=1, namespace="test")
    assert result["ci_low"] == result["ci_high"] == 1.0


def test_confidence_gap_analysis_separates_correct_and_incorrect():
    predictions = [
        _pred("a", True, confidence=0.95),
        _pred("a", True, confidence=0.85),
        _pred("a", False, confidence=0.6),
    ]
    result = confidence_gap_analysis(predictions)
    assert result["n_correct"] == 2
    assert result["n_incorrect"] == 1
    assert result["mean_confidence_correct"] == pytest.approx(0.9)
    assert result["mean_confidence_incorrect"] == 0.6


def test_confidence_gap_analysis_handles_no_incorrect_predictions():
    predictions = [_pred("a", True, confidence=0.9)]
    result = confidence_gap_analysis(predictions)
    assert result["n_incorrect"] == 0
    assert result["mean_confidence_incorrect"] is None
