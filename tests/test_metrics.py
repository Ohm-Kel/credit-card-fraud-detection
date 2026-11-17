import numpy as np

from fraud_detection.evaluation.metrics import classification_metrics


def test_perfect_scores():
    y = np.array([0, 0, 1, 1])
    metrics = classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5, np.ones(4))
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["recall"] == metrics["precision"] == 1.0
    assert metrics["fraud_amount_caught_pct"] == 100.0


def test_confusion_counts():
    y = np.array([0, 1, 1, 0])
    metrics = classification_metrics(y, np.array([0.9, 0.9, 0.1, 0.1]), 0.5)
    assert (metrics["true_positives"], metrics["false_positives"]) == (1, 1)
    assert (metrics["false_negatives"], metrics["true_negatives"]) == (1, 1)
    assert metrics["alerts"] == 2
    assert "fraud_amount_total" not in metrics
