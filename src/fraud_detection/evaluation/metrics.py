"""Evaluation metrics suited to rare-event classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    amounts: np.ndarray | None = None,
) -> dict[str, Any]:
    """Threshold-free and threshold-dependent metrics for one model on one split.

    PR-AUC (average precision) is the headline metric: with about 0.17% fraud, ROC-AUC and
    accuracy look excellent even for weak models, while PR-AUC reflects alert quality.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    metrics: dict[str, Any] = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "alerts": int(tp + fp),
        "fraud_rate": float(y_true.mean()),
    }
    if amounts is not None:
        amounts = np.asarray(amounts, dtype=float)
        fraud_amount = amounts[y_true == 1].sum()
        caught_amount = amounts[(y_true == 1) & (predicted == 1)].sum()
        metrics["fraud_amount_total"] = float(fraud_amount)
        metrics["fraud_amount_caught_pct"] = (
            float(100 * caught_amount / fraud_amount) if fraud_amount > 0 else None
        )
    return metrics
