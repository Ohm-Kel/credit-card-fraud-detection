"""Hyperparameter search scored on the validation split.

The test split is never used here. Validation PR-AUC picks the parameters; the decision
thresholds are then chosen on the same validation split, which is slightly optimistic, so
the test set remains the only unbiased estimate.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score

from fraud_detection.models.factory import fit_model, predict_scores

logger = logging.getLogger(__name__)


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """All combinations of a parameter grid, e.g. {"C": [1, 10]} -> [{"C": 1}, {"C": 10}]."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]


def tune_on_validation(
    name: str,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Try every combination in params["grid"] and return the best params and all trials.

    Candidates are fitted without calibration, because PR-AUC only depends on how the scores
    rank transactions, and calibration does not change that ranking.
    """
    base = {k: v for k, v in params.items() if k != "grid"}
    trials = []
    for candidate in expand_grid(params.get("grid") or {}):
        trial_params = {**base, **candidate}
        model = fit_model(name, trial_params, X_train, y_train, seed, calibrate=False)
        pr_auc = float(average_precision_score(y_val, predict_scores(model, X_val)))
        trials.append({"params": candidate, "val_pr_auc": pr_auc})
        logger.info("%s %s: validation PR-AUC %.3f", name, candidate, pr_auc)
    best = max(trials, key=lambda trial: trial["val_pr_auc"])
    logger.info(
        "%s best params %s (validation PR-AUC %.3f)", name, best["params"], best["val_pr_auc"]
    )
    return {**base, **best["params"]}, trials
