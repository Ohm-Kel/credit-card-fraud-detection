"""Model definitions. Every model is a Pipeline of the shared preprocessor and a classifier."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from fraud_detection.features.build import build_preprocessor

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ("logistic_regression", "svm", "xgboost")


def build_model(name: str, params: dict[str, Any], y_train: pd.Series, seed: int) -> Pipeline:
    """Create an unfitted pipeline for the named model.

    Class imbalance is handled inside each model (class weights or scale_pos_weight)
    rather than by oversampling, so the validation and test sets keep the real fraud rate.
    """
    params = {k: v for k, v in params.items() if k != "negatives_per_positive"}
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    if positives == 0:
        raise ValueError("Training labels contain no fraud cases")

    if name == "logistic_regression":
        classifier = LogisticRegression(class_weight="balanced", random_state=seed, **params)
    elif name == "svm":
        # SVMs output margins, not probabilities; sigmoid (Platt) calibration converts them.
        classifier = CalibratedClassifierCV(
            SVC(class_weight="balanced", random_state=seed, **params),
            method="sigmoid",
            cv=3,
            ensemble=False,
        )
    elif name == "xgboost":
        classifier = XGBClassifier(
            scale_pos_weight=negatives / positives,
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=-1,
            **params,
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Supported: {SUPPORTED_MODELS}")

    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", classifier)])


def undersample_negatives(
    X: pd.DataFrame, y: pd.Series, negatives_per_positive: int, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Keep every fraud and a random sample of legitimate transactions.

    Used for the RBF SVM, whose training time grows roughly with the square of the
    number of rows, which makes the full training set impractical.
    """
    rng = np.random.default_rng(seed)
    positive_idx = np.flatnonzero(y.to_numpy() == 1)
    negative_idx = np.flatnonzero(y.to_numpy() == 0)
    n_negatives = min(len(negative_idx), negatives_per_positive * len(positive_idx))
    sampled = rng.choice(negative_idx, size=n_negatives, replace=False)
    keep = np.sort(np.concatenate([positive_idx, sampled]))
    logger.info("Undersampled training set: %d rows (%d frauds)", len(keep), len(positive_idx))
    return X.iloc[keep], y.iloc[keep]


def fit_model(
    name: str, params: dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series, seed: int
) -> Pipeline:
    """Build and fit a model, applying any model-specific sampling."""
    if name == "svm" and params.get("negatives_per_positive"):
        X_train, y_train = undersample_negatives(
            X_train, y_train, int(params["negatives_per_positive"]), seed
        )
    model = build_model(name, params, y_train, seed)
    model.fit(X_train, y_train)
    return model


def predict_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Fraud probability for each transaction."""
    return model.predict_proba(X)[:, 1]
