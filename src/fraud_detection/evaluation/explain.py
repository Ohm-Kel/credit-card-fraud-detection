"""SHAP explanations: why the model scored a transaction as it did (see docs/adr/0007-shap.md).

- Global importance: mean absolute SHAP value per feature over a sample of transactions.
- Local reasons: the features that pushed one transaction's score up or down the most.

Explanations are computed on the model's input features (after preprocessing). The two
cyclical time features are summed into one "time_of_day" feature so reasons stay readable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

TIME_FEATURES = ("time_of_day_sin", "time_of_day_cos")
TIME_OF_DAY = "time_of_day"


EXACT_EXPLAINERS = ("XGBClassifier", "LogisticRegression")


def has_exact_explainer(model: Pipeline) -> bool:
    """Whether the model has a fast, exact SHAP explainer (tree or linear)."""
    return type(model.named_steps["model"]).__name__ in EXACT_EXPLAINERS


@dataclass(frozen=True)
class Explanation:
    """SHAP values for a set of transactions, with features already grouped."""

    values: np.ndarray  # shape (n_transactions, n_features)
    base_value: float
    feature_names: list[str]
    method: str
    units: str


def _group_time_features(values: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Sum the sine and cosine time contributions into a single time-of-day contribution."""
    if not all(f in names for f in TIME_FEATURES):
        return values, names
    time_idx = [names.index(f) for f in TIME_FEATURES]
    keep = [i for i in range(len(names)) if i not in time_idx]
    grouped = np.column_stack([values[:, time_idx].sum(axis=1), values[:, keep]])
    return grouped, [TIME_OF_DAY, *[names[i] for i in keep]]


def explain(
    model: Pipeline,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    max_background: int = 100,
    seed: int = 0,
) -> Explanation:
    """Compute SHAP values for X_explain.

    Tree models use the exact TreeExplainer (in log-odds); linear models use the
    LinearExplainer (in log-odds); other models (SVM, neural network) use the model-agnostic
    permutation explainer on the fraud probability, with a small background sample.
    """
    import shap

    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["model"]
    names = list(preprocessor.get_feature_names_out())
    background = preprocessor.transform(
        X_background.sample(n=min(max_background, len(X_background)), random_state=seed)
    )
    to_explain = preprocessor.transform(X_explain)
    class_name = type(classifier).__name__

    if class_name == "XGBClassifier":
        explainer = shap.TreeExplainer(classifier)
        result = explainer(to_explain)
        method, units = "tree", "log-odds"
    elif class_name == "LogisticRegression":
        explainer = shap.LinearExplainer(classifier, background)
        result = explainer(to_explain)
        method, units = "linear", "log-odds"
    else:

        def fraud_probability(rows: np.ndarray) -> np.ndarray:
            return classifier.predict_proba(rows)[:, 1]

        explainer = shap.Explainer(
            fraud_probability,
            shap.maskers.Independent(background),
            algorithm="permutation",
            seed=seed,
        )
        result = explainer(to_explain)
        method, units = "permutation", "probability"

    values = np.asarray(result.values, dtype=float)
    if values.ndim == 3:  # some explainers return one slice per class
        values = values[:, :, 1]
    base = float(np.ravel(result.base_values)[0])
    values, names = _group_time_features(values, names)
    logger.info("Explained %d transactions with the %s method", len(values), method)
    return Explanation(values, base, names, method, units)


def global_importance(explanation: Explanation) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, most important first."""
    importance = np.abs(explanation.values).mean(axis=0)
    return (
        pd.DataFrame({"feature": explanation.feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def top_reasons(
    explanation: Explanation, row: int, raw: pd.Series, k: int = 3
) -> list[dict[str, Any]]:
    """Return the k features that pushed one transaction's fraud score up the most.

    ``raw`` is the original (unprocessed) transaction, used to show readable values: the
    amount and PCA components as given, and time of day as an hour.
    """
    contributions = explanation.values[row]
    order = np.argsort(contributions)[::-1][:k]
    reasons = []
    for i in order:
        feature = explanation.feature_names[i]
        if feature == TIME_OF_DAY:
            value: float = round(float(raw["Time"]) % 86_400 / 3_600, 2)
        else:
            value = float(raw[feature])
        reasons.append(
            {
                "feature": feature,
                "value": value,
                "contribution": float(contributions[i]),
            }
        )
    return reasons
