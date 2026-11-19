"""Train an XGBoost model on synthetic data so the API can run without the Kaggle file.

Usage:  python scripts/make_demo_model.py   (writes models/xgboost/<version>/ and examples/)
The model is for demos and smoke tests only; real results need the Kaggle dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

from fraud_detection.data.split import time_based_split
from fraud_detection.data.synthetic import make_transactions
from fraud_detection.evaluation.threshold import (
    DecisionPolicy,
    choose_block_threshold,
    choose_cost_optimal_threshold,
)
from fraud_detection.features.build import feature_columns
from fraud_detection.models.factory import fit_model, predict_scores
from fraud_detection.models.registry import save_model


def main() -> None:
    """Train, choose thresholds on validation, save the model and an example request."""
    df = make_transactions(n_rows=20_000, fraud_rate=0.01, seed=42)
    split = time_based_split(df)
    cols = feature_columns()
    model = fit_model(
        "xgboost",
        {"n_estimators": 100, "max_depth": 3},
        split.train[cols],
        split.train["Class"],
        seed=42,
    )
    val_scores = predict_scores(model, split.validation[cols])
    review, _ = choose_cost_optimal_threshold(
        split.validation["Class"], val_scores, split.validation["Amount"], 5.0, 10.0
    )
    block = choose_block_threshold(split.validation["Class"], val_scores, review, 0.9)
    policy = DecisionPolicy(review, block)
    folder = save_model(
        model, "xgboost", {"policy": policy.to_dict(), "data": "synthetic demo"}, "models"
    )

    examples = Path("examples")
    examples.mkdir(exist_ok=True)
    fraud = split.test[split.test["Class"] == 1].iloc[0][cols].to_dict()
    (examples / "transaction.json").write_text(json.dumps(fraud, indent=2) + "\n")
    print(f"Demo model saved to {folder}; example request in examples/transaction.json")


if __name__ == "__main__":
    main()
