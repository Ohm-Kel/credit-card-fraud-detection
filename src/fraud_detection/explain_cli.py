"""Explain a trained model on the test set: global importance and reasons for top alerts.

Run with:  fraud-explain --model xgboost
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.config import TrainConfig, load_config
from fraud_detection.data.load import load_transactions
from fraud_detection.data.split import time_based_split
from fraud_detection.data.validate import validate_transactions
from fraud_detection.evaluation.explain import (
    explain,
    global_importance,
    has_exact_explainer,
    top_reasons,
)
from fraud_detection.features.build import feature_columns
from fraud_detection.models.factory import predict_scores
from fraud_detection.models.registry import latest_model, load_model

logger = logging.getLogger("fraud_detection")

# The permutation explainer costs about (2 x features + 1) x background predictions per row,
# so models without an exact explainer get a smaller sample (about 10x faster).
PERMUTATION_MAX_LEGIT = 100
PERMUTATION_BACKGROUND = 50


def select_rows(test: pd.DataFrame, target: str, n_legit: int, seed: int) -> pd.DataFrame:
    """Every fraud in the test set plus a random sample of legitimate transactions."""
    frauds = test[test[target] == 1]
    legit = test[test[target] == 0]
    sample = legit.sample(n=min(n_legit, len(legit)), random_state=seed)
    return pd.concat([frauds, sample]).sort_values("Time")


def save_importance_chart(importance: pd.DataFrame, title: str, path: Path, top: int = 15) -> None:
    """Horizontal bar chart of the most important features (grayscale)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(data["feature"], data["mean_abs_shap"], color="#444444")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_explain(
    config: TrainConfig,
    model_name: str,
    data_path: str | None = None,
    n_legit: int = 500,
    n_alerts: int = 5,
) -> Path:
    """Explain the latest saved model and write JSON and a chart to reports/explanations."""
    target = config.data.target
    df = validate_transactions(load_transactions(data_path or config.data.path), target)
    split = time_based_split(
        df,
        config.data.time_column,
        target,
        config.split.train_fraction,
        config.split.validation_fraction,
    )
    folder = latest_model(config.output.models_dir, model_name)
    model, metadata = load_model(folder)
    columns = feature_columns(config.data.time_column, config.data.amount_column)

    background = 100
    if not has_exact_explainer(model) and n_legit > PERMUTATION_MAX_LEGIT:
        logger.info(
            "%s has no exact explainer; explaining %d legitimate rows instead of %d",
            model_name,
            PERMUTATION_MAX_LEGIT,
            n_legit,
        )
        n_legit, background = PERMUTATION_MAX_LEGIT, PERMUTATION_BACKGROUND

    rows = select_rows(split.test, target, n_legit, config.seed)
    explanation = explain(
        model, split.train[columns], rows[columns], max_background=background, seed=config.seed
    )
    importance = global_importance(explanation)

    # Reasons for the highest-scoring test transactions (the alerts an analyst sees first).
    scores = predict_scores(model, rows[columns])
    alerts = []
    for position in np.argsort(scores)[::-1][:n_alerts]:
        raw = rows.iloc[position]
        alerts.append(
            {
                "score": float(scores[position]),
                "is_fraud": bool(raw[target] == 1),
                "amount": float(raw[config.data.amount_column]),
                "reasons": top_reasons(explanation, int(position), raw),
            }
        )

    out_dir = Path(config.output.reports_dir) / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": model_name,
        "version": metadata["version"],
        "method": explanation.method,
        "units": explanation.units,
        "base_value": explanation.base_value,
        "n_explained": len(rows),
        "n_frauds_explained": int(rows[target].sum()),
        "global_importance": importance.to_dict(orient="records"),
        "top_alerts": alerts,
    }
    report_path = out_dir / f"{model_name}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_importance_chart(
        importance,
        f"{model_name}: feature importance ({explanation.units})",
        out_dir / f"{model_name}_importance.png",
    )
    logger.info(
        "Top features for %s: %s",
        model_name,
        ", ".join(importance["feature"].head(5)),
    )
    return report_path


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Explain a trained fraud model with SHAP.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to YAML config")
    parser.add_argument("--model", default="xgboost", help="Model name to explain")
    parser.add_argument("--data", default=None, help="Override the dataset path")
    parser.add_argument("--n-legit", type=int, default=500, help="Legitimate rows to explain")
    parser.add_argument("--n-alerts", type=int, default=5, help="Top alerts to explain")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run_explain(load_config(args.config), args.model, args.data, args.n_legit, args.n_alerts)


if __name__ == "__main__":
    main()
