"""Check a served model for data drift, score drift and (when labels exist) performance decay.

Run with:
    fraud-monitor --model xgboost                      # test period as simulated production
    fraud-monitor --model xgboost --log predictions.jsonl   # real traffic logged by the API
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score

from fraud_detection.config import TrainConfig, load_config
from fraud_detection.data.load import load_transactions
from fraud_detection.data.split import time_based_split
from fraud_detection.data.validate import PCA_FEATURES, validate_transactions
from fraud_detection.evaluation.metrics import classification_metrics
from fraud_detection.features.build import feature_columns
from fraud_detection.models.factory import predict_scores
from fraud_detection.models.registry import latest_model, load_model
from fraud_detection.monitoring.drift import SIGNIFICANT, drift_status, feature_drift, psi

logger = logging.getLogger("fraud_detection")

# Raw Time only counts seconds since the start of the dataset, so it always "drifts";
# drift is checked on the amount and the PCA components instead.
DRIFT_COLUMNS = [*PCA_FEATURES, "Amount"]


def read_prediction_log(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read inputs and scores from the API's JSON Lines prediction log."""
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Prediction log {path} is empty")
    inputs = pd.DataFrame([r["inputs"] for r in records])
    scores = pd.Series([r["fraud_probability"] for r in records], name="fraud_probability")
    return inputs, scores


def run_monitor(
    config: TrainConfig,
    model_name: str,
    data_path: str | None = None,
    log_path: str | None = None,
    max_pr_auc_drop: float = 0.05,
) -> dict[str, Any]:
    """Compare the current period with the training period and write a monitoring report."""
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

    reference = split.train
    reference_scores = predict_scores(model, reference[columns])
    if log_path:
        current, current_scores = read_prediction_log(log_path)
        labels = None
        source = f"prediction log {log_path}"
    else:
        current = split.test
        current_scores = pd.Series(predict_scores(model, current[columns]))
        labels = current[target]
        source = "test period (simulated production with delayed labels)"

    drift = feature_drift(reference, current, DRIFT_COLUMNS)
    score_psi = psi(reference_scores, current_scores.to_numpy())
    alerts = [
        f"Significant drift in {row.feature} (PSI {row.psi:.2f})"
        for row in drift.itertuples()
        if row.status == SIGNIFICANT
    ]
    if drift_status(score_psi) == SIGNIFICANT:
        alerts.append(f"Significant drift in fraud scores (PSI {score_psi:.2f})")

    performance = None
    if labels is not None:
        review = metadata["policy"]["review_threshold"]
        current_metrics = classification_metrics(labels, current_scores, review)
        baseline = metadata.get("validation", {}).get("pr_auc")
        current_pr_auc = float(average_precision_score(labels, current_scores))
        performance = {
            "baseline_validation_pr_auc": baseline,
            "current_pr_auc": current_pr_auc,
            "current_recall": current_metrics["recall"],
            "current_precision": current_metrics["precision"],
            "current_alerts": current_metrics["alerts"],
        }
        if baseline is not None and baseline - current_pr_auc > max_pr_auc_drop:
            alerts.append(
                f"PR-AUC dropped from {baseline:.3f} to {current_pr_auc:.3f} "
                f"(more than {max_pr_auc_drop:.2f})"
            )

    report = {
        "model": model_name,
        "version": metadata["version"],
        "current_source": source,
        "n_reference": len(reference),
        "n_current": len(current),
        "score_psi": score_psi,
        "score_status": drift_status(score_psi),
        "feature_drift": drift.to_dict(orient="records"),
        "performance": performance,
        "alerts": alerts,
        "recommendation": "investigate and consider retraining" if alerts else "no action",
    }
    out_dir = Path(config.output.reports_dir) / "monitoring"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{model_name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "Score PSI %.3f (%s); %d alert(s); recommendation: %s",
        score_psi,
        report["score_status"],
        len(alerts),
        report["recommendation"],
    )
    for alert in alerts:
        logger.warning(alert)
    return report


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point; exits with code 1 when alerts are raised."""
    parser = argparse.ArgumentParser(description="Monitor a fraud model for drift and decay.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to YAML config")
    parser.add_argument("--model", default="xgboost", help="Model name to monitor")
    parser.add_argument("--data", default=None, help="Override the dataset path")
    parser.add_argument("--log", default=None, help="API prediction log (JSON Lines)")
    parser.add_argument("--max-pr-auc-drop", type=float, default=0.05)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    report = run_monitor(
        load_config(args.config), args.model, args.data, args.log, args.max_pr_auc_drop
    )
    sys.exit(1 if report["alerts"] else 0)


if __name__ == "__main__":
    main()
