"""End-to-end training pipeline: validate, split, train, choose thresholds, evaluate, save.

Run with:  fraud-train --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from fraud_detection.config import TrainConfig, load_config
from fraud_detection.data.load import file_fingerprint, load_transactions
from fraud_detection.data.split import DataSplit, time_based_split
from fraud_detection.data.validate import validate_transactions
from fraud_detection.evaluation.metrics import classification_metrics
from fraud_detection.evaluation.threshold import (
    DecisionPolicy,
    choose_block_threshold,
    choose_cost_optimal_threshold,
    expected_cost,
    fraud_costs,
)
from fraud_detection.features.build import feature_columns
from fraud_detection.models.factory import fit_model, predict_scores
from fraud_detection.models.registry import save_model
from fraud_detection.models.tuning import tune_on_validation
from fraud_detection.tracking import Tracker

logger = logging.getLogger("fraud_detection")


def set_seeds(seed: int) -> None:
    """Fix random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)


def train_and_evaluate(
    name: str, params: dict[str, Any], split: DataSplit, config: TrainConfig
) -> dict[str, Any]:
    """Train one model, pick thresholds on validation, and evaluate on validation and test."""
    target, amount_col = config.data.target, config.data.amount_column
    columns = feature_columns(config.data.time_column, amount_col)

    val, test = split.validation, split.test
    started = time.perf_counter()
    trials: list[dict[str, Any]] = []
    if params.get("grid"):
        params, trials = tune_on_validation(
            name,
            params,
            split.train[columns],
            split.train[target],
            val[columns],
            val[target],
            config.seed,
        )
    model = fit_model(name, params, split.train[columns], split.train[target], config.seed)
    train_seconds = time.perf_counter() - started

    val_scores = predict_scores(model, val[columns])
    test_scores = predict_scores(model, test[columns])

    # Thresholds are chosen on validation only, then frozen and applied to test.
    review_threshold, val_cost = choose_cost_optimal_threshold(
        val[target],
        val_scores,
        val[amount_col],
        config.costs.review_cost,
        config.costs.min_fraud_cost,
    )
    block_threshold = choose_block_threshold(
        val[target], val_scores, review_threshold, config.decision.block_precision_target
    )
    policy = DecisionPolicy(review_threshold, block_threshold)

    test_cost = expected_cost(
        test[target],
        test_scores,
        test[amount_col],
        review_threshold,
        config.costs.review_cost,
        config.costs.min_fraud_cost,
    )
    actions, counts = np.unique(policy.decide(test_scores), return_counts=True)

    result = {
        "model": name,
        "params": params,
        "train_seconds": round(train_seconds, 2),
        "tuning": trials,
        "policy": policy.to_dict(),
        "validation": {
            **classification_metrics(val[target], val_scores, review_threshold, val[amount_col]),
            "expected_cost": val_cost,
        },
        "test": {
            **classification_metrics(test[target], test_scores, review_threshold, test[amount_col]),
            "expected_cost": test_cost,
            "decisions": dict(zip(actions.tolist(), counts.tolist(), strict=True)),
        },
    }
    result["artifact"] = str(
        save_model(model, name, {**result, "data": config.data.path}, config.output.models_dir)
    )
    logger.info(
        "%s: test PR-AUC %.3f, recall %.3f, precision %.3f, cost %.0f",
        name,
        result["test"]["pr_auc"],
        result["test"]["recall"],
        result["test"]["precision"],
        test_cost,
    )
    return result


def results_table(results: list[dict[str, Any]], no_model_cost: float) -> str:
    """Markdown summary table of every model on the held-out test set."""
    lines = [
        "| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Recall | Precision | Alerts "
        "| Fraud amount caught | Cost | Saving vs no model |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda item: item["test"]["pr_auc"], reverse=True):
        t = r["test"]
        caught = t.get("fraud_amount_caught_pct")
        saving = 100 * (1 - t["expected_cost"] / no_model_cost) if no_model_cost else 0.0
        lines.append(
            f"| {r['model']} | {r['validation']['pr_auc']:.3f} | {t['pr_auc']:.3f} "
            f"| {t['roc_auc']:.3f} | {t['recall']:.3f} | {t['precision']:.3f} | {t['alerts']} "
            f"| {caught:.1f}% | {t['expected_cost']:,.0f} | {saving:.1f}% |"
        )
    return "\n".join(lines)


def run(
    config: TrainConfig,
    data_path: str | None = None,
    only: list[str] | None = None,
    track: bool = False,
) -> Path:
    """Run the whole pipeline and return the path of the JSON report.

    ``track=True`` turns on MLflow tracking even if the config leaves it disabled.
    """
    set_seeds(config.seed)
    path = data_path or config.data.path
    df = validate_transactions(load_transactions(path), config.data.target)
    split = time_based_split(
        df,
        config.data.time_column,
        config.data.target,
        config.split.train_fraction,
        config.split.validation_fraction,
    )

    names = only or list(config.models)
    unknown = sorted(set(names) - set(config.models))
    if unknown:
        raise ValueError(f"Models not defined in config: {unknown}")

    tracking = dict(config.raw.get("tracking") or {})
    if track:
        tracking["enabled"] = True
    tracker = Tracker(tracking)
    run_params = {
        "seed": config.seed,
        "split": config.raw.get("split", {}),
        "costs": config.raw.get("costs", {}),
        "decision": config.raw.get("decision", {}),
        "models": ",".join(names),
    }
    tags = {"data_path": str(path), "data_sha256": file_fingerprint(path)}
    results = []
    with tracker.pipeline_run(run_params, tags):
        for name in names:
            result = train_and_evaluate(name, dict(config.models[name] or {}), split, config)
            tracker.log_model_result(result)
            results.append(result)

    test = split.test
    no_model_cost = float(
        fraud_costs(test[config.data.amount_column], config.costs.min_fraud_cost)[
            test[config.data.target].to_numpy() == 1
        ].sum()
    )
    reports_dir = Path(config.output.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "data_path": str(path),
        "data_sha256": tags["data_sha256"],
        "seed": config.seed,
        "config": config.raw,
        "no_model_test_cost": no_model_cost,
        "results": results,
    }
    report_path = reports_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (reports_dir / "results.md").write_text(
        results_table(results, no_model_cost) + "\n", encoding="utf-8"
    )
    logger.info("Report written to %s", report_path)
    return report_path


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train and evaluate fraud detection models.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to YAML config")
    parser.add_argument("--data", default=None, help="Override the dataset path")
    parser.add_argument("--models", nargs="+", default=None, help="Train only these models")
    parser.add_argument("--track", action="store_true", help="Log the run to MLflow")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(load_config(args.config), data_path=args.data, only=args.models, track=args.track)


if __name__ == "__main__":
    main()
