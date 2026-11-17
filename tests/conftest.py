from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fraud_detection.data.validate import PCA_FEATURES


def make_transactions(n_rows: int = 6000, fraud_rate: float = 0.03, seed: int = 0) -> pd.DataFrame:
    """Synthetic data with the same schema as the Kaggle dataset and a learnable signal."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n_rows) < fraud_rate).astype(int)
    features = rng.normal(size=(n_rows, len(PCA_FEATURES)))
    features[y == 1, :4] += 2.5  # frauds differ on the first components
    df = pd.DataFrame(features, columns=PCA_FEATURES)
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_800, n_rows)))
    df["Amount"] = np.round(rng.lognormal(3, 1.2, n_rows), 2)
    df["Class"] = y
    return df


@pytest.fixture
def transactions() -> pd.DataFrame:
    return make_transactions()


@pytest.fixture
def config_file(tmp_path: Path, transactions: pd.DataFrame) -> Path:
    data_path = tmp_path / "creditcard.csv"
    transactions.to_csv(data_path, index=False)
    config = {
        "seed": 7,
        "data": {"path": str(data_path)},
        "split": {"train_fraction": 0.6, "validation_fraction": 0.2},
        "costs": {"review_cost": 5.0, "min_fraud_cost": 10.0},
        "decision": {"block_precision_target": 0.9},
        "models": {
            "logistic_regression": {"C": 1.0, "max_iter": 500},
            "svm": {"C": 1.0, "kernel": "rbf", "negatives_per_positive": 10},
            "xgboost": {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1},
        },
        "output": {
            "models_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
        },
    }
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path
