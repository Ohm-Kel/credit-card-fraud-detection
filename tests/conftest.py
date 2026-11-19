from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from fraud_detection.data.synthetic import make_transactions


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
            "svm": {"kernel": "rbf", "negatives_per_positive": 10, "grid": {"C": [0.1, 1.0]}},
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
