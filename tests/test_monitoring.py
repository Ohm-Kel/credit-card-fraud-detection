import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import load_config
from fraud_detection.monitoring.cli import main as monitor_main
from fraud_detection.monitoring.cli import run_monitor
from fraud_detection.monitoring.drift import (
    MODERATE,
    SIGNIFICANT,
    STABLE,
    drift_status,
    feature_drift,
    psi,
)
from fraud_detection.pipeline import run

RNG = np.random.default_rng(0)


def test_psi_near_zero_for_same_distribution():
    assert psi(RNG.normal(size=5000), RNG.normal(size=5000)) < 0.02


def test_psi_grows_with_shift():
    reference = RNG.normal(size=5000)
    small = psi(reference, RNG.normal(0.2, 1, 5000))
    large = psi(reference, RNG.normal(1.0, 1, 5000))
    assert small < large
    assert large > 0.25


def test_psi_handles_constant_reference_and_rejects_empty():
    assert psi(np.ones(100), np.ones(100)) < 1e-3
    assert psi(np.ones(100), np.zeros(100)) > 0.25
    with pytest.raises(ValueError):
        psi(np.array([]), np.ones(3))


def test_drift_status_thresholds():
    assert drift_status(0.05) == STABLE
    assert drift_status(0.15) == MODERATE
    assert drift_status(0.3) == SIGNIFICANT


def test_feature_drift_flags_the_shifted_column():
    reference = pd.DataFrame({"a": RNG.normal(size=3000), "b": RNG.normal(size=3000)})
    current = pd.DataFrame({"a": RNG.normal(size=3000), "b": RNG.normal(2, 1, 3000)})
    result = feature_drift(reference, current, ["a", "b"])
    assert result.iloc[0]["feature"] == "b"
    assert result.iloc[0]["status"] == SIGNIFICANT
    assert result.set_index("feature").loc["a", "status"] == STABLE


@pytest.fixture
def trained(config_file):
    config = load_config(config_file)
    run(config, only=["xgboost"])
    return config


def test_monitor_on_test_period_reports_performance(trained):
    report = run_monitor(trained, "xgboost")
    assert report["performance"]["current_pr_auc"] > 0.5
    assert len(report["feature_drift"]) == 29
    assert report["score_status"] in {STABLE, MODERATE, SIGNIFICANT}
    saved_path = Path(trained.output.reports_dir) / "monitoring" / "xgboost.json"
    assert json.loads(saved_path.read_text())["model"] == "xgboost"


def _write_log(path, transactions, amount_shift=0.0):
    with path.open("w") as handle:
        for _, row in transactions.drop(columns="Class").iterrows():
            inputs = row.to_dict()
            inputs["Amount"] = inputs["Amount"] + amount_shift
            handle.write(json.dumps({"inputs": inputs, "fraud_probability": 0.01}) + "\n")


def test_monitor_flags_drift_in_logged_traffic(trained, config_file, transactions, tmp_path):
    log = tmp_path / "predictions.jsonl"
    _write_log(log, transactions.head(800), amount_shift=5_000.0)
    report = run_monitor(trained, "xgboost", log_path=str(log))
    assert report["performance"] is None
    assert any("Amount" in alert for alert in report["alerts"])
    assert report["recommendation"] != "no action"
    with pytest.raises(SystemExit) as exit_info:
        monitor_main(["--config", str(config_file), "--log", str(log)])
    assert exit_info.value.code == 1


def test_api_logs_every_prediction(tmp_path, transactions):
    pytest.importorskip("fastapi")
    pytest.importorskip("shap")
    from fastapi.testclient import TestClient

    from fraud_detection.data.split import time_based_split
    from fraud_detection.features.build import feature_columns
    from fraud_detection.models.factory import fit_model
    from fraud_detection.models.registry import save_model
    from fraud_detection.serving.app import create_app

    cols = feature_columns()
    split = time_based_split(transactions)
    model = fit_model("xgboost", {"n_estimators": 20}, split.train[cols], split.train["Class"], 0)
    folder = save_model(
        model, "xgboost", {"policy": {"review_threshold": 0.5, "block_threshold": None}}, tmp_path
    )
    log = tmp_path / "logs" / "predictions.jsonl"
    rows = [{c: float(r[c]) for c in cols} for _, r in split.test.head(3).iterrows()]
    with TestClient(create_app(folder, prediction_log=log)) as client:
        for row in rows:
            client.post("/predict", json=row)
        client.post("/predict/batch", json={"transactions": rows[:2]})
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 5
    assert set(records[0]) == {
        "timestamp",
        "model_name",
        "model_version",
        "inputs",
        "fraud_probability",
        "decision",
    }
    assert records[0]["inputs"]["Amount"] == pytest.approx(rows[0]["Amount"])


def test_logging_failure_does_not_break_scoring(tmp_path, transactions):
    pytest.importorskip("fastapi")
    pytest.importorskip("shap")
    from fastapi.testclient import TestClient

    from fraud_detection.data.split import time_based_split
    from fraud_detection.features.build import feature_columns
    from fraud_detection.models.factory import fit_model
    from fraud_detection.models.registry import save_model
    from fraud_detection.serving.app import create_app

    cols = feature_columns()
    split = time_based_split(transactions)
    model = fit_model("xgboost", {"n_estimators": 20}, split.train[cols], split.train["Class"], 0)
    folder = save_model(
        model, "xgboost", {"policy": {"review_threshold": 0.5, "block_threshold": None}}, tmp_path
    )
    unwritable = tmp_path / "a_directory_not_a_file"
    unwritable.mkdir()
    row = {c: float(split.test.iloc[0][c]) for c in cols}
    with TestClient(create_app(folder, prediction_log=unwritable)) as client:
        assert client.post("/predict", json=row).status_code == 200
