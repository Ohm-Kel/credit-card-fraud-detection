import json

import pytest

from fraud_detection.config import load_config
from fraud_detection.models.registry import load_model
from fraud_detection.pipeline import main, run


def test_end_to_end_run(config_file):
    config = load_config(config_file)
    report_path = run(config)

    report = json.loads(report_path.read_text())
    assert {r["model"] for r in report["results"]} == {"logistic_regression", "svm", "xgboost"}
    assert (report_path.parent / "results.md").exists()

    for result in report["results"]:
        # The synthetic signal is strong, so every model should rank frauds well.
        assert result["test"]["pr_auc"] > 0.5
        assert result["test"]["expected_cost"] <= report["no_model_test_cost"]
        model, metadata = load_model(result["artifact"])
        assert metadata["policy"]["review_threshold"] == result["policy"]["review_threshold"]
        assert hasattr(model, "predict_proba")


def test_cli_runs_selected_models(config_file):
    main(["--config", str(config_file), "--models", "logistic_regression"])
    config = load_config(config_file)
    report = json.loads((config_file.parent / "reports" / "metrics.json").read_text())
    assert [r["model"] for r in report["results"]] == ["logistic_regression"]
    assert config.output.models_dir


def test_unknown_model_rejected(config_file):
    with pytest.raises(ValueError, match="not defined"):
        run(load_config(config_file), only=["random_forest"])


def test_missing_dataset_message(config_file):
    with pytest.raises(FileNotFoundError, match="kaggle"):
        run(load_config(config_file), data_path="does/not/exist.csv")
