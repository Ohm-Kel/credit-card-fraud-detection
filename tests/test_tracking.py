import pytest

from fraud_detection.config import load_config
from fraud_detection.pipeline import main, run
from fraud_detection.tracking import Tracker, _flatten


def test_flatten_nested_params():
    assert _flatten("", {"C": 1, "grid": {"gamma": [0.1, 1]}}) == {"C": 1, "grid.gamma": [0.1, 1]}


def test_disabled_tracker_is_a_no_op():
    tracker = Tracker({"enabled": False})
    with tracker.pipeline_run({"seed": 1}, {}):
        tracker.log_model_result({"anything": "ignored"})
    assert tracker.enabled is False


def test_run_logs_parent_and_one_child_per_model(config_file, tmp_path):
    mlflow = pytest.importorskip("mlflow")
    db = tmp_path / "mlflow.db"
    config = load_config(config_file)
    config.raw["tracking"] = {"enabled": True, "uri": str(db), "experiment": "test-exp"}
    run(config, only=["logistic_regression", "xgboost"])

    mlflow.set_tracking_uri(f"sqlite:///{db}")
    runs = mlflow.search_runs(experiment_names=["test-exp"])
    children = runs[runs["tags.mlflow.parentRunId"].notna()]
    assert sorted(children["tags.model"]) == ["logistic_regression", "xgboost"]
    assert (children["metrics.test_pr_auc"] > 0.5).all()
    assert children["metrics.review_threshold"].notna().all()
    parent = runs[runs["tags.mlflow.parentRunId"].isna()].iloc[0]
    assert parent["params.seed"] == "7"
    assert len(parent["tags.data_sha256"]) == 12


def test_track_flag_enables_tracking(config_file, tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)  # the default database mlflow.db is created here
    main(["--config", str(config_file), "--models", "logistic_regression", "--track"])
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    assert len(mlflow.search_runs(experiment_names=["fraud-detection"])) == 2


def test_metadata_artifact_is_logged(config_file, tmp_path):
    mlflow = pytest.importorskip("mlflow")
    db = tmp_path / "mlflow.db"
    config = load_config(config_file)
    config.raw["tracking"] = {"enabled": True, "uri": str(db), "experiment": "artifacts"}
    run(config, only=["xgboost"])
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    runs = mlflow.search_runs(experiment_names=["artifacts"])
    child = runs[runs["tags.mlflow.parentRunId"].notna()].iloc[0]
    files = [f.path for f in mlflow.MlflowClient().list_artifacts(child["run_id"])]
    assert files == ["metadata.json"]
    assert (tmp_path / "mlartifacts").exists()
