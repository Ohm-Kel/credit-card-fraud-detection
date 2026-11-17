import json

import numpy as np
import pytest

pytest.importorskip("shap")

from fraud_detection.config import load_config  # noqa: E402
from fraud_detection.data.split import time_based_split  # noqa: E402
from fraud_detection.evaluation.explain import (  # noqa: E402
    TIME_OF_DAY,
    Explanation,
    _group_time_features,
    explain,
    global_importance,
    top_reasons,
)
from fraud_detection.explain_cli import run_explain  # noqa: E402
from fraud_detection.features.build import feature_columns  # noqa: E402
from fraud_detection.models.factory import fit_model  # noqa: E402
from fraud_detection.pipeline import run  # noqa: E402

COLS = feature_columns()


@pytest.fixture
def split(transactions):
    return time_based_split(transactions)


def test_time_features_are_summed():
    values = np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 1.0]])
    grouped, names = _group_time_features(values, ["time_of_day_sin", "time_of_day_cos", "V1"])
    assert names == [TIME_OF_DAY, "V1"]
    np.testing.assert_allclose(grouped, [[3.0, 3.0], [0.0, 1.0]])


def test_tree_shap_adds_up_to_model_output(split):
    params = {"n_estimators": 30, "max_depth": 3}
    model = fit_model("xgboost", params, split.train[COLS], split.train["Class"], seed=0)
    rows = split.test[COLS].head(50)
    result = explain(model, split.train[COLS], rows)
    assert result.method == "tree"
    margin = model.named_steps["model"].predict(
        model.named_steps["preprocess"].transform(rows), output_margin=True
    )
    np.testing.assert_allclose(result.base_value + result.values.sum(axis=1), margin, atol=1e-4)


def test_linear_and_permutation_methods(split):
    linear = fit_model(
        "logistic_regression", {"max_iter": 500}, split.train[COLS], split.train["Class"], 0
    )
    assert explain(linear, split.train[COLS], split.test[COLS].head(10)).method == "linear"

    svm = fit_model(
        "svm", {"negatives_per_positive": 5}, split.train[COLS], split.train["Class"], 0
    )
    result = explain(svm, split.train[COLS], split.test[COLS].head(5), max_background=20)
    assert result.method == "permutation"
    assert result.values.shape == (5, 30)  # 31 model inputs, two time features grouped


def test_signal_features_rank_first(split):
    # The synthetic frauds differ on V1 to V4, so those should dominate importance.
    params = {"n_estimators": 50, "max_depth": 3}
    model = fit_model("xgboost", params, split.train[COLS], split.train["Class"], seed=0)
    importance = global_importance(explain(model, split.train[COLS], split.test[COLS]))
    assert set(importance["feature"].head(4)) == {"V1", "V2", "V3", "V4"}


def test_top_reasons_are_sorted_and_readable(transactions):
    explanation = Explanation(
        values=np.array([[0.2, -1.0, 3.0]]),
        base_value=0.0,
        feature_names=[TIME_OF_DAY, "Amount", "V1"],
        method="tree",
        units="log-odds",
    )
    raw = transactions.iloc[0].copy()
    raw["Time"] = 86_400 + 3 * 3_600  # 03:00 on day two
    reasons = top_reasons(explanation, 0, raw, k=2)
    assert [r["feature"] for r in reasons] == ["V1", TIME_OF_DAY]
    assert reasons[1]["value"] == 3.0


def test_explain_cli_end_to_end(config_file):
    config = load_config(config_file)
    run(config, only=["xgboost"])
    report_path = run_explain(config, "xgboost", n_legit=100, n_alerts=3)
    report = json.loads(report_path.read_text())
    assert report["method"] == "tree"
    assert len(report["top_alerts"]) == 3
    assert len(report["top_alerts"][0]["reasons"]) == 3
    assert (report_path.parent / "xgboost_importance.png").exists()


def test_explain_cli_without_trained_model(config_file):
    with pytest.raises(FileNotFoundError, match="fraud-train"):
        run_explain(load_config(config_file), "xgboost")
