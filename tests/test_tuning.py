from fraud_detection.data.split import time_based_split
from fraud_detection.features.build import feature_columns
from fraud_detection.models.factory import build_model, fit_model, predict_scores
from fraud_detection.models.tuning import expand_grid, tune_on_validation


def test_expand_grid():
    assert expand_grid({"C": [1, 10], "gamma": ["scale"]}) == [
        {"C": 1, "gamma": "scale"},
        {"C": 10, "gamma": "scale"},
    ]
    assert expand_grid({}) == [{}]


def test_svm_has_no_class_weights(transactions):
    model = build_model("svm", {"kernel": "rbf"}, transactions["Class"], seed=0)
    svc = model.named_steps["model"].estimator
    assert svc.class_weight is None


def test_uncalibrated_svm_scores_with_margin(transactions):
    split = time_based_split(transactions)
    cols = feature_columns()
    params = {"kernel": "rbf", "negatives_per_positive": 10}
    model = fit_model(
        "svm", params, split.train[cols], split.train["Class"], seed=0, calibrate=False
    )
    scores = predict_scores(model, split.validation[cols])
    assert scores.shape == (len(split.validation),)
    assert (scores < 0).any()  # margins, not probabilities


def test_tuning_picks_best_validation_candidate(transactions):
    split = time_based_split(transactions)
    cols = feature_columns()
    params = {"kernel": "rbf", "negatives_per_positive": 10, "grid": {"C": [0.01, 1.0]}}
    best, trials = tune_on_validation(
        "svm",
        params,
        split.train[cols],
        split.train["Class"],
        split.validation[cols],
        split.validation["Class"],
        seed=0,
    )
    assert len(trials) == 2
    top = max(trials, key=lambda t: t["val_pr_auc"])
    assert best["C"] == top["params"]["C"]
    assert "grid" not in best
