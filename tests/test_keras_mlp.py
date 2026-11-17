import joblib
import numpy as np
import pytest

pytest.importorskip("tensorflow")

from fraud_detection.data.split import time_based_split  # noqa: E402
from fraud_detection.features.build import feature_columns  # noqa: E402
from fraud_detection.models.factory import build_model, fit_model, predict_scores  # noqa: E402
from fraud_detection.models.keras_mlp import KerasMLPClassifier  # noqa: E402

FAST = {"hidden_units": [16], "epochs": 5, "patience": 2, "batch_size": 256}


def test_network_shape():
    network = KerasMLPClassifier(hidden_units=(8, 4), dropout=0.1)._build_network(5)
    dense_sizes = [layer.units for layer in network.layers if hasattr(layer, "units")]
    assert dense_sizes == [8, 4, 1]


def test_fit_predicts_probabilities_and_learns(transactions):
    split = time_based_split(transactions)
    cols = feature_columns()
    model = fit_model("keras_mlp", FAST, split.train[cols], split.train["Class"], seed=0)
    scores = predict_scores(model, split.validation[cols])
    assert scores.shape == (len(split.validation),)
    assert ((scores >= 0) & (scores <= 1)).all()
    frauds = split.validation["Class"].to_numpy() == 1
    assert scores[frauds].mean() > scores[~frauds].mean()


def test_same_seed_same_predictions(transactions):
    split = time_based_split(transactions)
    cols = feature_columns()
    first = fit_model("keras_mlp", FAST, split.train[cols], split.train["Class"], seed=3)
    second = fit_model("keras_mlp", FAST, split.train[cols], split.train["Class"], seed=3)
    np.testing.assert_allclose(
        predict_scores(first, split.test[cols]), predict_scores(second, split.test[cols]), atol=1e-5
    )


def test_pipeline_survives_save_and_load(tmp_path, transactions):
    split = time_based_split(transactions)
    cols = feature_columns()
    model = fit_model("keras_mlp", FAST, split.train[cols], split.train["Class"], seed=0)
    before = predict_scores(model, split.test[cols])
    joblib.dump(model, tmp_path / "model.joblib")
    after = predict_scores(joblib.load(tmp_path / "model.joblib"), split.test[cols])
    np.testing.assert_allclose(before, after, atol=1e-6)


def test_holdout_falls_back_when_no_late_frauds():
    y = np.zeros(1000)
    y[:20] = 1  # every fraud is early, so the time-ordered holdout would have none
    train_idx, holdout_idx = KerasMLPClassifier(holdout_fraction=0.1)._holdout_split(
        np.zeros((1000, 3)), y
    )
    assert y[holdout_idx].sum() > 0
    assert y[train_idx].sum() > 0
    assert not set(train_idx) & set(holdout_idx)


def test_factory_converts_hidden_units_to_tuple(transactions):
    model = build_model("keras_mlp", {"hidden_units": [8]}, transactions["Class"], seed=0)
    assert model.named_steps["model"].hidden_units == (8,)
