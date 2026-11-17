"""A small Keras neural network wrapped as a scikit-learn classifier.

Wrapping it lets the network sit inside the same Pipeline as the other models, so it shares
the preprocessing, thresholds, evaluation and registry (see docs/adr/0006-keras-mlp.md).
TensorFlow is imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


class KerasMLPClassifier(ClassifierMixin, BaseEstimator):
    """Feed-forward network for binary classification with dropout and early stopping.

    Parameters
    ----------
    hidden_units:
        Size of each hidden layer, e.g. (64, 32).
    dropout:
        Fraction of units dropped after each hidden layer during training.
    learning_rate:
        Adam learning rate.
    batch_size, epochs:
        Mini-batch size and maximum number of passes over the data.
    patience:
        Epochs without improvement in holdout PR-AUC before training stops.
    positive_weight:
        Loss weight for fraud cases. Thresholds are chosen later from business costs, so this
        only needs to give the rare class enough learning signal, not to fix the decision point.
    holdout_fraction:
        Last share of the (time-ordered) training rows used for early stopping, so the
        validation split stays untouched for thresholds.
    random_state:
        Seed for weights, dropout and shuffling.

    """

    def __init__(
        self,
        hidden_units: tuple[int, ...] = (64, 32),
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        batch_size: int = 2048,
        epochs: int = 50,
        patience: int = 5,
        positive_weight: float = 10.0,
        holdout_fraction: float = 0.1,
        random_state: int = 42,
    ) -> None:
        """Store hyperparameters only, as scikit-learn estimators require."""
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.positive_weight = positive_weight
        self.holdout_fraction = holdout_fraction
        self.random_state = random_state

    def _build_network(self, n_features: int) -> Any:
        import keras

        layers = [keras.Input(shape=(n_features,))]
        for units in self.hidden_units:
            layers.append(keras.layers.Dense(units, activation="relu"))
            layers.append(keras.layers.Dropout(self.dropout))
        layers.append(keras.layers.Dense(1, activation="sigmoid"))
        network = keras.Sequential(layers)
        network.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
        )
        return network

    def _holdout_split(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return train and early-stopping holdout indices (holdout = last rows in time)."""
        n_holdout = max(1, int(len(y) * self.holdout_fraction))
        train_idx = np.arange(len(y) - n_holdout)
        holdout_idx = np.arange(len(y) - n_holdout, len(y))
        if y[holdout_idx].sum() == 0 or y[train_idx].sum() == 0:
            # Too few frauds at the end of the period: fall back to a stratified random holdout.
            logger.warning("Time-ordered holdout has no frauds; using a stratified holdout")
            rng = np.random.default_rng(self.random_state)
            holdout_idx = np.concatenate(
                [
                    rng.choice(
                        np.flatnonzero(y == label),
                        size=max(1, int((y == label).sum() * self.holdout_fraction)),
                        replace=False,
                    )
                    for label in (0, 1)
                ]
            )
            train_idx = np.setdiff1d(np.arange(len(y)), holdout_idx)
        return train_idx, holdout_idx

    def fit(self, X: np.ndarray, y: np.ndarray) -> KerasMLPClassifier:
        """Train with early stopping on holdout PR-AUC, restoring the best weights."""
        import keras

        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        keras.utils.set_random_seed(self.random_state)
        train_idx, holdout_idx = self._holdout_split(X, y)
        self.model_ = self._build_network(X.shape[1])
        stopper = keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=self.patience,
            restore_best_weights=True,
        )
        history = self.model_.fit(
            X[train_idx],
            y[train_idx],
            validation_data=(X[holdout_idx], y[holdout_idx]),
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight={0: 1.0, 1: float(self.positive_weight)},
            callbacks=[stopper],
            verbose=0,
        )
        self.history_ = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        self.epochs_trained_ = len(self.history_["loss"])
        logger.info(
            "Keras MLP trained %d epochs, best holdout PR-AUC %.3f",
            self.epochs_trained_,
            max(self.history_["val_pr_auc"]),
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return [P(legitimate), P(fraud)] for each row."""
        fraud = self.model_.predict(np.asarray(X, dtype="float32"), verbose=0).reshape(-1)
        return np.column_stack([1 - fraud, fraud])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Class labels at a 0.5 threshold (the pipeline uses cost-based thresholds instead)."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __getstate__(self) -> dict[str, Any]:
        """Serialise the Keras model to bytes so joblib can save the whole pipeline."""
        state = self.__dict__.copy()
        model = state.pop("model_", None)
        if model is not None:
            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "model.keras")
                model.save(path)
                with open(path, "rb") as handle:
                    state["model_bytes_"] = handle.read()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuild the Keras model from bytes when loading."""
        model_bytes = state.pop("model_bytes_", None)
        self.__dict__.update(state)
        if model_bytes is not None:
            import keras

            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "model.keras")
                with open(path, "wb") as handle:
                    handle.write(model_bytes)
                self.model_ = keras.models.load_model(path)
