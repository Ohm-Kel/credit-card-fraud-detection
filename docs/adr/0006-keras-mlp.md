# ADR 0006: Keras neural network as a scikit-learn estimator

- Status: accepted
- Date: 2026-11-17

## Context

We want a deep learning comparison next to the classical models, without a second training
path, second preprocessing or second evaluation code (ADR 0002).

## Decision

- `KerasMLPClassifier` implements the scikit-learn estimator interface (`fit`,
  `predict_proba`), so it sits in the same `Pipeline` as the other models and shares
  preprocessing, thresholds, metrics and the registry.
- Architecture: Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.3) →
  Dense(1, sigmoid); binary cross-entropy; Adam (learning rate 0.001); batch size 2048.
- Early stopping on PR-AUC of an internal holdout (the last 10% of training rows in time),
  restoring the best weights. The validation split stays reserved for thresholds; if the
  holdout has no frauds, a stratified holdout is used instead.
- Fraud cases get a loss weight of 10. Decision thresholds come from business costs later,
  so the weight only needs to give the rare class enough learning signal. Full balancing
  (about 580) would make gradients unstable and repeat the over-correction seen with the SVM
  (ADR 0005).
- Reproducibility: `keras.utils.set_random_seed`; the Keras model is serialised to bytes
  inside the pickled pipeline so `joblib` saves one artifact.
- TensorFlow is an optional extra (`.[dl]`) so the core package and main CI stay light; a
  separate CI job installs it and runs the full suite.

## Consequences

- Training on the full dataset takes about 15 seconds on CPU.
- Tabular data with anonymised features usually favours gradient boosting; the network is a
  comparison, and the results table decides.
- Hyperparameters (width, dropout, positive weight) can later use the validation `grid`.
