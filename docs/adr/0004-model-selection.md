# ADR 0004: Model selection and class imbalance

- Status: accepted
- Date: 2026-11-17

## Context

The data is tabular with strong imbalance. We want a simple, interpretable baseline, a
kernel method, and a strong tabular model, compared under the same protocol.

## Decision

- Logistic regression: baseline, `class_weight="balanced"`.
- RBF SVM: training cost grows roughly with the square of the row count, so it trains on all
  frauds plus a random sample of legitimate transactions (`negatives_per_positive`). SVMs
  return margins, so scores are calibrated with Platt (sigmoid) scaling.
- XGBoost: `scale_pos_weight` = negatives / positives; usually the strongest on tabular data.
- Imbalance is handled with class weights rather than SMOTE, so no synthetic rows are created
  and validation / test keep the real fraud rate. SMOTE can be compared later inside training
  folds only.

## Consequences

- Undersampling and class weights distort predicted probabilities; thresholds are therefore
  chosen on validation scores rather than assumed at 0.5.
- A Keras network will be added as a further comparison (phase 1b).
