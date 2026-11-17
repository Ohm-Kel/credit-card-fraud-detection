# ADR 0002: Single pipeline artifact

- Status: accepted
- Date: 2026-11-17

## Context

If preprocessing (scaling, time encoding) is reimplemented in the serving code, small
differences between training and serving silently degrade predictions. This is
training-serving skew, a common production failure.

## Decision

Every model is a scikit-learn `Pipeline` of the shared `ColumnTransformer` and the
classifier, saved as one `joblib` file. Scalers are fitted on the training split only.

## Consequences

- Serving loads one file and calls `predict_proba` on raw columns.
- No statistics from validation or test leak into preprocessing.
- The Keras model (phase 1b) needs a wrapper or a separate saved preprocessor to keep
  this guarantee.
