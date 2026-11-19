# ADR 0010: MLflow experiment tracking

- Status: accepted
- Date: 2026-11-19

## Context

`reports/metrics.json` is overwritten on every run, so comparing today's results with last
week's means digging through git history. Our engineering standards require metrics to be
reported for every model or pipeline change.

## Decision

- Track runs with MLflow, kept optional (`tracking` extra) so the core package stays light.
- Turn it on with `tracking.enabled: true` in the config or `fraud-train --track`.
- Each `fraud-train` run is a parent MLflow run (seed, split, costs, data hash, data path) with
  one nested run per model: hyperparameters (including the tuning grid), validation and test
  metrics, decision thresholds, training time, and the model's `metadata.json` as an artifact.
- Storage: a local SQLite database (`mlflow.db`) with artifacts in `mlartifacts/`, because
  MLflow 3 has put its plain-folder backend in maintenance mode. A remote server can be used
  by setting `tracking.uri` to its URL.
- The model registry stays in `models/` (ADR 0002): the API loads from there, and MLflow is
  for comparing experiments, not serving.

## Consequences

- `mlflow ui --backend-store-uri sqlite:///mlflow.db` shows every run side by side.
- `mlflow.db` and `mlartifacts/` are git-ignored, like models and data.
