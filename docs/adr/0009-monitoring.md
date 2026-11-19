# ADR 0009: Prediction logging and drift monitoring

- Status: accepted
- Date: 2026-11-19

## Context

Fraud patterns change over time (ADR 0001 shows fraud spread unevenly across periods), and
fraud labels arrive weeks later through chargebacks. A deployed model needs a way to notice
that the data it sees no longer looks like its training data, before labelled performance
can confirm a drop.

## Decision

- The API can append every prediction (inputs, score, decision, model version, timestamp) to
  a JSON Lines file set by `FRAUD_PREDICTION_LOG`. Logging failures are caught and logged so
  monitoring can never break scoring.
- `fraud-monitor` compares a current period with the training period using the Population
  Stability Index (PSI), a standard drift measure in credit risk:
  - per feature (the amount and V1 to V28; raw `Time` is a counter that always grows, so it is
    excluded), and on the fraud scores themselves;
  - below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 significant.
- Two modes:
  - `--log predictions.jsonl`: real traffic from the API, no labels yet (early warning);
  - no log: the held-out test period stands in for production with delayed labels, so PR-AUC,
    recall and precision are also compared with the model's validation PR-AUC.
- Alerts: any feature or score with significant drift, or a PR-AUC drop above
  `--max-pr-auc-drop` (0.05 by default). The command exits with code 1 when alerts are raised,
  so a scheduler or CI job can act on it.

## Consequences

- Drift detection works without labels; performance checks need them and lag behind.
- PSI flags a change, not its cause: an alert means investigate, then retrain if confirmed.
- The prediction log contains raw transaction features. In a real bank it would go to a
  secured store with retention rules rather than a local file.
- Retraining is triggered by a person after an alert; automatic retraining is not in scope.
