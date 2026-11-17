# ADR 0003: Metrics and cost-based thresholds

- Status: accepted
- Date: 2026-11-17

## Context

With about 0.17% fraud, accuracy is meaningless (predicting "legitimate" always scores 99.8%)
and ROC-AUC looks strong even for weak models, because the huge number of true negatives
keeps the false positive rate low. The default threshold of 0.5 has no business meaning.

## Decision

- Headline metric: PR-AUC (average precision), which reflects alert quality.
- Review threshold: minimises expected cost on validation, where
  cost = sum of missed fraud amounts (at least `min_fraud_cost`) + `review_cost` per alert.
- Block threshold: lowest threshold whose alerts reach `block_precision_target` precision.
- Report recall, precision, alert volume, share of fraud amount caught and saving against
  having no model.

## Consequences

- Thresholds change when business costs change, without retraining.
- Cost assumptions (review cost of 5, minimum fraud cost of 10) are illustrative and should be
  set with the business.
