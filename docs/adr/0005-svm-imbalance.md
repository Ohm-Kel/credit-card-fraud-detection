# ADR 0005: SVM imbalance handling and tuning

- Status: accepted
- Date: 2026-11-17
- Supersedes: the SVM part of ADR 0004

## Context

The first run on the Kaggle dataset gave the RBF SVM a test ROC-AUC of 0.975 but a PR-AUC of
0.206 and a precision of 0.119 (469 alerts for 75 frauds). The SVM was trained on undersampled
data (all frauds plus 50 legitimate transactions per fraud) and also used
`class_weight="balanced"`, so class imbalance was corrected twice. `C` and `gamma` were defaults.

## Decision

- Undersampling alone rebalances the SVM's training data, so the SVM uses no class weights.
- Keep more legitimate transactions (100 per fraud) so the region near the decision boundary is
  better represented, while keeping training time practical.
- Choose `C` and `gamma` from a small grid by validation PR-AUC. Candidates are fitted without
  calibration, since calibration does not change the ranking; the chosen model is then calibrated.
- The test set is never used for tuning.

## Consequences

- The grid adds 9 extra fits (about 30 seconds on the full dataset).
- Tuning and threshold selection share the validation split, which is slightly optimistic; the
  test set remains the unbiased estimate.
- The same validation-based search can be reused for the other models via a `grid` entry.
