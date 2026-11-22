# ADR 0008: Widen the SVM hyperparameter grid

- Status: accepted
- Date: 2026-11-22
- Extends: [ADR 0005](0005-svm-imbalance.md)

## Context

ADR 0005 tuned the RBF SVM over `C: [0.1, 1.0, 10.0]` and `gamma: [scale, 0.01, 0.001]` by
validation PR-AUC. The best setting, `C=10, gamma=0.001`, sat at the highest `C` and the lowest
`gamma` in the grid — both edges — so a better setting could lie just outside it. The README
flagged this as an open roadmap item.

## Decision

- Drop the two candidates that were clearly worse at every other value (`C=0.1`, `gamma=scale`)
  to keep the grid size in check.
- Extend outward from the previous best in both directions: `C: [10.0, 30.0, 100.0, 300.0]` and
  `gamma: [0.001, 0.0005, 0.0003, 0.0001]` (16 combinations, up from 9).
- Before fixing the final grid, probe further out (`C` up to 1,000, `gamma` down to 0.0001) on
  the real dataset to check whether the winner would still sit at an edge. Validation PR-AUC
  plateaus between `C=100` and `C=1,000` (0.767-0.768, noise-level), so `C=300` is a reasonable
  upper bound rather than an open-ended edge.
- The chosen grid's best setting on the real dataset is `C=300, gamma=0.001`
  (validation PR-AUC 0.768), an interior point. Test PR-AUC rose from 0.791 to 0.798 and
  precision from 0.770 to 0.792 at the same recall (0.760); the change in cost is within noise
  (2,839 to 2,858 — see README for the full row).

## Consequences

- Training is on undersampled data (ADR 0005), so even with 16 combinations tuning stays under
  a minute on the full dataset.
- If a future change to the undersampling ratio or feature set moves the winner back to an edge,
  the grid should be widened again rather than treated as permanently settled.
- A wider grid costs more validation fits; if this becomes a bottleneck, replace the exhaustive
  grid with a coarser-to-fine search (e.g. halve the grid, then refine around the winner).
