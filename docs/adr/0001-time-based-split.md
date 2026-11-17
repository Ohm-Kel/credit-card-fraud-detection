# ADR 0001: Time-based split

- Status: accepted
- Date: 2026-11-17

## Context

The model will score future transactions using patterns learned from past ones. A random
split mixes later transactions into training, so the model is tested on a period it has
partly seen. Fraud also comes in campaigns (several related frauds close in time), which a
random split leaks across train and test.

## Decision

Sort by `Time` and split 60% train, 20% validation, 20% test. Thresholds are tuned on
validation; the test set is used once, for the final numbers.

## Consequences

- Test results reflect deployment conditions and are usually lower than with a random split.
- Each split must contain frauds; the pipeline fails loudly if one does not.
- Cross-validation, when added, must be time-aware (expanding window), not shuffled k-fold.
