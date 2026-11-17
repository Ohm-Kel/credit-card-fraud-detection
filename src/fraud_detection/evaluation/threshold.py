"""Cost-based decision thresholds (see docs/adr/0003-metrics-and-thresholds.md).

A model outputs a fraud probability. Turning it into an action needs thresholds, and the
right ones depend on what mistakes cost the bank, not on a default of 0.5:

- a missed fraud costs the transaction amount (at least ``min_fraud_cost``);
- every alert costs one analyst review (``review_cost``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

APPROVE, REVIEW, BLOCK = "approve", "review", "block"


def fraud_costs(amounts: np.ndarray, min_fraud_cost: float) -> np.ndarray:
    """Cost of missing each transaction if it is fraud."""
    return np.maximum(np.asarray(amounts, dtype=float), min_fraud_cost)


def expected_cost(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    review_cost: float,
    min_fraud_cost: float,
) -> float:
    """Total cost of flagging every transaction with a score at or above the threshold."""
    y_true = np.asarray(y_true)
    flagged = np.asarray(scores) >= threshold
    missed = (y_true == 1) & ~flagged
    return float(fraud_costs(amounts, min_fraud_cost)[missed].sum() + flagged.sum() * review_cost)


def candidate_thresholds(scores: np.ndarray, max_candidates: int = 1000) -> np.ndarray:
    """Distinct score quantiles to try, plus one value above every score ("flag nothing")."""
    scores = np.asarray(scores, dtype=float)
    quantiles = np.quantile(scores, np.linspace(0, 1, max_candidates))
    above_all = np.nextafter(scores.max(), np.inf)
    return np.unique(np.append(quantiles, above_all))


def choose_cost_optimal_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    review_cost: float,
    min_fraud_cost: float,
) -> tuple[float, float]:
    """Return the threshold with the lowest expected cost, and that cost."""
    best_threshold, best_cost = np.inf, np.inf
    for threshold in candidate_thresholds(scores):
        cost = expected_cost(y_true, scores, amounts, threshold, review_cost, min_fraud_cost)
        if cost < best_cost:
            best_threshold, best_cost = float(threshold), cost
    return best_threshold, best_cost


def choose_block_threshold(
    y_true: np.ndarray, scores: np.ndarray, review_threshold: float, precision_target: float
) -> float:
    """Lowest threshold above the review threshold whose alerts reach the precision target.

    Returns infinity when no threshold reaches the target, meaning nothing is blocked
    automatically and every alert goes to an analyst.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    for threshold in candidate_thresholds(scores):
        if threshold < review_threshold:
            continue
        flagged = scores >= threshold
        if flagged.sum() == 0:
            break
        if y_true[flagged].mean() >= precision_target:
            return float(threshold)
    return float("inf")


@dataclass(frozen=True)
class DecisionPolicy:
    """Map fraud scores to approve, review or block."""

    review_threshold: float
    block_threshold: float

    def __post_init__(self) -> None:
        """Ensure the block threshold never sits below the review threshold."""
        if self.block_threshold < self.review_threshold:
            raise ValueError("block_threshold must be greater than or equal to review_threshold")

    def decide(self, scores: np.ndarray) -> np.ndarray:
        """Return the action for each score."""
        scores = np.asarray(scores)
        return np.where(
            scores >= self.block_threshold,
            BLOCK,
            np.where(scores >= self.review_threshold, REVIEW, APPROVE),
        )

    def to_dict(self) -> dict[str, float | None]:
        """JSON-friendly representation (infinity becomes None)."""
        return {k: (None if np.isinf(v) else v) for k, v in asdict(self).items()}
