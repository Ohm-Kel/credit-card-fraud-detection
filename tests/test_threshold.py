import numpy as np
import pytest

from fraud_detection.evaluation.threshold import (
    APPROVE,
    BLOCK,
    REVIEW,
    DecisionPolicy,
    choose_block_threshold,
    choose_cost_optimal_threshold,
    expected_cost,
)

Y = np.array([0, 0, 0, 1, 1])
SCORES = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
AMOUNTS = np.array([10.0, 20.0, 30.0, 100.0, 5.0])


def test_expected_cost_by_hand():
    # Threshold 0.65 flags the two frauds: 2 reviews x 5 = 10, nothing missed.
    assert expected_cost(Y, SCORES, AMOUNTS, 0.65, review_cost=5, min_fraud_cost=10) == 10
    # Threshold 0.8 misses the 100 fraud and flags one: 100 + 5 = 105.
    assert expected_cost(Y, SCORES, AMOUNTS, 0.8, review_cost=5, min_fraud_cost=10) == 105
    # Flag nothing: 100 + max(5, 10) = 110.
    assert expected_cost(Y, SCORES, AMOUNTS, 1.1, review_cost=5, min_fraud_cost=10) == 110


def test_cost_optimal_threshold_flags_exactly_the_frauds():
    threshold, cost = choose_cost_optimal_threshold(Y, SCORES, AMOUNTS, 5, 10)
    assert 0.6 < threshold <= 0.7
    assert cost == 10


def test_expensive_reviews_mean_flagging_nothing():
    threshold, cost = choose_cost_optimal_threshold(Y, SCORES, AMOUNTS, 1_000, 10)
    assert threshold > SCORES.max()
    assert cost == 110


def test_block_threshold_meets_precision_target():
    threshold = choose_block_threshold(Y, SCORES, review_threshold=0.5, precision_target=1.0)
    assert (Y[SCORES >= threshold] == 1).all()


def test_block_threshold_infinite_when_target_unreachable():
    y = np.array([1, 0, 0, 0])
    scores = np.array([0.1, 0.5, 0.6, 0.9])
    assert np.isinf(choose_block_threshold(y, scores, 0.0, precision_target=0.9))


def test_policy_decisions():
    policy = DecisionPolicy(review_threshold=0.5, block_threshold=0.8)
    decisions = policy.decide(np.array([0.1, 0.5, 0.79, 0.8, 0.99]))
    assert decisions.tolist() == [APPROVE, REVIEW, REVIEW, BLOCK, BLOCK]


def test_policy_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        DecisionPolicy(review_threshold=0.8, block_threshold=0.5)


def test_policy_serialises_infinity_as_none():
    assert DecisionPolicy(0.5, float("inf")).to_dict() == {
        "review_threshold": 0.5,
        "block_threshold": None,
    }
