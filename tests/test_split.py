import pytest

from fraud_detection.data.split import time_based_split


def test_split_is_time_ordered_without_overlap(transactions):
    shuffled = transactions.sample(frac=1, random_state=1)
    split = time_based_split(shuffled)
    assert split.train["Time"].max() <= split.validation["Time"].min()
    assert split.validation["Time"].max() <= split.test["Time"].min()


def test_split_sizes_and_no_rows_lost(transactions):
    split = time_based_split(transactions, train_fraction=0.6, validation_fraction=0.2)
    total = len(split.train) + len(split.validation) + len(split.test)
    assert total == len(transactions)
    assert len(split.train) == int(len(transactions) * 0.6)


@pytest.mark.parametrize("train, val", [(0.0, 0.2), (0.6, 0.0), (0.8, 0.2)])
def test_invalid_fractions_rejected(transactions, train, val):
    with pytest.raises(ValueError):
        time_based_split(transactions, train_fraction=train, validation_fraction=val)


def test_split_without_frauds_rejected(transactions):
    # The test split is the last 20% of rows; remove every fraud from it.
    transactions.loc[transactions.index[-1500:], "Class"] = 0
    with pytest.raises(ValueError, match="no fraud"):
        time_based_split(transactions, train_fraction=0.6, validation_fraction=0.2)
