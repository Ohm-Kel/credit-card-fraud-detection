import numpy as np
import pytest

from fraud_detection.data.validate import DataValidationError, validate_transactions


def test_valid_data_passes(transactions):
    assert validate_transactions(transactions) is transactions


def test_missing_column_fails(transactions):
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_transactions(transactions.drop(columns=["V7"]))


def test_nulls_fail(transactions):
    transactions.loc[3, "Amount"] = np.nan
    with pytest.raises(DataValidationError, match="null values"):
        validate_transactions(transactions)


def test_non_binary_target_fails(transactions):
    transactions.loc[0, "Class"] = 2
    with pytest.raises(DataValidationError, match="binary"):
        validate_transactions(transactions)


def test_single_class_fails(transactions):
    transactions["Class"] = 0
    with pytest.raises(DataValidationError, match="both classes"):
        validate_transactions(transactions)


def test_negative_amount_fails(transactions):
    transactions.loc[0, "Amount"] = -1.0
    with pytest.raises(DataValidationError, match="negative"):
        validate_transactions(transactions)


def test_all_problems_reported_together(transactions):
    broken = transactions.drop(columns=["V1"])
    broken.loc[0, "Amount"] = -5.0
    with pytest.raises(DataValidationError) as error:
        validate_transactions(broken)
    assert "missing columns" in str(error.value)
    assert "negative" in str(error.value)
