"""Schema and quality checks run before any training."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
EXPECTED_COLUMNS = ["Time", *PCA_FEATURES, "Amount", "Class"]


class DataValidationError(ValueError):
    """Raised when the dataset does not meet the expected schema or quality rules."""


def validate_transactions(df: pd.DataFrame, target: str = "Class") -> pd.DataFrame:
    """Check schema and data quality, raising one error that lists every problem found.

    Duplicated rows are logged but not removed: the original dataset contains genuine
    repeated transactions, and dropping them is a modelling decision, not a validation one.
    """
    problems: list[str] = []

    if df.empty:
        raise DataValidationError("Dataset is empty")

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")

    present = [col for col in EXPECTED_COLUMNS if col in df.columns]
    non_numeric = [col for col in present if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        problems.append(f"non-numeric columns: {non_numeric}")

    null_counts = df[present].isna().sum()
    if null_counts.any():
        problems.append(f"null values: {null_counts[null_counts > 0].to_dict()}")

    if target in df.columns:
        labels = set(df[target].dropna().unique())
        if not labels <= {0, 1}:
            problems.append(f"target '{target}' must be binary 0/1, found {sorted(labels)}")
        elif labels != {0, 1}:
            problems.append(f"target '{target}' must contain both classes, found {labels}")

    if "Amount" in df.columns and (df["Amount"] < 0).any():
        problems.append("negative transaction amounts")

    if problems:
        raise DataValidationError("Data validation failed: " + "; ".join(problems))

    duplicates = int(df.duplicated().sum())
    if duplicates:
        logger.warning("Found %d duplicated rows (kept)", duplicates)

    fraud_rate = df[target].mean()
    logger.info("Validation passed: %d rows, fraud rate %.4f%%", len(df), 100 * fraud_rate)
    return df
