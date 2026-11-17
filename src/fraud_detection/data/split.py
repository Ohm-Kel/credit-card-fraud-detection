"""Time-ordered train / validation / test split (see docs/adr/0001-time-based-split.md)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataSplit:
    """The three time-ordered partitions of the dataset."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def time_based_split(
    df: pd.DataFrame,
    time_column: str = "Time",
    target: str = "Class",
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> DataSplit:
    """Split by time so the model is always evaluated on transactions after its training data.

    A random split would let the model learn from transactions that happen after the ones it
    is tested on, which overstates performance compared with production.
    """
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must leave room for a test set")

    ordered = df.sort_values(time_column, kind="mergesort").reset_index(drop=True)
    n_rows = len(ordered)
    train_end = int(n_rows * train_fraction)
    validation_end = int(n_rows * (train_fraction + validation_fraction))

    split = DataSplit(
        train=ordered.iloc[:train_end],
        validation=ordered.iloc[train_end:validation_end],
        test=ordered.iloc[validation_end:],
    )
    for name, part in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        positives = int(part[target].sum())
        if positives == 0:
            raise ValueError(f"The {name} split contains no fraud cases; adjust the fractions")
        logger.info("%s: %d rows, %d frauds", name, len(part), positives)
    return split
