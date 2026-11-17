"""Loading the raw transactions dataset."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

KAGGLE_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Read the transactions CSV, failing with a clear message if it is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download creditcard.csv from {KAGGLE_URL} "
            "and place it in the data/ folder."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %d rows and %d columns from %s", len(df), df.shape[1], path)
    return df


def file_fingerprint(path: str | Path, length: int = 12) -> str:
    """Return a short SHA-256 hash of a file, used to version the training data."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]
