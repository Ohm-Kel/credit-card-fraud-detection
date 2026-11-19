"""Synthetic transactions with the Kaggle schema, for tests, CI and demos.

The real dataset is not redistributed in this repository, so anything that must run without it
(tests, the Docker smoke test, a quick API demo) uses this generator instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.data.validate import PCA_FEATURES


def make_transactions(n_rows: int = 6000, fraud_rate: float = 0.03, seed: int = 0) -> pd.DataFrame:
    """Generate transactions where frauds differ on V1 to V4, so models can learn a signal."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n_rows) < fraud_rate).astype(int)
    features = rng.normal(size=(n_rows, len(PCA_FEATURES)))
    features[y == 1, :4] += 2.5
    df = pd.DataFrame(features, columns=PCA_FEATURES)
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_800, n_rows)))
    df["Amount"] = np.round(rng.lognormal(3, 1.2, n_rows), 2)
    df["Class"] = y
    return df
