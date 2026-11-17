"""Feature engineering, packaged as one scikit-learn transformer.

Preprocessing lives inside the model pipeline so training and serving apply exactly the same
transformations (see docs/adr/0002-single-pipeline-artifact.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

from fraud_detection.data.validate import PCA_FEATURES

SECONDS_PER_DAY = 24 * 60 * 60


def encode_time_of_day(times: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Turn seconds since the first transaction into a cyclical hour-of-day encoding.

    Sine and cosine keep 23:59 and 00:00 close together, which a raw hour number would not.
    """
    seconds = np.asarray(times, dtype=float).reshape(-1)
    angle = 2 * np.pi * (seconds % SECONDS_PER_DAY) / SECONDS_PER_DAY
    return np.column_stack([np.sin(angle), np.cos(angle)])


def time_feature_names(transformer: object, input_features: list[str]) -> list[str]:
    """Output names for the time-of-day encoder."""
    return ["time_of_day_sin", "time_of_day_cos"]


def build_preprocessor(
    time_column: str = "Time", amount_column: str = "Amount"
) -> ColumnTransformer:
    """Build the column transformer applied before every model.

    - Time: cyclical hour-of-day features (the raw counter only grows, so it would not
      generalise to later periods).
    - Amount: log1p then robust scaling, because amounts are heavily right-skewed.
    - V1 to V28: standard scaling (PCA outputs, already centred but not unit variance).
    """
    amount_pipeline = Pipeline(
        steps=[
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", RobustScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            (
                "time",
                FunctionTransformer(encode_time_of_day, feature_names_out=time_feature_names),
                [time_column],
            ),
            ("amount", amount_pipeline, [amount_column]),
            ("pca", StandardScaler(), PCA_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_columns(time_column: str = "Time", amount_column: str = "Amount") -> list[str]:
    """Raw input columns the model expects, in order."""
    return [time_column, *PCA_FEATURES, amount_column]
