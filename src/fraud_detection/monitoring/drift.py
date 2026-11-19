"""Data and score drift with the Population Stability Index (see docs/adr/0009-monitoring.md).

PSI compares the distribution of a variable in a reference period (training data) with a
current period (live traffic):

    PSI = sum over bins of (current% - reference%) * ln(current% / reference%)

Common reading, used in credit risk: below 0.1 stable, 0.1 to 0.25 moderate shift,
above 0.25 significant shift that warrants investigation or retraining.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STABLE, MODERATE, SIGNIFICANT = "stable", "moderate", "significant"
MODERATE_PSI, SIGNIFICANT_PSI = 0.1, 0.25
_EPSILON = 1e-6


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute the PSI between two samples, with bins from reference quantiles."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("Both samples must be non-empty")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # constant reference: compare share equal to that constant
        same_ref = 1.0
        same_cur = float(np.mean(current == reference[0]))
        ref_pct = np.array([same_ref, _EPSILON])
        cur_pct = np.clip(np.array([same_cur, 1 - same_cur]), _EPSILON, None)
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct = np.clip(ref_pct, _EPSILON, None)
    cur_pct = np.clip(cur_pct, _EPSILON, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_status(value: float) -> str:
    """Label a PSI value as stable, moderate or significant."""
    if value >= SIGNIFICANT_PSI:
        return SIGNIFICANT
    if value >= MODERATE_PSI:
        return MODERATE
    return STABLE


def feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], bins: int = 10
) -> pd.DataFrame:
    """PSI and status for each column, most drifted first."""
    rows = []
    for column in columns:
        value = psi(reference[column].to_numpy(), current[column].to_numpy(), bins)
        rows.append({"feature": column, "psi": value, "status": drift_status(value)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
