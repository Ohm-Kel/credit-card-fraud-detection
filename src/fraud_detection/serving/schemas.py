"""Request and response schemas for the scoring API (validated by Pydantic)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from fraud_detection.data.validate import PCA_FEATURES

_pca_fields = {
    name: (float, Field(..., description=f"Anonymised PCA component {name}"))
    for name in PCA_FEATURES
}

Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(extra="forbid"),
    Time=(float, Field(..., ge=0, description="Seconds since the first transaction")),
    Amount=(float, Field(..., ge=0, description="Transaction amount")),
    **_pca_fields,
)
Transaction.__doc__ = "One card transaction with the same columns as the training data."


class Reason(BaseModel):
    """A feature that pushed the fraud score up."""

    feature: str
    value: float = Field(description="Raw input value (time of day shown as an hour)")
    contribution: float = Field(description="SHAP contribution in log-odds")


class Prediction(BaseModel):
    """Fraud score, decision and explanation for one transaction."""

    fraud_probability: float = Field(ge=0, le=1)
    decision: Literal["approve", "review", "block"]
    review_threshold: float
    block_threshold: float | None = Field(
        description="None when no threshold reached the precision target (nothing is blocked)"
    )
    reasons: list[Reason] | None = Field(
        description="Top features pushing the score up; None if the model has no tree explainer"
    )
    model_name: str
    model_version: str


class BatchRequest(BaseModel):
    """Up to 1,000 transactions scored in one call."""

    model_config = ConfigDict(extra="forbid")

    transactions: list[Transaction] = Field(min_length=1, max_length=1000)  # type: ignore[valid-type]


class BatchPrediction(BaseModel):
    """Predictions in the same order as the request."""

    predictions: list[Prediction]


class Health(BaseModel):
    """Service status and the model it is serving."""

    status: Literal["ok"]
    model_name: str
    model_version: str
    explanations_available: bool
