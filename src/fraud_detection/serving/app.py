"""FastAPI scoring service (see docs/adr/0008-scoring-api.md).

Run locally with:  uvicorn fraud_detection.serving.app:app
Configuration (environment variables):
- FRAUD_MODEL_DIR: a specific saved model folder, or
- FRAUD_MODELS_DIR (default "models") and FRAUD_MODEL_NAME (default "xgboost"): the latest
  saved version of that model is served.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from sklearn.pipeline import Pipeline

from fraud_detection.evaluation.explain import TreeReasonExplainer, top_reasons
from fraud_detection.evaluation.threshold import DecisionPolicy
from fraud_detection.features.build import feature_columns
from fraud_detection.models.factory import predict_scores
from fraud_detection.models.registry import latest_model, load_model
from fraud_detection.serving.schemas import (
    BatchPrediction,
    BatchRequest,
    Health,
    Prediction,
    Reason,
    Transaction,
)

logger = logging.getLogger(__name__)
COLUMNS = feature_columns()


@dataclass
class ServedModel:
    """Everything the endpoints need, loaded once at startup."""

    model: Pipeline
    policy: DecisionPolicy
    name: str
    version: str
    explainer: TreeReasonExplainer | None


def resolve_model_dir() -> Path:
    """Model folder from FRAUD_MODEL_DIR, or the latest version of FRAUD_MODEL_NAME."""
    explicit = os.environ.get("FRAUD_MODEL_DIR")
    if explicit:
        return Path(explicit)
    return latest_model(
        os.environ.get("FRAUD_MODELS_DIR", "models"),
        os.environ.get("FRAUD_MODEL_NAME", "xgboost"),
    )


def load_served_model(folder: Path) -> ServedModel:
    """Load the pipeline, its decision thresholds and, for tree models, a SHAP explainer."""
    model, metadata = load_model(folder)
    policy_data = metadata["policy"]
    block = policy_data.get("block_threshold")
    policy = DecisionPolicy(
        review_threshold=float(policy_data["review_threshold"]),
        block_threshold=float("inf") if block is None else float(block),
    )
    try:
        explainer: TreeReasonExplainer | None = TreeReasonExplainer(model)
    except (ValueError, ImportError) as error:
        logger.warning("Explanations disabled: %s", error)
        explainer = None
    logger.info("Serving %s version %s from %s", metadata["name"], metadata["version"], folder)
    return ServedModel(model, policy, metadata["name"], metadata["version"], explainer)


def score(served: ServedModel, transactions: list[Any]) -> list[Prediction]:
    """Score transactions and build one Prediction each."""
    frame = pd.DataFrame([t.model_dump() for t in transactions])[COLUMNS]
    probabilities = predict_scores(served.model, frame)
    decisions = served.policy.decide(probabilities)
    explanation = served.explainer.explain(frame) if served.explainer else None
    policy = served.policy.to_dict()

    predictions = []
    for i, probability in enumerate(probabilities):
        reasons = (
            [Reason(**r) for r in top_reasons(explanation, i, frame.iloc[i])]
            if explanation is not None
            else None
        )
        predictions.append(
            Prediction(
                fraud_probability=float(np.clip(probability, 0.0, 1.0)),
                decision=str(decisions[i]),
                review_threshold=policy["review_threshold"],
                block_threshold=policy["block_threshold"],
                reasons=reasons,
                model_name=served.name,
                model_version=served.version,
            )
        )
    return predictions


def create_app(model_dir: Path | None = None) -> FastAPI:
    """Build the app; the model is loaded at startup so a bad model fails fast."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.served = load_served_model(model_dir or resolve_model_dir())
        yield

    app = FastAPI(
        title="Credit Card Fraud Scoring API",
        version="0.2.0",
        description="Scores card transactions and returns approve / review / block with reasons.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        """Liveness check that also reports which model is loaded."""
        served: ServedModel = request.app.state.served
        return Health(
            status="ok",
            model_name=served.name,
            model_version=served.version,
            explanations_available=served.explainer is not None,
        )

    @app.post("/predict", response_model=Prediction)
    def predict(transaction: Transaction, request: Request) -> Prediction:  # type: ignore[valid-type]
        """Score one transaction."""
        return score(request.app.state.served, [transaction])[0]

    @app.post("/predict/batch", response_model=BatchPrediction)
    def predict_batch(batch: BatchRequest, request: Request) -> BatchPrediction:
        """Score up to 1,000 transactions in one call."""
        return BatchPrediction(predictions=score(request.app.state.served, batch.transactions))

    return app


app = create_app()
