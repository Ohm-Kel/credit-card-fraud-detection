# ADR 0008: FastAPI scoring service

- Status: accepted
- Date: 2026-11-19

## Context

Phase 2 turns a trained model into a service that another system (a payment flow or an
analyst tool) can call per transaction, returning an action and its reasons.

## Decision

- FastAPI with Pydantic schemas: inputs are validated (all 30 features required, `Time` and
  `Amount` non-negative, unknown fields rejected), and OpenAPI docs are generated at `/docs`.
- Endpoints: `GET /health` (also reports the model name and version), `POST /predict` and
  `POST /predict/batch` (up to 1,000 transactions).
- The service loads one saved artifact (pipeline + thresholds, ADR 0002) at startup, so a broken
  model fails at deploy time rather than on the first request. The model is chosen by
  `FRAUD_MODEL_DIR`, or the latest version of `FRAUD_MODEL_NAME`.
- Each response carries the fraud probability, the approve / review / block decision from the
  stored policy (ADR 0003), the thresholds used, the model version (for audit) and, for tree
  models, the top three SHAP reasons. The tree explainer is built once at startup.
- Models are mounted into the container read-only, not baked into the image, so a new model
  does not require a new image.
- CI trains a demo model on synthetic data, builds the image with Docker Compose and scores an
  example transaction end to end.

## Consequences

- Latency on CPU, measured locally over 200 requests including SHAP reasons: p50 10.8 ms,
  p95 13.5 ms, p99 15.5 ms.
- Reasons are only returned for XGBoost; other models are served without them, because their
  explainers need background data or are too slow per request.
- The image does not include TensorFlow, so the Keras model is not servable from it.
- No authentication or rate limiting yet; both are required before any real deployment.
