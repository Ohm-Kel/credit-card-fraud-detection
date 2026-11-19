"""Optional MLflow experiment tracking (see docs/adr/0010-mlflow-tracking.md).

When `tracking.enabled` is true in the config, each `fraud-train` run creates one parent MLflow
run with a nested run per model, holding its parameters, validation and test metrics, decision
thresholds and saved metadata. When disabled, every call is a no-op and MLflow is not imported.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

METRIC_KEYS = (
    "pr_auc",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "alerts",
    "expected_cost",
    "fraud_amount_caught_pct",
)


def _flatten(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested dicts into dotted keys, e.g. {"grid": {"C": [1]}} -> {"grid.C": "[1]"}."""
    flat: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(f"{name}.", value))
        else:
            flat[name] = value
    return flat


class Tracker:
    """Thin wrapper that logs to MLflow when enabled and does nothing otherwise."""

    def __init__(self, settings: dict[str, Any] | None) -> None:
        """Read the `tracking` section of the config."""
        settings = settings or {}
        self.enabled = bool(settings.get("enabled", False))
        self.uri = str(settings.get("uri", "mlflow.db"))
        self.experiment = str(settings.get("experiment", "fraud-detection"))
        self._mlflow: Any = None
        if self.enabled:
            try:
                import mlflow
            except ImportError as error:
                raise ImportError(
                    "Tracking is enabled but MLflow is missing: pip install -e '.[tracking]'"
                ) from error
            if "://" in self.uri:
                uri, artifacts = self.uri, None
            else:
                # A plain path means a local SQLite database; artifacts sit next to it.
                db = Path(self.uri).resolve()
                db.parent.mkdir(parents=True, exist_ok=True)
                uri = f"sqlite:///{db}"
                artifacts = (db.parent / "mlartifacts").as_uri()
            mlflow.set_tracking_uri(uri)
            if mlflow.get_experiment_by_name(self.experiment) is None:
                mlflow.create_experiment(self.experiment, artifact_location=artifacts)
            mlflow.set_experiment(self.experiment)
            self._mlflow = mlflow
            logger.info("MLflow tracking to %s (experiment '%s')", uri, self.experiment)

    @contextmanager
    def pipeline_run(self, params: dict[str, Any], tags: dict[str, Any]) -> Iterator[None]:
        """Parent run covering one execution of the training pipeline."""
        if not self.enabled:
            with nullcontext():
                yield
            return
        with self._mlflow.start_run(run_name="fraud-train"):
            self._mlflow.log_params({k: str(v)[:500] for k, v in _flatten("", params).items()})
            self._mlflow.set_tags({k: str(v) for k, v in tags.items()})
            yield

    def log_model_result(self, result: dict[str, Any]) -> None:
        """Nested run with one model's parameters, metrics, thresholds and metadata file."""
        if not self.enabled:
            return
        mlflow = self._mlflow
        with mlflow.start_run(run_name=result["model"], nested=True):
            mlflow.set_tag("model", result["model"])
            params = _flatten("", result.get("params", {}))
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            metrics: dict[str, float] = {"train_seconds": float(result["train_seconds"])}
            for split in ("validation", "test"):
                for key in METRIC_KEYS:
                    value = result[split].get(key)
                    if value is not None:
                        metrics[f"{split}_{key}"] = float(value)
            for key, value in result["policy"].items():
                if value is not None:
                    metrics[key] = float(value)
            mlflow.log_metrics(metrics)
            artifact = Path(result["artifact"]) / "metadata.json"
            if artifact.exists():
                mlflow.log_artifact(str(artifact))
