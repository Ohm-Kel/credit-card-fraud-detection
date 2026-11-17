"""Saving and loading versioned model artifacts.

Each artifact folder holds the fitted pipeline, its decision thresholds and the metadata
needed to reproduce it (config, data hash, git commit, metrics).
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


def current_git_commit() -> str | None:
    """Return the current git commit hash, or None outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def save_model(
    model: Pipeline, name: str, metadata: dict[str, Any], models_dir: str | Path
) -> Path:
    """Save a model and its metadata under models_dir/name/version/ and return the folder."""
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(models_dir) / name / version
    folder.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, folder / MODEL_FILE)
    full_metadata = {
        "name": name,
        "version": version,
        "git_commit": current_git_commit(),
        **metadata,
    }
    (folder / METADATA_FILE).write_text(json.dumps(full_metadata, indent=2), encoding="utf-8")
    return folder


def load_model(folder: str | Path) -> tuple[Pipeline, dict[str, Any]]:
    """Load a saved model and its metadata."""
    folder = Path(folder)
    model = joblib.load(folder / MODEL_FILE)
    metadata = json.loads((folder / METADATA_FILE).read_text(encoding="utf-8"))
    return model, metadata
