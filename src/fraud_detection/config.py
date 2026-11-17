"""Typed loading of the YAML training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Where the data lives and which columns have special roles."""

    path: str
    target: str = "Class"
    time_column: str = "Time"
    amount_column: str = "Amount"


@dataclass(frozen=True)
class SplitConfig:
    """Fractions for the time-ordered train / validation / test split."""

    train_fraction: float = 0.6
    validation_fraction: float = 0.2


@dataclass(frozen=True)
class CostConfig:
    """Business costs used to choose the decision threshold."""

    review_cost: float = 5.0
    min_fraud_cost: float = 10.0


@dataclass(frozen=True)
class DecisionConfig:
    """Settings for the approve / review / block policy."""

    block_precision_target: float = 0.9


@dataclass(frozen=True)
class OutputConfig:
    """Folders for trained models and evaluation reports."""

    models_dir: str = "models"
    reports_dir: str = "reports"


@dataclass(frozen=True)
class TrainConfig:
    """Full training configuration."""

    data: DataConfig
    seed: int = 42
    split: SplitConfig = field(default_factory=SplitConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    output: OutputConfig = field(default_factory=OutputConfig)
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


def load_config(path: str | Path) -> TrainConfig:
    """Load and validate a training configuration from a YAML file."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if "data" not in raw:
        raise ValueError(f"Config {path} is missing the required 'data' section")
    if not raw.get("models"):
        raise ValueError(f"Config {path} must define at least one model under 'models'")
    split = SplitConfig(**raw.get("split", {}))
    if (
        split.train_fraction <= 0
        or split.validation_fraction <= 0
        or split.train_fraction + split.validation_fraction >= 1
    ):
        raise ValueError("Split fractions must be positive and leave room for a test set")
    return TrainConfig(
        data=DataConfig(**raw["data"]),
        seed=int(raw.get("seed", 42)),
        split=split,
        costs=CostConfig(**raw.get("costs", {})),
        decision=DecisionConfig(**raw.get("decision", {})),
        models=dict(raw["models"]),
        output=OutputConfig(**raw.get("output", {})),
        raw=raw,
    )
