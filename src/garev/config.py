"""Central, typed configuration for the whole pipeline.

Everything tunable lives here as frozen dataclasses, so a run is fully described
by a single ``PipelineConfig`` instance instead of magic numbers scattered across
notebook cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Project root resolved relative to this file (…/src/garev/config.py -> project).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR: Path = PROJECT_ROOT / "data"
DEFAULT_ARTIFACT_DIR: Path = PROJECT_ROOT / "artifacts"


@dataclass(frozen=True)
class TimeframeConfig:
    """Defines the sliding window used to turn sessions into a learning problem.

    The task is forward-looking: using a *feature window* of past activity we
    predict revenue in a disjoint *prediction window* that starts after a gap.
    The defaults mirror the competition framing (168 / 46 / 62 days).
    """

    feature_days: int = 168
    gap_days: int = 46
    prediction_days: int = 62


@dataclass(frozen=True)
class ModelConfig:
    """LightGBM settings shared by the classifier and regressor stages."""

    random_state: int = 13
    n_estimators: int = 1000
    learning_rate: float = 0.03
    num_leaves: int = 63
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    # Fraction of the training rows held out for early-stopping validation.
    valid_fraction: float = 0.2


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration aggregating every stage."""

    data_dir: Path = DEFAULT_DATA_DIR
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    timeframe: TimeframeConfig = field(default_factory=TimeframeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    # Optional Optuna budget; ``0`` disables tuning and uses ``ModelConfig``.
    tuning_trials: int = 0
