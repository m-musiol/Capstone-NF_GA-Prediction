"""garev — Google Analytics Customer Revenue Prediction.

A modern, two-stage gradient-boosting pipeline for the Kaggle GStore competition.
See the README for the methodology and quick start.
"""

from __future__ import annotations

from garev.config import ModelConfig, PipelineConfig, TimeframeConfig
from garev.pipeline import TrainedModel, load_sessions, make_timeframe, predict_submission, train

__version__ = "1.0.0"

__all__ = [
    "ModelConfig",
    "PipelineConfig",
    "TimeframeConfig",
    "TrainedModel",
    "load_sessions",
    "make_timeframe",
    "predict_submission",
    "train",
]
