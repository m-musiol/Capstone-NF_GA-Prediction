"""End-to-end orchestration: raw CSV -> trained model -> Kaggle submission.

This module wires the stages together so a full run is three function calls
(``load_sessions`` -> ``train`` -> ``predict_submission``) instead of dozens of
notebook cells with CSV round-trips in between.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from garev.config import PipelineConfig
from garev.data import clean, schema
from garev.data.loader import load_raw
from garev.features.encoders import CategoricalEncoder
from garev.features.timeframe import TimeframeDataset, aggregate_features, build_timeframe
from garev.models.two_stage import TwoStageRevenueModel


@dataclass
class TrainedModel:
    """A fitted model plus everything needed to score new visitors consistently."""

    model: TwoStageRevenueModel
    encoder: CategoricalEncoder
    feature_columns: list[str]

    def _align(self, features: pd.DataFrame) -> pd.DataFrame:
        """Encode and reindex features to match the training layout exactly."""
        encoded = self.encoder.transform(features)
        return encoded.reindex(columns=self.feature_columns, fill_value=0)


def load_sessions(
    path: str | Path,
    config: PipelineConfig | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load and clean a raw GA CSV into model-ready session rows."""
    config = config or PipelineConfig()
    raw = load_raw(path, nrows=nrows)
    return clean.clean(raw)


def make_timeframe(
    sessions: pd.DataFrame,
    config: PipelineConfig | None = None,
    window_start: pd.Timestamp | None = None,
) -> TimeframeDataset:
    """Build the training timeframe, defaulting the window to the earliest date."""
    config = config or PipelineConfig()
    start = window_start or sessions["visitStartTime"].min().normalize()
    return build_timeframe(sessions, start, config.timeframe)


def train(dataset: TimeframeDataset, config: PipelineConfig | None = None) -> TrainedModel:
    """Encode features and fit the two-stage model on a prepared timeframe."""
    config = config or PipelineConfig()
    categoricals = [c for c in schema.CATEGORICAL_FEATURES if c in dataset.features.columns]
    encoder = CategoricalEncoder(categoricals).fit(dataset.features)
    encoded = encoder.transform(dataset.features)

    model = TwoStageRevenueModel(config.model)
    model.fit(encoded, dataset.is_buyer, dataset.log_revenue)
    return TrainedModel(model=model, encoder=encoder, feature_columns=list(encoded.columns))


def predict_submission(trained: TrainedModel, sessions: pd.DataFrame) -> pd.DataFrame:
    """Score every visitor and return a Kaggle-format submission frame.

    Unlike training, inference uses *all* of a visitor's sessions and has no future
    window — the model extrapolates expected log-revenue from past behaviour.
    """
    reference_end = sessions["visitStartTime"].max()
    features = aggregate_features(sessions, reference_end=reference_end)
    aligned = trained._align(features)

    submission = pd.DataFrame({schema.VISITOR_ID: features.index.astype(str)})
    submission["PredictedLogRevenue"] = trained.model.predict_log_revenue(aligned)
    return submission.reset_index(drop=True)
