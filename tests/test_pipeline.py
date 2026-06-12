"""End-to-end tests: the full pipeline trains and scores without external data."""

from __future__ import annotations

import numpy as np

from garev.config import PipelineConfig
from garev.models.evaluate import evaluate, feature_importance
from garev.pipeline import make_timeframe, predict_submission, train


def test_train_and_predict_end_to_end(sessions):
    dataset = make_timeframe(sessions)
    trained = train(dataset)

    submission = predict_submission(trained, sessions)
    assert list(submission.columns) == ["fullVisitorId", "PredictedLogRevenue"]
    assert len(submission) > 0
    assert submission["PredictedLogRevenue"].notna().all()
    # Expected log-revenue is non-negative by construction.
    assert (submission["PredictedLogRevenue"] >= 0).all()
    # IDs must remain zero-padded strings for Kaggle scoring.
    assert submission["fullVisitorId"].str.len().gt(0).all()


def test_evaluation_report_is_well_formed(sessions):
    dataset = make_timeframe(sessions)
    trained = train(dataset)
    aligned = trained._align(dataset.features)
    report = evaluate(trained.model, aligned, dataset.is_buyer, dataset.log_revenue)
    assert report.n_visitors == len(dataset.features)
    assert 0.0 <= report.classifier_auc <= 1.0
    assert report.revenue_rmse >= 0.0


def test_feature_importance_is_ranked(sessions):
    dataset = make_timeframe(sessions)
    trained = train(dataset)
    importance = feature_importance(trained.model, top_n=5)
    assert len(importance) <= 5
    gains = importance["gain"].to_numpy()
    assert np.all(np.diff(gains) <= 0)  # sorted descending


def test_config_is_immutable():
    config = PipelineConfig()
    try:
        config.tuning_trials = 5  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("PipelineConfig should be frozen")
