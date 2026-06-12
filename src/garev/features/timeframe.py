"""Turn raw sessions into a per-visitor, forward-looking learning problem.

This is a cleaned, vectorised rewrite of the notebook's ``getTimeFramewithFeatures``.
The framing is unchanged because it is the right one for the competition: aggregate
each visitor's behaviour over a *feature window*, then label them with the revenue
they generate in a later, disjoint *prediction window*. Most visitors never return,
so the target is overwhelmingly zero — exactly what the two-stage model is built for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from garev.config import TimeframeConfig
from garev.data import schema

# Per-visitor aggregations applied to numeric session fields when present.
_NUMERIC_AGGS = {
    "totals_pageviews": ["sum", "mean", "max", "min"],
    "totals_hits": ["sum", "mean", "max", "min"],
    "totals_timeOnSite": ["sum", "mean", "max"],
    "totals_bounces": ["sum"],
    "totals_newVisits": ["sum"],
    "totals_sessionQualityDim": ["max", "mean"],
    "totals_transactions": ["sum"],
}


@dataclass(frozen=True)
class TimeframeDataset:
    """Feature matrix plus both target views for one timeframe."""

    features: pd.DataFrame  # per-visitor features, indexed by fullVisitorId
    log_revenue: pd.Series  # regression target: log1p of future revenue
    is_buyer: pd.Series  # classification target: returned and bought


def _revenue_units(sessions: pd.DataFrame) -> pd.Series:
    """Convert micro-unit revenue to currency units (0 where missing)."""
    micros = sessions.get(schema.REVENUE_COLUMN)
    if micros is None:
        return pd.Series(0.0, index=sessions.index)
    return micros.fillna(0) / schema.MICROS_PER_UNIT


def aggregate_features(sessions: pd.DataFrame, reference_end: pd.Timestamp) -> pd.DataFrame:
    """Aggregate sessions to one row per visitor.

    ``reference_end`` anchors the recency features so the same logic serves both
    the training windows and the final inference set.
    """
    sessions = sessions.copy()
    sessions["_revenue"] = _revenue_units(sessions)

    available = {c: a for c, a in _NUMERIC_AGGS.items() if c in sessions.columns}
    grouped = sessions.groupby(schema.VISITOR_ID)
    features = grouped.agg(available)
    features.columns = [f"{col}_{stat}" for col, stat in features.columns]

    # Behavioural summaries that don't fit the generic numeric agg.
    features["n_sessions"] = grouped.size()
    features["visit_number_max"] = grouped["visitNumber"].max()
    features["revenue_window_log"] = np.log1p(grouped["_revenue"].sum())

    # Recency / tenure, expressed in whole days relative to the window edges.
    first = grouped["visitStartTime"].min()
    last = grouped["visitStartTime"].max()
    window_start = sessions["visitStartTime"].min()
    features["days_since_first"] = (reference_end - first).dt.days
    features["days_since_last"] = (reference_end - last).dt.days
    features["active_span_days"] = (last - first).dt.days
    features["days_from_start"] = (first - window_start).dt.days

    # Most recent categorical value per visitor (forward-fill of identity).
    ordered = sessions.sort_values("visitStartTime")
    for col in schema.CATEGORICAL_FEATURES:
        if col in sessions.columns:
            features[col] = ordered.groupby(schema.VISITOR_ID)[col].last()

    return features.fillna(0)


def _future_revenue(sessions: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Sum each visitor's revenue (units) within the half-open window [start, end)."""
    mask = (sessions["visitStartTime"] >= start) & (sessions["visitStartTime"] < end)
    window = sessions.loc[mask].copy()
    window["_revenue"] = _revenue_units(window)
    return window.groupby(schema.VISITOR_ID)["_revenue"].sum()


def build_timeframe(
    sessions: pd.DataFrame,
    window_start: pd.Timestamp,
    config: TimeframeConfig,
) -> TimeframeDataset:
    """Build one (features, targets) timeframe starting at ``window_start``."""
    feature_end = window_start + pd.Timedelta(days=config.feature_days)
    pred_start = feature_end + pd.Timedelta(days=config.gap_days)
    pred_end = pred_start + pd.Timedelta(days=config.prediction_days)

    feature_mask = (sessions["visitStartTime"] >= window_start) & (
        sessions["visitStartTime"] < feature_end
    )
    feature_sessions = sessions.loc[feature_mask]
    if feature_sessions.empty:
        raise ValueError(f"No sessions in feature window starting {window_start.date()}")

    features = aggregate_features(feature_sessions, reference_end=feature_end)

    future = _future_revenue(sessions, pred_start, pred_end)
    revenue = future.reindex(features.index).fillna(0.0)
    log_revenue = np.log1p(revenue)
    is_buyer = (revenue > 0).astype(int)

    return TimeframeDataset(features=features, log_revenue=log_revenue, is_buyer=is_buyer)
