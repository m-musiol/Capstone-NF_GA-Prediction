"""Tests for categorical encoding and timeframe construction."""

from __future__ import annotations

import pandas as pd

from garev.config import TimeframeConfig
from garev.features.encoders import CategoricalEncoder
from garev.features.timeframe import build_timeframe


def test_encoder_is_consistent_across_splits():
    train = pd.DataFrame({"c": ["a", "b", "a", "c"]})
    test = pd.DataFrame({"c": ["b", "a"]})
    encoder = CategoricalEncoder(["c"]).fit(train)
    train_codes = encoder.transform(train)["c"].tolist()
    test_codes = encoder.transform(test)["c"].tolist()
    # "b" and "a" must map to the same codes in both splits — the bug we fixed.
    assert test_codes[0] == train_codes[1]  # "b"
    assert test_codes[1] == train_codes[0]  # "a"


def test_encoder_maps_unseen_category_to_sentinel():
    encoder = CategoricalEncoder(["c"]).fit(pd.DataFrame({"c": ["a", "b"]}))
    codes = encoder.transform(pd.DataFrame({"c": ["zzz"]}))["c"].tolist()
    assert codes == [-1]


def test_build_timeframe_aligns_features_and_targets(sessions):
    start = sessions["visitStartTime"].min().normalize()
    dataset = build_timeframe(sessions, start, TimeframeConfig())
    assert len(dataset.features) == len(dataset.log_revenue) == len(dataset.is_buyer)
    assert dataset.features.index.equals(dataset.log_revenue.index)
    # The synthetic sample is sized so the prediction window holds buyers.
    assert dataset.is_buyer.sum() >= 2
    assert not dataset.features.isna().any().any()


def test_build_timeframe_has_recency_features(sessions):
    start = sessions["visitStartTime"].min().normalize()
    dataset = build_timeframe(sessions, start, TimeframeConfig())
    for column in ("days_since_first", "days_since_last", "n_sessions"):
        assert column in dataset.features.columns
