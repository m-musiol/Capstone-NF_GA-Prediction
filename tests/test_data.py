"""Tests for sample generation, loading and cleaning."""

from __future__ import annotations

import json

import pandas as pd

from garev.data import schema
from garev.data.clean import clean, drop_constant_columns
from garev.data.loader import load_raw
from garev.data.sample import generate_sample


def test_generate_sample_has_raw_schema():
    frame = generate_sample(n_visitors=50, seed=1)
    for column in schema.ID_COLUMNS + schema.JSON_COLUMNS:
        assert column in frame.columns
    # JSON columns must be valid JSON documents.
    assert json.loads(frame["device"].iloc[0])["browser"]


def test_visitor_ids_keep_leading_zeros():
    frame = generate_sample(n_visitors=10, seed=1)
    assert frame["fullVisitorId"].str.startswith("0").any()


def test_load_raw_flattens_json(sample_csv):
    frame = load_raw(sample_csv, chunksize=None)
    # Raw JSON columns are gone, flattened children are present.
    assert not set(schema.JSON_COLUMNS).intersection(frame.columns)
    assert "device_browser" in frame.columns
    assert "totals_transactionRevenue" in frame.columns
    assert frame["fullVisitorId"].dtype == object
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])


def test_load_raw_chunked_matches_single(sample_csv):
    single = load_raw(sample_csv, chunksize=None)
    chunked = load_raw(sample_csv, chunksize=137)
    assert len(single) == len(chunked)
    assert set(single.columns) == set(chunked.columns)


def test_drop_constant_columns():
    frame = pd.DataFrame({"keep": [1, 2, 3], "drop": ["x", "x", "x"]})
    result = drop_constant_columns(frame)
    assert "keep" in result.columns
    assert "drop" not in result.columns


def test_clean_fills_revenue_with_zero(sample_csv):
    raw = load_raw(sample_csv, chunksize=None)
    cleaned = clean(raw)
    assert cleaned["totals_transactionRevenue"].isna().sum() == 0
