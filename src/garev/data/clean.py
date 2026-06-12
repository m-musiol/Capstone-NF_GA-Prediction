"""Cleaning steps shared by training and inference.

Two ideas from the original notebook are kept because they are sound — dropping
constant columns and filling structural missing values — but they are rewritten
as small, pure functions instead of inline cell mutations.
"""

from __future__ import annotations

import pandas as pd

from garev.data import schema

# Fields whose missingness is structural rather than random. Revenue and
# transaction counts are absent precisely when nothing was bought, so zero is the
# correct fill; the engagement flags default to ``0`` for the same reason.
_ZERO_FILL = (
    "totals_transactionRevenue",
    "totals_transactions",
    "totals_bounces",
    "totals_newVisits",
)


def drop_constant_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that carry a single value (incl. all-NaN) — they add no signal."""
    nunique = frame.nunique(dropna=False)
    constant = nunique[nunique <= 1].index.tolist()
    return frame.drop(columns=constant)


def fill_structural_missing(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values whose absence has a known meaning."""
    frame = frame.copy()
    for col in _ZERO_FILL:
        if col in frame.columns:
            frame[col] = frame[col].fillna(0)
    # ``pageviews`` is occasionally null while ``hits`` is present; hits is the
    # better proxy than zero, matching the original notebook's intent.
    if {"totals_pageviews", "totals_hits"}.issubset(frame.columns):
        frame["totals_pageviews"] = frame["totals_pageviews"].fillna(frame["totals_hits"])
    # Remaining categoricals get an explicit sentinel so encoders treat "unknown"
    # as its own category instead of dropping the row.
    for col in schema.CATEGORICAL_FEATURES:
        if col in frame.columns:
            frame[col] = frame[col].fillna("(missing)").astype(str)
    return frame


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning sequence."""
    return fill_structural_missing(drop_constant_columns(frame))
