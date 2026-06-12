"""Consistent categorical encoding.

This fixes the central modelling bug in the original notebook, where a fresh
``LabelEncoder`` was fitted *separately* on the train set and the test set. That
gives the same string different integer codes in each split, so the model learns
one mapping and is scored against another. Here a single encoder is fitted on the
training data and reused for every later transform; categories unseen at fit time
map to a dedicated ``-1`` code instead of crashing.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


class CategoricalEncoder:
    """Fit-once ordinal encoder over a fixed list of categorical columns."""

    def __init__(self, columns: list[str]):
        self.columns = list(columns)
        self._encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        self._fitted_columns: list[str] = []

    def fit(self, frame: pd.DataFrame) -> CategoricalEncoder:
        self._fitted_columns = [c for c in self.columns if c in frame.columns]
        if self._fitted_columns:
            self._encoder.fit(frame[self._fitted_columns].astype(str))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with the categorical columns replaced by integer codes."""
        frame = frame.copy()
        if self._fitted_columns:
            encoded = self._encoder.transform(frame[self._fitted_columns].astype(str))
            frame[self._fitted_columns] = encoded.astype("int32")
        return frame

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)
