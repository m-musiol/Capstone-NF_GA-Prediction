"""Shared fixtures: a synthetic raw CSV and cleaned sessions for the suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from garev.data.sample import write_sample
from garev.pipeline import load_sessions

# Enough visitors that the forward-looking prediction window reliably contains
# return-buyers, so the classifier and regressor both see positive examples.
SAMPLE_VISITORS = 1500
SAMPLE_SEED = 13


@pytest.fixture(scope="session")
def sample_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic raw GA CSV written once per test session."""
    path = tmp_path_factory.mktemp("data") / "sample.csv"
    return write_sample(path, n_visitors=SAMPLE_VISITORS, seed=SAMPLE_SEED)


@pytest.fixture(scope="session")
def sessions(sample_csv: Path) -> pd.DataFrame:
    """Loaded and cleaned sessions from the synthetic CSV."""
    return load_sessions(sample_csv)
