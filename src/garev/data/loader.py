"""Load the raw GA export and flatten its JSON columns.

This replaces the original notebook's hand-rolled recursive JSON parser (which
relied on brittle ``str.replace`` chains) with ``json.loads`` plus
``pandas.json_normalize`` — the standard, well-tested way to flatten nested JSON.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from garev.data import schema


def _flatten_json_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Expand one JSON-string column into ``{column}_{key}`` columns."""
    parsed = frame[column].apply(json.loads)
    flat = pd.json_normalize(parsed)
    flat.columns = [f"{column}_{sub}" for sub in flat.columns]
    flat.index = frame.index
    return flat


def _flatten_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Flatten every JSON column in a chunk and drop the raw originals."""
    present = [c for c in schema.JSON_COLUMNS if c in chunk.columns]
    expanded = [_flatten_json_column(chunk, col) for col in present]
    base = chunk.drop(columns=present)
    return pd.concat([base, *expanded], axis=1)


def _coerce_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric ``totals_*`` fields and parse the calendar date.

    ``fullVisitorId`` is forced to string because its leading zeros are
    significant for Kaggle scoring and must never be lost to integer casting.
    """
    frame = frame.copy()
    frame[schema.VISITOR_ID] = frame[schema.VISITOR_ID].astype(str)
    for col in schema.NUMERIC_TOTALS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    if "visitStartTime" in frame.columns:
        frame["visitStartTime"] = pd.to_datetime(frame["visitStartTime"], unit="s")
    return frame


def _read_chunks(path: Path, chunksize: int | None, nrows: int | None) -> Iterator[pd.DataFrame]:
    """Yield raw CSV chunks, keeping ``fullVisitorId`` as string on read."""
    reader = pd.read_csv(
        path,
        dtype={schema.VISITOR_ID: str},
        chunksize=chunksize,
        nrows=nrows,
    )
    if chunksize is None:
        yield reader  # ``read_csv`` returned a single DataFrame.
    else:
        yield from reader


def load_raw(
    path: str | Path,
    chunksize: int | None = 100_000,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load a raw GA CSV into a flat, typed DataFrame.

    Args:
        path: Path to ``train_v2.csv`` / ``test_v2.csv`` or a sample CSV.
        chunksize: Rows per chunk. Chunked reading keeps the 24 GB file within
            memory; pass ``None`` to read small files in one shot.
        nrows: Optional cap on rows read, handy for quick experiments.

    Returns:
        A DataFrame with JSON columns flattened and dtypes coerced.
    """
    path = Path(path)
    flattened = [_flatten_chunk(chunk) for chunk in _read_chunks(path, chunksize, nrows)]
    combined = pd.concat(flattened, ignore_index=True)
    return _coerce_types(combined)
