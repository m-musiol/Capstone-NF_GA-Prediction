"""Synthetic Google Analytics data generator.

The real Kaggle files are 24 GB + 7 GB and require a manual download, which makes
the pipeline impossible to run, test or demo out of the box. This module emits a
*small* CSV with the exact same shape — including the four JSON columns — so the
entire pipeline, and the test suite, run end to end without any external data.

The synthetic process is deliberately simple but produces the structure the model
needs: many returning visitors, a small fraction of buyers, and revenue spread
across time so the forward-looking target window contains positive examples.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_BROWSERS = ("Chrome", "Safari", "Firefox", "Edge", "Internet Explorer")
_OS = ("Windows", "Macintosh", "Android", "iOS", "Linux")
_DEVICES = ("desktop", "mobile", "tablet")
_CONTINENTS = ("Americas", "Europe", "Asia", "Africa", "Oceania")
_COUNTRIES = ("United States", "Germany", "India", "Japan", "Brazil", "France")
_CHANNELS = ("Organic Search", "Direct", "Referral", "Paid Search", "Social", "Display")
_SOURCES = ("google", "(direct)", "youtube.com", "facebook.com", "bing")
_MEDIUMS = ("organic", "(none)", "referral", "cpc", "social")


def _rng_choice(rng: np.random.Generator, values: tuple[str, ...]) -> str:
    return str(rng.choice(np.asarray(values, dtype=object)))


def _session_row(rng: np.random.Generator, visitor_id: str, day: date, visit_no: int) -> dict:
    """Build a single session as a dict mirroring one raw export row."""
    # ~1.5% of sessions convert; revenue is heavy-tailed, stored in micro-units.
    buys = rng.random() < 0.015
    revenue_micros = int(rng.lognormal(mean=4.0, sigma=0.6) * 1_000_000) if buys else None

    totals = {
        "hits": str(int(rng.integers(1, 40))),
        "pageviews": str(int(rng.integers(1, 30))),
        "newVisits": "1" if visit_no == 1 else None,
        "bounces": "1" if rng.random() < 0.4 else None,
        "sessionQualityDim": str(int(rng.integers(1, 100))),
        "timeOnSite": str(int(rng.integers(0, 1800))),
        "transactions": "1" if buys else None,
        "transactionRevenue": str(revenue_micros) if buys else None,
    }
    device = {
        "browser": _rng_choice(rng, _BROWSERS),
        "operatingSystem": _rng_choice(rng, _OS),
        "deviceCategory": _rng_choice(rng, _DEVICES),
        "isMobile": bool(rng.random() < 0.4),
    }
    geo = {
        "continent": _rng_choice(rng, _CONTINENTS),
        "subContinent": "Northern America",
        "country": _rng_choice(rng, _COUNTRIES),
        "region": _rng_choice(rng, ("California", "Bavaria", "Tokyo", "not available")),
        "metro": "not available in demo dataset",
        "city": _rng_choice(rng, ("Mountain View", "Hamburg", "Tokyo", "not available")),
        "networkDomain": _rng_choice(rng, ("comcast.net", "t-online.de", "(not set)", "unknown")),
    }
    traffic = {
        "source": _rng_choice(rng, _SOURCES),
        "medium": _rng_choice(rng, _MEDIUMS),
        "campaign": "(not set)",
        "keyword": _rng_choice(rng, ("(not provided)", "merchandise", "store", "(not set)")),
    }
    # ``json.dumps`` with default separators reproduces the export's quoting,
    # which the loader parses back with ``json.loads`` (no fragile string hacks).
    return {
        "fullVisitorId": visitor_id,
        "visitId": int(rng.integers(1, 2**31)),
        "visitNumber": visit_no,
        "visitStartTime": int(pd.Timestamp(day).timestamp()) + int(rng.integers(0, 86_400)),
        "date": int(day.strftime("%Y%m%d")),
        "channelGrouping": _rng_choice(rng, _CHANNELS),
        "device": json.dumps(device),
        "geoNetwork": json.dumps(geo),
        "totals": json.dumps({k: v for k, v in totals.items() if v is not None}),
        "trafficSource": json.dumps(traffic),
    }


def generate_sample(
    n_visitors: int = 800,
    start: date = date(2017, 1, 1),
    span_days: int = 330,
    seed: int = 13,
) -> pd.DataFrame:
    """Generate a synthetic, raw-shaped GA dataset.

    Args:
        n_visitors: Number of distinct visitors.
        start: First calendar day of activity.
        span_days: Length of the activity window (defaults cover one timeframe).
        seed: Seed for reproducibility.

    Returns:
        A DataFrame with the same columns and JSON encoding as the raw export.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for visitor in range(n_visitors):
        visitor_id = f"{visitor:019d}"  # 19-digit zero-padded, like the real IDs.
        n_sessions = int(rng.integers(1, 8))
        offsets = sorted(int(o) for o in rng.integers(0, span_days, size=n_sessions))
        for visit_no, offset in enumerate(offsets, start=1):
            rows.append(_session_row(rng, visitor_id, start + timedelta(days=offset), visit_no))
    return pd.DataFrame(rows)


def write_sample(path: Path, **kwargs) -> Path:
    """Generate a sample dataset and write it as CSV at ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_sample(**kwargs).to_csv(path, index=False)
    return path
