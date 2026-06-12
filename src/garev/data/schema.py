"""Schema of the Google Analytics export.

The raw Kaggle files (``train_v2.csv`` / ``test_v2.csv``) ship four columns whose
values are JSON documents — ``device``, ``geoNetwork``, ``totals`` and
``trafficSource``. Flattening those four into ordinary columns is the first job
of the loader. This module is the single source of truth for that structure so
the loader, the sample generator and the tests never drift apart.
"""

from __future__ import annotations

from typing import Final

# Top-level columns that are themselves JSON objects and must be flattened.
JSON_COLUMNS: Final[tuple[str, ...]] = (
    "device",
    "geoNetwork",
    "totals",
    "trafficSource",
)

# Identifier and timing columns carried verbatim from the export.
ID_COLUMNS: Final[tuple[str, ...]] = (
    "fullVisitorId",
    "visitId",
    "visitNumber",
    "visitStartTime",
    "date",
    "channelGrouping",
)

# The unique key used by Kaggle for scoring and submissions. It must survive the
# whole pipeline as a string — leading zeros are significant.
VISITOR_ID: Final[str] = "fullVisitorId"

# Session-level revenue, in micro-units of currency (1,000,000 == 1.0).
REVENUE_COLUMN: Final[str] = "totals_transactionRevenue"
MICROS_PER_UNIT: Final[int] = 1_000_000

# Numeric ``totals_*`` fields that arrive as strings and need coercion.
NUMERIC_TOTALS: Final[tuple[str, ...]] = (
    "totals_hits",
    "totals_pageviews",
    "totals_bounces",
    "totals_newVisits",
    "totals_sessionQualityDim",
    "totals_timeOnSite",
    "totals_transactions",
    "totals_transactionRevenue",
)

# Categorical fields kept for modelling. These are the high-signal, low-noise
# columns that consistently surfaced in strong public solutions; the hits-level
# fields from the original notebook were dropped on purpose (see README).
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "channelGrouping",
    "device_browser",
    "device_operatingSystem",
    "device_deviceCategory",
    "geoNetwork_continent",
    "geoNetwork_subContinent",
    "geoNetwork_country",
    "geoNetwork_region",
    "geoNetwork_metro",
    "geoNetwork_city",
    "geoNetwork_networkDomain",
    "trafficSource_source",
    "trafficSource_medium",
    "trafficSource_campaign",
    "trafficSource_keyword",
)
