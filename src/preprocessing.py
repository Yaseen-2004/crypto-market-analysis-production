"""Cleaning and preparation of the raw history.

Steps (in order)
----------------
1. Convert numeric columns (invalid values become NaN).
2. Drop rows without a coin id or timestamp; non-positive prices become NaN.
3. Convert Unix-millisecond timestamps to timezone-aware UTC datetimes.
4. Remove duplicate (coin, timestamp) records.
5. Sort chronologically within each coin.
6. Keep only the latest ``days`` of observations per coin.
7. Fill short interior gaps by time interpolation; drop prices still missing.
8. Exclude coins with too few observations.
9. Calculate periodic returns:  r_t = (P_t - P_{t-1}) / P_{t-1} x 100

Crypto markets trade 24/7, so no business-day calendar is applied: every
observation is a valid trading period and returns span weekends and nights.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NUMERIC_COLUMNS = [
    "timestamp_ms", "price", "market_cap", "total_volume", "market_cap_rank",
    "snapshot_price", "snapshot_market_cap", "snapshot_volume_24h",
    "price_change_24h_pct", "price_change_7d_pct",
]
METADATA_COLUMNS = [
    "name", "symbol", "market_cap_rank", "snapshot_price", "snapshot_market_cap",
    "snapshot_volume_24h", "price_change_24h_pct", "price_change_7d_pct",
]


@dataclass
class PreprocessingReport:
    """Audit trail of everything the cleaning step changed."""

    input_rows: int = 0
    coins_input: int = 0
    invalid_rows_dropped: int = 0
    invalid_prices_set_missing: int = 0
    duplicates_removed: int = 0
    rows_outside_window: int = 0
    prices_interpolated: int = 0
    rows_missing_price_dropped: int = 0
    coins_excluded: list[str] = field(default_factory=list)
    output_rows: int = 0
    coins_output: int = 0
    median_interval_minutes: float = float("nan")
    window_start_utc: str = ""
    window_end_utc: str = ""

    def to_frame(self) -> pd.DataFrame:
        data = asdict(self)
        data["coins_excluded"] = ", ".join(self.coins_excluded) or "none"
        return pd.DataFrame({"metric": list(data), "value": [str(v) for v in data.values()]})


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Periodic percentage return: (P_t - P_{t-1}) / P_{t-1} x 100 (first value NaN)."""
    previous = prices.shift(1)
    return (prices - previous) / previous * 100.0


def _fill_coin_gaps(group: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Time-interpolate short interior gaps for one coin (never extrapolates)."""
    g = group.set_index("timestamp")
    was_missing = g["price"].isna()
    for column in ("price", "market_cap", "total_volume"):
        if g[column].notna().sum() >= 2:
            g[column] = g[column].interpolate(method="time", limit=limit, limit_area="inside")
    g["price_imputed"] = was_missing & g["price"].notna()
    # Snapshot metadata is constant per coin; repair any holes from neighbours.
    for column in METADATA_COLUMNS:
        if column in g:
            g[column] = g[column].ffill().bfill()
    return g.reset_index()


def preprocess(
    raw: pd.DataFrame,
    days: int = 5,
    min_observations: int = 24,
    max_interpolation_gap: int = 3,
) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Clean the raw long-format history and add returns. See module docstring."""
    report = PreprocessingReport(input_rows=len(raw))
    if raw.empty:
        raise ValueError("No raw data to preprocess")
    df = raw.copy()

    # 1) numeric types -------------------------------------------------------
    for column in NUMERIC_COLUMNS:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ("coin_id", "name", "symbol"):
        df[column] = df[column].astype("string").str.strip()
    report.coins_input = int(df["coin_id"].nunique())

    # 2) invalid rows / values ---------------------------------------------
    invalid = df["coin_id"].isna() | df["coin_id"].fillna("").eq("") | df["timestamp_ms"].isna()
    report.invalid_rows_dropped = int(invalid.sum())
    df = df.loc[~invalid].copy()
    bad_price = df["price"].notna() & (df["price"] <= 0)
    report.invalid_prices_set_missing = int(bad_price.sum())
    df.loc[bad_price, "price"] = np.nan
    for column in ("market_cap", "total_volume"):
        df.loc[df[column] < 0, column] = np.nan

    # 3) readable timestamps -----------------------------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"].astype("int64"), unit="ms", utc=True)

    # 4) duplicates ---------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=["coin_id", "timestamp"], keep="last")
    report.duplicates_removed = before - len(df)

    # 5) chronological order ------------------------------------------------
    df = df.sort_values(["coin_id", "timestamp"], kind="mergesort")

    # 6) latest N-day window per coin --------------------------------------
    window_start = df.groupby("coin_id")["timestamp"].transform("max") - pd.Timedelta(days=days)
    outside = df["timestamp"] < window_start
    report.rows_outside_window = int(outside.sum())
    df = df.loc[~outside]

    # 7) missing values -----------------------------------------------------
    parts = [_fill_coin_gaps(g, max_interpolation_gap) for _, g in df.groupby("coin_id", sort=False)]
    df = pd.concat(parts, ignore_index=True)
    report.prices_interpolated = int(df["price_imputed"].sum())
    missing_price = df["price"].isna()
    report.rows_missing_price_dropped = int(missing_price.sum())
    df = df.loc[~missing_price]

    # 8) sufficient observations -------------------------------------------
    counts = df.groupby("coin_id")["price"].transform("size")
    excluded = sorted(df.loc[counts < min_observations, "coin_id"].unique().tolist())
    all_input = set(raw["coin_id"].dropna().astype(str))
    vanished = sorted(all_input - set(df["coin_id"].astype(str)))  # lost entirely in cleaning
    report.coins_excluded = sorted(set(excluded) | set(vanished))
    if report.coins_excluded:
        logger.warning("Excluded for insufficient data: %s", ", ".join(report.coins_excluded))
    df = df.loc[counts >= min_observations]
    if df.empty:
        raise ValueError(
            f"No cryptocurrency has at least {min_observations} valid observations in the last "
            f"{days} day(s); {len(report.coins_excluded)} coin(s) were excluded. Lower "
            "--min-observations, increase --days, or check data/raw/fetch_log.csv for fetch failures."
        )

    # 9) returns ------------------------------------------------------------
    df = df.sort_values(["market_cap_rank", "coin_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    grouped = df.groupby("coin_id", sort=False)
    df["interval_minutes"] = grouped["timestamp"].diff().dt.total_seconds() / 60.0
    df["pct_return"] = grouped["price"].transform(calculate_returns)

    df["datetime_utc"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["market_cap_rank"] = df["market_cap_rank"].round().astype("Int64")

    ordered = [
        "market_cap_rank", "coin_id", "name", "symbol", "timestamp", "datetime_utc", "date", "timestamp_ms",
        "price", "market_cap", "total_volume", "price_imputed", "interval_minutes", "pct_return",
        "snapshot_price", "snapshot_market_cap", "snapshot_volume_24h", "price_change_24h_pct",
        "price_change_7d_pct",
    ]
    df = df[[c for c in ordered if c in df.columns]]

    report.output_rows = len(df)
    report.coins_output = int(df["coin_id"].nunique())
    report.median_interval_minutes = float(df["interval_minutes"].median())
    report.window_start_utc = df["timestamp"].min().strftime("%Y-%m-%d %H:%M UTC")
    report.window_end_utc = df["timestamp"].max().strftime("%Y-%m-%d %H:%M UTC")
    logger.info(
        "Preprocessing: %d -> %d rows, %d coins, %d duplicates removed, %d prices interpolated",
        report.input_rows, report.output_rows, report.coins_output,
        report.duplicates_removed, report.prices_interpolated,
    )
    return df, report


def compute_daily_returns(clean: pd.DataFrame) -> pd.DataFrame:
    """Supplementary daily view: last price of each UTC calendar day and its daily return.

    The first and last calendar days of a rolling 5-day window are usually
    partial; they are flagged with ``is_partial_day``.
    """
    df = clean.sort_values(["coin_id", "timestamp"]).copy()
    df["utc_day"] = df["timestamp"].dt.floor("D")
    daily = (
        df.groupby(["coin_id", "utc_day"], sort=False)
        .agg(
            market_cap_rank=("market_cap_rank", "first"),
            name=("name", "first"),
            symbol=("symbol", "first"),
            close_time_utc=("timestamp", "last"),
            close_price=("price", "last"),
            observations=("price", "size"),
        )
        .reset_index()
    )
    daily = daily.sort_values(["market_cap_rank", "coin_id", "utc_day"]).reset_index(drop=True)
    daily["daily_return_pct"] = daily.groupby("coin_id", sort=False)["close_price"].transform(calculate_returns)

    interval = clean["interval_minutes"].median()
    expected = 24 * 60 / interval if interval and interval > 0 else 24
    daily["is_partial_day"] = daily["observations"] < 0.9 * expected
    daily["date"] = daily["utc_day"].dt.strftime("%Y-%m-%d")
    daily["close_time_utc"] = daily["close_time_utc"].dt.strftime("%Y-%m-%d %H:%M")
    return daily[["market_cap_rank", "coin_id", "name", "symbol", "date", "close_time_utc", "close_price",
                  "daily_return_pct", "observations", "is_partial_day"]]
