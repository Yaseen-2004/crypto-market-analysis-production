"""Simple moving averages and an analytical trend classification.

MA(k) at observation t is the arithmetic mean of the k most recent prices
*including* observation t (a standard trailing window):

    MA(k)_t = (P_t + P_{t-1} + ... + P_{t-k+1}) / k

The window is counted in *observations* (periods), so with CoinGecko's hourly
5-day data MA(3) spans ~3 hours and MA(5) ~5 hours.

Trend classification (analytical label, NOT a trading recommendation):

    Bullish : Price > MA(3) > MA(5)
    Bearish : Price < MA(3) < MA(5)
    Neutral : any other ordering
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BULLISH, BEARISH, NEUTRAL, INSUFFICIENT = "Bullish", "Bearish", "Neutral", "Insufficient data"


def add_moving_averages(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (3, 5),
    price_col: str = "price",
    group_col: str = "coin_id",
) -> pd.DataFrame:
    """Add ``ma_<k>`` columns computed independently for each coin.

    A value is only produced once ``k`` observations are available
    (``min_periods=k``), so the first k-1 rows of each coin are NaN.
    """
    out = df.sort_values([group_col, "timestamp"], kind="mergesort") if "timestamp" in df else df.copy()
    for window in windows:
        out[f"ma_{window}"] = out.groupby(group_col, sort=False)[price_col].transform(
            lambda s, w=window: s.rolling(window=w, min_periods=w).mean()
        )
    return out.sort_index()


def classify_trend(price: float, ma_short: float, ma_long: float) -> str:
    """Classify a single observation."""
    if any(pd.isna(v) for v in (price, ma_short, ma_long)):
        return INSUFFICIENT
    if price > ma_short > ma_long:
        return BULLISH
    if price < ma_short < ma_long:
        return BEARISH
    return NEUTRAL


def add_trend(df: pd.DataFrame, short: int = 3, long: int = 5, price_col: str = "price") -> pd.DataFrame:
    """Vectorised trend label for every row (requires ``ma_<short>`` / ``ma_<long>``)."""
    out = df.copy()
    price, ma_short, ma_long = out[price_col], out[f"ma_{short}"], out[f"ma_{long}"]
    missing = price.isna() | ma_short.isna() | ma_long.isna()
    out["trend"] = np.select(
        [missing, (price > ma_short) & (ma_short > ma_long), (price < ma_short) & (ma_short < ma_long)],
        [INSUFFICIENT, BULLISH, BEARISH],
        default=NEUTRAL,
    )
    return out


def latest_trend_snapshot(df: pd.DataFrame, short: int = 3, long: int = 5) -> pd.DataFrame:
    """Latest price, MA values and trend label for each coin."""
    latest = df.sort_values("timestamp").groupby("coin_id", sort=False).tail(1)
    snap = latest[["coin_id", "timestamp", "price", f"ma_{short}", f"ma_{long}"]].copy()
    snap["trend"] = [classify_trend(price, ma_short, ma_long) for price, ma_short, ma_long in
                     zip(snap["price"], snap[f"ma_{short}"], snap[f"ma_{long}"])]
    snap["price_vs_ma_long_pct"] = (snap["price"] / snap[f"ma_{long}"] - 1) * 100
    return snap.rename(columns={"timestamp": "latest_timestamp", "price": "latest_price"}).reset_index(drop=True)
