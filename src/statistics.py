"""Per-coin return and volatility statistics.

Volatility is measured on *returns*, not raw prices. A price standard deviation
is expressed in dollars and scales with the price level (a $70,000 coin moving
1% has a larger dollar spread than a $0.50 coin moving 20%), so it cannot be
compared across coins. Percentage returns are scale-free and comparable.

Population standard deviation (N in the denominator) is used, as specified:

    sigma = sqrt( sum((x - mu)^2) / N )

Import this module through the package (``from src import statistics``) - the
file name intentionally matches the specification but shadows the stdlib name.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config

QUADRANT_LABELS = {
    (True, True): "High return / High volatility",
    (True, False): "High return / Low volatility",
    (False, True): "Low return / High volatility",
    (False, False): "Low return / Low volatility",
}


def population_mean(values) -> float:
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    return float(x.sum() / len(x)) if len(x) else float("nan")


def population_std(values) -> float:
    """sigma = sqrt(sum((x - mu)^2) / N), ignoring NaN values."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return float("nan")
    mu = x.sum() / n
    return float(math.sqrt(((x - mu) ** 2).sum() / n))


def max_drawdown_pct(prices) -> float:
    """Largest peak-to-trough decline within the window, in percent (<= 0)."""
    p = np.asarray(prices, dtype=float)
    if len(p) == 0:
        return float("nan")
    running_peak = np.maximum.accumulate(p)
    return float(((p / running_peak) - 1).min() * 100)


def compute_coin_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """One row of return/volatility statistics per coin.

    ``df`` must contain the cleaned history with ``pct_return`` and
    ``interval_minutes``.
    """
    rows = []
    for coin_id, g in df.sort_values("timestamp").groupby("coin_id", sort=False):
        returns = g["pct_return"].dropna().to_numpy(dtype=float)
        prices = g["price"].to_numpy(dtype=float)
        first, last = g.iloc[0], g.iloc[-1]
        interval = float(g["interval_minutes"].median())
        std = population_std(returns)
        periods_per_year = config.MINUTES_PER_YEAR / interval if interval > 0 else float("nan")
        rows.append(
            {
                "market_cap_rank": last["market_cap_rank"],
                "coin_id": coin_id,
                "name": last["name"],
                "symbol": last["symbol"],
                "price": prices[-1],
                "market_cap": last.get("snapshot_market_cap"),
                "volume_24h": last.get("snapshot_volume_24h"),
                "start_time_utc": first["timestamp"],
                "end_time_utc": last["timestamp"],
                "n_observations": len(prices),
                "n_returns": len(returns),
                "median_interval_minutes": interval,
                "start_price": prices[0],
                "five_day_return_pct": (prices[-1] - prices[0]) / prices[0] * 100,
                "mean_return_pct": population_mean(returns),
                "std_return_pct": std,
                "annualized_volatility_pct": std * math.sqrt(periods_per_year),
                "mean_abs_return_pct": float(np.abs(returns).mean()) if len(returns) else float("nan"),
                "min_return_pct": float(returns.min()) if len(returns) else float("nan"),
                "max_return_pct": float(returns.max()) if len(returns) else float("nan"),
                "max_abs_return_pct": float(np.abs(returns).max()) if len(returns) else float("nan"),
                "price_range_pct": (prices.max() - prices.min()) / prices.mean() * 100,
                "max_drawdown_pct": max_drawdown_pct(prices),
                "price_change_24h_pct_api": last.get("price_change_24h_pct"),
                "price_change_7d_pct_api": last.get("price_change_7d_pct"),
            }
        )
    stats = pd.DataFrame(rows)
    stats["market_cap_rank"] = stats["market_cap_rank"].astype("Int64")
    return stats.sort_values("market_cap_rank").reset_index(drop=True)


def rank_by_volatility(stats: pd.DataFrame) -> pd.DataFrame:
    """Add ``volatility_rank`` (1 = highest standard deviation of returns)."""
    out = stats.copy()
    out["volatility_rank"] = out["std_return_pct"].rank(ascending=False, method="min").astype("Int64")
    return out


def volatility_fence(stats: pd.DataFrame, iqr_multiplier: float = 1.5) -> float:
    """Tukey upper fence Q3 + k x IQR of the cross-section of standard deviations."""
    q1, q3 = stats["std_return_pct"].quantile([0.25, 0.75])
    return float(q3 + iqr_multiplier * (q3 - q1))


def flag_unusual_volatility(stats: pd.DataFrame, iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """Flag coins whose volatility is unusually high relative to the other coins.

    Uses the robust Tukey fence (not a Z-score) because the cross-section of
    volatilities is itself strongly right-skewed. A cross-sectional Z-score is
    also reported for reference.
    """
    out = stats.copy()
    fence = volatility_fence(out, iqr_multiplier)
    std = out["std_return_pct"]
    out["volatility_zscore"] = (std - population_mean(std)) / population_std(std)
    out["high_volatility"] = std > fence
    return out


def flag_peg_like(stats: pd.DataFrame, peg: float = 1.0, tolerance: float = 0.03,
                  max_range_pct: float = 2.0) -> pd.DataFrame:
    """Heuristically flag USD-pegged assets (stablecoins).

    Their returns are tiny, so even economically trivial moves can produce
    large Z-scores. The flag lets readers and the dashboard filter them.
    """
    out = stats.copy()
    out["peg_like"] = ((out["price"] - peg).abs() <= tolerance) & (out["price_range_pct"] <= max_range_pct)
    return out


def flag_low_variation(stats: pd.DataFrame, max_range_pct: float = 0.5) -> pd.DataFrame:
    """Flag assets that barely move over the whole window.

    Stablecoins, tokenized money-market/treasury funds and similar instruments
    have a standard deviation near zero, so an economically negligible move (a
    few hundredths of a percent) becomes a 10-sigma event. Any ranking by |Z|
    is otherwise dominated by them. Unlike ``peg_like`` this is independent of
    the price level, so it also catches funds trading at $1.11 or $106.
    """
    out = stats.copy()
    out["low_variation"] = out["price_range_pct"] <= max_range_pct
    return out


def assign_risk_return_quadrant(stats: pd.DataFrame) -> pd.DataFrame:
    """Quadrants split at the cross-sectional medians of std and mean return."""
    out = stats.copy()
    high_return = out["mean_return_pct"] >= out["mean_return_pct"].median()
    high_vol = out["std_return_pct"] >= out["std_return_pct"].median()
    out["risk_return_quadrant"] = [QUADRANT_LABELS[(r, v)] for r, v in zip(high_return, high_vol)]
    return out


def volatility_table(stats: pd.DataFrame) -> pd.DataFrame:
    """Volatility ranking table, highest volatility first."""
    table = stats.sort_values("std_return_pct", ascending=False)
    return pd.DataFrame(
        {
            "Rank": table["volatility_rank"].to_numpy(),
            "Market Cap Rank": table["market_cap_rank"].to_numpy(),
            "Cryptocurrency": table["name"].to_numpy(),
            "Symbol": table["symbol"].to_numpy(),
            "Mean Return (%)": table["mean_return_pct"].to_numpy(),
            "Standard Deviation (%)": table["std_return_pct"].to_numpy(),
            "Unusually High Volatility": table["high_volatility"].to_numpy(),
        }
    )


def top_volatile(stats: pd.DataFrame, n: int = 10, most: bool = True) -> pd.DataFrame:
    """Top ``n`` most (or least) volatile coins with the report metrics."""
    ordered = stats.sort_values("std_return_pct", ascending=not most).head(n)
    columns = {
        "volatility_rank": "Volatility Rank",
        "name": "Name",
        "symbol": "Symbol",
        "std_return_pct": "Std Dev (%)",
        "mean_return_pct": "Average Return (%)",
        "five_day_return_pct": "5-Day Return (%)",
        "price": "Current Price (USD)",
        # No literal "|" in labels: it would break Markdown table rendering.
        "outlier_count": "Outlier Count",
    }
    return ordered[[c for c in columns if c in ordered]].rename(columns=columns).reset_index(drop=True)
