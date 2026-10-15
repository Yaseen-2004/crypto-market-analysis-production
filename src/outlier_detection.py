"""Statistical outlier detection on returns.

Z-score method (per coin)
-------------------------
    Z = (x - mu) / sigma

where x is one periodic return and mu / sigma are the coin's own mean and
population standard deviation of returns over the window.

    |Z| > 2  -> outlier           (about 4.6% of observations if returns were normal)
    |Z| > 3  -> extreme outlier   (about 0.3% if normal)

Z > 0 is a *Positive Outlier* (unusually large gain); Z < 0 a *Negative
Outlier* (unusually large loss). Because crypto returns are fat-tailed, more
outliers than the normal-distribution baseline are expected.

A supplementary cross-sectional check uses scikit-learn's Isolation Forest to
spot coins whose overall 5-day behaviour is unusual compared with the others.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .statistics import population_std

logger = logging.getLogger(__name__)

POSITIVE_OUTLIER, NEGATIVE_OUTLIER = "Positive Outlier", "Negative Outlier"


def add_zscores(df: pd.DataFrame, return_col: str = "pct_return", group_col: str = "coin_id",
                min_std: float = 1e-12) -> pd.DataFrame:
    """Add per-coin mean, standard deviation and Z-score of returns.

    If a coin's standard deviation is (numerically) zero the Z-score is NaN.
    """
    out = df.copy()
    grouped = out.groupby(group_col, sort=False)[return_col]
    mean = grouped.transform("mean")  # skips NaN -> divides by N valid returns
    std = grouped.transform(population_std)
    out["coin_mean_return_pct"] = mean
    out["coin_std_return_pct"] = std
    out["z_score"] = ((out[return_col] - mean) / std).where(std > min_std)
    return out


def flag_outliers(df: pd.DataFrame, threshold: float = 2.0, strict_threshold: float = 3.0) -> pd.DataFrame:
    """Flag |Z| > threshold (outlier) and |Z| > strict_threshold (extreme)."""
    out = df.copy()
    abs_z = out["z_score"].abs()
    out["is_outlier"] = (abs_z > threshold).fillna(False).astype(bool)
    out["is_extreme_outlier"] = (abs_z > strict_threshold).fillna(False).astype(bool)
    out["outlier_type"] = np.select(
        [out["is_outlier"] & (out["z_score"] > 0), out["is_outlier"] & (out["z_score"] < 0)],
        [POSITIVE_OUTLIER, NEGATIVE_OUTLIER],
        default="",
    )
    return out


def count_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Per-coin outlier counts."""
    return (
        df.groupby("coin_id", sort=False)
        .agg(
            outlier_count=("is_outlier", "sum"),
            extreme_outlier_count=("is_extreme_outlier", "sum"),
            positive_outliers=("outlier_type", lambda s: int((s == POSITIVE_OUTLIER).sum())),
            negative_outliers=("outlier_type", lambda s: int((s == NEGATIVE_OUTLIER).sum())),
        )
        .astype(int)
        .reset_index()
    )


def build_outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """Dedicated outlier report, largest |Z| first. Values come straight from the data."""
    o = df.loc[df["is_outlier"]].copy()
    o["abs_z"] = o["z_score"].abs()
    o = o.sort_values(["abs_z", "market_cap_rank"], ascending=[False, True])
    report = pd.DataFrame(
        {
            "Market Cap Rank": o["market_cap_rank"],
            "Cryptocurrency": o["name"],
            "Symbol": o["symbol"],
            "Date/Time (UTC)": o["datetime_utc"],
            "Price (USD)": o["price"],
            "Return (%)": o["pct_return"],
            "Z-score": o["z_score"],
            "MA(3)": o.get("ma_3"),
            "MA(5)": o.get("ma_5"),
            "Coin Mean Return (%)": o["coin_mean_return_pct"],
            "Coin Std Dev (%)": o["coin_std_return_pct"],
            "Direction": np.where(o["pct_return"] >= 0, "Up", "Down"),
            "Low-Variation Asset": (o["low_variation"].astype(bool) if "low_variation" in o
                                    else pd.Series(False, index=o.index)),
            "Outlier Type": o["outlier_type"],
            # Plain words: a literal "|" here would break Markdown table rendering.
            "Severity": np.where(o["is_extreme_outlier"], "Extreme", "Outlier"),
        }
    )
    return report.reset_index(drop=True)


def material_outliers(report: pd.DataFrame) -> pd.DataFrame:
    """Outliers excluding near-constant assets - the economically meaningful moves.

    A stablecoin's +0.03% blip is a real statistical outlier but not a market
    event; ranking by |Z| alone buries genuine double-digit moves beneath them.
    """
    if "Low-Variation Asset" not in report:
        return report
    return report.loc[~report["Low-Variation Asset"].astype(bool)].reset_index(drop=True)


def format_outlier_line(row: pd.Series) -> str:
    """e.g. ``Bitcoin | 2026-09-08 14:00 | +8.42% | Z=2.71 | Positive Outlier``."""
    return (f"{row['Cryptocurrency']} | {row['Date/Time (UTC)']} | {row['Return (%)']:+.2f}% | "
            f"Z={row['Z-score']:.2f} | {row['Outlier Type']}")


def detect_anomalous_coins(stats: pd.DataFrame, contamination: float = 0.05,
                           random_state: int = 42) -> pd.DataFrame:
    """Supplementary multivariate check with Isolation Forest (scikit-learn).

    Features are standardised first. ``anomaly_score`` is higher for more
    unusual coins. Returns ``stats`` with ``anomaly_score`` / ``is_anomalous_coin``.
    """
    features = ["std_return_pct", "five_day_return_pct", "max_abs_return_pct", "max_drawdown_pct"]
    out = stats.copy()
    out["anomaly_score"] = np.nan
    out["is_anomalous_coin"] = False
    usable = out[features].replace([np.inf, -np.inf], np.nan).dropna()
    if len(usable) < 10:
        logger.info("Isolation Forest skipped: only %d coins with complete features", len(usable))
        return out
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
    except ImportError:  # optional dependency
        logger.info("scikit-learn not installed; Isolation Forest check skipped")
        return out

    X = StandardScaler().fit_transform(usable.to_numpy())
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=random_state)
    labels = model.fit_predict(X)
    out.loc[usable.index, "anomaly_score"] = -model.score_samples(X)
    out.loc[usable.index, "is_anomalous_coin"] = labels == -1
    return out
