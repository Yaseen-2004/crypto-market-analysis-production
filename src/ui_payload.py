"""Compact JSON payload for the HTML interfaces.

Used two ways:

* ``python ui/build_data.py`` writes it to ``ui/analysis_data.js`` for the static pages;
* the Django app builds it after every scheduled refresh and serves it from the database.

Keys are deliberately short: the payload carries a price series for all 100
coins, and terse names keep it near 650 KB rather than several megabytes.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

REQUIRED_SUMMARY = ["coin_id", "name", "symbol", "market_cap_rank", "price", "std_return_pct", "trend"]


def sig(value, digits: int = 6):
    """Round to ``digits`` significant figures; NaN becomes null."""
    if value is None or pd.isna(value):
        return None
    return float(f"%.{digits}g" % float(value))


def build_payload(summary: pd.DataFrame, analysis: pd.DataFrame, outlier_report: pd.DataFrame,
                  refreshed_at: datetime | None = None, run_id: int | None = None) -> dict:
    """Assemble the payload from the pipeline's summary, analysis dataset and outlier report."""
    missing = [c for c in REQUIRED_SUMMARY if c not in summary.columns]
    if missing:
        raise KeyError(f"summary is missing column(s) {missing}")

    analysis = analysis.copy()
    analysis["timestamp"] = pd.to_datetime(analysis["timestamp"], utc=True)
    window_start = analysis["timestamp"].min()

    coins = []
    for _, s in summary.sort_values("market_cap_rank").iterrows():
        g = analysis.loc[analysis["coin_id"] == s["coin_id"]].sort_values("timestamp")
        if g.empty:
            continue
        flags = g["is_outlier"].fillna(False).astype(bool).to_numpy()
        extreme = g["is_extreme_outlier"].fillna(False).astype(bool).to_numpy()
        snapshot = g["snapshot_price"].dropna() if "snapshot_price" in g else g["price"].iloc[0:0]
        live = snapshot.iloc[-1] if len(snapshot) else None
        coins.append({
            "id": s["coin_id"], "n": s["name"], "s": s["symbol"], "r": int(s["market_cap_rank"]),
            # p = last hourly bar (what the statistics use); lp = live snapshot price
            "p": sig(s["price"]), "lp": sig(live),
            "mc": sig(s.get("market_cap"), 5), "v": sig(s.get("volume_24h"), 5),
            "d5": sig(s.get("five_day_return_pct"), 4), "c24": sig(s.get("price_change_24h_pct_api"), 4),
            "mean": sig(s.get("mean_return_pct"), 4), "sd": sig(s["std_return_pct"], 4),
            "av": sig(s.get("annualized_volatility_pct"), 4), "dd": sig(s.get("max_drawdown_pct"), 4),
            "ma3": sig(s.get("ma_3")), "ma5": sig(s.get("ma_5")), "t": s["trend"],
            "vr": int(s.get("volatility_rank", 0) or 0), "oc": int(s.get("outlier_count", 0)),
            "ec": int(s.get("extreme_outlier_count", 0)), "po": int(s.get("positive_outliers", 0)),
            "no": int(s.get("negative_outliers", 0)), "q": s.get("risk_return_quadrant", ""),
            "lv": bool(s.get("low_variation", False)), "hv": bool(s.get("high_volatility", False)),
            "an": bool(s.get("is_anomalous_coin", False)),
            # minutes since the window start - far smaller than repeating timestamps
            "ts": ((g["timestamp"] - window_start).dt.total_seconds() // 60).astype(int).tolist(),
            "pr": [sig(v) for v in g["price"]],
            "m3": [sig(v) for v in g["ma_3"]],
            "m5": [sig(v) for v in g["ma_5"]],
            "rt": [sig(v, 4) for v in g["pct_return"]],
            "z": [sig(v, 4) for v in g["z_score"]],
            "oi": [i for i, f in enumerate(flags) if f],
            "oe": [i for i, f in enumerate(extreme) if f],
        })

    rows = [{
        "s": o["Symbol"], "n": o["Cryptocurrency"], "d": o["Date/Time (UTC)"],
        "p": sig(o["Price (USD)"]), "rt": sig(o["Return (%)"], 4), "z": sig(o["Z-score"], 4),
        "t": "pos" if str(o["Outlier Type"]).startswith("Positive") else "neg",
        "x": o["Severity"] == "Extreme", "lv": bool(o.get("Low-Variation Asset", False)),
    } for _, o in outlier_report.iterrows()]

    return {
        "meta": {
            "windowStart": window_start.strftime("%Y-%m-%d %H:%M"),
            "windowEnd": analysis["timestamp"].max().strftime("%Y-%m-%d %H:%M"),
            "observations": int(len(analysis)),
            "coins": len(coins),
            "intervalMinutes": int(analysis["interval_minutes"].median()),
            "totalOutliers": len(rows),
            "materialOutliers": sum(not r["lv"] for r in rows),
            "extremeOutliers": sum(r["x"] for r in rows),
            "nearConstant": int(summary["low_variation"].astype(bool).sum()) if "low_variation" in summary else 0,
            "advancers": int((summary["five_day_return_pct"] > 0).sum()),
            "medianReturn": sig(summary["five_day_return_pct"].median(), 4),
            "medianSd": sig(summary["std_return_pct"].median(), 4),
            "source": "CoinGecko API",
            "refreshedAt": refreshed_at.isoformat(timespec="seconds") if refreshed_at else None,
            "runId": run_id,
        },
        "coins": coins,
        "outliers": rows,
    }


def to_script(payload: dict) -> str:
    """``window.CRYPTO_DATA=...;`` - loadable with a plain script tag, no fetch needed."""
    import json

    return "window.CRYPTO_DATA=" + json.dumps(payload, separators=(",", ":")) + ";"
