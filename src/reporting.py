"""Summary tables, CSV exports, key findings, console output and the Markdown report.

Every number written here is computed from the retrieved data; nothing is
hard-coded or illustrative.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config
from .formatting import compact_usd, format_price, interval_label, num, pct
from .outlier_detection import NEGATIVE_OUTLIER, POSITIVE_OUTLIER, format_outlier_line, material_outliers

logger = logging.getLogger(__name__)

SORT_OPTIONS = {
    "rank": ("market_cap_rank", True),
    "volatility": ("std_return_pct", False),
    "return": ("five_day_return_pct", False),
    "outliers": ("outlier_count", False),
}

DISCLAIMER = (
    "This is a data-science / statistical analysis of historical market data. Trend labels, volatility "
    "rankings and outlier flags are descriptive statistics, **not financial advice or trading "
    "recommendations**. Cryptocurrency prices are highly volatile and past behaviour does not predict "
    "future results."
)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def assemble_coin_summary(stats: pd.DataFrame, trend_snapshot: pd.DataFrame,
                          outlier_counts: pd.DataFrame) -> pd.DataFrame:
    """Join statistics, latest MA/trend and outlier counts into one row per coin."""
    trend_cols = [c for c in trend_snapshot.columns if c.startswith("ma_")] + ["trend", "price_vs_ma_long_pct"]
    summary = (
        stats.merge(trend_snapshot[["coin_id", *trend_cols]], on="coin_id", how="left")
        .merge(outlier_counts, on="coin_id", how="left")
    )
    count_cols = ["outlier_count", "extreme_outlier_count", "positive_outliers", "negative_outliers"]
    summary[count_cols] = summary[count_cols].fillna(0).astype(int)
    return summary.sort_values("market_cap_rank").reset_index(drop=True)


def sort_summary(summary: pd.DataFrame, by: str = "rank") -> pd.DataFrame:
    """Sort by market-cap rank (default), volatility, 5-day return or outlier count."""
    if by not in SORT_OPTIONS:
        raise ValueError(f"sort_by must be one of {list(SORT_OPTIONS)}")
    column, ascending = SORT_OPTIONS[by]
    return summary.sort_values([column, "market_cap_rank"], ascending=[ascending, True]).reset_index(drop=True)


def format_summary_table(summary: pd.DataFrame, windows=(3, 5)) -> pd.DataFrame:
    """Statistical summary with report-friendly column names (numeric values kept)."""
    short, long = windows
    table = pd.DataFrame(
        {
            "Rank": summary["market_cap_rank"],
            "Cryptocurrency": summary["name"],
            "Symbol": summary["symbol"],
            "Price (USD)": summary["price"],
            "5-Day Return (%)": summary["five_day_return_pct"],
            "Mean Return (%)": summary["mean_return_pct"],
            "Std Dev (%)": summary["std_return_pct"],
            f"MA({short})": summary[f"ma_{short}"],
            f"MA({long})": summary[f"ma_{long}"],
            "Trend": summary["trend"],
            "Outlier Count": summary["outlier_count"],
            "Extreme Outliers": summary["extreme_outlier_count"],
            "Volatility Rank": summary["volatility_rank"],
            "Market Cap (USD)": summary["market_cap"],
            "Unusually High Volatility": summary["high_volatility"],
            "Risk-Return Quadrant": summary["risk_return_quadrant"],
            "USD-Pegged (heuristic)": summary["peg_like"],
        }
    )
    if "low_variation" in summary:
        table["Near-Constant Asset"] = summary["low_variation"]
    if "is_anomalous_coin" in summary:
        table["Isolation Forest Anomaly"] = summary["is_anomalous_coin"]
    return table


def _display(df: pd.DataFrame, price_cols=(), pct_cols=(), signed_pct_cols=(), usd_cols=(), z_cols=()) -> pd.DataFrame:
    """String-formatted copy for Markdown / console output."""
    out = df.copy()
    for c in price_cols:
        out[c] = out[c].map(format_price)
    for c in pct_cols:
        out[c] = out[c].map(lambda v: pct(v, 3, signed=False))
    for c in signed_pct_cols:
        out[c] = out[c].map(lambda v: pct(v, 3))
    for c in usd_cols:
        out[c] = out[c].map(compact_usd)
    for c in z_cols:
        out[c] = out[c].map(lambda v: num(v, 2))
    return out


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------
def _coin(row: pd.Series) -> str:
    return f"{row['name']} ({row['symbol']})"


def key_findings(summary: pd.DataFrame, analysis: pd.DataFrame, cfg: config.AnalysisConfig) -> dict:
    """Compute headline numbers used by the console, Markdown report and dashboard."""
    s = summary.reset_index(drop=True)
    z_valid = int(analysis["z_score"].notna().sum())
    total = int(analysis["is_outlier"].sum())
    extreme = int(analysis["is_extreme_outlier"].sum())
    most_out = s.sort_values(["outlier_count", "extreme_outlier_count", "market_cap_rank"],
                             ascending=[False, False, True]).iloc[0]
    return {
        "n_coins": len(s),
        "window_start": analysis["timestamp"].min(),
        "window_end": analysis["timestamp"].max(),
        "median_interval_minutes": float(analysis["interval_minutes"].median()),
        "n_observations": len(analysis),
        "largest": s.sort_values("market_cap_rank").iloc[0],
        "most_volatile": s.loc[s["std_return_pct"].idxmax()],
        "least_volatile": s.loc[s["std_return_pct"].idxmin()],
        "best_return": s.loc[s["five_day_return_pct"].idxmax()],
        "worst_return": s.loc[s["five_day_return_pct"].idxmin()],
        "median_five_day_return": float(s["five_day_return_pct"].median()),
        "n_up": int((s["five_day_return_pct"] > 0).sum()),
        "trend_counts": s["trend"].value_counts().to_dict(),
        "z_valid_returns": z_valid,
        "total_outliers": total,
        "extreme_outliers": extreme,
        "positive_outliers": int((analysis["outlier_type"] == POSITIVE_OUTLIER).sum()),
        "negative_outliers": int((analysis["outlier_type"] == NEGATIVE_OUTLIER).sum()),
        "observed_outlier_rate": total / z_valid * 100 if z_valid else float("nan"),
        "observed_extreme_rate": extreme / z_valid * 100 if z_valid else float("nan"),
        "normal_outlier_rate": math.erfc(cfg.z_threshold / math.sqrt(2)) * 100,
        "normal_extreme_rate": math.erfc(cfg.strict_z_threshold / math.sqrt(2)) * 100,
        "most_outliers": most_out,
        "high_volatility": s.loc[s["high_volatility"]].sort_values("std_return_pct", ascending=False),
        "anomalous": (s.loc[s["is_anomalous_coin"]].sort_values("anomaly_score", ascending=False)
                      if "is_anomalous_coin" in s else s.iloc[0:0]),
        "quadrant_counts": s["risk_return_quadrant"].value_counts().to_dict(),
        "peg_like": s.loc[s["peg_like"]],
        "low_variation": s.loc[s["low_variation"]] if "low_variation" in s else s.iloc[0:0],
        "low_variation_outliers": (int(analysis.loc[analysis["is_outlier"], "low_variation"].sum())
                                   if "low_variation" in analysis else 0),
    }


def findings_bullets(f: dict, cfg: config.AnalysisConfig, fence: float) -> list[str]:
    trends = f["trend_counts"]
    lines = [
        f"**Largest market cap:** {_coin(f['largest'])} at {compact_usd(f['largest']['market_cap'])}.",
        f"**Most volatile:** {_coin(f['most_volatile'])} with σ = {f['most_volatile']['std_return_pct']:.3f}% "
        f"per period (5-day return {pct(f['most_volatile']['five_day_return_pct'])}).",
        f"**Least volatile:** {_coin(f['least_volatile'])} with σ = {f['least_volatile']['std_return_pct']:.4f}% "
        "per period.",
        f"**Highest 5-day return:** {_coin(f['best_return'])} at {pct(f['best_return']['five_day_return_pct'])}.",
        f"**Lowest 5-day return:** {_coin(f['worst_return'])} at {pct(f['worst_return']['five_day_return_pct'])}.",
        f"**Breadth:** {f['n_up']} of {f['n_coins']} coins rose over the window; the median 5-day return was "
        f"{pct(f['median_five_day_return'])}.",
        "**Trend labels at the latest observation:** "
        + " · ".join(f"{k} {trends.get(k, 0)}" for k in ("Bullish", "Bearish", "Neutral", "Insufficient data")
                     if trends.get(k, 0) or k != "Insufficient data") + ".",
        f"**Outliers (|Z| > {cfg.z_threshold:g}):** {f['total_outliers']:,} of {f['z_valid_returns']:,} returns "
        f"({f['observed_outlier_rate']:.2f}%; a normal distribution implies ≈{f['normal_outlier_rate']:.2f}%) — "
        f"{f['positive_outliers']:,} positive, {f['negative_outliers']:,} negative.",
        f"**Extreme outliers (|Z| > {cfg.strict_z_threshold:g}):** {f['extreme_outliers']:,} "
        f"({f['observed_extreme_rate']:.2f}% vs ≈{f['normal_extreme_rate']:.2f}% under normality), "
        "consistent with fat-tailed crypto returns." if f["observed_extreme_rate"] > f["normal_extreme_rate"] else
        f"**Extreme outliers (|Z| > {cfg.strict_z_threshold:g}):** {f['extreme_outliers']:,} "
        f"({f['observed_extreme_rate']:.2f}% vs ≈{f['normal_extreme_rate']:.2f}% under normality).",
        f"**Most outliers:** {_coin(f['most_outliers'])} with {int(f['most_outliers']['outlier_count'])} "
        f"({int(f['most_outliers']['extreme_outlier_count'])} extreme).",
    ]
    hv = f["high_volatility"]
    lines.append(
        f"**Unusually high volatility (σ above Q3 + 1.5×IQR = {fence:.3f}%):** "
        + (", ".join(f"{_coin(r)} {r['std_return_pct']:.3f}%" for _, r in hv.iterrows()) if len(hv) else "none")
        + "."
    )
    an = f["anomalous"]
    if len(an):
        lines.append("**Isolation Forest (multivariate) unusual coins:** " +
                     ", ".join(_coin(r) for _, r in an.iterrows()) + ".")
    lv = f["low_variation"]
    if len(lv):
        lines.append(
            f"**Near-constant assets:** {len(lv)} of {f['n_coins']} coins (stablecoins and tokenized "
            f"money-market/treasury funds) moved less than 0.5% across the entire window. They account for "
            f"{f['low_variation_outliers']:,} of the {f['total_outliers']:,} flagged outliers, because for an "
            "asset whose σ is ~0.003% a 0.03% blip is a 10σ event. Ranking purely by |Z| is dominated by these "
            "statistically-extreme-but-economically-trivial moves, so the largest *material* outliers are "
            "reported separately."
        )
    return lines


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def write_markdown_report(path: Path, *, cfg: config.AnalysisConfig, findings: dict, summary: pd.DataFrame,
                          top_most: pd.DataFrame, top_least: pd.DataFrame, outlier_report: pd.DataFrame,
                          prep_report, fetch_log: pd.DataFrame, fence: float, figures: dict[str, list[Path]],
                          fetched_at: datetime) -> Path:
    f = findings
    short, long = cfg.ma_windows
    interval = f["median_interval_minutes"]
    ok = int((fetch_log["status"] == "ok").sum())

    vol_fmt = dict(price_cols=["Current Price (USD)"], pct_cols=["Std Dev (%)"],
                   signed_pct_cols=["Average Return (%)", "5-Day Return (%)"])
    material = material_outliers(outlier_report)
    suppressed = len(outlier_report) - len(material)
    outlier_columns = ["Cryptocurrency", "Symbol", "Date/Time (UTC)", "Price (USD)", "Return (%)",
                       "Z-score", "Outlier Type", "Severity"]
    outliers_md = _display(material.head(25)[outlier_columns], price_cols=["Price (USD)"],
                           signed_pct_cols=["Return (%)"], z_cols=["Z-score"])
    extreme_z_md = _display(outlier_report.head(10)[outlier_columns + ["Low-Variation Asset"]],
                            price_cols=["Price (USD)"], signed_pct_cols=["Return (%)"], z_cols=["Z-score"])
    summary_md = _display(
        format_summary_table(summary, cfg.ma_windows)[
            ["Rank", "Cryptocurrency", "Symbol", "Price (USD)", "5-Day Return (%)", "Mean Return (%)", "Std Dev (%)",
             f"MA({short})", f"MA({long})", "Trend", "Outlier Count"]],
        price_cols=["Price (USD)", f"MA({short})", f"MA({long})"], pct_cols=["Std Dev (%)"],
        signed_pct_cols=["5-Day Return (%)", "Mean Return (%)"])
    quadrants = pd.DataFrame({"Quadrant": list(f["quadrant_counts"]), "Coins": list(f["quadrant_counts"].values())})
    prep = prep_report.to_frame()

    def rel(p: Path) -> str:
        return p.relative_to(path.parent).as_posix()

    fig_md = []
    titles = {"market_cap": "Graph 1 — Market capitalization", "price_trend": "Graph 2 — Price trend",
              "moving_average": "Graph 3 — Moving averages", "volatility": "Graph 4 — Volatility ranking",
              "outliers": "Graph 5 — Outlier detection", "risk_return": "Graph 6 — Risk vs return"}
    for key, title in titles.items():
        for p in figures.get(key, []):
            fig_md.append(f"### {title}: {p.stem.replace('_', ' ')}\n\n![{title}]({rel(p)})\n")

    examples = "\n".join(format_outlier_line(r) for _, r in material.head(5).iterrows()) or "No outliers."

    text = f"""# Cryptocurrency Market Volatility & Trend Analysis

> {DISCLAIMER}

| | |
|---|---|
| Generated | {fetched_at.strftime('%Y-%m-%d %H:%M UTC')} |
| Data source | CoinGecko API (`/coins/markets`, `/coins/{{id}}/market_chart`) |
| Window | {f['window_start']:%Y-%m-%d %H:%M} → {f['window_end']:%Y-%m-%d %H:%M} UTC (latest {cfg.days} days) |
| Resolution | {interval_label(interval)} observations (median interval {interval:.0f} min, original API granularity) |
| Coins analysed | {f['n_coins']} (history retrieved for {ok} of {len(fetch_log)} requested) |
| Observations | {f['n_observations']:,} price points |

## 1. Key findings

{chr(10).join('- ' + b for b in findings_bullets(f, cfg, fence))}

## 2. Top {cfg.top_k} most volatile cryptocurrencies

{_display(top_most, **vol_fmt).to_markdown(index=False)}

## 3. Top {cfg.top_k} least volatile cryptocurrencies

{_display(top_least, **vol_fmt).to_markdown(index=False)}

## 4. Outlier report

{len(outlier_report):,} observations were flagged in total (full list: `outlier_report.csv`).
{suppressed:,} of them belong to near-constant assets - stablecoins and tokenized money-market funds whose
sigma is so small that a few hundredths of a percent registers as a 10-sigma event. Those are real
statistical outliers but not market events, so the table below ranks the **{len(material):,} material
outliers** (near-constant assets excluded) by |Z|. Examples from this run:

```text
{examples}
```

{outliers_md.to_markdown(index=False)}

### Largest |Z| overall, including near-constant assets

Shown to make the distinction concrete: the biggest Z-scores in the data set belong to assets that barely
moved in price terms.

{extreme_z_md.to_markdown(index=False)}

## 5. Risk vs return quadrants

Split at the cross-sectional medians of standard deviation and mean return.

{quadrants.to_markdown(index=False)}

## 6. Statistical summary (market-cap order)

Sortable copy: `crypto_analysis_report.csv` (or `python main.py --sort-by volatility|return|outliers`).

{summary_md.to_markdown(index=False)}

## 7. Figures

{chr(10).join(fig_md) if fig_md else 'Figures were not generated for this run.'}

## 8. Data quality

{prep.to_markdown(index=False)}

## 9. Methodology

* **Periodic return:** r_t = (P_t − P_(t−1)) / P_(t−1) × 100, computed at the original timestamp resolution.
* **5-day return:** (P_last − P_first) / P_first × 100 over the window.
* **Moving averages:** MA(k)_t = mean of the k most recent prices including t (k = {short}, {long} periods).
* **Trend:** Bullish if Price > MA({short}) > MA({long}); Bearish if Price < MA({short}) < MA({long});
  otherwise Neutral.
* **Volatility:** population standard deviation of returns, σ = √(Σ(x − μ)² / N). Returns (not prices) are used
  so that coins with very different price levels are comparable.
* **Outliers:** Z = (x − μ) / σ per coin; |Z| > {cfg.z_threshold:g} flagged, |Z| > {cfg.strict_z_threshold:g} extreme.
* **Unusual volatility:** σ above the Tukey fence Q3 + 1.5 × IQR of all coins' σ.
* **24/7 markets:** no business-day calendar; every hour, night and weekend is a valid observation.

## 10. Limitations

* Five days of hourly data is a short sample; statistics are noisy and regime-dependent.
* MA(3)/MA(5) on hourly data describe hour-scale momentum only.
* Z-scores use each coin's own mean/σ, which include the outliers themselves (masking effect), and assume a
  roughly stable distribution; crypto returns are fat-tailed, so more |Z| > 2 events than the normal
  baseline are expected.
* CoinGecko prices are aggregated across exchanges and may differ from any single venue; the last point of each
  series may cover a partial interval.
* The market-cap snapshot and the historical series are retrieved seconds to minutes apart.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------
def print_console_summary(findings: dict, cfg: config.AnalysisConfig, fence: float, top_most: pd.DataFrame,
                          top_least: pd.DataFrame, outlier_report: pd.DataFrame) -> None:
    rule = "=" * 96
    vol_fmt = dict(price_cols=["Current Price (USD)"], pct_cols=["Std Dev (%)"],
                   signed_pct_cols=["Average Return (%)", "5-Day Return (%)"])
    print(f"\n{rule}\nCRYPTOCURRENCY MARKET VOLATILITY & TREND ANALYSIS\n{rule}")
    print(f"Window: {findings['window_start']:%Y-%m-%d %H:%M} -> {findings['window_end']:%Y-%m-%d %H:%M} UTC | "
          f"{findings['n_coins']} coins | {findings['n_observations']:,} observations | "
          f"{interval_label(findings['median_interval_minutes'])} data\n")
    print("KEY FINDINGS")
    for line in findings_bullets(findings, cfg, fence):
        print("  - " + line.replace("**", ""))
    print(f"\nTOP {cfg.top_k} MOST VOLATILE")
    print(_display(top_most, **vol_fmt).to_string(index=False))
    print(f"\nTOP {cfg.top_k} LEAST VOLATILE")
    print(_display(top_least, **vol_fmt).to_string(index=False))
    material = material_outliers(outlier_report)
    suppressed = len(outlier_report) - len(material)
    print("\nLARGEST MATERIAL OUTLIERS (by |Z|, excluding near-constant assets)")
    for _, row in material.head(10).iterrows():
        print("  " + format_outlier_line(row))
    if suppressed:
        print(f"  ({suppressed:,} outliers from stablecoins / tokenized funds omitted here — "
              "they are statistically extreme but economically tiny; see outlier_report.csv)")
    print(f"\n{rule}\nNot financial advice: descriptive statistics of historical data only.\n{rule}")
