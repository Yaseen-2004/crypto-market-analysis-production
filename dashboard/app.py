"""Interactive Streamlit dashboard.

    streamlit run dashboard/app.py

Reads the CSV files written by ``python main.py``. The sidebar can re-run the
whole pipeline (fetching fresh data) without leaving the browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config, reporting  # noqa: E402
from src import statistics as coin_stats  # noqa: E402
from src import visualization as viz  # noqa: E402
from src.config import AnalysisConfig  # noqa: E402
from src.formatting import compact_usd, format_price, interval_label, pct  # noqa: E402

st.set_page_config(page_title="Crypto Market Volatility & Trend Analysis", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def load_outputs(stamp: float):  # noqa: ARG001 - part of the cache key, not read directly
    """Load the pipeline outputs. ``stamp`` busts the cache when files change."""
    analysis = pd.read_csv(config.ANALYSIS_DATASET_CSV)
    analysis["timestamp"] = pd.to_datetime(analysis["timestamp"], utc=True, format="ISO8601")
    summary = pd.read_csv(config.COIN_STATISTICS_CSV)
    outliers = pd.read_csv(config.OUTLIER_REPORT_CSV)
    daily = pd.read_csv(config.DAILY_RETURNS_CSV) if config.DAILY_RETURNS_CSV.exists() else pd.DataFrame()
    return analysis, summary, outliers, daily


def outputs_stamp() -> float | None:
    needed = [config.ANALYSIS_DATASET_CSV, config.COIN_STATISTICS_CSV, config.OUTLIER_REPORT_CSV]
    return max(p.stat().st_mtime for p in needed) if all(p.exists() for p in needed) else None


def run_analysis(top_n: int, days: int) -> None:
    from src.main import run_pipeline

    with st.spinner(f"Fetching the top {top_n} coins and {days} days of history from CoinGecko "
                    "(rate-limited, this can take several minutes)..."):
        cfg = AnalysisConfig(top_n=top_n, days=days, make_plots=False, make_interactive=False)
        run_pipeline(cfg)
    st.cache_data.clear()
    st.success("Analysis complete.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Data")
stamp = outputs_stamp()
if stamp:
    st.sidebar.caption(f"Last analysis: {pd.Timestamp(stamp, unit='s', tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}")
else:
    st.sidebar.warning("No analysis found yet.")
top_n = st.sidebar.number_input("Coins (by market cap)", min_value=5, max_value=250, value=100, step=5)
days = st.sidebar.number_input("History window (days)", min_value=1, max_value=90, value=5)
if st.sidebar.button("Fetch fresh data & re-run", type="primary", width="stretch"):
    run_analysis(int(top_n), int(days))
    stamp = outputs_stamp()

if stamp is None:
    st.title("Cryptocurrency Market Volatility & Trend Analysis")
    st.info("Run `python main.py` in the project root, or use **Fetch fresh data & re-run** in the sidebar.")
    st.stop()

analysis, summary, outliers, daily = load_outputs(stamp)
flag = "low_variation" if "low_variation" in summary else "peg_like"
hide_pegged = st.sidebar.checkbox("Hide near-constant assets in rankings", value=False)
st.sidebar.caption("Stablecoins and tokenized money-market funds have a near-zero standard deviation, so they "
                   "dominate the 'least volatile' end and show huge Z-scores for economically tiny moves.")
ranked = summary.loc[~summary[flag].astype(bool)] if hide_pegged else summary
fence = coin_stats.volatility_fence(ranked)
windows = config.AnalysisConfig().ma_windows
short, long = windows

# ---------------------------------------------------------------------------
# Header & market overview
# ---------------------------------------------------------------------------
st.title("Cryptocurrency Market Volatility & Trend Analysis")
st.caption(
    f"{len(summary)} coins · {analysis['timestamp'].min():%Y-%m-%d %H:%M} → "
    f"{analysis['timestamp'].max():%Y-%m-%d %H:%M} UTC · "
    f"{interval_label(analysis['interval_minutes'].median())} observations · data: CoinGecko. "
    "Statistical analysis only — not financial advice."
)

st.subheader("Market overview")
most_vol = ranked.loc[ranked["std_return_pct"].idxmax()]
best = summary.loc[summary["five_day_return_pct"].idxmax()]
worst = summary.loc[summary["five_day_return_pct"].idxmin()]
largest = summary.sort_values("market_cap_rank").iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("Cryptocurrencies analysed", len(summary))
c1.metric("Total outliers (|Z| > 2)", f"{int(summary['outlier_count'].sum()):,}",
          f"{int(summary['extreme_outlier_count'].sum()):,} extreme (|Z| > 3)", delta_color="off")
c2.metric(f"Highest market cap — {largest['symbol']}", compact_usd(largest["market_cap"]), largest["name"],
          delta_color="off")
c2.metric(f"Most volatile — {most_vol['symbol']}", f"{most_vol['std_return_pct']:.3f}% σ", most_vol["name"],
          delta_color="off")
c3.metric(f"Highest 5-day return — {best['symbol']}", pct(best["five_day_return_pct"]), best["name"],
          delta_color="off")
c3.metric(f"Lowest 5-day return — {worst['symbol']}", pct(worst["five_day_return_pct"]), worst["name"],
          delta_color="off")

# ---------------------------------------------------------------------------
# Coin selector
# ---------------------------------------------------------------------------
st.subheader("Cryptocurrency")
options = summary.sort_values("market_cap_rank")
labels = {row["coin_id"]: f"{int(row['market_cap_rank'])}. {row['name']} ({row['symbol']})"
          for _, row in options.iterrows()}
coin_id = st.selectbox("Select a cryptocurrency", list(labels), format_func=lambda c: labels[c])
row = summary.loc[summary["coin_id"] == coin_id].iloc[0]

m = st.columns(4)
m[0].metric("Current price", format_price(row["price"]), pct(row["price_change_24h_pct_api"]) + " (24h)")
m[1].metric("Market cap", compact_usd(row["market_cap"]), f"rank {int(row['market_cap_rank'])}", delta_color="off")
m[2].metric("5-day return", pct(row["five_day_return_pct"]))
m[3].metric("Std dev of returns", f"{row['std_return_pct']:.3f}%",
            f"volatility rank {int(row['volatility_rank'])}", delta_color="off")
m2 = st.columns(4)
m2[0].metric(f"MA({short})", format_price(row[f"ma_{short}"]))
m2[1].metric(f"MA({long})", format_price(row[f"ma_{long}"]))
m2[2].metric("Trend", row["trend"], "analytical label", delta_color="off")
m2[3].metric("Outliers", int(row["outlier_count"]), f"{int(row['extreme_outlier_count'])} extreme",
             delta_color="off")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
tabs = st.tabs(["Price trend", "Moving averages", "Returns & outliers", "Volatility", "Risk vs return",
                "Market cap", "Tables"])

with tabs[0]:
    st.plotly_chart(viz.plotly_price_figure(analysis, coin_id, int(days)), width="stretch")
with tabs[1]:
    st.plotly_chart(viz.plotly_moving_average_figure(analysis, coin_id, windows), width="stretch")
    st.caption(f"Trend classification compares the latest price with MA({short}) and MA({long}); "
               "it describes recent momentum and is not a trading signal.")
with tabs[2]:
    st.plotly_chart(viz.plotly_outlier_figure(analysis, coin_id), width="stretch")
    coin_outliers = outliers.loc[outliers["Symbol"] == row["symbol"]]
    st.write(f"**{len(coin_outliers)} outlier(s)** for {row['name']}")
    st.dataframe(coin_outliers, width="stretch", hide_index=True)
with tabs[3]:
    st.plotly_chart(viz.plotly_volatility_figure(ranked, 10, fence), width="stretch")
with tabs[4]:
    st.plotly_chart(viz.plotly_risk_return_figure(ranked), width="stretch")
with tabs[5]:
    st.plotly_chart(viz.plotly_market_cap_figure(summary), width="stretch")

with tabs[6]:
    st.markdown("#### Statistical summary")
    sort_by = st.radio("Sort by", list(reporting.SORT_OPTIONS), horizontal=True,
                       format_func=lambda s: {"rank": "Market-cap rank", "volatility": "Volatility",
                                              "return": "5-day return", "outliers": "Outlier count"}[s])
    table = reporting.format_summary_table(reporting.sort_summary(summary, sort_by), windows)
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button("Download summary CSV", table.to_csv(index=False).encode("utf-8"),
                       "crypto_analysis_report.csv", "text/csv")

    st.markdown("#### Top 10 most / least volatile")
    left, right = st.columns(2)
    left.dataframe(coin_stats.top_volatile(ranked, 10, most=True), width="stretch", hide_index=True)
    right.dataframe(coin_stats.top_volatile(ranked, 10, most=False), width="stretch", hide_index=True)

    st.markdown("#### Outlier report")
    st.dataframe(outliers, width="stretch", hide_index=True)
    st.download_button("Download outlier CSV", outliers.to_csv(index=False).encode("utf-8"),
                       "outlier_report.csv", "text/csv")

    if not daily.empty:
        st.markdown("#### Daily closes and daily returns")
        st.dataframe(daily.loc[daily["coin_id"] == coin_id], width="stretch", hide_index=True)

if config.DATABASE_PATH.exists():
    with st.expander("Stored history (SQLite)"):
        from src import database

        with database.session(config.DATABASE_PATH) as conn:
            stored = database.totals(conn)
            runs = database.run_history(conn)
            cover = database.coverage(conn)
        cols = st.columns(4)
        cols[0].metric("Runs stored", f"{stored['runs']:,}")
        cols[1].metric("Observations", f"{stored['observations']:,}")
        cols[2].metric("Coins tracked", f"{stored['coins']:,}")
        cols[3].metric("Days covered", f"{stored['days_covered']:.1f}")
        st.caption("Every run appends to `data/crypto_market.db`. Observations are keyed by coin and "
                   "timestamp, so overlapping runs update rows instead of duplicating them — run the "
                   "pipeline regularly and the history grows past the 5-day API window.")
        st.dataframe(runs[["run_id", "started_at_utc", "window_start_utc", "window_end_utc", "n_coins",
                           "n_observations", "n_outliers"]], width="stretch", hide_index=True)
        st.dataframe(cover, width="stretch", hide_index=True)

st.divider()
st.caption(reporting.DISCLAIMER.replace("**", ""))
