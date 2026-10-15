"""Static (Matplotlib) and interactive (Plotly) visualisations.

Design conventions
------------------
* Thin 2px lines, hairline solid gridlines, quiet axes; the data is the loudest ink.
* Fixed colour roles: price = blue, MA(3) = orange, MA(5) = aqua. Positive and
  negative outliers use opposite hues AND different marker shapes (up / down
  triangles) so identity never relies on colour alone.
* Emphasis charts (volatility ranking) colour the highlighted group and gray
  the rest.
* The time axis is continuous calendar time: crypto trades 24/7, so there are
  no weekend or overnight gaps to remove.
* Different price scales are never put on a dual axis; each coin gets its own chart.

Every Matplotlib function returns the Figure and saves it when ``save_path``
is given. Plotly builders return ``plotly.graph_objects.Figure`` objects that
are reused by the HTML report and the Streamlit dashboard.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from . import config
from .formatting import compact_usd, format_price, interval_label, pct
from .outlier_detection import NEGATIVE_OUTLIER, POSITIVE_OUTLIER

logger = logging.getLogger(__name__)


# ===========================================================================
# Matplotlib helpers
# ===========================================================================
def apply_style() -> None:
    """Apply the shared chart style to Matplotlib."""
    plt.rcParams.update(
        {
            "figure.facecolor": config.SURFACE,
            "axes.facecolor": config.SURFACE,
            "savefig.facecolor": config.SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": config.FONT_FAMILY,
            "font.size": 10,
            "axes.edgecolor": config.BASELINE,
            "axes.linewidth": 1.0,
            "axes.labelcolor": config.INK_SECONDARY,
            "axes.titlecolor": config.INK_PRIMARY,
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": config.GRIDLINE,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": config.BASELINE,
            "ytick.color": config.BASELINE,
            "xtick.labelcolor": config.INK_SECONDARY,
            "ytick.labelcolor": config.INK_SECONDARY,
            "legend.frameon": False,
            "legend.labelcolor": config.INK_SECONDARY,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def _save(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.debug("Saved figure %s", path)


def _heading(ax: plt.Axes, title: str, subtitle: str = "", pad: float = 30) -> None:
    ax.set_title(title, pad=pad)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, pad - 12),
                    textcoords="offset points", fontsize=9.5, color=config.INK_SECONDARY, va="top")


def _time_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_xlabel("Time (UTC, continuous 24/7)")


def _end_dot(ax: plt.Axes, x, y, color: str, size: float = 64) -> None:
    ax.scatter([x], [y], s=size, color=color, edgecolors=config.SURFACE, linewidths=2, zorder=6)


NEAR_CONSTANT_LABEL = "Near-constant (stablecoin / tokenized fund)"


def _near_constant(stats: pd.DataFrame) -> pd.Series:
    """Boolean mask of assets that barely move (de-emphasised in cross-sectional charts)."""
    for column in ("low_variation", "peg_like"):
        if column in stats:
            return stats[column].astype(bool)
    return pd.Series(False, index=stats.index)


def coin_slice(df: pd.DataFrame, coin_id: str) -> pd.DataFrame:
    g = df.loc[df["coin_id"] == coin_id].sort_values("timestamp")
    if g.empty:
        raise ValueError(f"No data for coin '{coin_id}'")
    return g


def _price_formatter() -> FuncFormatter:
    return FuncFormatter(lambda v, _: format_price(v))


# ===========================================================================
# Graph 1 - market capitalisation
# ===========================================================================
def plot_market_cap(stats: pd.DataFrame, save_path=None) -> plt.Figure:
    """Ranked horizontal bar chart of market capitalisation (log scale)."""
    d = stats.dropna(subset=["market_cap"]).sort_values("market_cap", ascending=False).reset_index(drop=True)
    n = len(d)
    fig, ax = plt.subplots(figsize=(11, max(6.0, 0.2 * n + 2.2)))
    y = np.arange(n)
    caps = d["market_cap"].to_numpy(dtype=float)
    base = 10 ** np.floor(np.log10(caps.min()))

    ax.barh(y, caps - base, left=base, height=0.62, color=config.SERIES_1)
    ax.set_xscale("log")
    ax.set_xlim(base, caps.max() * 8)
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_yticks(y, [f"{int(r)}. {nm} ({sym})" for r, nm, sym in
                      zip(d["market_cap_rank"], d["name"], d["symbol"])])
    ax.tick_params(axis="y", length=0, labelsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: compact_usd(v)))
    for yi, value in zip(y, caps):
        ax.text(value * 1.12, yi, compact_usd(value), va="center", fontsize=6.8, color=config.INK_SECONDARY)

    ax.set_xlabel("Market capitalization (USD, logarithmic scale)")
    ax.set_ylabel("Cryptocurrency (market-cap rank)")
    ratio = caps.max() / caps.min()
    _heading(ax, f"Top {n} cryptocurrencies by market capitalization",
             f"Snapshot from CoinGecko. Log scale: #1 is {ratio:,.0f}x the size of #{n}; "
             "values are printed at each bar tip.", pad=34)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ===========================================================================
# Graph 2 - price trend
# ===========================================================================
def plot_price_trend(df: pd.DataFrame, coin_id: str, days: int = 5, save_path=None) -> plt.Figure:
    """Price over the latest window for one coin."""
    g = coin_slice(df, coin_id)
    name, symbol = g["name"].iloc[0], g["symbol"].iloc[0]
    t, p = g["timestamp"], g["price"]
    change = (p.iloc[-1] - p.iloc[0]) / p.iloc[0] * 100

    fig, ax = plt.subplots(figsize=(11, 4.8))
    span = p.max() - p.min() or abs(p.iloc[-1]) * 0.01
    floor = p.min() - 0.08 * span
    ax.plot(t, p, color=config.SERIES_1, lw=2)
    ax.fill_between(t, p, floor, color=config.SERIES_1, alpha=0.10, lw=0)
    ax.set_ylim(floor, p.max() + 0.12 * span)

    i_max, i_min = p.idxmax(), p.idxmin()
    for idx, va, dy in ((i_max, "bottom", 6), (i_min, "top", -6)):
        ax.scatter([g.at[idx, "timestamp"]], [g.at[idx, "price"]], s=36, color=config.SERIES_1,
                   edgecolors=config.SURFACE, linewidths=1.5, zorder=5)
        label = "High" if idx == i_max else "Low"
        ax.annotate(f"{label} {format_price(g.at[idx, 'price'])}", (g.at[idx, "timestamp"], g.at[idx, "price"]),
                    xytext=(0, dy), textcoords="offset points", ha="center", va=va, fontsize=8.5,
                    color=config.INK_SECONDARY)
    _end_dot(ax, t.iloc[-1], p.iloc[-1], config.SERIES_1)
    ax.annotate(format_price(p.iloc[-1]), (t.iloc[-1], p.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, color=config.INK_PRIMARY, fontweight="semibold")

    ax.yaxis.set_major_formatter(_price_formatter())
    ax.set_ylabel("Price (USD)")
    _time_axis(ax)
    ax.margins(x=0.06)
    interval = g["interval_minutes"].median()
    _heading(ax, f"{name} price — last {days} days",
             f"{symbol}/USD · {len(g)} {interval_label(interval)} observations · "
             f"change over window {pct(change)}")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ===========================================================================
# Graph 3 - moving averages
# ===========================================================================
def plot_moving_averages(df: pd.DataFrame, coin_id: str, windows=(3, 5), zoom_periods: int = 24,
                         save_path=None) -> plt.Figure:
    """Price with MA(short) and MA(long): full window (top) and recent zoom (bottom)."""
    from .moving_average import classify_trend

    short, long = windows
    g = coin_slice(df, coin_id)
    name, symbol = g["name"].iloc[0], g["symbol"].iloc[0]
    last = g.iloc[-1]
    trend = classify_trend(last["price"], last[f"ma_{short}"], last[f"ma_{long}"])

    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(11, 8.2), gridspec_kw={"height_ratios": [3, 2]})
    series = (("price", "Price", config.SERIES_1), (f"ma_{short}", f"MA({short})", config.SERIES_2),
              (f"ma_{long}", f"MA({long})", config.SERIES_3))
    zoom = g.tail(zoom_periods)
    for ax, data in ((ax_full, g), (ax_zoom, zoom)):
        for column, label, color in series:
            ax.plot(data["timestamp"], data[column], color=color, lw=2, label=label)
        for column, _, color in series:
            _end_dot(ax, data["timestamp"].iloc[-1], data[column].iloc[-1], color, size=48)
        ax.yaxis.set_major_formatter(_price_formatter())
        ax.set_ylabel("Price (USD)")
        _time_axis(ax)

    ax_full.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9, handlelength=1.6,
                   borderaxespad=0.2)
    relation = {"Bullish": f"Price > MA({short}) > MA({long})", "Bearish": f"Price < MA({short}) < MA({long})"}
    _heading(ax_full, f"{name} ({symbol}) price with MA({short}) and MA({long})",
             f"Latest: price {format_price(last['price'])} · MA({short}) {format_price(last[f'ma_{short}'])} · "
             f"MA({long}) {format_price(last[f'ma_{long}'])} → {trend}"
             f"{' (' + relation[trend] + ')' if trend in relation else ''}. "
             "Analytical label, not a trading recommendation.", pad=48)
    ax_zoom.set_title(f"Zoom: last {len(zoom)} observations", fontsize=11, pad=8)
    fig.tight_layout(h_pad=2.5)
    _save(fig, save_path)
    return fig


# ===========================================================================
# Graph 4 - volatility ranking
# ===========================================================================
def plot_volatility_ranking(stats: pd.DataFrame, top_k: int = 10, fence: float | None = None,
                            save_path=None) -> plt.Figure:
    """Standard deviation of returns, highest to lowest, top-k highlighted."""
    d = stats.dropna(subset=["std_return_pct"]).sort_values("std_return_pct", ascending=False).reset_index(drop=True)
    n = len(d)
    fig, ax = plt.subplots(figsize=(11, max(6.0, 0.2 * n + 2.2)))
    y = np.arange(n)
    colors = [config.SERIES_1 if i < top_k else config.DE_EMPHASIS for i in range(n)]
    ax.barh(y, d["std_return_pct"], height=0.62, color=colors)
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_yticks(y, [f"{i + 1}. {nm} ({sym})" for i, (nm, sym) in enumerate(zip(d["name"], d["symbol"]))])
    ax.tick_params(axis="y", length=0, labelsize=7.5)
    ax.grid(axis="y", visible=False)

    for i, value in enumerate(d["std_return_pct"]):
        flagged = bool(d.at[i, "high_volatility"]) if "high_volatility" in d else False
        if i < top_k or flagged:
            ax.text(value, i, f"  {value:.3f}%", va="center", fontsize=7.5, color=config.INK_SECONDARY)

    handles = [Patch(color=config.SERIES_1, label=f"Top {top_k} most volatile"),
               Patch(color=config.DE_EMPHASIS, label="Other coins")]
    if fence is not None and np.isfinite(fence):
        ax.axvline(fence, color=config.INK_MUTED, lw=1)
        handles.append(Line2D([], [], color=config.INK_MUTED, lw=1,
                              label=f"Unusually high threshold: Q3 + 1.5×IQR = {fence:.3f}%"))
    ax.set_xlim(0, d["std_return_pct"].max() * 1.15)
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    interval = d["median_interval_minutes"].median() if "median_interval_minutes" in d else np.nan
    ax.set_xlabel(f"Standard deviation of {interval_label(interval)} returns (%)")
    ax.set_ylabel("Cryptocurrency (volatility rank)")
    _heading(ax, "Volatility ranking — highest to lowest",
             "Population standard deviation of percentage returns (scale-free, comparable across price levels).",
             pad=34)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ===========================================================================
# Graph 5 - outliers
# ===========================================================================
def plot_outliers(df: pd.DataFrame, coin_id: str, threshold: float = 2.0, strict_threshold: float = 3.0,
                  save_path=None) -> plt.Figure:
    """Returns over time with the mean, the ±threshold·σ band and flagged outliers."""
    g = coin_slice(df, coin_id).dropna(subset=["pct_return"])
    name, symbol = g["name"].iloc[0], g["symbol"].iloc[0]
    mean, std = float(g["coin_mean_return_pct"].iloc[0]), float(g["coin_std_return_pct"].iloc[0])

    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.axhspan(mean - threshold * std, mean + threshold * std, color=config.GRIDLINE, alpha=0.6, lw=0,
               label=f"Mean ± {threshold:g}σ (not an outlier)")
    ax.plot(g["timestamp"], g["pct_return"], color=config.INK_SECONDARY, lw=1.5, label="Return")
    ax.axhline(mean, color=config.INK_MUTED, lw=1.2, label=f"Mean return ({mean:+.3f}%)")

    pos = g.loc[g["outlier_type"] == POSITIVE_OUTLIER]
    neg = g.loc[g["outlier_type"] == NEGATIVE_OUTLIER]
    for subset, marker, color, label in ((pos, "^", config.POSITIVE, f"Positive outlier (Z > {threshold:g})"),
                                         (neg, "v", config.NEGATIVE, f"Negative outlier (Z < −{threshold:g})")):
        sizes = np.where(subset["is_extreme_outlier"], 150, 80)
        ax.scatter(subset["timestamp"], subset["pct_return"], s=sizes, marker=marker, color=color,
                   edgecolors=config.SURFACE, linewidths=1.5, zorder=5, label=f"{label} · {len(subset)}")
    n_extreme = int(g["is_extreme_outlier"].sum())
    if n_extreme:
        ax.scatter([], [], s=150, marker="o", facecolors="none", edgecolors=config.INK_MUTED,
                   label=f"Larger marker = extreme (|Z| > {strict_threshold:g}) · {n_extreme}")

    for subset, dy, va in ((pos, 8, "bottom"), (neg, -8, "top")):  # label only the most extreme of each
        if not subset.empty:
            row = subset.loc[subset["z_score"].abs().idxmax()]
            ax.annotate(f"{row['pct_return']:+.2f}%  Z={row['z_score']:.2f}", (row["timestamp"], row["pct_return"]),
                        xytext=(0, dy), textcoords="offset points", ha="center", va=va, fontsize=8.5,
                        color=config.INK_PRIMARY)

    low, high = g["pct_return"].min(), g["pct_return"].max()
    pad_y = (high - low) * 0.15 or 0.1
    ax.set_ylim(low - pad_y, high + pad_y)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.2f}%" if v else "0%"))
    ax.set_ylabel("Return per period (%)")
    _time_axis(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8.5, borderaxespad=0.2)
    interval = g["interval_minutes"].median()
    _heading(ax, f"{name} ({symbol}) {interval_label(interval)} returns and Z-score outliers",
             f"Z = (return − mean) / σ with σ = {std:.3f}% · {len(pos) + len(neg)} outliers (|Z| > {threshold:g}), "
             f"{n_extreme} extreme (|Z| > {strict_threshold:g})", pad=64)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ===========================================================================
# Graph 6 - risk vs return
# ===========================================================================
def plot_risk_return(stats: pd.DataFrame, n_label_vol: int = 5, n_label_ret: int = 3,
                     save_path=None) -> plt.Figure:
    """Scatter of standard deviation (risk) vs mean return, split at the medians."""
    d = stats.dropna(subset=["std_return_pct", "mean_return_pct"]).reset_index(drop=True)
    peg = _near_constant(d)
    x_med, y_med = d["std_return_pct"].median(), d["mean_return_pct"].median()

    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.axvline(x_med, color=config.BASELINE, lw=1, zorder=1)
    ax.axhline(y_med, color=config.BASELINE, lw=1, zorder=1)
    ax.scatter(d.loc[~peg, "std_return_pct"], d.loc[~peg, "mean_return_pct"], s=64, color=config.SERIES_1,
               edgecolors=config.SURFACE, linewidths=1.5, zorder=3, label="Cryptocurrency")
    if peg.any():
        ax.scatter(d.loc[peg, "std_return_pct"], d.loc[peg, "mean_return_pct"], s=64, color=config.DE_EMPHASIS,
                   edgecolors=config.SURFACE, linewidths=1.5, zorder=3, label=NEAR_CONSTANT_LABEL)

    corners = ((0.99, 0.98, "right", "top", "High return / High volatility"),
               (0.01, 0.98, "left", "top", "High return / Low volatility"),
               (0.01, 0.02, "left", "bottom", "Low return / Low volatility"),
               (0.99, 0.02, "right", "bottom", "Low return / High volatility"))
    for xa, ya, ha, va, text in corners:
        ax.text(xa, ya, text, transform=ax.transAxes, ha=ha, va=va, fontsize=9, color=config.INK_MUTED)

    core = d.loc[~peg]
    to_label = set(core.nlargest(n_label_vol, "std_return_pct").index)
    to_label |= set(core.nlargest(n_label_ret, "mean_return_pct").index)
    to_label |= set(core.nsmallest(n_label_ret, "mean_return_pct").index)
    for idx in to_label:
        ax.annotate(d.at[idx, "symbol"], (d.at[idx, "std_return_pct"], d.at[idx, "mean_return_pct"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8.5, color=config.INK_PRIMARY)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.3f}%" if v else "0%"))
    ax.set_xlabel("Risk: standard deviation of returns (%)")
    ax.set_ylabel("Return: mean periodic return (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9, borderaxespad=0.2)
    _heading(ax, "Risk vs return",
             f"Each point is one cryptocurrency. Quadrants split at the medians "
             f"(σ = {x_med:.3f}%, mean return = {y_med:+.4f}%).", pad=48)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ===========================================================================
# Plotly (interactive)
# ===========================================================================
def _go():
    import plotly.graph_objects as go  # imported lazily so Matplotlib-only use needs no Plotly
    return go


def _plotly_layout(fig, title: str, height: int = 460, hovermode: str = "x unified"):
    axis = dict(gridcolor=config.GRIDLINE, linecolor=config.BASELINE, zeroline=False,
                tickfont=dict(color=config.INK_SECONDARY))
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16, color=config.INK_PRIMARY)),
        height=height, hovermode=hovermode, paper_bgcolor=config.SURFACE, plot_bgcolor=config.SURFACE,
        font=dict(family="Segoe UI, system-ui, -apple-system, sans-serif", color=config.INK_SECONDARY, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=90, b=50), xaxis=axis, yaxis=axis,
    )
    return fig


def _coin_title(df: pd.DataFrame, coin_id: str) -> str:
    g = df.loc[df["coin_id"] == coin_id]
    return f"{g['name'].iloc[0]} ({g['symbol'].iloc[0]})"


def _price_hover(label: str) -> str:
    return f"{label}: $%{{y:,.6~g}}<extra></extra>"


def price_traces(df: pd.DataFrame, coin_id: str, include_ma: bool = True, windows=(3, 5)) -> list:
    go = _go()
    g = coin_slice(df, coin_id)
    traces = [go.Scatter(x=g["timestamp"], y=g["price"], name="Price", mode="lines",
                         line=dict(color=config.SERIES_1, width=2), hovertemplate=_price_hover("Price"))]
    if include_ma:
        for window, color in zip(windows, (config.SERIES_2, config.SERIES_3)):
            traces.append(go.Scatter(x=g["timestamp"], y=g[f"ma_{window}"], name=f"MA({window})", mode="lines",
                                     line=dict(color=color, width=2), hovertemplate=_price_hover(f"MA({window})")))
    return traces


def outlier_traces(df: pd.DataFrame, coin_id: str, threshold: float = 2.0) -> list:
    go = _go()
    g = coin_slice(df, coin_id).dropna(subset=["pct_return"])
    mean, std = float(g["coin_mean_return_pct"].iloc[0]), float(g["coin_std_return_pct"].iloc[0])
    t = g["timestamp"]
    traces = [
        go.Scatter(x=t, y=np.full(len(g), mean + threshold * std), mode="lines", line=dict(width=0),
                   hoverinfo="skip", showlegend=False, name="upper"),
        go.Scatter(x=t, y=np.full(len(g), mean - threshold * std), mode="lines", line=dict(width=0),
                   fill="tonexty", fillcolor="rgba(225,224,217,0.6)", hoverinfo="skip",
                   name=f"Mean ± {threshold:g}σ"),
        go.Scatter(x=t, y=g["pct_return"], mode="lines", name="Return",
                   line=dict(color=config.INK_SECONDARY, width=1.5),
                   customdata=g["z_score"], hovertemplate="Return: %{y:+.3f}% (Z=%{customdata:.2f})<extra></extra>"),
        go.Scatter(x=t, y=np.full(len(g), mean), mode="lines", name=f"Mean return ({mean:+.3f}%)",
                   line=dict(color=config.INK_MUTED, width=1.2), hoverinfo="skip"),
    ]
    for kind, symbol, color in ((POSITIVE_OUTLIER, "triangle-up", config.POSITIVE),
                                (NEGATIVE_OUTLIER, "triangle-down", config.NEGATIVE)):
        o = g.loc[g["outlier_type"] == kind]
        traces.append(go.Scatter(
            x=o["timestamp"], y=o["pct_return"], mode="markers", name=f"{kind} ({len(o)})",
            marker=dict(symbol=symbol, color=color, size=np.where(o["is_extreme_outlier"], 16, 11),
                        line=dict(color=config.SURFACE, width=1.5)),
            customdata=np.stack([o["z_score"], o["price"]], axis=-1) if len(o) else None,
            hovertemplate=f"{kind}: %{{y:+.3f}}%<br>Z=%{{customdata[0]:.2f}} · price $%{{customdata[1]:,.6~g}}"
                          "<extra></extra>"))
    return traces


def plotly_price_figure(df: pd.DataFrame, coin_id: str, days: int = 5):
    go = _go()
    fig = go.Figure(price_traces(df, coin_id, include_ma=False))
    _plotly_layout(fig, f"{_coin_title(df, coin_id)} price — last {days} days")
    fig.update_yaxes(title_text="Price (USD)", tickprefix="$")
    fig.update_xaxes(title_text="Time (UTC, 24/7)")
    return fig


def plotly_moving_average_figure(df: pd.DataFrame, coin_id: str, windows=(3, 5)):
    go = _go()
    fig = go.Figure(price_traces(df, coin_id, include_ma=True, windows=windows))
    _plotly_layout(fig, f"{_coin_title(df, coin_id)} price with MA({windows[0]}) and MA({windows[1]})")
    fig.update_yaxes(title_text="Price (USD)", tickprefix="$")
    fig.update_xaxes(title_text="Time (UTC, 24/7)")
    return fig


def plotly_outlier_figure(df: pd.DataFrame, coin_id: str, threshold: float = 2.0):
    go = _go()
    fig = go.Figure(outlier_traces(df, coin_id, threshold))
    _plotly_layout(fig, f"{_coin_title(df, coin_id)} returns and Z-score outliers", hovermode="closest")
    fig.update_yaxes(title_text="Return per period (%)", ticksuffix="%")
    fig.update_xaxes(title_text="Time (UTC, 24/7)")
    return fig


def plotly_market_cap_figure(stats: pd.DataFrame):
    go = _go()
    d = stats.dropna(subset=["market_cap"]).sort_values("market_cap", ascending=False)
    labels = [f"{int(r)}. {n} ({s})" for r, n, s in zip(d["market_cap_rank"], d["name"], d["symbol"])]
    fig = go.Figure(go.Bar(x=d["market_cap"], y=labels, orientation="h", marker_color=config.SERIES_1,
                           text=[compact_usd(v) for v in d["market_cap"]], textposition="outside",
                           hovertemplate="%{y}<br>Market cap: %{text}<extra></extra>"))
    _plotly_layout(fig, f"Top {len(d)} cryptocurrencies by market capitalization (log scale)",
                   height=max(500, 20 * len(d) + 120), hovermode="closest")
    fig.update_xaxes(type="log", title_text="Market capitalization (USD, log scale)", tickprefix="$")
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    return fig


def plotly_volatility_figure(stats: pd.DataFrame, top_k: int = 10, fence: float | None = None):
    go = _go()
    d = stats.dropna(subset=["std_return_pct"]).sort_values("std_return_pct", ascending=False).reset_index(drop=True)
    colors = [config.SERIES_1 if i < top_k else config.DE_EMPHASIS for i in range(len(d))]
    fig = go.Figure(go.Bar(
        x=d["std_return_pct"], y=[f"{i + 1}. {n} ({s})" for i, (n, s) in enumerate(zip(d["name"], d["symbol"]))],
        orientation="h", marker_color=colors, customdata=d["mean_return_pct"],
        hovertemplate="%{y}<br>Std dev: %{x:.3f}%<br>Mean return: %{customdata:+.4f}%<extra></extra>"))
    if fence is not None and np.isfinite(fence):
        fig.add_vline(x=fence, line_color=config.INK_MUTED, line_width=1,
                      annotation_text=f"Unusually high: > {fence:.3f}%", annotation_position="top right")
    _plotly_layout(fig, f"Volatility ranking (top {top_k} highlighted)", height=max(500, 20 * len(d) + 120),
                   hovermode="closest")
    fig.update_xaxes(title_text="Standard deviation of returns (%)", ticksuffix="%")
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    return fig


def plotly_risk_return_figure(stats: pd.DataFrame):
    go = _go()
    d = stats.dropna(subset=["std_return_pct", "mean_return_pct"])
    peg = _near_constant(d)
    fig = go.Figure()
    for mask, label, color in ((~peg, "Cryptocurrency", config.SERIES_1),
                               (peg, NEAR_CONSTANT_LABEL, config.DE_EMPHASIS)):
        s = d.loc[mask]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["std_return_pct"], y=s["mean_return_pct"], mode="markers", name=label,
            marker=dict(color=color, size=11, line=dict(color=config.SURFACE, width=1.5)),
            text=[f"{n} ({sym})" for n, sym in zip(s["name"], s["symbol"])],
            customdata=s["five_day_return_pct"],
            hovertemplate="%{text}<br>Std dev: %{x:.3f}%<br>Mean return: %{y:+.4f}%"
                          "<br>5-day return: %{customdata:+.2f}%<extra></extra>"))
    fig.add_vline(x=d["std_return_pct"].median(), line_color=config.BASELINE, line_width=1)
    fig.add_hline(y=d["mean_return_pct"].median(), line_color=config.BASELINE, line_width=1)
    _plotly_layout(fig, "Risk vs return (quadrants split at the medians)", height=560, hovermode="closest")
    fig.update_xaxes(title_text="Risk: standard deviation of returns (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Return: mean periodic return (%)", ticksuffix="%")
    return fig


def _coin_selector_figure(df: pd.DataFrame, coin_ids: list[str], builder, title_fn, height: int = 480):
    """One figure holding every coin's traces with a dropdown that toggles visibility."""
    go = _go()
    fig = go.Figure()
    counts = []
    for i, coin_id in enumerate(coin_ids):
        traces = builder(coin_id)
        for trace in traces:
            trace.visible = i == 0
            fig.add_trace(trace)
        counts.append(len(traces))
    total, start, buttons = sum(counts), 0, []
    for coin_id, count in zip(coin_ids, counts):
        visible = [False] * total
        visible[start:start + count] = [True] * count
        start += count
        buttons.append(dict(label=_coin_title(df, coin_id), method="update",
                            args=[{"visible": visible}, {"title.text": title_fn(coin_id)}]))
    _plotly_layout(fig, title_fn(coin_ids[0]), height=height)
    fig.update_layout(updatemenus=[dict(buttons=buttons, direction="down", showactive=True, x=1, xanchor="right",
                                        y=1.18, yanchor="bottom", bgcolor=config.SURFACE,
                                        bordercolor=config.BASELINE)])
    fig.update_xaxes(title_text="Time (UTC, 24/7)")
    return fig


def build_interactive_report(df: pd.DataFrame, stats: pd.DataFrame, save_path: str | Path,
                             threshold: float = 2.0, top_k: int = 10, windows=(3, 5),
                             fence: float | None = None, subtitle: str = "") -> Path:
    """Write a self-contained HTML page with every interactive chart and a coin selector."""
    coin_ids = stats.sort_values("market_cap_rank")["coin_id"].tolist()
    price_fig = _coin_selector_figure(
        df, coin_ids, lambda c: price_traces(df, c, include_ma=True, windows=windows),
        lambda c: f"{_coin_title(df, c)} — price, MA({windows[0]}) and MA({windows[1]})")
    price_fig.update_yaxes(title_text="Price (USD)", tickprefix="$")
    outlier_fig = _coin_selector_figure(
        df, coin_ids, lambda c: outlier_traces(df, c, threshold),
        lambda c: f"{_coin_title(df, c)} — returns and Z-score outliers (|Z| > {threshold:g})")
    outlier_fig.update_layout(hovermode="closest")
    outlier_fig.update_yaxes(title_text="Return per period (%)", ticksuffix="%")

    figures = [
        ("Price trend and moving averages", "Pick any coin from the dropdown (top right of the chart). "
         "Click legend entries to hide MA lines and see the price alone.", price_fig),
        ("Returns and outliers", "Triangles mark |Z| above the threshold; larger triangles are extreme (|Z| > 3).",
         outlier_fig),
        ("Volatility ranking", "Standard deviation of percentage returns.",
         plotly_volatility_figure(stats, top_k, fence)),
        ("Risk vs return", "Hover a point for details.", plotly_risk_return_figure(stats)),
        ("Market capitalization", "Logarithmic scale.", plotly_market_cap_figure(stats)),
    ]
    sections = []
    for i, (heading, note, fig) in enumerate(figures):
        html = fig.to_html(full_html=False, include_plotlyjs=(i == 0),
                           config={"displaylogo": False, "responsive": True})
        sections.append(f"<section><h2>{heading}</h2><p class='note'>{note}</p>{html}</section>")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Crypto Market Analysis — Interactive Charts</title>
<style>
 body {{ margin:0; padding:24px 16px; background:{config.PAGE}; color:{config.INK_PRIMARY};
        font-family:"Segoe UI", system-ui, -apple-system, sans-serif; }}
 main {{ max-width:1180px; margin:0 auto; }}
 h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:0 0 2px; }}
 .sub, .note {{ color:{config.INK_SECONDARY}; margin:4px 0 12px; }}
 section {{ background:{config.SURFACE}; border:1px solid rgba(11,11,11,0.10); border-radius:10px;
            padding:16px; margin:16px 0; overflow-x:auto; }}
 .disclaimer {{ font-size:13px; color:{config.INK_SECONDARY}; border-left:3px solid {config.BASELINE};
               padding-left:10px; }}
</style></head><body><main>
<h1>Cryptocurrency Market Volatility &amp; Trend Analysis</h1>
<p class="sub">{subtitle}</p>
<p class="disclaimer">Statistical analysis of historical market data for educational and research purposes.
Trend labels, volatility rankings and outlier flags describe past data; they are not financial advice or
trading recommendations.</p>
{''.join(sections)}
</main></body></html>"""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path
