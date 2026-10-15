"""Tests for the chart builders (Matplotlib and Plotly).

These assert the structure a reader depends on - the right number of marks, the
series that must be present, labelled axes, saved files - rather than pixels.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: never open a window during tests

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from src import config  # noqa: E402
from src import visualization as viz
from src.formatting import compact_usd, format_price  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def frames(analysis_frames):
    cfg, analysis, summary = analysis_frames
    viz.apply_style()
    return cfg, analysis, summary


# ---------------------------------------------------------------------------
# Matplotlib
# ---------------------------------------------------------------------------
def test_market_cap_chart_has_one_bar_per_coin_and_saves(frames, tmp_path):
    _, _, summary = frames
    path = tmp_path / "market_cap.png"
    fig = viz.plot_market_cap(summary, save_path=path)
    ax = fig.axes[0]
    assert len(ax.patches) == len(summary)                     # one bar per coin
    assert ax.get_xscale() == "log"                            # wide range needs a log axis
    assert len(ax.get_yticklabels()) == len(summary)           # every coin labelled
    assert path.exists() and path.stat().st_size > 0


def test_price_trend_plots_the_whole_series(frames):
    _, analysis, _ = frames
    coin = "bitcoin"
    fig = viz.plot_price_trend(analysis, coin, days=5)
    ax = fig.axes[0]
    expected = int((analysis["coin_id"] == coin).sum())
    assert len(ax.lines[0].get_ydata()) == expected
    assert "Bitcoin" in ax.get_title(loc="left")
    assert "Price" in ax.get_ylabel()


def test_moving_average_chart_draws_price_and_both_averages(frames):
    cfg, analysis, _ = frames
    fig = viz.plot_moving_averages(analysis, "ethereum", cfg.ma_windows)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ["Price", "MA(3)", "MA(5)"]
    assert len(fig.axes) == 2                                   # full window + recent zoom
    assert "Zoom" in fig.axes[1].get_title(loc="left")


def test_moving_average_subtitle_reports_the_trend(frames):
    cfg, analysis, summary = frames
    fig = viz.plot_moving_averages(analysis, "solana", cfg.ma_windows)
    texts = " ".join(t.get_text() for t in fig.axes[0].texts)
    expected = summary.loc[summary["coin_id"] == "solana", "trend"].iloc[0]
    assert expected in texts                                    # chart agrees with the summary table
    assert "not a trading recommendation" in texts              # required framing


def test_volatility_chart_highlights_the_top_k(frames):
    _, _, summary = frames
    top_k = 2
    fig = viz.plot_volatility_ranking(summary, top_k=top_k, fence=1.0)
    ax = fig.axes[0]
    colors = [p.get_facecolor() for p in ax.patches]
    highlighted = matplotlib.colors.to_rgba(config.SERIES_1)
    assert sum(c == highlighted for c in colors) == top_k       # only the top k are emphasised
    assert len(ax.patches) == len(summary)


def test_volatility_bars_are_sorted_descending(frames):
    _, _, summary = frames
    fig = viz.plot_volatility_ranking(summary, top_k=3, fence=None)
    widths = [p.get_width() for p in fig.axes[0].patches]
    assert widths == sorted(widths, reverse=True)


def test_outlier_chart_marks_positive_and_negative_separately(frames):
    cfg, analysis, _ = frames
    coin = "solana"
    fig = viz.plot_outliers(analysis, coin, cfg.z_threshold, cfg.strict_z_threshold)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("Positive outlier" in label for label in labels)
    assert any("Negative outlier" in label for label in labels)
    coin_rows = analysis.loc[analysis["coin_id"] == coin]
    plotted = sum(len(c.get_offsets()) for c in fig.axes[0].collections)
    assert plotted >= int(coin_rows["is_outlier"].sum())        # every outlier is drawn


def test_risk_return_scatter_plots_every_coin(frames):
    _, _, summary = frames
    fig = viz.plot_risk_return(summary)
    ax = fig.axes[0]
    plotted = sum(len(c.get_offsets()) for c in ax.collections)
    assert plotted == len(summary)
    assert "Risk" in ax.get_xlabel() and "Return" in ax.get_ylabel()
    assert len(ax.get_legend().get_texts()) >= 1


def test_charts_use_the_shared_palette(frames):
    _, analysis, _ = frames
    fig = viz.plot_price_trend(analysis, "bitcoin")
    assert fig.axes[0].lines[0].get_color() == config.SERIES_1
    assert fig.get_facecolor() == matplotlib.colors.to_rgba(config.SURFACE)


def test_coin_slice_rejects_unknown_coins(frames):
    _, analysis, _ = frames
    with pytest.raises(ValueError, match="No data for coin"):
        viz.coin_slice(analysis, "not-a-coin")


# ---------------------------------------------------------------------------
# Formatting helpers used by the charts
# ---------------------------------------------------------------------------
def test_price_formatting_adapts_to_magnitude():
    assert format_price(76350.4) == "$76,350"          # no cents on large prices
    assert format_price(2.5) == "$2.50"
    assert format_price(0.000012345) == "$0.00001234"  # 4 significant digits, not 0.00
    assert format_price(None) == "n/a"


def test_compact_usd_uses_magnitude_suffixes():
    assert compact_usd(1.52e12) == "$1.52T"
    assert compact_usd(8.5e8) == "$850.00M"
    assert compact_usd(float("nan")) == "n/a"


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------
def test_plotly_price_and_moving_average_traces(frames):
    cfg, analysis, _ = frames
    assert len(viz.plotly_price_figure(analysis, "bitcoin").data) == 1
    ma_fig = viz.plotly_moving_average_figure(analysis, "bitcoin", cfg.ma_windows)
    assert [t.name for t in ma_fig.data] == ["Price", "MA(3)", "MA(5)"]


def test_plotly_outlier_figure_uses_direction_marker_shapes(frames):
    _, analysis, _ = frames
    fig = viz.plotly_outlier_figure(analysis, "solana")
    symbols = {t.marker.symbol for t in fig.data if t.mode == "markers"}
    assert symbols == {"triangle-up", "triangle-down"}          # shape, not colour alone


def test_plotly_cross_section_figures(frames):
    _, _, summary = frames
    assert len(viz.plotly_market_cap_figure(summary).data[0].x) == len(summary)
    assert len(viz.plotly_volatility_figure(summary, 3, 1.0).data[0].x) == len(summary)
    points = sum(len(t.x) for t in viz.plotly_risk_return_figure(summary).data)
    assert points == len(summary)


def test_interactive_report_has_a_selector_for_every_coin(frames, tmp_path):
    cfg, analysis, summary = frames
    path = viz.build_interactive_report(analysis, summary, tmp_path / "charts.html",
                                        cfg.z_threshold, cfg.top_k, cfg.ma_windows, fence=1.0,
                                        subtitle="test run")
    html = path.read_text(encoding="utf-8")
    assert html.count("<section>") == 5                          # one per chart family
    assert html.count("Plotly.newPlot") == 5
    assert "not financial advice" in html.lower()
    for name in summary["name"]:
        assert name in html                                      # every coin selectable
