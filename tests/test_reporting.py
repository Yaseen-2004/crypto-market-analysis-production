"""Tests for the summary tables, findings and the Markdown report.

The report is the deliverable a reader actually reads, so these check that its
numbers agree with the data and that its tables are valid Markdown.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src import preprocessing, reporting
from src.outlier_detection import material_outliers


@pytest.fixture
def report_inputs(analysis_frames):
    """Everything ``write_markdown_report`` needs, built the way the pipeline builds it."""
    cfg, analysis, summary = analysis_frames
    from src import outlier_detection
    from src import statistics as coin_stats

    outliers = outlier_detection.build_outlier_report(analysis)
    findings = reporting.key_findings(summary, analysis, cfg)
    fence = coin_stats.volatility_fence(summary)
    _, prep = preprocessing.preprocess(analysis, cfg.days, cfg.min_observations)
    report_args = dict(
        cfg=cfg, findings=findings, summary=summary,
        top_most=coin_stats.top_volatile(summary, 3, most=True),
        top_least=coin_stats.top_volatile(summary, 3, most=False),
        outlier_report=outliers, prep_report=prep,
        fetch_log=pd.DataFrame({"coin_id": summary["coin_id"], "name": summary["name"], "status": "ok"}),
        fence=fence, figures={}, fetched_at=datetime.now(timezone.utc),
    )
    return cfg, analysis, summary, outliers, findings, fence, report_args


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def test_summary_table_matches_the_specified_columns(report_inputs):
    cfg, _, summary, _, _, _, _ = report_inputs
    table = reporting.format_summary_table(summary, cfg.ma_windows)
    required = ["Rank", "Cryptocurrency", "Symbol", "Price (USD)", "5-Day Return (%)", "Mean Return (%)",
                "Std Dev (%)", "MA(3)", "MA(5)", "Trend", "Outlier Count"]
    assert list(table.columns)[:len(required)] == required
    assert len(table) == len(summary)


def test_summary_values_come_straight_from_the_statistics(report_inputs):
    cfg, _, summary, _, _, _, _ = report_inputs
    table = reporting.format_summary_table(summary, cfg.ma_windows)
    row = summary.iloc[0]
    out = table.iloc[0]
    assert out["Cryptocurrency"] == row["name"]
    assert out["Std Dev (%)"] == pytest.approx(row["std_return_pct"])
    assert out["5-Day Return (%)"] == pytest.approx(row["five_day_return_pct"])
    assert out["Outlier Count"] == row["outlier_count"]


def test_assemble_summary_gives_coins_without_outliers_a_zero(analysis_frames):
    from src.reporting import assemble_coin_summary

    _, _, summary = analysis_frames
    stats = summary.drop(columns=["outlier_count", "extreme_outlier_count", "positive_outliers",
                                  "negative_outliers", "trend", "price_vs_ma_long_pct", "ma_3", "ma_5"])
    trend = pd.DataFrame({"coin_id": summary["coin_id"], "ma_3": 1.0, "ma_5": 1.0,
                          "trend": "Neutral", "price_vs_ma_long_pct": 0.0})
    counts = pd.DataFrame({"coin_id": [summary["coin_id"].iloc[0]], "outlier_count": [3],
                           "extreme_outlier_count": [1], "positive_outliers": [2], "negative_outliers": [1]})
    merged = assemble_coin_summary(stats, trend, counts)
    assert merged["outlier_count"].iloc[0] == 3
    assert (merged["outlier_count"].iloc[1:] == 0).all()        # missing counts become 0, not NaN
    assert merged["outlier_count"].dtype.kind in "iu"


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------
def test_key_findings_agree_with_the_data(report_inputs):
    _, analysis, summary, _, findings, _, _ = report_inputs
    assert findings["n_coins"] == len(summary)
    assert findings["n_observations"] == len(analysis)
    assert findings["total_outliers"] == int(analysis["is_outlier"].sum())
    assert findings["extreme_outliers"] == int(analysis["is_extreme_outlier"].sum())
    assert findings["positive_outliers"] + findings["negative_outliers"] == findings["total_outliers"]
    assert findings["most_volatile"]["std_return_pct"] == summary["std_return_pct"].max()
    assert findings["least_volatile"]["std_return_pct"] == summary["std_return_pct"].min()
    assert findings["best_return"]["five_day_return_pct"] == summary["five_day_return_pct"].max()
    assert findings["n_up"] == int((summary["five_day_return_pct"] > 0).sum())


def test_normal_baseline_rates_are_the_textbook_values(report_inputs):
    _, _, _, _, findings, _, _ = report_inputs
    assert findings["normal_outlier_rate"] == pytest.approx(4.55, abs=0.01)    # P(|Z|>2)
    assert findings["normal_extreme_rate"] == pytest.approx(0.27, abs=0.01)    # P(|Z|>3)


def test_findings_bullets_mention_the_headline_facts(report_inputs):
    cfg, _, _, _, findings, fence, _ = report_inputs
    text = " ".join(reporting.findings_bullets(findings, cfg, fence))
    assert findings["most_volatile"]["name"] in text
    assert findings["largest"]["name"] in text
    assert "Near-constant assets" in text                       # the stablecoin caveat is always stated


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _markdown_table_rows(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("|")]


def test_markdown_report_is_written_with_every_section(report_inputs, tmp_path):
    *_, report_args = report_inputs
    path = reporting.write_markdown_report(tmp_path / "report.md", **report_args)
    text = path.read_text(encoding="utf-8")
    for heading in ("# Cryptocurrency Market Volatility & Trend Analysis", "## 1. Key findings",
                    "## 4. Outlier report", "## 6. Statistical summary", "## 9. Methodology",
                    "## 10. Limitations"):
        assert heading in text
    assert "not financial advice" in text.lower()


def test_markdown_tables_have_no_unescaped_pipes(report_inputs, tmp_path):
    """Regression: a label like 'Outliers (|Z|>2)' silently breaks Markdown tables."""
    *_, report_args = report_inputs
    path = reporting.write_markdown_report(tmp_path / "report.md", **report_args)
    text = path.read_text(encoding="utf-8")

    rows = _markdown_table_rows(text)
    assert rows, "the report should contain Markdown tables"
    assert not any("|Z|" in row for row in rows)                # never inside a table cell
    # Every row of a table must have the same cell count as its header.
    header_cells, checked = None, 0
    for row in rows:
        cells = row.count("|")
        if header_cells is None:
            header_cells = cells
        elif set(row) <= set("|-: "):                           # separator row ends a header
            assert cells == header_cells
        elif cells != header_cells:
            header_cells = cells                                # a new table started
        checked += 1
    assert checked == len(rows)


def test_report_outlier_table_excludes_near_constant_assets(report_inputs, tmp_path):
    *_, outliers, _, _, report_args = report_inputs
    path = reporting.write_markdown_report(tmp_path / "report.md", **report_args)
    text = path.read_text(encoding="utf-8")
    material = material_outliers(outliers)
    assert f"{len(material):,} material" in text
    assert "Largest |Z| overall, including near-constant assets" in text


def test_console_summary_prints_the_headlines(report_inputs, capsys):
    cfg, _, _, outliers, findings, fence, report_args = report_inputs
    reporting.print_console_summary(findings, cfg, fence, report_args["top_most"], report_args["top_least"],
                                    outliers)
    printed = capsys.readouterr().out
    assert "CRYPTOCURRENCY MARKET VOLATILITY & TREND ANALYSIS" in printed
    assert "KEY FINDINGS" in printed
    assert "LARGEST MATERIAL OUTLIERS" in printed
    assert "Not financial advice" in printed
    assert "**" not in printed                                  # Markdown bold stripped for the terminal


def test_sort_summary_is_stable_on_ties():
    summary = pd.DataFrame({"market_cap_rank": [3, 1, 2], "std_return_pct": [1.0, 1.0, 1.0],
                            "five_day_return_pct": [0.0, 0.0, 0.0], "outlier_count": [0, 0, 0]})
    assert reporting.sort_summary(summary, "volatility")["market_cap_rank"].tolist() == [1, 2, 3]
