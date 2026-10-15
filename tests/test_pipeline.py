"""End-to-end tests for the pipeline, using a fake API client (no network).

These run the real orchestration - fetch, clean, analyse, chart, report, store -
against synthetic payloads, so an orchestration regression is caught in seconds
instead of needing a 20-minute rate-limited download.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src import config, database  # noqa: E402
from src.config import AnalysisConfig  # noqa: E402
from src.main import resolve_coins, run_pipeline  # noqa: E402
from tests.helpers import FakeCoinGeckoClient  # noqa: E402


def make_config(**overrides) -> AnalysisConfig:
    defaults = dict(top_n=6, days=5, min_observations=24, make_plots=False, make_interactive=False,
                    use_database=False)
    defaults.update(overrides)
    return AnalysisConfig(**defaults)


@pytest.fixture
def pipeline(isolated_paths):
    """Run the full pipeline offline; returns (result, tmp_root)."""
    def _run(**overrides):
        client = overrides.pop("client", None) or FakeCoinGeckoClient()
        return run_pipeline(make_config(**overrides), client=client), isolated_paths
    return _run


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def test_pipeline_produces_every_expected_file(pipeline):
    result, root = pipeline()
    expected = [config.RAW_MARKETS_CSV, config.RAW_HISTORY_CSV, config.FETCH_LOG_CSV, config.CLEAN_CSV,
                config.DAILY_RETURNS_CSV, config.PREPROCESSING_REPORT_CSV, config.ANALYSIS_DATASET_CSV,
                config.SUMMARY_REPORT_CSV, config.COIN_STATISTICS_CSV, config.VOLATILITY_RANKING_CSV,
                config.TOP_VOLATILE_CSV, config.LEAST_VOLATILE_CSV, config.OUTLIER_REPORT_CSV,
                config.ANOMALOUS_COINS_CSV, config.MARKDOWN_REPORT]
    for path in expected:
        assert path.exists() and path.stat().st_size > 0, f"missing output: {path.name}"
    assert root in config.SUMMARY_REPORT_CSV.parents        # nothing written outside the tmp dir
    assert len(result.summary) == 6


def test_pipeline_results_are_internally_consistent(pipeline):
    result, _ = pipeline()
    analysis, summary = result.analysis, result.summary
    assert set(summary["coin_id"]) == set(analysis["coin_id"])
    assert summary["outlier_count"].sum() == int(analysis["is_outlier"].sum())
    assert result.findings["n_coins"] == len(summary)
    assert len(result.outlier_report) == int(analysis["is_outlier"].sum())
    assert result.volatility_ranking["Standard Deviation (%)"].is_monotonic_decreasing
    assert len(result.top_most) == len(result.top_least) == result.cfg.top_k or len(summary) < result.cfg.top_k


def test_summary_csv_is_written_in_the_requested_order(pipeline):
    result, _ = pipeline(sort_by="volatility")
    table = pd.read_csv(config.SUMMARY_REPORT_CSV)
    assert table["Std Dev (%)"].is_monotonic_decreasing
    assert len(table) == len(result.summary)

    result, _ = pipeline(sort_by="rank")
    table = pd.read_csv(config.SUMMARY_REPORT_CSV)
    assert table["Rank"].is_monotonic_increasing


def test_analysis_dataset_carries_every_derived_column(pipeline):
    pipeline()
    dataset = pd.read_csv(config.ANALYSIS_DATASET_CSV)
    for column in ("price", "pct_return", "ma_3", "ma_5", "trend", "z_score", "is_outlier",
                   "is_extreme_outlier", "outlier_type", "low_variation"):
        assert column in dataset.columns


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def test_pipeline_generates_the_six_graph_families(pipeline):
    result, _ = pipeline(make_plots=True, make_interactive=True)
    written = {p.name for paths in result.figures.values() for p in paths}
    assert any(n.startswith("01_market_cap") for n in written)
    assert any(n.startswith("04_volatility_ranking") for n in written)
    assert any(n.startswith("06_risk_vs_return") for n in written)
    for coin in result.selected_coins:                       # per-coin graphs 2, 3 and 5
        assert f"02_price_trend_{coin}.png" in written
        assert f"03_moving_average_{coin}.png" in written
        assert f"05_outliers_{coin}.png" in written
    assert config.INTERACTIVE_REPORT_HTML.exists()
    assert all(p.exists() for paths in result.figures.values() for p in paths)


def test_rerun_clears_stale_figures(pipeline, isolated_paths):
    pipeline(make_plots=True)
    stale = config.FIGURES_DIR / "02_price_trend_delisted.png"
    stale.write_bytes(b"stale")
    pipeline(make_plots=True)
    assert not stale.exists()                                # old per-coin charts are removed


# ---------------------------------------------------------------------------
# Coin selection
# ---------------------------------------------------------------------------
def test_default_coins_skip_stablecoins(pipeline):
    result, _ = pipeline()
    assert "tether" not in result.selected_coins             # a peg-like coin is a poor default chart
    assert result.selected_coins == ["bitcoin", "ethereum", "solana"]


def test_coins_can_be_chosen_by_id_symbol_or_name(pipeline):
    result, _ = pipeline(selected_coins=["dogecoin", "ETH", "Solana"])
    assert result.selected_coins == ["dogecoin", "ethereum", "solana"]


def test_unknown_coins_are_skipped_with_a_warning(pipeline, caplog):
    result, _ = pipeline(selected_coins=["bitcoin", "not-a-coin"])
    assert result.selected_coins == ["bitcoin"]
    assert "not-a-coin" in caplog.text


def test_resolve_coins_falls_back_when_nothing_matches():
    summary = pd.DataFrame({"coin_id": ["a", "b", "c"], "symbol": ["A", "B", "C"],
                            "name": ["Alpha", "Beta", "Gamma"], "market_cap_rank": [1, 2, 3],
                            "peg_like": [False, False, False]})
    assert resolve_coins(summary, ["nope"], n_default=2) == ["a", "b"]
    assert resolve_coins(summary, [], n_default=1) == ["a"]
    assert resolve_coins(summary, ["gamma", "GAMMA"], n_default=2) == ["c"]   # case-insensitive, deduped


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------
def test_a_failing_coin_is_logged_and_skipped(pipeline):
    client = FakeCoinGeckoClient(fail={"dogecoin"})
    result, _ = pipeline(client=client)
    assert "dogecoin" not in set(result.summary["coin_id"])   # excluded from the analysis
    assert len(result.summary) == 5
    log = pd.read_csv(config.FETCH_LOG_CSV)
    failed = log.loc[log["status"] == "failed"]
    assert failed["coin_id"].tolist() == ["dogecoin"]
    assert client.requested.count("dogecoin") == 2            # retried once after the cool-down


def test_a_newly_listed_coin_with_too_little_history_is_excluded(pipeline):
    """A coin with only a few hours of data is dropped; the rest are analysed normally."""
    client = FakeCoinGeckoClient(short={"dogecoin": 10})
    result, _ = pipeline(client=client, min_observations=24)
    assert "dogecoin" in result.prep_report.coins_excluded
    assert "dogecoin" not in set(result.summary["coin_id"])
    assert len(result.summary) == 5                          # the others are unaffected


def test_too_little_data_everywhere_fails_with_a_clear_message(pipeline):
    """Regression: this used to crash with 'NaTType does not support strftime'."""
    client = FakeCoinGeckoClient(n_points=30)
    with pytest.raises(ValueError, match="No cryptocurrency has at least 40 valid observations"):
        pipeline(client=client, min_observations=40)


def test_cli_reports_insufficient_data_without_a_traceback(isolated_paths, monkeypatch, capsys):
    """The CLI must exit 1 with a readable message - the CLI configures its own log handler,
    so the message is asserted on stderr rather than through caplog."""
    from src import main as main_module

    def explode(*args, **kwargs):
        raise ValueError("No cryptocurrency has at least 40 valid observations")

    monkeypatch.setattr(main_module, "run_pipeline", explode)
    assert main_module.main(["--top-n", "6", "--no-plots", "--no-db"]) == 1
    assert "Analysis could not run" in capsys.readouterr().err


def test_cli_reports_a_failed_download_without_a_traceback(isolated_paths, monkeypatch, capsys):
    from src import data_fetcher
    from src import main as main_module

    def explode(*args, **kwargs):
        raise data_fetcher.CoinGeckoError("simulated network outage")

    monkeypatch.setattr(main_module, "run_pipeline", explode)
    assert main_module.main(["--no-plots", "--no-db"]) == 1
    assert "Data collection failed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def test_pipeline_persists_the_run_when_storage_is_enabled(pipeline):
    result, _ = pipeline(use_database=True)
    assert config.DATABASE_PATH.exists()
    assert result.run_id == 1
    with database.session(config.DATABASE_PATH) as conn:
        stored = database.totals(conn)
    assert stored["runs"] == 1
    assert stored["observations"] == len(result.analysis)
    assert stored["coins"] == len(result.summary)


def test_storage_can_be_switched_off(pipeline):
    result, _ = pipeline(use_database=False)
    assert not config.DATABASE_PATH.exists()
    assert result.run_id is None


def test_a_storage_failure_does_not_break_the_analysis(pipeline, monkeypatch, caplog):
    def explode(*args, **kwargs):
        raise sqlite_error("disk is full")

    from src import database as db_module

    class sqlite_error(RuntimeError):
        pass

    monkeypatch.setattr(db_module, "save_run", explode)
    result, _ = pipeline(use_database=True)
    assert result.summary is not None and len(result.summary) == 6    # analysis still completed
    assert config.MARKDOWN_REPORT.exists()                            # reports still written
    assert "Could not write to the database" in caplog.text
