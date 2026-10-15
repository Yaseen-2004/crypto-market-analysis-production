"""End-to-end pipeline.

Run from the project root:

    python main.py                          # full run: top 100, last 5 days
    python main.py --coins bitcoin eth sol  # choose coins for the per-coin charts
    python main.py --sort-by volatility     # sort the summary CSV
    python -m src.main --help

Steps: fetch -> download history -> clean -> returns -> moving averages ->
standard deviation -> outliers -> visualisations -> CSV/Markdown reports -> display.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    # Executed as a file (python src/main.py): make the project root importable and
    # drop src/ from sys.path so src/statistics.py cannot shadow the stdlib module.
    _SRC = Path(__file__).resolve().parent
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != _SRC]
    sys.path.insert(0, str(_SRC.parent))

import pandas as pd  # noqa: E402

from src import (  # noqa: E402
    config,
    data_fetcher,
    database,
    moving_average,
    outlier_detection,
    preprocessing,
    reporting,
)
from src import statistics as coin_stats  # noqa: E402
from src.config import AnalysisConfig  # noqa: E402

logger = logging.getLogger("crypto_analysis")


@dataclass
class PipelineResult:
    cfg: AnalysisConfig
    fetch: data_fetcher.FetchResult
    clean: pd.DataFrame
    prep_report: preprocessing.PreprocessingReport
    daily: pd.DataFrame
    analysis: pd.DataFrame          # clean data + MA, trend, Z-scores and outlier flags
    summary: pd.DataFrame           # one row per coin, all metrics
    volatility_ranking: pd.DataFrame
    top_most: pd.DataFrame
    top_least: pd.DataFrame
    outlier_report: pd.DataFrame
    fence: float
    findings: dict
    selected_coins: list[str]
    figures: dict[str, list[Path]] = field(default_factory=dict)
    outputs: dict[str, Path] = field(default_factory=dict)
    run_id: int | None = None       # row in the SQLite `runs` table, when storage is enabled


def _step(number: int, text: str) -> None:
    logger.info("Step %d/10 - %s", number, text)


def resolve_coins(summary: pd.DataFrame, requested: list[str] | None, n_default: int = 3) -> list[str]:
    """Map ids / symbols / names (case-insensitive) to coin ids.

    With nothing requested, the ``n_default`` largest non-pegged coins are used.
    """
    by_rank = summary.sort_values("market_cap_rank")
    if not requested:
        pool = by_rank.loc[~by_rank["peg_like"]] if "peg_like" in by_rank else by_rank
        return pool["coin_id"].head(n_default).astype(str).tolist()
    resolved: list[str] = []
    keys = {col: by_rank[col].astype(str).str.lower() for col in ("coin_id", "symbol", "name")}
    for query in requested:
        q = query.strip().lower()
        match = by_rank.loc[(keys["coin_id"] == q) | (keys["symbol"] == q) | (keys["name"] == q)]
        if match.empty:
            logger.warning("Coin '%s' is not among the analysed coins; skipping", query)
            continue
        coin_id = str(match["coin_id"].iloc[0])
        if coin_id not in resolved:
            resolved.append(coin_id)
    return resolved or resolve_coins(summary, None, n_default)


def generate_figures(cfg: AnalysisConfig, analysis: pd.DataFrame, summary: pd.DataFrame, coins: list[str],
                     fence: float, findings: dict) -> dict[str, list[Path]]:
    import matplotlib

    if not cfg.show_plots:
        matplotlib.use("Agg")  # headless: write files without opening windows
    import matplotlib.pyplot as plt

    from src import visualization as viz

    for old in config.FIGURES_DIR.glob("*.png"):  # remove charts from previous runs
        old.unlink()
    viz.apply_style()
    figures: dict[str, list[Path]] = {}

    def keep(key: str, fig, name: str) -> None:
        path = config.FIGURES_DIR / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        figures.setdefault(key, []).append(path)
        if not cfg.show_plots:
            plt.close(fig)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    keep("market_cap", viz.plot_market_cap(summary), "01_market_cap_top100.png")
    for coin_id in coins:
        keep("price_trend", viz.plot_price_trend(analysis, coin_id, cfg.days), f"02_price_trend_{coin_id}.png")
        keep("moving_average", viz.plot_moving_averages(analysis, coin_id, cfg.ma_windows),
             f"03_moving_average_{coin_id}.png")
    keep("volatility", viz.plot_volatility_ranking(summary, cfg.top_k, fence), "04_volatility_ranking.png")
    for coin_id in coins:
        keep("outliers", viz.plot_outliers(analysis, coin_id, cfg.z_threshold, cfg.strict_z_threshold),
             f"05_outliers_{coin_id}.png")
    keep("risk_return", viz.plot_risk_return(summary), "06_risk_vs_return.png")

    if cfg.make_interactive:
        try:
            subtitle = (f"{findings['n_coins']} coins · {findings['window_start']:%Y-%m-%d %H:%M} → "
                        f"{findings['window_end']:%Y-%m-%d %H:%M} UTC · data: CoinGecko")
            path = viz.build_interactive_report(analysis, summary, config.INTERACTIVE_REPORT_HTML, cfg.z_threshold,
                                                cfg.top_k, cfg.ma_windows, fence, subtitle)
            figures["interactive"] = [path]
        except ImportError:
            logger.warning("Plotly not installed; interactive HTML report skipped")
    if cfg.show_plots:
        plt.show()
    return figures


def run_pipeline(cfg: AnalysisConfig | None = None, client: data_fetcher.CoinGeckoClient | None = None,
                 save_outputs: bool = True) -> PipelineResult:
    """Run every analysis step and return all intermediate results."""
    cfg = cfg or AnalysisConfig()
    for directory in (config.RAW_DIR, config.PROCESSED_DIR, config.REPORTS_DIR, config.FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    short, long = cfg.ma_windows

    # 1-2) data collection ---------------------------------------------------
    _step(1, f"Fetching the current top {cfg.top_n} cryptocurrencies by market cap")
    _step(2, f"Downloading the latest {cfg.days} days of history for each coin (rate-limited, please wait)")
    fetch = data_fetcher.fetch_market_data(cfg.top_n, cfg.days, cfg.vs_currency, cfg.cache_minutes, client=client,
                                           history_cache_minutes=cfg.history_cache_minutes)
    if save_outputs:
        outputs["raw_markets"] = reporting.save_csv(fetch.markets, config.RAW_MARKETS_CSV)
        outputs["raw_history"] = reporting.save_csv(fetch.history, config.RAW_HISTORY_CSV)
        outputs["fetch_log"] = reporting.save_csv(fetch.log, config.FETCH_LOG_CSV)

    # 3) cleaning -------------------------------------------------------------
    _step(3, "Cleaning data (types, timestamps, duplicates, missing values, window)")
    clean, prep = preprocessing.preprocess(fetch.history, cfg.days, cfg.min_observations, cfg.max_interpolation_gap)

    # 4) returns ------------------------------------------------------------
    _step(4, "Calculating periodic returns")
    daily = preprocessing.compute_daily_returns(clean)
    if save_outputs:
        outputs["clean"] = reporting.save_csv(clean, config.CLEAN_CSV)
        outputs["daily"] = reporting.save_csv(daily, config.DAILY_RETURNS_CSV)
        outputs["prep_report"] = reporting.save_csv(prep.to_frame(), config.PREPROCESSING_REPORT_CSV)

    # 5) moving averages ----------------------------------------------------
    _step(5, f"Calculating MA({short}) and MA({long}) and trend labels")
    analysis = moving_average.add_moving_averages(clean, cfg.ma_windows)
    analysis = moving_average.add_trend(analysis, short, long)
    trend_snapshot = moving_average.latest_trend_snapshot(analysis, short, long)

    # 6) volatility -----------------------------------------------------------
    _step(6, "Calculating standard deviation and volatility rankings")
    stats = coin_stats.compute_coin_statistics(analysis)
    stats = coin_stats.rank_by_volatility(stats)
    stats = coin_stats.flag_unusual_volatility(stats)
    stats = coin_stats.flag_peg_like(stats)
    stats = coin_stats.flag_low_variation(stats)
    stats = coin_stats.assign_risk_return_quadrant(stats)
    fence = coin_stats.volatility_fence(stats)
    # Carry the flag onto every observation so the outlier report can mark them.
    analysis["low_variation"] = analysis["coin_id"].map(
        stats.set_index("coin_id")["low_variation"]).fillna(False)

    # 7) outliers -------------------------------------------------------------
    _step(7, f"Detecting outliers (|Z| > {cfg.z_threshold:g}, extreme |Z| > {cfg.strict_z_threshold:g})")
    analysis = outlier_detection.add_zscores(analysis)
    analysis = outlier_detection.flag_outliers(analysis, cfg.z_threshold, cfg.strict_z_threshold)
    counts = outlier_detection.count_outliers(analysis)
    outlier_report = outlier_detection.build_outlier_report(analysis)

    summary = reporting.assemble_coin_summary(stats, trend_snapshot, counts)
    summary = outlier_detection.detect_anomalous_coins(summary, random_state=cfg.random_state)
    top_most = coin_stats.top_volatile(summary, cfg.top_k, most=True)
    top_least = coin_stats.top_volatile(summary, cfg.top_k, most=False)
    volatility_ranking = coin_stats.volatility_table(summary)
    findings = reporting.key_findings(summary, analysis, cfg)
    coins = resolve_coins(summary, cfg.selected_coins, cfg.n_default_coins)

    result = PipelineResult(cfg, fetch, clean, prep, daily, analysis, summary, volatility_ranking, top_most,
                            top_least, outlier_report, fence, findings, coins, outputs=outputs)

    # 8) visualisations -----------------------------------------------------
    if cfg.make_plots:
        _step(8, f"Generating visualisations (per-coin charts for: {', '.join(coins)})")
        result.figures = generate_figures(cfg, analysis, summary, coins, fence, findings)
    else:
        _step(8, "Visualisations skipped (--no-plots)")

    # 9) reports --------------------------------------------------------------
    _step(9, "Writing summary CSV files and the analytical report")
    if save_outputs:
        dataset = analysis.copy()
        dataset["timestamp"] = dataset["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        outputs["analysis_dataset"] = reporting.save_csv(dataset, config.ANALYSIS_DATASET_CSV)
        table = reporting.format_summary_table(reporting.sort_summary(summary, cfg.sort_by), cfg.ma_windows)
        outputs["summary"] = reporting.save_csv(table, config.SUMMARY_REPORT_CSV)
        stats_out = summary.copy()
        for col in ("start_time_utc", "end_time_utc"):
            stats_out[col] = stats_out[col].dt.strftime("%Y-%m-%d %H:%M")
        outputs["coin_statistics"] = reporting.save_csv(stats_out, config.COIN_STATISTICS_CSV)
        outputs["volatility_ranking"] = reporting.save_csv(volatility_ranking, config.VOLATILITY_RANKING_CSV)
        outputs["top_most"] = reporting.save_csv(top_most, config.TOP_VOLATILE_CSV)
        outputs["top_least"] = reporting.save_csv(top_least, config.LEAST_VOLATILE_CSV)
        outputs["outliers"] = reporting.save_csv(outlier_report, config.OUTLIER_REPORT_CSV)
        anomalies = summary.sort_values("anomaly_score", ascending=False)[
            ["market_cap_rank", "name", "symbol", "anomaly_score", "is_anomalous_coin", "std_return_pct",
             "five_day_return_pct", "max_abs_return_pct", "max_drawdown_pct"]]
        outputs["anomalies"] = reporting.save_csv(anomalies, config.ANOMALOUS_COINS_CSV)
        if cfg.use_database:
            try:
                with database.session(config.DATABASE_PATH) as conn:
                    run_id = database.save_run(conn, cfg=cfg, analysis=analysis, summary=summary,
                                               markets=fetch.markets)
                    stored = database.totals(conn)
                result.run_id = run_id
                outputs["database"] = config.DATABASE_PATH
                logger.info(
                    "Database now holds %d run(s), %s observations for %d coins spanning %.1f days "
                    "(the API window alone is %d days)",
                    stored["runs"], f"{stored['observations']:,}", stored["coins"], stored["days_covered"],
                    cfg.days)
            except Exception as exc:  # storage must never break the analysis
                logger.warning("Could not write to the database: %s", exc)

        outputs["markdown"] = reporting.write_markdown_report(
            config.MARKDOWN_REPORT, cfg=cfg, findings=findings, summary=summary, top_most=top_most,
            top_least=top_least, outlier_report=outlier_report, prep_report=prep, fetch_log=fetch.log,
            fence=fence, figures=result.figures, fetched_at=fetch.fetched_at)
    return result


def parse_args(argv: list[str] | None = None) -> AnalysisConfig:
    parser = argparse.ArgumentParser(description="Cryptocurrency market volatility & trend analysis (CoinGecko).")
    parser.add_argument("--top-n", type=int, default=100, help="number of coins by market cap (default 100)")
    parser.add_argument("--days", type=int, default=5, help="history window in days (default 5)")
    parser.add_argument("--coins", nargs="*", default=[],
                        help="coins for per-coin charts: ids, symbols or names (default: 3 largest non-stablecoins)")
    parser.add_argument("--sort-by", choices=list(reporting.SORT_OPTIONS), default="rank",
                        help="sort order of the summary CSV")
    parser.add_argument("--z-threshold", type=float, default=2.0, help="outlier threshold for |Z| (default 2)")
    parser.add_argument("--strict-z-threshold", type=float, default=3.0, help="extreme-outlier threshold (default 3)")
    parser.add_argument("--min-observations", type=int, default=24, help="minimum price points per coin")
    parser.add_argument("--cache-minutes", type=float, default=0.0,
                        help="reuse API responses younger than this many minutes (default 0 = always fresh)")
    parser.add_argument("--no-plots", action="store_true", help="skip all charts")
    parser.add_argument("--no-interactive", action="store_true", help="skip the Plotly HTML report")
    parser.add_argument("--no-db", action="store_true",
                        help="do not append this run to data/crypto_market.db")
    parser.add_argument("--show", action="store_true", help="open Matplotlib windows after saving")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)
    if args.top_n < 2 or args.days < 1:
        parser.error("--top-n must be >= 2 and --days >= 1")

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S", force=True)
    if not args.verbose:
        for noisy in ("urllib3", "matplotlib", "PIL"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    return AnalysisConfig(
        top_n=args.top_n, days=args.days, selected_coins=args.coins, sort_by=args.sort_by,
        z_threshold=args.z_threshold, strict_z_threshold=args.strict_z_threshold,
        min_observations=args.min_observations, cache_minutes=args.cache_minutes,
        make_plots=not args.no_plots, make_interactive=not args.no_interactive, show_plots=args.show,
        use_database=not args.no_db,
    )


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    cfg = parse_args(argv)
    started = time.monotonic()
    try:
        result = run_pipeline(cfg)
    except data_fetcher.CoinGeckoError as exc:
        logger.error("Data collection failed: %s", exc)
        logger.error("Check your internet connection or try again in a minute (CoinGecko rate limits). "
                     "Setting COINGECKO_API_KEY raises the limits.")
        return 1
    except ValueError as exc:  # not enough usable data - report it plainly, without a traceback
        logger.error("Analysis could not run: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Interrupted by user")
        return 130

    _step(10, "Displaying the final analysis")
    reporting.print_console_summary(result.findings, cfg, result.fence, result.top_most, result.top_least,
                                    result.outlier_report)
    print("\nOutputs:")
    for path in list(result.outputs.values()) + [p for paths in result.figures.values() for p in paths]:
        print(f"  {Path(path).relative_to(config.PROJECT_ROOT).as_posix()}")
    print(f"\nFinished in {time.monotonic() - started:.0f}s at {datetime.now():%Y-%m-%d %H:%M:%S}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
