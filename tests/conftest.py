"""Shared pytest fixtures. Also makes the project root importable for ``from src import ...``."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import moving_average, outlier_detection, preprocessing  # noqa: E402
from src import statistics as coin_stats  # noqa: E402
from src.config import AnalysisConfig  # noqa: E402
from tests.helpers import synthetic_market_data  # noqa: E402


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Redirect every configured output path into ``tmp_path``.

    Tests must never write into the real data/ and reports/ directories.
    """
    from src import config

    real_root = config.PROJECT_ROOT
    for name in dir(config):
        if not name.isupper():
            continue
        value = getattr(config, name)
        if isinstance(value, Path):
            try:
                relative = value.relative_to(real_root)
            except ValueError:
                continue
            monkeypatch.setattr(config, name, tmp_path / relative)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def raw_history() -> pd.DataFrame:
    """Raw long-format history for a handful of synthetic coins (as the fetcher returns it)."""
    markets, charts = synthetic_market_data()
    frames = []
    for record in markets:
        coin_id = record["id"]
        chart = charts[coin_id]
        frame = pd.DataFrame(chart["prices"], columns=["timestamp_ms", "price"])
        frame["market_cap"] = [v for _, v in chart["market_caps"]]
        frame["total_volume"] = [v for _, v in chart["total_volumes"]]
        frame.insert(0, "coin_id", coin_id)
        frame["name"] = record["name"]
        frame["symbol"] = record["symbol"].upper()
        frame["market_cap_rank"] = record["market_cap_rank"]
        frame["snapshot_price"] = record["current_price"]
        frame["snapshot_market_cap"] = record["market_cap"]
        frame["snapshot_volume_24h"] = record["total_volume"]
        frame["price_change_24h_pct"] = record["price_change_percentage_24h_in_currency"]
        frame["price_change_7d_pct"] = record["price_change_percentage_7d_in_currency"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def analysis_frames(raw_history):
    """Run the real analysis steps offline: returns (cfg, analysis, summary)."""
    cfg = AnalysisConfig(top_n=6, days=5)
    short, long = cfg.ma_windows
    clean, _ = preprocessing.preprocess(raw_history, cfg.days, cfg.min_observations)
    analysis = moving_average.add_trend(moving_average.add_moving_averages(clean, cfg.ma_windows), short, long)

    stats = coin_stats.compute_coin_statistics(analysis)
    stats = coin_stats.rank_by_volatility(stats)
    stats = coin_stats.flag_unusual_volatility(stats)
    stats = coin_stats.flag_peg_like(stats)
    stats = coin_stats.flag_low_variation(stats)
    stats = coin_stats.assign_risk_return_quadrant(stats)

    analysis["low_variation"] = analysis["coin_id"].map(stats.set_index("coin_id")["low_variation"])
    analysis = outlier_detection.add_zscores(analysis)
    analysis = outlier_detection.flag_outliers(analysis, cfg.z_threshold, cfg.strict_z_threshold)

    trend = moving_average.latest_trend_snapshot(analysis, short, long)
    counts = outlier_detection.count_outliers(analysis)
    from src.reporting import assemble_coin_summary

    summary = assemble_coin_summary(stats, trend, counts)
    summary = outlier_detection.detect_anomalous_coins(summary, random_state=cfg.random_state)
    return cfg, analysis, summary
