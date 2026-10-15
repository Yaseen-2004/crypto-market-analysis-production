"""Tests for the optional SQLite persistence layer."""

from __future__ import annotations

import pandas as pd
import pytest

from src import database


@pytest.fixture
def pipeline_frames(analysis_frames):
    """The shared offline analysis output (cfg, analysis, summary)."""
    return analysis_frames


def test_connect_creates_every_table(tmp_path):
    path = tmp_path / "nested" / "market.db"
    with database.session(path) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert path.exists()                                    # parent directory created
    assert {"runs", "coins", "market_snapshots", "prices", "coin_statistics", "outliers"} <= tables


def test_save_run_populates_tables_and_returns_run_id(tmp_path, pipeline_frames):
    cfg, analysis, summary = pipeline_frames
    with database.session(tmp_path / "market.db") as conn:
        run_id = database.save_run(conn, cfg=cfg, analysis=analysis, summary=summary)
        assert run_id == 1
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("runs", "coins", "prices", "coin_statistics", "outliers")}
        stored = database.totals(conn)
    assert counts["runs"] == 1
    assert counts["coins"] == summary["coin_id"].nunique()
    assert counts["prices"] == len(analysis)
    assert counts["coin_statistics"] == len(summary)
    assert counts["outliers"] == int(analysis["is_outlier"].sum()) > 0
    assert stored["observations"] == len(analysis)


def test_reruns_upsert_instead_of_duplicating(tmp_path, pipeline_frames):
    """Overlapping windows must update rows in place - this is what lets history accumulate."""
    cfg, analysis, summary = pipeline_frames
    with database.session(tmp_path / "market.db") as conn:
        first = database.save_run(conn, cfg=cfg, analysis=analysis, summary=summary)
        second = database.save_run(conn, cfg=cfg, analysis=analysis, summary=summary)
        rows = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        coins = conn.execute("SELECT COUNT(*) FROM coins").fetchone()[0]
        latest_run = conn.execute("SELECT DISTINCT run_id FROM prices").fetchall()
    assert second == first + 1
    assert runs == 2                                        # both runs recorded
    assert rows == len(analysis)                            # but observations not duplicated
    assert coins == summary["coin_id"].nunique()
    assert [r[0] for r in latest_run] == [second]           # rows point at the newest run


def test_new_observations_extend_the_history(tmp_path, pipeline_frames):
    cfg, analysis, summary = pipeline_frames
    later = analysis.copy()
    later["timestamp"] = later["timestamp"] + pd.Timedelta(days=10)   # a later, non-overlapping window
    with database.session(tmp_path / "market.db") as conn:
        database.save_run(conn, cfg=cfg, analysis=analysis, summary=summary)
        database.save_run(conn, cfg=cfg, analysis=later, summary=summary)
        stored = database.totals(conn)
        cover = database.coverage(conn)
    assert stored["observations"] == 2 * len(analysis)      # window grew beyond a single API fetch
    assert stored["days_covered"] > cfg.days
    assert set(cover["coin_id"]) == set(summary["coin_id"])


def test_load_prices_filters_by_coin_and_date(tmp_path, pipeline_frames):
    cfg, analysis, summary = pipeline_frames
    with database.session(tmp_path / "market.db") as conn:
        database.save_run(conn, cfg=cfg, analysis=analysis, summary=summary)
        one = database.load_prices(conn, coin_id="bitcoin")
        cutoff = analysis["timestamp"].median().strftime(database.TIME_FORMAT)
        recent = database.load_prices(conn, since=cutoff)
        history = database.run_history(conn)
    assert set(one["coin_id"]) == {"bitcoin"}
    assert one["timestamp_utc"].is_monotonic_increasing
    assert len(recent) < len(analysis) and (recent["timestamp_utc"] >= pd.Timestamp(cutoff, tz="UTC")).all()
    assert len(history) == 1 and history["n_coins"].iloc[0] == summary["coin_id"].nunique()


def test_failed_save_leaves_the_database_unchanged(tmp_path, pipeline_frames):
    """A broken frame must roll the whole run back, not half-write it."""
    cfg, analysis, summary = pipeline_frames
    broken = summary.drop(columns=["std_return_pct"])
    with database.session(tmp_path / "market.db") as conn:
        database.save_run(conn, cfg=cfg, analysis=analysis, summary=summary)
        before = database.totals(conn)
        with pytest.raises(KeyError, match="missing column"):
            database.save_run(conn, cfg=cfg, analysis=analysis.drop(columns=["price"]), summary=broken)
        after = database.totals(conn)
    assert after["observations"] == before["observations"]
    assert after["runs"] == before["runs"]
