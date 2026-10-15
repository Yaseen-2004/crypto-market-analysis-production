"""Optional SQLite persistence.

Every run is appended to a local database, which turns a sequence of runs into
something the API alone cannot provide: CoinGecko's free `market_chart` window
only reaches back 5 days, but `prices` accumulates every observation ever
downloaded, so a project that runs daily builds a long history over time.

Tables
------
runs              one row per pipeline execution (parameters + headline counts)
coins             dimension table of every coin ever seen
market_snapshots  per-run market-cap snapshot (rank, price, market cap, volume)
prices            per-coin observations, keyed (coin_id, timestamp) - upserted
coin_statistics   per-run statistics for each coin
outliers          per-coin flagged observations, keyed (coin_id, timestamp)

`prices` and `outliers` use ON CONFLICT upserts, so overlapping windows from
consecutive runs update rows in place instead of duplicating them.

Uses only the standard library's ``sqlite3`` - no server, no extra dependency.
The same schema ports to PostgreSQL with minor type changes.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at_utc          TEXT NOT NULL,
    window_start_utc        TEXT,
    window_end_utc          TEXT,
    top_n                   INTEGER,
    days                    INTEGER,
    vs_currency             TEXT,
    ma_short_window         INTEGER,
    ma_long_window          INTEGER,
    z_threshold             REAL,
    n_coins                 INTEGER,
    n_observations          INTEGER,
    n_outliers              INTEGER,
    median_interval_minutes REAL
);

CREATE TABLE IF NOT EXISTS coins (
    coin_id        TEXT PRIMARY KEY,
    symbol         TEXT,
    name           TEXT,
    first_seen_utc TEXT,
    last_seen_utc  TEXT
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    run_id               INTEGER NOT NULL,
    coin_id              TEXT    NOT NULL,
    market_cap_rank      INTEGER,
    price                REAL,
    market_cap           REAL,
    volume_24h           REAL,
    price_change_24h_pct REAL,
    price_change_7d_pct  REAL,
    PRIMARY KEY (run_id, coin_id),
    FOREIGN KEY (run_id)  REFERENCES runs(run_id)  ON DELETE CASCADE,
    FOREIGN KEY (coin_id) REFERENCES coins(coin_id)
);

CREATE TABLE IF NOT EXISTS prices (
    coin_id       TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    price         REAL,
    market_cap    REAL,
    total_volume  REAL,
    pct_return    REAL,
    ma_short      REAL,
    ma_long       REAL,
    z_score       REAL,
    is_outlier    INTEGER,
    outlier_type  TEXT,
    run_id        INTEGER,
    PRIMARY KEY (coin_id, timestamp_utc),
    FOREIGN KEY (coin_id) REFERENCES coins(coin_id)
);

CREATE TABLE IF NOT EXISTS coin_statistics (
    run_id                    INTEGER NOT NULL,
    coin_id                   TEXT    NOT NULL,
    market_cap_rank           INTEGER,
    price                     REAL,
    market_cap                REAL,
    five_day_return_pct       REAL,
    mean_return_pct           REAL,
    std_return_pct            REAL,
    annualized_volatility_pct REAL,
    max_drawdown_pct          REAL,
    volatility_rank           INTEGER,
    trend                     TEXT,
    outlier_count             INTEGER,
    extreme_outlier_count     INTEGER,
    high_volatility           INTEGER,
    low_variation             INTEGER,
    risk_return_quadrant      TEXT,
    PRIMARY KEY (run_id, coin_id),
    FOREIGN KEY (run_id)  REFERENCES runs(run_id)  ON DELETE CASCADE,
    FOREIGN KEY (coin_id) REFERENCES coins(coin_id)
);

CREATE TABLE IF NOT EXISTS outliers (
    coin_id       TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    price         REAL,
    pct_return    REAL,
    z_score       REAL,
    outlier_type  TEXT,
    is_extreme    INTEGER,
    low_variation INTEGER,
    run_id        INTEGER,
    PRIMARY KEY (coin_id, timestamp_utc),
    FOREIGN KEY (coin_id) REFERENCES coins(coin_id)
);

CREATE INDEX IF NOT EXISTS idx_prices_timestamp    ON prices(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_outliers_timestamp  ON outliers(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_stats_run           ON coin_statistics(run_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_run       ON market_snapshots(run_id);
"""

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the database and make sure the schema exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


@contextmanager
def session(path: str | Path):
    """Context manager yielding a connection that is always closed."""
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()


def _rows(df: pd.DataFrame, columns: list[str]) -> list[tuple]:
    """Convert a frame to plain Python tuples (sqlite3 cannot bind NumPy scalars).

    Missing columns raise rather than silently storing NULLs: a column that
    quietly disappears would corrupt the stored analysis without any signal.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Cannot store rows: missing column(s) {missing}")
    frame = df[columns].astype(object)
    frame = frame.where(pd.notna(frame), None)
    return [tuple(row) for row in frame.to_numpy()]


def _times(series: pd.Series) -> pd.Series:
    return series.dt.strftime(TIME_FORMAT)


def _upsert(conn: sqlite3.Connection, table: str, columns: list[str], key: list[str],
            rows: list[tuple]) -> None:
    """INSERT ... ON CONFLICT(key) DO UPDATE - so re-running never duplicates."""
    if not rows:
        return
    placeholders = ", ".join("?" * len(columns))
    updates = ", ".join(f"{c} = excluded.{c}" for c in columns if c not in key)
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT({', '.join(key)}) DO UPDATE SET {updates}",
        rows,
    )


def save_run(conn: sqlite3.Connection, *, cfg, analysis: pd.DataFrame, summary: pd.DataFrame,
             markets: pd.DataFrame | None = None) -> int:
    """Persist one pipeline run. Returns the new ``run_id``.

    Everything is written in a single transaction: a failure leaves the
    database exactly as it was.
    """
    short, long = cfg.ma_windows
    now = datetime.now(timezone.utc).strftime(TIME_FORMAT)
    data = analysis.copy()
    data["timestamp_utc"] = _times(data["timestamp"])

    with conn:  # transaction: commit on success, roll back on exception
        cursor = conn.execute(
            "INSERT INTO runs (started_at_utc, window_start_utc, window_end_utc, top_n, days, vs_currency,"
            " ma_short_window, ma_long_window, z_threshold, n_coins, n_observations, n_outliers,"
            " median_interval_minutes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (now,
             data["timestamp"].min().strftime(TIME_FORMAT),
             data["timestamp"].max().strftime(TIME_FORMAT),
             int(cfg.top_n), int(cfg.days), cfg.vs_currency, int(short), int(long), float(cfg.z_threshold),
             int(summary["coin_id"].nunique()), int(len(data)), int(data["is_outlier"].sum()),
             float(data["interval_minutes"].median())),
        )
        run_id = int(cursor.lastrowid)

        # Dimension table: insert new coins, refresh last_seen for known ones.
        coins = summary[["coin_id", "symbol", "name"]].drop_duplicates("coin_id").copy()
        coins["first_seen_utc"] = now
        coins["last_seen_utc"] = now
        conn.executemany(
            "INSERT INTO coins (coin_id, symbol, name, first_seen_utc, last_seen_utc) VALUES (?,?,?,?,?) "
            "ON CONFLICT(coin_id) DO UPDATE SET symbol = excluded.symbol, name = excluded.name, "
            "last_seen_utc = excluded.last_seen_utc",
            _rows(coins, ["coin_id", "symbol", "name", "first_seen_utc", "last_seen_utc"]),
        )

        if markets is not None and not markets.empty:
            snapshot = markets.copy()
            snapshot["run_id"] = run_id
            columns = ["run_id", "coin_id", "market_cap_rank", "current_price", "market_cap",
                       "total_volume_24h", "price_change_24h_pct", "price_change_7d_pct"]
            conn.executemany(
                "INSERT OR REPLACE INTO market_snapshots (run_id, coin_id, market_cap_rank, price, market_cap,"
                " volume_24h, price_change_24h_pct, price_change_7d_pct) VALUES (?,?,?,?,?,?,?,?)",
                _rows(snapshot.loc[snapshot["coin_id"].isin(set(coins["coin_id"]))], columns),
            )

        data["run_id"] = run_id
        data["is_outlier_int"] = data["is_outlier"].astype(int)
        price_columns = ["coin_id", "timestamp_utc", "price", "market_cap", "total_volume", "pct_return",
                         f"ma_{short}", f"ma_{long}", "z_score", "is_outlier_int", "outlier_type", "run_id"]
        _upsert(conn, "prices",
                ["coin_id", "timestamp_utc", "price", "market_cap", "total_volume", "pct_return",
                 "ma_short", "ma_long", "z_score", "is_outlier", "outlier_type", "run_id"],
                ["coin_id", "timestamp_utc"], _rows(data, price_columns))

        stats = summary.copy()
        stats["run_id"] = run_id
        for column in ("high_volatility", "low_variation"):
            if column in stats:
                stats[column] = stats[column].astype(bool).astype(int)
            else:
                stats[column] = 0
        stat_columns = ["run_id", "coin_id", "market_cap_rank", "price", "market_cap", "five_day_return_pct",
                        "mean_return_pct", "std_return_pct", "annualized_volatility_pct", "max_drawdown_pct",
                        "volatility_rank", "trend", "outlier_count", "extreme_outlier_count",
                        "high_volatility", "low_variation", "risk_return_quadrant"]
        _upsert(conn, "coin_statistics", stat_columns, ["run_id", "coin_id"], _rows(stats, stat_columns))

        flagged = data.loc[data["is_outlier"]].copy()
        flagged["run_id"] = run_id
        flagged["is_extreme"] = flagged["is_extreme_outlier"].astype(int)
        flagged["low_variation_int"] = (flagged["low_variation"].astype(bool).astype(int)
                                        if "low_variation" in flagged else 0)
        outlier_columns = ["coin_id", "timestamp_utc", "price", "pct_return", "z_score", "outlier_type",
                           "is_extreme", "low_variation_int", "run_id"]
        _upsert(conn, "outliers",
                ["coin_id", "timestamp_utc", "price", "pct_return", "z_score", "outlier_type",
                 "is_extreme", "low_variation", "run_id"],
                ["coin_id", "timestamp_utc"], _rows(flagged, outlier_columns))

    logger.info("Stored run #%d in the database", run_id)
    return run_id


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
def run_history(conn: sqlite3.Connection) -> pd.DataFrame:
    """All runs, newest first."""
    return pd.read_sql_query("SELECT * FROM runs ORDER BY run_id DESC", conn)


def load_prices(conn: sqlite3.Connection, coin_id: str | None = None, since: str | None = None) -> pd.DataFrame:
    """Accumulated observations, optionally for one coin and/or from a date onwards."""
    query = "SELECT * FROM prices"
    clauses, params = [], []
    if coin_id:
        clauses.append("coin_id = ?")
        params.append(coin_id)
    if since:
        clauses.append("timestamp_utc >= ?")
        params.append(since)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    frame = pd.read_sql_query(query + " ORDER BY coin_id, timestamp_utc", conn, params=params)
    if not frame.empty:
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame


def coverage(conn: sqlite3.Connection) -> pd.DataFrame:
    """How much history has accumulated per coin across every run so far."""
    return pd.read_sql_query(
        "SELECT p.coin_id, c.symbol, c.name, COUNT(*) AS observations,"
        " MIN(p.timestamp_utc) AS first_observation, MAX(p.timestamp_utc) AS last_observation,"
        " ROUND(julianday(MAX(p.timestamp_utc)) - julianday(MIN(p.timestamp_utc)), 2) AS days_covered"
        " FROM prices p LEFT JOIN coins c ON c.coin_id = p.coin_id"
        " GROUP BY p.coin_id, c.symbol, c.name ORDER BY observations DESC", conn)


def totals(conn: sqlite3.Connection) -> dict:
    """Headline counts for logging and the dashboard."""
    row = conn.execute(
        "SELECT (SELECT COUNT(*) FROM runs), (SELECT COUNT(*) FROM coins), (SELECT COUNT(*) FROM prices),"
        " (SELECT COUNT(*) FROM outliers), (SELECT MIN(timestamp_utc) FROM prices),"
        " (SELECT MAX(timestamp_utc) FROM prices)").fetchone()
    keys = ("runs", "coins", "observations", "outliers", "first_observation", "last_observation")
    stats = dict(zip(keys, row))
    first, last = stats["first_observation"], stats["last_observation"]
    stats["days_covered"] = (
        (datetime.strptime(last, TIME_FORMAT) - datetime.strptime(first, TIME_FORMAT)).total_seconds() / 86400
        if first and last else 0.0
    )
    return stats
