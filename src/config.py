"""Central configuration: file paths, analysis parameters, API settings and palette.

Everything that a user might reasonably want to tune lives here so that the
analysis modules stay free of magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = RAW_DIR / "cache"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Optional persistent storage: accumulates history across runs (see src/database.py)
DATABASE_PATH = DATA_DIR / "crypto_market.db"

# Raw data (exactly as retrieved, before cleaning)
RAW_MARKETS_CSV = RAW_DIR / "top_coins_markets.csv"
RAW_HISTORY_CSV = RAW_DIR / "crypto_data.csv"
FETCH_LOG_CSV = RAW_DIR / "fetch_log.csv"

# Processed data
CLEAN_CSV = PROCESSED_DIR / "crypto_data_cleaned.csv"
ANALYSIS_DATASET_CSV = PROCESSED_DIR / "crypto_analysis_dataset.csv"
DAILY_RETURNS_CSV = PROCESSED_DIR / "daily_returns.csv"
PREPROCESSING_REPORT_CSV = PROCESSED_DIR / "preprocessing_report.csv"

# Reports
SUMMARY_REPORT_CSV = REPORTS_DIR / "crypto_analysis_report.csv"
COIN_STATISTICS_CSV = REPORTS_DIR / "coin_statistics.csv"
VOLATILITY_RANKING_CSV = REPORTS_DIR / "volatility_ranking.csv"
TOP_VOLATILE_CSV = REPORTS_DIR / "top10_most_volatile.csv"
LEAST_VOLATILE_CSV = REPORTS_DIR / "top10_least_volatile.csv"
OUTLIER_REPORT_CSV = REPORTS_DIR / "outlier_report.csv"
ANOMALOUS_COINS_CSV = REPORTS_DIR / "anomalous_coins_isolation_forest.csv"
MARKDOWN_REPORT = REPORTS_DIR / "crypto_analysis_report.md"
INTERACTIVE_REPORT_HTML = REPORTS_DIR / "interactive_charts.html"

# ---------------------------------------------------------------------------
# API settings
# ---------------------------------------------------------------------------
# Optional: set COINGECKO_API_KEY (and COINGECKO_API_PLAN=demo|pro) in the
# environment for higher rate limits. The public endpoint works without a key.
API_KEY_ENV = "COINGECKO_API_KEY"
API_PLAN_ENV = "COINGECKO_API_PLAN"
PUBLIC_BASE_URL = "https://api.coingecko.com/api/v3"
PRO_BASE_URL = "https://pro-api.coingecko.com/api/v3"

# Minimum seconds between requests. The public tier allows only a small number
# of calls per minute; the client also slows down adaptively after HTTP 429.
MIN_INTERVAL_PUBLIC = 2.5
MIN_INTERVAL_DEMO = 2.1
MIN_INTERVAL_PRO = 0.25
MAX_RETRIES = 6
REQUEST_TIMEOUT_SECONDS = 30

# Cryptocurrencies trade 24/7/365, so annualisation uses calendar time.
MINUTES_PER_YEAR = 365 * 24 * 60


@dataclass
class AnalysisConfig:
    """All run-time parameters of the pipeline."""

    top_n: int = 100                      # number of coins by market cap
    days: int = 5                         # historical window in days
    vs_currency: str = "usd"
    ma_windows: tuple[int, int] = (3, 5)  # short and long moving-average periods
    z_threshold: float = 2.0              # |Z| > 2  -> outlier
    strict_z_threshold: float = 3.0       # |Z| > 3  -> extreme outlier
    min_observations: int = 24            # minimum price points per coin
    max_interpolation_gap: int = 3        # max consecutive missing prices to fill
    top_k: int = 10                       # size of the "top volatile" tables
    cache_minutes: float = 0.0            # 0 = always fetch fresh data
    history_cache_minutes: float | None = None  # history cache; None = same as cache_minutes
    selected_coins: list[str] = field(default_factory=list)  # ids / symbols / names
    n_default_coins: int = 3              # coins charted when none are selected
    sort_by: str = "rank"                 # rank | volatility | return | outliers
    make_plots: bool = True
    make_interactive: bool = True
    use_database: bool = True             # append each run to data/crypto_market.db
    show_plots: bool = False
    random_state: int = 42                # reproducible Isolation Forest


# ---------------------------------------------------------------------------
# Chart palette (validated reference palette; light surface)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES_1 = "#2a78d6"  # blue   - primary series (price)
SERIES_2 = "#eb6834"  # orange - MA(3)
SERIES_3 = "#1baf7a"  # aqua   - MA(5)
DE_EMPHASIS = "#c3c2b7"  # gray for "the rest" in emphasis charts

# Polarity (diverging poles) for positive / negative outliers. Always paired
# with a marker shape (up / down triangle) so colour is never the only cue.
POSITIVE = "#2a78d6"  # blue
NEGATIVE = "#e34948"  # red

FONT_FAMILY = ["Segoe UI", "system-ui", "-apple-system", "Helvetica Neue", "Arial", "DejaVu Sans"]
