# Cryptocurrency Market Volatility & Trend Analysis System

A Python data-analysis pipeline that retrieves the **top 100 cryptocurrencies by market capitalisation**
from the CoinGecko API, downloads the **latest 5 days** of market history, and analyses price trends,
moving averages, volatility and statistical outliers — producing CSV tables, charts, a written report and
an optional interactive dashboard.

> **This is a data-science / statistical analysis project, not financial advice.** Trend labels, volatility
> rankings and outlier flags are descriptive statistics of historical data. They are not trading
> recommendations. Cryptocurrency prices are highly volatile and past behaviour does not predict future results.

---

## Quick start

Requires **Python 3.10 or newer**.

```bash
pip install -r requirements.txt
python main.py
```

Everything is fetched live on each run — no prices or coin lists are hard-coded. The full 100-coin download
is rate-limited by the public API and takes roughly **15–25 minutes** without an API key (a measured run
took 23 minutes: 105 requests, 4 rate-limit responses handled automatically); progress is logged per coin.
Analysis, charts and reports then take under a minute — with `--cache-minutes`, re-running on already
downloaded data takes about 30 seconds.

```bash
python main.py --top-n 20                       # smaller, faster run
python main.py --coins bitcoin ETH Solana       # choose the coins charted individually
python main.py --sort-by volatility             # sort the summary CSV (rank|volatility|return|outliers)
python main.py --days 5 --z-threshold 2.5       # tune the window / outlier threshold
python main.py --cache-minutes 60               # reuse recent API responses while experimenting
python main.py --no-db --no-plots               # skip SQLite storage / charts
python main.py --help                           # all options
```

Optional interfaces:

```bash
streamlit run dashboard/app.py                  # interactive dashboard
jupyter notebook notebooks/crypto_analysis.ipynb
pytest                                          # 88 tests, no network needed (~1 minute)
```

### API key (optional)

The public CoinGecko endpoint needs no key. A free demo key raises the rate limit considerably:

```bash
# Windows PowerShell
$env:COINGECKO_API_KEY="CG-xxxx"; $env:COINGECKO_API_PLAN="demo"   # or "pro"
# macOS / Linux
export COINGECKO_API_KEY=CG-xxxx COINGECKO_API_PLAN=demo
```

### If HTTPS requests fail with `CERTIFICATE_VERIFY_FAILED`

Corporate proxies and some antivirus products intercept TLS with their own certificate authority, which is
not in Python's bundled `certifi` roots. The client automatically uses the operating-system certificate
store when [`truststore`](https://pypi.org/project/truststore/) is installed (it is in `requirements.txt`).
Certificate verification is never disabled. Alternatively point `REQUESTS_CA_BUNDLE` at your CA bundle.

---

## What the pipeline does

`python main.py` runs ten steps end to end:

1. Fetch the current top 100 coins by market cap (`/coins/markets`).
2. Download 5 days of price, market-cap and volume history per coin (`/coins/{id}/market_chart`).
3. Clean the data (types, timestamps, duplicates, missing values, window).
4. Calculate periodic and daily returns.
5. Calculate MA(3) and MA(5) and classify trends.
6. Calculate standard deviation and volatility rankings.
7. Detect Z-score outliers (|Z| > 2, extreme |Z| > 3).
8. Generate six chart families (PNG) plus an interactive HTML report.
9. Write the summary CSVs and the Markdown report.
10. Print the findings to the console.

### Data collection and robustness

The API layer (`src/data_fetcher.py`) handles the messy realities of a public API:

| Situation | Handling |
|---|---|
| Rate limits (HTTP 429) | Honours `Retry-After`, exponential back-off with jitter, and **adaptively slows** the request interval for the rest of the run |
| Network errors / timeouts | Retried with exponential back-off (up to 6 retries) |
| Server errors (5xx) | Retried |
| TLS certificate failures | Fail fast with a fix hint (retrying cannot help) |
| Invalid / malformed responses | Schema-validated; bad records dropped and counted |
| Error payloads returned with HTTP 200 | Detected and raised |
| Missing values | Short interior gaps time-interpolated (and flagged), the rest dropped |
| Duplicate records | Removed per coin and timestamp |
| Coins that fail entirely | Logged in `data/raw/fetch_log.csv`, retried once after a cool-down, then skipped |
| Coins with too little history | Excluded (`--min-observations`, default 24) and reported |

---

## Project structure

```text
crypto_market_analysis/
├── data/
│   ├── raw/                         # exactly as retrieved
│   │   ├── top_coins_markets.csv    # top-100 snapshot
│   │   ├── crypto_data.csv          # 5-day history, long format
│   │   ├── fetch_log.csv            # per-coin fetch status
│   │   └── cache/                   # optional JSON cache (--cache-minutes)
│   └── processed/
│       ├── crypto_data_cleaned.csv       # cleaned history + returns
│       ├── crypto_analysis_dataset.csv   # + MA, trend, Z-scores, outlier flags
│       ├── daily_returns.csv             # daily closes and daily returns
│       └── preprocessing_report.csv      # data-quality audit trail
├── notebooks/crypto_analysis.ipynb  # exploratory walkthrough
├── src/
│   ├── config.py                    # paths, parameters, palette
│   ├── data_fetcher.py              # CoinGecko client
│   ├── preprocessing.py             # cleaning + returns
│   ├── statistics.py                # volatility metrics, rankings, quadrants
│   ├── moving_average.py            # MA(3)/MA(5) + trend classification
│   ├── outlier_detection.py         # Z-scores + Isolation Forest
│   ├── visualization.py             # Matplotlib + Plotly charts
│   ├── reporting.py                 # tables, CSVs, Markdown report
│   ├── formatting.py                # number formatting
│   ├── database.py                  # optional SQLite persistence
│   └── main.py                      # pipeline orchestration
├── reports/
│   ├── crypto_analysis_report.csv   # statistical summary (one row per coin)
│   ├── crypto_analysis_report.md    # written analytical report
│   ├── coin_statistics.csv          # every computed metric per coin
│   ├── volatility_ranking.csv       # ranked by standard deviation
│   ├── top10_most_volatile.csv
│   ├── top10_least_volatile.csv
│   ├── outlier_report.csv           # every |Z| > 2 observation
│   ├── anomalous_coins_isolation_forest.csv
│   ├── interactive_charts.html      # Plotly charts with a coin selector
│   └── figures/*.png                # graphs 1–6
├── data/crypto_market.db            # SQLite store, appended to on every run
├── dashboard/app.py                 # Streamlit dashboard
├── tests/                           # 88 tests, all offline
│   ├── helpers.py                   # synthetic data + a fake API client
│   ├── test_calculations.py         # formulas, preprocessing, API client behaviour
│   ├── test_visualization.py        # chart structure (Matplotlib + Plotly)
│   ├── test_reporting.py            # summary tables, findings, Markdown report
│   ├── test_database.py             # SQLite storage
│   └── test_pipeline.py             # end-to-end runs against the fake client
├── requirements.txt
└── main.py
```

---

## Methodology

### Resolution: hourly, not daily

For a 2–90 day window CoinGecko returns **hourly** observations, so a 5-day window gives ~121 points per
coin instead of 5. The pipeline preserves that original resolution: returns, moving averages, standard
deviations and Z-scores are computed per hourly period, which makes the statistics far more reliable than
four daily returns would be. A daily view (last price of each UTC day and its daily return) is also written
to `data/processed/daily_returns.csv`.

### Formulas

| Quantity | Formula |
|---|---|
| Periodic return | `r_t = (P_t − P_{t−1}) / P_{t−1} × 100` |
| 5-day return | `(P_last − P_first) / P_first × 100` |
| Moving average | `MA(k)_t = mean(P_t, P_{t−1}, …, P_{t−k+1})` |
| Standard deviation | `σ = sqrt( Σ(x − μ)² / N )` (population, N in the denominator) |
| Z-score | `Z = (x − μ) / σ`, per coin |
| Annualised volatility | `σ × sqrt(365 × 24)` for hourly data (24/7 calendar) |
| Unusual volatility | `σ > Q3 + 1.5 × IQR` across all coins (Tukey fence) |

### Why volatility is measured on returns, not prices

A standard deviation of raw prices is denominated in dollars and scales with the price level. Bitcoin near
\$76,000 can have a price standard deviation in the hundreds of dollars while a \$0.50 coin has one of
fractions of a cent — even if the \$0.50 coin swings far more violently in percentage terms. Ranking those
two by price standard deviation measures *price level*, not *risk*. Percentage returns are scale-free, so
they are directly comparable across the top 100, and they are the standard input to volatility statistics
in finance. This is why every volatility figure, Z-score and risk/return coordinate in this project is
computed from returns.

### Cryptocurrencies trade 24/7

There is no market close, no weekend gap and no overnight jump to special-case: every hour is a valid
observation, the time axis of each chart is continuous calendar time, and annualisation uses 365 × 24
hourly periods rather than the 252 trading days used for equities.

### Distinguishing the measures

| Term | What it is |
|---|---|
| **Price** | The level, in USD, at one timestamp. |
| **Return** | The percentage change between consecutive observations — scale-free. |
| **Volatility** | The dispersion of those returns (standard deviation) — how much a coin typically moves. |
| **Moving average** | A smoothed price over the last k observations — the trend, with noise reduced. |
| **Outlier** | A single return far from that coin's own mean relative to its own σ (|Z| > 2). |

A coin can be highly volatile with no outliers (consistently large moves) or calm with several (a quiet
series punctuated by shocks).

### Trend classification

`Bullish` when Price > MA(3) > MA(5), `Bearish` when Price < MA(3) < MA(5), otherwise `Neutral`. These are
**analytical labels describing the ordering of recent averages**, not signals to act on; on hourly data they
describe momentum over a few hours only.

### Outliers

Z-scores are computed per coin, so "unusual" always means unusual *for that coin*. Because crypto returns
are fat-tailed, considerably more than the ~4.6% of observations implied by a normal distribution typically
exceed |Z| > 2 — the report states the observed rate against the normal baseline on every run. Two
supplementary checks are included: the Tukey fence for coins whose overall volatility is unusual, and a
scikit-learn `IsolationForest` over several per-coin features for multivariate oddities.

### Statistically extreme ≠ economically meaningful

Roughly a quarter of the top 100 are **near-constant assets** — stablecoins, but also tokenized
money-market and treasury funds (BUIDL, JTRSY, USYC, USTB, JAAA) that trade at \$1.12, \$11.21 or \$106.54
rather than at \$1. Their σ is around 0.003%, so a 0.03% blip is a 10σ event. In a measured run they
produced 110 of the 614 flagged outliers and occupied the entire top of the |Z| ranking, burying genuine
moves such as −9.4% in one hour.

Two flags separate these cases, and both are in the outputs:

* `peg_like` — price within 3% of \$1 (identifies stablecoins specifically).
* `low_variation` — total price range across the window ≤ 0.5%, **independent of price level**, so it also
  catches the tokenized funds.

The outlier report keeps every flagged observation, marking each with `Low-Variation Asset`. The report,
the console summary and `outlier_detection.material_outliers()` rank the **material** outliers with those
assets excluded, and the report additionally shows the raw |Z| leaderboard so the distinction is explicit.
The dashboard can hide them from the rankings.

---

## Outputs

**Charts** (`reports/figures/`, and interactively in `reports/interactive_charts.html`):

1. Top 100 market capitalisation — ranked horizontal bars, log scale.
2. Price trend — price over the last 5 days, per selected coin.
3. Moving averages — price with MA(3) and MA(5), plus a recent-window zoom.
4. Volatility ranking — standard deviation for all 100 coins, top 10 highlighted.
5. Outlier detection — returns with the mean, the ±2σ band and flagged outliers.
6. Risk vs return — standard deviation against mean return, split into quadrants.

**Tables:** statistical summary (`crypto_analysis_report.csv`), volatility ranking, top 10 most/least
volatile, the full outlier report, and the written report (`crypto_analysis_report.md`).

The summary table is sorted by market-cap rank by default and can be sorted by volatility, return or
outlier count via `--sort-by` (or interactively in the dashboard).

---

## Persistent storage (SQLite)

Every run is also appended to `data/crypto_market.db` (stdlib `sqlite3`, no server required; disable with
`--no-db`). This is what lets the project outgrow its own data source: the free `market_chart` endpoint
only reaches back 5 days, but the `prices` table keeps every observation ever downloaded, so running the
pipeline on a schedule accumulates a history far longer than any single fetch.

| Table | Contents |
|---|---|
| `runs` | one row per execution — parameters, window, coin/observation/outlier counts |
| `coins` | every coin ever seen, with first/last seen timestamps |
| `market_snapshots` | per-run market-cap rank, price, market cap and volume |
| `prices` | every observation with returns, MAs, Z-score and outlier flags |
| `coin_statistics` | per-run volatility metrics, trend and outlier counts per coin |
| `outliers` | every flagged observation, including the near-constant marker |

`prices` and `outliers` are keyed by `(coin_id, timestamp)` and written with `ON CONFLICT … DO UPDATE`, so
re-running on an overlapping window refreshes rows instead of duplicating them. Each run is written in a
single transaction, and a storage failure only logs a warning — it never invalidates the analysis.

### Building history on a schedule

The store only accumulates if the pipeline runs repeatedly. A daily run is enough to extend the history
well past the API's 5-day window:

```bash
# Linux/macOS - crontab -e, 06:00 daily
0 6 * * * cd /path/to/crypto_market_analysis && python main.py --no-plots >> run.log 2>&1

# Windows - Task Scheduler, daily
schtasks /create /tn "Crypto analysis" /tr "python C:\path\to\crypto_market_analysis\main.py --no-plots" /sc daily /st 06:00
```

`--no-plots` keeps a scheduled run quick; the CSVs, report and database are still written.

```python
from src import config, database

with database.session(config.DATABASE_PATH) as conn:
    print(database.totals(conn))            # runs, coins, observations, days covered
    history = database.run_history(conn)    # every past run
    btc = database.load_prices(conn, "bitcoin", since="2026-09-12")
    coverage = database.coverage(conn)      # accumulated history per coin
```

Plain SQL works too — the biggest genuine moves, excluding near-constant assets:

```sql
SELECT c.name, o.timestamp_utc, ROUND(o.pct_return, 2) AS ret, ROUND(o.z_score, 2) AS z, o.outlier_type
FROM outliers o JOIN coins c USING (coin_id)
WHERE o.low_variation = 0
ORDER BY ABS(o.z_score) DESC LIMIT 10;
```

## Dashboard

```bash
streamlit run dashboard/app.py
```

* **Market overview** — coins analysed, largest market cap, most volatile, highest/lowest 5-day return, total outliers.
* **Cryptocurrency selector** — price, market cap, 5-day return, σ, MA(3), MA(5), trend and outlier count for any coin.
* **Charts** — price trend, moving averages, returns and outliers, volatility ranking, risk vs return, market cap.
* **Tables** — sortable summary, top 10 most/least volatile, outlier report, CSV downloads.
* A sidebar button re-runs the whole pipeline with fresh data.

---

## Tests

```bash
pytest          # 88 tests, about a minute, no network access
```

The suite runs entirely offline: a fake client serves synthetic payloads (including a near-constant
stablecoin, a very volatile coin and coins with deliberate spikes), and all output paths are redirected to
a temporary directory, so tests never touch `data/` or `reports/`.

| File | Covers |
|---|---|
| `test_calculations.py` | return/σ/Z-score formulas against hand-computed values, preprocessing, retries, rate limits, response validation |
| `test_visualization.py` | chart structure — bar counts, series present, sort order, marker shapes, palette, saved files |
| `test_reporting.py` | summary columns, findings matching the data, valid Markdown tables |
| `test_database.py` | schema, upsert-not-duplicate, history growth, transaction rollback |
| `test_pipeline.py` | full runs end to end: outputs, figures, coin selection, failure handling, storage |

Running `python main.py` against the live API remains the real integration test.

Linting uses [ruff](https://docs.astral.sh/ruff/) (`ruff check .`); its configuration, together with
pytest's, lives in `pyproject.toml`. GitHub Actions runs both on every push
([.github/workflows/tests.yml](.github/workflows/tests.yml)) across Python 3.10-3.13.

## Limitations

* Five days of hourly data is a short sample; all statistics are noisy and regime-dependent.
* MA(3)/MA(5) on hourly data describe hour-scale momentum, not multi-day trends.
* Z-scores use each coin's own mean and σ, which are themselves inflated by the outliers they are meant to
  detect (masking), and assume a roughly stable distribution within the window.
* CoinGecko prices are aggregated across exchanges and may differ from any single venue; the most recent
  observation of each series can cover a partial interval.
* The market-cap snapshot and the historical series are retrieved seconds to minutes apart, so a fast-moving
  coin's snapshot price and last historical price may differ slightly.
* Newly listed coins may have fewer than 5 days of history; those below `--min-observations` are excluded
  and listed in the preprocessing report.

## Technology

Python · pandas · NumPy · requests · Matplotlib · Plotly · scikit-learn · Streamlit · Jupyter · pytest

Data source: [CoinGecko API](https://www.coingecko.com/en/api) (public endpoints; please respect their
rate limits and terms of use).
