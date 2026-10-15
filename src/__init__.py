"""Cryptocurrency Market Volatility & Trend Analysis System.

Package layout
--------------
config             Paths, analysis parameters and chart palette.
data_fetcher       CoinGecko API client (retries, rate limits, validation, caching).
preprocessing      Cleaning, de-duplication, missing values, returns.
moving_average     MA(3)/MA(5) and the analytical trend classification.
statistics         Per-coin volatility statistics, rankings, risk/return quadrants.
outlier_detection  Z-score outliers and a supplementary Isolation Forest check.
visualization      Matplotlib static charts and Plotly interactive charts.
reporting          Summary tables, CSV exports and the Markdown report.
main               End-to-end pipeline orchestration (``python main.py``).

Note: ``src/statistics.py`` shares its name with the standard-library module.
Always import it through the package (``from src import statistics``) and run the
pipeline from the project root (``python main.py`` or ``python -m src.main``).
"""

__version__ = "1.0.0"
