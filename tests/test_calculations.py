"""Unit tests for the statistical calculations and the API client's error handling.

All expected values are computed by hand (or with an independent NumPy call) so
the tests verify the formulas rather than restating the implementation.

    pytest            # from the project root
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import requests

from src import data_fetcher, moving_average, outlier_detection, preprocessing, reporting
from src import statistics as coin_stats
from tests.helpers import START_MS, make_history


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------
def test_calculate_returns_matches_formula():
    prices = pd.Series([100.0, 110.0, 99.0])
    returns = preprocessing.calculate_returns(prices)
    assert math.isnan(returns.iloc[0])                     # no previous price
    assert returns.iloc[1] == pytest.approx(10.0)          # (110-100)/100*100
    assert returns.iloc[2] == pytest.approx(-10.0)         # (99-110)/110*100


def test_returns_are_computed_per_coin_not_across_coins():
    raw = pd.concat([make_history("a", (10.0, 11.0), rank=1), make_history("b", (50.0, 55.0), rank=2)])
    clean, _ = preprocessing.preprocess(raw, days=5, min_observations=2)
    first_of_b = clean.loc[clean["coin_id"] == "b"].iloc[0]
    assert pd.isna(first_of_b["pct_return"])               # not chained to coin "a"
    assert clean.loc[clean["coin_id"] == "b", "pct_return"].iloc[1] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Standard deviation
# ---------------------------------------------------------------------------
def test_population_std_uses_n_not_n_minus_1():
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]      # textbook example, sigma = 2
    assert coin_stats.population_std(values) == pytest.approx(2.0)
    assert coin_stats.population_std(values) == pytest.approx(float(np.std(values, ddof=0)))
    assert coin_stats.population_std(values) != pytest.approx(float(np.std(values, ddof=1)))


def test_population_std_ignores_nan_and_handles_empty():
    assert coin_stats.population_std([1.0, np.nan, 1.0]) == pytest.approx(0.0)
    assert math.isnan(coin_stats.population_std([]))
    assert math.isnan(coin_stats.population_std([np.nan]))


def test_max_drawdown_is_peak_to_trough():
    assert coin_stats.max_drawdown_pct([100.0, 120.0, 60.0, 90.0]) == pytest.approx(-50.0)
    assert coin_stats.max_drawdown_pct([100.0, 101.0, 102.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Moving averages and trend
# ---------------------------------------------------------------------------
def test_moving_average_values_and_min_periods():
    raw = make_history(prices=(10.0, 20.0, 30.0, 40.0, 50.0))
    clean, _ = preprocessing.preprocess(raw, days=5, min_observations=2)
    out = moving_average.add_moving_averages(clean, (3, 5))
    assert out["ma_3"].isna().tolist() == [True, True, False, False, False]
    assert out["ma_3"].iloc[2] == pytest.approx(20.0)      # (10+20+30)/3
    assert out["ma_3"].iloc[4] == pytest.approx(40.0)      # (30+40+50)/3
    assert out["ma_5"].iloc[4] == pytest.approx(30.0)      # (10+...+50)/5
    assert out["ma_5"].iloc[:4].isna().all()


def test_classify_trend_rules():
    assert moving_average.classify_trend(10, 9, 8) == "Bullish"
    assert moving_average.classify_trend(8, 9, 10) == "Bearish"
    assert moving_average.classify_trend(10, 8, 9) == "Neutral"
    assert moving_average.classify_trend(9, 9, 9) == "Neutral"      # equality is not strict
    assert moving_average.classify_trend(10, np.nan, 8) == "Insufficient data"


def test_add_trend_matches_scalar_classifier():
    raw = make_history(prices=tuple(float(p) for p in (10, 12, 14, 16, 18, 20)))
    clean, _ = preprocessing.preprocess(raw, days=5, min_observations=2)
    out = moving_average.add_trend(moving_average.add_moving_averages(clean, (3, 5)))
    expected = [moving_average.classify_trend(r["price"], r["ma_3"], r["ma_5"]) for _, r in out.iterrows()]
    assert out["trend"].tolist() == expected
    assert out["trend"].iloc[-1] == "Bullish"              # steadily rising prices


# ---------------------------------------------------------------------------
# Z-scores and outliers
# ---------------------------------------------------------------------------
def _z_frame(returns: list[float]) -> pd.DataFrame:
    n = len(returns)
    return pd.DataFrame(
        {
            "coin_id": ["c"] * n,
            "name": ["Coin"] * n,
            "symbol": ["C"] * n,
            "market_cap_rank": [1] * n,
            "datetime_utc": [f"2026-01-01 {h:02d}:00" for h in range(n)],
            "price": np.linspace(100, 110, n),
            "pct_return": returns,
        }
    )


def test_zscore_formula():
    returns = [1.0, 2.0, 3.0, 4.0, 90.0]
    out = outlier_detection.add_zscores(_z_frame(returns))
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=0))
    assert out["coin_mean_return_pct"].iloc[0] == pytest.approx(mu)
    assert out["coin_std_return_pct"].iloc[0] == pytest.approx(sigma)
    assert out["z_score"].iloc[-1] == pytest.approx((90.0 - mu) / sigma)


def test_zscore_is_nan_when_sigma_is_zero():
    out = outlier_detection.add_zscores(_z_frame([1.0, 1.0, 1.0, 1.0]))
    assert out["z_score"].isna().all()                     # constant returns -> undefined


def test_flag_outliers_thresholds_and_direction():
    # 20 small returns plus one large positive and one large negative move
    returns = [0.1, -0.1] * 10 + [12.0, -12.0]
    out = outlier_detection.flag_outliers(outlier_detection.add_zscores(_z_frame(returns)), 2.0, 3.0)
    flagged = out.loc[out["is_outlier"]]
    assert set(flagged["outlier_type"]) == {"Positive Outlier", "Negative Outlier"}
    assert (out.loc[out["pct_return"] == 12.0, "outlier_type"] == "Positive Outlier").all()
    assert (out.loc[out["pct_return"] == -12.0, "outlier_type"] == "Negative Outlier").all()
    # every flagged row really exceeds the threshold, and nothing else does
    assert (flagged["z_score"].abs() > 2.0).all()
    assert (out.loc[~out["is_outlier"], "z_score"].abs() <= 2.0).all()
    assert out["is_extreme_outlier"].sum() == int((out["z_score"].abs() > 3.0).sum())


def test_count_outliers_totals():
    returns = [0.1, -0.1] * 10 + [12.0, -12.0]
    out = outlier_detection.flag_outliers(outlier_detection.add_zscores(_z_frame(returns)))
    counts = outlier_detection.count_outliers(out).iloc[0]
    assert counts["outlier_count"] == int(out["is_outlier"].sum())
    assert counts["positive_outliers"] + counts["negative_outliers"] == counts["outlier_count"]


def test_outlier_report_columns_and_order():
    returns = [0.1, -0.1] * 10 + [12.0, -20.0]
    out = outlier_detection.flag_outliers(outlier_detection.add_zscores(_z_frame(returns)))
    out["ma_3"] = 1.0
    out["ma_5"] = 1.0
    report = outlier_detection.build_outlier_report(out)
    for column in ("Cryptocurrency", "Date/Time (UTC)", "Price (USD)", "Return (%)", "Z-score", "Outlier Type"):
        assert column in report.columns
    assert report["Z-score"].abs().is_monotonic_decreasing          # largest |Z| first
    assert report["Direction"].iloc[0] in {"Up", "Down"}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def test_preprocess_removes_duplicates_and_sorts():
    raw = make_history(prices=(10.0, 11.0, 12.0, 13.0))
    raw = pd.concat([raw.iloc[[3]], raw, raw.iloc[[1]]], ignore_index=True)   # shuffled + duplicated
    clean, report = preprocessing.preprocess(raw, days=5, min_observations=2)
    assert report.duplicates_removed == 2
    assert len(clean) == 4
    assert clean["timestamp"].is_monotonic_increasing


def test_preprocess_interpolates_interior_gap_and_flags_it():
    raw = make_history(prices=(10.0, np.nan, 30.0, 40.0))
    clean, report = preprocessing.preprocess(raw, days=5, min_observations=2)
    assert report.prices_interpolated == 1
    assert clean["price"].iloc[1] == pytest.approx(20.0)             # midpoint in time
    assert bool(clean["price_imputed"].iloc[1]) is True


def test_preprocess_drops_invalid_prices_and_short_series():
    good = make_history("good", (10.0, 11.0, 12.0), rank=1)
    short = make_history("short", (5.0, 6.0), rank=2)
    raw = pd.concat([good, short], ignore_index=True)
    raw.loc[raw["coin_id"] == "good", "price"] = [10.0, -5.0, 12.0]  # negative price is invalid
    clean, report = preprocessing.preprocess(raw, days=5, min_observations=3)
    assert report.invalid_prices_set_missing == 1
    assert "short" in report.coins_excluded
    assert set(clean["coin_id"]) == {"good"}


def test_preprocess_keeps_only_the_requested_window():
    old = make_history("c", (1.0, 2.0), start_ms=START_MS)
    new = make_history("c", (3.0, 4.0, 5.0), start_ms=START_MS + 10 * 86_400_000)
    clean, report = preprocessing.preprocess(pd.concat([old, new]), days=5, min_observations=2)
    assert report.rows_outside_window == 2
    assert len(clean) == 3


def test_daily_returns_use_the_last_price_of_each_utc_day():
    hourly = tuple(float(100 + i) for i in range(48))                # two full UTC days
    raw = make_history(prices=hourly, start_ms=1_700_006_400_000)    # 00:00 UTC start
    clean, _ = preprocessing.preprocess(raw, days=5, min_observations=2)
    daily = preprocessing.compute_daily_returns(clean)
    assert len(daily) == 2
    assert pd.isna(daily["daily_return_pct"].iloc[0])
    expected = (daily["close_price"].iloc[1] - daily["close_price"].iloc[0]) / daily["close_price"].iloc[0] * 100
    assert daily["daily_return_pct"].iloc[1] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Coin statistics
# ---------------------------------------------------------------------------
def _stats_for(prices):
    raw = make_history(prices=prices)
    clean, _ = preprocessing.preprocess(raw, days=5, min_observations=2)
    return coin_stats.compute_coin_statistics(clean).iloc[0]


def test_coin_statistics_core_metrics():
    prices = (100.0, 110.0, 99.0, 99.0, 108.9)
    row = _stats_for(prices)
    returns = [10.0, -10.0, 0.0, 10.0]
    assert row["n_observations"] == 5
    assert row["n_returns"] == 4
    assert row["five_day_return_pct"] == pytest.approx((108.9 - 100) / 100 * 100)
    assert row["mean_return_pct"] == pytest.approx(float(np.mean(returns)))
    assert row["std_return_pct"] == pytest.approx(float(np.std(returns, ddof=0)))
    assert row["max_abs_return_pct"] == pytest.approx(10.0)
    assert row["price"] == pytest.approx(108.9)


def test_annualized_volatility_scales_by_periods_per_year():
    row = _stats_for((100.0, 110.0, 99.0, 99.0, 108.9))     # hourly data -> 8760 periods/year
    assert row["annualized_volatility_pct"] == pytest.approx(row["std_return_pct"] * math.sqrt(365 * 24))


def test_volatility_ranking_and_fence():
    stats = pd.DataFrame({"std_return_pct": [0.1, 5.0, 0.2, 0.3], "market_cap_rank": [1, 2, 3, 4]})
    ranked = coin_stats.rank_by_volatility(stats)
    assert ranked.loc[ranked["std_return_pct"] == 5.0, "volatility_rank"].iloc[0] == 1
    flagged = coin_stats.flag_unusual_volatility(ranked)
    assert bool(flagged.loc[flagged["std_return_pct"] == 5.0, "high_volatility"].iloc[0]) is True
    assert not flagged.loc[flagged["std_return_pct"] == 0.1, "high_volatility"].iloc[0]


def test_risk_return_quadrants_cover_all_rows():
    stats = pd.DataFrame({"std_return_pct": [0.1, 0.2, 0.3, 0.4], "mean_return_pct": [-1.0, 2.0, -2.0, 1.0]})
    out = coin_stats.assign_risk_return_quadrant(stats)
    assert set(out["risk_return_quadrant"]) <= set(coin_stats.QUADRANT_LABELS.values())
    assert out["risk_return_quadrant"].notna().all()


def test_peg_like_flags_stablecoins_only():
    stats = pd.DataFrame({"price": [1.0005, 68000.0, 0.999], "price_range_pct": [0.05, 4.0, 0.2]})
    out = coin_stats.flag_peg_like(stats)
    assert out["peg_like"].tolist() == [True, False, True]


def test_low_variation_flag_is_independent_of_price_level():
    # A tokenized fund at $11 that barely moves must be flagged even though it is not near $1.
    stats = pd.DataFrame({"price": [1.0, 11.2, 106.5, 76000.0], "price_range_pct": [0.04, 0.05, 0.05, 4.0]})
    out = coin_stats.flag_low_variation(stats)
    assert out["low_variation"].tolist() == [True, True, True, False]


def test_material_outliers_exclude_near_constant_assets():
    frame = _z_frame([0.1, -0.1] * 10 + [12.0, -12.0])
    frame["low_variation"] = [False] * len(frame)
    stable = _z_frame([0.001, -0.001] * 10 + [0.05, -0.05])
    stable["coin_id"] = "stable"
    stable["symbol"] = "STBL"
    stable["name"] = "Stable"
    stable["low_variation"] = [True] * len(stable)
    combined = pd.concat([frame, stable], ignore_index=True)
    out = outlier_detection.flag_outliers(outlier_detection.add_zscores(combined))
    out["ma_3"] = out["ma_5"] = 1.0
    report = outlier_detection.build_outlier_report(out)
    material = outlier_detection.material_outliers(report)
    assert report["Low-Variation Asset"].any()                      # stablecoin rows are flagged
    assert not material["Low-Variation Asset"].any()                # and excluded from the material view
    assert set(material["Symbol"]) == {"C"}
    assert len(material) < len(report)


def test_sort_summary_orders_by_requested_metric():
    summary = pd.DataFrame({
        "market_cap_rank": [1, 2, 3],
        "std_return_pct": [0.5, 2.0, 1.0],
        "five_day_return_pct": [1.0, -3.0, 8.0],
        "outlier_count": [2, 9, 4],
    })
    assert reporting.sort_summary(summary, "volatility")["std_return_pct"].tolist() == [2.0, 1.0, 0.5]
    assert reporting.sort_summary(summary, "return")["five_day_return_pct"].tolist() == [8.0, 1.0, -3.0]
    assert reporting.sort_summary(summary, "outliers")["outlier_count"].tolist() == [9, 4, 2]
    assert reporting.sort_summary(summary, "rank")["market_cap_rank"].tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        reporting.sort_summary(summary, "nonsense")


# ---------------------------------------------------------------------------
# API client: validation and error handling
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


class FakeSession:
    """Returns queued responses (or raises queued exceptions) in order."""

    def __init__(self, queue):
        self.headers = {}
        self.queue = list(queue)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(queue, **kwargs):
    kwargs.setdefault("min_interval", 0)
    return data_fetcher.CoinGeckoClient(session=FakeSession(queue), sleep=lambda s: None, **kwargs)


def test_client_retries_rate_limit_then_succeeds():
    client = make_client([FakeResponse(429, headers={"Retry-After": "1"}), FakeResponse(200, {"ok": True})])
    assert client.get("ping") == {"ok": True}
    assert client.rate_limit_hits == 1
    assert len(client.session.calls) == 2


def test_client_slows_down_after_rate_limit():
    client = make_client([FakeResponse(429), FakeResponse(200, {"ok": True})], min_interval=2.0)
    client.get("ping")
    assert client.min_interval > 2.0                       # adaptive throttling


def test_client_recovers_interval_after_a_run_of_successes():
    client = make_client([FakeResponse(429)] + [FakeResponse(200, {"ok": True})] * 12, min_interval=2.0)
    client.get("ping")                                     # 429 -> slows down
    slowed = client.min_interval
    assert slowed > 2.0
    for _ in range(11):                                    # sustained success -> eases back
        client.get("ping")
    assert 2.0 <= client.min_interval < slowed             # never below the base interval


def test_client_retries_server_errors_and_network_errors():
    client = make_client([FakeResponse(503), requests.ConnectionError("boom"), FakeResponse(200, [1])])
    assert client.get("ping") == [1]
    assert len(client.session.calls) == 3


def test_client_does_not_retry_ssl_errors():
    client = make_client([requests.exceptions.SSLError("bad certificate")])
    with pytest.raises(data_fetcher.CoinGeckoError, match="certificate"):
        client.get("ping")
    assert len(client.session.calls) == 1                   # failed fast


def test_client_raises_on_client_error_and_gives_up_eventually():
    with pytest.raises(data_fetcher.CoinGeckoError, match="404"):
        make_client([FakeResponse(404, text="not found")]).get("coins/nope")
    client = make_client([FakeResponse(429)] * 3, max_retries=2)
    with pytest.raises(data_fetcher.CoinGeckoError, match="Giving up"):
        client.get("ping")


def test_client_detects_error_payload_with_http_200():
    client = make_client([FakeResponse(200, {"status": {"error_code": 10005, "error_message": "exclusive"}})])
    with pytest.raises(data_fetcher.CoinGeckoError, match="API error"):
        client.get("coins/markets")


def test_validate_market_records_drops_invalid_and_duplicates():
    records = [
        {"id": "bitcoin", "name": "Bitcoin", "symbol": "btc", "current_price": 70000, "market_cap": 1e12},
        {"id": "bitcoin", "name": "Bitcoin", "symbol": "btc", "current_price": 70000, "market_cap": 1e12},  # dup
        {"id": "broken", "name": "Broken", "symbol": "brk", "current_price": None, "market_cap": 1e9},
        {"id": "nocap", "name": "No cap", "symbol": "nc", "current_price": 1.0, "market_cap": 0},
        "not-a-dict",
        {"id": "ether", "name": "Ether", "symbol": "eth", "current_price": 3000, "market_cap": 4e11},
    ]
    valid = data_fetcher.validate_market_records(records)
    assert [r["id"] for r in valid] == ["bitcoin", "ether"]            # sorted by market cap, deduplicated


def test_validate_market_chart_rejects_unusable_payloads():
    with pytest.raises(data_fetcher.InvalidResponseError):
        data_fetcher.validate_market_chart({"market_caps": []}, "x")
    with pytest.raises(data_fetcher.InvalidResponseError):
        data_fetcher.validate_market_chart({"prices": [[1, 2]]}, "x")   # fewer than 2 points
    chart = data_fetcher.validate_market_chart(
        {"prices": [[1, 2], [2, 3], ["bad", 4], [3, None]], "market_caps": [[1, 9]]}, "x")
    assert chart["prices"][:2] == [(1.0, 2.0), (2.0, 3.0)]
    assert math.isnan(chart["prices"][2][1])                            # None value kept as NaN
    assert len(chart["prices"]) == 3                                    # malformed pair dropped


def test_chart_to_frame_merges_series_and_drops_duplicate_timestamps():
    chart = {
        "prices": [(1_000.0, 10.0), (2_000.0, 11.0), (2_000.0, 12.0)],
        "market_caps": [(1_000.0, 100.0), (2_000.0, 110.0)],
        "total_volumes": [(1_000.0, 5.0)],
    }
    frame, duplicates = data_fetcher.chart_to_frame("coin", chart)
    assert duplicates == 1
    assert frame["price"].tolist() == [10.0, 12.0]                      # keeps the last duplicate
    assert frame["market_cap"].tolist() == [100.0, 110.0]
    assert math.isnan(frame["total_volume"].iloc[1])                    # missing -> NaN, not dropped


def test_get_top_coins_returns_requested_number():
    payload = [{"id": f"c{i}", "name": f"Coin {i}", "symbol": f"c{i}", "current_price": 1.0 + i,
                "market_cap": 1e9 - i} for i in range(20)]
    client = make_client([FakeResponse(200, payload)])
    assert len(client.get_top_coins(5)) == 5
