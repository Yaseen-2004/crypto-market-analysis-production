"""Shared synthetic data builders for the test-suite (no network access)."""

from __future__ import annotations

import numpy as np
import pandas as pd

HOUR_MS = 3_600_000
START_MS = 1_789_059_600_000  # a fixed instant, so every test run is reproducible


def make_history(coin_id="testcoin", prices=(100.0, 110.0, 99.0, 99.0, 108.9), rank=1, start_ms=START_MS,
                 step_ms=HOUR_MS, name="Test Coin", symbol="TEST") -> pd.DataFrame:
    """Raw-format history (as produced by the fetcher) for one synthetic coin."""
    return pd.DataFrame(
        {
            "coin_id": coin_id,
            "timestamp_ms": [start_ms + i * step_ms for i in range(len(prices))],
            "price": list(prices),
            "market_cap": [p * 1_000 for p in prices],
            "total_volume": [5_000.0] * len(prices),
            "name": name,
            "symbol": symbol,
            "market_cap_rank": rank,
            "snapshot_price": prices[-1],
            "snapshot_market_cap": prices[-1] * 1_000,
            "snapshot_volume_24h": 5_000.0,
            "price_change_24h_pct": 1.0,
            "price_change_7d_pct": 2.0,
        }
    )


def synthetic_prices(n: int, start: float, volatility: float, seed: int, spike_at: int | None = None,
                     spike: float = 0.12) -> list[float]:
    """A reproducible random walk, optionally with one large jump (a guaranteed outlier)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, volatility, n)
    if spike_at is not None and 0 <= spike_at < n:
        steps[spike_at] = spike
    return list(start * np.cumprod(1 + steps))


def synthetic_market_data(n_coins: int = 6, n_points: int = 80) -> tuple[list[dict], dict[str, dict]]:
    """Build ``/coins/markets`` records and ``market_chart`` payloads for a fake API.

    The set deliberately contains a near-constant stablecoin, a very volatile
    coin and a coin with a single large spike, so the whole analysis (volatility
    ranking, outliers, low-variation flag) is exercised.
    """
    profiles = [
        ("bitcoin", "Bitcoin", "btc", 76_000.0, 0.004, 1.5e12, 40),
        ("ethereum", "Ethereum", "eth", 2_400.0, 0.007, 2.9e11, None),
        ("tether", "Tether", "usdt", 1.0, 0.00003, 1.8e11, None),     # near-constant
        ("solana", "Solana", "sol", 118.0, 0.010, 5.7e10, 20),
        ("dogecoin", "Dogecoin", "doge", 0.0825, 0.014, 1.2e10, None),
        ("volatilecoin", "Volatile Coin", "vol", 0.5, 0.035, 9.0e8, 60),
    ][:n_coins]

    markets, charts = [], {}
    for rank, (coin_id, name, symbol, start, vol, cap, spike_at) in enumerate(profiles, start=1):
        prices = synthetic_prices(n_points, start, vol, seed=rank, spike_at=spike_at)
        timestamps = [START_MS + i * HOUR_MS for i in range(n_points)]
        charts[coin_id] = {
            "prices": [[t, p] for t, p in zip(timestamps, prices)],
            "market_caps": [[t, p * cap / start] for t, p in zip(timestamps, prices)],
            "total_volumes": [[t, cap / 20] for t in timestamps],
        }
        markets.append({
            "id": coin_id, "name": name, "symbol": symbol, "current_price": prices[-1], "market_cap": cap,
            "market_cap_rank": rank, "total_volume": cap / 20,
            "price_change_percentage_24h_in_currency": (prices[-1] / prices[-24] - 1) * 100,
            "price_change_percentage_7d_in_currency": (prices[-1] / prices[0] - 1) * 100,
            "circulating_supply": cap / prices[-1], "last_updated": "2026-09-16T04:00:00.000Z",
        })
    return markets, charts


class FakeCoinGeckoClient:
    """Stands in for ``CoinGeckoClient`` - serves synthetic payloads, no network."""

    def __init__(self, n_coins: int = 6, n_points: int = 80, fail: set[str] | None = None,
                 short: dict[str, int] | None = None):
        self.markets, self.charts = synthetic_market_data(n_coins, n_points)
        for coin_id, points in (short or {}).items():  # simulate a newly listed coin
            for key in ("prices", "market_caps", "total_volumes"):
                self.charts[coin_id][key] = self.charts[coin_id][key][-points:]
        self.fail = fail or set()
        self.request_count = 0
        self.rate_limit_hits = 0
        self.requested: list[str] = []

    @staticmethod
    def _sleep(seconds: float) -> None:
        return None

    def get_top_coins(self, n: int = 100, vs_currency: str = "usd") -> list[dict]:
        self.request_count += 1
        return self.markets[:n]

    def get_market_chart(self, coin_id: str, days: int = 5, vs_currency: str = "usd") -> dict:
        from src.data_fetcher import CoinGeckoError

        self.request_count += 1
        self.requested.append(coin_id)
        if coin_id in self.fail:
            raise CoinGeckoError(f"simulated failure for {coin_id}")
        return self.charts[coin_id]
