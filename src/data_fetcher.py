"""Data collection from the CoinGecko REST API.

Two endpoints are used:

* ``/coins/markets``            - current top-N coins by market capitalisation
* ``/coins/{id}/market_chart``  - price, market-cap and volume history

For a window of 2-90 days CoinGecko returns roughly hourly observations; the
original timestamp resolution is preserved.

Robustness features
-------------------
* Client-side throttling with an adaptive interval that grows after HTTP 429.
* Retries with exponential back-off + jitter for network errors, timeouts,
  HTTP 429 (honouring ``Retry-After``) and HTTP 5xx.
* Schema validation of every response; malformed records are dropped and logged.
* De-duplication of coins and of repeated timestamps inside a response.
* A second pass that retries coins that failed in the first pass.
* Optional on-disk JSON cache (disabled by default so every run is fresh).
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from . import config

logger = logging.getLogger(__name__)

HISTORY_COLUMNS = ["coin_id", "timestamp_ms", "price", "market_cap", "total_volume"]


_TRUST_STORE_READY = False


def enable_system_trust_store() -> bool:
    """Verify TLS against the operating-system certificate store, if possible.

    On machines behind a TLS-inspecting proxy or antivirus, the bundled
    ``certifi`` roots do not contain the intercepting CA and every HTTPS request
    fails with CERTIFICATE_VERIFY_FAILED. ``truststore`` makes Python use the OS
    store instead, which does trust it. Certificates are still verified - this
    never disables verification.
    """
    global _TRUST_STORE_READY
    if _TRUST_STORE_READY:
        return True
    try:
        import truststore

        truststore.inject_into_ssl()
        _TRUST_STORE_READY = True
        logger.debug("Using the OS certificate store for TLS verification (truststore)")
    except ImportError:
        logger.debug("truststore not installed; using certifi roots")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not enable the OS certificate store: %s", exc)
    return _TRUST_STORE_READY


class CoinGeckoError(RuntimeError):
    """The API could not return usable data (after all retries, if retryable)."""


class InvalidResponseError(CoinGeckoError):
    """A response was received but is structurally invalid."""


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------
class CoinGeckoClient:
    """Minimal, polite CoinGecko client with throttling and retries."""

    def __init__(
        self,
        api_key: str | None = None,
        plan: str | None = None,
        min_interval: float | None = None,
        max_retries: int = config.MAX_RETRIES,
        timeout: float = config.REQUEST_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if session is None:
            enable_system_trust_store()
        self.api_key = api_key if api_key is not None else (os.environ.get(config.API_KEY_ENV) or None)
        default_plan = "demo" if self.api_key else "public"
        self.plan = (plan or os.environ.get(config.API_PLAN_ENV) or default_plan).lower()

        if self.plan == "pro":
            self.base_url = config.PRO_BASE_URL
            key_header, default_interval = "x-cg-pro-api-key", config.MIN_INTERVAL_PRO
        else:
            self.base_url = config.PUBLIC_BASE_URL
            key_header = "x-cg-demo-api-key"
            default_interval = config.MIN_INTERVAL_DEMO if self.api_key else config.MIN_INTERVAL_PUBLIC

        self.session = session or requests.Session()
        self.session.headers.update({"Accept": "application/json", "User-Agent": "crypto-market-analysis/1.0"})
        if self.api_key:
            self.session.headers[key_header] = self.api_key

        self.min_interval = default_interval if min_interval is None else min_interval
        self.base_interval = self.min_interval  # floor that recovery cannot go below
        self._consecutive_ok = 0
        self.max_retries = max_retries
        self.timeout = timeout
        self._sleep = sleep
        self._last_request = -math.inf
        self.request_count = 0
        self.rate_limit_hits = 0

    # -- helpers -----------------------------------------------------------
    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            self._sleep(wait)

    @staticmethod
    def _backoff(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
        return min(cap, base * (2 ** attempt)) + random.uniform(0, 1)

    @staticmethod
    def _retry_after(response: requests.Response) -> float | None:
        value = response.headers.get("Retry-After")
        try:
            return min(float(value), 120.0) if value is not None else None
        except ValueError:
            return None

    def _on_rate_limit(self, attempt: int, response: requests.Response | None = None) -> None:
        self.rate_limit_hits += 1
        self._consecutive_ok = 0
        self.min_interval = min(self.min_interval * 1.5, 15.0)  # adapt: slow down
        delay = (self._retry_after(response) if response is not None else None) or self._backoff(
            attempt, base=10.0, cap=90.0
        )
        logger.warning(
            "Rate limited by CoinGecko; waiting %.0fs (attempt %d/%d, new interval %.1fs)",
            delay, attempt + 1, self.max_retries + 1, self.min_interval,
        )
        self._sleep(delay)

    def _speed_up(self, every: int = 10, factor: float = 0.85) -> None:
        """Ease back toward the base interval after a run of successful requests.

        Without this the throttle only ever slows down, so a single early 429
        would penalise the remaining requests of the whole run.
        """
        self._consecutive_ok += 1
        if self._consecutive_ok >= every and self.min_interval > self.base_interval:
            self.min_interval = max(self.base_interval, self.min_interval * factor)
            self._consecutive_ok = 0
            logger.debug("Recovering request interval to %.1fs", self.min_interval)

    # -- core request --------------------------------------------------------
    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """GET ``endpoint`` and return decoded JSON, retrying transient failures."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.exceptions.SSLError as exc:
                # A local trust-store problem, not a transient one: retrying cannot help.
                raise CoinGeckoError(
                    f"TLS certificate verification failed for {url}: {exc}. "
                    "If you are behind a corporate proxy or antivirus, install 'truststore' "
                    "(pip install truststore) or point REQUESTS_CA_BUNDLE at your CA bundle."
                ) from exc
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                delay = self._backoff(attempt)
                logger.warning("Network error on %s (%s); retrying in %.0fs", url, type(exc).__name__, delay)
                self._sleep(delay)
                continue
            except requests.RequestException as exc:  # malformed URL etc. - not retryable
                raise CoinGeckoError(f"Request failed for {url}: {exc}") from exc
            finally:
                self._last_request = time.monotonic()
                self.request_count += 1

            status = response.status_code
            if status == 429:
                last_error = CoinGeckoError("HTTP 429 Too Many Requests")
                self._on_rate_limit(attempt, response)
                continue
            if 500 <= status < 600:
                last_error = CoinGeckoError(f"HTTP {status}")
                delay = self._backoff(attempt)
                logger.warning("Server error %d on %s; retrying in %.0fs", status, url, delay)
                self._sleep(delay)
                continue
            if status >= 400:
                raise CoinGeckoError(f"HTTP {status} for {url}: {response.text[:200]}")

            try:
                payload = response.json()
            except ValueError:
                last_error = InvalidResponseError(f"Non-JSON response from {url}")
                self._sleep(self._backoff(attempt))
                continue

            # CoinGecko occasionally reports errors inside an HTTP 200 body.
            if isinstance(payload, dict):
                status_block = payload.get("status")
                if isinstance(status_block, dict) and status_block.get("error_code"):
                    if status_block.get("error_code") == 429:
                        last_error = CoinGeckoError("Rate limited (error payload)")
                        self._on_rate_limit(attempt)
                        continue
                    raise CoinGeckoError(f"API error for {url}: {status_block}")
                if "error" in payload and len(payload) <= 2:
                    raise InvalidResponseError(f"API error for {url}: {payload['error']}")
            self._speed_up()
            return payload

        raise CoinGeckoError(f"Giving up on {url} after {self.max_retries + 1} attempts: {last_error}")

    # -- endpoints -----------------------------------------------------------
    def get_top_coins(self, n: int = 100, vs_currency: str = "usd") -> list[dict[str, Any]]:
        """Return the top ``n`` valid coins ordered by market capitalisation."""
        per_page = min(250, n + 10)  # small buffer in case some records are invalid
        records: list[dict[str, Any]] = []
        valid: list[dict[str, Any]] = []
        page = 1
        while len(valid) < n and page <= 5:
            payload = self.get(
                "coins/markets",
                {
                    "vs_currency": vs_currency,
                    "order": "market_cap_desc",
                    "per_page": per_page,
                    "page": page,
                    "sparkline": "false",
                    "price_change_percentage": "24h,7d",
                },
            )
            if not isinstance(payload, list):
                raise InvalidResponseError(f"/coins/markets returned {type(payload).__name__}, expected list")
            if not payload:
                break
            records.extend(payload)
            valid = validate_market_records(records)
            page += 1
        if len(valid) < n:
            logger.warning("Only %d valid coins available (requested %d)", len(valid), n)
        return valid[:n]

    def get_market_chart(self, coin_id: str, days: int = 5, vs_currency: str = "usd") -> dict[str, list]:
        """Return validated price / market-cap / volume history for one coin."""
        payload = self.get(f"coins/{coin_id}/market_chart", {"vs_currency": vs_currency, "days": days})
        return validate_market_chart(payload, coin_id)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_market_records(records: list[Any]) -> list[dict[str, Any]]:
    """Drop invalid / duplicate market records and sort by market cap (desc)."""
    seen: set[str] = set()
    valid: list[dict[str, Any]] = []
    rejected = 0
    for rec in records:
        if not isinstance(rec, dict):
            rejected += 1
            continue
        coin_id, name, symbol = rec.get("id"), rec.get("name"), rec.get("symbol")
        if not (isinstance(coin_id, str) and coin_id and isinstance(name, str) and isinstance(symbol, str)):
            rejected += 1
            continue
        if not (_is_number(rec.get("current_price")) and rec["current_price"] > 0):
            rejected += 1
            continue
        if not (_is_number(rec.get("market_cap")) and rec["market_cap"] > 0):
            rejected += 1
            continue
        if coin_id in seen:  # duplicate coin across pages
            rejected += 1
            continue
        seen.add(coin_id)
        valid.append(rec)
    if rejected:
        logger.info("Dropped %d invalid or duplicate market records", rejected)
    valid.sort(key=lambda r: r["market_cap"], reverse=True)
    return valid


def _clean_pairs(pairs: Any) -> list[tuple[float, float]]:
    """Keep ``[timestamp_ms, value]`` pairs with a numeric timestamp; bad values -> NaN."""
    if not isinstance(pairs, list):
        return []
    cleaned = []
    for pair in pairs:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2 and _is_number(pair[0]):
            value = float(pair[1]) if _is_number(pair[1]) else float("nan")
            cleaned.append((float(pair[0]), value))
    return cleaned


def validate_market_chart(payload: Any, coin_id: str = "?") -> dict[str, list]:
    """Validate a ``market_chart`` payload; raise ``InvalidResponseError`` if unusable."""
    if not isinstance(payload, dict) or "prices" not in payload:
        raise InvalidResponseError(f"{coin_id}: market_chart response has no 'prices'")
    prices = _clean_pairs(payload.get("prices"))
    if len(prices) < 2:
        raise InvalidResponseError(f"{coin_id}: fewer than 2 valid price points")
    return {
        "prices": prices,
        "market_caps": _clean_pairs(payload.get("market_caps")),
        "total_volumes": _clean_pairs(payload.get("total_volumes")),
    }


# ---------------------------------------------------------------------------
# Conversion to DataFrames
# ---------------------------------------------------------------------------
def chart_to_frame(coin_id: str, chart: dict[str, list]) -> tuple[pd.DataFrame, int]:
    """Merge the three history arrays on timestamp. Returns (frame, duplicates_dropped)."""
    frames = []
    duplicates = 0
    for key, column in (("prices", "price"), ("market_caps", "market_cap"), ("total_volumes", "total_volume")):
        part = pd.DataFrame(chart.get(key, []), columns=["timestamp_ms", column])
        before = len(part)
        part = part.drop_duplicates(subset="timestamp_ms", keep="last")
        if key == "prices":
            duplicates = before - len(part)
        frames.append(part)
    merged = frames[0]
    for part in frames[1:]:
        merged = merged.merge(part, on="timestamp_ms", how="left")
    merged.insert(0, "coin_id", coin_id)
    merged["timestamp_ms"] = merged["timestamp_ms"].astype("int64")
    return merged[HISTORY_COLUMNS], duplicates


def markets_to_frame(records: list[dict[str, Any]], fetched_at: datetime) -> pd.DataFrame:
    """Flatten validated market records into a tidy snapshot table."""
    rows = []
    for position, rec in enumerate(records, start=1):
        rows.append(
            {
                "coin_id": rec["id"],
                "name": rec["name"],
                "symbol": str(rec["symbol"]).upper(),
                "market_cap_rank": rec.get("market_cap_rank") or position,
                "current_price": rec.get("current_price"),
                "market_cap": rec.get("market_cap"),
                "total_volume_24h": rec.get("total_volume"),
                "price_change_24h_pct": rec.get("price_change_percentage_24h_in_currency",
                                                rec.get("price_change_percentage_24h")),
                "price_change_7d_pct": rec.get("price_change_percentage_7d_in_currency"),
                "circulating_supply": rec.get("circulating_supply"),
                "last_updated": rec.get("last_updated"),
                "fetched_at_utc": fetched_at.isoformat(timespec="seconds"),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def _read_cache(path: Path, max_age_minutes: float) -> Any | None:
    if max_age_minutes <= 0 or not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(blob["fetched_at"])
        age_minutes = (datetime.now(timezone.utc) - fetched).total_seconds() / 60
        return blob["payload"] if age_minutes <= max_age_minutes else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache(path: Path, payload: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {"fetched_at": datetime.now(timezone.utc).isoformat(), "payload": payload}
        path.write_text(json.dumps(blob), encoding="utf-8")
    except OSError as exc:  # caching is best-effort only
        logger.debug("Could not write cache %s: %s", path, exc)


# ---------------------------------------------------------------------------
# High-level fetch
# ---------------------------------------------------------------------------
@dataclass
class FetchResult:
    markets: pd.DataFrame   # one row per coin (current snapshot)
    history: pd.DataFrame   # long format: one row per coin x timestamp
    log: pd.DataFrame       # per-coin fetch status
    fetched_at: datetime


def fetch_market_data(
    top_n: int = 100,
    days: int = 5,
    vs_currency: str = "usd",
    cache_minutes: float = 0.0,
    cache_dir: Path | None = None,
    client: CoinGeckoClient | None = None,
    retry_cooldown: float = 30.0,
    history_cache_minutes: float | None = None,
) -> FetchResult:
    """Fetch the top-N snapshot plus ``days`` of history for every coin.

    ``cache_minutes`` governs the market snapshot; ``history_cache_minutes``
    (default: the same value) governs the per-coin history. They differ for a
    frequent refresh loop: CoinGecko's multi-day history is hourly, so it only
    gains a bar once an hour, while live prices change constantly - refreshing
    the one-call snapshot every cycle and the 100-call history hourly keeps the
    loop inside the free tier's rate limit.
    """
    if history_cache_minutes is None:
        history_cache_minutes = cache_minutes
    # Resolved at call time (not as a default) so the location stays configurable.
    cache_dir = Path(cache_dir) if cache_dir is not None else config.CACHE_DIR
    client = client or CoinGeckoClient()
    fetched_at = datetime.now(timezone.utc)

    # 1) Top-N snapshot ----------------------------------------------------
    markets_cache = cache_dir / f"markets_top{top_n}_{vs_currency}.json"
    records = _read_cache(markets_cache, cache_minutes)
    if records is not None:
        records = validate_market_records(records)[:top_n]
        logger.info("Loaded top-%d market snapshot from cache", len(records))
    else:
        records = client.get_top_coins(top_n, vs_currency)
        _write_cache(markets_cache, records)
    if not records:
        raise CoinGeckoError("No valid market records returned")
    markets = markets_to_frame(records, fetched_at)
    logger.info("Top %d coins retrieved (largest: %s)", len(markets), markets.iloc[0]["name"])

    # 2) Per-coin history ----------------------------------------------------
    frames: dict[str, pd.DataFrame] = {}
    log: dict[str, dict[str, Any]] = {}

    def fetch_one(coin_id: str, name: str) -> None:
        cache_path = cache_dir / f"chart_{coin_id}_{days}d_{vs_currency}.json"
        source = "cache"
        try:
            payload = _read_cache(cache_path, history_cache_minutes)
            if payload is not None:
                chart = validate_market_chart(payload, coin_id)
            else:
                source = "api"
                chart = client.get_market_chart(coin_id, days, vs_currency)
                _write_cache(cache_path, {k: [list(p) for p in v] for k, v in chart.items()})
            frame, dups = chart_to_frame(coin_id, chart)
            frames[coin_id] = frame
            log[coin_id] = {"coin_id": coin_id, "name": name, "status": "ok", "source": source,
                            "points": len(frame), "duplicate_timestamps_dropped": dups, "error": ""}
        except CoinGeckoError as exc:
            log[coin_id] = {"coin_id": coin_id, "name": name, "status": "failed", "source": source,
                            "points": 0, "duplicate_timestamps_dropped": 0, "error": str(exc)[:300]}
            logger.warning("Failed to fetch %s: %s", coin_id, exc)

    total = len(markets)
    for i, (coin_id, name) in enumerate(zip(markets["coin_id"], markets["name"]), start=1):
        fetch_one(coin_id, name)
        if log[coin_id]["status"] == "ok":
            logger.info("[%3d/%d] %-26s %4d points (%s)", i, total, name[:26], log[coin_id]["points"],
                        log[coin_id]["source"])

    failed = [(c, n) for c, n in zip(markets["coin_id"], markets["name"]) if log[c]["status"] != "ok"]
    if failed:
        logger.info("Retrying %d failed coin(s) after a %.0fs cool-down", len(failed), retry_cooldown)
        client._sleep(retry_cooldown)
        for coin_id, name in failed:
            fetch_one(coin_id, name)

    if not frames:
        raise CoinGeckoError("History could not be retrieved for any coin")

    history = pd.concat([frames[c] for c in markets["coin_id"] if c in frames], ignore_index=True)
    snapshot = markets.rename(
        columns={
            "current_price": "snapshot_price",
            "market_cap": "snapshot_market_cap",
            "total_volume_24h": "snapshot_volume_24h",
        }
    )[["coin_id", "name", "symbol", "market_cap_rank", "snapshot_price", "snapshot_market_cap",
       "snapshot_volume_24h", "price_change_24h_pct", "price_change_7d_pct"]]
    history = history.merge(snapshot, on="coin_id", how="left")
    history["timestamp_utc"] = pd.to_datetime(history["timestamp_ms"], unit="ms", utc=True).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    log_df = pd.DataFrame([log[c] for c in markets["coin_id"]])
    logger.info(
        "History retrieved for %d/%d coins (%d requests, %d rate-limit responses)",
        len(frames), total, client.request_count, client.rate_limit_hits,
    )
    return FetchResult(markets=markets, history=history, log=log_df, fetched_at=fetched_at)
