"""
data_feed.py — Fetches all live market data needed by the bot.

Sources (all free, no API key required):
  Coinbase Exchange REST → BTC 1-min OHLCV candles  (Binance is geo-blocked in the US)
  OKX public REST        → BTC perpetual funding rate
  Hardcoded calendar     → High-impact macro events (FOMC, CPI, NFP)
  Kalshi public REST     → KXBTC15M / KXBTC Yes/No prices (no auth needed)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

# ── API endpoint constants ────────────────────────────────────────────────────
# Coinbase Exchange (Advanced Trade) — public, no auth, works in the US
COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
# OKX perpetual swap funding rate — public, no auth, works in the US
OKX_FUNDING_URL      = "https://www.okx.com/api/v5/public/funding-rate"
KALSHI_PUBLIC_BASE   = "https://api.elections.kalshi.com/trade-api/v2"

REQUEST_TIMEOUT = 10   # seconds before giving up on an API call

# ── Hardcoded high-impact macro event timestamps (Unix seconds, UTC) ──────────
# These fire a 30-minute blackout window: bot sits out near major news events.
# Update this list from https://www.forexfactory.com before each month.
# Format: add the Unix timestamp for each FOMC/CPI/NFP release.
_MACRO_EVENTS_UTC: list[int] = [
    # ── 2025 FOMC meetings (2:00 PM ET = 18:00 UTC) ───────────────────────────
    1748030400,   # 2025-05-21 18:00 UTC
    1753214400,   # 2025-07-22 18:00 UTC
    1758398400,   # 2025-09-17 18:00 UTC
    1763582400,   # 2025-11-05 18:00 UTC
    1766260800,   # 2025-12-17 18:00 UTC
    # ── 2025 CPI releases (8:30 AM ET = 13:30 UTC, 12:30 UTC in winter) ───────
    1746451800,   # 2025-05-13 12:30 UTC
    1749043800,   # 2025-06-11 12:30 UTC
    1751635800,   # 2025-07-15 12:30 UTC
    1754314200,   # 2025-08-12 12:30 UTC
    1756906200,   # 2025-09-10 12:30 UTC
    # ── 2025 NFP (first Friday of month, 8:30 AM ET) ──────────────────────────
    1746368400,   # 2025-05-02 12:30 UTC
    1748960400,   # 2025-06-06 12:30 UTC
    1751466000,   # 2025-07-04 12:30 UTC
    1754144400,   # 2025-08-01 12:30 UTC
    1756736400,   # 2025-09-05 12:30 UTC
]

# How far before/after a macro event to block trading
MACRO_BLACKOUT_BEFORE_SECS = 30 * 60   # 30 min before
MACRO_BLACKOUT_AFTER_SECS  = 10 * 60   # 10 min after (dust settling)


# ─────────────────────────────────────────────────────────────────────────────
# A) BTC OHLCV candles from Binance
# ─────────────────────────────────────────────────────────────────────────────

def fetch_btc_candles(interval: str = "1m", limit: int = 60) -> pd.DataFrame:
    """
    Fetch BTC-USD OHLCV candles from Coinbase Exchange public API.

    Coinbase candle format (each row): [time, low, high, open, close, volume]
    where time is a Unix timestamp in seconds.

    Parameters
    ----------
    interval : candle width — only "1m" (60s) and "5m" (300s) are used by this bot
    limit    : number of candles to return (Coinbase max per request: 300)

    Returns
    -------
    DataFrame with columns: timestamp (UTC datetime), open, high, low, close, volume
    sorted oldest → newest
    """
    # Map interval string to Coinbase granularity in seconds
    granularity_map = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}
    granularity = granularity_map.get(interval, 60)

    try:
        resp = requests.get(
            COINBASE_CANDLES_URL,
            params={"granularity": granularity, "limit": limit},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"[data_feed] BTC candles fetch failed: {exc}") from exc

    if not raw:
        raise RuntimeError("[data_feed] Coinbase returned empty candle list")

    # Coinbase returns newest-first; reverse to get oldest-first (standard convention)
    raw = list(reversed(raw))

    # Each row: [unix_ts_sec, low, high, open, close, volume]
    df = pd.DataFrame(raw, columns=["ts", "low", "high", "open", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    return df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# B) BTC perpetual futures funding rate from OKX
# ─────────────────────────────────────────────────────────────────────────────

def fetch_funding_rate() -> float:
    """
    Fetch the most recent BTC perpetual funding rate from OKX public API.
    (Binance futures API is geo-blocked in the US; OKX works fine.)

    Positive = longs pay shorts (longs overextended — bearish lean).
    Negative = shorts pay longs (shorts overextended — bullish lean).

    Returns
    -------
    float, e.g. -0.000167 means -0.0167% per 8-hour period
    """
    try:
        resp = requests.get(
            OKX_FUNDING_URL,
            params={"instId": "BTC-USD-SWAP"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"[data_feed] Funding rate fetch failed: {exc}") from exc

    data = body.get("data", [])
    if not data:
        raise RuntimeError("[data_feed] OKX returned empty funding rate response")

    return float(data[0]["fundingRate"])


# ─────────────────────────────────────────────────────────────────────────────
# C) Macro event blackout check
# ─────────────────────────────────────────────────────────────────────────────

def is_macro_event_soon() -> bool:
    """
    Returns True if a high-impact USD macro event (FOMC, CPI, NFP) is within
    the blackout window (30 min before or 10 min after the event).

    Update _MACRO_EVENTS_UTC with actual upcoming event timestamps each month.
    """
    now_ts = time.time()
    for event_ts in _MACRO_EVENTS_UTC:
        before_ok = now_ts < event_ts and event_ts - now_ts < MACRO_BLACKOUT_BEFORE_SECS
        after_ok  = event_ts <= now_ts < event_ts + MACRO_BLACKOUT_AFTER_SECS
        if before_ok or after_ok:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# D) Kalshi market data (public, no auth required)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_kalshi_market(series_ticker: str) -> dict[str, Any]:
    """
    Fetch the soonest-closing open Kalshi market for the given series.

    Parameters
    ----------
    series_ticker : "KXBTC15M" (15-minute) or "KXBTC" (1-hour)

    Returns
    -------
    dict with keys:
      ticker         — market ticker (e.g. "KXBTC15M-25MAY01-T95000")
      yes_ask        — best ask price for YES contracts (cents, 1–99)
      yes_bid        — best bid price for YES contracts (cents)
      no_ask         — best ask price for NO contracts (cents)
      no_bid         — best bid price for NO contracts (cents)
      yes_price      — midpoint of yes_ask and yes_bid
      implied_prob   — yes_price / 100  (market-implied P(BTC goes up))
      close_time     — UTC datetime when this window closes
      open_time      — UTC datetime when this window opened
      seconds_to_close — seconds remaining in this window
    """
    try:
        resp = requests.get(
            f"{KALSHI_PUBLIC_BASE}/markets",
            params={"series_ticker": series_ticker, "status": "open", "limit": 10},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"[data_feed] Kalshi market fetch failed for {series_ticker}: {exc}"
        ) from exc

    markets = data.get("markets", [])
    if not markets:
        raise RuntimeError(f"[data_feed] No open markets found for series {series_ticker}")

    # Find the market with the soonest close_time that hasn't passed yet
    now_utc = datetime.now(timezone.utc)
    best: dict[str, Any] | None = None
    best_close: datetime | None = None

    for m in markets:
        close_raw = m.get("close_time") or m.get("expiration_time")
        if not close_raw:
            continue
        close_dt = datetime.fromisoformat(close_raw.replace("Z", "+00:00"))
        if close_dt > now_utc:
            if best_close is None or close_dt < best_close:
                best = m
                best_close = close_dt

    if best is None or best_close is None:
        raise RuntimeError(f"[data_feed] No active future markets found for {series_ticker}")

    open_raw = best.get("open_time") or best.get("start_time")
    open_dt  = (
        datetime.fromisoformat(open_raw.replace("Z", "+00:00"))
        if open_raw else now_utc
    )

    # Prices are in cents (1–99). Use ask for the side we want to buy.
    yes_ask = float(best.get("yes_ask", 50))
    yes_bid = float(best.get("yes_bid", 50))
    no_ask  = float(best.get("no_ask",  50))
    no_bid  = float(best.get("no_bid",  50))

    # Midpoint price as the "fair" cost estimate
    yes_price = (yes_ask + yes_bid) / 2.0

    return {
        "ticker":           best["ticker"],
        "yes_ask":          yes_ask,
        "yes_bid":          yes_bid,
        "no_ask":           no_ask,
        "no_bid":           no_bid,
        "yes_price":        yes_price,
        "implied_prob":     yes_price / 100.0,
        "close_time":       best_close,
        "open_time":        open_dt,
        "seconds_to_close": (best_close - now_utc).total_seconds(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper: fetch everything at once
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all(series_tickers: list[str]) -> dict[str, Any]:
    """
    Fetch all data needed for one bot loop iteration.

    Parameters
    ----------
    series_tickers : list of series tickers to fetch, e.g. ["KXBTC15M", "KXBTC"]

    Returns
    -------
    dict with keys:
      candles          — pd.DataFrame of 60 1-min OHLCV candles
      funding_rate     — float, BTC perp funding rate
      macro_event_soon — bool, True if a macro event blackout is active
      markets          — dict[series_ticker → market dict | None]
    """
    candles      = fetch_btc_candles(interval="1m", limit=60)
    funding_rate = fetch_funding_rate()
    macro_soon   = is_macro_event_soon()

    markets: dict[str, Any] = {}
    for ticker in series_tickers:
        try:
            markets[ticker] = fetch_kalshi_market(ticker)
        except RuntimeError as exc:
            print(f"  [data_feed] WARNING — could not fetch {ticker}: {exc}")
            markets[ticker] = None

    return {
        "candles":          candles,
        "funding_rate":     funding_rate,
        "macro_event_soon": macro_soon,
        "markets":          markets,
    }
