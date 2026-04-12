"""
Optional **settlement reference** helpers when Kalshi’s official index is unavailable.

``coinbase_close_usd_near`` returns the Coinbase BTC-USD **5m candle close** nearest a
Unix instant (not identical to Kalshi settlement). Use only for diagnostics and
proxy labels — production fair value should use Kalshi’s published rules when possible.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from btc_trend_features import COINBASE_CANDLES


def coinbase_close_usd_near(unix_ts: int, *, timeout: float = 35.0) -> float | None:
    """Nearest 5m bar close to ``unix_ts`` (UTC seconds)."""
    s = int(unix_ts) - 600
    e = int(unix_ts) + 600
    r = requests.get(
        COINBASE_CANDLES,
        params={"granularity": 300, "start": s, "end": e},
        timeout=timeout,
    )
    r.raise_for_status()
    rows: list[list[Any]] = r.json()
    if not rows:
        return None
    best = min(rows, key=lambda row: abs(int(row[0]) - unix_ts))
    return float(best[4])


def sleep_s_paced(last_call: list[float], min_interval: float) -> None:
    """Mutable ``last_call[0]`` holds last time.time(); sleep to respect min_interval."""
    now = time.time()
    wait = min_interval - (now - last_call[0])
    if wait > 0:
        time.sleep(wait)
    last_call[0] = time.time()
