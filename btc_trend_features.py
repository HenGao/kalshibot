"""
BTC trend features from Coinbase 5m candles + Kalshi strike / target context.

Used by training scripts, edge_bot --btc-trend-model, and KXBTC15M backtests.

Feature set v2 adds realized vol, path vs strike, interaction terms, and optional
order-book context (filled live from Kalshi orderbook; zeros in historical training).

v3 adds UTC time-of-week cyclical encodings and vol_15m z-score vs the local kline window
(regime). Pass ``decision_ts_utc`` (seconds); if omitted, uses last candle start + 300s.
"""

from __future__ import annotations

import math
import re
import statistics
import time
from datetime import datetime, timezone
from typing import Any

import requests

from edge_math import parse_orderbook

# Coinbase Exchange public API (US-friendly). Candles: [time, low, high, open, close, vol]
COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
COINBASE_SPOT = "https://api.coinbase.com/v2/prices/BTC-USD/spot"

# Ordered list expected by saved models (logistic or HGB).
BTC_TREND_FEATURE_NAMES: tuple[str, ...] = (
    "ret_5m",
    "ret_15m",
    "ret_1h",
    "ret_6h",
    "dist_pct",
    "log1p_seconds_before_close",
    "vol_15m",
    "vol_1h",
    "range_15m_pct",
    "path_extreme_pct",
    "dist_x_vol15",
    "ret15_x_logsec",
    "book_yes_spread",
    "book_yes_imbalance",
    "hour_utc_sin",
    "hour_utc_cos",
    "dow_sin",
    "dow_cos",
    "vol_15m_z",
)

_KLINES_CACHE: tuple[float, list[list[Any]]] | None = None
_KLINES_TTL_SEC = 25.0


def _parse_iso_ms(iso: str) -> int:
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_btc_spot_usd() -> float:
    r = requests.get(COINBASE_SPOT, timeout=20)
    r.raise_for_status()
    return float(r.json()["data"]["amount"])


def fetch_coinbase_5m_range(
    start_ts: int,
    end_ts: int,
    *,
    sleep_s: float = 0.0,
) -> list[list[Any]]:
    """
    Merge Coinbase 5m candles [time, low, high, open, close, vol] for [start_ts, end_ts).
    Used by Kalshi-label training and backtests.
    """
    by_time: dict[int, list[Any]] = {}
    e = end_ts
    while e > start_ts:
        s = max(start_ts, e - 300 * 300)
        r = requests.get(
            COINBASE_CANDLES,
            params={"start": s, "end": e, "granularity": 300},
            timeout=60,
        )
        r.raise_for_status()
        chunk = r.json()
        for c in chunk:
            t0 = int(c[0])
            if start_ts <= t0 < end_ts:
                by_time[t0] = c
        e = s
        if sleep_s > 0:
            time.sleep(sleep_s)
    return [by_time[k] for k in sorted(by_time)]


def fetch_klines_5m(*, limit: int = 300, end_ts: int | None = None) -> list[list[Any]]:
    """
    Coinbase allows up to 300 candles per request (granularity 300 = 5m).
    `end_ts`: exclusive end in seconds since epoch (optional).
    """
    global _KLINES_CACHE
    now = time.time()
    if (
        _KLINES_CACHE is not None
        and end_ts is None
        and now - _KLINES_CACHE[0] < _KLINES_TTL_SEC
        and limit <= len(_KLINES_CACHE[1])
    ):
        return _KLINES_CACHE[1][-limit:]

    cap = min(limit, 300)
    end = int(end_ts if end_ts is not None else now)
    start = end - cap * 300
    r = requests.get(
        COINBASE_CANDLES,
        params={"granularity": 300, "start": start, "end": end},
        timeout=35,
    )
    r.raise_for_status()
    data = r.json()
    data.sort(key=lambda row: int(row[0]))
    if end_ts is None:
        _KLINES_CACHE = (now, data)
    return data


def _close_series(klines: list[list[Any]]) -> list[float]:
    return [float(k[4]) for k in klines]


def log_returns_from_closes(closes: list[float], spans: list[int]) -> dict[str, float]:
    """For each span S, log(close[-1]/close[-1-S]); missing -> 0.0."""
    out: dict[str, float] = {}
    last = closes[-1]
    for s in spans:
        key = f"ret_{s}b"
        if len(closes) <= s or last <= 0 or closes[-1 - s] <= 0:
            out[key] = 0.0
        else:
            out[key] = math.log(last / closes[-1 - s])
    return out


def _pstdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(statistics.pstdev(xs))


def _pairwise_log_returns(closes: list[float]) -> list[float]:
    out: list[float] = []
    for i in range(1, len(closes)):
        a, b = closes[i - 1], closes[i]
        if a > 0 and b > 0:
            out.append(math.log(b / a))
    return out


def _vol_15m_z_from_closes(closes: list[float]) -> float:
    """Z-score of terminal 15m vol vs prior 15m vols along the same close path."""
    vols: list[float] = []
    for end in range(2, len(closes)):
        sub = closes[: end + 1]
        r = _pairwise_log_returns(sub)
        if len(r) >= 3:
            vols.append(_pstdev(r[-3:]))
        else:
            vols.append(0.0)
    if len(vols) < 6:
        return 0.0
    cur = vols[-1]
    hist = vols[:-1]
    mu = float(statistics.mean(hist))
    sd = float(statistics.pstdev(hist))
    if sd < 1e-12:
        return 0.0
    z = (cur - mu) / sd
    return max(-5.0, min(5.0, z))


def _time_cyclic_features(decision_ts_utc: int) -> dict[str, float]:
    dt = datetime.fromtimestamp(float(decision_ts_utc), tz=timezone.utc)
    dow = float(dt.weekday())
    hod = float(dt.hour) + float(dt.minute) / 60.0 + float(dt.second) / 3600.0
    return {
        "hour_utc_sin": math.sin(2.0 * math.pi * hod / 24.0),
        "hour_utc_cos": math.cos(2.0 * math.pi * hod / 24.0),
        "dow_sin": math.sin(2.0 * math.pi * dow / 7.0),
        "dow_cos": math.cos(2.0 * math.pi * dow / 7.0),
    }


def features_from_klines_window(
    klines: list[list[Any]],
    *,
    seconds_before_close: float,
    target_usd: float,
    book_yes_spread: float = 0.0,
    book_yes_imbalance: float = 0.0,
    decision_ts_utc: int | None = None,
) -> dict[str, float]:
    """
    Full v2 feature dict from a chronological 5m kline window ending at decision bar.
    Each row: [time, low, high, open, close, vol].
    """
    if len(klines) < 80:
        raise ValueError(f"need >=80 klines, got {len(klines)}")
    if target_usd <= 0:
        raise ValueError("target_usd must be positive")

    closes = _close_series(klines)
    highs = [float(k[2]) for k in klines]
    lows = [float(k[1]) for k in klines]
    spot = closes[-1]

    lr1 = log_returns_from_closes(closes, [1, 3, 12, 72])
    dist_pct = (spot - target_usd) / target_usd * 100.0
    logsec = math.log1p(max(0.0, seconds_before_close))
    ret_15m = float(lr1["ret_3b"])

    rets = _pairwise_log_returns(closes)
    vol_15m = _pstdev(rets[-3:]) if len(rets) >= 3 else 0.0
    vol_1h = _pstdev(rets[-12:]) if len(rets) >= 12 else _pstdev(rets) if rets else 0.0

    n3 = min(3, len(klines))
    w3 = klines[-n3:]
    hh = max(float(x[2]) for x in w3)
    ll = min(float(x[1]) for x in w3)
    range_15m_pct = (hh - ll) / target_usd * 100.0

    n12 = min(12, len(klines))
    w12 = klines[-n12:]
    mx = 0.0
    for x in w12:
        h, lowv = float(x[2]), float(x[1])
        mx = max(mx, abs(h - target_usd), abs(lowv - target_usd))
    path_extreme_pct = mx / target_usd * 100.0

    if decision_ts_utc is None:
        decision_ts_utc = int(klines[-1][0]) + 300
    tc = _time_cyclic_features(int(decision_ts_utc))
    vol_z = _vol_15m_z_from_closes(closes)

    return {
        "ret_5m": float(lr1["ret_1b"]),
        "ret_15m": ret_15m,
        "ret_1h": float(lr1["ret_12b"]),
        "ret_6h": float(lr1["ret_72b"]),
        "dist_pct": float(dist_pct),
        "log1p_seconds_before_close": float(logsec),
        "vol_15m": float(vol_15m),
        "vol_1h": float(vol_1h),
        "range_15m_pct": float(range_15m_pct),
        "path_extreme_pct": float(path_extreme_pct),
        "dist_x_vol15": float(dist_pct * vol_15m),
        "ret15_x_logsec": float(ret_15m * logsec),
        "book_yes_spread": float(book_yes_spread),
        "book_yes_imbalance": float(book_yes_imbalance),
        "hour_utc_sin": float(tc["hour_utc_sin"]),
        "hour_utc_cos": float(tc["hour_utc_cos"]),
        "dow_sin": float(tc["dow_sin"]),
        "dow_cos": float(tc["dow_cos"]),
        "vol_15m_z": float(vol_z),
    }


def book_features_from_orderbook(orderbook_json: dict[str, Any]) -> dict[str, float]:
    """Top-of-book YES spread and a simple bid-size imbalance proxy."""
    ex = parse_orderbook(orderbook_json)
    ysp = float(ex.yes_spread)
    ob = orderbook_json.get("orderbook_fp") or orderbook_json.get("orderbook") or {}
    yl = ob.get("yes_dollars") or []
    nl = ob.get("no_dollars") or []

    def _best_sz(levels: list[Any]) -> float:
        if not levels:
            return 0.0
        row = levels[-1]
        if not isinstance(row, (list, tuple)) or len(row) < 1:
            return 0.0
        if len(row) > 1:
            try:
                return float(row[1])
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    ysz = _best_sz(yl)
    nsz = _best_sz(nl)
    tot = ysz + nsz
    imb = (ysz - nsz) / tot if tot > 1e-9 else 0.0
    return {"book_yes_spread": ysp, "book_yes_imbalance": imb}


def parse_ticker_strike_usd(ticker: str) -> float | None:
    """KXBTC-...-T80799.99 style suffix."""
    m = re.search(r"-T([\d.]+)\s*$", ticker, re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def resolve_kalshi_btc_target_usd(
    market: dict[str, Any],
    *,
    ticker: str,
    spot_hint: float | None = None,
) -> tuple[float | None, str]:
    """
    Return (target_usd, source_tag) for distance / context.
    For 'greater' markets, target is the hurdle (floor_strike).
    """
    m = market
    st = m.get("strike_type") or ""
    floor = m.get("floor_strike")
    cap = m.get("cap_strike")

    if st in ("greater", "greater_or_equal") and floor is not None:
        return float(floor), "floor_strike"
    if st == "less" and cap is not None:
        return float(cap), "cap_strike"
    if st == "between" and floor is not None and cap is not None:
        return (float(floor) + float(cap)) / 2.0, "mid_range"
    if st == "between" and floor is not None:
        return float(floor), "floor_only"

    t = parse_ticker_strike_usd(ticker)
    if t is not None:
        return t, "ticker"

    ot = m.get("open_time")
    if ot:
        try:
            ref = btc_usd_near_timestamp(str(ot))
            return ref, "open_time"
        except Exception:
            pass
    if spot_hint is not None:
        return float(spot_hint), "spot_fallback"
    return None, "unknown"


def btc_usd_near_timestamp(open_time_iso: str) -> float:
    """Approximate BTC-USD at open using 5m candles bracketing open_time."""
    ms = _parse_iso_ms(open_time_iso)
    tsec = ms // 1000
    r = requests.get(
        COINBASE_CANDLES,
        params={
            "granularity": 300,
            "start": tsec - 900,
            "end": tsec + 900,
        },
        timeout=35,
    )
    r.raise_for_status()
    kl = r.json()
    if not kl:
        raise ValueError("empty candles")
    best = min(kl, key=lambda row: abs(int(row[0]) - tsec))
    return float(best[4])


def build_btc_trend_features_from_closes(
    closes: list[float],
    *,
    seconds_before_close: float,
    target_usd: float,
    book_yes_spread: float = 0.0,
    book_yes_imbalance: float = 0.0,
    decision_ts_utc: int | None = None,
) -> dict[str, float]:
    """Sim / fallback: use close-only path (high=low=close per bar)."""
    if len(closes) < 80:
        raise ValueError(f"need >=80 closes, got {len(closes)}")
    klines = [[i, c, c, c, c, 0.0] for i, c in enumerate(closes)]
    return features_from_klines_window(
        klines,
        seconds_before_close=seconds_before_close,
        target_usd=target_usd,
        book_yes_spread=book_yes_spread,
        book_yes_imbalance=book_yes_imbalance,
        decision_ts_utc=decision_ts_utc,
    )


def build_btc_trend_feature_dict(
    *,
    seconds_before_close: float,
    target_usd: float,
    spot_usd: float,
    klines_5m: list[list[Any]] | None = None,
    orderbook_json: dict[str, Any] | None = None,
    decision_ts_utc: int | None = None,
) -> dict[str, float]:
    """
    Live path: fetch Coinbase 5m (unless klines provided), align last close to spot_usd,
    optional Kalshi order book for book_* features.
    """
    if decision_ts_utc is None:
        decision_ts_utc = int(time.time())

    if klines_5m is None:
        klines_5m = fetch_klines_5m(limit=300)
    klines = [list(r) for r in klines_5m]
    closes = _close_series(klines)
    if closes and abs(closes[-1] - spot_usd) / max(spot_usd, 1e-9) > 0.002:
        row = klines[-1]
        su = float(spot_usd)
        row[4] = su
        row[1] = min(float(row[1]), su)
        row[2] = max(float(row[2]), su)

    book_spread = 0.0
    book_imb = 0.0
    if orderbook_json is not None:
        bf = book_features_from_orderbook(orderbook_json)
        book_spread = bf["book_yes_spread"]
        book_imb = bf["book_yes_imbalance"]

    return features_from_klines_window(
        klines,
        seconds_before_close=seconds_before_close,
        target_usd=target_usd,
        book_yes_spread=book_spread,
        book_yes_imbalance=book_imb,
        decision_ts_utc=decision_ts_utc,
    )
