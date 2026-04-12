#!/usr/bin/env python3
"""
Fetch Kalshi public trades for settled BTC short-horizon markets, label each
trade by whether the *taker* side won at settlement, and write:
  - per-trade CSV (timing + outcome)
  - JSON summary (counts and win-rate by time-to-close bucket)

Default series: KXBTC15M (Bitcoin up/down in next 15 minutes).

Note on "hourly": Kalshi's catalog exposes this up/down cadence for BTC as
KXBTC15M only. Hourly BTC products (e.g. KXBTC range, KXBTCD above/below) are
different contract types; pass --series if you extend this script after
verifying resolution rules match your research question.

This is descriptive statistics on historical prints — not a trading signal.
Run with -h for options.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Buckets are seconds BEFORE market close_time (trading cutoff).
DEFAULT_BUCKETS = [
    (0, 30),
    (30, 60),
    (60, 120),
    (120, 180),
    (180, 300),
    (300, 450),
    (450, 600),
    (600, 900),
]


def _parse_ts(iso: str) -> datetime:
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    return datetime.fromisoformat(iso)


def _http_get(url: str, sleep_s: float) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {e.code} for {url}\n{body}") from e
    if sleep_s > 0:
        time.sleep(sleep_s)
    return json.loads(raw.decode("utf-8"))


def iter_markets(
    series_ticker: str,
    *,
    max_markets: int,
    sleep_s: float,
) -> Iterator[dict[str, Any]]:
    cursor: str | None = None
    count = 0
    while True:
        q: dict[str, str] = {
            "series_ticker": series_ticker,
            "status": "settled",
            "limit": "200",
        }
        if cursor:
            q["cursor"] = cursor
        url = f"{BASE}/markets?{urllib.parse.urlencode(q)}"
        data = _http_get(url, sleep_s)
        markets = data.get("markets") or []
        for m in markets:
            yield m
            count += 1
            if max_markets and count >= max_markets:
                return
        cursor = data.get("cursor") or ""
        if not cursor:
            return


def iter_trades(ticker: str, sleep_s: float) -> Iterator[dict[str, Any]]:
    cursor: str | None = None
    while True:
        q: dict[str, str] = {"ticker": ticker, "limit": "1000"}
        if cursor:
            q["cursor"] = cursor
        url = f"{BASE}/markets/trades?{urllib.parse.urlencode(q)}"
        data = _http_get(url, sleep_s)
        for t in data.get("trades") or []:
            yield t
        cursor = data.get("cursor") or ""
        if not cursor:
            return


def taker_won(result: str, taker_side: str) -> bool:
    r = result.lower()
    s = taker_side.lower()
    if r not in {"yes", "no"} or s not in {"yes", "no"}:
        return False
    return r == s


def outcome_yes_int(result: str) -> int | None:
    r = (result or "").lower()
    if r == "yes":
        return 1
    if r == "no":
        return 0
    return None


def taker_pnl_per_contract(
    *, outcome_yes: int, taker_side: str, yes_price: float
) -> float:
    """
    Approximate taker PnL per YES-priced contract (ignores fees).
    YES taker: pays yes_price, receives 1 iff market resolves YES.
    NO taker: pays (1 - yes_price), receives 1 iff market resolves NO.
    """
    s = taker_side.lower()
    if s == "yes":
        return float(outcome_yes) - yes_price
    if s == "no":
        return float(1 - outcome_yes) - (1.0 - yes_price)
    return float("nan")


def log1p_seconds(seconds_before_close: float) -> float:
    return math.log1p(max(0.0, seconds_before_close))


def bucket_label(seconds_before_close: float, buckets: list[tuple[int, int]]) -> str:
    s = max(0.0, seconds_before_close)
    for lo, hi in buckets:
        if lo <= s < hi:
            return f"{lo}-{hi}s"
    if buckets:
        return f">={buckets[-1][1]}s"
    return "unknown"


@dataclass
class BucketAgg:
    trades: int = 0
    contracts: float = 0.0
    taker_wins: int = 0
    taker_win_contracts: float = 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="Kalshi BTC short-horizon trade timing dataset")
    p.add_argument(
        "--series",
        default="KXBTC15M",
        help="Kalshi series_ticker (default: KXBTC15M)",
    )
    p.add_argument(
        "--max-markets",
        type=int,
        default=0,
        help="Cap settled markets processed (0 = no cap)",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Seconds to sleep after each HTTP request (rate limiting)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Output directory",
    )
    args = p.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    series_safe = args.series.replace("/", "-")
    trades_path = out_dir / f"{series_safe.lower()}_trades_enriched.csv"
    summary_path = out_dir / f"{series_safe.lower()}_timing_summary.json"

    fieldnames = [
        "series_ticker",
        "market_ticker",
        "trade_id",
        "created_time",
        "close_time",
        "seconds_before_close",
        "log1p_seconds_before_close",
        "time_bucket",
        "taker_side",
        "taker_side_yes",
        "yes_price_dollars",
        "yes_price",
        "market_result",
        "outcome_yes",
        "count_fp",
        "taker_won",
        "taker_pnl_per_contract",
    ]

    bucket_aggs: dict[str, BucketAgg] = defaultdict(BucketAgg)
    total_trades = 0
    total_wins = 0
    market_tickers: set[str] = set()

    with trades_path.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fieldnames)
        w.writeheader()

        for m in iter_markets(args.series, max_markets=args.max_markets, sleep_s=args.sleep):
            ticker = m.get("ticker")
            result = (m.get("result") or "").lower()
            close_raw = m.get("close_time")
            oy = outcome_yes_int(result)
            if not ticker or not close_raw or oy is None:
                continue
            market_tickers.add(ticker)
            close_dt = _parse_ts(close_raw)

            for tr in iter_trades(ticker, args.sleep):
                created_raw = tr.get("created_time")
                if not created_raw:
                    continue
                trade_dt = _parse_ts(created_raw)
                if trade_dt >= close_dt:
                    continue
                sec_before = (close_dt - trade_dt).total_seconds()
                side = (tr.get("taker_side") or "").lower()
                if side not in {"yes", "no"}:
                    continue
                try:
                    yes_price = float(tr.get("yes_price_dollars") or "")
                except (TypeError, ValueError):
                    continue
                if not 0.0 < yes_price < 1.0:
                    continue
                won = taker_won(result, side)
                cnt = float(tr.get("count_fp") or 0)
                bl = bucket_label(sec_before, DEFAULT_BUCKETS)
                lp = log1p_seconds(sec_before)
                pnl = taker_pnl_per_contract(
                    outcome_yes=oy, taker_side=side, yes_price=yes_price
                )

                w.writerow(
                    {
                        "series_ticker": args.series,
                        "market_ticker": ticker,
                        "trade_id": tr.get("trade_id"),
                        "created_time": created_raw,
                        "close_time": close_raw,
                        "seconds_before_close": f"{sec_before:.3f}",
                        "log1p_seconds_before_close": f"{lp:.6f}",
                        "time_bucket": bl,
                        "taker_side": side,
                        "taker_side_yes": "1" if side == "yes" else "0",
                        "yes_price_dollars": tr.get("yes_price_dollars"),
                        "yes_price": f"{yes_price:.4f}",
                        "market_result": result,
                        "outcome_yes": str(oy),
                        "count_fp": tr.get("count_fp"),
                        "taker_won": str(won).lower(),
                        "taker_pnl_per_contract": f"{pnl:.6f}",
                    }
                )

                ba = bucket_aggs[bl]
                ba.trades += 1
                ba.contracts += cnt
                total_trades += 1
                if won:
                    ba.taker_wins += 1
                    ba.taker_win_contracts += cnt
                    total_wins += 1

    def bucket_sort_key(label: str) -> tuple[int, int]:
        if label.endswith("s") and "-" in label and not label.startswith(">="):
            a, b = label[:-1].split("-", 1)
            try:
                return (int(a), int(b))
            except ValueError:
                pass
        return (10_000, 0)

    by_bucket = []
    for label in sorted(bucket_aggs.keys(), key=bucket_sort_key):
        ba = bucket_aggs[label]
        wr = ba.taker_wins / ba.trades if ba.trades else 0.0
        by_bucket.append(
            {
                "time_bucket": label,
                "trade_count": ba.trades,
                "taker_win_count": ba.taker_wins,
                "taker_win_rate": round(wr, 4),
                "contracts": round(ba.contracts, 2),
                "taker_win_contracts": round(ba.taker_win_contracts, 2),
            }
        )

    summary = {
        "series_ticker": args.series,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "markets_cap": args.max_markets or None,
        "markets_distinct": len(market_tickers),
        "total_trades_written": total_trades,
        "overall_taker_win_rate": round(total_wins / total_trades, 4) if total_trades else None,
        "buckets_seconds_before_close": [f"{a}-{b}s" for a, b in DEFAULT_BUCKETS]
        + [f">={DEFAULT_BUCKETS[-1][1]}s"],
        "by_bucket": by_bucket,
        "caveats": [
            "taker_won means the aggressive side matched at settlement; it is not your PnL or a tradeable rule.",
            "Win-rate by bucket mixes all prices; high win rates near close often coincide with cheap lottery tickets.",
            "Past timing distributions are not a validated forward strategy — test out-of-sample with costs.",
            "yes_price / outcome_yes / taker_pnl_per_contract support ML; pnl ignores fees.",
            "Train with a group split on market_ticker so rows from the same market stay in one fold.",
        ],
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {trades_path} ({total_trades} trades)")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
