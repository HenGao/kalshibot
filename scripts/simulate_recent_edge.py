#!/usr/bin/env python3
"""
Simulate the edge_bot decision rule on *recent settled* Kalshi BTC markets.

Fetches public data for the last ``--days`` (default 30) for:
  - KXBTC15M (15-minute up/down)
  - KXBTC (hourly range / strike markets)

For each market, picks the public trade whose time is closest to ``--target-sec``
before ``close_time``, uses that trade's YES price as the mid, applies the same
half-spread executable approximation and fee/EV logic as ``backtest_edge.py``,
then scores PnL **as if** you held one contract to settlement (no intraday stop).

This does **not** replay live stop-losses or IOC partial fills; it answers
"would this rule + model have made money holding to resolution on this sample?"

Rate-limits lightly between HTTP calls. Use ``--max-markets-per-series`` for a
quick smoke test; 0 = no cap (can take a long time).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from edge_math import (
    expected_value_buy_no,
    expected_value_buy_yes,
    quadratic_taker_fee_total_usd,
)
from fair_model import FairYesModel, predict_fair_yes
from kalshi_client import public_get


def implied_exec_from_mid(yes_mid: float, half_spread: float) -> tuple[Decimal, Decimal]:
    hm = half_spread
    ymid = yes_mid
    yb = max(0.01, min(0.99, ymid - hm))
    ya = max(0.01, min(0.99, ymid + hm))
    nb = max(0.01, min(0.99, 1.0 - ymid - hm))
    na = max(0.01, min(0.99, 1.0 - ymid + hm))
    return Decimal(str(ya)), Decimal(str(na))


def parse_ts(iso: str) -> Any:
    from datetime import datetime

    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    return datetime.fromisoformat(iso)


def iter_settled_markets(
    series_ticker: str,
    *,
    min_settled_ts: int,
    max_settled_ts: int,
    public_base: str,
    sleep_s: float,
) -> Iterator[dict[str, Any]]:
    cursor: str | None = None
    while True:
        q: dict[str, str] = {
            "series_ticker": series_ticker,
            "status": "settled",
            "min_settled_ts": str(min_settled_ts),
            "max_settled_ts": str(max_settled_ts),
            "limit": "1000",
        }
        if cursor:
            q["cursor"] = cursor
        path = "markets?" + urlencode(q)
        data = public_get(path, base_url=public_base)
        markets = data.get("markets") or []
        for m in markets:
            yield m
        cursor = data.get("cursor") or ""
        if not cursor:
            break
        if sleep_s > 0:
            time.sleep(sleep_s)


def fetch_trades_for_ticker(ticker: str, *, public_base: str, sleep_s: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        q: dict[str, str] = {"ticker": ticker, "limit": "1000"}
        if cursor:
            q["cursor"] = cursor
        data = public_get("markets/trades?" + urlencode(q), base_url=public_base)
        for t in data.get("trades") or []:
            out.append(t)
        cursor = data.get("cursor") or ""
        if not cursor:
            break
        if sleep_s > 0:
            time.sleep(sleep_s)
    return out


def pick_decision_trade(
    trades: list[dict[str, Any]],
    close_dt: Any,
    *,
    target_sec: float,
    max_lookback_sec: float,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_d = 1e18
    for t in trades:
        ct = parse_ts(str(t["created_time"]))
        if ct >= close_dt:
            continue
        sec_before = (close_dt - ct).total_seconds()
        if sec_before < 0 or sec_before > max_lookback_sec:
            continue
        d = abs(sec_before - target_sec)
        if d < best_d:
            best_d = d
            best = t
    return best


def outcome_yes_from_market(m: dict[str, Any]) -> int | None:
    r = (m.get("result") or "").strip().lower()
    if r == "yes":
        return 1
    if r == "no":
        return 0
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Simulate edge rule on recent settled BTC markets")
    p.add_argument("--days", type=int, default=30, help="Lookback window (settled time)")
    p.add_argument(
        "--max-markets-per-series",
        type=int,
        default=0,
        help="Cap markets processed per series (0 = all returned by API)",
    )
    p.add_argument("--target-sec", type=float, default=120.0)
    p.add_argument("--half-spread", type=float, default=0.02)
    p.add_argument("--min-edge", type=float, default=0.02)
    p.add_argument("--fee-coefficient", type=float, default=0.07)
    p.add_argument("--fee-multiplier", type=float, default=1.0)
    p.add_argument("--model", type=Path, default=Path("data/models/fair_yes_logistic.json"))
    p.add_argument(
        "--model-hourly",
        type=Path,
        default=None,
        help="Model for KXBTC (non-15m); defaults to --model",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.02,
        help="Seconds between paginated trade/list requests",
    )
    p.add_argument(
        "--max-lookback-sec",
        type=float,
        default=3600.0,
        help="Ignore trades earlier than this many seconds before close",
    )
    p.add_argument(
        "--write-report",
        type=Path,
        default=None,
        help="Write machine-readable gate JSON (for edge_bot --require-strategy-gate)",
    )
    p.add_argument(
        "--gate-min-avg-pnl-per-trade",
        type=float,
        default=0.0,
        help="Report passed=True only if avg PnL/trade >= this (and min trades met)",
    )
    p.add_argument(
        "--gate-min-trades",
        type=int,
        default=50,
        help="Report passed=True only if at least this many trades in sample",
    )
    args = p.parse_args()

    public_base = "https://api.elections.kalshi.com/trade-api/v2"
    now = int(time.time())
    min_ts = now - int(args.days * 86400)

    if not args.model.is_file():
        print(f"Missing model {args.model}", file=sys.stderr)
        sys.exit(2)
    model_15m = FairYesModel.load_path(args.model)
    hourly_path = args.model_hourly if args.model_hourly is not None else args.model
    if not hourly_path.is_file():
        print(f"Missing hourly model {hourly_path}", file=sys.stderr)
        sys.exit(2)
    model_hr = FairYesModel.load_path(hourly_path)

    half = args.half_spread
    me = Decimal(str(args.min_edge))
    fc = Decimal(str(args.fee_coefficient))
    fm = Decimal(str(args.fee_multiplier))
    c1 = Decimal("1")

    series_list = [("KXBTC15M", model_15m), ("KXBTC", model_hr)]

    print("=== Recent settled-market simulation (hold to settlement) ===", flush=True)
    print(f"Window: last {args.days} days by settled time", flush=True)
    print(
        f"Series: KXBTC15M + KXBTC  |  target_sec~{args.target_sec}  |  half_spread={args.half_spread}",
        flush=True,
    )
    print(
        f"min_edge={args.min_edge}  fee_coeff={args.fee_coefficient}  fee_mult={args.fee_multiplier}",
        flush=True,
    )
    if args.max_markets_per_series:
        print(f"Cap: {args.max_markets_per_series} markets per series (not full universe)", flush=True)
    print("Fetching markets and trades...", flush=True)

    pnls: list[float] = []
    n_trade = 0
    n_skip_rule = 0
    n_skip_data = 0
    by_series: dict[str, list[float]] = {"KXBTC15M": [], "KXBTC": []}

    processed = 0
    for series_ticker, model in series_list:
        cap = args.max_markets_per_series
        seen = 0
        for m in iter_settled_markets(
            series_ticker,
            min_settled_ts=min_ts,
            max_settled_ts=now,
            public_base=public_base,
            sleep_s=args.sleep,
        ):
            if cap and seen >= cap:
                break
            seen += 1
            ticker = m.get("ticker")
            if not ticker:
                n_skip_data += 1
                continue
            oy = outcome_yes_from_market(m)
            if oy is None:
                n_skip_data += 1
                continue
            close_raw = m.get("close_time")
            if not close_raw:
                n_skip_data += 1
                continue
            close_dt = parse_ts(str(close_raw))

            if args.sleep > 0:
                time.sleep(args.sleep)
            trades = fetch_trades_for_ticker(str(ticker), public_base=public_base, sleep_s=args.sleep)
            row = pick_decision_trade(
                trades,
                close_dt,
                target_sec=args.target_sec,
                max_lookback_sec=args.max_lookback_sec,
            )
            if row is None:
                n_skip_data += 1
                continue

            ymid = float(row["yes_price_dollars"])
            if not (0 < ymid < 1):
                n_skip_data += 1
                continue

            sec_before = (close_dt - parse_ts(str(row["created_time"]))).total_seconds()
            feats = {
                "log1p_seconds_before_close": math.log1p(max(0.0, sec_before)),
                "yes_price": ymid,
            }
            fy = Decimal(str(predict_fair_yes(model, feats)))
            ya, na = implied_exec_from_mid(ymid, half)
            fee_y = quadratic_taker_fee_total_usd(ya, c1, coefficient=fc, fee_multiplier=fm) / c1
            fee_n = quadratic_taker_fee_total_usd(na, c1, coefficient=fc, fee_multiplier=fm) / c1
            ev_y = expected_value_buy_yes(fy, ya, fee_y)
            ev_n = expected_value_buy_no(fy, na, fee_n)

            if ev_y >= me and ev_y >= ev_n:
                n_trade += 1
                pay = Decimal(oy) * Decimal("1") - ya - fee_y
                x = float(pay)
                pnls.append(x)
                by_series[series_ticker].append(x)
            elif ev_n >= me and ev_n > ev_y:
                n_trade += 1
                pay = Decimal(1 - oy) * Decimal("1") - na - fee_n
                x = float(pay)
                pnls.append(x)
                by_series[series_ticker].append(x)
            else:
                n_skip_rule += 1

            processed += 1
            if processed % 200 == 0:
                print(f"  ... processed {processed} markets so far ...", flush=True)

    total = sum(pnls)
    print("--- Results ---", flush=True)
    print(f"Trades (model took signal): {n_trade}  No trade (rule): {n_skip_rule}  Skipped (no tape): {n_skip_data}")
    if pnls:
        print(f"Total PnL ($ per 1-lot, after est. fee at entry): {total:.4f}")
        print(f"Avg per trade: {total / len(pnls):.4f}  Min/Max: {min(pnls):.4f} / {max(pnls):.4f}")
        for s in ("KXBTC15M", "KXBTC"):
            xs = by_series[s]
            if xs:
                print(f"  {s}: n={len(xs)}  subtotal={sum(xs):.4f}  avg={sum(xs)/len(xs):.4f}")
    else:
        print("No trades met min_edge in sample.")
    print(
        "\nCaveats: tape mid != full book; no stop-loss; same model used for hourly unless "
        "--model-hourly set; past performance != future."
    )

    avg_pnl = (total / len(pnls)) if pnls else None
    passed = (
        n_trade >= args.gate_min_trades
        and avg_pnl is not None
        and avg_pnl >= args.gate_min_avg_pnl_per_trade
    )
    report: dict[str, object] = {
        "schema": "strategy_gate_report/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "n_trades": n_trade,
        "n_skip_rule": n_skip_rule,
        "n_skip_data": n_skip_data,
        "total_pnl": total,
        "avg_pnl_per_trade": avg_pnl,
        "by_series": {
            s: {"n": len(xs), "subtotal": sum(xs), "avg": (sum(xs) / len(xs) if xs else None)}
            for s, xs in by_series.items()
            if xs
        },
        "criteria": {
            "min_avg_pnl_per_trade": args.gate_min_avg_pnl_per_trade,
            "min_trades": args.gate_min_trades,
        },
        "args": {
            "days": args.days,
            "max_markets_per_series": args.max_markets_per_series,
            "target_sec": args.target_sec,
            "half_spread": args.half_spread,
            "min_edge": args.min_edge,
            "fee_coefficient": args.fee_coefficient,
            "fee_multiplier": args.fee_multiplier,
            "model": str(args.model),
            "model_hourly": str(args.model_hourly) if args.model_hourly else None,
        },
    }
    if args.write_report is not None:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote gate report: {args.write_report}  passed={passed}", flush=True)


if __name__ == "__main__":
    main()
