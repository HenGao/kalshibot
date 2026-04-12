#!/usr/bin/env python3
"""
Time-based walk-forward on KXBTC15M btc-trend backtest: split **UTC settlement dates**
in the pool into K contiguous folds; evaluate each fold’s markets (out-of-fold are not
used here — each fold is a **held-out time slice**).

Uses the same PnL logic as ``backtest_btc_trend_kxbtc15m.py`` with ``preloaded_markets``.
Supports fill stress (``--half-spread-mult``, ``--ask-slip``), fee args, optional
``--calibration-bins`` per fold, and the same entry filters as the backtest.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from datetime import date, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_bt_path = _ROOT / "scripts" / "backtest_btc_trend_kxbtc15m.py"
_bt_spec = importlib.util.spec_from_file_location("btkx", _bt_path)
assert _bt_spec and _bt_spec.loader
_btk = importlib.util.module_from_spec(_bt_spec)
sys.modules[_bt_spec.name] = _btk
_bt_spec.loader.exec_module(_btk)

_sre_path = _ROOT / "scripts" / "simulate_recent_edge.py"
_sre_spec = importlib.util.spec_from_file_location("simulate_recent_edge", _sre_path)
assert _sre_spec and _sre_spec.loader
_sre = importlib.util.module_from_spec(_sre_spec)
sys.modules[_sre_spec.name] = _sre
_sre_spec.loader.exec_module(_sre)
iter_settled_markets = _sre.iter_settled_markets
parse_ts = _sre.parse_ts


def _utc_date(close_raw: str) -> date:
    dt = parse_ts(str(close_raw))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).date()


def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward folds by settlement day")
    p.add_argument("--days", type=int, default=30, help="Pool: last N days settled")
    p.add_argument("--folds", type=int, default=3, help="Contiguous day-bucket folds")
    p.add_argument("--model", type=Path, default=Path("data/models/btc_trend_logistic.json"))
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--trade-pages", type=int, default=8)
    p.add_argument("--min-edge", type=float, default=0.04)
    p.add_argument("--max-spread", type=float, default=0.12)
    p.add_argument("--edge-spread-mult", type=float, default=0.30)
    p.add_argument("--half-spread", type=float, default=0.02)
    p.add_argument(
        "--target-sec",
        type=float,
        default=120.0,
        help="Seconds before close for tape decision row (same as backtest).",
    )
    p.add_argument(
        "--min-ev-surplus",
        type=str,
        default=None,
        help="EV minus threshold cushion ($/contract, 0=off). Env MIN_EV_SURPLUS_DOLLARS if unset (default 0.02).",
    )
    p.add_argument(
        "--min-direction-confidence",
        type=str,
        default=None,
        help="YES needs fair_yes>=this, NO needs fair_yes<=1-this (0=off). Env MIN_DIRECTION_CONFIDENCE if unset (default 0.62).",
    )
    p.add_argument(
        "--half-spread-mult",
        type=float,
        default=1.0,
        help="Multiply half-spread (pessimistic fills; >1 stresses wider effective spread).",
    )
    p.add_argument(
        "--ask-slip",
        type=float,
        default=0.0,
        help="Add to tape-derived asks ($/contract), same as backtest.",
    )
    p.add_argument("--fee-coefficient", type=float, default=0.07)
    p.add_argument("--fee-multiplier", type=float, default=1.0)
    p.add_argument(
        "--coinbase-pad-days",
        type=float,
        default=2.0,
        help="Extra Coinbase history before the pool window.",
    )
    p.add_argument(
        "--max-vol-15m",
        type=float,
        default=0.0,
        help="Skip rows with vol_15m above this (0=off).",
    )
    p.add_argument(
        "--calibration-bins",
        type=int,
        default=0,
        help="If >0, print fair_yes vs empirical YES bins per fold (post-vol rows).",
    )
    args = p.parse_args()

    if args.folds < 2:
        print("--folds must be >= 2", file=sys.stderr)
        sys.exit(2)
    if not args.model.is_file():
        print(f"Missing model {args.model}", file=sys.stderr)
        sys.exit(2)

    if args.min_ev_surplus is not None:
        min_ev_surplus = float(Decimal(args.min_ev_surplus))
    else:
        min_ev_surplus = float(Decimal(os.environ.get("MIN_EV_SURPLUS_DOLLARS", "0.02")))
    if min_ev_surplus < 0:
        print("--min-ev-surplus must be >= 0", file=sys.stderr)
        sys.exit(2)

    if args.min_direction_confidence is not None:
        min_direction_confidence = float(Decimal(args.min_direction_confidence))
    else:
        min_direction_confidence = float(Decimal(os.environ.get("MIN_DIRECTION_CONFIDENCE", "0.62")))
    if min_direction_confidence < 0 or min_direction_confidence >= 1:
        print("--min-direction-confidence must be in [0, 1)", file=sys.stderr)
        sys.exit(2)
    if min_direction_confidence > 0 and min_direction_confidence <= 0.5:
        print(
            "--min-direction-confidence must be 0 or > 0.5 (otherwise YES/NO regions overlap)",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.half_spread_mult <= 0:
        print("--half-spread-mult must be > 0", file=sys.stderr)
        sys.exit(2)
    if args.ask_slip < 0:
        print("--ask-slip must be >= 0", file=sys.stderr)
        sys.exit(2)
    if args.calibration_bins < 0 or args.calibration_bins > 50:
        print("--calibration-bins must be in [0, 50]", file=sys.stderr)
        sys.exit(2)

    public_base = "https://api.elections.kalshi.com/trade-api/v2"
    now = int(time.time())
    min_s = now - args.days * 86400

    by_day: dict[date, list[dict[str, Any]]] = {}
    for m in iter_settled_markets(
        "KXBTC15M",
        min_settled_ts=min_s,
        max_settled_ts=now,
        public_base=public_base,
        sleep_s=args.sleep,
    ):
        cr = m.get("close_time")
        if not cr:
            continue
        d = _utc_date(str(cr))
        by_day.setdefault(d, []).append(m)

    days_sorted = sorted(by_day.keys())
    if len(days_sorted) < args.folds:
        print(f"Need at least {args.folds} distinct settlement days, got {len(days_sorted)}", file=sys.stderr)
        sys.exit(2)

    n_d = len(days_sorted)
    k = args.folds
    sizes = [n_d // k + (1 if i < (n_d % k) else 0) for i in range(k)]
    fold_ranges: list[list[date]] = []
    off = 0
    for sz in sizes:
        if sz <= 0:
            continue
        fold_ranges.append(days_sorted[off : off + sz])
        off += sz

    approx_days_per_fold = max(1, n_d // k)
    print(
        f"Pool: {len(days_sorted)} UTC days, {args.folds} folds (~{approx_days_per_fold} days each), "
        f"model={args.model}  target_sec={args.target_sec}  "
        f"min_ev_surplus={min_ev_surplus}  min_dir_conf={min_direction_confidence}  "
        f"half_spread_mult={args.half_spread_mult}  ask_slip={args.ask_slip}  "
        f"calibration_bins={args.calibration_bins or 'off'}",
        flush=True,
    )

    base_cfg_kwargs = dict(
        days=args.days,
        target_sec=float(args.target_sec),
        half_spread=args.half_spread,
        half_spread_mult=float(args.half_spread_mult),
        min_edge=args.min_edge,
        max_spread=args.max_spread,
        edge_spread_mult=args.edge_spread_mult,
        fee_coefficient=float(args.fee_coefficient),
        fee_multiplier=float(args.fee_multiplier),
        model_path=args.model,
        sleep=args.sleep,
        max_markets=0,
        trade_pages=args.trade_pages,
        max_lookback_sec=3600.0,
        coinbase_pad_days=float(args.coinbase_pad_days),
        max_vol_15m=float(args.max_vol_15m),
        verbose=False,
        news_tilt=0.0,
        random_sample_days=0,
        random_seed=42,
        ask_slip_dollars=float(args.ask_slip),
        min_ev_surplus=min_ev_surplus,
        min_direction_confidence=min_direction_confidence,
        calibration_bins=int(args.calibration_bins),
    )

    for fi, dr in enumerate(fold_ranges):
        preload: list[dict[str, Any]] = []
        for d in dr:
            preload.extend(by_day[d])
        cfg = _btk.KxBtc15mBacktestConfig(**base_cfg_kwargs, preloaded_markets=preload)
        r = _btk.run_kxbtc15m_btc_trend_backtest(cfg)
        dr_s = f"{dr[0].isoformat()}..{dr[-1].isoformat()}"
        print(
            f"Fold {fi + 1}/{len(fold_ranges)}  days={dr_s}  markets={r['processed']}  "
            f"trades={r['n_trade']}  total_pnl={r['total_pnl']:.4f}",
            flush=True,
        )
        cal = r.get("calibration")
        if cal:
            print(f"  calibration (fold {fi + 1})", flush=True)
            for row in cal:
                n = int(row["n"])
                if n <= 0:
                    continue
                print(
                    f"    [{row['bin_lo']:.2f},{row['bin_hi']:.2f})  n={n}  "
                    f"avg_fair={float(row['avg_fair_yes']):.4f}  emp_YES%={100 * float(row['empirical_yes_rate']):.2f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
