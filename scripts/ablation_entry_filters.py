#!/usr/bin/env python3
"""
Run the KXBTC15M btc-trend backtest four times with different entry-filter combos
(same window) to see whether ``min_ev_surplus`` or ``min_direction_confidence``
helps or hurts total PnL.

Example:
  python scripts/ablation_entry_filters.py --days 14
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_bt_path = _ROOT / "scripts" / "backtest_btc_trend_kxbtc15m.py"
_bt_spec = importlib.util.spec_from_file_location("btkx_ablation", _bt_path)
assert _bt_spec and _bt_spec.loader
_bt = importlib.util.module_from_spec(_bt_spec)
sys.modules[_bt_spec.name] = _bt
_bt_spec.loader.exec_module(_bt)


def main() -> None:
    p = argparse.ArgumentParser(description="Ablation: min_ev_surplus x min_direction_confidence")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--model", type=Path, default=Path("data/models/btc_trend_logistic.json"))
    p.add_argument("--target-sec", type=float, default=120.0)
    p.add_argument("--calibration-bins", type=int, default=0, help="If >0, print bins after each run")
    args = p.parse_args()

    if not args.model.is_file():
        print(f"Missing model {args.model}", file=sys.stderr)
        sys.exit(2)
    if args.calibration_bins < 0 or args.calibration_bins > 50:
        print("--calibration-bins must be in [0, 50]", file=sys.stderr)
        sys.exit(2)

    base = dict(
        days=args.days,
        target_sec=args.target_sec,
        half_spread=0.02,
        half_spread_mult=1.0,
        min_edge=0.04,
        max_spread=0.12,
        edge_spread_mult=0.30,
        fee_coefficient=0.07,
        fee_multiplier=1.0,
        model_path=args.model,
        sleep=0.0,
        max_markets=0,
        trade_pages=8,
        max_lookback_sec=3600.0,
        coinbase_pad_days=2.0,
        max_vol_15m=0.0,
        verbose=False,
        news_tilt=0.0,
        random_sample_days=0,
        random_seed=42,
        ask_slip_dollars=0.0,
        calibration_bins=args.calibration_bins,
    )

    scenarios: list[tuple[str, float, float]] = [
        ("live_like (0.02 / 0.62)", 0.02, 0.62),
        ("no_dir_conf (0.02 / 0)", 0.02, 0.0),
        ("no_ev_surplus (0 / 0.62)", 0.0, 0.62),
        ("both_off (0 / 0)", 0.0, 0.0),
    ]

    print(f"Ablation | {args.days}d | model={args.model}\n")
    print(f"{'scenario':<28} {'trades':>7} {'total_pnl':>11} {'skip_dir':>9} {'skip_surp':>10}")
    for label, mes, mdc in scenarios:
        cfg = _bt.KxBtc15mBacktestConfig(
            **base,
            min_ev_surplus=mes,
            min_direction_confidence=mdc,
        )
        r = _bt.run_kxbtc15m_btc_trend_backtest(cfg)
        print(
            f"{label:<28} {r['n_trade']:7d} {r['total_pnl']:11.4f} "
            f"{r['n_skip_dir']:9d} {r['n_skip_surplus']:10d}"
        )
        cal = r.get("calibration")
        if cal:
            print(f"  --- calibration ({label}) ---")
            for row in cal:
                n = int(row["n"])
                if n <= 0:
                    continue
                print(
                    f"  [{row['bin_lo']:.2f},{row['bin_hi']:.2f})  n={n}  "
                    f"avg_fair={float(row['avg_fair_yes']):.4f}  emp_YES%={100 * float(row['empirical_yes_rate']):.2f}"
                )


if __name__ == "__main__":
    main()
