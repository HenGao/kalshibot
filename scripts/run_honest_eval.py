#!/usr/bin/env python3
"""
Honest evaluation workflow (public API; can take many minutes).

1. Run this script to refresh ``data/strategy_gate_report.json`` via
   ``simulate_recent_edge.py`` (hold-to-settlement, tape-based mids).

2. Optionally tighten the gate thresholds (defaults: min 50 trades, avg PnL/trade >= 0).

3. Start ``edge_bot`` with ``REQUIRE_STRATEGY_GATE=1`` or ``--require-strategy-gate`` so
   **new entries** are blocked until ``passed: true`` in the report. Stops still run.

4. For BTC-trend models, generate calibration bins with
   ``train_btc_trend_kalshi.py --calibration-report ...`` and inspect with
   ``python scripts/report_calibration.py <json>``.

5. Walk-forward style checks: ``python scripts/walk_forward_eval.py`` (see that script).

This is not financial advice.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sim = root / "scripts" / "simulate_recent_edge.py"
    p = argparse.ArgumentParser(description="Regenerate strategy_gate_report.json via API simulation")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--max-markets-per-series", type=int, default=250)
    p.add_argument(
        "--report",
        type=Path,
        default=Path("data/strategy_gate_report.json"),
        help="Path relative to repo root unless absolute",
    )
    p.add_argument(
        "--gate-min-avg-pnl-per-trade",
        type=float,
        default=0.0,
        help="Forwarded to simulate_recent_edge (default 0 = non-negative avg)",
    )
    p.add_argument(
        "--gate-min-trades",
        type=int,
        default=50,
        help="Minimum trades required for passed=True",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.01,
        help="Throttle between Kalshi API pages (simulate_recent_edge --sleep)",
    )
    args = p.parse_args()
    report = args.report if args.report.is_absolute() else root / args.report

    cmd = [
        sys.executable,
        str(sim),
        "--days",
        str(args.days),
        "--max-markets-per-series",
        str(args.max_markets_per_series),
        "--sleep",
        str(args.sleep),
        "--write-report",
        str(report),
        "--gate-min-avg-pnl-per-trade",
        str(args.gate_min_avg_pnl_per_trade),
        "--gate-min-trades",
        str(args.gate_min_trades),
    ]
    print("Running:", " ".join(cmd), flush=True)
    if os.environ.get("REQUIRE_STRATEGY_GATE", "").strip().lower() in ("1", "true", "yes"):
        print("(REQUIRE_STRATEGY_GATE is set; edge_bot will enforce this file once updated.)", flush=True)
    proc = subprocess.run(cmd, cwd=str(root), env=os.environ.copy())
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
