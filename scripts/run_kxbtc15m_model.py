#!/usr/bin/env python3
"""
Run the trained fair P(YES) model against the **live** open KXBTC15M contract.

Uses real Kalshi quotes (public API). By default runs in **--dry-run** mode
(no orders). Use **--paper** for virtual cash, or **--trade** for real IOC
orders (requires KALSHI_ACCESS_KEY_ID and KALSHI_PRIVATE_KEY_PATH).

Example:
  python scripts/run_kxbtc15m_model.py --once
  python scripts/run_kxbtc15m_model.py --trade --interval 3

Extra args are forwarded to edge_bot.py (e.g. --min-edge 0.03).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDGE = ROOT / "edge_bot.py"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Model-driven bot on live open KXBTC15M (wrapper around edge_bot.py)"
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--dry-run",
        action="store_true",
        help="Log only, no orders (default)",
    )
    g.add_argument(
        "--paper",
        action="store_true",
        help="Paper trade: real quotes, virtual wallet",
    )
    g.add_argument(
        "--trade",
        action="store_true",
        help="Real orders: need API keys in environment",
    )
    p.add_argument("--once", action="store_true", help="Single poll cycle then exit")
    p.add_argument("--interval", type=float, default=5.0, help="Seconds between polls")
    args, rest = p.parse_known_args()

    if not EDGE.is_file():
        print(f"Missing {EDGE}", file=sys.stderr)
        sys.exit(2)

    cmd: list[str] = [
        sys.executable,
        str(EDGE),
        "--auto-markets",
        "15m",
        "--interval",
        str(args.interval),
    ]
    if args.once:
        cmd.append("--once")

    if args.trade:
        pass
    elif args.paper:
        cmd.append("--paper")
    else:
        cmd.append("--dry-run")

    cmd.extend(rest)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
