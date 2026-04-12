#!/usr/bin/env python3
"""
Compare Kalshi KXBTC15M ``result`` (yes/no) to a **Coinbase 5m close near close_time**
vs ``floor_strike`` (greater markets): proxy_yes = (close >= strike).

Reports agreement rate — **not** Kalshi’s legal settlement. Use to gauge Coinbase gap.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kalshi_settlement_proxy import coinbase_close_usd_near, sleep_s_paced  # noqa: E402

_sre_path = _ROOT / "scripts" / "simulate_recent_edge.py"
_sre_spec = importlib.util.spec_from_file_location("simulate_recent_edge", _sre_path)
assert _sre_spec and _sre_spec.loader
_sre = importlib.util.module_from_spec(_sre_spec)
_sre_spec.loader.exec_module(_sre)
iter_settled_markets = _sre.iter_settled_markets
outcome_yes_from_market = _sre.outcome_yes_from_market
parse_ts = _sre.parse_ts


def main() -> None:
    p = argparse.ArgumentParser(description="Kalshi result vs Coinbase-strike proxy")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--max-markets", type=int, default=40)
    p.add_argument("--sleep", type=float, default=0.15)
    args = p.parse_args()

    public_base = "https://api.elections.kalshi.com/trade-api/v2"
    now = int(time.time())
    min_s = now - args.days * 86400
    last = [0.0]
    n = 0
    agree = 0
    skipped = 0

    for m in iter_settled_markets(
        "KXBTC15M",
        min_settled_ts=min_s,
        max_settled_ts=now,
        public_base=public_base,
        sleep_s=args.sleep,
    ):
        if n >= args.max_markets:
            break
        oy = outcome_yes_from_market(m)
        if oy is None:
            skipped += 1
            continue
        st = str(m.get("strike_type") or "")
        floor = m.get("floor_strike")
        close_raw = m.get("close_time")
        if st not in ("greater", "greater_or_equal") or floor is None or not close_raw:
            skipped += 1
            continue
        close_dt = parse_ts(str(close_raw))
        ts = int(close_dt.timestamp())
        sleep_s_paced(last, args.sleep)
        px = coinbase_close_usd_near(ts)
        if px is None:
            skipped += 1
            continue
        proxy_yes = int(px >= float(floor))
        if proxy_yes == int(oy):
            agree += 1
        n += 1

    if n == 0:
        print("No comparable markets.", file=sys.stderr)
        sys.exit(2)
    print(f"Compared: {n}  agreement: {agree}/{n} ({100.0 * agree / n:.1f}%)  skipped: {skipped}")
    print("Coinbase 5m close near Kalshi close_time is not Kalshi’s settlement index.")


if __name__ == "__main__":
    main()
