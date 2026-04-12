#!/usr/bin/env python3
"""Pretty-print JSON from ``train_btc_trend_kalshi.py --calibration-report``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Print reliability bins from calibration JSON")
    p.add_argument("path", type=Path, help="JSON from --calibration-report")
    args = p.parse_args()
    if not args.path.is_file():
        print(f"Not found: {args.path}", file=sys.stderr)
        sys.exit(2)
    data = json.loads(args.path.read_text(encoding="utf-8"))
    bins = data.get("bins") or []
    print(f"n_bins={data.get('n_bins', len(bins))}")
    print(f"{'bin_lo':>8} {'bin_hi':>8} {'n':>6} {'mean_p':>10} {'mean_y':>10}")
    for b in bins:
        print(
            f"{b.get('bin_lo', 0):8.3f} {b.get('bin_hi', 0):8.3f} {b.get('n', 0):6d} "
            f"{(b.get('mean_p') if b.get('mean_p') is not None else float('nan')):10.4f} "
            f"{(b.get('mean_y') if b.get('mean_y') is not None else float('nan')):10.4f}"
        )


if __name__ == "__main__":
    main()
