#!/usr/bin/env python3
"""
Backtest the edge_bot decision rule on *held-out* settled markets (no overlap
with training markets when using the same GroupShuffleSplit as training).

Uses one synthetic decision per test market: row closest to --target-sec before
close. Executable prices are approximated from the tape (yes_price ± half-spread)
because historical order books are not in the CSV.

Also optional --live-snapshot: current open KXBTC15M book vs model (no PnL until
settlement).
"""

from __future__ import annotations

import argparse
import math
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# repo root on sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from edge_math import (
    expected_value_buy_no,
    expected_value_buy_yes,
    parse_orderbook,
    quadratic_taker_fee_total_usd,
)
from fair_model import FairYesModel, predict_fair_yes
from kalshi_client import public_get


def ensure_ml_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "outcome_yes" not in out.columns and "market_result" in out.columns:
        r = out["market_result"].astype(str).str.lower()
        out["outcome_yes"] = r.map({"yes": 1, "no": 0})
    if "yes_price" not in out.columns and "yes_price_dollars" in out.columns:
        out["yes_price"] = pd.to_numeric(out["yes_price_dollars"], errors="coerce")
    if "log1p_seconds_before_close" not in out.columns:
        sec = pd.to_numeric(out["seconds_before_close"], errors="coerce")
        out["log1p_seconds_before_close"] = np.log1p(sec.clip(lower=0))
    return out


def implied_exec_from_mid(yes_mid: float, half_spread: float) -> tuple[Decimal, Decimal]:
    """Toy bid/ask around trade-derived YES level."""
    hm = half_spread
    yb = max(0.01, min(0.99, yes_mid - hm))
    ya = max(0.01, min(0.99, yes_mid + hm))
    nb = max(0.01, min(0.99, 1.0 - yes_mid - hm))
    na = max(0.01, min(0.99, 1.0 - yes_mid + hm))
    return Decimal(str(ya)), Decimal(str(na))


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest edge rule on holdout markets")
    p.add_argument("--csv", type=Path, default=Path("data/kxbtc15m_trades_enriched.csv"))
    p.add_argument("--model", type=Path, default=Path("data/models/fair_yes_logistic.json"))
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-sec", type=float, default=120.0, help="Decision time before close")
    p.add_argument("--half-spread", type=float, default=0.02, help="Half-spread around tape mid")
    p.add_argument("--min-edge", type=float, default=0.02)
    p.add_argument("--fee-coefficient", type=float, default=0.07)
    p.add_argument("--fee-multiplier", type=float, default=1.0)
    p.add_argument(
        "--live-snapshot",
        action="store_true",
        help="Print model vs live KXBTC15M orderbook (first open market)",
    )
    args = p.parse_args()

    if args.live_snapshot:
        base = "https://api.elections.kalshi.com/trade-api/v2"
        q = "series_ticker=KXBTC15M&status=open&limit=1"
        mk = public_get(f"markets?{q}", base_url=base).get("markets") or []
        if not mk:
            print("No open KXBTC15M market for snapshot.")
            return
        ticker = mk[0]["ticker"]
        if not args.model.is_file():
            print("Missing model", args.model, file=sys.stderr)
            sys.exit(2)
        model = FairYesModel.load_path(args.model)
        ob = public_get(f"markets/{ticker}/orderbook", base_url=base)
        ex = parse_orderbook(ob)
        mdata = public_get(f"markets/{ticker}", base_url=base)
        m = mdata.get("market", mdata)
        close_raw = m.get("close_time")
        if not close_raw:
            print("No close_time", file=sys.stderr)
            sys.exit(2)
        if close_raw.endswith("Z"):
            close_raw = close_raw[:-1] + "+00:00"
        from datetime import datetime, timezone

        close_dt = datetime.fromisoformat(close_raw)
        now = datetime.now(timezone.utc)
        sec = max(0.0, (close_dt - now).total_seconds())
        feats = {
            "log1p_seconds_before_close": math.log1p(sec),
            "yes_price": float(ex.best_yes_ask),
        }
        fy = Decimal(str(predict_fair_yes(model, feats)))
        c = Decimal("1")
        fc = Decimal(str(args.fee_coefficient))
        fm = Decimal(str(args.fee_multiplier))
        fee_y = quadratic_taker_fee_total_usd(ex.best_yes_ask, c, coefficient=fc, fee_multiplier=fm) / c
        fee_n = quadratic_taker_fee_total_usd(ex.best_no_ask, c, coefficient=fc, fee_multiplier=fm) / c
        ev_y = expected_value_buy_yes(fy, ex.best_yes_ask, fee_y)
        ev_n = expected_value_buy_no(fy, ex.best_no_ask, fee_n)
        me = Decimal(str(args.min_edge))
        print(f"Live snapshot: {ticker}")
        print(f"  sec_to_close={sec:.1f} fair_yes={fy:.4f}")
        print(
            f"  yes bid/ask {ex.best_yes_bid}/{ex.best_yes_ask} "
            f"no bid/ask {ex.best_no_bid}/{ex.best_no_ask}"
        )
        print(f"  EV/ct YES={ev_y:.4f} NO={ev_n:.4f} min_edge={me}")
        if ev_y >= me and ev_y >= ev_n:
            print("  Rule: would IOC buy YES (if keys configured).")
        elif ev_n >= me and ev_n > ev_y:
            print("  Rule: would IOC buy NO (if keys configured).")
        else:
            print("  Rule: no trade (EV below min_edge or flat).")
        print("  Settlement PnL unknown until market closes.")
        return

    if not args.csv.is_file():
        print(f"Missing CSV {args.csv}", file=sys.stderr)
        sys.exit(2)
    if not args.model.is_file():
        print(f"Missing model {args.model}", file=sys.stderr)
        sys.exit(2)

    df = ensure_ml_columns(pd.read_csv(args.csv))
    df = df.dropna(subset=["outcome_yes", "yes_price", "log1p_seconds_before_close", "seconds_before_close"])
    df = df[(df["yes_price"] > 0) & (df["yes_price"] < 1)]
    df["outcome_yes"] = df["outcome_yes"].astype(int)
    groups = df["market_ticker"].astype(str).to_numpy()
    X = df[["log1p_seconds_before_close", "yes_price"]].astype(float).to_numpy()
    y = df["outcome_yes"].to_numpy()

    gss = GroupShuffleSplit(
        n_splits=1, test_size=args.test_fraction, random_state=args.seed
    )
    train_idx, test_idx = next(gss.split(X, y, groups))
    test_markets = set(df.iloc[test_idx]["market_ticker"].unique())
    train_markets = set(df.iloc[train_idx]["market_ticker"].unique())
    overlap = test_markets & train_markets
    if overlap:
        print("warning: unexpected train/test overlap", overlap, file=sys.stderr)

    model = FairYesModel.load_path(args.model)
    dftest = df[df["market_ticker"].isin(test_markets)].copy()
    sec_col = pd.to_numeric(dftest["seconds_before_close"], errors="coerce")

    half = args.half_spread
    me = Decimal(str(args.min_edge))
    fc = Decimal(str(args.fee_coefficient))
    fm = Decimal(str(args.fee_multiplier))
    c1 = Decimal("1")

    pnls: list[float] = []
    n_trade = 0
    n_skip = 0

    for mkt, g in dftest.groupby("market_ticker"):
        g2 = g.assign(_d=(sec_col.loc[g.index] - args.target_sec).abs())
        row = g2.sort_values("_d").iloc[0]
        ymid = float(row["yes_price"])
        ya, na = implied_exec_from_mid(ymid, half)
        feats = {
            "log1p_seconds_before_close": float(row["log1p_seconds_before_close"]),
            "yes_price": ymid,
        }
        fy = Decimal(str(predict_fair_yes(model, feats)))
        fee_y = quadratic_taker_fee_total_usd(ya, c1, coefficient=fc, fee_multiplier=fm) / c1
        fee_n = quadratic_taker_fee_total_usd(na, c1, coefficient=fc, fee_multiplier=fm) / c1
        ev_y = expected_value_buy_yes(fy, ya, fee_y)
        ev_n = expected_value_buy_no(fy, na, fee_n)
        oy = int(row["outcome_yes"])

        if ev_y >= me and ev_y >= ev_n:
            n_trade += 1
            pay = Decimal(oy) * Decimal("1") - ya - fee_y
            pnls.append(float(pay))
        elif ev_n >= me and ev_n > ev_y:
            n_trade += 1
            pay = Decimal(1 - oy) * Decimal("1") - na - fee_n
            pnls.append(float(pay))
        else:
            n_skip += 1

    total = sum(pnls)
    print("=== Holdout backtest (markets not used in training split) ===")
    print(f"Train markets: {len(train_markets)}  Test markets: {len(test_markets)}")
    print(f"Decision: one row per test market nearest {args.target_sec}s before close")
    print(f"Spread model: yes_mid=tape yes_price, half_spread={half}")
    print(f"Trades taken: {n_trade}  No trade: {n_skip}")
    if pnls:
        print(f"Total PnL (est. $ per 1-lot, after est. taker fee): {total:.4f}")
        print(f"Avg per trade: {total / len(pnls):.4f}  Min/Max: {min(pnls):.4f} / {max(pnls):.4f}")
    else:
        print("No trades met min_edge on holdout.")
    print(
        "\nCaveats: tape price is not full book; fees estimated; "
        "tiny test set is very noisy; past is not future."
    )


if __name__ == "__main__":
    main()
