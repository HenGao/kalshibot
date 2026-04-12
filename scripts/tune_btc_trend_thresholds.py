#!/usr/bin/env python3
"""
Grid-search ``min_edge``, ``max_spread``, ``edge_spread_mult``, optional ``news_tilt``,
optional **min_ev_surplus**, **min_direction_confidence**, **target_sec** (decision
lead time), **half_spread_mult**, and **ask_slip** (fill stress), plus optional
**multiple random day samples** (seeds) on the KXBTC15M
btc-trend backtest. Preloads Coinbase candles and the model once.

Default grids use a single ``0`` for surplus and direction so behavior matches the
legacy tuner; pass e.g. ``--min-ev-surpluses 0,0.02`` and
``--min-direction-confidences 0,0.62`` for ablations or live-like search.

Default is **in-sample** on ``--days``. Use ``--random-sample-days`` + ``--random-seeds``
to stress one window; use ``walk_forward_eval.py`` for time-slice holdouts.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys
import time
from decimal import Decimal
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_bt_path = _ROOT / "scripts" / "backtest_btc_trend_kxbtc15m.py"
_bt_spec = importlib.util.spec_from_file_location("backtest_btc_trend_kxbtc15m", _bt_path)
assert _bt_spec and _bt_spec.loader
_bt = importlib.util.module_from_spec(_bt_spec)
sys.modules[_bt_spec.name] = _bt
_bt_spec.loader.exec_module(_bt)

from btc_trend_features import fetch_coinbase_5m_range  # noqa: E402
from btc_trend_predict import load_btc_trend_predictor  # noqa: E402


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _validate_direction_confidence(x: float) -> None:
    if x < 0 or x >= 1:
        print(f"min_direction_confidence must be in [0, 1), got {x}", file=sys.stderr)
        sys.exit(2)
    if x > 0 and x <= 0.5:
        print(
            "min_direction_confidence must be 0 or > 0.5 (otherwise YES/NO regions overlap)",
            file=sys.stderr,
        )
        sys.exit(2)


def main() -> None:
    p = argparse.ArgumentParser(description="Grid-search EV thresholds (+ optional news / random days)")
    p.add_argument("--model", type=Path, default=Path("data/models/btc_trend_logistic.json"))
    p.add_argument("--days", type=int, default=7)
    p.add_argument(
        "--target-sec",
        type=float,
        default=120.0,
        help="Seconds before close for tape row; used when --target-secs is omitted.",
    )
    p.add_argument("--half-spread", type=float, default=0.02)
    p.add_argument(
        "--half-spread-mult",
        type=float,
        default=1.0,
        help="Used when --half-spread-mults is omitted.",
    )
    p.add_argument(
        "--half-spread-mults",
        type=str,
        default=None,
        help="Comma-separated pessimistic half-spread multipliers (overrides --half-spread-mult). Example: 1,1.25,1.5",
    )
    p.add_argument("--min-edges", type=str, default="0.03,0.04,0.05,0.06")
    p.add_argument("--max-spreads", type=str, default="0.08,0.12,0.16,0")
    p.add_argument("--edge-spread-mults", type=str, default="0.2,0.3,0.4")
    p.add_argument("--news-tilts", type=str, default="0", help="Comma-separated BTC news tilt values (0=off)")
    p.add_argument(
        "--random-sample-days",
        type=int,
        default=0,
        help="If >0, sample this many random UTC days from pool per run (see backtest)",
    )
    p.add_argument(
        "--random-seeds",
        type=str,
        default="42",
        help="Comma seeds for --random-sample-days (e.g. 42,43,44 for bootstrap-style spread)",
    )
    p.add_argument(
        "--ask-slip",
        type=float,
        default=0.0,
        help="Used when --ask-slips is omitted.",
    )
    p.add_argument(
        "--ask-slips",
        type=str,
        default=None,
        help="Comma-separated extra $/contract on asks (slippage stress). Example: 0,0.01,0.02",
    )
    p.add_argument(
        "--min-ev-surpluses",
        type=str,
        default="0",
        help="Comma-separated EV surplus ($/contract, 0=off). Env MIN_EV_SURPLUS_DOLLARS if set to "
        "the literal 'env' (single grid point). Example ablation: 0,0.02,0.04",
    )
    p.add_argument(
        "--min-direction-confidences",
        type=str,
        default="0",
        help="Comma-separated min fair P(side) (0=off). Env MIN_DIRECTION_CONFIDENCE if value is "
        "the literal 'env'. Example: 0,0.58,0.62",
    )
    p.add_argument(
        "--target-secs",
        type=str,
        default=None,
        help="Comma-separated seconds-before-close (overrides --target-sec). Example: 60,120,180",
    )
    p.add_argument("--fee-coefficient", type=float, default=0.07)
    p.add_argument("--fee-multiplier", type=float, default=1.0)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--max-markets", type=int, default=0)
    p.add_argument("--trade-pages", type=int, default=8)
    p.add_argument("--max-lookback-sec", type=float, default=3600.0)
    p.add_argument("--coinbase-pad-days", type=float, default=2.0)
    p.add_argument("--max-vol-15m", type=float, default=0.0)
    args = p.parse_args()

    if not args.model.is_file():
        print(f"Missing model {args.model}", file=sys.stderr)
        sys.exit(2)

    min_edges = _parse_float_list(args.min_edges)
    max_spreads = _parse_float_list(args.max_spreads)
    edge_mults = _parse_float_list(args.edge_spread_mults)
    news_tilts = _parse_float_list(args.news_tilts)
    seeds = _parse_int_list(args.random_seeds)
    target_secs = (
        _parse_float_list(args.target_secs) if args.target_secs else [float(args.target_sec)]
    )
    raw_mes = [x.strip() for x in args.min_ev_surpluses.split(",") if x.strip()]
    raw_mdc = [x.strip() for x in args.min_direction_confidences.split(",") if x.strip()]
    min_ev_surpluses: list[float] = []
    for token in raw_mes:
        if token.lower() == "env":
            min_ev_surpluses.append(float(Decimal(os.environ.get("MIN_EV_SURPLUS_DOLLARS", "0.02"))))
        else:
            v = float(Decimal(token))
            if v < 0:
                print("min-ev-surpluses values must be >= 0", file=sys.stderr)
                sys.exit(2)
            min_ev_surpluses.append(v)
    min_direction_confidences: list[float] = []
    for token in raw_mdc:
        if token.lower() == "env":
            min_direction_confidences.append(
                float(Decimal(os.environ.get("MIN_DIRECTION_CONFIDENCE", "0.62")))
            )
        else:
            v = float(Decimal(token))
            _validate_direction_confidence(v)
            min_direction_confidences.append(v)
    for ts in target_secs:
        if ts <= 0:
            print("--target-secs values must be > 0", file=sys.stderr)
            sys.exit(2)

    half_spread_mults = (
        _parse_float_list(args.half_spread_mults)
        if args.half_spread_mults
        else [float(args.half_spread_mult)]
    )
    for h in half_spread_mults:
        if h <= 0:
            print("--half-spread-mults values must be > 0", file=sys.stderr)
            sys.exit(2)
    ask_slips = _parse_float_list(args.ask_slips) if args.ask_slips else [float(args.ask_slip)]
    for s in ask_slips:
        if s < 0:
            print("--ask-slips values must be >= 0", file=sys.stderr)
            sys.exit(2)

    now = int(time.time())
    min_settled = now - int(args.days * 86400)
    chart_start = min_settled - int(args.coinbase_pad_days * 86400) - 3600

    print("Loading model + Coinbase once ...", flush=True)
    mdl = load_btc_trend_predictor(args.model)
    candles = fetch_coinbase_5m_range(chart_start, now + 60, sleep_s=args.sleep)
    starts = _bt.candle_starts(candles)
    print(f"  candles={len(candles)}", flush=True)

    results: list[tuple[float, float, float, float, float, int, float, int, float, float, float, float, float]] = []

    for seed in seeds:
        for nt in news_tilts:
            for ts in target_secs:
                for me in min_edges:
                    for ms in max_spreads:
                        for em in edge_mults:
                            for mes in min_ev_surpluses:
                                for mdc in min_direction_confidences:
                                    for hsm in half_spread_mults:
                                        for slip in ask_slips:
                                            cfg = _bt.KxBtc15mBacktestConfig(
                                                days=args.days,
                                                target_sec=ts,
                                                half_spread=args.half_spread,
                                                half_spread_mult=hsm,
                                                min_edge=me,
                                                max_spread=ms,
                                                edge_spread_mult=em,
                                                fee_coefficient=args.fee_coefficient,
                                                fee_multiplier=args.fee_multiplier,
                                                model_path=args.model,
                                                sleep=args.sleep,
                                                max_markets=args.max_markets,
                                                trade_pages=args.trade_pages,
                                                max_lookback_sec=args.max_lookback_sec,
                                                coinbase_pad_days=args.coinbase_pad_days,
                                                max_vol_15m=args.max_vol_15m,
                                                verbose=False,
                                                preloaded_candles=candles,
                                                preloaded_starts=starts,
                                                preloaded_model=mdl,
                                                news_tilt=float(nt),
                                                random_sample_days=args.random_sample_days,
                                                random_seed=int(seed),
                                                ask_slip_dollars=slip,
                                                min_ev_surplus=mes,
                                                min_direction_confidence=mdc,
                                            )
                                            r = _bt.run_kxbtc15m_btc_trend_backtest(cfg)
                                            pnls = r["pnls"]
                                            avg = r["total_pnl"] / len(pnls) if pnls else 0.0
                                            results.append(
                                                (
                                                    r["total_pnl"],
                                                    avg,
                                                    me,
                                                    ms,
                                                    em,
                                                    r["n_trade"],
                                                    float(nt),
                                                    int(seed),
                                                    mes,
                                                    mdc,
                                                    ts,
                                                    hsm,
                                                    slip,
                                                )
                                            )

    results.sort(key=lambda x: x[0], reverse=True)
    print("\n=== Top grid points by total PnL ($/contract, after est. fee) ===\n")
    hdr = (
        f"{'total_pnl':>12} {'avg':>10} {'trades':>8}  min_e  max_sp  edge_m  "
        f"ev_s  dir_c  t_sec  hsm  slip  news  seed"
    )
    print(hdr)
    for total, avg, me, ms, em, ntr, news, seed, mes, mdc, ts, hsm, slip in results[:30]:
        print(
            f"{total:12.4f} {avg:10.4f} {ntr:8d}  {me:.3f} {ms:.3f} {em:.2f}  "
            f"{mes:.3f} {mdc:.2f}  {ts:5.0f}  {hsm:.2f} {slip:.3f}  {news:.3f}  {seed}"
        )

    if len(seeds) > 1 and args.random_sample_days > 0:
        by_key: dict[tuple[float, float, float, float, float, float, float, float], list[float]] = {}
        for total, _avg, me, ms, em, _ntr, _news, _sd, mes, mdc, ts, hsm, slip in results:
            by_key.setdefault((me, ms, em, mes, mdc, ts, hsm, slip), []).append(total)
        print("\n=== Mean total_pnl over seeds (random day draws; same thresholds) ===\n")
        agg: list[tuple[float, float, float, float, float, float, float, float, float]] = []
        for (me, ms, em, mes, mdc, ts, hsm, slip), totals in by_key.items():
            agg.append((statistics.mean(totals), me, ms, em, mes, mdc, ts, hsm, slip))
        agg.sort(key=lambda x: x[0], reverse=True)
        for mean_pnl, me, ms, em, mes, mdc, ts, hsm, slip in agg[:15]:
            print(
                f"mean_pnl={mean_pnl:10.4f}  (n_seeds={len(seeds)})  min_e={me:.3f} max_sp={ms:.3f} "
                f"edge_m={em:.2f}  ev_s={mes:.3f} dir_c={mdc:.2f} t_sec={ts:.0f}  "
                f"hsm={hsm:.2f} slip={slip:.3f}"
            )

    print(
        "\nNote: in-sample unless you hold out time (walk_forward_eval.py) or use fresh ``--days``.",
        flush=True,
    )


if __name__ == "__main__":
    main()
