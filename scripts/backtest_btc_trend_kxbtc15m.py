#!/usr/bin/env python3
"""
Backtest --btc-trend-model on settled KXBTC15M markets (hold winning side to settlement).

Uses public Kalshi trades (tape mid ~ target_sec before close) like simulate_recent_edge.py,
and fair P(YES) from Coinbase 5m + v2 feature set (vol, path, book placeholders).

Optional ``--random-sample-days``: build a pool from the last ``--days``, then randomly
pick that many distinct UTC settlement dates and backtest only those markets (shuffled),
so results are not tied to one contiguous window (use ``--random-seed`` to reproduce).

PnL: $/contract after paying executable ask + est. quadratic taker fee at entry (no stop).
"""

from __future__ import annotations

import argparse
import bisect
import importlib.util
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import requests
from urllib.parse import urlencode

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from btc_news_sentiment import apply_btc_news_tilt_to_fair_yes, sentiment_for_decision_utc  # noqa: E402
from btc_trend_features import (  # noqa: E402
    fetch_coinbase_5m_range,
    features_from_klines_window,
)
from btc_trend_predict import load_btc_trend_predictor, predict_btc_trend_yes  # noqa: E402
from edge_math import (  # noqa: E402
    expected_value_buy_no,
    expected_value_buy_yes,
    quadratic_taker_fee_total_usd,
)

_sre_path = _ROOT / "scripts" / "simulate_recent_edge.py"
_sre_spec = importlib.util.spec_from_file_location("simulate_recent_edge", _sre_path)
assert _sre_spec and _sre_spec.loader
_sre = importlib.util.module_from_spec(_sre_spec)
_sre_spec.loader.exec_module(_sre)
implied_exec_from_mid = _sre.implied_exec_from_mid
iter_settled_markets = _sre.iter_settled_markets
outcome_yes_from_market = _sre.outcome_yes_from_market
parse_ts = _sre.parse_ts
pick_decision_trade = _sre.pick_decision_trade


def fetch_trades_for_ticker_limited(
    ticker: str,
    *,
    public_base: str,
    sleep_s: float,
    max_pages: int,
) -> list[dict[str, Any]]:
    """Paginate Kalshi trades (newest first); cap pages so deep tape does not hang."""
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    for _ in range(max_pages):
        q: dict[str, str] = {"ticker": ticker, "limit": "1000"}
        if cursor:
            q["cursor"] = cursor
        r = requests.get(
            f"{public_base}/markets/trades?{urlencode(q)}",
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        for t in data.get("trades") or []:
            out.append(t)
        cursor = (data.get("cursor") or "").strip() or None
        if not cursor:
            break
        if sleep_s > 0:
            time.sleep(sleep_s)
    return out


def parse_iso_to_unix(iso: str) -> int:
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    from datetime import datetime, timezone

    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def candle_starts(candles: list[list[Any]]) -> list[int]:
    return [int(c[0]) for c in candles]


def last_closed_bar_index(starts: list[int], t_dec: int) -> int:
    return bisect.bisect_right(starts, t_dec - 300) - 1


def btc_target_at_open(candles: list[list[Any]], starts: list[int], open_unix: int) -> float:
    j = bisect.bisect_right(starts, open_unix) - 1
    if j < 0:
        raise ValueError("no candle for open")
    return float(candles[j][3])


def _settlement_date_utc(close_dt: Any) -> date:
    if close_dt.tzinfo is None:
        close_dt = close_dt.replace(tzinfo=timezone.utc)
    return close_dt.astimezone(timezone.utc).date()


def _markets_for_random_days(
    *,
    series_ticker: str,
    min_settled_ts: int,
    max_settled_ts: int,
    public_base: str,
    sleep_s: float,
    pool_days: int,
    random_sample_days: int,
    random_seed: int,
    max_markets: int,
    verbose: bool,
) -> tuple[list[dict[str, Any]], list[date]]:
    """
    Load all settled markets in the pool window, pick ``random_sample_days`` distinct
    UTC settlement dates at random, return those markets (shuffled) with optional cap.
    """
    if random_sample_days <= 0:
        raise ValueError("random_sample_days must be > 0")
    by_day: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for m in iter_settled_markets(
        series_ticker,
        min_settled_ts=min_settled_ts,
        max_settled_ts=max_settled_ts,
        public_base=public_base,
        sleep_s=sleep_s,
    ):
        close_raw = m.get("close_time")
        if not close_raw:
            continue
        try:
            close_dt = parse_ts(str(close_raw))
        except (TypeError, ValueError):
            continue
        by_day[_settlement_date_utc(close_dt)].append(m)

    days_sorted = sorted(by_day.keys())
    if not days_sorted:
        return [], []

    rng = random.Random(random_seed)
    k = min(random_sample_days, len(days_sorted))
    picked = sorted(rng.sample(days_sorted, k))

    if verbose:
        print(
            f"  Random day sample: {k} of {len(days_sorted)} UTC days in {pool_days}d pool "
            f"(seed={random_seed})",
            flush=True,
        )
        print(f"    Days: {', '.join(d.isoformat() for d in picked)}", flush=True)

    out: list[dict[str, Any]] = []
    for d in picked:
        out.extend(by_day[d])
    rng.shuffle(out)

    if max_markets > 0 and len(out) > max_markets:
        out = out[:max_markets]
        if verbose:
            print(f"    Capped to --max-markets={max_markets}", flush=True)

    return out, picked


@dataclass
class KxBtc15mBacktestConfig:
    days: int
    target_sec: float
    half_spread: float
    half_spread_mult: float
    min_edge: float
    max_spread: float
    edge_spread_mult: float
    fee_coefficient: float
    fee_multiplier: float
    model_path: Path
    sleep: float
    max_markets: int
    trade_pages: int
    max_lookback_sec: float
    coinbase_pad_days: float
    max_vol_15m: float
    verbose: bool = True
    # When set (e.g. threshold tuning), skip refetch / reload for each run.
    preloaded_candles: list[list[Any]] | None = None
    preloaded_starts: list[int] | None = None
    preloaded_model: Any | None = None
    news_tilt: float = 0.0
    # 0 = use contiguous last ``days`` (API order). >0 = sample that many random UTC settlement days from the pool.
    random_sample_days: int = 0
    random_seed: int = 42
    ask_slip_dollars: float = 0.0
    min_ev_surplus: float = 0.02
    min_direction_confidence: float = 0.62
    preloaded_markets: list[dict[str, Any]] | None = None
    # If >0, collect fair_yes vs settlement YES rate in that many bins (same rows as trading path after vol filter).
    calibration_bins: int = 0


def run_kxbtc15m_btc_trend_backtest(cfg: KxBtc15mBacktestConfig) -> dict[str, Any]:
    model = (
        cfg.preloaded_model
        if cfg.preloaded_model is not None
        else load_btc_trend_predictor(cfg.model_path)
    )
    public_base = "https://api.elections.kalshi.com/trade-api/v2"
    now = int(time.time())
    min_settled = now - int(cfg.days * 86400)
    chart_start = min_settled - int(cfg.coinbase_pad_days * 86400) - 3600

    if cfg.preloaded_candles is not None:
        candles = cfg.preloaded_candles
        starts = cfg.preloaded_starts or candle_starts(candles)
    else:
        candles = fetch_coinbase_5m_range(chart_start, now + 60, sleep_s=cfg.sleep)
        starts = candle_starts(candles)
    if cfg.verbose:
        print(f"  {len(candles)} candles loaded.", flush=True)

    half = cfg.half_spread * cfg.half_spread_mult
    me = Decimal(str(cfg.min_edge))
    max_spread = Decimal(str(cfg.max_spread))
    edge_mult = Decimal(str(cfg.edge_spread_mult))
    fc = Decimal(str(cfg.fee_coefficient))
    fm = Decimal(str(cfg.fee_multiplier))
    c1 = Decimal("1")

    pnls: list[float] = []
    trade_close_unix: list[int] = []
    n_trade = 0
    n_skip_rule = 0
    n_skip_wide = 0
    n_skip_edge = 0
    n_skip_data = 0
    n_skip_vol = 0
    n_skip_surplus = 0
    n_skip_dir = 0
    processed = 0
    n_seen = 0
    random_days_picked: list[date] = []
    calib_n = cfg.calibration_bins
    calib_count = [0] * calib_n if calib_n > 0 else []
    calib_sum_yes = [0] * calib_n if calib_n > 0 else []
    calib_sum_fy = [0.0] * calib_n if calib_n > 0 else []

    if cfg.preloaded_markets is not None:
        market_iter = cfg.preloaded_markets
        cap_m = 0
        random_days_picked = []
    elif cfg.random_sample_days > 0:
        market_list, random_days_picked = _markets_for_random_days(
            series_ticker="KXBTC15M",
            min_settled_ts=min_settled,
            max_settled_ts=now,
            public_base=public_base,
            sleep_s=cfg.sleep,
            pool_days=cfg.days,
            random_sample_days=cfg.random_sample_days,
            random_seed=cfg.random_seed,
            max_markets=cfg.max_markets,
            verbose=cfg.verbose,
        )
        market_iter: Any = market_list
        cap_m = 0
    else:
        market_iter = iter_settled_markets(
            "KXBTC15M",
            min_settled_ts=min_settled,
            max_settled_ts=now,
            public_base=public_base,
            sleep_s=cfg.sleep,
        )
        cap_m = cfg.max_markets

    for m in market_iter:
        if cap_m and n_seen >= cap_m:
            break
        n_seen += 1
        ticker = m.get("ticker")
        if not ticker:
            n_skip_data += 1
            continue
        oy = outcome_yes_from_market(m)
        if oy is None:
            n_skip_data += 1
            continue
        close_raw = m.get("close_time")
        open_raw = m.get("open_time")
        if not close_raw or not open_raw:
            n_skip_data += 1
            continue
        close_dt = parse_ts(str(close_raw))
        open_unix = parse_iso_to_unix(str(open_raw))

        if cfg.sleep > 0:
            time.sleep(cfg.sleep)
        trades = fetch_trades_for_ticker_limited(
            str(ticker),
            public_base=public_base,
            sleep_s=cfg.sleep,
            max_pages=max(1, cfg.trade_pages),
        )
        row = pick_decision_trade(
            trades,
            close_dt,
            target_sec=cfg.target_sec,
            max_lookback_sec=cfg.max_lookback_sec,
        )
        if row is None:
            n_skip_data += 1
            continue

        ymid = float(row["yes_price_dollars"])
        if not (0 < ymid < 1):
            n_skip_data += 1
            continue

        decision_dt = parse_ts(str(row["created_time"]))
        t_dec = int(decision_dt.timestamp())
        sec_before = (close_dt - decision_dt).total_seconds()

        try:
            tgt = btc_target_at_open(candles, starts, open_unix)
        except ValueError:
            n_skip_data += 1
            continue

        j = last_closed_bar_index(starts, t_dec)
        if j < 72:
            n_skip_data += 1
            continue
        lo = max(0, j - 299)
        klines_win = candles[lo : j + 1]
        if len(klines_win) < 80:
            n_skip_data += 1
            continue

        try:
            feats = features_from_klines_window(
                klines_win,
                seconds_before_close=float(max(0.0, sec_before)),
                target_usd=float(tgt),
                decision_ts_utc=t_dec,
            )
        except ValueError:
            n_skip_data += 1
            continue

        if cfg.max_vol_15m > 0.0 and feats.get("vol_15m", 0.0) > cfg.max_vol_15m:
            n_skip_vol += 1
            continue

        fy = Decimal(str(predict_btc_trend_yes(model, feats)))
        if cfg.news_tilt > 0.0:
            sc, _ns = sentiment_for_decision_utc(decision_dt)
            fy = apply_btc_news_tilt_to_fair_yes(fy, sc, Decimal(str(cfg.news_tilt)))
        if calib_n > 0:
            fv = float(fy)
            bi = min(calib_n - 1, max(0, int(fv * calib_n)))
            calib_count[bi] += 1
            calib_sum_yes[bi] += int(oy)
            calib_sum_fy[bi] += fv
        ya, na = implied_exec_from_mid(ymid, half)
        slip = Decimal(str(cfg.ask_slip_dollars))
        if slip > 0:
            ya = min(Decimal("0.99"), ya + slip)
            na = min(Decimal("0.99"), na + slip)
        hm = Decimal(str(half))
        ym_d = Decimal(str(ymid))
        yb = max(Decimal("0.01"), min(Decimal("0.99"), ym_d - hm))
        bn = max(Decimal("0.01"), min(Decimal("0.99"), Decimal("1") - ym_d - hm))
        yes_spread = ya - yb
        no_spread = na - bn
        thr_yes = me + edge_mult * yes_spread
        thr_no = me + edge_mult * no_spread
        yes_ok = max_spread <= 0 or yes_spread <= max_spread
        no_ok = max_spread <= 0 or no_spread <= max_spread

        fee_y = quadratic_taker_fee_total_usd(ya, c1, coefficient=fc, fee_multiplier=fm) / c1
        fee_n = quadratic_taker_fee_total_usd(na, c1, coefficient=fc, fee_multiplier=fm) / c1
        ev_y = expected_value_buy_yes(fy, ya, fee_y)
        ev_n = expected_value_buy_no(fy, na, fee_n)

        surplus_need = Decimal(str(cfg.min_ev_surplus))
        dc = Decimal(str(cfg.min_direction_confidence))
        c_no = Decimal("1") - dc
        yes_dir_ok = dc <= 0 or fy >= dc
        no_dir_ok = dc <= 0 or fy <= c_no

        if yes_ok and ev_y >= ev_n and ev_y >= thr_yes:
            if surplus_need > 0 and ev_y - thr_yes < surplus_need:
                n_skip_surplus += 1
            elif not yes_dir_ok:
                n_skip_dir += 1
            else:
                n_trade += 1
                pay = Decimal(oy) * Decimal("1") - ya - fee_y
                pnls.append(float(pay))
                trade_close_unix.append(int(close_dt.timestamp()))
        elif no_ok and ev_n > ev_y and ev_n >= thr_no:
            if surplus_need > 0 and ev_n - thr_no < surplus_need:
                n_skip_surplus += 1
            elif not no_dir_ok:
                n_skip_dir += 1
            else:
                n_trade += 1
                pay = Decimal(1 - oy) * Decimal("1") - na - fee_n
                pnls.append(float(pay))
                trade_close_unix.append(int(close_dt.timestamp()))
        elif max_spread > 0 and (
            (ev_y >= ev_n and not yes_ok) or (ev_n > ev_y and not no_ok)
        ):
            n_skip_wide += 1
        elif ev_y < thr_yes and ev_n < thr_no:
            n_skip_edge += 1
        else:
            n_skip_rule += 1

        processed += 1
        if cfg.verbose and processed % 50 == 0:
            print(f"  ... markets processed {processed} ...", flush=True)

    total = float(sum(pnls))
    calibration: list[dict[str, Any]] | None = None
    if calib_n > 0:
        calibration = []
        for bi in range(calib_n):
            cnt = calib_count[bi]
            if cnt <= 0:
                calibration.append(
                    {
                        "bin_lo": bi / calib_n,
                        "bin_hi": (bi + 1) / calib_n,
                        "n": 0,
                        "avg_fair_yes": float("nan"),
                        "empirical_yes_rate": float("nan"),
                    }
                )
                continue
            calibration.append(
                {
                    "bin_lo": bi / calib_n,
                    "bin_hi": (bi + 1) / calib_n,
                    "n": cnt,
                    "avg_fair_yes": calib_sum_fy[bi] / cnt,
                    "empirical_yes_rate": calib_sum_yes[bi] / cnt,
                }
            )
    return {
        "total_pnl": total,
        "n_trade": n_trade,
        "pnls": pnls,
        "n_skip_wide": n_skip_wide,
        "n_skip_edge": n_skip_edge,
        "n_skip_rule": n_skip_rule,
        "n_skip_data": n_skip_data,
        "n_skip_vol": n_skip_vol,
        "n_skip_surplus": n_skip_surplus,
        "n_skip_dir": n_skip_dir,
        "processed": processed,
        "random_days_sampled": [d.isoformat() for d in random_days_picked],
        "calibration": calibration,
        "trade_close_unix": trade_close_unix,
        "equity_stats": equity_path_stats(trade_close_unix, pnls),
    }


def equity_path_stats(trade_close_unix: list[int], pnls: list[float]) -> dict[str, float]:
    """Min cumulative PnL and max drawdown (from 0) after sorting trades by settlement time."""
    if len(trade_close_unix) != len(pnls) or not pnls:
        return {"min_cumulative_pnl": float("nan"), "max_drawdown": float("nan")}
    pairs = sorted(zip(trade_close_unix, pnls), key=lambda x: x[0])
    cum = 0.0
    peak = 0.0
    min_cum = 0.0
    max_dd = 0.0
    for _t, p in pairs:
        cum += float(p)
        min_cum = min(min_cum, cum)
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return {"min_cumulative_pnl": min_cum, "max_drawdown": max_dd}


def write_equity_plot(
    *,
    trade_close_unix: list[int],
    pnls: list[float],
    out_path: Path,
    title: str,
    bankroll_start: float = 0.0,
) -> None:
    """Cumulative PnL vs market settlement time (UTC), sorted by close time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if len(trade_close_unix) != len(pnls):
        raise ValueError("trade_close_unix and pnls length mismatch")
    if not pnls:
        raise ValueError("no trades to plot")

    pairs = sorted(zip(trade_close_unix, pnls), key=lambda x: x[0])
    times = [datetime.fromtimestamp(t, tz=timezone.utc) for t, _ in pairs]
    pays = [p for _, p in pairs]
    cum: list[float] = []
    s = float(bankroll_start)
    for p in pays:
        s += p
        cum.append(s)

    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    ax.plot(times, cum, color="#1f77b4", linewidth=1.4, drawstyle="steps-post")
    ax.axhline(float(bankroll_start), color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Market settlement close (UTC)")
    ax.set_ylabel(
        "Bankroll ($) = start + cumulative PnL"
        if bankroll_start
        else "Cumulative PnL ($)"
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.25)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest btc-trend model on settled KXBTC15M")
    p.add_argument(
        "--days",
        type=int,
        default=10,
        help="Settled-time lookback (pool of days). With --random-sample-days, all markets in this window are loaded, then random days are chosen.",
    )
    p.add_argument(
        "--random-sample-days",
        type=int,
        default=0,
        help="If >0, pick this many random UTC calendar days from the --days pool (reproducible via --random-seed). Markets are shuffled; use --max-markets to cap.",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="RNG seed for --random-sample-days and market shuffle",
    )
    p.add_argument("--target-sec", type=float, default=120.0)
    p.add_argument("--half-spread", type=float, default=0.02)
    p.add_argument(
        "--half-spread-mult",
        type=float,
        default=1.0,
        help="Multiply half-spread (pessimistic execution; 1.5-2 simulates worse fills)",
    )
    p.add_argument(
        "--min-edge",
        type=float,
        default=0.04,
        help="Base min EV $/contract (matches edge_bot default)",
    )
    p.add_argument(
        "--max-spread",
        type=float,
        default=0.12,
        help="Skip side if spread wider; 0 disables (matches edge_bot)",
    )
    p.add_argument(
        "--edge-spread-mult",
        type=float,
        default=0.30,
        help="Extra EV = mult * spread on traded leg (matches edge_bot)",
    )
    p.add_argument("--fee-coefficient", type=float, default=0.07)
    p.add_argument("--fee-multiplier", type=float, default=1.0)
    p.add_argument(
        "--model",
        type=Path,
        default=Path("data/models/btc_trend_logistic.json"),
    )
    p.add_argument("--sleep", type=float, default=0.0, help="Kalshi + Coinbase pacing (0 = fastest)")
    p.add_argument(
        "--max-markets",
        type=int,
        default=0,
        help="Cap markets (0 = no cap)",
    )
    p.add_argument(
        "--trade-pages",
        type=int,
        default=8,
        help="Max Kalshi trade API pages per market (1000 trades/page)",
    )
    p.add_argument("--max-lookback-sec", type=float, default=3600.0)
    p.add_argument(
        "--coinbase-pad-days",
        type=float,
        default=2.0,
        help="Extra history before window for 6h returns",
    )
    p.add_argument(
        "--max-vol-15m",
        type=float,
        default=0.0,
        help="Skip if vol_15m exceeds this (0 = off). Tune from training-set percentiles.",
    )
    p.add_argument(
        "--news-tilt",
        type=float,
        default=None,
        help="Fair P(YES) shift from BTC headlines (24h to decision). Env BTC_NEWS_TILT if unset. 0=off.",
    )
    p.add_argument(
        "--ask-slip",
        type=float,
        default=0.0,
        help="Add to tape-derived executable asks (slippage stress).",
    )
    p.add_argument(
        "--min-ev-surplus",
        type=str,
        default=None,
        help="Require EV minus spread-adjusted threshold >= this ($/contract). 0=off. Env MIN_EV_SURPLUS_DOLLARS (default 0.02).",
    )
    p.add_argument(
        "--min-direction-confidence",
        type=str,
        default=None,
        help="YES needs fair_yes>=this, NO needs fair_yes<=1-this (0=off). Env MIN_DIRECTION_CONFIDENCE (default 0.62).",
    )
    p.add_argument(
        "--calibration-bins",
        type=int,
        default=0,
        help="If >0, print fair_yes vs empirical YES rate in that many bins (rows after vol filter).",
    )
    p.add_argument(
        "--equity-plot",
        type=Path,
        default=None,
        help="Write PNG: cumulative PnL vs settlement close time (UTC), trades sorted by time.",
    )
    p.add_argument(
        "--bankroll-start",
        type=float,
        default=0.0,
        help="Y-axis offset: plot bankroll = this + cumulative PnL (0 = raw cumulative PnL).",
    )
    args = p.parse_args()

    if not args.model.is_file():
        print(f"Missing model {args.model}", file=sys.stderr)
        sys.exit(2)

    if args.news_tilt is not None:
        news_tilt = float(args.news_tilt)
    else:
        news_tilt = float(os.environ.get("BTC_NEWS_TILT", "0"))

    if args.min_ev_surplus is not None:
        min_ev_surplus = float(Decimal(args.min_ev_surplus))
    else:
        min_ev_surplus = float(Decimal(os.environ.get("MIN_EV_SURPLUS_DOLLARS", "0.02")))
    if min_ev_surplus < 0:
        print("min-ev-surplus must be >= 0", file=sys.stderr)
        sys.exit(2)

    if args.min_direction_confidence is not None:
        min_direction_confidence = float(Decimal(args.min_direction_confidence))
    else:
        min_direction_confidence = float(Decimal(os.environ.get("MIN_DIRECTION_CONFIDENCE", "0.62")))
    if min_direction_confidence < 0 or min_direction_confidence >= 1:
        print("min-direction-confidence must be in [0, 1)", file=sys.stderr)
        sys.exit(2)
    if min_direction_confidence > 0 and min_direction_confidence <= 0.5:
        print("min-direction-confidence must be 0 or > 0.5 (otherwise YES/NO regions overlap)", file=sys.stderr)
        sys.exit(2)

    if args.calibration_bins < 0 or args.calibration_bins > 50:
        print("--calibration-bins must be in [0, 50]", file=sys.stderr)
        sys.exit(2)

    if args.equity_plot is not None and args.bankroll_start < 0:
        print("--bankroll-start must be >= 0", file=sys.stderr)
        sys.exit(2)

    rs = args.random_sample_days
    mode = (
        f"random {rs} UTC day(s) from {args.days}d pool (seed={args.random_seed})"
        if rs > 0
        else f"contiguous last {args.days}d"
    )
    print(
        f"KXBTC15M backtest | {mode} | btc-trend model | "
        f"target_sec~{args.target_sec} half_spread={args.half_spread} "
        f"half_spread_mult={args.half_spread_mult} "
        f"min_edge={args.min_edge} max_spread={args.max_spread or 'off'} "
        f"edge_spread_mult={args.edge_spread_mult} max_vol_15m={args.max_vol_15m or 'off'} "
        f"news_tilt={news_tilt or 'off'} ask_slip={args.ask_slip or 0} "
        f"min_ev_surplus={min_ev_surplus or 'off'} "
        f"min_dir_conf={min_direction_confidence or 'off'}",
        flush=True,
    )
    print("Fetching Coinbase BTC-USD 5m ...", flush=True)

    cfg = KxBtc15mBacktestConfig(
        days=args.days,
        target_sec=args.target_sec,
        half_spread=args.half_spread,
        half_spread_mult=args.half_spread_mult,
        min_edge=args.min_edge,
        max_spread=args.max_spread,
        edge_spread_mult=args.edge_spread_mult,
        fee_coefficient=args.fee_coefficient,
        fee_multiplier=args.fee_multiplier,
        model_path=args.model,
        sleep=args.sleep,
        max_markets=args.max_markets,
        trade_pages=args.trade_pages,
        max_lookback_sec=args.max_lookback_sec,
        coinbase_pad_days=args.coinbase_pad_days,
        max_vol_15m=args.max_vol_15m,
        verbose=True,
        news_tilt=news_tilt,
        random_sample_days=args.random_sample_days,
        random_seed=args.random_seed,
        ask_slip_dollars=float(args.ask_slip),
        min_ev_surplus=min_ev_surplus,
        min_direction_confidence=min_direction_confidence,
        calibration_bins=args.calibration_bins,
    )

    r = run_kxbtc15m_btc_trend_backtest(cfg)

    print("\n=== KXBTC15M + btc-trend backtest (hold to settlement) ===", flush=True)
    if r.get("random_days_sampled"):
        print(f"Random UTC days in sample: {r['random_days_sampled']}", flush=True)
    print(f"Markets with decision tape + Coinbase context: {r['processed']}")
    print(
        f"Trades: {r['n_trade']}  skip_wide_spread: {r['n_skip_wide']}  skip_edge: {r['n_skip_edge']}  "
        f"skip_vol: {r['n_skip_vol']}  skip_surplus: {r['n_skip_surplus']}  "
        f"skip_dir_conf: {r['n_skip_dir']}  "
        f"skip_other: {r['n_skip_rule']}  skipped_data: {r['n_skip_data']}"
    )
    pnls = r["pnls"]
    if pnls:
        t = r["total_pnl"]
        print(f"Total PnL ($ per 1 contract, after est. entry fee): {t:.4f}")
        print(f"Avg per trade: {t / len(pnls):.4f}  trades={len(pnls)}")
        print(f"Min / Max trade PnL: {min(pnls):.4f} / {max(pnls):.4f}")
        es = r.get("equity_stats") or {}
        mdd = float(es.get("max_drawdown", float("nan")))
        mic = float(es.get("min_cumulative_pnl", float("nan")))
        if not math.isnan(mdd) and not math.isnan(mic):
            print(
                f"Equity path (sorted by settlement): min cumulative PnL={float(mic):.4f}  "
                f"max drawdown={float(mdd):.4f}"
            )
    else:
        print("No trades met min_edge (or insufficient data).")
    if args.equity_plot is not None:
        tc = r.get("trade_close_unix") or []
        if not pnls or len(tc) != len(pnls):
            print("Skipping --equity-plot (no trades or missing timestamps).", flush=True)
        else:
            write_equity_plot(
                trade_close_unix=tc,
                pnls=pnls,
                out_path=args.equity_plot,
                title=(
                    f"KXBTC15M btc-trend | {args.days}d contiguous | "
                    f"{len(pnls)} trades | 1 contract"
                ),
                bankroll_start=float(args.bankroll_start),
            )
            print(f"Wrote equity curve: {args.equity_plot.resolve()}", flush=True)
    cal = r.get("calibration")
    if cal:
        print("\n=== Calibration (fair_yes vs settled YES), post-vol-filter rows ===", flush=True)
        print(f"{'bin':>14} {'n':>6}  {'avg_fair':>9}  {'emp_YES%':>9}", flush=True)
        for row in cal:
            lo, hi = row["bin_lo"], row["bin_hi"]
            n = int(row["n"])
            if n <= 0:
                print(f"[{lo:.2f},{hi:.2f})  {n:6d}         —          —", flush=True)
            else:
                af = float(row["avg_fair_yes"])
                er = float(row["empirical_yes_rate"])
                print(
                    f"[{lo:.2f},{hi:.2f})  {n:6d}  {af:9.4f}  {er * 100:9.2f}",
                    flush=True,
                )
    print(
        "\nCaveats: tape mid != live book; Coinbase != Kalshi settlement index; "
        "RSS/NewsAPI news may miss old decision times (neutral tilt); past != future.",
        flush=True,
    )


if __name__ == "__main__":
    main()
