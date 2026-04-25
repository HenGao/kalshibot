#!/usr/bin/env python3
"""
Kalshi edge-trading bot (template).

Trades only when *your* fair probability implies positive expected value vs the
**executable** ask (implied from the order book), after a **quadratic taker-fee
estimate** and a **minimum edge** cushion.

Includes a **stop-loss** on premium at risk: for a long YES, exits (reduce-only
sell) if (entry_ask - current_yes_bid) / entry_ask >= stop fraction.

Multi-market:
  - Default **KXBTC15M only** (``--auto-markets`` defaults to ``15m``, or env
    ``KALSHI_AUTO_MARKETS``). Tokens: ``hourly`` (**KXBTC**), ``daily`` (**KXBTCD**),
    ``both`` (15m+hourly), ``all`` or ``btc`` (15m+hourly+daily). Comma mix e.g.
    ``15m,hourly,daily``. Auto mode picks the open contract with soonest close_time
    per series.
  - Explicit ``--ticker`` / ``KALSHI_MARKET_TICKER`` overrides auto-markets.
  - Per-ticker state under --state-dir (default data/bot_state/).

Simulation:
  - --simulate  → synthetic books + fills; same JSONL as live for
    scripts/live_graph.py (no API keys).

Paper trading:
  - --paper --paper-balance 50  → real Kalshi quotes, virtual cash only (no API
    keys). Wallet in data/paper_wallet.json; each poll writes a paper_portfolio
    row (NAV + PnL vs starting stake). Per-ticker "equity" in JSONL is
    bid-vs-entry (misleading right after a buy); use the portfolio row / dashboard
    "Paper profit" card for bankroll.

Strategy gate (honest workflow):
  - Regenerate ``data/strategy_gate_report.json`` with
    ``python scripts/run_honest_eval.py`` (wraps ``simulate_recent_edge.py``).
  - Set ``REQUIRE_STRATEGY_GATE=1`` or ``--require-strategy-gate`` so **new**
    entries are blocked unless the report exists and ``"passed": true``.
  - Stops and state management still run. Use ``--strategy-gate-file`` or
    ``STRATEGY_GATE_FILE`` to point at a custom report path.

Portfolio kill switch:
  - --max-capital-loss-pct 0.30  (or MAX_CAPITAL_LOSS_PCT): exit the process when
    equity falls 30%% below baseline. **Live:** baseline = first successful
    GET /portfolio/balance total (available cents + portfolio_value cents) in USD
    after startup. **Paper:** baseline = wallet ``starting_capital`` vs current NAV
    (cash + positions marked at bids). Does **not** auto-close positions; use
    ``--no-entry`` first or flatten manually. 0 = disabled (default).

Configuration:
  - Fair P(YES): --btc-trend-model (Coinbase v2 features + Kalshi target + live book
    spread/imbalance; JSON logistic, Platt optional, or HGB manifest/joblib from
    train_btc_trend_kalshi.py), or --fair-yes, or JSON models (train_fair_yes_model.py):
    --fair-model, optional --fair-model-hourly.

This is not financial advice. Test on demo API first. Fees are *estimated*;
actual fees follow Kalshi rounding rules.

Entry quality (defaults tuned to reduce taker bleed):
  MAX_BOOK_SPREAD / --max-spread: skip if that side's spread is wider (0 = disable).
  EDGE_SPREAD_MULT / --edge-spread-mult: extra EV required = mult * spread on the traded leg.
  MIN_EV_SURPLUS_DOLLARS / --min-ev-surplus: require EV minus threshold >= this (skips marginal edges).
  ASK_SLIP_DOLLARS / --ask-slip: add to both asks for EV math + IOC limit prices (slippage stress).
  MIN_SECONDS_BEFORE_CLOSE / --min-seconds-before-close: skip **new** entries when this many
  seconds remain (0 = off). Cuts lottery-style noise near expiry; stops still run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from edge_math import (
    expected_value_buy_no,
    expected_value_buy_yes,
    parse_orderbook,
    quadratic_taker_fee_total_usd,
)
from btc_trend_features import (
    book_features_from_orderbook,
    build_btc_trend_feature_dict,
    build_btc_trend_features_from_closes,
    fetch_btc_spot_usd,
    resolve_kalshi_btc_target_usd,
)
from btc_news_sentiment import apply_btc_news_tilt_to_fair_yes, sentiment_for_decision_utc
from btc_trend_predict import BtcTrendPredictor, load_btc_trend_predictor, predict_btc_trend_yes
from fair_model import FairYesModel, fair_yes_decimal
from kalshi_client import KalshiClient, public_get
from live_metrics import LiveMetricsSink
from paper_broker import PaperBroker
from sim_session import SimSession
from strategy_gate import load_strategy_gate

_REPO_ROOT = Path(__file__).resolve().parent


def series_from_market_ticker(ticker: str) -> str:
    return ticker.split("-", 1)[0]


def safe_ticker_fs(ticker: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in ticker)[:200]


def _parse_close_time_iso(iso: str) -> datetime:
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    return datetime.fromisoformat(iso)


def best_open_market_ticker(series: str, *, public_base: str) -> str | None:
    """
    Pick the **current** open contract for a series: among status=open markets,
    choose the one with the **soonest** close_time (front month / active window).
    Paginates the markets list so the active contract is not missed past the
    first page. Falls back to the first open row if no future close_time parses.
    """
    now = datetime.now(timezone.utc)
    candidates: list[tuple[datetime, str]] = []
    fallback_ticker: str | None = None
    cursor: str | None = None
    while True:
        q: dict[str, str] = {
            "series_ticker": series,
            "status": "open",
            "limit": "200",
        }
        if cursor:
            q["cursor"] = cursor
        d = public_get(f"markets?{urlencode(q)}", base_url=public_base)
        mk = d.get("markets") or []
        for m in mk:
            t = m.get("ticker")
            raw = m.get("close_time")
            if not t or not raw:
                continue
            if fallback_ticker is None:
                fallback_ticker = str(t)
            try:
                ct = _parse_close_time_iso(str(raw))
            except (TypeError, ValueError):
                continue
            if ct > now:
                candidates.append((ct, str(t)))
        cursor = (d.get("cursor") or "").strip() or None
        if not cursor:
            break
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    return fallback_ticker


def resolve_tickers(
    *,
    single: str | None,
    tickers_csv: str | None,
    auto_markets: str | None,
    public_base: str,
    simulate: bool,
) -> list[str]:
    if tickers_csv:
        return [x.strip() for x in tickers_csv.split(",") if x.strip()]
    if single:
        return [single]
    if auto_markets:
        parts = {p.strip().lower() for p in auto_markets.split(",") if p.strip()}
        if "both" in parts:
            parts |= {"15m", "hourly"}
        if "all" in parts or "btc" in parts:
            parts |= {"15m", "hourly", "daily"}
        out: list[str] = []
        if "15m" in parts:
            t = best_open_market_ticker("KXBTC15M", public_base=public_base)
            if t:
                out.append(t)
            else:
                print("[warn] No open KXBTC15M market", file=sys.stderr)
        if "hourly" in parts:
            t = best_open_market_ticker("KXBTC", public_base=public_base)
            if t:
                out.append(t)
            else:
                print("[warn] No open KXBTC (hourly range) market", file=sys.stderr)
        if "daily" in parts:
            t = best_open_market_ticker("KXBTCD", public_base=public_base)
            if t:
                out.append(t)
            else:
                print("[warn] No open KXBTCD (daily) market", file=sys.stderr)
        return out
    if simulate:
        return ["KXBTC15M-SIMDEMO-00"]
    return []


def load_series_fee_multiplier(series_ticker: str, *, public_base: str) -> Decimal:
    try:
        data = public_get(f"series/{series_ticker}", base_url=public_base)
        m = data.get("series", data)
        return Decimal(str(m.get("fee_multiplier", 1)))
    except Exception:
        return Decimal("1")


def paper_net_asset_value(paper: PaperBroker, *, public_base: str) -> Decimal:
    """
    Cash plus open positions marked at the bid you could hit to exit (same side
    as position). This matches how the wallet would look if you flattened now.
    """
    nav = paper.balance
    for t, sz in paper.positions.items():
        if sz == 0:
            continue
        ob = public_get(f"markets/{t}/orderbook", base_url=public_base)
        ex = parse_orderbook(ob)
        if sz > 0:
            nav += sz * ex.best_yes_bid
        else:
            nav += abs(sz) * ex.best_no_bid
    return nav


def fetch_portfolio_equity_usd(client: KalshiClient) -> Decimal:
    """
    Kalshi GET /portfolio/balance: available balance + portfolio value (both cents) → USD.
    Response may be flat or nested under ``balance``.
    """
    r = client.get("/portfolio/balance")
    if not isinstance(r, dict):
        raise RuntimeError(f"unexpected /portfolio/balance response: {r!r}")
    inner = r.get("balance")
    if isinstance(inner, dict):
        bal_c = int(inner.get("balance", 0))
        pv_c = int(inner.get("portfolio_value", 0))
    else:
        bal_c = int(r.get("balance", 0))
        pv_c = int(r.get("portfolio_value", 0))
    return (Decimal(bal_c) + Decimal(pv_c)) / Decimal("100")


def dollars_str(d: Decimal) -> str:
    q = d.quantize(Decimal("0.0001"))
    return format(q, "f")


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def clear_state(path: Path) -> None:
    if path.is_file():
        path.unlink()


def seconds_before_close_from_market_payload(
    data: dict[str, Any], *, now: datetime | None = None
) -> float:
    m = data.get("market", data)
    close_raw = m.get("close_time")
    if not close_raw:
        raise ValueError("market response missing close_time")
    close_dt = _parse_close_time_iso(close_raw)
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return max(0.0, (close_dt - now).total_seconds())


def position_size_signed(pos: dict[str, Any] | None) -> Decimal:
    if not pos:
        return Decimal("0")
    return Decimal(str(pos.get("position_fp") or "0"))


def state_from_market_position_row(ticker: str, pos_row: dict[str, Any]) -> dict[str, Any] | None:
    """
    Build bot stop/entry state from Kalshi GET /portfolio/positions row.
    Positive position_fp = YES contracts; negative = NO contracts.
    entry_ask_dollars = market_exposure_dollars / |contracts| (avg cost; OK for stop math).
    """
    fp = Decimal(str(pos_row.get("position_fp") or "0"))
    if fp == 0:
        return None
    side = "yes" if fp > 0 else "no"
    contracts = abs(fp)
    exp_raw = pos_row.get("market_exposure_dollars")
    if exp_raw is None or str(exp_raw).strip() == "":
        return None
    try:
        exposure = Decimal(str(exp_raw))
    except Exception:
        return None
    if contracts <= 0 or exposure <= 0:
        return None
    avg = exposure / contracts
    if avg <= 0:
        return None
    return {
        "ticker": ticker,
        "side": side,
        "contracts": format(contracts.quantize(Decimal("0.01")), "f"),
        "entry_ask_dollars": format(avg.quantize(Decimal("0.0001")), "f"),
        "order_id": None,
        "adopted_from_api": True,
    }


def place_ioc(
    client: KalshiClient,
    *,
    ticker: str,
    action: str,
    side: str,
    count_fp: str,
    yes_price_dollars: str | None,
    no_price_dollars: str | None,
    reduce_only: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "ticker": ticker,
        "client_order_id": str(uuid.uuid4()),
        "action": action,
        "side": side,
        "type": "limit",
        "time_in_force": "immediate_or_cancel",
        "count_fp": count_fp,
        "reduce_only": reduce_only,
    }
    if yes_price_dollars is not None:
        body["yes_price_dollars"] = yes_price_dollars
    if no_price_dollars is not None:
        body["no_price_dollars"] = no_price_dollars
    return client.post("/portfolio/orders", body)


def maybe_stop_out(
    *,
    client: KalshiClient | None,
    sim: SimSession | None,
    paper: PaperBroker | None,
    ticker: str,
    ex,
    state: dict[str, Any],
    stop_pct: Decimal,
    dry_run: bool,
    fee_coefficient: Decimal,
    fee_multiplier: Decimal,
) -> bool:
    """
    Returns True if a stop (or dry-run stop) fired and position should be flat.
    """
    side = state.get("side")
    contracts = Decimal(str(state.get("contracts") or "0"))
    if contracts <= 0:
        return False

    if side == "yes":
        entry = Decimal(str(state["entry_ask_dollars"]))
        mark = ex.best_yes_bid
        if entry <= 0:
            return False
        dd = (entry - mark) / entry
        if dd < stop_pct:
            return False
        print(
            f"[stop] YES drawdown {float(dd):.2%} >= {float(stop_pct):.2%} "
            f"(entry_ask {entry} vs bid {mark})"
        )
        if sim is not None:
            sim.apply_sell_ioc(
                ticker, side="yes", count_fp=contracts
            )
            return True
        if paper is not None:
            paper.sell_yes_reduce(
                ticker,
                contracts,
                ex.best_yes_bid,
                fee_coefficient=fee_coefficient,
                fee_multiplier=fee_multiplier,
            )
            return True
        if dry_run or client is None:
            return True
        place_ioc(
            client,
            ticker=ticker,
            action="sell",
            side="yes",
            count_fp=f"{contracts:.2f}",
            yes_price_dollars=dollars_str(ex.best_yes_bid),
            no_price_dollars=None,
            reduce_only=True,
        )
        return True

    if side == "no":
        entry = Decimal(str(state["entry_ask_dollars"]))
        mark = ex.best_no_bid
        if entry <= 0:
            return False
        dd = (entry - mark) / entry
        if dd < stop_pct:
            return False
        print(
            f"[stop] NO drawdown {float(dd):.2%} >= {float(stop_pct):.2%} "
            f"(entry_ask {entry} vs bid {mark})"
        )
        if sim is not None:
            sim.apply_sell_ioc(ticker, side="no", count_fp=contracts)
            return True
        if paper is not None:
            paper.sell_no_reduce(
                ticker,
                contracts,
                ex.best_no_bid,
                fee_coefficient=fee_coefficient,
                fee_multiplier=fee_multiplier,
            )
            return True
        if dry_run or client is None:
            return True
        place_ioc(
            client,
            ticker=ticker,
            action="sell",
            side="no",
            count_fp=f"{contracts:.2f}",
            yes_price_dollars=None,
            no_price_dollars=dollars_str(ex.best_no_bid),
            reduce_only=True,
        )
        return True

    return False


def run_cycle(
    *,
    client: KalshiClient | None,
    sim: SimSession | None,
    paper: PaperBroker | None,
    ticker: str,
    fair_yes: Decimal | None,
    fair_model: FairYesModel | None,
    fair_model_hourly: FairYesModel | None = None,
    btc_trend_model: BtcTrendPredictor | None = None,
    contracts: Decimal,
    min_edge: Decimal,
    stop_pct: Decimal,
    fee_coefficient: Decimal,
    fee_multiplier: Decimal,
    state_path: Path,
    dry_run: bool,
    allow_entry: bool,
    entry_block_event: str | None = None,
    metrics: LiveMetricsSink | None = None,
    max_book_spread: Decimal = Decimal("0"),
    edge_spread_mult: Decimal = Decimal("0"),
    btc_news_tilt: Decimal = Decimal("0"),
    min_ev_surplus: Decimal = Decimal("0"),
    ask_slip_dollars: Decimal = Decimal("0"),
    min_direction_confidence: Decimal = Decimal("0"),
    adopt_live_position: bool = False,
    min_seconds_before_close: float = 0.0,
) -> None:
    public_base = os.environ.get(
        "KALSHI_PUBLIC_BASE", "https://api.elections.kalshi.com/trade-api/v2"
    )
    if sim is not None:
        ob = sim.step_book(ticker)
    else:
        ob = public_get(f"markets/{ticker}/orderbook", base_url=public_base)
    ex = parse_orderbook(ob)

    if sim is not None:
        mdata = sim.market_payload_for_features(ticker)
    else:
        mdata = public_get(f"markets/{ticker}", base_url=public_base)
    mkt = mdata.get("market", mdata)
    sec = seconds_before_close_from_market_payload(mdata)

    fm = fair_model
    if fair_model_hourly is not None and not ticker.startswith("KXBTC15M"):
        fm = fair_model_hourly

    btc_spot: float | None = None
    btc_tgt: float | None = None
    btc_tgt_src: str | None = None
    btc_news_snap: tuple[float, str] | None = None

    if btc_trend_model is not None:
        try:
            if sim is not None:
                closes = sim.btc_closes(ticker)
                btc_tgt = float(mkt.get("floor_strike") or sim.btc_target.get(ticker, closes[-1]))
                btc_spot = float(closes[-1])
                btc_tgt_src = "simulate"
                bf = book_features_from_orderbook(ob)
                feats = build_btc_trend_features_from_closes(
                    closes,
                    seconds_before_close=sec,
                    target_usd=btc_tgt,
                    book_yes_spread=bf["book_yes_spread"],
                    book_yes_imbalance=bf["book_yes_imbalance"],
                    decision_ts_utc=int(time.time()),
                )
            else:
                btc_spot = float(fetch_btc_spot_usd())
                tgt, btc_tgt_src = resolve_kalshi_btc_target_usd(
                    mkt, ticker=ticker, spot_hint=btc_spot
                )
                if tgt is None:
                    print(
                        f"[{ticker}] [btc-trend] cannot resolve Kalshi BTC target; skip",
                        file=sys.stderr,
                    )
                    if metrics is not None:
                        metrics.update(
                            ts=time.time(),
                            ticker=ticker,
                            cycle_event="btc_no_target",
                        )
                    return
                btc_tgt = float(tgt)
                feats = build_btc_trend_feature_dict(
                    seconds_before_close=sec,
                    target_usd=btc_tgt,
                    spot_usd=btc_spot,
                    orderbook_json=ob,
                    decision_ts_utc=int(time.time()),
                )
            fair_yes_eff = Decimal(str(predict_btc_trend_yes(btc_trend_model, feats)))
            stype = str(mkt.get("strike_type") or "")
            if stype == "less" and mkt.get("cap_strike") is not None:
                fair_yes_eff = Decimal("1") - fair_yes_eff
                fair_yes_eff = max(Decimal("0.01"), min(Decimal("0.99"), fair_yes_eff))
        except Exception as e:
            print(f"[{ticker}] [btc-trend] feature error: {e}", file=sys.stderr)
            if metrics is not None:
                metrics.update(
                    ts=time.time(),
                    ticker=ticker,
                    cycle_event="btc_trend_error",
                )
            return
    elif fm is not None:
        feats = {
            "log1p_seconds_before_close": math.log1p(sec),
            "yes_price": float(ex.best_yes_ask),
        }
        fair_yes_eff = fair_yes_decimal(fm, feats)
    elif fair_yes is not None:
        fair_yes_eff = fair_yes
    else:
        raise ValueError("fair_yes or at least one fair model is required")

    if btc_news_tilt > 0 and sim is None:
        btc_news_snap = sentiment_for_decision_utc(datetime.now(timezone.utc))
        sc, nsrc = btc_news_snap
        fair_yes_eff = apply_btc_news_tilt_to_fair_yes(fair_yes_eff, sc, btc_news_tilt)
        print(
            f"[{ticker}] [btc-news] sentiment={sc:+.3f} source={nsrc} "
            f"tilt={btc_news_tilt} fair_yes={fair_yes_eff}"
        )

    if sim is not None:
        state = sim.load_state(ticker)
    elif paper is not None:
        state = paper.load_state(ticker)
    else:
        state = load_state(state_path)
    pos_row = None
    if sim is not None:
        size = sim.position_fp(ticker)
    elif paper is not None:
        size = paper.position_fp(ticker)
    elif client is not None:
        pos = client.get("/portfolio/positions", params={"ticker": ticker})
        for p in pos.get("market_positions") or []:
            if p.get("ticker") == ticker:
                pos_row = p
                break
        size = position_size_signed(pos_row)
    else:
        size = Decimal("0")

    if (
        adopt_live_position
        and client is not None
        and sim is None
        and paper is None
        and state is None
        and size != 0
        and pos_row is not None
    ):
        adopted = state_from_market_position_row(ticker, pos_row)
        if adopted is not None:
            save_state(state_path, adopted)
            state = adopted
            print(
                f"[state] Adopted live position from API → {state_path} "
                f"(side={adopted['side']} contracts={adopted['contracts']} "
                f"entry_ask≈{adopted['entry_ask_dollars']}; use for stop-loss). "
                "Review if exposure is not your true avg entry."
            )
        else:
            print(
                f"[warn] --adopt-live-position set but could not build state "
                f"(need market_exposure_dollars + position_fp). {state_path}",
                file=sys.stderr,
            )

    print(
        f"[{ticker}] [book] yes bid/ask {ex.best_yes_bid}/{ex.best_yes_ask} "
        f"no bid/ask {ex.best_no_bid}/{ex.best_no_ask} "
        f"spreads yes={ex.yes_spread} no={ex.no_spread} pos={size}"
    )
    if btc_trend_model is not None and btc_spot is not None and btc_tgt is not None:
        dist = (btc_spot - btc_tgt) / btc_tgt * 100.0
        print(
            f"[{ticker}] [btc-trend] spot=${btc_spot:,.2f} target=${btc_tgt:,.2f} "
            f"({btc_tgt_src}) dist={dist:+.3f}% fair_yes={fair_yes_eff}"
        )

    def _metrics_base() -> None:
        if metrics is None:
            return
        bot_side = None
        bot_contracts = None
        bot_entry = None
        if state and state.get("ticker") == ticker:
            bot_side = state.get("side")
            bot_contracts = state.get("contracts")
            bot_entry = state.get("entry_ask_dollars")
        u: dict[str, Any] = dict(
            ts=time.time(),
            ticker=ticker,
            best_yes_bid=ex.best_yes_bid,
            best_yes_ask=ex.best_yes_ask,
            best_no_bid=ex.best_no_bid,
            best_no_ask=ex.best_no_ask,
            position_fp=size,
            fair_yes=float(fair_yes_eff),
            bot_side=bot_side,
            bot_contracts=bot_contracts,
            bot_entry_ask=bot_entry,
            cycle_event="poll",
        )
        if btc_spot is not None:
            u["btc_spot_usd"] = float(btc_spot)
        if btc_news_snap is not None:
            u["btc_news_sentiment"] = btc_news_snap[0]
            u["btc_news_source"] = btc_news_snap[1]
            u["btc_news_tilt"] = float(btc_news_tilt)
        if btc_tgt is not None:
            u["kalshi_target_usd"] = float(btc_tgt)
        if btc_tgt_src:
            u["btc_target_source"] = btc_tgt_src
        if paper is not None:
            u["paper_balance"] = float(paper.balance)
            u["paper_mode"] = True
        if (
            client is not None
            and pos_row is not None
            and size != 0
            and not (state and state.get("ticker") == ticker)
        ):
            ost = state_from_market_position_row(ticker, pos_row)
            if ost and ost.get("entry_ask_dollars"):
                u["orphan_entry_ask_dollars"] = ost["entry_ask_dollars"]
        metrics.update(**u)

    _metrics_base()

    # --- Stop management for bot-opened positions ---
    if state and state.get("ticker") == ticker:
        if size == 0:
            print("[state] position flat; clearing state file")
            if sim is not None:
                sim.clear_state(ticker)
            elif paper is not None:
                paper.clear_state(ticker)
            else:
                clear_state(state_path)
            state = None
            if metrics is not None:
                metrics.update(
                    bot_side=None,
                    bot_contracts=None,
                    bot_entry_ask=None,
                    cycle_event="position_flat",
                )
        else:
            fired = maybe_stop_out(
                client=client,
                sim=sim,
                paper=paper,
                ticker=ticker,
                ex=ex,
                state=state,
                stop_pct=stop_pct,
                dry_run=dry_run,
                fee_coefficient=fee_coefficient,
                fee_multiplier=fee_multiplier,
            )
            if fired and not dry_run and client is not None:
                if metrics is not None:
                    metrics.update(cycle_event="stop_exit")
                clear_state(state_path)
                return
            if fired and sim is not None:
                if metrics is not None:
                    metrics.update(cycle_event="stop_exit")
                sim.clear_state(ticker)
                return
            if fired and paper is not None:
                if metrics is not None:
                    metrics.update(cycle_event="stop_exit")
                paper.clear_state(ticker)
                return
            if fired:
                if metrics is not None:
                    metrics.update(cycle_event="stop_exit")
                return

    # --- Entry: require flat book position ---
    if size != 0:
        if state is None:
            print(
                "[skip] Non-zero position but no bot state (manual trade, other algo, or "
                "lost state file). Flatten on Kalshi, or run once with "
                "`--adopt-live-position` to seed state from API cost (then stop-loss applies). "
                f"Expected file: {state_path}"
            )
            hev = "hold_nonbot"
        else:
            hev = "hold"
        if metrics is not None:
            metrics.update(cycle_event=hev)
        return

    if not allow_entry:
        if metrics is not None:
            metrics.update(cycle_event=entry_block_event or "no_new_entry")
        return

    if min_seconds_before_close > 0 and sec < min_seconds_before_close:
        print(
            f"[{ticker}] [entry] Skip: {sec:.1f}s to close < "
            f"min_seconds_before_close={min_seconds_before_close:.0f}s "
            "(model/spread unreliable near expiry)"
        )
        if metrics is not None:
            metrics.update(cycle_event="too_close_to_expiry")
        return

    yes_ask_exec = min(Decimal("0.99"), ex.best_yes_ask + ask_slip_dollars)
    no_ask_exec = min(Decimal("0.99"), ex.best_no_ask + ask_slip_dollars)

    fee_yes = quadratic_taker_fee_total_usd(
        yes_ask_exec, contracts, coefficient=fee_coefficient, fee_multiplier=fee_multiplier
    ) / contracts
    fee_no = quadratic_taker_fee_total_usd(
        no_ask_exec, contracts, coefficient=fee_coefficient, fee_multiplier=fee_multiplier
    ) / contracts

    ev_yes = expected_value_buy_yes(fair_yes_eff, yes_ask_exec, fee_yes)
    ev_no = expected_value_buy_no(fair_yes_eff, no_ask_exec, fee_no)

    print(
        f"[edge] fair_yes={fair_yes_eff} EV/contract yes={dollars_str(ev_yes)} "
        f"no={dollars_str(ev_no)} (est fee/ct yes={dollars_str(fee_yes)} no={dollars_str(fee_no)})"
    )

    if metrics is not None:
        metrics.update(ev_yes=float(ev_yes), ev_no=float(ev_no))

    thr_yes = min_edge + edge_spread_mult * ex.yes_spread
    thr_no = min_edge + edge_spread_mult * ex.no_spread
    yes_spread_ok = max_book_spread <= 0 or ex.yes_spread <= max_book_spread
    no_spread_ok = max_book_spread <= 0 or ex.no_spread <= max_book_spread

    c_yes = min_direction_confidence
    c_no = Decimal("1") - min_direction_confidence
    yes_dir_ok = c_yes <= 0 or fair_yes_eff >= c_yes
    no_dir_ok = c_yes <= 0 or fair_yes_eff <= c_no

    best_side = None
    best_ev = Decimal("-999")
    if yes_spread_ok and ev_yes >= ev_no and ev_yes >= thr_yes:
        if (min_ev_surplus <= 0 or ev_yes - thr_yes >= min_ev_surplus) and yes_dir_ok:
            best_side, best_ev = "yes", ev_yes
    if best_side is None and no_spread_ok and ev_no > ev_yes and ev_no >= thr_no:
        if (min_ev_surplus <= 0 or ev_no - thr_no >= min_ev_surplus) and no_dir_ok:
            best_side, best_ev = "no", ev_no

    if best_side is None:
        spread_blocked = max_book_spread > 0 and (
            (ev_yes >= ev_no and not yes_spread_ok) or (ev_no > ev_yes and not no_spread_ok)
        )
        if spread_blocked:
            print(
                f"[entry] Skip: spread too wide on favorable side "
                f"(yes={ex.yes_spread} no={ex.no_spread} cap={max_book_spread})."
            )
            evc = "wide_spread"
        elif ev_yes < thr_yes and ev_no < thr_no:
            print(
                f"[entry] No side meets spread-adjusted edge "
                f"(thr_yes={dollars_str(thr_yes)} thr_no={dollars_str(thr_no)} "
                f"ev_yes={dollars_str(ev_yes)} ev_no={dollars_str(ev_no)})."
            )
            evc = "no_edge"
        elif min_ev_surplus > 0 and (
            (yes_spread_ok and ev_yes >= ev_no and ev_yes >= thr_yes and ev_yes - thr_yes < min_ev_surplus)
            or (no_spread_ok and ev_no > ev_yes and ev_no >= thr_no and ev_no - thr_no < min_ev_surplus)
        ):
            print(
                f"[entry] Skip: EV surplus below min_ev_surplus={dollars_str(min_ev_surplus)} "
                f"(thr_yes={dollars_str(thr_yes)} thr_no={dollars_str(thr_no)})."
            )
            evc = "no_edge_surplus"
        elif c_yes > 0 and (
            (
                yes_spread_ok
                and ev_yes >= ev_no
                and ev_yes >= thr_yes
                and (min_ev_surplus <= 0 or ev_yes - thr_yes >= min_ev_surplus)
                and not yes_dir_ok
            )
            or (
                no_spread_ok
                and ev_no > ev_yes
                and ev_no >= thr_no
                and (min_ev_surplus <= 0 or ev_no - thr_no >= min_ev_surplus)
                and not no_dir_ok
            )
        ):
            print(
                f"[entry] Skip: direction confidence (YES needs fair_yes>={c_yes}, "
                f"NO needs fair_yes<={c_no}; fair_yes={fair_yes_eff})"
            )
            evc = "no_direction_confidence"
        else:
            print(
                f"[entry] No trade after filters (EV vs tie/threshold; "
                f"thr_yes={dollars_str(thr_yes)} thr_no={dollars_str(thr_no)})."
            )
            evc = "no_edge_tie"
        if metrics is not None:
            metrics.update(cycle_event=evc)
        return

    print(f"[entry] IOC buy {best_side} count={contracts} EV~{dollars_str(best_ev)}/contract")

    if dry_run and sim is None and paper is None:
        if metrics is not None:
            metrics.update(cycle_event="dry_entry", entry_side=best_side, entry_ev=float(best_ev))
        return

    if sim is not None:
        resp = sim.apply_buy_ioc(
            ticker,
            side=best_side,
            count_fp=contracts,
            yes_ask=yes_ask_exec,
            no_ask=no_ask_exec,
        )
        entry_ask = yes_ask_exec if best_side == "yes" else no_ask_exec
        filled = contracts
    elif paper is not None:
        if best_side == "yes":
            ok, resp = paper.buy_yes(
                ticker,
                contracts,
                yes_ask_exec,
                fee_coefficient=fee_coefficient,
                fee_multiplier=fee_multiplier,
            )
            entry_ask = yes_ask_exec
        else:
            ok, resp = paper.buy_no(
                ticker,
                contracts,
                no_ask_exec,
                fee_coefficient=fee_coefficient,
                fee_multiplier=fee_multiplier,
            )
            entry_ask = no_ask_exec
        if not ok:
            print(f"[paper] Skip entry (cash): {resp}")
            if metrics is not None:
                metrics.update(cycle_event="paper_no_cash")
            return
        filled = contracts
    elif client is None:
        if metrics is not None:
            metrics.update(cycle_event="dry_entry", entry_side=best_side, entry_ev=float(best_ev))
        return
    elif best_side == "yes":
        resp = place_ioc(
            client,
            ticker=ticker,
            action="buy",
            side="yes",
            count_fp=f"{contracts:.2f}",
            yes_price_dollars=dollars_str(yes_ask_exec),
            no_price_dollars=None,
            reduce_only=False,
        )
        entry_ask = yes_ask_exec
        filled = Decimal(str(resp["order"].get("fill_count_fp") or "0"))
    else:
        resp = place_ioc(
            client,
            ticker=ticker,
            action="buy",
            side="no",
            count_fp=f"{contracts:.2f}",
            yes_price_dollars=None,
            no_price_dollars=dollars_str(no_ask_exec),
            reduce_only=False,
        )
        entry_ask = no_ask_exec
        filled = Decimal(str(resp["order"].get("fill_count_fp") or "0"))

    if filled <= 0:
        print("[entry] IOC had no fill; not saving state.")
        if metrics is not None:
            metrics.update(cycle_event="no_fill")
        return

    st = {
        "ticker": ticker,
        "side": best_side,
        "contracts": str(filled),
        "entry_ask_dollars": str(entry_ask),
        "order_id": (resp.get("order") or {}).get("order_id"),
    }
    if sim is not None:
        sim.save_state(ticker, st)
        print(f"[state] Saved sim state (filled {filled})")
    elif paper is not None:
        paper.save_state(ticker, st)
        print(f"[paper] Filled {filled}; cash balance ${paper.balance:.4f}")
    else:
        save_state(state_path, st)
        print(f"[state] Saved {state_path} (filled {filled})")
    if metrics is not None:
        metrics.update(
            cycle_event="filled",
            entry_side=best_side,
            filled=float(filled),
            bot_side=best_side,
            bot_contracts=str(filled),
            bot_entry_ask=str(entry_ask),
        )


def load_project_dotenv() -> None:
    """Load project-root .env into os.environ (does not override existing vars)."""
    path = Path(__file__).resolve().parent / ".env"
    if not path.is_file():
        return
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ[key] = val


def main() -> None:
    load_project_dotenv()
    p = argparse.ArgumentParser(description="Kalshi edge + stop-loss bot (multi-market + simulate)")
    p.add_argument("--ticker", default=os.environ.get("KALSHI_MARKET_TICKER"), required=False)
    p.add_argument(
        "--tickers",
        default=os.environ.get("KALSHI_MARKET_TICKERS"),
        help="Comma-separated market tickers (overrides --ticker)",
    )
    p.add_argument(
        "--auto-markets",
        default=os.environ.get("KALSHI_AUTO_MARKETS") or "15m",
        help="Discover open BTC markets each poll (default: 15m = KXBTC15M). "
        "Tokens: hourly (KXBTC), daily (KXBTCD), both (15m+hourly), all|btc (15m+hourly+daily), "
        "or comma mix e.g. 15m,hourly,daily",
    )
    p.add_argument(
        "--simulate",
        action="store_true",
        help="Fake books + fills; writes same metrics JSONL for live_graph (no API keys)",
    )
    p.add_argument(
        "--paper",
        action="store_true",
        help="Paper trade: real Kalshi quotes, virtual cash only (no API keys)",
    )
    p.add_argument(
        "--paper-balance",
        type=str,
        default=os.environ.get("PAPER_BALANCE", "50"),
        help="Starting USD when creating a new paper wallet (ignored if wallet exists)",
    )
    p.add_argument(
        "--paper-wallet",
        type=Path,
        default=Path(os.environ.get("PAPER_WALLET", "data/paper_wallet.json")),
    )
    p.add_argument(
        "--paper-reset",
        action="store_true",
        help="Delete paper wallet file before start (fresh balance)",
    )
    p.add_argument(
        "--fair-yes",
        type=str,
        default=os.environ.get("FAIR_YES"),
        help="Fair P(YES), e.g. 0.58 (ignored if --fair-model is set)",
    )
    p.add_argument(
        "--fair-model",
        type=Path,
        default=(
            Path(x)
            if (x := os.environ.get("FAIR_YES_MODEL"))
            else None
        ),
        help="JSON model for KXBTC15M-style markets (default path used if file exists)",
    )
    p.add_argument(
        "--fair-model-hourly",
        type=Path,
        default=(
            Path(x)
            if (x := os.environ.get("FAIR_YES_MODEL_HOURLY"))
            else None
        ),
        help="Optional JSON for non-KXBTC15M tickers (e.g. KXBTC hourly); else uses --fair-model",
    )
    p.add_argument(
        "--btc-trend-model",
        type=Path,
        default=None,
        help="JSON from scripts/train_btc_trend_model.py; Coinbase BTC trends + Kalshi target (overrides fair-model)",
    )
    p.add_argument(
        "--btc-news-tilt",
        type=str,
        default=None,
        help="Max fair_yes shift from daily BTC headlines (0=off). Env BTC_NEWS_TILT. Positive news -> lean YES/up.",
    )
    p.add_argument(
        "--min-ev-surplus",
        type=str,
        default=None,
        help="Require EV minus threshold >= this (0=off). Env MIN_EV_SURPLUS_DOLLARS.",
    )
    p.add_argument(
        "--ask-slip",
        type=str,
        default=None,
        help="Add to executable ask(s) for EV + orders (stress slippage). Env ASK_SLIP_DOLLARS.",
    )
    p.add_argument(
        "--min-direction-confidence",
        type=str,
        default=None,
        help="Min fair P(side): YES needs fair_yes>=this, NO needs fair_yes<=1-this (0=off). "
        "Env MIN_DIRECTION_CONFIDENCE (default 0.62 for rarer, higher-conviction trades).",
    )
    p.add_argument("--contracts", type=str, default=os.environ.get("CONTRACTS", "1"))
    p.add_argument(
        "--min-seconds-before-close",
        type=float,
        default=float(os.environ.get("MIN_SECONDS_BEFORE_CLOSE", "90")),
        help="Skip new entries if fewer seconds remain (0=off). Env MIN_SECONDS_BEFORE_CLOSE.",
    )
    p.add_argument(
        "--min-edge",
        type=str,
        default=os.environ.get("MIN_EDGE_DOLLARS", "0.05"),
        help="Base min EV $/contract after ask + est fee (before spread cushion)",
    )
    p.add_argument(
        "--max-spread",
        type=str,
        default=os.environ.get("MAX_BOOK_SPREAD", "0.10"),
        help="Skip side if bid/ask spread exceeds this (0 = disable cap)",
    )
    p.add_argument(
        "--edge-spread-mult",
        type=str,
        default=os.environ.get("EDGE_SPREAD_MULT", "0.40"),
        help="Extra EV required: mult * (spread on traded leg), on top of --min-edge",
    )
    p.add_argument(
        "--stop-pct",
        type=str,
        default=os.environ.get("STOP_LOSS_PCT", "0.20"),
        help="Stop if (entry_ask - bid)/entry_ask reaches this (0.20 = 20%%)",
    )
    p.add_argument(
        "--max-capital-loss-pct",
        type=str,
        default=os.environ.get("MAX_CAPITAL_LOSS_PCT", "0"),
        help="Exit process if equity falls this fraction below baseline (0=off). "
        "Live: baseline from first /portfolio/balance. Paper: NAV vs starting_capital. "
        "Example: 0.30 = halt after 30%% drawdown from baseline.",
    )
    p.add_argument(
        "--fee-coefficient",
        type=str,
        default=os.environ.get("FEE_COEFFICIENT", "0.07"),
        help="Quadratic taker coefficient (Kalshi default often 0.07 before multiplier)",
    )
    p.add_argument("--interval", type=float, default=5.0, help="Seconds between polls")
    p.add_argument("--once", action="store_true", help="Single cycle then exit")
    p.add_argument("--dry-run", action="store_true", help="Log only; no orders (not used with --simulate)")
    p.add_argument(
        "--no-entry",
        action="store_true",
        help="Only manage stop / state; do not open new positions",
    )
    p.add_argument(
        "--require-strategy-gate",
        action="store_true",
        help="Block new entries unless STRATEGY_GATE_FILE report has passed=true "
        "(also set REQUIRE_STRATEGY_GATE=1). Stops still run.",
    )
    p.add_argument(
        "--strategy-gate-file",
        type=Path,
        default=None,
        help="JSON from scripts/run_honest_eval.py / simulate_recent_edge --write-report. "
        "Default: STRATEGY_GATE_FILE env or data/strategy_gate_report.json",
    )
    p.add_argument(
        "--state-dir",
        type=Path,
        default=Path(os.environ.get("BOT_STATE_DIR", "data/bot_state")),
        help="Per-ticker state files: state_<ticker>.json",
    )
    p.add_argument(
        "--adopt-live-position",
        action="store_true",
        help="Live only: if missing state file but portfolio has this market's position, "
        "create state from position_fp + market_exposure_dollars (enables stop-loss). "
        "Or set BOT_ADOPT_LIVE_POSITION=1.",
    )
    p.add_argument(
        "--metrics-file",
        type=Path,
        default=Path(os.environ.get("BOT_METRICS_FILE", "data/live_metrics.jsonl")),
        help="Append JSONL metrics for scripts/live_graph.py (omit to disable)",
    )
    p.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable writing --metrics-file",
    )
    args = p.parse_args()
    if os.environ.get("BOT_ADOPT_LIVE_POSITION", "").strip().lower() in ("1", "true", "yes"):
        args.adopt_live_position = True

    if args.btc_trend_model is None and os.environ.get("BTC_TREND_MODEL"):
        args.btc_trend_model = Path(os.environ["BTC_TREND_MODEL"])

    if args.btc_news_tilt is not None:
        btc_news_tilt = Decimal(args.btc_news_tilt)
    else:
        btc_news_tilt = Decimal(os.environ.get("BTC_NEWS_TILT", "0"))
    if btc_news_tilt < 0:
        print("btc-news-tilt must be >= 0", file=sys.stderr)
        sys.exit(2)

    if args.min_ev_surplus is not None:
        min_ev_surplus = Decimal(args.min_ev_surplus)
    else:
        min_ev_surplus = Decimal(os.environ.get("MIN_EV_SURPLUS_DOLLARS", "0.03"))
    if args.min_direction_confidence is not None:
        min_direction_confidence = Decimal(args.min_direction_confidence)
    else:
        min_direction_confidence = Decimal(os.environ.get("MIN_DIRECTION_CONFIDENCE", "0.64"))
    if args.ask_slip is not None:
        ask_slip_dollars = Decimal(args.ask_slip)
    else:
        ask_slip_dollars = Decimal(os.environ.get("ASK_SLIP_DOLLARS", "0.005"))
    if min_ev_surplus < 0 or ask_slip_dollars < 0:
        print("min-ev-surplus and ask-slip must be >= 0", file=sys.stderr)
        sys.exit(2)
    min_seconds_before_close = float(args.min_seconds_before_close)
    if min_seconds_before_close < 0:
        print("min-seconds-before-close must be >= 0", file=sys.stderr)
        sys.exit(2)
    if min_direction_confidence < 0 or min_direction_confidence >= 1:
        print("min-direction-confidence must be in [0, 1) (0 disables)", file=sys.stderr)
        sys.exit(2)
    if min_direction_confidence > 0 and min_direction_confidence <= Decimal("0.5"):
        print("min-direction-confidence must be > 0.5 (e.g. 0.62) or 0 to disable", file=sys.stderr)
        sys.exit(2)

    max_capital_loss_pct = Decimal(args.max_capital_loss_pct)
    if max_capital_loss_pct < 0 or max_capital_loss_pct >= 1:
        print("max-capital-loss-pct must be in [0, 1) (0 disables)", file=sys.stderr)
        sys.exit(2)

    if args.paper and args.simulate:
        print("Use either --paper or --simulate, not both.", file=sys.stderr)
        sys.exit(2)

    public_base = os.environ.get(
        "KALSHI_PUBLIC_BASE", "https://api.elections.kalshi.com/trade-api/v2"
    )

    tickers = resolve_tickers(
        single=args.ticker,
        tickers_csv=args.tickers,
        auto_markets=args.auto_markets,
        public_base=public_base,
        simulate=bool(args.simulate),
    )
    if not tickers:
        print(
            "Set --ticker, KALSHI_MARKET_TICKER, --tickers, or --auto-markets (default 15m; "
            "also hourly, both, btc), or use --simulate / --paper",
            file=sys.stderr,
        )
        sys.exit(2)

    btc_trend_model_obj: BtcTrendPredictor | None = None
    if args.btc_trend_model is not None:
        if not args.btc_trend_model.is_file():
            print(f"BTC trend model not found: {args.btc_trend_model}", file=sys.stderr)
            sys.exit(2)
        btc_trend_model_obj = load_btc_trend_predictor(args.btc_trend_model)
        print(f"[model] BTC trend model {args.btc_trend_model}")

    if btc_trend_model_obj is None and args.fair_model is None and not args.fair_yes:
        fallback = Path("data/models/fair_yes_logistic.json")
        if fallback.is_file():
            args.fair_model = fallback
            print(f"[model] Using default {fallback}")

    fair_model: FairYesModel | None = None
    fair_model_hourly: FairYesModel | None = None
    fair_yes: Decimal | None = None
    if btc_trend_model_obj is not None:
        pass
    elif args.fair_model is not None:
        if not args.fair_model.is_file():
            print(f"Fair model not found: {args.fair_model}", file=sys.stderr)
            sys.exit(2)
        fair_model = FairYesModel.load_path(args.fair_model)
    else:
        if not args.fair_yes:
            print(
                "Set --btc-trend-model, --fair-yes / FAIR_YES, FAIR_YES_MODEL / --fair-model, "
                "or run scripts/train_btc_trend_model.py / train_fair_yes_model.py",
                file=sys.stderr,
            )
            sys.exit(2)
        fair_yes = Decimal(args.fair_yes)
        if not Decimal("0") < fair_yes < Decimal("1"):
            print("fair_yes must be in (0,1)", file=sys.stderr)
            sys.exit(2)

    if btc_trend_model_obj is None and args.fair_model_hourly is not None:
        if not args.fair_model_hourly.is_file():
            print(f"Hourly fair model not found: {args.fair_model_hourly}", file=sys.stderr)
            sys.exit(2)
        fair_model_hourly = FairYesModel.load_path(args.fair_model_hourly)

    contracts = Decimal(args.contracts)
    if contracts <= 0:
        print("contracts must be > 0", file=sys.stderr)
        sys.exit(2)

    min_edge = Decimal(args.min_edge)
    max_book_spread = Decimal(args.max_spread)
    edge_spread_mult = Decimal(args.edge_spread_mult)
    if min_edge < 0 or max_book_spread < 0 or edge_spread_mult < 0:
        print("min-edge, max-spread, edge-spread-mult must be >= 0", file=sys.stderr)
        sys.exit(2)
    stop_pct = Decimal(args.stop_pct)
    fee_coefficient = Decimal(args.fee_coefficient)

    print(
        f"[filters] min_edge={min_edge} max_spread={max_book_spread or 'off'} "
        f"edge_spread_mult={edge_spread_mult} btc_news_tilt={btc_news_tilt or 'off'} "
        f"min_ev_surplus={min_ev_surplus or 'off'} ask_slip={ask_slip_dollars or 'off'} "
        f"min_dir_conf={min_direction_confidence or 'off'} "
        f"min_sec_to_close={min_seconds_before_close or 'off'} "
        f"max_capital_loss_pct={max_capital_loss_pct or 'off'}"
    )

    sim: SimSession | None = None
    paper: PaperBroker | None = None
    if args.paper:
        if args.paper_reset and args.paper_wallet.is_file():
            args.paper_wallet.unlink()
        paper = PaperBroker.load(args.paper_wallet, Decimal(args.paper_balance))
        print(
            f"[paper] tickers={tickers} cash=${paper.balance} "
            f"start=${paper.starting_capital} wallet={args.paper_wallet}"
        )
    elif args.simulate:
        sim = SimSession(tickers=tuple(tickers))
        print(f"[simulate] tickers={tickers}")

    client: KalshiClient | None = None
    if not args.simulate and not args.paper and not args.dry_run:
        key_id = os.environ.get("KALSHI_ACCESS_KEY_ID")
        key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        host = os.environ.get("KALSHI_HOST", "https://api.elections.kalshi.com")
        if not key_id or not key_path:
            print(
                "Need KALSHI_ACCESS_KEY_ID and KALSHI_PRIVATE_KEY_PATH "
                "(or use --dry-run / --simulate / --paper)",
                file=sys.stderr,
            )
            sys.exit(2)
        client = KalshiClient(api_key_id=key_id, private_key_pem_path=key_path, host=host)

    allow_entry_cli = not args.no_entry
    require_strategy_gate = bool(args.require_strategy_gate) or (
        os.environ.get("REQUIRE_STRATEGY_GATE", "").strip().lower() in ("1", "true", "yes")
    )
    strategy_gate_path = (
        args.strategy_gate_file
        if args.strategy_gate_file is not None
        else Path(os.environ.get("STRATEGY_GATE_FILE", "data/strategy_gate_report.json"))
    )
    if not strategy_gate_path.is_absolute():
        strategy_gate_path = _REPO_ROOT / strategy_gate_path

    metrics_sink: LiveMetricsSink | None = None
    if not args.no_metrics:
        metrics_sink = LiveMetricsSink(args.metrics_file)

    args.state_dir.mkdir(parents=True, exist_ok=True)

    rediscover_btc = (
        bool(args.auto_markets)
        and not args.tickers
        and not args.ticker
        and not args.simulate
    )
    last_announced: tuple[str, ...] | None = None
    live_equity_baseline: Decimal | None = None
    last_strategy_gate_log = 0.0

    while True:
        try:
            if (
                max_capital_loss_pct > 0
                and client is not None
                and live_equity_baseline is None
            ):
                try:
                    live_equity_baseline = fetch_portfolio_equity_usd(client)
                    print(
                        f"[risk] Live equity baseline ${dollars_str(live_equity_baseline)} "
                        "(cash + portfolio_value from /portfolio/balance)",
                        flush=True,
                    )
                except Exception as ex:
                    print(f"[warn] capital guard: could not read baseline balance: {ex}", file=sys.stderr)

            entry_effective = allow_entry_cli
            entry_block_event: str | None = None
            if allow_entry_cli and require_strategy_gate:
                ok_gate, gate_reason, _ = load_strategy_gate(strategy_gate_path)
                if not ok_gate:
                    entry_effective = False
                    entry_block_event = "strategy_gate"
                    now_l = time.time()
                    if now_l - last_strategy_gate_log >= 25.0:
                        print(f"[strategy-gate] {gate_reason}", flush=True)
                        last_strategy_gate_log = now_l

            if rediscover_btc:
                tickers = resolve_tickers(
                    single=None,
                    tickers_csv=None,
                    auto_markets=args.auto_markets,
                    public_base=public_base,
                    simulate=False,
                )
                if not tickers:
                    print(
                        "[warn] Auto-discovery returned no open BTC markets; retrying after interval.",
                        file=sys.stderr,
                    )
                    if metrics_sink is not None:
                        metrics_sink.flush()
                    time.sleep(args.interval)
                    continue
                tkey = tuple(tickers)
                if tkey != last_announced:
                    print(f"[markets] Active Kalshi BTC tickers: {tickers}")
                    last_announced = tkey
            for ticker in tickers:
                fee_mult = load_series_fee_multiplier(
                    series_from_market_ticker(ticker), public_base=public_base
                )
                print(
                    f"[fees] {ticker} series={series_from_market_ticker(ticker)} "
                    f"fee_multiplier={fee_mult} coeff={fee_coefficient}"
                )
                sp = args.state_dir / f"state_{safe_ticker_fs(ticker)}.json"
                run_cycle(
                    client=client,
                    sim=sim,
                    paper=paper,
                    ticker=ticker,
                    fair_yes=fair_yes,
                    fair_model=fair_model,
                    fair_model_hourly=fair_model_hourly,
                    btc_trend_model=btc_trend_model_obj,
                    contracts=contracts,
                    min_edge=min_edge,
                    stop_pct=stop_pct,
                    fee_coefficient=fee_coefficient,
                    fee_multiplier=fee_mult,
                    state_path=sp,
                    dry_run=args.dry_run,
                    allow_entry=entry_effective,
                    entry_block_event=entry_block_event,
                    metrics=metrics_sink,
                    max_book_spread=max_book_spread,
                    edge_spread_mult=edge_spread_mult,
                    btc_news_tilt=btc_news_tilt,
                    min_ev_surplus=min_ev_surplus,
                    ask_slip_dollars=ask_slip_dollars,
                    min_direction_confidence=min_direction_confidence,
                    adopt_live_position=bool(args.adopt_live_position),
                    min_seconds_before_close=min_seconds_before_close,
                )
                if metrics_sink is not None:
                    metrics_sink.flush()
            if paper is not None and metrics_sink is not None:
                nav = paper_net_asset_value(paper, public_base=public_base)
                pnl = nav - paper.starting_capital
                metrics_sink.update(
                    ts=time.time(),
                    ticker="_paper_",
                    cycle_event="paper_portfolio",
                    paper_balance=float(paper.balance),
                    paper_nav=float(nav),
                    paper_pnl=float(pnl),
                    paper_start=float(paper.starting_capital),
                    paper_position_tickers=list(paper.positions.keys()),
                )
                metrics_sink.flush()
                print(
                    f"[paper] NAV ${nav:.4f} (cash ${paper.balance:.4f}) "
                    f"PnL vs start ${pnl:.4f} (started ${paper.starting_capital})"
                )
            if max_capital_loss_pct > 0:
                trip = False
                halt_msg = ""
                if paper is not None:
                    nav_chk = paper_net_asset_value(paper, public_base=public_base)
                    base = paper.starting_capital
                    if base > 0:
                        floor = base * (Decimal("1") - max_capital_loss_pct)
                        if nav_chk <= floor:
                            trip = True
                            halt_msg = (
                                f"paper NAV {dollars_str(nav_chk)} <= floor {dollars_str(floor)} "
                                f"(starting_capital {dollars_str(base)}, "
                                f"max_loss {float(max_capital_loss_pct):.0%})"
                            )
                elif client is not None and live_equity_baseline is not None:
                    try:
                        eq = fetch_portfolio_equity_usd(client)
                        floor = live_equity_baseline * (Decimal("1") - max_capital_loss_pct)
                        if eq <= floor:
                            trip = True
                            halt_msg = (
                                f"live equity ${dollars_str(eq)} <= floor ${dollars_str(floor)} "
                                f"(baseline ${dollars_str(live_equity_baseline)}, "
                                f"max_loss {float(max_capital_loss_pct):.0%})"
                            )
                    except Exception as ex:
                        print(f"[warn] capital guard: {ex}", file=sys.stderr)
                if trip:
                    print(f"[halt] Max capital loss — {halt_msg}", flush=True)
                    if metrics_sink is not None:
                        halt_row = {
                            "ts": time.time(),
                            "ticker": "_halt_",
                            "cycle_event": "capital_halt",
                            "halt_reason": halt_msg,
                        }
                        with metrics_sink.path.open("a", encoding="utf-8") as hf:
                            hf.write(json.dumps(halt_row) + "\n")
                    sys.exit(3)
            if sim is not None:
                sim.tick_time(args.interval)
        except Exception as e:
            print(f"[error] {e}", file=sys.stderr)
        finally:
            if metrics_sink is not None:
                metrics_sink.flush()
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
