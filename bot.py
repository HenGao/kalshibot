#!/usr/bin/env python3
"""
bot.py -- Main loop for the Kalshi BTC 15-minute / 1-hour directional bot.

Usage:
  Paper mode (default, no credentials needed):
    python bot.py

  Summary only (print stats and exit):
    python bot.py --summary

  Live trading (requires Kalshi API credentials in config.py):
    Set PAPER_TRADE = False in config.py, then:
    python bot.py

The loop:
  Every 60 seconds:
    1. Resolve any open trades from prior windows
    2. Fetch fresh market data (BTC candles, funding rate, Kalshi prices)
    3. Print a status line
    4. For each active series (15M, 1HR):
       - Check if we're in the entry window (2-4 min before next window opens)
       - If yes: run signals. If strong signal: size and place a trade
       - If no:  log the reason we're skipping and wait
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import config
import data_feed
import signals as sig_module
import risk as risk_module
import logger
from bot_client import BotKalshiClient


# -----------------------------------------------------------------------------
# Window timing utilities
# -----------------------------------------------------------------------------

def next_15m_window_open(now: datetime) -> datetime:
    """
    Returns the UTC datetime when the next 15-minute Kalshi window opens.
    Windows open at :00, :15, :30, :45 of every hour.
    """
    minute      = now.minute
    next_mark   = ((minute // 15) + 1) * 15
    if next_mark >= 60:
        return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return now.replace(minute=next_mark, second=0, microsecond=0)


def next_1hr_window_open(now: datetime) -> datetime:
    """
    Returns the UTC datetime when the next 1-hour Kalshi window opens.
    Windows open at :00 of every hour.
    """
    return (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)


def minutes_until(target: datetime, now: datetime) -> float:
    """How many minutes until `target`, from `now`."""
    return (target - now).total_seconds() / 60.0


# -----------------------------------------------------------------------------
# Single-market trade attempt
# -----------------------------------------------------------------------------

def maybe_trade(
    *,
    client:        BotKalshiClient,
    series_ticker: str,
    feed_data:     dict[str, Any],
) -> bool:
    """
    Run the full signal pipeline for one market series.
    If signals are strong enough, calculate position size and place the trade.

    Returns True if a trade was placed (or paper-logged), False otherwise.
    """
    market = feed_data["markets"].get(series_ticker)
    if market is None:
        print(f"  [{series_ticker}] No market data available -- skipping.")
        return False

    seconds_to_close = market.get("seconds_to_close", 0)
    if seconds_to_close < config.MIN_SECONDS_BEFORE_CLOSE:
        print(
            f"  [{series_ticker}] Only {seconds_to_close:.0f}s until close "
            f"(threshold: {config.MIN_SECONDS_BEFORE_CLOSE}s) -- too late to enter."
        )
        return False

    # -- Run all signals -------------------------------------------------------
    aggregated, vol_sig, momentum, crowd, funding = sig_module.run_signals(
        candles      = feed_data["candles"],
        funding_rate = feed_data["funding_rate"],
        implied_prob = market["implied_prob"],
        macro_soon   = feed_data["macro_event_soon"],
        min_strength = config.MIN_SIGNAL_STRENGTH,
    )

    direction = aggregated["direction"]
    strength  = aggregated["strength"]

    print(f"  [{series_ticker}] Signal: {direction}  strength={strength:.2f}")
    for reason in aggregated.get("reasons", []):
        print(f"             ↳ {reason}")

    if direction == "NEUTRAL":
        reasons = " | ".join(aggregated.get("reasons", ["unknown"]))
        logger.log_no_trade(reasons, series_ticker)
        return False

    # -- Position sizing -------------------------------------------------------
    side = direction.lower()   # "yes" or "no"

    # Use the ask price for the side we're buying (what we'll actually pay)
    if side == "yes":
        contract_price_cents = market["yes_ask"]
    else:
        contract_price_cents = market["no_ask"]

    balance_cents   = client.get_balance_cents()
    balance_dollars = balance_cents / 100.0

    # Map signal strength -> estimated win probability (conservative blend)
    p_win = risk_module.signal_to_win_prob(strength, direction, market["yes_ask"])

    effective_bankroll = min(config.BANKROLL_DOLLARS, balance_dollars)
    num_contracts, bet_dollars = risk_module.kelly_size(
        bankroll          = effective_bankroll,
        win_probability   = p_win,
        yes_price_cents   = market["yes_ask"],
        kelly_fraction    = config.KELLY_FRACTION,
        max_bet_pct       = config.MAX_BET_PCT,
        side              = side,
    )

    if num_contracts < 1:
        logger.log_no_trade("Kelly criterion: 0 contracts (insufficient edge)", series_ticker)
        return False

    # Enforce balance check: don't spend more than we have
    num_contracts = risk_module.check_balance(
        num_contracts  = num_contracts,
        price_cents    = contract_price_cents,
        balance_cents  = balance_cents,
    )
    if num_contracts < 1:
        logger.log_no_trade("Insufficient balance for even 1 contract", series_ticker)
        return False

    bet_dollars = num_contracts * contract_price_cents / 100.0

    print(
        f"  [{series_ticker}] Sizing: {num_contracts} × {side.upper()} @ "
        f"{contract_price_cents:.0f}¢  = ${bet_dollars:.2f}  "
        f"(p_win={p_win:.0%}, bankroll=${effective_bankroll:.0f})"
    )

    # -- Place the order -------------------------------------------------------
    limit_price = max(1, min(99, int(contract_price_cents)))
    client.place_order(
        ticker             = market["ticker"],
        side               = side,
        num_contracts      = num_contracts,
        limit_price_cents  = limit_price,
    )

    # -- Log the trade to CSV --------------------------------------------------
    btc_price = float(feed_data["candles"]["close"].iloc[-1])
    logger.log_trade(
        series_ticker        = series_ticker,
        window_open          = market["open_time"],
        window_close         = market["close_time"],
        btc_price            = btc_price,
        kalshi_ticker        = market["ticker"],
        direction            = direction,
        side                 = side,
        num_contracts        = num_contracts,
        contract_price_cents = contract_price_cents,
        bet_size_dollars     = bet_dollars,
        vol_signal           = vol_sig,
        momentum_signal      = momentum,
        crowd_signal         = crowd,
        funding_signal       = funding,
        aggregated           = aggregated,
        paper_trade          = config.PAPER_TRADE,
    )
    return True


# -----------------------------------------------------------------------------
# Main bot loop
# -----------------------------------------------------------------------------

def run() -> None:
    """Main trading loop. Runs forever until interrupted with Ctrl+C."""

    # -- Startup banner --------------------------------------------------------
    mode_tag = "[PAPER MODE -- no real orders]" if config.PAPER_TRADE else "[LIVE TRADING !!️]"
    markets_tag = " + ".join(
        [s for s, flag in [("KXBTC15M", config.TRADE_15M), ("KXBTC", config.TRADE_1HR)] if flag]
    )
    print(f"\n{'=' * 62}")
    print(f"  Kalshi BTC Directional Bot")
    print(f"  Mode     : {mode_tag}")
    print(f"  Markets  : {markets_tag or 'NONE ENABLED'}")
    print(f"  Bankroll : ${config.BANKROLL_DOLLARS:.2f}  "
          f"Kelly: {config.KELLY_FRACTION}  "
          f"MaxBet: {config.MAX_BET_PCT:.0%}")
    print(f"  Min signal strength: {config.MIN_SIGNAL_STRENGTH}")
    print(f"  Entry window: {config.ENTRY_WINDOW_MIN}-{config.ENTRY_WINDOW_MAX} min before window open")
    print(f"{'=' * 62}\n")

    if not config.PAPER_TRADE:
        print("  *** LIVE TRADING ENABLED -- REAL MONEY WILL BE SPENT ***")
        print("  Press Ctrl+C in the next 5 seconds to abort...")
        time.sleep(5)
        print("  Proceeding with live trading.\n")

    # Determine which series to watch
    series: list[str] = []
    if config.TRADE_15M:
        series.append("KXBTC15M")
    if config.TRADE_1HR:
        series.append("KXBTC")

    if not series:
        print("ERROR: Both TRADE_15M and TRADE_1HR are False in config.py. Nothing to do.")
        sys.exit(1)

    client = BotKalshiClient()

    # Track which ticker we already traded this window (avoid double-entry)
    last_traded_ticker: dict[str, str] = {}
    loop_count = 0

    while True:
        loop_count += 1
        now_utc = datetime.now(timezone.utc)
        now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

        print(f"\n{'-' * 62}")
        print(f"  Loop #{loop_count}  |  {now_str}")
        print(f"{'-' * 62}")

        # -- Step 1: Resolve open trades from prior windows --------------------
        try:
            resolved = logger.resolve_open_trades()
            if resolved:
                print(f"  Resolved {resolved} trade(s).")
        except Exception as exc:
            print(f"  [WARNING] resolve_open_trades error: {exc}")

        # -- Step 2: Fetch fresh market data -----------------------------------
        print("  Fetching market data...")
        try:
            feed_data = data_feed.fetch_all(series)
        except RuntimeError as exc:
            print(f"  [ERROR] Data fetch failed: {exc}")
            print(f"  Retrying in {config.POLL_INTERVAL_SECONDS}s...")
            time.sleep(config.POLL_INTERVAL_SECONDS)
            continue

        # -- Step 3: Status line -----------------------------------------------
        btc_price    = float(feed_data["candles"]["close"].iloc[-1])
        funding_rate = feed_data["funding_rate"]
        macro_tag    = "!!️ MACRO BLACKOUT" if feed_data["macro_event_soon"] else "clear"

        print(f"  BTC spot : ${btc_price:>10,.2f}")
        print(f"  Funding  : {funding_rate:>+.6f}  ({funding_rate * 100:+.4f}%/8h)")
        print(f"  Macro    : {macro_tag}")

        for s in series:
            mkt = feed_data["markets"].get(s)
            if mkt:
                mins = mkt["seconds_to_close"] / 60
                print(
                    f"  {s:<10}: {mkt['ticker']:<40}  "
                    f"Yes={mkt['yes_ask']:.0f}¢  No={mkt['no_ask']:.0f}¢  "
                    f"implied={mkt['implied_prob']:.0%}  "
                    f"closes in {mins:.1f}min"
                )

        # -- Step 4: Check entry timing for each series ------------------------
        next_15m = next_15m_window_open(now_utc)
        next_1hr = next_1hr_window_open(now_utc)

        for s in series:
            mkt = feed_data["markets"].get(s)
            if mkt is None:
                continue

            ticker = mkt["ticker"]

            if s == "KXBTC15M":
                mins_to_next = minutes_until(next_15m, now_utc)
            else:
                mins_to_next = minutes_until(next_1hr, now_utc)

            in_entry_window = config.ENTRY_WINDOW_MIN <= mins_to_next <= config.ENTRY_WINDOW_MAX
            already_traded  = (last_traded_ticker.get(s) == ticker)

            print(
                f"\n  [{s}] {mins_to_next:.1f}min to next window open  "
                f"| entry window: {config.ENTRY_WINDOW_MIN}-{config.ENTRY_WINDOW_MAX}min"
                f"{'  [ALREADY TRADED]' if already_traded else ''}"
            )

            if already_traded:
                print(f"  [{s}] Already traded {ticker} this window -- waiting.")
                continue

            if not in_entry_window:
                if mins_to_next > config.ENTRY_WINDOW_MAX:
                    print(f"  [{s}] Too early -- {mins_to_next:.1f}min until window (need <={config.ENTRY_WINDOW_MAX}min).")
                else:
                    print(f"  [{s}] Too late -- only {mins_to_next:.1f}min left (need >={config.ENTRY_WINDOW_MIN}min).")
                continue

            # We're in the entry window -- run signals and possibly trade
            print(f"  [{s}] *** IN ENTRY WINDOW -- running signals ***")
            try:
                traded = maybe_trade(
                    client        = client,
                    series_ticker = s,
                    feed_data     = feed_data,
                )
                if traded:
                    last_traded_ticker[s] = ticker
            except Exception as exc:
                print(f"  [{s}] [ERROR] Trade attempt failed: {exc}")

        # -- Sleep until next loop ---------------------------------------------
        print(f"\n  Sleeping {config.POLL_INTERVAL_SECONDS}s...", end="", flush=True)
        time.sleep(config.POLL_INTERVAL_SECONDS)
        print(" done.")


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kalshi BTC 15-min/1-hr directional trading bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bot.py              # Run in paper mode (default)
  python bot.py --summary    # Print performance stats and exit
        """,
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print trade log performance summary and exit",
    )
    args = parser.parse_args()

    if args.summary:
        logger.print_summary()
        return

    try:
        run()
    except KeyboardInterrupt:
        print("\n\nBot stopped (Ctrl+C).")
        print("Printing final summary...\n")
        logger.print_summary()
        sys.exit(0)


if __name__ == "__main__":
    main()
