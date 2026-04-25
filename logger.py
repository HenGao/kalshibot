"""
logger.py -- Persistent trade log with outcome resolution and performance summary.

All trades are appended to trades.csv (configurable via config.LOG_FILE).
After each window closes, resolve_open_trades() fetches the final BTC price
and fills in outcome (WIN/LOSS) and P&L for any open positions.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

import config

# -- CSV schema ----------------------------------------------------------------
COLUMNS = [
    "timestamp",              # When the bot made this decision (UTC ISO)
    "series_ticker",          # KXBTC15M or KXBTC
    "window_open",            # UTC ISO -- when the trading window opened
    "window_close",           # UTC ISO -- when the trading window closes
    "btc_price_at_open",      # BTC spot price when we entered
    "kalshi_ticker",          # Full Kalshi market ticker
    "direction",              # YES or NO (which way we bet)
    "side",                   # "yes" or "no" (matches direction but lowercase)
    "num_contracts",          # How many $1-face contracts we bought
    "contract_price_cents",   # What we paid per contract (cents)
    "bet_size_dollars",       # Total dollars risked
    "signal_vol_direction",   # Volatility gate output
    "signal_vol_strength",
    "signal_momentum_direction",
    "signal_momentum_strength",
    "signal_crowd_direction",
    "signal_crowd_strength",
    "signal_funding_direction",
    "signal_funding_strength",
    "combined_direction",     # Final aggregated direction
    "combined_strength",      # Final aggregated strength
    "signal_reasons",         # Pipe-separated list of signal reason strings
    "paper_trade",            # True = paper trade, False = real money
    "outcome",                # WIN / LOSS -- filled in after window closes
    "btc_price_at_close",     # BTC price at settlement (filled in at resolution)
    "pnl_dollars",            # Profit or loss in dollars (filled in at resolution)
]

COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _ensure_csv() -> None:
    """Create the CSV file with a header row if it doesn't exist yet."""
    log_path = Path(config.LOG_FILE)
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()


def _fetch_btc_price_near(target_time: datetime, timeout: int = 10) -> float | None:
    """
    Fetch the BTC 1-minute close price nearest to target_time from Coinbase.
    Used to determine settlement price when resolving open trades.

    Coinbase candle format: [time_sec, low, high, open, close, volume]
    """
    try:
        ts_sec   = int(target_time.timestamp())
        start    = ts_sec - 90     # 1.5 minutes before
        end      = ts_sec + 90     # 1.5 minutes after

        resp = requests.get(
            COINBASE_CANDLES_URL,
            params={"granularity": 60, "start": start, "end": end},
            timeout=timeout,
        )
        resp.raise_for_status()
        candles = resp.json()

        if not candles:
            return None

        # Pick the candle whose timestamp is closest to the target
        best = min(candles, key=lambda c: abs(int(c[0]) - ts_sec))
        return float(best[4])    # index 4 = close price

    except Exception:
        return None


# -----------------------------------------------------------------------------
# Public logging functions
# -----------------------------------------------------------------------------

def log_trade(
    *,
    series_ticker:        str,
    window_open:          datetime,
    window_close:         datetime,
    btc_price:            float,
    kalshi_ticker:        str,
    direction:            str,
    side:                 str,
    num_contracts:        int,
    contract_price_cents: float,
    bet_size_dollars:     float,
    vol_signal:           dict[str, Any],
    momentum_signal:      dict[str, Any],
    crowd_signal:         dict[str, Any],
    funding_signal:       dict[str, Any],
    aggregated:           dict[str, Any],
    paper_trade:          bool,
) -> None:
    """
    Append a new trade row to trades.csv.

    outcome and pnl_dollars are left blank -- they will be filled in later
    by resolve_open_trades() once the window closes and we know the result.
    """
    _ensure_csv()

    reasons_str = " | ".join(aggregated.get("reasons", []))

    row = {
        "timestamp":                  datetime.now(timezone.utc).isoformat(),
        "series_ticker":              series_ticker,
        "window_open":                window_open.isoformat(),
        "window_close":               window_close.isoformat(),
        "btc_price_at_open":          f"{btc_price:.2f}",
        "kalshi_ticker":              kalshi_ticker,
        "direction":                  direction,
        "side":                       side,
        "num_contracts":              num_contracts,
        "contract_price_cents":       f"{contract_price_cents:.1f}",
        "bet_size_dollars":           f"{bet_size_dollars:.2f}",
        "signal_vol_direction":       vol_signal.get("direction", ""),
        "signal_vol_strength":        f"{vol_signal.get('strength', 0):.3f}",
        "signal_momentum_direction":  momentum_signal.get("direction", ""),
        "signal_momentum_strength":   f"{momentum_signal.get('strength', 0):.3f}",
        "signal_crowd_direction":     crowd_signal.get("direction", ""),
        "signal_crowd_strength":      f"{crowd_signal.get('strength', 0):.3f}",
        "signal_funding_direction":   funding_signal.get("direction", ""),
        "signal_funding_strength":    f"{funding_signal.get('strength', 0):.3f}",
        "combined_direction":         aggregated.get("direction", ""),
        "combined_strength":          f"{aggregated.get('strength', 0):.3f}",
        "signal_reasons":             reasons_str,
        "paper_trade":                str(paper_trade),
        "outcome":                    "",
        "btc_price_at_close":         "",
        "pnl_dollars":                "",
    }

    with Path(config.LOG_FILE).open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)

    mode = "[PAPER]" if paper_trade else "[LIVE] "
    print(
        f"  {mode} Logged trade: {direction} {num_contracts}× {kalshi_ticker} "
        f"@ {contract_price_cents:.0f}¢  (${bet_size_dollars:.2f})  "
        f"strength={aggregated.get('strength', 0):.2f}"
    )


def log_no_trade(reason: str, series_ticker: str = "") -> None:
    """Print a skipped-trade message (no CSV write -- saves log space)."""
    label = f"[{series_ticker}] " if series_ticker else ""
    print(f"  [SKIP] {label}No trade -- {reason}")


# -----------------------------------------------------------------------------
# Trade resolution
# -----------------------------------------------------------------------------

def resolve_open_trades() -> int:
    """
    Look for unresolved trades in trades.csv and fill in outcome + P&L.

    Resolution logic:
      1. Find rows where outcome is blank and window_close has passed
      2. Fetch the BTC price near that close time from Binance
      3. Compare btc_price_at_open vs btc_price_at_close
      4. If side=yes and BTC went up -> WIN; otherwise LOSS (and vice versa for no)
      5. P&L: win = (1.00 - price_paid) × contracts; loss = -price_paid × contracts

    Returns the number of trades resolved in this call.
    """
    log_path = Path(config.LOG_FILE)
    if not log_path.exists():
        return 0

    now_utc = datetime.now(timezone.utc)

    # Read all rows
    with log_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    changed       = False
    resolved_count = 0

    for row in rows:
        # Skip already-resolved or blank rows
        if row.get("outcome") or not row.get("kalshi_ticker") or not row.get("side"):
            continue

        # Parse window close time
        try:
            close_str  = row["window_close"]
            close_time = datetime.fromisoformat(close_str)
            if close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
        except (ValueError, KeyError):
            continue

        # Only resolve after the window has closed
        if now_utc <= close_time:
            continue

        # Parse entry data
        try:
            btc_open      = float(row["btc_price_at_open"])
            price_cents   = float(row["contract_price_cents"])
            num_contracts = int(row["num_contracts"])
            side          = row["side"].lower()
        except (ValueError, KeyError):
            continue

        if btc_open <= 0 or price_cents <= 0:
            continue

        # Fetch settlement price from Binance
        btc_close = _fetch_btc_price_near(close_time)
        if btc_close is None:
            continue   # Try again next loop

        # Determine win/loss based on BTC price movement
        btc_went_up = btc_close > btc_open
        if side == "yes":
            win = btc_went_up
        elif side == "no":
            win = not btc_went_up
        else:
            continue

        # Calculate P&L:
        #   Win:  receive $1 per contract, paid price_cents/100 -> net = (1 - price) × n
        #   Loss: receive $0 per contract, paid price_cents/100 -> net = -(price/100) × n
        price_dollars = price_cents / 100.0
        if win:
            pnl     = num_contracts * (1.0 - price_dollars)
            outcome = "WIN"
        else:
            pnl     = num_contracts * (-price_dollars)
            outcome = "LOSS"

        row["outcome"]            = outcome
        row["btc_price_at_close"] = f"{btc_close:.2f}"
        row["pnl_dollars"]        = f"{pnl:.2f}"
        changed        = True
        resolved_count += 1

        pnl_sign = "+" if pnl >= 0 else ""
        print(
            f"  [RESOLVE] {row['kalshi_ticker']}: {outcome}  "
            f"BTC {btc_open:,.0f} -> {btc_close:,.0f}  "
            f"P&L: {pnl_sign}${pnl:.2f}  ({num_contracts}× @ {price_cents:.0f}¢)"
        )

    # Write back if anything changed
    if changed:
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    return resolved_count


# -----------------------------------------------------------------------------
# Performance summary
# -----------------------------------------------------------------------------

def print_summary() -> None:
    """
    Print a performance summary from trades.csv.

    Shows overall stats (win rate, P&L, avg bet) plus a breakdown by
    signal combination so you can see which signal configs work best.
    """
    log_path = Path(config.LOG_FILE)
    if not log_path.exists():
        print("\n[SUMMARY] No trade log found. Run the bot first.")
        return

    with log_path.open("r", newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    resolved = [r for r in all_rows if r.get("outcome") in ("WIN", "LOSS")]
    pending  = [r for r in all_rows if not r.get("outcome") and r.get("kalshi_ticker")]

    if not resolved and not pending:
        print("\n[SUMMARY] No trades logged yet.")
        return

    print(f"\n{'=' * 60}")
    print(f"  TRADE SUMMARY  ({config.LOG_FILE})")
    print(f"{'=' * 60}")
    print(f"  Total logged    : {len(all_rows)}  ({len(pending)} pending resolution)")

    if not resolved:
        print("  No resolved trades yet -- come back after windows close.")
        print(f"{'=' * 60}")
        return

    def _print_group(trades: list[dict], label: str) -> None:
        if not trades:
            return

        wins   = [t for t in trades if t["outcome"] == "WIN"]
        losses = [t for t in trades if t["outcome"] == "LOSS"]
        pnls   = []
        bets   = []
        for t in trades:
            try:
                pnls.append(float(t["pnl_dollars"]))
                bets.append(float(t["bet_size_dollars"]))
            except (ValueError, KeyError):
                pass

        win_rate  = len(wins) / len(trades) * 100
        total_pnl = sum(pnls)
        avg_bet   = sum(bets) / len(bets) if bets else 0.0
        pnl_sign  = "+" if total_pnl >= 0 else ""

        print(f"\n  -- {label} --")
        print(f"  Resolved trades : {len(trades)}")
        print(f"  Win rate        : {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
        print(f"  Total P&L       : {pnl_sign}${total_pnl:.2f}")
        print(f"  Avg bet size    : ${avg_bet:.2f}")

        if pnls:
            best_idx  = pnls.index(max(pnls))
            worst_idx = pnls.index(min(pnls))
            print(f"  Best trade      : +${pnls[best_idx]:.2f}  "
                  f"({trades[best_idx].get('kalshi_ticker', '?')})")
            print(f"  Worst trade     :  ${pnls[worst_idx]:.2f}  "
                  f"({trades[worst_idx].get('kalshi_ticker', '?')})")

        # Win rate breakdown by signal configuration
        combos: dict[str, list[dict]] = {}
        for t in trades:
            key = (
                f"mom={t.get('signal_momentum_direction', '?'):>4}  "
                f"crowd={t.get('signal_crowd_direction', '?'):>4}  "
                f"fund={t.get('signal_funding_direction', '?'):>4}"
            )
            combos.setdefault(key, []).append(t)

        if len(combos) > 1:
            print(f"\n  Win rate by signal combo:")
            stats = [
                (combo, sum(1 for t in ts if t["outcome"] == "WIN") / len(ts), len(ts))
                for combo, ts in combos.items()
            ]
            stats.sort(key=lambda x: (-x[1], -x[2]))   # sort by win rate desc
            for combo, wr, n in stats[:8]:
                bar = "#" * int(wr * 20)
                print(f"    {combo}   {wr:.0%} {bar:<20} n={n}")

    # Split by paper vs live
    paper_trades = [r for r in resolved if r.get("paper_trade", "True") == "True"]
    live_trades  = [r for r in resolved if r.get("paper_trade", "True") == "False"]

    _print_group(paper_trades, "Paper Trades")
    _print_group(live_trades,  "Live Trades")

    print(f"\n{'=' * 60}")
