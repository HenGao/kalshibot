# Kalshi BTC Directional Bot

A paper/live trading bot for Kalshi's KXBTC15M (15-minute) and KXBTC (1-hour)
BTC Up/Down binary markets. Built for a $50 bankroll with conservative sizing.

---

## How It Works

Every 60 seconds the bot:
1. Resolves any past trades that have now settled (fills in WIN/LOSS + P&L)
2. Fetches fresh data: BTC candles from Binance, the funding rate, Kalshi prices
3. For each active market series, checks if it's 2–4 minutes before the next window opens
4. If yes, runs four signals and decides whether to trade:
   - **Volatility gate** — skips low-volatility periods (no movement = no edge)
   - **Multi-timeframe momentum** — 5min, 15min, 60min trends must agree (2 of 3)
   - **Crowd sentiment fade** — fades extreme one-sided Kalshi positioning
   - **Funding rate** — uses perp funding as a contrarian lean
5. If signals agree and are strong enough (> 0.55), sizes the position with
   quarter-Kelly criterion and places the order

---

## File Overview

| File | Purpose |
|------|---------|
| `config.py` | All settings: credentials, PAPER_TRADE flag, bankroll, thresholds |
| `data_feed.py` | Fetches BTC candles, funding rate, macro events, Kalshi prices |
| `signals.py` | All four signal functions + the aggregation logic |
| `bot_client.py` | Kalshi API wrapper: market data, order placement, balance |
| `risk.py` | Fractional Kelly sizing + balance checks |
| `logger.py` | Writes to `trades.csv`, resolves outcomes, prints summary |
| `bot.py` | Main loop — ties everything together |

---

## Installation

```bash
# 1. Clone or download this project, then navigate to it
cd kalshibot

# 2. Install Python dependencies
pip install -r requirements.txt
```

Requirements: Python 3.10+, internet connection.

---

## Running in Paper Mode (default — no money at risk)

Paper mode is **on by default** (`PAPER_TRADE = True` in `config.py`).  
No Kalshi credentials are needed. The bot uses real Kalshi market prices but
never sends any orders.

```bash
python bot.py
```

You'll see output like:
```
══════════════════════════════════════════════════════════════
  Kalshi BTC Directional Bot
  Mode     : [PAPER MODE — no real orders]
  Markets  : KXBTC15M + KXBTC
  Bankroll : $50.00  Kelly: 0.25  MaxBet: 8%
══════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────
  Loop #1  |  2025-05-15 14:32:00 UTC
──────────────────────────────────────────────────────────────
  BTC spot :  $65,432.10
  Funding  : +0.000102  (+0.0102%/8h)
  Macro    : clear
  KXBTC15M : KXBTC15M-25MAY15-T65500  Yes=47¢  No=53¢  implied=47%  closes in 8.3min
  ...
  [KXBTC15M] Too early — 11.7min until window (need ≤4min).
```

When the bot decides to trade (paper):
```
  [KXBTC15M] *** IN ENTRY WINDOW — running signals ***
  [KXBTC15M] Signal: NO  strength=0.62
             ↳ volatility OK (24.3% ann vol, above 30th pct)
             ↳ momentum NO (2/3 agree): 5m=NO, 15m=NO, 60m=YES
             ↳ crowd overextended YES (74% implied) — fading
             ↳ perp funding neutral (+0.000102)
  [KXBTC15M] Sizing: 3 × NO @ 53¢ = $1.59  (p_win=57%, bankroll=$50)
  [PAPER ORDER] NO 3 × KXBTC15M-25MAY15-T65500 @ 53¢  ($1.59 total)
  [PAPER] Logged trade: NO 3× KXBTC15M-... @ 53¢  ($1.59)  strength=0.62
```

---

## Reading the Trade Log

All trades are written to `trades.csv`. Open it in Excel or view it in the terminal:

```bash
python bot.py --summary
```

Key columns:
| Column | Meaning |
|--------|---------|
| `direction` | YES (bet BTC goes up) or NO (bet BTC goes down) |
| `num_contracts` | How many $1-face contracts were bought |
| `contract_price_cents` | What you paid per contract (e.g. 47¢) |
| `bet_size_dollars` | Total dollars at risk for this trade |
| `outcome` | WIN or LOSS — filled in after the window closes |
| `pnl_dollars` | Your profit (positive) or loss (negative) |
| `paper_trade` | True = simulated, False = real money |

**How P&L works:**
- You buy 3 NO contracts at 53¢ each = $1.59 total cost
- If BTC goes down (you win): each contract pays $1.00, so you get $3.00 back → P&L = +$1.41
- If BTC goes up (you lose): contracts expire at $0.00 → P&L = -$1.59

---

## Printing a Performance Summary

```bash
python bot.py --summary
```

Shows win rate, total P&L, average bet size, and a breakdown of which signal
combinations have performed best.

---

## Flipping to Live Trading

> **Warning:** This uses real money. Start with paper mode and run it for at
> least a week to check that the bot behaves as expected before going live.

**Step 1 — Get Kalshi API credentials:**
1. Log in at https://kalshi.com
2. Go to Account → API Keys → Create New Key
3. Download the RSA private key `.pem` file
4. Copy the Key ID shown on the page

**Step 2 — Configure `config.py`:**
```python
KALSHI_API_KEY_ID       = "your-key-id-here"
KALSHI_PRIVATE_KEY_PATH = "kalshi_private_key.pem"   # path to your downloaded PEM file
PAPER_TRADE             = False                       # ← this enables live orders
BANKROLL_DOLLARS        = 50.0                        # your actual bankroll
```

**Step 3 — Run:**
```bash
python bot.py
```

The bot will print a 5-second warning and count down before starting live trading.

---

## Adjusting Risk Settings (config.py)

| Setting | Default | What it does |
|---------|---------|-------------|
| `BANKROLL_DOLLARS` | 50 | Total bankroll the bot sizes off of |
| `KELLY_FRACTION` | 0.25 | 0.25 = quarter-Kelly (conservative). Never set > 0.5 |
| `MIN_SIGNAL_STRENGTH` | 0.55 | Raise to 0.65+ to trade less but only on stronger signals |
| `MAX_BET_PCT` | 0.08 | Max 8% of bankroll per trade. Lower = safer |
| `ENTRY_WINDOW_MIN/MAX` | 2–4 | Minutes before window open to enter |

---

## Which Markets Are Traded

Both are enabled by default. Set to `False` in `config.py` to disable:
```python
TRADE_15M = True    # KXBTC15M — 15-minute windows
TRADE_1HR = True    # KXBTC    — 1-hour windows
```

**15-minute markets** settle every :00, :15, :30, :45. More frequent = more
trades but also more noise and spread cost per day.

**1-hour markets** settle at the top of each hour. Fewer trades but each signal
has more time to play out.

---

## Updating the Macro Event Calendar

The bot skips trading 30 minutes before and 10 minutes after major USD economic
events (FOMC, CPI, NFP) to avoid getting caught in the spike.

The event timestamps are hardcoded in `data_feed.py` at `_MACRO_EVENTS_UTC`.
Update this list at the start of each month using scheduled dates from
[Forex Factory](https://www.forexfactory.com) (free, no account needed).

---

## Troubleshooting

**"No open markets found for series KXBTC15M"**  
Kalshi markets may be briefly unavailable between settlement and the next
window opening. The bot will retry on the next loop.

**"RSA private key not found"**  
Check that `KALSHI_PRIVATE_KEY_PATH` in config.py points to the correct path
for your downloaded `.pem` file.

**Bot is never trading**  
- Check that the signal strength threshold (`MIN_SIGNAL_STRENGTH = 0.55`) isn't too high
- Run `python bot.py --summary` to see if any trades have been logged
- The volatility gate may be filtering everything out — this is intentional during quiet periods

---

## Disclaimer

This software is for educational purposes. Binary prediction markets carry
significant risk. Past performance does not guarantee future results. Never
trade more than you can afford to lose entirely.
