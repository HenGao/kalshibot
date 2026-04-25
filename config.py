"""
config.py — Bot configuration. Fill in your Kalshi credentials before running.

IMPORTANT: Kalshi uses API key + RSA private key authentication (NOT email/password).
To get your credentials:
  1. Log in to https://kalshi.com → Account → API Keys
  2. Create a new key, download the PEM private key file
  3. Fill in KALSHI_API_KEY_ID below and set KALSHI_PRIVATE_KEY_PATH to the PEM file path

For safe paper-trading testing, leave credentials blank — paper mode uses
no API calls for order placement.
"""

# ── Kalshi API credentials ────────────────────────────────────────────────────
KALSHI_API_KEY_ID       = "574c0ce4-3899-4eb1-8595-5f6706981935"  # Your Key ID from Kalshi dashboard
KALSHI_PRIVATE_KEY_PATH = r"C:\Users\hgao1\Downloads\claude.txt"  # Path to your downloaded RSA PEM file

# API host: switch to demo for risk-free testing with fake money
# Demo:  https://demo-api.kalshi.co
# Live:  https://api.elections.kalshi.com
KALSHI_HOST = "https://api.elections.kalshi.com"

# ── Trading mode ──────────────────────────────────────────────────────────────
# True  = paper mode: logs what the bot WOULD do, no real orders sent
# False = live mode:  real orders placed with real money (requires credentials)
PAPER_TRADE = True

# ── Markets to trade ──────────────────────────────────────────────────────────
TRADE_15M = True    # KXBTC15M — BTC direction over 15-minute windows
TRADE_1HR = True    # KXBTC    — BTC direction over 1-hour windows

# ── Risk management ───────────────────────────────────────────────────────────
BANKROLL_DOLLARS    = 50.0   # Your total trading bankroll in dollars
KELLY_FRACTION      = 0.25   # Fraction of full Kelly to use (0.25 = quarter-Kelly, conservative)
MIN_SIGNAL_STRENGTH = 0.40   # Combined signal strength must exceed this to place a trade
MAX_BET_PCT         = 0.08   # Hard cap: never risk more than 8% of bankroll per trade

# ── Bot loop timing ───────────────────────────────────────────────────────────
POLL_INTERVAL_SECONDS   = 60   # How often the main loop runs (seconds)
ENTRY_WINDOW_MIN        = 2    # Start entering trades when this many minutes remain before window open
ENTRY_WINDOW_MAX        = 5    # Stop entering trades after this many minutes before window open
MIN_SECONDS_BEFORE_CLOSE = 90  # Skip new entries when fewer than this many seconds remain in window

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = "trades.csv"
