# Paper "P1": live Kalshi quotes — KXBTC15M (15m) + KXBTC (hourly) + KXBTCD (daily).
# Optional headline tilt: BTC_NEWS_TILT (e.g. 0.06) or --btc-news-tilt below.
# Metrics + wallet paths match trade_dashboard when you set the same env vars.
# Usage (repo root):  powershell -File scripts/paper_p1.ps1
# Then in another terminal (same env):  python trade_dashboard.py
# Optional: run scripts/run_honest_eval.py then set REQUIRE_STRATEGY_GATE=1 and
# add --require-strategy-gate to edge_bot to block entries until the report passes.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$env:BTC_TREND_MODEL = ""
$env:BOT_METRICS_FILE = "data/paper_p1_metrics.jsonl"
$env:PAPER_WALLET = "data/paper_wallet_p1.json"
if (-not $env:BTC_NEWS_TILT) { $env:BTC_NEWS_TILT = "0.06" }

Write-Host "Starting paper P1 (15m + hourly + daily, news tilt env BTC_NEWS_TILT=$env:BTC_NEWS_TILT). Dashboard: scripts/dashboard_p1.ps1"
python -u edge_bot.py `
  --paper `
  --auto-markets all `
  --btc-news-tilt $env:BTC_NEWS_TILT `
  --fair-model data/models/fair_yes_logistic.json `
  --fair-model-hourly data/models/fair_yes_logistic.json `
  --metrics-file data/paper_p1_metrics.jsonl `
  --paper-wallet data/paper_wallet_p1.json `
  --paper-balance 50
