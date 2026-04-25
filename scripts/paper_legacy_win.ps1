# Legacy paper profile (closer to early loose settings)
# Usage: powershell -File scripts/paper_legacy_win.ps1
# Dashboard: powershell -File scripts/dashboard_legacy_win.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$env:BTC_TREND_MODEL = ""
$env:BOT_METRICS_FILE = "data/paper_legacy_metrics.jsonl"
$env:PAPER_WALLET = "data/paper_wallet_legacy.json"

Write-Host "Starting legacy paper profile (15m + hourly, loose entry filters)."
python -u edge_bot.py `
  --paper `
  --paper-reset `
  --auto-markets both `
  --fair-model data/models/fair_yes_logistic.json `
  --fair-model-hourly data/models/fair_yes_logistic.json `
  --metrics-file data/paper_legacy_metrics.jsonl `
  --paper-wallet data/paper_wallet_legacy.json `
  --paper-balance 50 `
  --min-edge 0.02 `
  --max-spread 0 `
  --edge-spread-mult 0 `
  --min-ev-surplus 0 `
  --ask-slip 0 `
  --min-direction-confidence 0 `
  --min-seconds-before-close 0 `
  --btc-news-tilt 0
