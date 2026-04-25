# Local dashboard for legacy profile (must match paper_legacy_win.ps1 paths).
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:BOT_METRICS_FILE = "data/paper_legacy_metrics.jsonl"
$env:PAPER_WALLET = "data/paper_wallet_legacy.json"
python trade_dashboard.py
