# Local dashboard for paper P1 (must match paper_p1.ps1 paths).
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:BOT_METRICS_FILE = "data/paper_p1_metrics.jsonl"
$env:PAPER_WALLET = "data/paper_wallet_p1.json"
python trade_dashboard.py
