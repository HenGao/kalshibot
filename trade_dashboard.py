#!/usr/bin/env python3
"""
Local web dashboard for bot trade status.

Reads:
  - data/live_metrics.jsonl (written by edge_bot.py)
  - data/bot_state/state_*.json (per-ticker positions)

Run (from repo root):
  pip install flask
  python trade_dashboard.py

Open: http://127.0.0.1:8765/

Set PORT=9000 to change the port.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "data" / "live_metrics.jsonl"
STATE_DIR = ROOT / "data" / "bot_state"
PAPER_WALLET_PATH = Path(os.environ.get("PAPER_WALLET", str(ROOT / "data" / "paper_wallet.json")))

# Max bytes to read from end of JSONL (avoid huge files).
TAIL_BYTES = 2_000_000
MAX_JSONL_LINES = 8000
RECENT_LINES = 80

app = Flask(__name__)


def tail_jsonl_lines(path: Path, max_lines: int) -> list[dict]:
    if not path.is_file():
        return []
    raw = path.read_bytes()
    if not raw:
        return []
    chunk = raw[-TAIL_BYTES:] if len(raw) > TAIL_BYTES else raw
    text = chunk.decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows = []
    for line in lines[-max_lines:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load_paper_wallet() -> dict | None:
    if not PAPER_WALLET_PATH.is_file():
        return None
    try:
        return json.loads(PAPER_WALLET_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_bot_states() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not STATE_DIR.is_dir():
        return out
    for p in sorted(STATE_DIR.glob("state_*.json")):
        try:
            key = p.stem.replace("state_", "", 1)
            out[key] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def build_status() -> dict:
    rows = tail_jsonl_lines(METRICS_PATH, MAX_JSONL_LINES)
    now = time.time()
    latest_by_ticker: dict[str, dict] = {}
    for r in rows:
        t = str(r.get("ticker") or "_")
        if t == "_paper_":
            continue
        latest_by_ticker[t] = r

    latest_global = None
    if rows:
        latest_global = max(rows, key=lambda x: float(x.get("ts") or 0))

    latest_paper = None
    for r in reversed(rows):
        if r.get("cycle_event") == "paper_portfolio":
            latest_paper = r
            break

    return {
        "server_time": now,
        "metrics_path": str(METRICS_PATH),
        "metrics_exists": METRICS_PATH.is_file(),
        "metrics_size_bytes": METRICS_PATH.stat().st_size if METRICS_PATH.is_file() else 0,
        "metrics_line_count_est": len(rows),
        "latest_global": latest_global,
        "latest_paper": latest_paper,
        "latest_by_ticker": latest_by_ticker,
        "recent": rows[-RECENT_LINES:],
        "bot_states": load_bot_states(),
        "paper_wallet_path": str(PAPER_WALLET_PATH),
        "paper_wallet": load_paper_wallet(),
    }


@app.route("/api/status")
def api_status():
    return jsonify(build_status())


@app.route("/api/stream")
def api_stream():
    """Optional SSE stream (1 Hz) for smoother updates without polling config."""

    def gen():
        while True:
            payload = json.dumps(build_status())
            yield f"data: {payload}\n\n"
            time.sleep(float(request.args.get("interval", "1")))

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Kalshi bot — trade status</title>
  <style>
    :root {
      --bg: #0f1419;
      --card: #1a2332;
      --text: #e7ecf3;
      --muted: #8b9cb3;
      --accent: #3d9cf5;
      --yes: #2ecc71;
      --no: #e74c3c;
      --warn: #f39c12;
    }
    * { box-sizing: border-box; }
    body {
      font-family: ui-sans-serif, system-ui, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 1rem 1.25rem 2rem;
      line-height: 1.45;
    }
    h1 { font-size: 1.25rem; font-weight: 600; margin: 0 0 0.5rem; }
    .sub { color: var(--muted); font-size: 0.85rem; margin-bottom: 1.25rem; }
    .grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); }
    .card {
      background: var(--card);
      border-radius: 10px;
      padding: 1rem 1.1rem;
      border: 1px solid #2a3545;
    }
    .card h2 { font-size: 0.75rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 0 0 .6rem; }
    .big { font-size: 1.5rem; font-weight: 700; font-variant-numeric: tabular-nums; }
    .row { display: flex; justify-content: space-between; gap: .5rem; font-size: 0.9rem; margin: .35rem 0; }
    .row span:last-child { font-variant-numeric: tabular-nums; color: var(--muted); }
    .event { font-size: 0.85rem; padding: .2rem .45rem; border-radius: 4px; display: inline-block; margin-top: .4rem;}
    .event-poll { background: #2a3545; color: var(--muted); }
    .event-filled { background: #1e3d2f; color: var(--yes); }
    .event-hold { background: #3d3420; color: var(--warn); }
    .event-stop { background: #3d2020; color: var(--no); }
    .event-edge { background: #2a3545; color: var(--muted); }
    table { width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-top: .75rem; }
    th, td { text-align: left; padding: .35rem .5rem; border-bottom: 1px solid #2a3545; }
    th { color: var(--muted); font-weight: 500; }
    .mono { font-family: ui-monospace, Consolas, monospace; font-size: 0.78rem; }
    .err { color: var(--no); }
    a { color: var(--accent); }
  </style>
</head>
<body>
  <h1>Kalshi bot — live status</h1>
  <p class="sub">Polling <code class="mono">/api/status</code> — run <code class="mono">edge_bot.py</code> with metrics enabled.</p>
  <div id="banner" class="sub"></div>

  <div class="grid" id="summary"></div>
  <div class="grid" id="tickers" style="margin-top:1rem;"></div>

  <div class="card" style="margin-top:1rem;">
    <h2>Recent events</h2>
    <div style="overflow-x:auto;">
      <table id="recent"><thead><tr>
        <th>Time</th><th>Ticker</th><th>Event</th><th>Equity</th><th>YES mid</th><th>Pos</th><th>Fair YES</th>
      </tr></thead><tbody></tbody></table>
    </div>
  </div>

  <div class="card" style="margin-top:1rem;">
    <h2>Paper wallet (virtual positions)</h2>
    <pre id="paper" class="mono" style="margin:0;white-space:pre-wrap;word-break:break-all;"></pre>
  </div>

  <div class="card" style="margin-top:1rem;">
    <h2>On-disk bot state (data/bot_state)</h2>
    <pre id="states" class="mono" style="margin:0;white-space:pre-wrap;word-break:break-all;"></pre>
  </div>

<script>
const fmtUsd = (n) => (n == null ? "—" : Number(n).toFixed(4));
const fmtTs = (ts) => {
  if (ts == null) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleString();
};
const evClass = (ev) => {
  if (!ev) return "event event-poll";
  if (ev.includes("fill")) return "event event-filled";
  if (ev.includes("hold") || ev.includes("nonbot")) return "event event-hold";
  if (ev.includes("stop")) return "event event-stop";
  if (ev.includes("edge") || ev.includes("entry")) return "event event-edge";
  return "event event-poll";
};

async function refresh() {
  let data;
  try {
    const r = await fetch("/api/status", { cache: "no-store" });
    data = await r.json();
  } catch (e) {
    document.getElementById("banner").innerHTML = '<span class="err">Cannot reach API.</span>';
    return;
  }
  document.getElementById("banner").innerHTML =
    `Metrics file: <span class="mono">${data.metrics_path}</span> — ` +
    (data.metrics_exists ? `${data.metrics_size_bytes} bytes, ~${data.metrics_line_count_est} lines loaded` : '<span class="err">missing (start the bot with metrics on)</span>') +
    (data.paper_wallet_path ? ` — paper wallet: <span class="mono">${data.paper_wallet_path}</span>` : "");

  const sum = document.getElementById("summary");
  const g = data.latest_global;
  const lp = data.latest_paper;
  const pnlColor = (x) => (x == null ? "var(--text)" : (x >= 0 ? "var(--yes)" : "var(--no)"));
  sum.innerHTML = `
    ${lp ? `
    <div class="card" style="border:1px solid var(--warn);">
      <h2>Paper profit (total bankroll)</h2>
      <p style="margin:0 0 .6rem;font-size:.8rem;color:var(--muted);">NAV = cash + open contracts valued at the bid you would hit to exit. This is the number to watch vs your starting stake — not the per-market &quot;equity&quot; column, which marks bid vs entry and looks like a loss right after you buy at the ask.</p>
      <div class="big" style="color:${pnlColor(lp.paper_pnl)}">${fmtUsd(lp.paper_pnl)} <span style="font-size:.9rem;color:var(--muted)">PnL vs ${fmtUsd(lp.paper_start)} start</span></div>
      <div class="row"><span>Net asset value</span><span>${fmtUsd(lp.paper_nav)}</span></div>
      <div class="row"><span>Cash</span><span>${fmtUsd(lp.paper_balance)}</span></div>
      <div class="row"><span>Open positions</span><span style="text-align:right;max-width:65%">${(lp.paper_position_tickers || []).join(", ") || "—"}</span></div>
      <div class="row"><span>Time</span><span>${fmtTs(lp.ts)}</span></div>
    </div>` : ""}
    <div class="card">
      <h2>Latest per-market metric row</h2>
      ${g ? `
        ${g.paper_mode && !lp ? `<p style="margin:0 0 .5rem;font-size:.8rem;color:var(--warn);">Waiting for paper portfolio snapshot — update the bot, or use cash row below.</p>` : ""}
        <p style="margin:0 0 .5rem;font-size:.75rem;color:var(--muted);">Per-ticker &quot;equity&quot; is a rough mark (bid vs entry); it is not your full $50 wallet.</p>
        <div class="big">${fmtUsd(g.equity)} <span style="font-size:.9rem;color:var(--muted)">ledger est.</span></div>
        <div class="row"><span>Realized</span><span>${fmtUsd(g.realized_pnl)}</span></div>
        <div class="row"><span>Unrealized</span><span>${fmtUsd(g.unrealized_pnl)}</span></div>
        ${g.paper_balance != null ? `<div class="row"><span>Cash (one market)</span><span>${fmtUsd(g.paper_balance)}</span></div>` : ""}
        <div class="row"><span>Time</span><span>${fmtTs(g.ts)}</span></div>
        <span class="${evClass(g.cycle_event)}">${g.cycle_event || "poll"}</span>
      ` : '<p class="muted">No metrics yet.</p>'}
    </div>
    <div class="card">
      <h2>Refresh</h2>
      <p style="margin:0;color:var(--muted);font-size:.9rem;">Page auto-refreshes every 1s.</p>
      <p style="margin:.5rem 0 0;font-size:.8rem;color:var(--muted);">Paper: <code class="mono">python edge_bot.py --paper --paper-balance 50</code></p>
      <p style="margin:.35rem 0 0;font-size:.8rem;color:var(--muted);">Sim: <code class="mono">python edge_bot.py --simulate --interval 1 --min-edge 0.001</code></p>
    </div>`;

  const tickers = document.getElementById("tickers");
  const entries = Object.entries(data.latest_by_ticker || {}).sort((a,b) => a[0].localeCompare(b[0]));
  tickers.innerHTML = entries.map(([tk, r]) => `
    <div class="card">
      <h2 class="mono">${tk}</h2>
      <div class="row"><span>YES bid / ask</span><span>${fmtUsd(r.best_yes_bid)} / ${fmtUsd(r.best_yes_ask)}</span></div>
      <div class="row"><span>NO bid / ask</span><span>${fmtUsd(r.best_no_bid)} / ${fmtUsd(r.best_no_ask)}</span></div>
      <div class="row"><span>YES mid</span><span>${fmtUsd(r.yes_mid)}</span></div>
      <div class="row"><span>Fair YES (model)</span><span>${r.fair_yes != null ? Number(r.fair_yes).toFixed(4) : "—"}</span></div>
      <div class="row"><span>EV YES / NO</span><span>${r.ev_yes != null ? fmtUsd(r.ev_yes) : "—"} / ${r.ev_no != null ? fmtUsd(r.ev_no) : "—"}</span></div>
      <div class="row"><span>Position</span><span>${r.position_fp != null ? r.position_fp : "—"}</span></div>
      ${r.paper_balance != null ? `<div class="row"><span>Paper USD</span><span>${fmtUsd(r.paper_balance)}</span></div>` : ""}
      <div class="row"><span>Bot side / contracts</span><span>${r.bot_side || "—"} ${r.bot_contracts != null ? r.bot_contracts : ""}</span></div>
      <div class="row"><span>Updated</span><span>${fmtTs(r.ts)}</span></div>
      <span class="${evClass(r.cycle_event)}">${r.cycle_event || "poll"}</span>
    </div>
  `).join("") || '<div class="card"><p>No per-ticker rows yet.</p></div>';

  const tbody = document.querySelector("#recent tbody");
  const rev = (data.recent || []).slice().reverse();
  tbody.innerHTML = rev.map(r => `<tr>
    <td>${fmtTs(r.ts)}</td>
    <td class="mono">${(r.ticker || "").slice(0, 28)}</td>
    <td>${r.cycle_event || ""}</td>
    <td>${r.cycle_event === "paper_portfolio" ? fmtUsd(r.paper_nav) : fmtUsd(r.equity)}</td>
    <td>${fmtUsd(r.yes_mid)}</td>
    <td>${r.position_fp != null ? r.position_fp : ""}</td>
    <td>${r.fair_yes != null ? Number(r.fair_yes).toFixed(3) : ""}</td>
  </tr>`).join("");

  document.getElementById("paper").textContent = data.paper_wallet
    ? JSON.stringify(data.paper_wallet, null, 2)
    : "(no paper_wallet.json yet — start edge_bot with --paper)";

  document.getElementById("states").textContent =
    Object.keys(data.bot_states || {}).length
      ? JSON.stringify(data.bot_states, null, 2)
      : "(no state_*.json files — flat, sim-only, or paper-only in wallet)";
}

refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html; charset=utf-8")


def main() -> None:
    port = int(os.environ.get("PORT", "8765"))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"Dashboard: http://{host}:{port}/")
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
