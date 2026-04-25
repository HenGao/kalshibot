#!/usr/bin/env python3
"""
Local web dashboard for bot trade status.

Reads:
  - JSONL from edge_bot.py (--metrics-file or env BOT_METRICS_FILE, default data/live_metrics.jsonl)
  - Bot state: BOT_STATE_DIR (default data/bot_state)
  - Paper wallet: PAPER_WALLET (default data/paper_wallet.json)

Run (from repo root):
  pip install flask
  python trade_dashboard.py

Open: http://127.0.0.1:8765/

Set PORT=9000 to change the port.

Paper P1 (15m + hourly, same metrics file as bot):
  set BOT_METRICS_FILE=data/paper_p1_metrics.jsonl
  set PAPER_WALLET=data/paper_wallet_p1.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

ROOT = Path(__file__).resolve().parent


def _resolve_path(env_key: str, default: str) -> Path:
    raw = os.environ.get(env_key, default)
    p = Path(raw)
    return p if p.is_absolute() else (ROOT / p)


def metrics_path() -> Path:
    return _resolve_path("BOT_METRICS_FILE", "data/live_metrics.jsonl")


def state_dir() -> Path:
    return _resolve_path("BOT_STATE_DIR", "data/bot_state")


def paper_wallet_path() -> Path:
    return _resolve_path("PAPER_WALLET", "data/paper_wallet.json")


# Max bytes to read from end of JSONL (avoid huge files).
TAIL_BYTES = 2_000_000
MAX_JSONL_LINES = 8000
RECENT_LINES = 80

app = Flask(__name__)


@app.after_request
def _no_cache_api(resp: Response) -> Response:
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


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
    pw = paper_wallet_path()
    if not pw.is_file():
        return None
    try:
        return json.loads(pw.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def live_positions_from_paper_wallet(wallet: dict | None) -> list[dict[str, Any]]:
    """Open legs from paper_wallet.json: positions + bot_states entry."""
    if not wallet:
        return []
    positions = wallet.get("positions") or {}
    states = wallet.get("bot_states") or {}
    out: list[dict[str, Any]] = []
    for ticker in sorted(positions.keys()):
        raw = positions.get(ticker)
        try:
            fp = float(raw)
        except (TypeError, ValueError):
            continue
        if abs(fp) < 1e-12:
            continue
        side = "YES" if fp > 0 else "NO"
        n = abs(fp)
        st = states.get(ticker)
        st_d = st if isinstance(st, dict) else {}
        entry = st_d.get("entry_ask_dollars")
        e_f: float | None
        try:
            e_f = float(entry) if entry is not None else None
        except (TypeError, ValueError):
            e_f = None
        out.append(
            {
                "ticker": ticker,
                "side": side,
                "contracts": n,
                "entry_ask_dollars": e_f,
            }
        )
    return out


def paper_wallet_cash_summary(wallet: dict | None) -> dict | None:
    """Cash + start from on-disk wallet (NAV still needs bot paper_portfolio or manual mark)."""
    if not wallet:
        return None
    try:
        bal = float(wallet.get("balance", 0))
        start = float(wallet.get("starting_capital", 0))
    except (TypeError, ValueError):
        return None
    pos = wallet.get("positions") or {}
    npos = len([k for k, v in pos.items() if v and str(v) not in ("0", "0.0", "0.00")])
    return {
        "cash_balance": bal,
        "starting_capital": start,
        "open_leg_count": npos,
        "cash_vs_start": bal - start,
    }


def load_bot_states() -> dict[str, dict]:
    out: dict[str, dict] = {}
    sd = state_dir()
    if not sd.is_dir():
        return out
    for p in sorted(sd.glob("state_*.json")):
        try:
            key = p.stem.replace("state_", "", 1)
            out[key] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _decimate_pairs(pairs: list[list[float]], max_n: int = 2000) -> list[list[float]]:
    if len(pairs) <= max_n:
        return pairs
    step = max(1, len(pairs) // max_n)
    return pairs[::step]


def build_chart_series(rows: list[dict]) -> dict:
    """
    Paper NAV/PnL use absolute epoch seconds on X so the chart does not rescale
    when the JSONL tail window moves (avoids flicker in the dashboard).
    """
    if not rows:
        return {
            "t0": time.time(),
            "paper_nav": [],
            "paper_pnl": [],
            "fair_yes": {},
            "yes_mid": {},
            "unrealized": {},
            "ev_yes": {},
            "ev_no": {},
        }
    t0 = min(float(r.get("ts") or 0) for r in rows)
    paper_nav: list[list[float]] = []
    paper_pnl: list[list[float]] = []
    fair_yes: dict[str, list[list[float]]] = {}
    yes_mid: dict[str, list[list[float]]] = {}
    unrealized: dict[str, list[list[float]]] = {}
    ev_yes: dict[str, list[list[float]]] = {}
    ev_no: dict[str, list[list[float]]] = {}

    for r in rows:
        ts = float(r.get("ts") or 0)
        tr = ts - t0
        if r.get("cycle_event") == "paper_portfolio":
            nav = r.get("paper_nav")
            pnl = r.get("paper_pnl")
            if nav is not None:
                paper_nav.append([ts, float(nav)])
            if pnl is not None:
                paper_pnl.append([ts, float(pnl)])
            continue
        tk = str(r.get("ticker") or "")
        if not tk or tk.startswith("_"):
            continue
        if r.get("fair_yes") is not None:
            fair_yes.setdefault(tk, []).append([tr, float(r["fair_yes"])])
        if r.get("yes_mid") is not None:
            yes_mid.setdefault(tk, []).append([tr, float(r["yes_mid"])])
        if r.get("unrealized_pnl") is not None:
            unrealized.setdefault(tk, []).append([tr, float(r["unrealized_pnl"])])
        if r.get("ev_yes") is not None:
            ev_yes.setdefault(tk, []).append([tr, float(r["ev_yes"])])
        if r.get("ev_no") is not None:
            ev_no.setdefault(tk, []).append([tr, float(r["ev_no"])])

    return {
        "t0": t0,
        "paper_nav": _decimate_pairs(paper_nav),
        "paper_pnl": _decimate_pairs(paper_pnl),
        "fair_yes": {k: _decimate_pairs(v) for k, v in fair_yes.items()},
        "yes_mid": {k: _decimate_pairs(v) for k, v in yes_mid.items()},
        "unrealized": {k: _decimate_pairs(v) for k, v in unrealized.items()},
        "ev_yes": {k: _decimate_pairs(v) for k, v in ev_yes.items()},
        "ev_no": {k: _decimate_pairs(v) for k, v in ev_no.items()},
    }


def build_status() -> dict:
    mp = metrics_path()
    rows = tail_jsonl_lines(mp, MAX_JSONL_LINES)
    now = time.time()
    latest_by_ticker: dict[str, dict] = {}
    for r in rows:
        t = str(r.get("ticker") or "_")
        if t == "_paper_":
            continue
        latest_by_ticker[t] = r

    latest_global = None
    if rows:
        market_rows = [
            r for r in rows if str(r.get("ticker") or "_") not in ("_paper_", "_")
        ]
        pool = market_rows if market_rows else rows
        latest_global = max(pool, key=lambda x: float(x.get("ts") or 0))

    paper_rows_list = [r for r in rows if r.get("cycle_event") == "paper_portfolio"]
    latest_paper = None
    if paper_rows_list:
        latest_paper = max(paper_rows_list, key=lambda x: float(x.get("ts") or 0))

    latest_global_ts = float(latest_global.get("ts") or 0) if latest_global else 0.0
    latest_paper_ts = float(latest_paper.get("ts") or 0) if latest_paper else 0.0
    paper_snapshot_stale = bool(
        latest_paper and latest_global and latest_global_ts > latest_paper_ts + 2.0
    )

    w = load_paper_wallet()
    cash_h = paper_wallet_cash_summary(w)
    paper_rows_n = len(paper_rows_list)
    hint_parts: list[str] = []
    if w and cash_h is not None and paper_rows_n == 0:
        hint_parts.append(
            "This metrics file has no paper_portfolio rows. Point BOT_METRICS_FILE "
            "(and PAPER_WALLET) at the same paths as edge_bot.py so NAV/PnL stay in sync."
        )
    if paper_snapshot_stale:
        hint_parts.append(
            "Paper portfolio snapshot is older than the latest market row — "
            "total capital below may be missing; fix metrics path or restart the paper bot."
        )
    metrics_hint = "<br/>".join(hint_parts) if hint_parts else None

    metrics_age_sec: float | None = None
    if latest_global_ts > 0:
        metrics_age_sec = max(0.0, now - latest_global_ts)
    metrics_stale = metrics_age_sec is not None and metrics_age_sec > 90.0

    chart_series = build_chart_series(rows)
    if paper_snapshot_stale:
        chart_series["paper_nav"] = []
        chart_series["paper_pnl"] = []
    chart_series = {
        "t0": chart_series.get("t0", now),
        "paper_nav": chart_series.get("paper_nav", []),
        "paper_pnl": chart_series.get("paper_pnl", []),
    }

    return {
        "server_time": now,
        "metrics_path": str(mp),
        "metrics_exists": mp.is_file(),
        "metrics_size_bytes": mp.stat().st_size if mp.is_file() else 0,
        "metrics_line_count_est": len(rows),
        "latest_global": latest_global,
        "latest_paper": latest_paper,
        "latest_by_ticker": latest_by_ticker,
        "recent": rows[-RECENT_LINES:],
        "bot_states": load_bot_states(),
        "paper_wallet_path": str(paper_wallet_path()),
        "paper_wallet": w,
        "paper_wallet_cash": cash_h,
        "metrics_hint": metrics_hint,
        "paper_snapshot_stale": paper_snapshot_stale,
        "metrics_age_sec": metrics_age_sec,
        "metrics_stale": metrics_stale,
        "chart_series": chart_series,
        "live_positions": live_positions_from_paper_wallet(w),
    }


@app.route("/api/status")
def api_status():
    return jsonify(build_status())


@app.route("/api/chart_series")
def api_chart_series():
    st = build_status()
    return jsonify(st["chart_series"])


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
  <title>Paper trading — capital</title>
  <style>
    :root {
      --bg: #0d1117;
      --surface: #161b22;
      --border: #30363d;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --yes: #3fb950;
      --no: #f85149;
    }
    * { box-sizing: border-box; }
    body {
      font-family: ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 1.5rem clamp(1rem, 4vw, 2.5rem) 2.5rem;
      line-height: 1.5;
      max-width: 56rem;
      margin-left: auto;
      margin-right: auto;
    }
    h1 { font-size: 1.125rem; font-weight: 600; margin: 0 0 0.25rem; color: var(--muted); letter-spacing: 0.02em; }
    .hero {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem 1.75rem;
      margin-top: 1rem;
    }
    .hero-label { font-size: 0.8125rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.35rem; }
    .hero-cash {
      font-size: clamp(2.75rem, 10vw, 4rem);
      font-weight: 700;
      font-variant-numeric: tabular-nums;
      letter-spacing: -0.03em;
      line-height: 1.1;
      color: var(--accent);
    }
    .hero-nav-row { margin-top: 1.25rem; padding-top: 1.25rem; border-top: 1px solid var(--border); }
    .hero-nav-label { font-size: 0.8125rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.25rem; }
    .hero-nav-val { font-size: clamp(1.35rem, 4vw, 1.75rem); font-weight: 600; font-variant-numeric: tabular-nums; }
    .hero-sub { margin-top: 0.65rem; font-size: 0.9375rem; color: var(--muted); display: flex; flex-wrap: wrap; gap: 0.75rem 1.25rem; }
    .hero-sub strong { color: var(--text); font-weight: 600; }
    .pnl-pos { color: var(--yes); }
    .pnl-neg { color: var(--no); }
    .section {
      margin-top: 1.75rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem 1.5rem;
    }
    .section h2 { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin: 0 0 1rem; font-weight: 600; }
    table { width: 100%; border-collapse: collapse; font-size: 0.9375rem; }
    th, td { text-align: left; padding: 0.65rem 0.5rem; border-bottom: 1px solid var(--border); }
    th { color: var(--muted); font-weight: 500; font-size: 0.8125rem; }
    td { font-variant-numeric: tabular-nums; }
    .mono { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: 0.84rem; word-break: break-all; }
    tr:last-child td { border-bottom: none; }
    .empty { color: var(--muted); font-size: 0.9375rem; margin: 0; }
    .banner { font-size: 0.8125rem; color: var(--muted); margin-bottom: 0.75rem; line-height: 1.55; }
    .banner .err { color: var(--no); }
    .section--chart {
      margin-top: 0.75rem;
    }
    .chart-box {
      position: relative;
      width: 100%;
      height: min(70vh, 780px);
      min-height: 520px;
      margin-top: 0.35rem;
    }
    .side-yes { color: var(--yes); font-weight: 600; }
    .side-no { color: var(--no); font-weight: 600; }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body>
  <h1>Paper trading</h1>
  <div id="banner" class="banner"></div>

  <div class="hero" id="hero">
    <div class="hero-label">Cash balance</div>
    <div class="hero-cash" id="heroCash">—</div>
    <div class="hero-nav-row">
      <div class="hero-nav-label">Total capital (NAV)</div>
      <div class="hero-nav-val" id="heroNav">—</div>
    </div>
    <div class="hero-sub" id="heroSub"></div>
  </div>

  <div class="section section--chart">
    <h2>P/L &amp; capital over time</h2>
    <p class="empty" style="margin-bottom:0.35rem;">NAV (blue) and gain vs starting stake (green). Straight lines — no curve smoothing — so the plot does not shift when new points arrive.</p>
    <div class="chart-box"><canvas id="chCapital"></canvas></div>
  </div>

  <div class="section">
    <h2>Live trades</h2>
    <div id="positionsWrap"></div>
  </div>

<script>
const fmtUsd = (n) => {
  if (n == null || n === "") return "—";
  const x = Number(n);
  return Number.isFinite(x) ? x.toFixed(2) : "—";
};
const fmtUsd4 = (n) => {
  if (n == null || n === "") return "—";
  const x = Number(n);
  return Number.isFinite(x) ? x.toFixed(4) : "—";
};
const fmtTs = (ts) => {
  if (ts == null) return "—";
  return new Date(ts * 1000).toLocaleString();
};
const pnlClass = (x) => (x == null || !Number.isFinite(Number(x)) ? "" : (Number(x) >= 0 ? "pnl-pos" : "pnl-neg"));

const xy = (arr) => (arr || []).map(([x, y]) => ({ x, y }));

function tickTimeShort(v) {
  if (v == null || !Number.isFinite(v)) return "";
  try {
    return new Date(v * 1000).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch (e) {
    return String(v);
  }
}

const chartOpts = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  interaction: { mode: "index", intersect: false },
  plugins: {
    legend: {
      position: "top",
      labels: { color: "#8b949e", font: { size: 13 }, boxWidth: 14 },
    },
  },
  scales: {
    x: {
      type: "linear",
      title: { display: true, text: "Time", color: "#8b949e", font: { size: 12 } },
      ticks: {
        color: "#8b949e",
        maxTicksLimit: 12,
        callback: (v) => tickTimeShort(v),
      },
      grid: { color: "rgba(139,148,158,0.12)" },
    },
    y: {
      title: { display: true, text: "USD", color: "#8b949e", font: { size: 12 } },
      ticks: { color: "#8b949e" },
      grid: { color: "rgba(139,148,158,0.12)" },
      grace: "12%",
    },
  },
};

function seriesKey(nav, pnl) {
  return JSON.stringify({ nav, pnl });
}

function updateCapitalChart(s) {
  if (typeof Chart === "undefined") return;
  const nav = s.paper_nav || [];
  const pnl = s.paper_pnl || [];
  const key = seriesKey(nav, pnl);
  if (key === window.__capSeriesKey) return;
  window.__capSeriesKey = key;

  const navData = xy(nav);
  const pnlData = xy(pnl);
  const el = document.getElementById("chCapital");
  if (!el) return;
  const ctx = el.getContext("2d");

  if (!window.__cap) {
    if (!navData.length && !pnlData.length) return;
    window.__cap = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: "Total capital (NAV)",
            data: navData,
            borderColor: "#58a6ff",
            backgroundColor: "rgba(88,166,255,0.08)",
            fill: true,
            tension: 0,
            pointRadius: 0,
            pointHoverRadius: 0,
            borderWidth: 2,
            spanGaps: true,
          },
          {
            label: "Gain vs start ($)",
            data: pnlData,
            borderColor: "#3fb950",
            borderDash: [6, 4],
            tension: 0,
            pointRadius: 0,
            pointHoverRadius: 0,
            borderWidth: 1.75,
            fill: false,
            spanGaps: true,
          },
        ],
      },
      options: chartOpts,
    });
    return;
  }

  window.__cap.data.datasets[0].data = navData;
  window.__cap.data.datasets[1].data = pnlData;
  window.__cap.update("none");
}

async function refresh() {
  let data;
  try {
    const r = await fetch("/api/status", { cache: "no-store" });
    data = await r.json();
  } catch (e) {
    document.getElementById("banner").innerHTML = '<span class="err">Cannot reach API.</span>';
    return;
  }

  let ban = `<span class="mono">${data.metrics_path}</span>`;
  if (!data.metrics_exists) ban += ` <span class="err">(file missing)</span>`;
  if (data.paper_wallet_path) ban += `<br>Wallet: <span class="mono">${data.paper_wallet_path}</span>`;
  if (data.metrics_hint) ban += `<br><span class="err">${data.metrics_hint}</span>`;
  if (data.metrics_age_sec != null && Number.isFinite(data.metrics_age_sec)) {
    const sec = Math.floor(data.metrics_age_sec);
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    const h = Math.floor(m / 60);
    const mm = m % 60;
    const ago = h > 0 ? `${h}h ${mm}m` : `${m}m ${s}s`;
    ban += `<br>Last market update in file: <strong>${ago}</strong> ago`;
  }
  if (data.metrics_stale) {
    ban += `<br><span class="err">Metrics look stale — is edge_bot still running?</span>`;
  }
  document.getElementById("banner").innerHTML = ban;

  const lp = data.latest_paper;
  const stale = data.paper_snapshot_stale;
  const cash = data.paper_wallet_cash;
  const w = data.paper_wallet;

  const heroCash = document.getElementById("heroCash");
  const heroNav = document.getElementById("heroNav");
  const heroSub = document.getElementById("heroSub");

  if (lp && !stale) {
    heroCash.textContent = "$" + fmtUsd(lp.paper_balance);
    heroNav.textContent = "$" + fmtUsd(lp.paper_nav);
    heroNav.className = "hero-nav-val " + pnlClass(lp.paper_pnl);
    heroSub.innerHTML = `
      <span>P/L <strong class="${pnlClass(lp.paper_pnl)}">$${fmtUsd(lp.paper_pnl)}</strong> vs <strong>$${fmtUsd(lp.paper_start)}</strong> started</span>
      <span>Updated ${fmtTs(lp.ts)}</span>`;
  } else if (w && cash) {
    heroCash.textContent = "$" + fmtUsd(cash.cash_balance);
    heroNav.textContent = stale || !lp ? "—" : "$" + fmtUsd(lp.paper_nav);
    heroNav.className = "hero-nav-val " + (lp && lp.paper_pnl != null ? pnlClass(lp.paper_pnl) : "");
    const hint = stale || !lp
      ? "NAV needs a current paper_portfolio row from the bot."
      : "";
    heroSub.innerHTML = `
      <span>Started <strong>$${fmtUsd(cash.starting_capital)}</strong></span>
      <span>Cash vs start <strong class="${pnlClass(cash.cash_vs_start)}">$${fmtUsd(cash.cash_vs_start)}</strong></span>
      ${hint ? `<span style="width:100%;color:var(--muted);font-size:0.875rem;">${hint}</span>` : ""}`;
  } else {
    heroCash.textContent = "—";
    heroNav.textContent = "—";
    heroNav.className = "hero-nav-val";
    heroSub.innerHTML = '<span style="color:var(--muted);">Start edge_bot with <code class="mono">--paper</code> and matching BOT_METRICS_FILE / PAPER_WALLET.</span>';
  }

  const pos = data.live_positions || [];
  const wrap = document.getElementById("positionsWrap");
  if (!pos.length) {
    wrap.innerHTML = '<p class="empty">No open positions in the paper wallet.</p>';
  } else {
    wrap.innerHTML = `
      <table>
        <thead><tr>
          <th>Market</th><th>Side</th><th>Contracts</th><th>Entry (avg)</th>
        </tr></thead>
        <tbody>
        ${pos.map((p) => `
          <tr>
            <td class="mono">${p.ticker}</td>
            <td class="${p.side === "YES" ? "side-yes" : "side-no"}">${p.side}</td>
            <td>${p.contracts}</td>
            <td>${p.entry_ask_dollars != null ? "$" + fmtUsd4(p.entry_ask_dollars) : "—"}</td>
          </tr>`).join("")}
        </tbody>
      </table>`;
  }

  updateCapitalChart(data.chart_series || {});
}

refresh();
setInterval(refresh, 3000);
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
