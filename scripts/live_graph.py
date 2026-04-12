#!/usr/bin/env python3
"""
Live matplotlib dashboard for data/live_metrics.jsonl (written by edge_bot.py).

Run in a second terminal while the bot runs:
  python scripts/live_graph.py
  python scripts/live_graph.py --file data/live_metrics.jsonl

With multiple markets, YES mid is one line per ticker (color-coded). Equity is
combined across tickers (see live_metrics.LiveMetricsSink).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Max lines to load from end of file (each poll is one line).
TAIL_LINES = 50_000


def read_metrics_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    raw = path.read_bytes()
    if not raw:
        return []
    chunk = raw[-4_194_304:] if len(raw) > 4_194_304 else raw
    text = chunk.decode("utf-8", errors="replace")
    lines = text.splitlines()
    rows = []
    for line in lines[-TAIL_LINES:]:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Live graph for Kalshi bot metrics JSONL")
    p.add_argument(
        "--file",
        type=Path,
        default=Path("data/live_metrics.jsonl"),
        help="Path to JSONL written by edge_bot --metrics-file",
    )
    p.add_argument("--interval-ms", type=int, default=750, help="Redraw interval")
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required: pip install matplotlib\n" + str(e)
        ) from e

    fig, (ax_eq, ax_px) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Kalshi bot — equity & YES mid (multi-ticker)")

    (eq_line,) = ax_eq.plot([], [], color="#2ecc71", linewidth=1.5, label="equity ($)")
    (rl_line,) = ax_eq.plot([], [], color="#3498db", linewidth=1, alpha=0.7, label="realized ($)")
    ax_eq.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax_eq.set_ylabel("USD (est.)")
    ax_eq.grid(True, alpha=0.3)

    trade_x: list[float] = []
    trade_y: list[float] = []
    scat = ax_eq.scatter([], [], s=36, c="#e74c3c", marker="^", zorder=5, label="entry / fill")
    ax_eq.legend(loc="upper left", fontsize=8)

    colors = list(plt.cm.tab10.colors)
    ax_px.set_ylabel("YES mid")
    ax_px.set_xlabel("time (session seconds)")
    ax_px.grid(True, alpha=0.3)

    t0: list[float | None] = [None]

    def update(_frame: int) -> tuple:
        rows = read_metrics_jsonl(args.file)
        if not rows:
            return (eq_line, rl_line, scat)

        if t0[0] is None:
            t0[0] = float(rows[0]["ts"])

        t_rel = [float(r["ts"]) - t0[0] for r in rows]
        eq = [float(r["equity"]) for r in rows]
        rl = [float(r["realized_pnl"]) for r in rows]
        eq_line.set_data(t_rel, eq)
        rl_line.set_data(t_rel, rl)

        trade_x.clear()
        trade_y.clear()
        for r, tx in zip(rows, t_rel):
            ev = r.get("cycle_event")
            if ev in ("dry_entry", "filled", "no_fill"):
                trade_x.append(tx)
                trade_y.append(float(r["equity"]))
        if trade_x:
            scat.set_offsets(list(zip(trade_x, trade_y)))
        else:
            scat.set_offsets([])

        per_ticker: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
        for r, tx in zip(rows, t_rel):
            tk = str(r.get("ticker") or "_")
            per_ticker[tk][0].append(tx)
            per_ticker[tk][1].append(float(r["yes_mid"]))

        ax_px.clear()
        ax_px.set_ylabel("YES mid")
        ax_px.set_xlabel("time (session seconds)")
        ax_px.grid(True, alpha=0.3)
        mid_artists = []
        for i, (tk, (xs, ys)) in enumerate(sorted(per_ticker.items())):
            (ln,) = ax_px.plot(
                xs,
                ys,
                color=colors[i % len(colors)],
                linewidth=1.2,
                label=tk[:32] + ("…" if len(tk) > 32 else ""),
            )
            mid_artists.append(ln)
        ax_px.legend(loc="upper left", fontsize=7)

        for ax in (ax_eq, ax_px):
            ax.relim()
            ax.autoscale_view()

        return (eq_line, rl_line, scat, *mid_artists)

    _ = FuncAnimation(fig, update, interval=args.interval_ms, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
