"""Append-only metrics for live training / monitoring graphs (JSONL)."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any


def _f(d: Decimal) -> float:
    return float(d)


@dataclass
class _TickerLedger:
    realized: Decimal = field(default_factory=lambda: Decimal("0"))
    prev_unrealized: Decimal = field(default_factory=lambda: Decimal("0"))
    prev_abs_pos: Decimal = field(default_factory=lambda: Decimal("0"))


class LiveMetricsSink:
    """
    Buffers one row per bot cycle, then appends a JSON line on flush().
    Tracks **per-ticker** realized PnL; equity = sum(realized) + sum(current
    unrealized across tickers last reported). Suitable for multi-market loops.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._row: dict[str, Any] = {}
        self._ledger: dict[str, _TickerLedger] = defaultdict(_TickerLedger)
        self._last_unreal: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))

    def update(self, **kwargs: Any) -> None:
        self._row.update(kwargs)

    def flush(self) -> None:
        if not self._row:
            return
        row = self._row
        self._row = {}

        if row.get("cycle_event") == "paper_portfolio":
            out: dict[str, Any] = {
                "ts": float(row.get("ts") or time.time()),
                "ticker": str(row.get("ticker") or "_paper_"),
                "cycle_event": "paper_portfolio",
                "paper_mode": True,
            }
            for k in (
                "paper_balance",
                "paper_nav",
                "paper_pnl",
                "paper_start",
                "paper_position_tickers",
            ):
                if k in row and row[k] is not None:
                    out[k] = row[k]
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(out) + "\n")
            return

        ts = float(row.get("ts") or time.time())
        ticker = str(row.get("ticker") or "_")
        led = self._ledger[ticker]

        pos = Decimal(str(row.get("position_fp") or "0"))
        abs_pos = abs(pos)

        side = row.get("bot_side")
        contracts = Decimal(str(row.get("bot_contracts") or "0"))
        entry = row.get("bot_entry_ask")
        entry_d = Decimal(str(entry)) if entry is not None else None

        orphan_entry = row.get("orphan_entry_ask_dollars")
        need_orphan = abs_pos > 0 and not (
            side in ("yes", "no") and contracts > 0 and entry_d is not None
        )
        if need_orphan and orphan_entry is not None:
            side = "yes" if pos > 0 else "no"
            contracts = abs_pos
            entry_d = Decimal(str(orphan_entry))

        yb = Decimal(str(row.get("best_yes_bid") or "0"))
        nb = Decimal(str(row.get("best_no_bid") or "0"))

        unrealized = Decimal("0")
        if abs_pos > 0 and side in ("yes", "no") and contracts > 0 and entry_d is not None:
            if side == "yes":
                unrealized = contracts * (yb - entry_d)
            else:
                unrealized = contracts * (nb - entry_d)

        if led.prev_abs_pos > 0 and abs_pos == 0:
            led.realized += led.prev_unrealized

        led.prev_unrealized = unrealized if abs_pos > 0 else Decimal("0")
        led.prev_abs_pos = abs_pos

        self._last_unreal[ticker] = unrealized if abs_pos > 0 else Decimal("0")

        total_realized = sum(x.realized for x in self._ledger.values())
        total_unrealized = sum(self._last_unreal.values())
        equity = total_realized + total_unrealized

        out = {
            "ts": ts,
            "ticker": ticker,
            "best_yes_bid": _f(yb),
            "best_yes_ask": _f(Decimal(str(row.get("best_yes_ask") or "0"))),
            "best_no_bid": _f(nb),
            "best_no_ask": _f(Decimal(str(row.get("best_no_ask") or "0"))),
            "yes_mid": _f((yb + Decimal(str(row.get("best_yes_ask") or "0"))) / 2),
            "position_fp": _f(pos),
            "fair_yes": row.get("fair_yes"),
            "ev_yes": row.get("ev_yes"),
            "ev_no": row.get("ev_no"),
            "cycle_event": row.get("cycle_event") or "poll",
            "unrealized_pnl": _f(unrealized),
            "realized_pnl": _f(total_realized),
            "equity": _f(equity),
            "bot_side": side,
            "bot_contracts": _f(contracts) if contracts else None,
        }
        if row.get("paper_balance") is not None:
            out["paper_balance"] = float(row["paper_balance"])
        if row.get("paper_mode"):
            out["paper_mode"] = True
        for k in ("btc_spot_usd", "kalshi_target_usd", "btc_target_source"):
            if k in row and row[k] is not None:
                out[k] = row[k]
        if row.get("orphan_entry_ask_dollars") is not None:
            out["orphan_entry_ask_dollars"] = float(row["orphan_entry_ask_dollars"])
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")
