"""Load JSON gate reports produced by ``scripts/simulate_recent_edge.py --write-report``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_strategy_gate(path: Path) -> tuple[bool, str, dict[str, Any] | None]:
    """
    Returns (ok, reason, data). ``ok`` is True only when the report exists and
    ``passed`` is exactly True.
    """
    if not path.is_file():
        return False, f"missing gate file {path}", None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, f"invalid JSON in {path}: {e}", None
    if not isinstance(data, dict):
        return False, "gate file must be a JSON object", None
    if data.get("passed") is True:
        return True, "strategy gate passed", data
    avg = data.get("avg_pnl_per_trade")
    n = data.get("n_trades")
    return (
        False,
        f"gate not passed (passed={data.get('passed')!r} avg_pnl={avg} n_trades={n})",
        data,
    )
