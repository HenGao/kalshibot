"""In-memory session for --simulate (fake books, positions, bot state)."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any


def synthetic_orderbook_fp(mid: Decimal, spread: Decimal) -> dict[str, Any]:
    """
    Build orderbook_fp like Kalshi API: reciprocal bids only.
    Target yes_mid ~= mid, half-spread ~= spread/2.
    """
    half = spread / 2
    by = (mid - half).quantize(Decimal("0.0001"))
    bn = (Decimal("1") - mid - half).quantize(Decimal("0.0001"))
    if by < Decimal("0.01"):
        by = Decimal("0.01")
    if bn < Decimal("0.01"):
        bn = Decimal("0.01")
    if by + bn > Decimal("0.99"):
        bn = Decimal("0.99") - by
    return {
        "orderbook_fp": {
            "yes_dollars": [["0.0100", "100.00"], [str(by), "50.00"]],
            "no_dollars": [["0.0100", "100.00"], [str(bn), "50.00"]],
        }
    }


def _default_horizon_sec(ticker: str) -> float:
    if ticker.upper().startswith("KXBTC15M"):
        return 600.0
    return 3600.0


@dataclass
class SimSession:
    tickers: tuple[str, ...]
    rng: random.Random = field(default_factory=random.Random)
    drift: Decimal = Decimal("0.002")
    spread: Decimal = Decimal("0.04")
    mids: dict[str, Decimal] = field(default_factory=dict)
    seconds_left: dict[str, float] = field(default_factory=dict)
    positions: dict[str, Decimal] = field(default_factory=dict)
    bot_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Synthetic BTC path for --btc-trend-model + --simulate
    btc_hist: dict[str, list[float]] = field(default_factory=dict)
    btc_target: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for t in self.tickers:
            if t not in self.mids:
                self.mids[t] = Decimal(str(round(self.rng.uniform(0.42, 0.58), 4)))
            if t not in self.seconds_left:
                self.seconds_left[t] = _default_horizon_sec(t)
            if t not in self.btc_hist:
                base = 97000.0 + self.rng.uniform(-800.0, 800.0)
                self.btc_hist[t] = [
                    base * (1.0 + self.rng.uniform(-0.0008, 0.0008)) for _ in range(220)
                ]
            if t not in self.btc_target:
                self.btc_target[t] = float(self.btc_hist[t][-1]) * float(
                    self.rng.uniform(0.996, 0.999)
                )

    def btc_closes(self, ticker: str) -> list[float]:
        return list(self.btc_hist.get(ticker, [97000.0]))

    def market_payload_for_features(self, ticker: str) -> dict[str, Any]:
        sec = max(0.0, self.seconds_left.get(ticker, 0.0))
        close = datetime.now(timezone.utc) + timedelta(seconds=sec)
        hz = _default_horizon_sec(ticker)
        open_t = close - timedelta(seconds=hz)
        iso_close = close.strftime("%Y-%m-%dT%H:%M:%SZ")
        iso_open = open_t.strftime("%Y-%m-%dT%H:%M:%SZ")
        tgt = self.btc_target.get(ticker, self.btc_hist[ticker][-1] * 0.998)
        return {
            "market": {
                "close_time": iso_close,
                "open_time": iso_open,
                "floor_strike": float(tgt),
                "strike_type": "greater",
            }
        }

    def tick_time(self, dt: float) -> None:
        for t in self.tickers:
            self.seconds_left[t] = max(0.0, self.seconds_left[t] - dt)

    def step_book(self, ticker: str) -> dict[str, Any]:
        m = float(self.mids[ticker])
        m += self.rng.uniform(-float(self.drift), float(self.drift))
        m = max(0.08, min(0.92, m))
        self.mids[ticker] = Decimal(str(round(m, 4)))
        hist = self.btc_hist.get(ticker, [])
        if hist:
            px = hist[-1] * (1.0 + self.rng.uniform(-0.0025, 0.0025))
            hist.append(px)
            self.btc_hist[ticker] = hist[-499:]
        return synthetic_orderbook_fp(self.mids[ticker], self.spread)

    def position_fp(self, ticker: str) -> Decimal:
        return self.positions.get(ticker, Decimal("0"))

    def load_state(self, ticker: str) -> dict[str, Any] | None:
        return self.bot_states.get(ticker)

    def save_state(self, ticker: str, state: dict[str, Any]) -> None:
        self.bot_states[ticker] = state

    def clear_state(self, ticker: str) -> None:
        self.bot_states.pop(ticker, None)

    def apply_buy_ioc(
        self,
        ticker: str,
        *,
        side: str,
        count_fp: Decimal,
        yes_ask: Decimal,
        no_ask: Decimal,
    ) -> dict[str, Any]:
        """Always fully fill at displayed ask; update signed position."""
        n = count_fp
        if side == "yes":
            cur = self.positions.get(ticker, Decimal("0"))
            self.positions[ticker] = cur + n
        else:
            cur = self.positions.get(ticker, Decimal("0"))
            self.positions[ticker] = cur - n
        return {
            "order": {
                "order_id": str(uuid.uuid4()),
                "fill_count_fp": f"{n:.2f}",
            }
        }

    def apply_sell_ioc(
        self,
        ticker: str,
        *,
        side: str,
        count_fp: Decimal,
    ) -> None:
        n = count_fp
        if side == "yes":
            cur = self.positions.get(ticker, Decimal("0"))
            self.positions[ticker] = cur - n
        else:
            cur = self.positions.get(ticker, Decimal("0"))
            self.positions[ticker] = cur + n
        if abs(self.positions.get(ticker, Decimal("0"))) < Decimal("0.0001"):
            self.positions.pop(ticker, None)
