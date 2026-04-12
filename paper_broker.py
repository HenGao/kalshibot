"""Paper (virtual) wallet: real Kalshi quotes, no authenticated orders."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from edge_math import quadratic_taker_fee_total_usd


def _dec(x: Any) -> Decimal:
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


@dataclass
class PaperBroker:
    """
    Signed position: positive = YES contracts, negative = NO contracts.
    Cash balance in USD. Persists to JSON for dashboard restarts.

    starting_capital is fixed at wallet creation (for PnL vs your fake bankroll).
    """

    path: Path
    balance: Decimal
    starting_capital: Decimal
    positions: dict[str, Decimal] = field(default_factory=dict)
    bot_states: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, default_balance: Decimal) -> PaperBroker:
        if not path.is_file():
            b = PaperBroker(
                path=path,
                balance=default_balance,
                starting_capital=default_balance,
            )
            b.persist()
            return b
        raw = json.loads(path.read_text(encoding="utf-8"))
        pos = {k: _dec(v) for k, v in (raw.get("positions") or {}).items()}
        states = raw.get("bot_states") or {}
        bal = _dec(raw.get("balance", default_balance))
        start_raw = raw.get("starting_capital")
        if start_raw is None:
            starting = default_balance
        else:
            starting = _dec(start_raw)
        broker = cls(
            path=path,
            balance=bal,
            starting_capital=starting,
            positions=pos,
            bot_states=states,
        )
        if "starting_capital" not in raw:
            broker.persist()
        return broker

    def persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "balance": str(self.balance),
            "starting_capital": str(self.starting_capital),
            "positions": {k: str(v) for k, v in self.positions.items()},
            "bot_states": self.bot_states,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def position_fp(self, ticker: str) -> Decimal:
        return self.positions.get(ticker, Decimal("0"))

    def load_state(self, ticker: str) -> dict[str, Any] | None:
        return self.bot_states.get(ticker)

    def save_state(self, ticker: str, state: dict[str, Any]) -> None:
        self.bot_states[ticker] = state
        self.persist()

    def clear_state(self, ticker: str) -> None:
        self.bot_states.pop(ticker, None)
        self.persist()

    def _fee(
        self,
        price: Decimal,
        contracts: Decimal,
        *,
        coeff: Decimal,
        mult: Decimal,
    ) -> Decimal:
        return quadratic_taker_fee_total_usd(
            price, contracts, coefficient=coeff, fee_multiplier=mult
        )

    def buy_yes(
        self,
        ticker: str,
        contracts: Decimal,
        yes_ask: Decimal,
        *,
        fee_coefficient: Decimal,
        fee_multiplier: Decimal,
    ) -> tuple[bool, dict[str, Any]]:
        fee = self._fee(yes_ask, contracts, coeff=fee_coefficient, mult=fee_multiplier)
        cost = contracts * yes_ask + fee
        if self.balance < cost:
            return False, {"reason": "insufficient_cash", "need": str(cost)}
        self.balance -= cost
        self.positions[ticker] = self.positions.get(ticker, Decimal("0")) + contracts
        self.persist()
        return True, {
            "order": {
                "order_id": f"paper-{uuid.uuid4()}",
                "fill_count_fp": str(contracts),
            }
        }

    def buy_no(
        self,
        ticker: str,
        contracts: Decimal,
        no_ask: Decimal,
        *,
        fee_coefficient: Decimal,
        fee_multiplier: Decimal,
    ) -> tuple[bool, dict[str, Any]]:
        fee = self._fee(no_ask, contracts, coeff=fee_coefficient, mult=fee_multiplier)
        cost = contracts * no_ask + fee
        if self.balance < cost:
            return False, {"reason": "insufficient_cash", "need": str(cost)}
        self.balance -= cost
        self.positions[ticker] = self.positions.get(ticker, Decimal("0")) - contracts
        self.persist()
        return True, {
            "order": {
                "order_id": f"paper-{uuid.uuid4()}",
                "fill_count_fp": str(contracts),
            }
        }

    def sell_yes_reduce(
        self,
        ticker: str,
        contracts: Decimal,
        yes_bid: Decimal,
        *,
        fee_coefficient: Decimal,
        fee_multiplier: Decimal,
    ) -> None:
        fee = self._fee(yes_bid, contracts, coeff=fee_coefficient, mult=fee_multiplier)
        proceeds = contracts * yes_bid - fee
        self.balance += proceeds
        cur = self.positions.get(ticker, Decimal("0"))
        self.positions[ticker] = cur - contracts
        if abs(self.positions[ticker]) < Decimal("0.0001"):
            self.positions.pop(ticker, None)
        self.persist()

    def sell_no_reduce(
        self,
        ticker: str,
        contracts: Decimal,
        no_bid: Decimal,
        *,
        fee_coefficient: Decimal,
        fee_multiplier: Decimal,
    ) -> None:
        fee = self._fee(no_bid, contracts, coeff=fee_coefficient, mult=fee_multiplier)
        proceeds = contracts * no_bid - fee
        self.balance += proceeds
        cur = self.positions.get(ticker, Decimal("0"))
        self.positions[ticker] = cur + contracts
        if abs(self.positions[ticker]) < Decimal("0.0001"):
            self.positions.pop(ticker, None)
        self.persist()
