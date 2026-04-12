"""Executable prices, spread, quadratic taker-fee estimate, and edge checks."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal


@dataclass(frozen=True)
class ExecutablePrices:
    best_yes_bid: Decimal
    best_yes_ask: Decimal
    best_no_bid: Decimal
    best_no_ask: Decimal
    yes_spread: Decimal
    no_spread: Decimal


def _d(x: str | float | Decimal) -> Decimal:
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


def parse_orderbook(orderbook_json: dict) -> ExecutablePrices:
    ob = orderbook_json.get("orderbook_fp") or orderbook_json.get("orderbook") or {}
    yes_levels = ob.get("yes_dollars") or []
    no_levels = ob.get("no_dollars") or []

    best_yes_bid = _d(yes_levels[-1][0]) if yes_levels else Decimal("0")
    best_no_bid = _d(no_levels[-1][0]) if no_levels else Decimal("0")

    # Implied asks from reciprocal bids (Kalshi binary book).
    best_yes_ask = Decimal("1") - best_no_bid if no_levels else Decimal("1")
    best_no_ask = Decimal("1") - best_yes_bid if yes_levels else Decimal("1")

    yes_spread = best_yes_ask - best_yes_bid
    no_spread = best_no_ask - best_no_bid

    return ExecutablePrices(
        best_yes_bid=best_yes_bid,
        best_yes_ask=best_yes_ask,
        best_no_bid=best_no_bid,
        best_no_ask=best_no_ask,
        yes_spread=yes_spread,
        no_spread=no_spread,
    )


def quadratic_taker_fee_total_usd(
    price_per_contract: Decimal,
    contracts: Decimal,
    *,
    coefficient: Decimal = Decimal("0.07"),
    fee_multiplier: Decimal = Decimal("1"),
) -> Decimal:
    """
    Kalshi-style quadratic taker fee: scales with p*(1-p). This is an estimate;
    actual fees include cent rounding / accumulators (see Kalshi fee docs).

    Uses coefficient * multiplier * sum over contracts of p*(1-p) as a single
    lump (equivalent to per-contract when using one price level).
    """
    p = price_per_contract
    if p <= 0 or p >= 1:
        return Decimal("0")
    unit = p * (Decimal("1") - p)
    raw = coefficient * fee_multiplier * contracts * unit
    return raw.quantize(Decimal("0.0001"), rounding=ROUND_CEILING)


def expected_value_buy_yes(
    fair_yes: Decimal, yes_ask: Decimal, est_fee_per_contract: Decimal
) -> Decimal:
    """Per-contract EV in dollars after paying ask and estimated taker fee."""
    return fair_yes - yes_ask - est_fee_per_contract


def expected_value_buy_no(
    fair_yes: Decimal, no_ask: Decimal, est_fee_per_contract: Decimal
) -> Decimal:
    fair_no = Decimal("1") - fair_yes
    return fair_no - no_ask - est_fee_per_contract
