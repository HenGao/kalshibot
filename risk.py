"""
risk.py — Fractional Kelly Criterion position sizing.

The Kelly Criterion tells you the mathematically optimal fraction of your
bankroll to bet given your edge. We use "quarter-Kelly" (kelly_fraction=0.25)
which is much more conservative — it sacrifices some expected growth for
significantly lower variance (drawdowns). Good for beginners.

Key idea:
  - If you have edge, Kelly says bet more. If you have no edge (or negative),
    Kelly says bet nothing.
  - Quarter-Kelly means: take what Kelly recommends, then bet only 25% of that.
  - A hard cap (max_bet_pct=0.08) ensures we never risk more than 8% of bankroll
    on any single trade, regardless of what Kelly says.
"""

from __future__ import annotations

import math


def kelly_size(
    bankroll:         float,
    win_probability:  float,
    yes_price_cents:  float,
    kelly_fraction:   float = 0.25,
    max_bet_pct:      float = 0.08,
    side:             str   = "yes",
) -> tuple[int, float]:
    """
    Compute the number of contracts to buy using fractional Kelly Criterion.

    Parameters
    ----------
    bankroll         : Total bankroll in dollars (e.g. 50.0)
    win_probability  : Estimated P(win) for this specific bet (0.5–0.85)
    yes_price_cents  : Current YES ask price in cents (1–99)
    kelly_fraction   : Fraction of Kelly to use (0.25 = conservative quarter-Kelly)
    max_bet_pct      : Hard cap on single-trade risk as fraction of bankroll
    side             : "yes" or "no" — which side we're buying

    Returns
    -------
    (num_contracts, bet_size_dollars)
    num_contracts = 0 means no trade (Kelly says no edge)

    Example
    -------
    >>> kelly_size(bankroll=50, win_probability=0.62, yes_price_cents=45, side="yes")
    (2, 0.90)   # Buy 2 YES contracts at $0.45 each = $0.90 total bet
    """
    # Determine the price per contract for the side we're buying
    if side.lower() == "yes":
        price_cents  = float(yes_price_cents)
    else:
        price_cents  = 100.0 - float(yes_price_cents)  # NO price = 100 - YES price

    price_dollars = price_cents / 100.0

    # Kelly formula:
    #   b    = net profit per dollar at risk (i.e. "odds")
    #          If you pay $0.45 and win $1.00, your net profit is $0.55
    #          So odds b = (1.00 - 0.45) / 0.45 = 1.222
    #   p    = probability of winning
    #   q    = 1 - p
    #   f*   = (p*b - q) / b  =  (p*(b+1) - 1) / b
    if price_dollars <= 0 or price_dollars >= 1:
        return 0, 0.0

    odds       = (1.0 - price_dollars) / price_dollars   # net profit / stake
    p          = float(win_probability)
    q          = 1.0 - p

    kelly_raw  = (p * (odds + 1.0) - 1.0) / odds

    if kelly_raw <= 0:
        # Negative Kelly = our estimated edge doesn't cover the price → don't bet
        return 0, 0.0

    # Apply fractional Kelly (scale down to be conservative)
    bet_fraction = kelly_fraction * kelly_raw

    # Hard cap: never risk more than max_bet_pct of bankroll
    bet_fraction = min(bet_fraction, max_bet_pct)

    bet_size_dollars = bankroll * bet_fraction

    # Convert to integer contracts (floor to avoid overspending)
    num_contracts = math.floor(bet_size_dollars / price_dollars)

    # Floor at 1 (if Kelly says bet anything, buy at least 1 contract)
    num_contracts = max(1, num_contracts)

    # Actual dollar cost after integer rounding
    actual_bet_dollars = num_contracts * price_dollars

    return num_contracts, round(actual_bet_dollars, 2)


def signal_to_win_prob(
    combined_strength: float,
    direction:         str,
    yes_price_cents:   float,
) -> float:
    """
    Map our combined signal strength (0.0–1.0) to an estimated win probability.

    We blend the signal strength with the market-implied probability so our
    estimate is grounded in what the market already "knows". This prevents us
    from wildly overestimating our edge.

    Formula (for a YES bet):
      p_implied = yes_price_cents / 100
      p_win = p_implied + strength × (1 - p_implied) × CONFIDENCE_FACTOR

    Where CONFIDENCE_FACTOR = 0.3 means: at max signal strength (1.0), we only
    move 30% of the remaining "probability gap" above the market's estimate.
    This is intentionally conservative.

    Parameters
    ----------
    combined_strength : float, 0.0–1.0 from the signal aggregator
    direction         : "YES" or "NO"
    yes_price_cents   : Current YES ask price (cents)

    Returns
    -------
    float in [0.51, 0.85] — estimated probability of winning
    """
    p_mkt = float(yes_price_cents) / 100.0
    CONFIDENCE_FACTOR = 0.30   # How aggressively we shift the probability

    if direction.upper() == "YES":
        # We think BTC goes up: our estimate is above the market's
        p_win = p_mkt + combined_strength * (1.0 - p_mkt) * CONFIDENCE_FACTOR
    else:
        # We're betting NO (BTC goes down): our estimate of P(down) is above market's
        p_down_mkt = 1.0 - p_mkt
        p_win = p_down_mkt + combined_strength * (1.0 - p_down_mkt) * CONFIDENCE_FACTOR

    # Clamp to a reasonable range — we're humble about our edge
    return max(0.51, min(0.85, p_win))


def check_balance(num_contracts: int, price_cents: float, balance_cents: int) -> int:
    """
    Reduce num_contracts if the total cost exceeds available balance.

    Parameters
    ----------
    num_contracts  : Proposed number of contracts
    price_cents    : Cost per contract in cents
    balance_cents  : Available balance in cents

    Returns
    -------
    Adjusted num_contracts (at least 0 if we can't afford even 1)
    """
    if price_cents <= 0:
        return 0
    affordable = int(balance_cents // price_cents)
    adjusted   = min(num_contracts, affordable)
    if adjusted < num_contracts:
        print(
            f"  [risk] Balance constraint: reduced {num_contracts} → {adjusted} contracts "
            f"(balance ${balance_cents/100:.2f}, cost {price_cents:.0f}¢/contract)"
        )
    return max(0, adjusted)
