"""
signals.py — Trading signal computation for the Kalshi BTC directional bot.

Each signal function returns a dict:
  direction : "YES" | "NO" | "NEUTRAL"  (YES = bet BTC goes up)
  strength  : float 0.0–1.0
  reason    : human-readable explanation string

The aggregate_signals() function combines all signals into a final decision.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SignalResult = dict[str, Any]   # type alias for clarity


# ─────────────────────────────────────────────────────────────────────────────
# Signal 1: Volatility Gate  (HARD GATE — blocks all trading if triggered)
# ─────────────────────────────────────────────────────────────────────────────

def signal_volatility_gate(
    candles: pd.DataFrame,
    low_vol_percentile: float = 15.0,
) -> SignalResult:
    """
    Computes annualized realized volatility and blocks trading during quiet periods.

    Logic:
      - Compute log returns for every consecutive pair of 1-min closes
      - Calculate a rolling 20-period realized vol over the last 60 candles
      - If the most recent vol window is below the 30th percentile of all
        windows, sit out — there's not enough movement to profit from

    Why: low-vol periods mean BTC is barely moving; our binary edge disappears
    and the spread cost eats any potential profit.
    """
    closes = candles["close"].values.astype(float)
    if len(closes) < 22:
        # Not enough data to compute anything meaningful
        return {"direction": "NEUTRAL", "strength": 0.0, "reason": "insufficient candle history"}

    # Log returns: ln(close[t] / close[t-1]) for each 1-min interval
    log_returns = np.log(closes[1:] / closes[:-1])

    # Rolling 20-period realized vol, annualized to minutes-per-year basis
    # sqrt(525_600) because there are 525,600 minutes in a year
    ANNUALIZE = np.sqrt(525_600)
    rolling_vols: list[float] = []
    for i in range(len(log_returns) - 19):
        window_vol = np.std(log_returns[i : i + 20]) * ANNUALIZE
        rolling_vols.append(float(window_vol))

    if not rolling_vols:
        return {"direction": "NEUTRAL", "strength": 0.0, "reason": "not enough return history"}

    current_vol = rolling_vols[-1]
    threshold   = float(np.percentile(rolling_vols, low_vol_percentile))

    if current_vol < threshold:
        return {
            "direction": "NEUTRAL",
            "strength":  0.0,
            "reason":    f"low volatility ({current_vol:.1%} ann) — sitting out",
        }

    # Strength = how far above the threshold we are (normalized 0–1)
    max_vol = max(rolling_vols)
    denom   = max(max_vol - threshold, 1e-9)
    strength = min(1.0, (current_vol - threshold) / denom)

    return {
        "direction": "PASS",   # Not directional — just clears the gate
        "strength":  strength,
        "reason":    f"volatility OK ({current_vol:.1%} ann vol, above {low_vol_percentile:.0f}th pct)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Signal 2: Multi-Timeframe Momentum
# ─────────────────────────────────────────────────────────────────────────────

def signal_momentum(candles: pd.DataFrame) -> SignalResult:
    """
    Checks momentum across three timeframes using 1-minute closes.

    Timeframes:
      5-min:  avg(last 5 closes)  vs  avg(prior 5 closes)
      15-min: avg(last 15 closes) vs  avg(prior 15 closes)
      60-min: last close          vs  close 60 candles ago

    Decision rule: at least 2 of 3 timeframes must agree on direction.
    Strength = agreeing_count / 3  (0.67 for 2/3, 1.0 for unanimous)

    Why: multi-timeframe agreement filters out noise in a single timeframe
    and requires a more consistent trend before we act.
    """
    closes = candles["close"].values.astype(float)
    if len(closes) < 60:
        return {"direction": "NEUTRAL", "strength": 0.0, "reason": "need 60 candles for momentum"}

    # 5-minute trend
    avg_last_5  = closes[-5:].mean()
    avg_prev_5  = closes[-10:-5].mean()
    trend_5m    = "YES" if avg_last_5 > avg_prev_5 else "NO"

    # 15-minute trend
    avg_last_15 = closes[-15:].mean()
    avg_prev_15 = closes[-30:-15].mean()
    trend_15m   = "YES" if avg_last_15 > avg_prev_15 else "NO"

    # 60-minute trend (full look-back window)
    trend_60m   = "YES" if closes[-1] > closes[0] else "NO"

    trends    = [trend_5m, trend_15m, trend_60m]
    yes_count = trends.count("YES")
    no_count  = trends.count("NO")
    desc      = f"5m={trend_5m}, 15m={trend_15m}, 60m={trend_60m}"

    if yes_count >= 2:
        return {
            "direction": "YES",
            "strength":  yes_count / 3.0,
            "reason":    f"momentum YES ({yes_count}/3 agree): {desc}",
        }
    elif no_count >= 2:
        return {
            "direction": "NO",
            "strength":  no_count / 3.0,
            "reason":    f"momentum NO ({no_count}/3 agree): {desc}",
        }
    else:
        return {
            "direction": "NEUTRAL",
            "strength":  0.0,
            "reason":    f"mixed momentum (no agreement): {desc}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Signal 3: Crowd Sentiment Fade
# ─────────────────────────────────────────────────────────────────────────────

def signal_crowd_fade(implied_prob: float) -> SignalResult:
    """
    Fades extreme crowd positioning on Kalshi.

    When 72%+ of Kalshi bettors are saying YES (implied_prob > 0.72), the crowd
    is likely overextended — fade them by going NO. Vice versa for extreme NO.

    Why: in binary prediction markets, extreme one-sided positioning often
    overshoots fair value because of recency bias and herd behavior.

    Strength = distance from 0.50, normalized to [0, 1]
    (e.g. implied_prob=0.80 → dist=0.30, strength=0.60)
    """
    BULL_EXTREME = 0.72
    BEAR_EXTREME = 0.28

    # Strength: how far from neutral (0.5) the crowd is leaning
    dist_from_neutral = abs(implied_prob - 0.5)
    strength = min(1.0, dist_from_neutral / 0.5)

    if implied_prob > BULL_EXTREME:
        return {
            "direction": "NO",
            "strength":  strength,
            "reason":    f"crowd overextended YES ({implied_prob:.0%} implied) — fading",
        }
    elif implied_prob < BEAR_EXTREME:
        return {
            "direction": "YES",
            "strength":  strength,
            "reason":    f"crowd overextended NO ({implied_prob:.0%} implied) — fading",
        }
    else:
        return {
            "direction": "NEUTRAL",
            "strength":  0.0,
            "reason":    f"crowd neutral ({implied_prob:.0%} implied) — no fade signal",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Signal 4: Perpetual Futures Funding Rate
# ─────────────────────────────────────────────────────────────────────────────

def signal_funding_rate(funding_rate: float) -> SignalResult:
    """
    Uses the BTC perp funding rate as a contrarian / mean-reversion signal.

    Positive funding (> +0.0002): longs are paying shorts, meaning leveraged
      longs are overextended → lean NO (short-side bets)
    Negative funding (< -0.0002): shorts are paying longs, meaning leveraged
      shorts are overextended → lean YES (long-side bets)

    Why: extreme funding rates signal over-leveraged directional positioning in
    the perp market; these positions tend to be forcibly unwound (liquidations),
    which can create short-term price moves against the crowded side.
    """
    BULL_THRESHOLD =  0.0002
    BEAR_THRESHOLD = -0.0002

    if funding_rate > BULL_THRESHOLD:
        return {
            "direction": "NO",
            "strength":  0.6,
            "reason":    f"perp funding {funding_rate:+.4f} → longs extended, lean NO",
        }
    elif funding_rate < BEAR_THRESHOLD:
        return {
            "direction": "YES",
            "strength":  0.6,
            "reason":    f"perp funding {funding_rate:+.4f} → shorts extended, lean YES",
        }
    else:
        return {
            "direction": "NEUTRAL",
            "strength":  0.0,
            "reason":    f"perp funding neutral ({funding_rate:+.6f})",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Signal Aggregator
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_signals(
    vol_signal:   SignalResult,
    momentum:     SignalResult,
    crowd_fade:   SignalResult,
    funding:      SignalResult,
    macro_soon:   bool,
    min_strength: float = 0.55,
) -> dict[str, Any]:
    """
    Combines individual signals into a single actionable trading decision.

    Hard gates (override everything else):
      1. Volatility gate fired (low vol) → NEUTRAL
      2. Macro event within 30 min       → NEUTRAL

    Weighted vote (signals 2–4):
      Momentum   weight = 0.50
      Crowd fade weight = 0.35
      Funding    weight = 0.15

    Final direction only returned if combined strength > min_strength.
    Otherwise NEUTRAL (skip the trade).

    Returns
    -------
    dict with keys: direction, strength, reasons (list of strings)
    """
    reasons: list[str] = []

    # ── Hard gate 1: Volatility ───────────────────────────────────────────────
    if vol_signal["direction"] == "NEUTRAL":
        return {
            "direction": "NEUTRAL",
            "strength":  0.0,
            "reasons":   [vol_signal["reason"]],
        }
    reasons.append(vol_signal["reason"])

    # ── Hard gate 2: Macro event blackout ─────────────────────────────────────
    if macro_soon:
        return {
            "direction": "NEUTRAL",
            "strength":  0.0,
            "reasons":   ["macro event blackout — not trading"],
        }

    # ── Weighted vote across directional signals ──────────────────────────────
    weights = {"momentum": 0.65, "crowd_fade": 0.20, "funding": 0.15}

    signal_pairs = [
        (momentum,   weights["momentum"]),
        (crowd_fade, weights["crowd_fade"]),
        (funding,    weights["funding"]),
    ]

    yes_score = 0.0
    no_score  = 0.0

    for sig, w in signal_pairs:
        reasons.append(sig["reason"])
        if sig["direction"] == "YES":
            yes_score += w * sig["strength"]
        elif sig["direction"] == "NO":
            no_score  += w * sig["strength"]
        # NEUTRAL contributes 0 to either side

    if yes_score == 0.0 and no_score == 0.0:
        return {
            "direction": "NEUTRAL",
            "strength":  0.0,
            "reasons":   reasons + ["all directional signals neutral"],
        }

    if yes_score >= no_score:
        direction = "YES"
        strength  = yes_score
    else:
        direction = "NO"
        strength  = no_score

    # Final strength gate — only trade when we have clear conviction
    if strength < min_strength:
        return {
            "direction": "NEUTRAL",
            "strength":  strength,
            "reasons":   reasons + [
                f"combined strength {strength:.2f} below threshold {min_strength} — skipping"
            ],
        }

    return {
        "direction": direction,
        "strength":  round(strength, 4),
        "reasons":   reasons,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Top-level convenience function
# ─────────────────────────────────────────────────────────────────────────────

def run_signals(
    candles:      pd.DataFrame,
    funding_rate: float,
    implied_prob: float,
    macro_soon:   bool,
    min_strength: float = 0.55,
) -> tuple[dict[str, Any], SignalResult, SignalResult, SignalResult, SignalResult]:
    """
    Compute all four signals and aggregate them.

    Returns
    -------
    (aggregated, vol_signal, momentum_signal, crowd_signal, funding_signal)
    """
    vol_sig  = signal_volatility_gate(candles)
    momentum = signal_momentum(candles)
    crowd    = signal_crowd_fade(implied_prob)
    funding  = signal_funding_rate(funding_rate)
    agg      = aggregate_signals(
        vol_signal   = vol_sig,
        momentum     = momentum,
        crowd_fade   = crowd,
        funding      = funding,
        macro_soon   = macro_soon,
        min_strength = min_strength,
    )
    return agg, vol_sig, momentum, crowd, funding
