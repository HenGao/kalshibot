"""
bot_client.py — High-level Kalshi trading client for the BTC directional bot.

This wraps the low-level signed HTTP client (kalshi_client.KalshiClient) with
the specific operations the bot needs: fetching market prices, placing YES/NO
orders, and reading available balance.

NOTE: Kalshi uses API key + RSA private key authentication, NOT email/password.
      In PAPER_TRADE mode (config.PAPER_TRADE = True), no credentials are needed
      because orders are never sent — only logged locally.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
from kalshi_client import KalshiClient, public_get


class BotKalshiClient:
    """
    High-level Kalshi client for the BTC directional bot.

    Usage:
        client = BotKalshiClient()
        market = client.get_market_info("KXBTC15M-25MAY01-T95000")
        client.place_order(ticker="KXBTC15M-...", side="yes",
                           num_contracts=2, limit_price_cents=48)
    """

    def __init__(self) -> None:
        self._client: KalshiClient | None = None

        if not config.PAPER_TRADE:
            # Live mode requires valid credentials
            if not config.KALSHI_API_KEY_ID:
                raise ValueError(
                    "KALSHI_API_KEY_ID is empty in config.py.\n"
                    "Get your API key from https://kalshi.com → Account → API Keys,\n"
                    "then fill it in config.py before running live."
                )
            key_path = Path(config.KALSHI_PRIVATE_KEY_PATH)
            if not key_path.exists():
                raise FileNotFoundError(
                    f"RSA private key not found at '{key_path}'.\n"
                    "Download it when creating your API key at https://kalshi.com/account/api-keys"
                )
            self._client = KalshiClient(
                api_key_id          = config.KALSHI_API_KEY_ID,
                private_key_pem_path= key_path,
                host                = config.KALSHI_HOST,
            )
            print(f"  [bot_client] Authenticated with Kalshi API (key: {config.KALSHI_API_KEY_ID[:8]}...)")
        else:
            print("  [bot_client] Paper trade mode — no Kalshi auth needed for orders")

    # ── Public market data (no auth required) ─────────────────────────────────

    def get_market_info(self, ticker: str) -> dict[str, Any]:
        """
        Fetch a single market by ticker from the Kalshi public API.
        Returns the raw market dict.
        """
        try:
            data = public_get(
                f"markets/{ticker}",
                base_url=config.KALSHI_HOST + "/trade-api/v2",
            )
            return data.get("market", data)
        except Exception as exc:
            raise RuntimeError(f"[bot_client] get_market_info({ticker}) failed: {exc}") from exc

    # ── Authenticated portfolio operations ────────────────────────────────────

    def get_balance_cents(self) -> int:
        """
        Return available Kalshi balance in cents.

        In paper mode: returns a mock value (bankroll from config).
        In live mode:  calls the Kalshi portfolio balance endpoint.
        """
        if config.PAPER_TRADE:
            return int(config.BANKROLL_DOLLARS * 100)

        try:
            data = self._client.get("/portfolio/balance")
            # Response structure: {"balance": {"available_balance_cents": 5000, ...}}
            balance_obj = data.get("balance", {})
            return int(balance_obj.get("available_balance_cents", 0))
        except Exception as exc:
            raise RuntimeError(f"[bot_client] get_balance_cents() failed: {exc}") from exc

    # ── Order placement ───────────────────────────────────────────────────────

    def place_order(
        self,
        *,
        ticker:            str,
        side:              str,   # "yes" or "no"
        num_contracts:     int,
        limit_price_cents: int,   # the price for YOUR side (1–99 cents)
    ) -> dict[str, Any]:
        """
        Place a YES or NO limit order.

        Parameters
        ----------
        ticker            : Market ticker (e.g. "KXBTC15M-25MAY01-T95000")
        side              : "yes" to bet BTC goes up, "no" to bet BTC goes down
        num_contracts     : Number of $1-face-value contracts to buy
        limit_price_cents : Max price you'll pay per contract (1–99 cents)
                            For YES: how much you pay for a yes contract
                            For NO:  how much you pay for a no contract

        In PAPER_TRADE mode: prints the order and returns a mock confirmation.
        In LIVE mode: submits an IOC (immediate-or-cancel) limit order to Kalshi.

        Returns
        -------
        dict with order details (real API response or paper trade mock)
        """
        side = side.lower()
        if side not in ("yes", "no"):
            raise ValueError(f"side must be 'yes' or 'no', got: {side!r}")
        if num_contracts < 1:
            raise ValueError(f"num_contracts must be >= 1, got {num_contracts}")
        if not (1 <= limit_price_cents <= 99):
            raise ValueError(f"limit_price_cents must be 1–99, got {limit_price_cents}")

        order_id = str(uuid.uuid4())[:8].upper()
        total_cost = num_contracts * limit_price_cents / 100

        # ── Paper trade mode ──────────────────────────────────────────────────
        if config.PAPER_TRADE:
            mock = {
                "order_id":           f"PAPER-{order_id}",
                "ticker":             ticker,
                "side":               side,
                "num_contracts":      num_contracts,
                "limit_price_cents":  limit_price_cents,
                "total_cost_dollars": round(total_cost, 2),
                "status":             "paper_filled",
                "paper_trade":        True,
                "placed_at":          datetime.now(timezone.utc).isoformat(),
            }
            print(
                f"  [PAPER ORDER] {side.upper()} {num_contracts} × {ticker} "
                f"@ {limit_price_cents}¢  (${total_cost:.2f} total)"
            )
            return mock

        # ── Live order ────────────────────────────────────────────────────────
        # Kalshi order API v2:
        # POST /portfolio/orders
        # Both yes_price and no_price must always be provided; they must sum to 100.
        if side == "yes":
            yes_price = limit_price_cents
            no_price  = 100 - limit_price_cents
        else:
            no_price  = limit_price_cents
            yes_price = 100 - limit_price_cents

        body = {
            "ticker":        ticker,
            "action":        "buy",
            "side":          side,
            "type":          "limit",
            "time_in_force": "ioc",       # Immediate-or-cancel: fill or cancel instantly
            "count":         num_contracts,
            "yes_price":     yes_price,
            "no_price":      no_price,
        }

        try:
            resp = self._client.post("/portfolio/orders", json_body=body)
            print(
                f"  [LIVE ORDER] {side.upper()} {num_contracts} × {ticker} "
                f"@ {limit_price_cents}¢  order_id={resp.get('order', {}).get('order_id', '?')}"
            )
            return resp or {}
        except Exception as exc:
            raise RuntimeError(
                f"[bot_client] Live order failed ({side.upper()} {num_contracts}×{ticker}): {exc}"
            ) from exc
