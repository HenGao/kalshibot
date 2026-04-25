"""Minimal signed HTTP client for Kalshi Trade API v2."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiClient:
    def __init__(
        self,
        *,
        api_key_id: str,
        private_key_pem_path: str | Path,
        host: str = "https://api.elections.kalshi.com",
        api_prefix: str = "/trade-api/v2",
    ) -> None:
        self.api_key_id = api_key_id
        self.host = host.rstrip("/")
        self.api_prefix = api_prefix.rstrip("/")
        self.base_url = f"{self.host}{self.api_prefix}"
        path = Path(private_key_pem_path)
        with path.open("rb") as f:
            self._private_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )

    def _sign(self, timestamp_ms: str, method: str, sign_path: str) -> str:
        msg = f"{timestamp_ms}{method}{sign_path}".encode("utf-8")
        sig = self._private_key.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("utf-8")

    def _headers(self, method: str, url: str) -> dict[str, str]:
        ts = str(int(time.time() * 1000))
        path_only = urlparse(url).path
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, path_only),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        if not path.startswith("/"):
            path = "/" + path
        url = self.base_url + path
        headers = self._headers(method.upper(), url)
        if json_body is not None:
            headers["Content-Type"] = "application/json"
        resp = requests.request(
            method.upper(),
            url,
            headers=headers,
            params=params,
            json=json_body,
            timeout=60,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"{method} {url} -> {resp.status_code}: {resp.text}"
            )
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self.request("GET", path, params=params)

    def post(self, path: str, json_body: dict[str, Any]) -> Any:
        return self.request("POST", path, json_body=json_body)

    def delete(self, path: str) -> Any:
        return self.request("DELETE", path)


def public_get(
    path: str,
    *,
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
    retries: int = 4,
) -> Any:
    """Unauthenticated GET (market data). Path is relative to base_url, e.g. markets/FOO/orderbook."""
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    last: BaseException | None = None
    for attempt in range(max(1, retries)):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code >= 400:
                raise RuntimeError(f"GET {url} -> {resp.status_code}: {resp.text}")
            return resp.json()
        except RuntimeError:
            raise
        except (requests.RequestException, OSError) as e:
            last = e
            if attempt < retries - 1:
                time.sleep(0.5 * (2**attempt))
            else:
                raise
    raise last  # pragma: no cover
