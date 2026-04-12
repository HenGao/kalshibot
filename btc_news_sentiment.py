"""
Daily BTC headline “compression” + lexicon sentiment -> score in [-1, 1].

Used to tilt fair P(YES): positive news nudges fair_yes up, negative down, which
changes EV vs asks (same as shifting effective edge on YES vs NO).

Data sources (first match wins per call, with in-memory + disk cache):
  1. NEWSAPI_KEY env: NewsAPI everything (better for historical backtests).
  2. CoinDesk RSS (free; only contains recent items — sparse for old dates).

For backtests far in the past, set NEWSAPI_KEY or expect frequent neutral (0).
"""

from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import requests

_CACHE_TTL_SEC = 900.0
_rss_raw_cache: tuple[float, str] | None = None

# Disk cache: date -> { "score": float, "snippet_len": int, "source": str }
_DEFAULT_DISK = Path(__file__).resolve().parent / "data" / "btc_news_sentiment_cache.json"


_POS_WORDS = frozenset(
    {
        "bull",
        "bullish",
        "rally",
        "surge",
        "soar",
        "gain",
        "gains",
        "up",
        "high",
        "record",
        "breakout",
        "adoption",
        "approve",
        "approved",
        "etf",
        "inflow",
        "accumulation",
        "optimism",
        "upgrade",
        "positive",
        "momentum",
        "recovery",
        "rebound",
        "green",
        "support",
        "buy",
        "long",
    }
)
_NEG_WORDS = frozenset(
    {
        "bear",
        "bearish",
        "crash",
        "plunge",
        "drop",
        "falls",
        "fall",
        "down",
        "low",
        "hack",
        "stolen",
        "ban",
        "banned",
        "sec",
        "reject",
        "rejected",
        "lawsuit",
        "fraud",
        "scam",
        "outflow",
        "selloff",
        "sell",
        "short",
        "fear",
        "liquidation",
        "liquidations",
        "warning",
        "negative",
        "cut",
        "cuts",
        "loss",
        "losses",
    }
)


def _disk_cache_path() -> Path:
    p = os.environ.get("BTC_NEWS_CACHE_PATH")
    return Path(p) if p else _DEFAULT_DISK


def _load_disk() -> dict[str, Any]:
    path = _disk_cache_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_disk_row(key: str, payload: dict[str, Any]) -> None:
    path = _disk_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_disk()
    data[key] = payload
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def lexicon_sentiment(text: str) -> float:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    pos = sum(1 for w in words if w in _POS_WORDS)
    neg = sum(1 for w in words if w in _NEG_WORDS)
    tot = pos + neg
    if tot == 0:
        return 0.0
    # Smooth toward 0; clamp
    raw = (pos - neg) / (tot + 4.0)
    return max(-1.0, min(1.0, raw * 2.5))


def compress_headlines(titles_and_snippets: list[str], max_chars: int = 8000) -> str:
    """Single blob for lexicon scoring."""
    blob = "\n".join(s.strip() for s in titles_and_snippets if s.strip())
    return blob[:max_chars]


def _fetch_coindesk_rss_raw() -> str:
    global _rss_raw_cache
    now = time.time()
    if _rss_raw_cache and now - _rss_raw_cache[0] < _CACHE_TTL_SEC:
        return _rss_raw_cache[1]
    r = requests.get(
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        timeout=25,
        headers={"User-Agent": "kalshibot/1.0 (research)"},
    )
    r.raise_for_status()
    _rss_raw_cache = (now, r.text)
    return r.text


def _parse_rss_items(xml_text: str) -> list[tuple[datetime, str]]:
    """Return (published_utc, text) per item."""
    out: list[tuple[datetime, str]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    channel = root.find("channel")
    if channel is None:
        return out
    for item in channel.findall("item"):
        title_el = item.find("title")
        desc_el = item.find("description")
        pub_el = item.find("pubDate")
        title = (title_el.text or "").strip() if title_el is not None and title_el.text else ""
        desc = (desc_el.text or "").strip() if desc_el is not None and desc_el.text else ""
        if not title and not desc:
            continue
        pub_raw = (pub_el.text or "").strip() if pub_el is not None and pub_el.text else ""
        try:
            pub_dt = parsedate_to_datetime(pub_raw)
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            else:
                pub_dt = pub_dt.astimezone(timezone.utc)
        except (TypeError, ValueError):
            continue
        text = title
        if desc:
            text = f"{title}\n{re.sub('<[^<]+?>', '', desc)[:400]}"
        out.append((pub_dt, text))
    return out


def _fetch_newsapi_window(
    api_key: str,
    q: str,
    decision_utc: datetime,
) -> list[str]:
    """Articles published in [decision-24h, decision]."""
    end = decision_utc.astimezone(timezone.utc)
    start = end - timedelta(hours=24)
    # NewsAPI date params are calendar dates (UTC)
    from_s = start.strftime("%Y-%m-%d")
    to_s = end.strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": q,
        "from": from_s,
        "to": to_s,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": "100",
    }
    r = requests.get(url, params=params, headers={"X-Api-Key": api_key}, timeout=35)
    r.raise_for_status()
    data = r.json()
    arts = data.get("articles") or []
    texts: list[str] = []
    for a in arts:
        pt = a.get("publishedAt") or ""
        if not pt:
            continue
        try:
            if pt.endswith("Z"):
                pub = datetime.fromisoformat(pt.replace("Z", "+00:00"))
            else:
                pub = datetime.fromisoformat(pt)
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            pub = pub.astimezone(timezone.utc)
        except ValueError:
            continue
        if not (start <= pub <= end):
            continue
        t = (a.get("title") or "").strip()
        d = (a.get("description") or "").strip()
        if t or d:
            texts.append(f"{t}\n{d[:400]}")
    return texts


def sentiment_for_decision_utc(decision_utc: datetime) -> tuple[float, str]:
    """
    Sentiment in [-1,1] from compressed headlines in the 24h window ending at decision_utc.
    Returns (score, source_tag).
    """
    if decision_utc.tzinfo is None:
        decision_utc = decision_utc.replace(tzinfo=timezone.utc)
    else:
        decision_utc = decision_utc.astimezone(timezone.utc)

    key = f"{decision_utc.date().isoformat()}_{int(decision_utc.timestamp()) // 3600}"
    disk = _load_disk()
    if key in disk and isinstance(disk[key], dict) and "score" in disk[key]:
        return float(disk[key]["score"]), str(disk[key].get("source", "cache"))

    api_key = (os.environ.get("NEWSAPI_KEY") or os.environ.get("NEWSAPI_API_KEY") or "").strip()
    texts: list[str] = []
    src = "none"

    if api_key:
        try:
            texts = _fetch_newsapi_window(api_key, "bitcoin OR BTC OR crypto", decision_utc)
            src = "newsapi"
        except Exception:
            texts = []
            src = "newsapi_error"

    if not texts:
        try:
            raw = _fetch_coindesk_rss_raw()
            items = _parse_rss_items(raw)
            start = decision_utc - timedelta(hours=24)
            for pub_dt, txt in items:
                if start <= pub_dt <= decision_utc:
                    if "bitcoin" in txt.lower() or "btc" in txt.lower() or "crypto" in txt.lower():
                        texts.append(txt)
            src = "coindesk_rss"
        except Exception:
            texts = []
            src = "rss_error"

    blob = compress_headlines(texts)
    score = lexicon_sentiment(blob) if blob.strip() else 0.0
    _save_disk_row(
        key,
        {
            "score": score,
            "snippet_len": len(blob),
            "source": src,
            "n_items": len(texts),
        },
    )
    return score, src


def apply_btc_news_tilt_to_fair_yes(
    fair_yes: Decimal,
    sentiment: float,
    tilt: Decimal,
) -> Decimal:
    """
    Shift fair P(YES) by tilt * sentiment (sentiment in [-1,1]).
    Positive sentiment -> higher fair_yes (lean long YES / up).
    """
    if tilt <= 0:
        return fair_yes
    adj = float(fair_yes) + float(tilt) * float(sentiment)
    adj = max(0.01, min(0.99, adj))
    return Decimal(str(round(adj, 6)))
