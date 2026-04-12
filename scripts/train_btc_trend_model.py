#!/usr/bin/env python3
"""
Train P(BTC finishes at/above a nearby strike after H minutes) from Coinbase 5m candles.

Labels are synthetic (strike jitter around spot). Feature set matches live edge_bot v2
(vol, path, interactions, book placeholders).

Writes JSON compatible with fair_model.FairYesModel (same schema as train_fair_yes_model.py).

Requires network access to Coinbase public API.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from btc_trend_features import (  # noqa: E402
    BTC_TREND_FEATURE_NAMES,
    features_from_klines_window,
)

COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/BTC-USD/candles"


def download_coinbase_btcusd_5m(*, max_batches: int, sleep_s: float) -> list[list]:
    """Chronological candles [time, low, high, open, close, vol]; 300 bars per batch."""
    by_time: dict[int, list] = {}
    end_ts = int(time.time())
    for _ in range(max_batches):
        start_ts = end_ts - 300 * 300
        r = requests.get(
            COINBASE_CANDLES,
            params={"start": start_ts, "end": end_ts, "granularity": 300},
            timeout=50,
        )
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        for c in chunk:
            by_time[int(c[0])] = c
        end_ts = min(int(x[0]) for x in chunk) - 1
        if sleep_s > 0:
            time.sleep(sleep_s)
    return [by_time[k] for k in sorted(by_time)]


def main() -> None:
    p = argparse.ArgumentParser(description="Train BTC trend logistic from Coinbase 5m candles")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/models/btc_trend_logistic.json"),
        help="Output JSON model path",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=200,
        help="Coinbase requests (300 x 5m bars each); ~200 batches ≈ 200 days of 5m bars",
    )
    p.add_argument("--sleep", type=float, default=0.08, help="Pause between Coinbase calls")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for synthetic strikes/horizons")
    p.add_argument("--test-fraction", type=float, default=0.2, help="Holdout tail fraction (time-ordered)")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Downloading Coinbase BTC-USD 5m candles...", flush=True)
    raw = download_coinbase_btcusd_5m(max_batches=args.max_batches, sleep_s=args.sleep)
    closes = [float(row[4]) for row in raw]
    n = len(closes)
    if n < 500:
        print(f"Too few candles: {n}", file=sys.stderr)
        sys.exit(2)

    max_h = 36
    rows: list[dict] = []
    t0 = 100
    t1 = n - max_h - 2
    feature_list = list(BTC_TREND_FEATURE_NAMES)
    for t in range(t0, t1):
        h = int(rng.choice([3, 6, 12, 18, 36]))
        strike = float(closes[t] * (1.0 + rng.uniform(-0.012, 0.012)))
        if strike <= 0:
            continue
        y = int(closes[t + h] >= strike)
        klines = raw[: t + 1]
        try:
            bar_t = int(raw[t][0])
            fd = features_from_klines_window(
                klines,
                seconds_before_close=float(h * 300),
                target_usd=strike,
                decision_ts_utc=bar_t + 300,
            )
        except ValueError:
            continue
        row = {k: fd[k] for k in feature_list}
        row["y"] = y
        rows.append(row)

    if len(rows) < 5000:
        print(f"Warning: only {len(rows)} training samples", file=sys.stderr)

    X = np.array([[r[k] for k in feature_list] for r in rows], dtype=np.float64)
    y = np.array([r["y"] for r in rows], dtype=np.int32)
    split = int(len(rows) * (1.0 - args.test_fraction))
    split = max(split, 1000)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=args.seed)),
        ]
    )
    pipe.fit(X_tr, y_tr)
    clf: LogisticRegression = pipe.named_steps["clf"]
    scaler: StandardScaler = pipe.named_steps["scaler"]

    proba = pipe.predict_proba(X_te)[:, 1]
    try:
        auc = float(roc_auc_score(y_te, proba))
    except ValueError:
        auc = float("nan")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "logistic",
        "feature_names": feature_list,
        "coef": clf.coef_.ravel().tolist(),
        "intercept": float(clf.intercept_[0]),
        "means": scaler.mean_.tolist(),
        "stds": scaler.scale_.tolist(),
        "meta": {
            "train_rows": int(X_tr.shape[0]),
            "test_rows": int(X_te.shape[0]),
            "roc_auc_holdout": auc,
            "source": "coinbase BTC-USD 5m synthetic strike labels",
            "horizons_bars": [3, 6, 12, 18, 36],
            "feature_schema": "v3",
            "note": "Synthetic labels; align with Kalshi settlement for production.",
        },
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}  rows={len(rows)}  holdout ROC-AUC={auc:.4f}")


if __name__ == "__main__":
    main()
