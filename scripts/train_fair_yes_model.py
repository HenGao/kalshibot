#!/usr/bin/env python3
"""
Train a small logistic model: P(YES settlement) from trade-time features.

Default features match fair_model.py / edge_bot inference:
  - log1p_seconds_before_close
  - yes_price   (trade print; at runtime edge_bot uses best YES ask)

Uses GroupShuffleSplit on market_ticker to reduce same-market leakage.

Writes JSON suitable for fair_model.FairYesModel.load_path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def ensure_ml_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add outcome_yes, yes_price, log1p_seconds_before_close if absent (old CSV)."""
    out = df.copy()
    if "outcome_yes" not in out.columns and "market_result" in out.columns:
        r = out["market_result"].astype(str).str.lower()
        out["outcome_yes"] = r.map({"yes": 1, "no": 0})
    if "yes_price" not in out.columns and "yes_price_dollars" in out.columns:
        out["yes_price"] = pd.to_numeric(out["yes_price_dollars"], errors="coerce")
    if "log1p_seconds_before_close" not in out.columns:
        if "seconds_before_close" not in out.columns:
            raise ValueError("CSV needs seconds_before_close or log1p_seconds_before_close")
        sec = pd.to_numeric(out["seconds_before_close"], errors="coerce")
        out["log1p_seconds_before_close"] = np.log1p(sec.clip(lower=0))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train fair YES logistic from enriched Kalshi CSV")
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("data/kxbtc15m_trades_enriched.csv"),
        help="Enriched trades CSV (from kalshi_btc_trade_timing.py)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/models/fair_yes_logistic.json"),
        help="Output JSON model path",
    )
    p.add_argument("--test-fraction", type=float, default=0.25, help="Holdout fraction by market")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    if not args.csv.is_file():
        print(f"Missing CSV: {args.csv}", file=sys.stderr)
        sys.exit(2)

    df = ensure_ml_columns(pd.read_csv(args.csv))
    if "market_ticker" not in df.columns:
        print("CSV missing market_ticker", file=sys.stderr)
        sys.exit(2)

    df = df.dropna(subset=["outcome_yes", "yes_price", "log1p_seconds_before_close"])
    df = df[(df["yes_price"] > 0) & (df["yes_price"] < 1)]
    df["outcome_yes"] = df["outcome_yes"].astype(int)

    y = df["outcome_yes"].to_numpy()
    groups = df["market_ticker"].astype(str).to_numpy()
    X = df[["log1p_seconds_before_close", "yes_price"]].astype(float).to_numpy()

    gss = GroupShuffleSplit(
        n_splits=1, test_size=args.test_fraction, random_state=args.seed
    )
    train_idx, test_idx = next(gss.split(X, y, groups))

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    pipe.fit(X[train_idx], y[train_idx])
    clf: LogisticRegression = pipe.named_steps["clf"]
    scaler: StandardScaler = pipe.named_steps["scaler"]

    proba = pipe.predict_proba(X[test_idx])[:, 1]
    try:
        auc = float(roc_auc_score(y[test_idx], proba))
    except ValueError:
        auc = float("nan")

    feature_names = ["log1p_seconds_before_close", "yes_price"]
    means = scaler.mean_.tolist()
    stds = scaler.scale_.tolist()
    coef = clf.coef_.ravel().tolist()
    intercept = float(clf.intercept_[0])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_names": feature_names,
        "coef": coef,
        "intercept": intercept,
        "means": means,
        "stds": stds,
        "meta": {
            "train_rows": int(train_idx.size),
            "test_rows": int(test_idx.size),
            "train_markets": int(pd.Series(groups[train_idx]).nunique()),
            "test_markets": int(pd.Series(groups[test_idx]).nunique()),
            "roc_auc_holdout": auc,
            "source_csv": str(args.csv.as_posix()),
        },
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} (holdout ROC-AUC={auc:.4f})")
    print(
        "Note: High AUC often means yes_price tracks settlement in-sample; "
        "that does not prove profitable forward edge after fees (see edge_bot).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
