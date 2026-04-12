#!/usr/bin/env python3
"""
Train btc-trend models from **settled KXBTC15M** outcomes (Kalshi public API) +
Coinbase 5m candles.

Default training recipe (override as needed):
  - **HistGradientBoostingClassifier** with early stopping on a holdout slice of train
  - **Time validation**: last ``--val-last-days`` of market close times (or ``--val-after-iso``)
  - **Platt** logit calibration on that validation slice (disable with ``--no-platt``)
  - Rows with non-finite features are dropped

Use ``--tape-aligned`` to match backtest tape timing (slow). Synthetic ``close_time - target_sec`` is the default.

Requires network.
"""

from __future__ import annotations

import argparse
import bisect
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
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
    fetch_coinbase_5m_range,
)

_sre_path = _ROOT / "scripts" / "simulate_recent_edge.py"
_sre_spec = importlib.util.spec_from_file_location("simulate_recent_edge", _sre_path)
assert _sre_spec and _sre_spec.loader
_sre = importlib.util.module_from_spec(_sre_spec)
sys.modules[_sre_spec.name] = _sre
_sre_spec.loader.exec_module(_sre)
iter_settled_markets = _sre.iter_settled_markets
outcome_yes_from_market = _sre.outcome_yes_from_market
parse_ts = _sre.parse_ts
pick_decision_trade = _sre.pick_decision_trade

_bt_path = _ROOT / "scripts" / "backtest_btc_trend_kxbtc15m.py"
_bt_spec = importlib.util.spec_from_file_location("btkx15m", _bt_path)
assert _bt_spec and _bt_spec.loader
_btk = importlib.util.module_from_spec(_bt_spec)
sys.modules[_bt_spec.name] = _btk
_bt_spec.loader.exec_module(_btk)
fetch_trades_for_ticker_limited = _btk.fetch_trades_for_ticker_limited

FEATURE_LIST = list(BTC_TREND_FEATURE_NAMES)
public_base = "https://api.elections.kalshi.com/trade-api/v2"


def row_features_finite(fd: dict[str, float]) -> bool:
    for k in FEATURE_LIST:
        v = fd.get(k)
        if v is None:
            return False
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(fv):
            return False
    return True


def parse_iso_to_unix(iso: str) -> int:
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def candle_starts(candles: list[list[Any]]) -> list[int]:
    return [int(c[0]) for c in candles]


def last_closed_bar_index(starts: list[int], t_dec: int) -> int:
    return bisect.bisect_right(starts, t_dec - 300) - 1


def btc_target_at_open(candles: list[list[Any]], starts: list[int], open_unix: int) -> float:
    j = bisect.bisect_right(starts, open_unix) - 1
    if j < 0:
        raise ValueError("no candle for open")
    return float(candles[j][3])


def write_calibration_report_json(path: Path, p: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_out: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            bins_out.append({"bin_lo": lo, "bin_hi": hi, "n": 0, "mean_p": None, "mean_y": None})
            continue
        bins_out.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "n": int(np.sum(mask)),
                "mean_p": float(np.mean(p[mask])),
                "mean_y": float(np.mean(y[mask])),
            }
        )
    path.write_text(json.dumps({"n_bins": n_bins, "bins": bins_out}, indent=2), encoding="utf-8")


def fit_platt_logit(p_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float]:
    """Return (coef, intercept) for sigmoid(coef * logit(p) + intercept)."""
    eps = 1e-6
    pc = np.clip(p_val, eps, 1.0 - eps)
    logit = np.log(pc / (1.0 - pc)).reshape(-1, 1)
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(logit, y_val)
    return float(lr.coef_.ravel()[0]), float(lr.intercept_[0])


def main() -> None:
    p = argparse.ArgumentParser(description="Train btc-trend on settled KXBTC15M + Coinbase")
    p.add_argument(
        "--days",
        type=int,
        default=120,
        help="Settled-time lookback (Kalshi markets + matching Coinbase window); use 180+ for longer history",
    )
    p.add_argument(
        "--target-sec",
        type=float,
        default=120.0,
        help="Seconds before close used as synthetic decision time",
    )
    p.add_argument("--test-fraction", type=float, default=0.2, help="Holdout tail by market close time")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model-kind",
        choices=("logistic", "hgb"),
        default="hgb",
        help="Default hgb (often generalizes better); use logistic for a linear baseline.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/models/btc_trend_kalshi_logistic.json"),
        help="Output JSON (logistic weights or HGB manifest)",
    )
    p.add_argument(
        "--hgb-joblib",
        type=Path,
        default=None,
        help="Path for HGB joblib (default: same stem as --out with .joblib)",
    )
    p.add_argument(
        "--no-platt",
        action="store_true",
        help="Disable Platt logit calibration on validation predictions (default: Platt on).",
    )
    p.add_argument(
        "--sample-weight",
        choices=("none", "vol"),
        default="none",
        help="vol: upweight high vol_15m rows on train",
    )
    p.add_argument("--sleep", type=float, default=0.02)
    p.add_argument(
        "--coinbase-pad-days",
        type=float,
        default=2.0,
        help="Extra Coinbase history before window",
    )
    p.add_argument("--max-markets", type=int, default=0, help="Cap markets (0=all)")
    p.add_argument(
        "--tape-aligned",
        action="store_true",
        help="Use public tape row nearest --target-sec (slow; matches backtest). Else synthetic close−target_sec.",
    )
    p.add_argument(
        "--max-lookback-sec",
        type=float,
        default=3600.0,
        help="With --tape-aligned, max seconds before close to search tape",
    )
    p.add_argument(
        "--trade-pages",
        type=int,
        default=8,
        help="Kalshi trade API pages per market when --tape-aligned",
    )
    p.add_argument(
        "--val-last-days",
        type=float,
        default=14.0,
        help="Validation = markets with close_time in the last N days (by train 'now'); train on older. "
        "Falls back to --test-fraction if too few rows. 0 = use --test-fraction only.",
    )
    p.add_argument(
        "--val-after-iso",
        type=str,
        default=None,
        help="UTC instant (e.g. 2025-03-01T00:00:00Z): validation = markets with close_time >= this; train = before. "
        "Overrides --val-last-days and --test-fraction.",
    )
    p.add_argument(
        "--calibration-report",
        type=Path,
        default=None,
        help="Write validation-set reliability bins (JSON) after training",
    )
    args = p.parse_args()

    now = int(time.time())
    min_settled = now - int(args.days * 86400)
    chart_start = min_settled - int(args.coinbase_pad_days * 86400) - 3600

    use_platt = not args.no_platt
    print(
        f"KXBTC15M Kalshi-label train | {args.days}d settled | target_sec={args.target_sec} | "
        f"{args.model_kind} | sample_weight={args.sample_weight} | "
        f"tape_aligned={args.tape_aligned} | platt={use_platt} | val_last_days={args.val_last_days}",
        flush=True,
    )
    print(f"Fetching Coinbase {chart_start} .. {now} ...", flush=True)
    candles = fetch_coinbase_5m_range(chart_start, now + 60, sleep_s=args.sleep)
    starts = candle_starts(candles)
    print(f"  {len(candles)} candles.", flush=True)

    rows: list[tuple[int, dict[str, float], int]] = []
    n_seen = 0
    n_skip_nonfinite = 0
    for m in iter_settled_markets(
        "KXBTC15M",
        min_settled_ts=min_settled,
        max_settled_ts=now,
        public_base=public_base,
        sleep_s=args.sleep,
    ):
        if args.max_markets and n_seen >= args.max_markets:
            break
        n_seen += 1
        ticker = m.get("ticker")
        if not ticker:
            continue
        oy = outcome_yes_from_market(m)
        if oy is None:
            continue
        close_raw = m.get("close_time")
        open_raw = m.get("open_time")
        if not close_raw or not open_raw:
            continue
        close_dt = parse_ts(str(close_raw))
        close_unix = int(close_dt.timestamp())
        open_unix = parse_iso_to_unix(str(open_raw))

        if args.tape_aligned:
            if args.sleep > 0:
                time.sleep(args.sleep)
            trades = fetch_trades_for_ticker_limited(
                str(ticker),
                public_base=public_base,
                sleep_s=args.sleep,
                max_pages=max(1, args.trade_pages),
            )
            tr = pick_decision_trade(
                trades,
                close_dt,
                target_sec=args.target_sec,
                max_lookback_sec=args.max_lookback_sec,
            )
            if tr is None:
                continue
            decision_dt = parse_ts(str(tr["created_time"]))
            t_dec = int(decision_dt.timestamp())
            sec_before = float(max(0.0, (close_dt - decision_dt).total_seconds()))
        else:
            t_dec = close_unix - int(args.target_sec)
            sec_before = float(max(0.0, close_unix - t_dec))

        if t_dec <= open_unix:
            continue

        try:
            tgt = btc_target_at_open(candles, starts, open_unix)
        except ValueError:
            continue

        j = last_closed_bar_index(starts, t_dec)
        if j < 72:
            continue
        lo = max(0, j - 299)
        klines_win = candles[lo : j + 1]
        try:
            fd = features_from_klines_window(
                klines_win,
                seconds_before_close=sec_before,
                target_usd=float(tgt),
                decision_ts_utc=t_dec,
            )
        except ValueError:
            continue

        if not row_features_finite(fd):
            n_skip_nonfinite += 1
            continue

        rows.append((close_unix, fd, int(oy)))

    rows.sort(key=lambda x: x[0])
    if len(rows) < 200:
        print(f"Too few rows: {len(rows)}", file=sys.stderr)
        sys.exit(2)

    if n_skip_nonfinite:
        print(f"  skipped_nonfinite_features={n_skip_nonfinite}", flush=True)

    vts: int | None = None
    split_mode: str
    if args.val_after_iso:
        vts = parse_iso_to_unix(args.val_after_iso)
        train_rows = [r for r in rows if r[0] < vts]
        val_rows = [r for r in rows if r[0] >= vts]
        if len(train_rows) < 100 or len(val_rows) < 30:
            print(
                f"Too few rows after val-after split: train={len(train_rows)} val={len(val_rows)}",
                file=sys.stderr,
            )
            sys.exit(2)
        split_mode = "val_after_iso"
    elif args.val_last_days > 0:
        vts = now - int(args.val_last_days * 86400)
        train_rows = [r for r in rows if r[0] < vts]
        val_rows = [r for r in rows if r[0] >= vts]
        if len(train_rows) < 100 or len(val_rows) < 30:
            print(
                f"Warning: val-last-days split weak (train={len(train_rows)} val={len(val_rows)}); "
                f"falling back to test_fraction={args.test_fraction}",
                flush=True,
            )
            split = int(len(rows) * (1.0 - args.test_fraction))
            split = max(split, 50)
            train_rows = rows[:split]
            val_rows = rows[split:]
            split_mode = "test_fraction_fallback"
            vts = None
        else:
            split_mode = "val_last_days"
    else:
        split = int(len(rows) * (1.0 - args.test_fraction))
        split = max(split, 50)
        train_rows = rows[:split]
        val_rows = rows[split:]
        split_mode = "test_fraction"
        vts = None

    meta_common: dict[str, Any] = {
        "split_mode": split_mode,
        "val_cutoff_unix": vts,
        "val_last_days_config": float(args.val_last_days),
        "skipped_nonfinite": int(n_skip_nonfinite),
        "platt_enabled": use_platt,
    }

    X_tr = np.array([[r[1][k] for k in FEATURE_LIST] for r in train_rows], dtype=np.float64)
    y_tr = np.array([r[2] for r in train_rows], dtype=np.int32)
    X_va = np.array([[r[1][k] for k in FEATURE_LIST] for r in val_rows], dtype=np.float64)
    y_va = np.array([r[2] for r in val_rows], dtype=np.int32)

    sw_tr: np.ndarray | None = None
    if args.sample_weight == "vol":
        col = FEATURE_LIST.index("vol_15m")
        med = float(np.median(X_tr[:, col])) or 1e-12
        w = 1.0 + np.clip(X_tr[:, col] / med, 0.0, 3.0)
        sw_tr = w.astype(np.float64)

    platt_coef: float | None = None
    platt_intercept: float | None = None

    if args.model_kind == "logistic":
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=4000, random_state=args.seed)),
            ]
        )
        if sw_tr is not None:
            pipe.fit(X_tr, y_tr, clf__sample_weight=sw_tr)
        else:
            pipe.fit(X_tr, y_tr)
        clf: LogisticRegression = pipe.named_steps["clf"]
        scaler: StandardScaler = pipe.named_steps["scaler"]
        proba_va = pipe.predict_proba(X_va)[:, 1]
        if use_platt and len(val_rows) >= 30:
            platt_coef, platt_intercept = fit_platt_logit(proba_va, y_va)
        try:
            auc = float(roc_auc_score(y_va, proba_va))
        except ValueError:
            auc = float("nan")

        if args.calibration_report is not None and len(val_rows) >= 20:
            write_calibration_report_json(args.calibration_report, proba_va, y_va)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model_type": "logistic",
            "feature_names": FEATURE_LIST,
            "coef": clf.coef_.ravel().tolist(),
            "intercept": float(clf.intercept_[0]),
            "means": scaler.mean_.tolist(),
            "stds": scaler.scale_.tolist(),
            "meta": {
                "train_rows": int(X_tr.shape[0]),
                "val_rows": int(X_va.shape[0]),
                "roc_auc_val": auc,
                "source": "KXBTC15M settled + Coinbase (tape or synthetic decision time)",
                "feature_schema": "v3",
                "tape_aligned": bool(args.tape_aligned),
                "val_after_iso": args.val_after_iso,
                "target_sec": args.target_sec,
                "days": args.days,
                "sample_weight": args.sample_weight,
                **meta_common,
            },
        }
        if platt_coef is not None and platt_intercept is not None:
            payload["platt_logit"] = {"coef": platt_coef, "intercept": platt_intercept}
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}  train={len(train_rows)} val={len(val_rows)} ROC-AUC(val)={auc:.4f}")
        return

    # HGB
    hgb = HistGradientBoostingClassifier(
        max_depth=6,
        max_iter=500,
        learning_rate=0.06,
        random_state=args.seed,
        l2_regularization=1e-2,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=25,
    )
    if sw_tr is not None:
        hgb.fit(X_tr, y_tr, sample_weight=sw_tr)
    else:
        hgb.fit(X_tr, y_tr)
    proba_va = hgb.predict_proba(X_va)[:, 1]
    if use_platt and len(val_rows) >= 30:
        platt_coef, platt_intercept = fit_platt_logit(proba_va, y_va)
    try:
        auc = float(roc_auc_score(y_va, proba_va))
    except ValueError:
        auc = float("nan")

    if args.calibration_report is not None and len(val_rows) >= 20:
        write_calibration_report_json(args.calibration_report, proba_va, y_va)

    job_path = args.hgb_joblib
    if job_path is None:
        job_path = args.out.with_suffix(".joblib")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "clf": hgb,
        "feature_names": FEATURE_LIST,
    }
    if platt_coef is not None and platt_intercept is not None:
        blob["platt_logit"] = {"coef": platt_coef, "intercept": platt_intercept}
    joblib.dump(blob, job_path)

    manifest: dict[str, Any] = {
        "model_type": "hist_gradient_boosting",  # loaders also accept: hgb, sklearn_hgb
        "feature_names": FEATURE_LIST,
        "sklearn_joblib": job_path.name,
        "meta": {
            "train_rows": int(X_tr.shape[0]),
            "val_rows": int(X_va.shape[0]),
            "roc_auc_val": auc,
            "source": "KXBTC15M settled + Coinbase (tape or synthetic decision time)",
            "feature_schema": "v3",
            "tape_aligned": bool(args.tape_aligned),
            "val_after_iso": args.val_after_iso,
            "target_sec": args.target_sec,
            "days": args.days,
            "sample_weight": args.sample_weight,
            **meta_common,
        },
    }
    if platt_coef is not None and platt_intercept is not None:
        manifest["platt_logit"] = {"coef": platt_coef, "intercept": platt_intercept}
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} + {job_path}  train={len(train_rows)} val={len(val_rows)} ROC-AUC(val)={auc:.4f}")


if __name__ == "__main__":
    main()
