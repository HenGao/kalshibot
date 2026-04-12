"""
Load btc-trend models: logistic JSON (FairYesModel + optional Platt) or
HistGradientBoostingClassifier saved via joblib (manifest JSON).

Used by edge_bot and backtests.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import joblib
import numpy as np

from fair_model import FairYesModel, predict_fair_yes

BtcTrendPredictor = Union[FairYesModel, "HgbBtcTrendModel"]


@dataclass
class HgbBtcTrendModel:
    """Histogram-based gradient boosting + optional Platt on top of predicted logit."""

    clf: Any
    feature_names: tuple[str, ...]
    platt_coef: float | None = None
    platt_intercept: float | None = None

    def predict_proba_yes(self, features: dict[str, float]) -> float:
        x = np.array([[float(features[name]) for name in self.feature_names]], dtype=np.float64)
        p = float(self.clf.predict_proba(x)[0, 1])
        return _apply_platt_on_prob(p, self.platt_coef, self.platt_intercept)


def _apply_platt_on_prob(
    p: float,
    platt_coef: float | None,
    platt_intercept: float | None,
) -> float:
    if platt_coef is None or platt_intercept is None:
        return min(1.0 - 1e-9, max(1e-9, p))
    eps = 1e-9
    pc = min(1.0 - eps, max(eps, p))
    logit = math.log(pc / (1.0 - pc))
    z = platt_coef * logit + platt_intercept
    if z >= 0:
        ez = math.exp(-z)
        out = 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        out = ez / (1.0 + ez)
    return min(1.0 - 1e-9, max(1e-9, out))


def predict_btc_trend_yes(model: BtcTrendPredictor, features: dict[str, float]) -> float:
    if isinstance(model, HgbBtcTrendModel):
        return model.predict_proba_yes(features)
    return predict_fair_yes(model, features)


def load_btc_trend_predictor(path: str | Path) -> BtcTrendPredictor:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    if p.suffix.lower() in (".joblib", ".pkl", ".pickle"):
        blob: Any = joblib.load(p)
        if isinstance(blob, dict) and "clf" in blob and "feature_names" in blob:
            names = tuple(str(x) for x in blob["feature_names"])
            pl = blob.get("platt_logit") or {}
            return HgbBtcTrendModel(
                clf=blob["clf"],
                feature_names=names,
                platt_coef=(float(pl["coef"]) if pl.get("coef") is not None else None),
                platt_intercept=(float(pl["intercept"]) if pl.get("intercept") is not None else None),
            )
        raise ValueError(f"Unrecognized joblib payload in {p}")

    obj = json.loads(p.read_text(encoding="utf-8"))
    mt = (obj.get("model_type") or "logistic").lower()
    if mt in ("hist_gradient_boosting", "hgb", "sklearn_hgb"):
        rel = obj.get("sklearn_joblib")
        if not rel:
            raise ValueError("manifest missing sklearn_joblib")
        jp = (p.parent / rel).resolve()
        if not jp.is_file():
            raise FileNotFoundError(f"HGB joblib not found: {jp}")
        blob = joblib.load(jp)
        if isinstance(blob, dict) and "clf" in blob:
            clf = blob["clf"]
            blob_names = blob.get("feature_names")
        else:
            clf = blob
            blob_names = None
        names = tuple(str(x) for x in (obj.get("feature_names") or blob_names or ()))
        if not names:
            raise ValueError("HGB manifest missing feature_names")
        pl = obj.get("platt_logit") or {}
        if not pl and isinstance(blob, dict):
            pl = blob.get("platt_logit") or {}
        return HgbBtcTrendModel(
            clf=clf,
            feature_names=names,
            platt_coef=(float(pl["coef"]) if pl.get("coef") is not None else None),
            platt_intercept=(float(pl["intercept"]) if pl.get("intercept") is not None else None),
        )

    return FairYesModel.from_json_obj(obj)
