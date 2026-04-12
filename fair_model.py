"""Load a JSON fair-YES model (logistic on standardized features) and predict."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FairYesModel:
    feature_names: tuple[str, ...]
    coef: tuple[float, ...]
    intercept: float
    means: tuple[float, ...]
    stds: tuple[float, ...]
    # Optional Platt scaling on logit(p), fit on a time-ordered validation slice.
    platt_coef: float | None = None
    platt_intercept: float | None = None

    @classmethod
    def from_json_obj(cls, obj: dict[str, Any]) -> FairYesModel:
        names = tuple(obj["feature_names"])
        coef = tuple(float(x) for x in obj["coef"])
        means = tuple(float(x) for x in obj["means"])
        stds = tuple(float(x) for x in obj["stds"])
        if not (len(names) == len(coef) == len(means) == len(stds)):
            raise ValueError("feature_names, coef, means, stds length mismatch")
        for i, s in enumerate(stds):
            if s <= 0:
                raise ValueError(f"stds[{i}] must be positive, got {s}")
        platt = obj.get("platt_logit") or {}
        pc = platt.get("coef")
        pi = platt.get("intercept")
        return cls(
            feature_names=names,
            coef=coef,
            intercept=float(obj["intercept"]),
            means=means,
            stds=stds,
            platt_coef=float(pc) if pc is not None else None,
            platt_intercept=float(pi) if pi is not None else None,
        )

    @classmethod
    def load_path(cls, path: str | Path) -> FairYesModel:
        raw = Path(path).read_text(encoding="utf-8")
        return cls.from_json_obj(json.loads(raw))


def predict_fair_yes(model: FairYesModel, features: dict[str, float]) -> float:
    z = model.intercept
    for name, c, m, s in zip(
        model.feature_names, model.coef, model.means, model.stds, strict=True
    ):
        if name not in features:
            raise KeyError(f"missing feature {name!r}")
        x = float(features[name])
        z += c * (x - m) / s
    # numerically stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        p = 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        p = ez / (1.0 + ez)
    p = min(1.0 - 1e-9, max(1e-9, p))
    if model.platt_coef is not None and model.platt_intercept is not None:
        logit = math.log(p / (1.0 - p))
        z2 = model.platt_coef * logit + model.platt_intercept
        if z2 >= 0:
            ez2 = math.exp(-z2)
            p = 1.0 / (1.0 + ez2)
        else:
            ez2 = math.exp(z2)
            p = ez2 / (1.0 + ez2)
        p = min(1.0 - 1e-9, max(1e-9, p))
    return p


def fair_yes_decimal(model: FairYesModel, features: dict[str, float]) -> Decimal:
    return Decimal(str(predict_fair_yes(model, features)))
