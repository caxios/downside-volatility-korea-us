"""
S1v2: 변동성 예측 기반 비중 조절의 개선판 (12단계).

S1_B(로그 LHAR, 내일 하루 변동성 예측 → 변동성 타기팅)를 두 방향으로 넓힌다.
  1) 예측 모형: LHAR (하락 폭만) vs LUHAR (하락 폭 + 상승 폭, 반분산 분석의 대칭 버전)
  2) 예측 지평: h = 1일 vs h = 22일 (앞으로 한 달 평균 변동성)

후보 4개 = {LHAR, LUHAR} × {1, 22}. LHAR·h=1은 기존 S1_B와 같은 모형이다(기준선).

시점 규칙
  - t행의 설명변수는 t일 장 마감까지의 정보, 목표는 t+1 ~ t+h일 평균 RV.
  - 재추정 시점 start에서, 목표가 이미 모두 실현된 행(i + h ≤ start - 1 + 1 → i ≤ start - h)만 학습에 쓴다.
    h = 1이면 기존 har.walk_forward와 같다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .har import har_features

BASE = ["lRV", "lRV_w", "lRV_m"]
LEV = ["L", "L_w", "L_m"]
UP = ["U", "U_w", "U_m"]
MODELS = {"LHAR": BASE + LEV, "LUHAR": BASE + LEV + UP}
HORIZONS = (1, 22)


def features(rv: pd.Series, ret_pct: pd.Series, h: int) -> pd.DataFrame:
    d = har_features(rv, ret_pct)
    U = np.maximum(ret_pct.reindex(d.index), 0.0)
    d["U"] = U
    d["U_w"] = U.rolling(5).mean()
    d["U_m"] = U.rolling(22).mean()
    # 목표: t+1 ~ t+h 평균 RV
    d["y"] = d["RV"][::-1].rolling(h).mean()[::-1].shift(-1)
    return d


def walk_forward_h(d: pd.DataFrame, cols: list, h: int, min_train: int = 1000,
                   refit_every: int = 22) -> pd.Series:
    """로그 목표 확장 창 워크포워드. 학습은 목표가 모두 실현된 행만 (h행 간격)."""
    valid = d[cols].notna().all(axis=1)
    dd = d.loc[valid]
    preds = pd.Series(np.nan, index=d.index, name="pred")
    n = len(dd)
    X = np.column_stack([np.ones(n), dd[cols].to_numpy(float)])
    y = dd["y"].to_numpy(float)
    for start in range(min_train, n, refit_every):
        hi = start - h + 1                    # 학습 행: [0, hi) → 마지막 행 i = start - h, 목표 끝 = start
        Xtr, ytr = X[:hi], y[:hi]
        ok = np.isfinite(ytr) & (ytr > 0)
        Xtr, ytr = Xtr[ok], ytr[ok]
        if len(ytr) < len(cols) + 10:
            continue
        t = np.log(ytr)
        beta, *_ = np.linalg.lstsq(Xtr, t, rcond=None)
        p = np.exp(X[start:start + refit_every] @ beta + 0.5 * np.var(t - Xtr @ beta))
        preds.loc[dd.index[start:start + refit_every]] = np.clip(p, ytr.min(), ytr.max())
    return preds


def all_forecasts(rv: pd.Series, ret_pct: pd.Series, end: str | None = None, **kw) -> dict:
    """{(모형, h): (예측 Series, 특징 DataFrame)}. end가 있으면 그 날짜까지의 자료만 쓴다."""
    if end is not None:
        rv, ret_pct = rv.loc[:end], ret_pct.loc[:end]
    out = {}
    for h in HORIZONS:
        d = features(rv, ret_pct, h)
        for name, cols in MODELS.items():
            out[(name, h)] = (walk_forward_h(d, cols, h, **kw), d)
    return out


def vt_weights(forecast: pd.Series, sigma_target: float) -> pd.Series:
    """변동성 타기팅: w = min(1, σ* / √예측분산). S1_B와 같은 형태."""
    return (sigma_target / np.sqrt(forecast.clip(lower=1e-8))).clip(upper=1.0).rename("w")


def cand_name(model: str, h: int) -> str:
    return f"{model}_h{h}"
