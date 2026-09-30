"""여러 단계에서 공통으로 쓰는 시장 데이터 적재와 S1 구성요소 계산."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (DATA_START, FIRST_FIT_END, HAR_MIN_TRAIN, HAR_REFIT_EVERY, JM_CLIP, JM_FEATURES,
                     JM_HALFLIVES, MARKETS, REGIME_MODEL)
from .data_io import download_daily
from .har import har_features, spec, walk_forward
from .quality import clean_daily
from .regime import hmm_features, online_hmm, online_jm
from .strategies import s1_weights
from .vol import parkinson


def load_market(market: str, refresh: bool = False) -> dict:
    cfg = MARKETS[market]
    tick = [cfg["index"], cfg["etf"]] + ([cfg["vix"]] if cfg["vix"] else [])
    raw = download_daily(tick, start=DATA_START, refresh=refresh)
    kr = market == "KR"
    idx = clean_daily(raw[cfg["index"]], kr=kr)
    etf = clean_daily(raw[cfg["etf"]], kr=kr)
    return {
        "cfg": cfg,
        "idx": idx,
        "etf": etf,
        "r_pct": (idx["r"] * 100.0).rename("r_pct"),      # 지수 로그수익률 (%)
        "rv": parkinson(idx).rename("RV"),                  # 지수 일별 분산 (%²)
        "etf_ret": np.expm1(etf["r"]).rename("etf_ret"),     # ETF 단순수익률 (배당 포함 수정주가)
        "close": idx["Close"],
        "vix": clean_daily(raw[cfg["vix"]]) if cfg["vix"] and cfg["vix"] in raw else None,
    }


def lhar_forecast(mk: dict, end: str | None = None) -> pd.Series:
    """로그 LHAR 워크포워드 예측 (t행 = t일 장 마감에 만든 t+1일 분산 예측)"""
    rv, r = mk["rv"], mk["r_pct"]
    if end is not None:
        rv, r = rv.loc[:end], r.loc[:end]
    d = har_features(rv, r)
    return walk_forward(d, spec("LHAR", log=True), HAR_MIN_TRAIN, HAR_REFIT_EVERY, log_target=True)


def market_regime(r_pct: pd.Series, rv: pd.Series, lam: float | None, model: str = REGIME_MODEL) -> pd.Series:
    """online 시장 레짐 (0 평온 / 1 혼란). model='jm'이면 lam 필요, 'hmm'이면 필터링 확률 > 0.5"""
    if model == "jm":
        return online_jm(r_pct, lam, JM_HALFLIVES, FIRST_FIT_END, clip=JM_CLIP, rv=rv, kind=JM_FEATURES)
    if model == "hmm":
        p = online_hmm(hmm_features(r_pct, rv.reindex(r_pct.index)), FIRST_FIT_END)
        return (p > 0.5).astype(int).rename("regime")
    raise ValueError(model)


def s1_all(mk: dict, lam: float | None, sigma_target: float | None = None, end: str | None = None,
           dev_end: str | None = None, turbulent_weight_c: float = 0.0, model: str = REGIME_MODEL) -> dict:
    """레짐, LHAR 예측, 규칙 A/B/C 비중을 한꺼번에 계산"""
    r = mk["r_pct"] if end is None else mk["r_pct"].loc[:end]
    states = market_regime(r, mk["rv"], lam, model)
    fc = lhar_forecast(mk, end)
    if sigma_target is None:
        ref = fc.loc[:dev_end] if dev_end else fc
        sigma_target = float(np.sqrt(ref.dropna()).mean())   # 개발 구간 평균 예측 변동성
    w = {rule: s1_weights(states, fc, sigma_target, rule, turbulent_weight_c) for rule in ("A", "B", "C")}
    return {"states": states, "forecast": fc, "sigma_target": sigma_target, "weights": w}


# ----------------------------------------------------------------- 패널 저장/불러오기 (Period ↔ 문자열)
def save_panel(panel: pd.DataFrame, path) -> None:
    p = panel.copy()
    p["month"] = p["month"].astype(str)
    p.to_parquet(path)


def load_panel(path) -> pd.DataFrame:
    p = pd.read_parquet(path)
    p["month"] = pd.PeriodIndex(p["month"], freq="M")
    return p


def save_series(s: pd.Series, path) -> None:
    s.to_frame(s.name or "value").to_parquet(path)


def load_series(path) -> pd.Series:
    df = pd.read_parquet(path)
    return df.iloc[:, 0]
