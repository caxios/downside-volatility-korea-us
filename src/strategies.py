"""
전략 규칙.

S1: 주식 비중(얼마나)  — 레짐 스위치(A), LHAR 변동성 타기팅(B), 결합(C)
S2: 종목 선택(무엇을)  — 횡단면 시그널 결합 점수 상위 롱온리, 버퍼 규칙
S3: S2 종목 구성 × S1 비중
벤치마크: 바이앤홀드, 200일 이동평균, 단순 변동성 타기팅(과거 22일 실현변동성)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cross_section import to_unit_rank


# ======================================================================
# S1과 벤치마크 (모두 't일 장 마감 후 결정한 비중'을 반환)
# ======================================================================

def s1_weights(regime: pd.Series | None, rv_forecast: pd.Series | None, sigma_target: float | None,
               rule: str, turbulent_weight_c: float = 0.0) -> pd.Series:
    """
    regime: 0/1, rv_forecast: 내일 분산 예측(%²), sigma_target: 목표 일변동성(%)
    """
    if rule == "A":
        return (1.0 - regime.astype(float)).rename("w")
    wB = (sigma_target / np.sqrt(rv_forecast.clip(lower=1e-8))).clip(upper=1.0).rename("w")
    if rule == "B":
        return wB
    if rule == "C":
        reg = regime.reindex(wB.index)
        return wB.where(reg == 0, turbulent_weight_c).where(reg.notna()).rename("w")
    raise ValueError(rule)


def ma_rule_weights(close: pd.Series, window: int = 200) -> pd.Series:
    ma = close.rolling(window).mean()
    return (close > ma).astype(float).where(ma.notna()).rename("w")


def naive_voltarget_weights(ret_pct: pd.Series, sigma_target: float, window: int = 22) -> pd.Series:
    sig = ret_pct.rolling(window).std()
    return (sigma_target / sig).clip(upper=1.0).rename("w")


def buy_hold_weights(index: pd.Index) -> pd.Series:
    return pd.Series(1.0, index=index, name="w")


# ======================================================================
# S2: 종목 선택
# ======================================================================

def s2_scores(month_df: pd.DataFrame, signals: dict) -> pd.Series:
    """signals: {컬럼명: 방향(+1 높을수록 좋음, -1 낮을수록 좋음)} → 결합 점수(높을수록 좋음)"""
    parts = [to_unit_rank(month_df[c]) * float(d) for c, d in signals.items()]
    sc = pd.concat(parts, axis=1)
    return sc.mean(axis=1).where(sc.notna().all(axis=1)).dropna()


def s2_select(month_df: pd.DataFrame, signals: dict, entry_pct: float, buffer_pct: float,
              prev: set | None = None, max_weight: float = 0.05) -> pd.Series:
    sc = s2_scores(month_df.set_index("ticker") if "ticker" in month_df else month_df, signals)
    if sc.empty:
        return pd.Series(dtype=float)
    n_target = max(1, int(round(entry_pct * len(sc))))
    pct = sc.rank(pct=True)
    keep = [t for t in (prev or set()) if t in pct.index and pct[t] >= 1.0 - (entry_pct + buffer_pct)]
    keep = sorted(keep, key=lambda t: -sc[t])[:n_target]
    ranked = sc.sort_values(ascending=False).index
    new = [t for t in ranked if t not in keep][: max(0, n_target - len(keep))]
    names = keep + new
    w = pd.Series(min(1.0 / len(names), max_weight), index=names)
    return w / w.sum() if w.sum() > 1 else w          # max_weight에 걸리면 나머지는 현금


def run_s2(panel: pd.DataFrame, signals: dict, entry_pct: float, buffer_pct: float,
           max_weight: float = 0.05, months=None) -> dict:
    """반환: {Period(t): Series(ticker → 비중)} — t월 말 결정, t+1월 보유"""
    holdings, prev = {}, set()
    for m, g in panel.groupby("month"):
        if months is not None and m not in months:
            continue
        w = s2_select(g, signals, entry_pct, buffer_pct, prev, max_weight)
        if len(w):
            holdings[m] = w
            prev = set(w.index)
    return holdings


# ======================================================================
# S3: S2 종목 구성 × S1 비중
# ======================================================================

def run_s3(sleeve: pd.DataFrame, w_s1: pd.Series, cost_oneway: float, sell_tax: float,
           overlay_cost: float, rf: pd.Series | None = None) -> pd.DataFrame:
    """
    sleeve: backtest.sleeve_daily 결과 (gross, turnover, sells)
    w_s1  : t일 장 마감 후 결정한 주식 비중 → t+1일 적용
    """
    idx = sleeve.index
    held = w_s1.reindex(w_s1.index.union(idx)).ffill().shift(1).reindex(idx).fillna(0.0)
    rf_ = rf.reindex(idx).ffill().fillna(0.0) if rf is not None else 0.0
    sleeve_cost = held * (cost_oneway * sleeve["turnover"] + sell_tax * sleeve["sells"])
    d_held = held.diff().abs().fillna(held.abs())
    overlay = overlay_cost * d_held
    gross = held * sleeve["gross"] + (1.0 - held) * rf_
    return pd.DataFrame({"w": held, "gross": gross, "net": gross - sleeve_cost - overlay,
                         "turnover": sleeve["turnover"] * held + d_held})
