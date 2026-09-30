"""
H4 (공매도 금지 효과)와 H5 (한미 비교).

국가 간 이중차분은 두 나라의 월별 시그널 효과 차이 Δb_t = b_KR,t − b_US,t 를
금지 기간 더미에 회귀하는 형태로 구현한다. 이때 Ban 계수가 이중차분 추정치 θ3 이다.
(같은 달의 글로벌 충격은 차이를 취하면서 상쇄된다.)

금지 기간이 짧으므로(약 17개월) 표준오차만 믿지 말고,
개발 구간 안의 같은 길이 가짜 기간들로 만든 플라시보 분포와 비교한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .cross_section import to_unit_rank


def ban_dummies(months: pd.PeriodIndex, ban_start: str, ban_end: str) -> pd.DataFrame:
    s, e = pd.Period(ban_start, "M"), pd.Period(ban_end, "M")
    ban = ((months >= s) & (months <= e)).astype(float)
    post = (months > e).astype(float)
    return pd.DataFrame({"Ban": ban, "Post": post}, index=months)


def _hac(y, X, lags):
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def country_did(b_kr: pd.Series, b_us: pd.Series, ban_start: str, ban_end: str,
                D_kr: pd.Series | None = None, D_us: pd.Series | None = None, lags: int = 3,
                exclude_months=None) -> pd.DataFrame:
    delta = (b_kr - b_us).dropna().rename("delta")
    if exclude_months:
        delta = delta.drop([pd.Period(m, "M") for m in exclude_months if pd.Period(m, "M") in delta.index])
    X = ban_dummies(delta.index, ban_start, ban_end)
    if D_kr is not None:
        X["D_KR"] = D_kr.reindex(delta.index)
    if D_us is not None:
        X["D_US"] = D_us.reindex(delta.index)
    X = X.loc[:, X.std() > 0].dropna()
    y = delta.reindex(X.index)
    res = _hac(y, X, lags)
    return pd.DataFrame({"coef": res.params, "t_NW": res.tvalues})


def months_in(windows) -> set:
    """[(시작월, 끝월), ...] → 포함된 Period 월 집합"""
    out = set()
    for s, e in windows or ():
        out |= set(pd.period_range(s, e, freq="M"))
    return out


def placebo_ban(delta: pd.Series, window_len: int, dev_end: str, step: int = 1,
                lags: int = 3, exclude_windows=None) -> pd.Series:
    """
    개발 구간 안에서 길이 window_len개월짜리 가짜 금지 기간을 옮겨가며 Ban 계수를 추정.
    exclude_windows: 실제 금지 기간 [(시작월, 끝월)] — 그 달들을 표본에서 빼고,
    빠진 달을 건너뛰는(연속되지 않는) 가짜 기간은 쓰지 않는다.
    """
    d = delta.dropna()
    d = d[d.index <= pd.Period(dev_end, "M")]
    drop = months_in(exclude_windows)
    if drop:
        d = d[~d.index.isin(list(drop))]
    est = {}
    for i in range(0, len(d) - window_len + 1, step):
        start, end = d.index[i], d.index[i + window_len - 1]
        if (end - start).n != window_len - 1:          # 제외된 달을 건너뛰는 창
            continue
        ban = ((d.index >= start) & (d.index <= end)).astype(float)
        res = _hac(d.to_numpy(float), pd.DataFrame({"Ban": ban}, index=d.index), lags)
        est[str(start)] = float(res.params["Ban"])
    return pd.Series(est, name="placebo_ban_coef")


def placebo_pvalue(real: float, placebo: pd.Series) -> float:
    """가짜 기간 추정치 중 실제보다 절댓값이 크거나 같은 비율"""
    p = placebo.dropna()
    return float((p.abs() >= abs(real)).mean()) if len(p) else np.nan


def high_low_short(short_ratio: pd.Series, top: float = 0.3) -> tuple[set, set]:
    s = short_ratio.dropna()
    hi = set(s[s >= s.quantile(1 - top)].index)
    lo = set(s[s <= s.quantile(top)].index)
    return hi, lo


def within_korea_did(panel: pd.DataFrame, signal: str, high: set, low: set, ban_start: str,
                     ban_end: str, controls=("REV", "SIZE"), ret: str = "fwd_ret",
                     start_month: str | None = None) -> pd.DataFrame:
    """
    r_{i,t+1} = α_t + β1 S + β2 S·Ban + β3 S·High + β4 S·Ban·High + γ1 High + γ2 High·Ban + 통제 + e
    월 고정효과(α_t)는 월별 평균 차감으로 처리, 표준오차는 월 단위 군집.
    β4 : 공매도 제약이 걸린 종목에서만 금지 기간에 달라진 시그널 효과
    """
    p = panel[panel["ticker"].isin(high | low)].copy()
    if start_month:
        p = p[p["month"] >= pd.Period(start_month, "M")]
    p = p.dropna(subset=[signal, ret] + list(controls))
    p["S"] = p.groupby("month")[signal].transform(to_unit_rank)
    for c in controls:
        p[c + "_r"] = p.groupby("month")[c].transform(to_unit_rank)
    bd = ban_dummies(pd.PeriodIndex(p["month"]), ban_start, ban_end)
    p["Ban"] = bd["Ban"].to_numpy()
    p["High"] = p["ticker"].isin(high).astype(float)
    p["S_Ban"] = p["S"] * p["Ban"]
    p["S_High"] = p["S"] * p["High"]
    p["S_Ban_High"] = p["S"] * p["Ban"] * p["High"]
    p["High_Ban"] = p["High"] * p["Ban"]
    xcols = ["S", "S_Ban", "S_High", "S_Ban_High", "High", "High_Ban"] + [c + "_r" for c in controls]
    Z = p[xcols + [ret]].copy()
    Z = Z - Z.groupby(p["month"]).transform("mean")               # 월 고정효과 제거
    groups = pd.factorize(p["month"])[0]
    res = sm.OLS(Z[ret].to_numpy(float), Z[xcols].to_numpy(float)).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    return pd.DataFrame({"coef": res.params, "t_cluster": res.tvalues}, index=xcols)


def ic_difference(ic_kr: pd.Series, ic_us: pd.Series, D: pd.Series | None = None, lags: int = 3) -> pd.Series:
    """H5: 같은 달 두 나라 IC 차이의 평균 (레짐 조건부 선택)"""
    d = (ic_kr - ic_us).dropna().rename("dIC")
    if D is None:
        res = sm.OLS(d.to_numpy(float), np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        return pd.Series({"mean_KR_minus_US": res.params[0], "t_NW": res.tvalues[0], "n": len(d)})
    X = D.reindex(d.index).rename("D").to_frame().dropna()
    res = _hac(d.reindex(X.index), X, lags)
    return pd.Series({"calm_KR_minus_US": res.params["const"], "turb_extra": res.params["D"],
                      "t_turb_extra": res.tvalues["D"], "n": len(X)})
