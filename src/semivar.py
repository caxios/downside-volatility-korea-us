"""
상승/하락 반분산(realized semivariance)의 예측 관계 — H1의 직접 검정.

  RS⁺_t = r_t² · 1{r_t > 0},  RS⁻_t = r_t² · 1{r_t < 0},  RS⁺_t + RS⁻_t = r_t²   (vol.measures 와 같은 정의)

회귀 (Patton & Sheppard 2015 의 반분산 HAR 을 미래 상승/하락 반분산 각각에 적용):

  y_{t+1:t+h} = c + β⁺_d RS⁺_t + β⁻_d RS⁻_t + β⁺_w RS⁺_w,t + β⁻_w RS⁻_w,t + β⁺_m RS⁺_m,t + β⁻_m RS⁻_m,t + ε

  y ∈ {미래 RS⁺, 미래 RS⁻, 미래 RV}: t+1 ~ t+h 일의 일평균
  _w: 최근 5일 평균, _m: 최근 22일 평균
  표준오차: Newey-West (h일 목표가 겹치므로 시차 ≥ h)

검정
  개별 계수 β = 0            → t, p (양측, 정규근사)
  β⁻_k = β⁺_k (k = d, w, m)  → 하락과 상승의 영향이 같은가
  세 쌍 동시                 → Wald χ²(3)
  Σβ⁻ = Σβ⁺                  → 일·주·월을 합친 총영향이 같은가
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

POS = ["RSp_d", "RSp_w", "RSp_m"]
NEG = ["RSn_d", "RSn_w", "RSn_m"]
REGRESSORS = ["RSp_d", "RSn_d", "RSp_w", "RSn_w", "RSp_m", "RSn_m"]
TARGETS = {"RSp": "미래 상승 반분산", "RSn": "미래 하락 반분산", "RV": "미래 전체 분산"}


def semivar_features(r_pct: pd.Series, horizons=(1, 5, 22), windows=None) -> pd.DataFrame:
    """r_pct: 일별 로그수익률(%). 하루 수익률 하나로 그날의 RS±를 만든다 (일봉 버전)."""
    r = r_pct.dropna().astype(float)
    rsp = pd.Series(np.where(r > 0, r ** 2, 0.0), index=r.index)
    rsn = pd.Series(np.where(r < 0, r ** 2, 0.0), index=r.index)
    return semivar_frame(rsp, rsn, horizons, windows)


def semivar_frame(rsp: pd.Series, rsn: pd.Series, horizons=(1, 5, 22), windows=None) -> pd.DataFrame:
    """
    일별 RS⁺, RS⁻ (일봉 1개로 만든 것이든 장중 봉을 합산한 것이든) → 설명변수와 목표.
    t행의 설명변수는 t일 장 마감까지, 목표는 t+1 ~ t+h 일의 평균.
    windows: {접미사: 평균 일수}. 기본은 주(5일)·월(22일). 표본이 짧으면 {"w": 5}만 쓴다.
    """
    windows = {"w": 5, "m": 22} if windows is None else windows
    d = pd.DataFrame({"RSp_d": rsp.astype(float), "RSn_d": rsn.astype(float)}).dropna()
    d["RV_d"] = d["RSp_d"] + d["RSn_d"]
    for s in ("RSp", "RSn"):
        for k, n in windows.items():
            d[f"{s}_{k}"] = d[f"{s}_d"].rolling(n).mean()
    for h in horizons:
        for s in ("RSp", "RSn", "RV"):
            # t+1 ~ t+h 평균: 앞으로 h개를 보는 rolling 을 -h 만큼 당김
            d[f"y_{s}_h{h}"] = d[f"{s}_d"].rolling(h).mean().shift(-h)
    return d


def regressors(windows=None) -> tuple[list, list, list]:
    """(상승 변수, 하락 변수, 회귀 순서)"""
    keys = ["d"] + list(({"w": 5, "m": 22} if windows is None else windows).keys())
    pos, neg = [f"RSp_{k}" for k in keys], [f"RSn_{k}" for k in keys]
    return pos, neg, [c for pair in zip(pos, neg) for c in pair]


def nw_lags(h: int) -> int:
    """사전 확정: 겹치는 목표(h일)를 덮도록 max(22, 2h)"""
    return max(22, 2 * h)


def winsorize(df: pd.DataFrame, cols: list, q: float = 0.995) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c].clip(upper=out[c].quantile(q))
    return out


def fit(d: pd.DataFrame, y: str, xs: list, lags: int):
    dd = d.dropna(subset=xs + [y])
    return sm.OLS(dd[y], sm.add_constant(dd[xs])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def fit_panel(d: pd.DataFrame, y: str, xs: list, time_col: str, lags: int):
    """
    여러 종목을 묶은 합동 OLS. 표준오차는 Driscoll-Kraay (같은 날 종목 간 상관 + 날짜 간 자기상관에 강건).
    """
    dd = d.dropna(subset=xs + [y]).sort_values(time_col)
    t = pd.factorize(dd[time_col], sort=True)[0]
    return sm.OLS(dd[y], sm.add_constant(dd[xs])).fit(cov_type="hac-groupsum",
                                                      cov_kwds={"time": t, "maxlags": lags})


def coef_table(res, d: pd.DataFrame, y: str) -> pd.DataFrame:
    """계수, 표준오차, t, p, 표준화 계수(설명변수 1표준편차 변화 → 목표의 표준편차 단위 변화)"""
    dd = d.loc[res.model.data.row_labels]
    sd_y = dd[y].std()
    rows = {}
    for k in res.params.index:
        sd_x = dd[k].std() if k in dd else np.nan
        rows[k] = {"coef": res.params[k], "se_NW": res.bse[k], "t": res.tvalues[k], "p": res.pvalues[k],
                   "std_coef": res.params[k] * sd_x / sd_y if k != "const" else np.nan}
    return pd.DataFrame(rows).T


def equality_tests(res, pos: list = POS, neg: list = NEG) -> pd.DataFrame:
    """
    각 방향의 영향이 있는가: Σβ⁺ = 0, Σβ⁻ = 0 (총영향), 방향별 계수 동시 0
    하락 계수 = 상승 계수 검정: diff = β⁻ − β⁺ (양수면 하락의 영향이 더 큼)
    """
    keys = [c.split("_")[-1] for c in pos]
    k = len(keys)
    rows = {}

    def one(expr):
        t = res.t_test(expr)
        return {"diff": float(np.squeeze(t.effect)), "se_NW": float(np.squeeze(t.sd)),
                "stat": float(np.squeeze(t.tvalue)), "df": 1, "p": float(np.squeeze(t.pvalue))}

    def joint(expr):
        w = res.wald_test(expr, scalar=True, use_f=False)
        return {"diff": np.nan, "se_NW": np.nan, "stat": float(w.statistic), "df": k, "p": float(w.pvalue)}

    for name, cols in (("상승", pos), ("하락", neg)):
        rows[f"{name} 총영향: Σβ = 0"] = one(" + ".join(cols) + " = 0")
        rows[f"{name} 동시: " + " = ".join(f"β_{x}" for x in keys) + " = 0"] = joint(", ".join(f"{c} = 0" for c in cols))
    for p, n in zip(pos, neg):
        rows[f"{n.split('_')[-1]}: β⁻ − β⁺"] = one(f"{n} - {p} = 0")
    rows["합계: Σβ⁻ − Σβ⁺"] = one(" + ".join(neg) + " - " + " - ".join(pos) + " = 0")
    rows[f"동시: β⁻ = β⁺ ({','.join(keys)})"] = joint(", ".join(f"{n} = {p}" for p, n in zip(pos, neg)))
    return pd.DataFrame(rows).T


def univariate_assoc(d: pd.DataFrame, xs: list, ys: list, lags_of) -> pd.DataFrame:
    """
    단순 연관성: 표준화한 x, y 의 단순회귀 기울기 = 피어슨 상관. p 는 NW 표준오차로 계산
    (겹치는 목표 때문에 보통의 상관 p값은 과대평가됨). 스피어만 순위상관도 함께 보고.
    """
    rows = []
    for y, h in ys:
        for x in xs:
            dd = d[[x, y]].dropna()
            z = (dd - dd.mean()) / dd.std()
            res = sm.OLS(z[y], sm.add_constant(z[x])).fit(cov_type="HAC", cov_kwds={"maxlags": lags_of(h)})
            rows.append({"y": y, "h": h, "x": x, "pearson": res.params[x], "t_NW": res.tvalues[x],
                         "p_NW": res.pvalues[x], "spearman": stats.spearmanr(dd[x], dd[y])[0], "n": len(dd)})
    return pd.DataFrame(rows)
