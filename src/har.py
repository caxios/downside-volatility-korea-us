"""
HAR 계열 변동성 예측.

종속변수는 모든 모형에서 같다: 내일의 일별 변동성 RV_{t+1} (%²).
모형은 오른쪽(설명변수)만 다르다.

  HAR   : RV_t, RV_t^(w), RV_t^(m)
  LHAR  : HAR + 하락 크기 L_t = max(-r_t, 0) 와 그 주·월 평균
  RLHAR : LHAR + 레짐 더미 D_t 와 (RV_t, L_t) 의 상호작용
  SHAR  : RV_t 대신 RS+_t, RS-_t (장중 봉 필요)
  HARQ  : HAR + (sqrt(RQ_t) - 과거평균) * RV_t (장중 봉 필요)

로그 버전(권장): 분산 항을 로그로 바꾸고 log(RV_{t+1})을 추정한 뒤 되돌린다.
  - 선형(수준) 모형은 예측이 0 이하로 떨어질 수 있고, QLIKE는 과소예측에 매우 민감해서
    몇 개의 비현실적 예측이 비교 결과를 지배할 수 있다. 로그 모형은 예측이 항상 양수다.
  - 하락 크기 L은 수준 그대로 쓴다.

시점 규칙: t행의 설명변수는 t일 장 마감까지의 정보, y는 t+1일 RV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

HAR = ["RV", "RV_w", "RV_m"]
LEV = ["L", "L_w", "L_m"]
SPECS = {
    "HAR": HAR,
    "LHAR": HAR + LEV,
    "RLHAR": HAR + LEV + ["D", "RV_xD", "L_xD"],
    "SHAR": ["RSp", "RSn", "RV_w", "RV_m"],
    "HARQ": HAR + ["RQ_int"],
    "SHAR_L": ["RSp", "RSn", "RV_w", "RV_m"] + LEV,
}
LOG_SPECS = {
    "HAR": ["lRV", "lRV_w", "lRV_m"],
    "LHAR": ["lRV", "lRV_w", "lRV_m"] + LEV,
    "RLHAR": ["lRV", "lRV_w", "lRV_m"] + LEV + ["D", "lRV_xD", "L_xD"],
    "SHAR": ["lRSp", "lRSn", "lRV_w", "lRV_m"],
    "SHAR_L": ["lRSp", "lRSn", "lRV_w", "lRV_m"] + LEV,
}


def spec(name: str, log: bool = True) -> list:
    return (LOG_SPECS if log else SPECS)[name]


def har_features(rv: pd.Series, ret_pct: pd.Series | None = None,
                 rs_pos: pd.Series | None = None, rs_neg: pd.Series | None = None,
                 rq: pd.Series | None = None, regime: pd.Series | None = None) -> pd.DataFrame:
    """
    rv: 일별 변동성(%²), ret_pct: 일별 로그수익률(%), regime: t일 장 마감 기준 레짐(0/1)
    """
    d = pd.DataFrame({"RV": rv.astype(float)})
    d = d[d["RV"] > 0].dropna()
    d["RV_w"] = d["RV"].rolling(5).mean()
    d["RV_m"] = d["RV"].rolling(22).mean()
    if ret_pct is not None:
        L = np.maximum(-ret_pct.reindex(d.index), 0.0)
        d["L"] = L
        d["L_w"] = L.rolling(5).mean()
        d["L_m"] = L.rolling(22).mean()
    if rs_pos is not None and rs_neg is not None:
        d["RSp"] = rs_pos.reindex(d.index)
        d["RSn"] = rs_neg.reindex(d.index)
    if rq is not None:
        s = np.sqrt(rq.reindex(d.index))
        past_mean = s.expanding().mean().shift(1)          # 과거 평균만 사용
        d["RQ_int"] = (s - past_mean) * d["RV"]
    if regime is not None:
        D = regime.reindex(d.index).astype(float)
        d["D"] = D
        d["RV_xD"] = d["RV"] * D
        if "L" in d:
            d["L_xD"] = d["L"] * D
    # 로그 버전 컬럼
    for c in ("RV", "RV_w", "RV_m"):
        d["l" + c] = np.log(d[c])
    if "RSp" in d:
        eps = 1e-3 * d["RV"].median()
        d["lRSp"] = np.log(d["RSp"] + eps)
        d["lRSn"] = np.log(d["RSn"] + eps)
    if "D" in d:
        d["lRV_xD"] = d["lRV"] * d["D"]
    d["y"] = d["RV"].shift(-1)
    return d


def walk_forward(d: pd.DataFrame, cols: list, min_train: int = 1000, refit_every: int = 22,
                 window: int | None = None, log_target: bool = False) -> pd.Series:
    """
    확장(window=None) 또는 롤링 윈도우 워크포워드.
    각 재추정 시점에 그 이전 행들로만 OLS → 다음 refit_every 행을 예측.
    t행 예측은 t일 장 마감 시점에 가능 (학습에 쓰인 마지막 y = RV_t 는 이미 관측됨).
    예측값은 학습 구간 y의 [최소, 최대]로 절단 (비현실적 값 방지).
    """
    valid = d[cols].notna().all(axis=1)
    dd = d.loc[valid]
    preds = pd.Series(np.nan, index=d.index, name="pred")
    n = len(dd)
    Xall = np.column_stack([np.ones(n), dd[cols].to_numpy(float)])
    yall = dd["y"].to_numpy(float)
    for start in range(min_train, n, refit_every):
        lo = 0 if window is None else max(0, start - window)
        Xtr, ytr = Xall[lo:start], yall[lo:start]
        ok = np.isfinite(ytr)
        Xtr, ytr = Xtr[ok], ytr[ok]
        if len(ytr) < len(cols) + 10:
            continue
        target = np.log(ytr) if log_target else ytr
        beta, *_ = np.linalg.lstsq(Xtr, target, rcond=None)
        p = Xall[start:start + refit_every] @ beta
        if log_target:
            resid_var = np.var(target - Xtr @ beta)
            p = np.exp(p + 0.5 * resid_var)
        p = np.clip(p, ytr.min(), ytr.max())
        preds.loc[dd.index[start:start + refit_every]] = p
    return preds


def qlike(y, f):
    y = np.asarray(y, float)
    f = np.maximum(np.asarray(f, float), 1e-12)
    return y / f - np.log(y / f) - 1.0


def hac_mean_t(x, lags: int) -> tuple[float, float]:
    x = pd.Series(x).dropna()
    res = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params[0]), float(res.tvalues[0])


def compare_models(d: pd.DataFrame, preds: dict, bench: str = "HAR", lags: int = 10) -> pd.DataFrame:
    """
    preds: {모형명: 예측 Series}. 공통 표본(모든 예측과 y가 있는 날)에서 비교.
    DM_t > 0 이면 해당 모형이 bench보다 손실이 작음(더 좋음).
    """
    P = pd.DataFrame(preds)
    ok = P.notna().all(axis=1) & d["y"].reindex(P.index).notna() & (d["y"].reindex(P.index) > 0)
    y = d["y"].reindex(P.index)[ok]
    P = P[ok]
    rows = []
    loss_b = qlike(y, P[bench])
    for m in P.columns:
        lq = qlike(y, P[m])
        mse = (y - P[m]) ** 2
        dm = hac_mean_t(loss_b - lq, lags) if m != bench else (np.nan, np.nan)
        rows.append({"model": m, "n_oos": int(ok.sum()), "QLIKE": lq.mean(), "MSE": mse.mean(),
                     "QLIKE_improve_pct_vs_bench": 100 * (1 - lq.mean() / loss_b.mean()),
                     "DM_t_vs_bench": dm[1]})
    return pd.DataFrame(rows).set_index("model")


def full_sample_fit(d: pd.DataFrame, cols: list, lags: int = 22, log_target: bool = False) -> pd.DataFrame:
    """
    해석용 전체 표본 추정 (예측력 주장에는 쓰지 말 것).
    log_target=True: 로그 모형(LOG_SPECS)과 짝을 맞춰 log(y)를 추정한다 (walk_forward와 같은 모형).
    로그 설명변수에 수준 y를 쓰면 계수가 워크포워드 예측 모형과 다른 모형의 것이 된다.
    """
    dd = d.dropna(subset=cols + ["y"])
    dd = dd[dd["y"] > 0] if log_target else dd
    y = np.log(dd["y"]) if log_target else dd["y"]
    res = sm.OLS(y, sm.add_constant(dd[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return pd.DataFrame({"coef": res.params, "t_NW": res.tvalues, "p": res.pvalues})


def lookahead_sanity(d: pd.DataFrame, cols: list, log_target: bool = False, **kw) -> float:
    """
    미래값(y)을 일부러 넣었을 때 QLIKE가 거의 0이 되는지 확인 → 평가 과정이 누수를 감지하는지 점검.
    cheat는 모형의 예측 대상과 같은 단위로 넣어야 한다 (로그 모형이면 log(y)).
    """
    dd = d.copy()
    dd["cheat"] = np.log(dd["y"]) if log_target else dd["y"]
    p = walk_forward(dd, cols + ["cheat"], log_target=log_target, **kw)
    ok = p.notna() & dd["y"].notna()
    return float(np.mean(qlike(dd.loc[ok, "y"], p[ok])))