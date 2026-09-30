"""
시장 레짐 판정 (0 = 평온, 1 = 혼란).

JM (Statistical Jump Model)
  판정표(레짐 라벨의 줄) s_1..s_T 중에서
     Σ_t ½‖x_t − θ_{s_t}‖²  +  λ × (레짐이 바뀐 횟수)
  가 가장 작은 것을 고른다. θ_k 는 레짐 k의 전형적인 피처 값(중심).

  online 판정: 매일 V_t(k) = "t일까지의 데이터에 대해, t일을 k로 끝내는 판정표 중 최저 점수"를
  계산하고, 더 작은 쪽을 t일의 레짐으로 정한다. 과거 판정은 다시 고치지 않는다(역추적 안 함).
  중심 θ와 표준화 기준은 매월 말 그때까지의 데이터로만 다시 추정한다.

HMM (강건성 확인용)
  같은 일정으로 재추정하고, 필터링 확률 P(S_t = 혼란 | t일까지)를 직접 계산한다.
  hmmlearn 의 predict_proba 는 스무딩(미래 사용)이므로 쓰지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ======================================================================
# 피처
# ======================================================================

def jm_features(r: pd.Series, halflives=(5, 10, 21), rv: pd.Series | None = None,
                kind: str = "vol") -> pd.DataFrame:
    """
    r: 지수 일별 로그수익률(%), rv: 지수 일별 분산(%², 예: Parkinson)

    kind="vol" (기본, 이 프로젝트의 레짐 정의 = 변동성의 크기):
        로그 EWMA 분산(lrv_h), 하방편차(dd_h)
    kind="shu" (Shu-Yu-Mulvey 2024 방식, 추세까지 섞인 레짐):
        EWMA 수익률(ret_h), 하방편차(dd_h), 소르티노(sortino_h)
    """
    r = r.dropna()
    neg2 = np.minimum(r, 0.0) ** 2
    f = {}
    for h in halflives:
        dd = np.sqrt(neg2.ewm(halflife=h, min_periods=2 * h).mean())
        f[f"dd_{h}"] = dd
        if kind == "vol":
            if rv is None:
                raise ValueError("kind='vol'에는 rv가 필요함")
            f[f"lrv_{h}"] = np.log(rv.reindex(r.index).ewm(halflife=h, min_periods=2 * h).mean())
        elif kind == "shu":
            mu = r.ewm(halflife=h, min_periods=2 * h).mean()
            f[f"ret_{h}"] = mu
            f[f"sortino_{h}"] = mu / dd.replace(0.0, np.nan)
        else:
            raise ValueError(kind)
    return pd.DataFrame(f).replace([np.inf, -np.inf], np.nan).dropna()


def _standardize(train: pd.DataFrame, X: pd.DataFrame, clip: float):
    mu, sd = train.mean(), train.std().replace(0.0, 1.0)
    return ((X - mu) / sd).clip(-clip, clip).to_numpy(float), mu, sd


# ======================================================================
# JM 핵심 계산
# ======================================================================

def _loss(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return 0.5 * ((X[:, None, :] - theta[None, :, :]) ** 2).sum(-1)      # T × K


def _forward_pass_k2(L: np.ndarray, lam: float):
    """K=2 전용 빠른 경로 (순수 파이썬 스칼라 연산)"""
    T = L.shape[0]
    l0, l1 = L[:, 0].tolist(), L[:, 1].tolist()
    V0, V1 = [0.0] * T, [0.0] * T
    B0, B1 = [0] * T, [1] * T
    a, b = l0[0], l1[0]
    V0[0], V1[0] = a, b
    for t in range(1, T):
        sw_b = b + lam
        if sw_b < a:
            na, B0[t] = l0[t] + sw_b, 1
        else:
            na, B0[t] = l0[t] + a, 0
        sw_a = a + lam
        if sw_a < b:
            nb, B1[t] = l1[t] + sw_a, 0
        else:
            nb, B1[t] = l1[t] + b, 1
        a, b = na, nb
        V0[t], V1[t] = a, b
    return np.column_stack([V0, V1]), np.column_stack([B0, B1])


def forward_pass(L: np.ndarray, lam: float):
    """V[t,k] = t일까지, t일을 k로 끝내는 판정표의 최저 점수. back[t,k] = 그때 어제의 레짐"""
    if L.shape[1] == 2:
        return _forward_pass_k2(L, lam)
    T, K = L.shape
    Ll = L.tolist()
    V = [None] * T
    B = [None] * T
    prev = list(Ll[0])
    V[0] = prev
    B[0] = list(range(K))
    for t in range(1, T):
        row = Ll[t]
        cur = [0.0] * K
        bk = [0] * K
        for k in range(K):
            best, arg = prev[k], k
            for j in range(K):
                if j != k:
                    c = prev[j] + lam
                    if c < best:
                        best, arg = c, j
            cur[k] = row[k] + best
            bk[k] = arg
        V[t] = cur
        B[t] = bk
        prev = cur
    return np.asarray(V), np.asarray(B)


def dp_assign(X: np.ndarray, theta: np.ndarray, lam: float):
    """전체 기간 최적 판정표 (역추적 포함 → 사후 판정, 진단용)"""
    V, B = forward_pass(_loss(X, theta), lam)
    T = len(V)
    s = np.empty(T, dtype=int)
    s[-1] = int(V[-1].argmin())
    for t in range(T - 1, 0, -1):
        s[t - 1] = B[t, s[t]]
    return s, float(V[-1].min())


def fit_jump_model(X: np.ndarray, K: int = 2, lam: float = 30.0, n_init: int = 10,
                   max_iter: int = 100, seed: int = 0, init_thetas=None):
    """중심 갱신 ↔ 판정표 갱신을 번갈아 반복. 초기값 여러 개 중 최저 점수 채택."""
    rng = np.random.default_rng(seed)
    T = len(X)
    starts = [np.array(t, dtype=float) for t in (init_thetas or [])]
    while len(starts) < max(n_init, 1):
        starts.append(X[rng.choice(T, K, replace=False)].astype(float))
    best = (None, None, np.inf)
    for theta in starts:
        theta = theta.copy()
        prev = None
        for _ in range(max_iter):
            s, obj = dp_assign(X, theta, lam)
            for k in range(K):
                m = s == k
                theta[k] = X[m].mean(0) if m.any() else X[rng.integers(T)]
            if prev is not None and np.array_equal(s, prev):
                break
            prev = s
        s, obj = dp_assign(X, theta, lam)
        if obj < best[2]:
            best = (theta.copy(), s.copy(), obj)
    return best


def _order_by_downside(theta: np.ndarray, cols: list) -> np.ndarray:
    idx = [i for i, c in enumerate(cols) if c.startswith("dd_") or c.startswith("lrv_")]
    return np.argsort(theta[:, idx].mean(axis=1))          # 변동성·하방편차 작은 쪽 = 0(평온)


def _refit_dates(index: pd.DatetimeIndex, first_fit_end: str, freq: str = "M") -> list:
    s = index.to_series()
    ends = s.groupby(index.to_period(freq)).max()
    first = pd.Timestamp(first_fit_end)
    ends = ends[ends >= ends[ends <= first].max()] if (ends <= first).any() else ends
    return list(ends.values)


def online_jm(r_pct: pd.Series, lam: float, halflives=(5, 10, 21), first_fit_end: str = "2004-12-31",
              K: int = 2, clip: float = 5.0, refit: str = "M", n_init: int = 10, seed: int = 0,
              rv: pd.Series | None = None, kind: str = "vol") -> pd.Series:
    """
    반환: 날짜별 레짐 (0/1). 값은 그날 장 마감 시점에 알 수 있는 것만으로 계산됨.
    첫 재추정 시점 이전(burn-in) 날짜는 결과에 없음.
    """
    F = jm_features(r_pct, halflives, rv, kind)
    cols = list(F.columns)
    refits = [pd.Timestamp(x) for x in _refit_dates(F.index, first_fit_end, refit)]
    states = pd.Series(np.nan, index=F.index)
    theta_raw_prev = None
    for i, tau in enumerate(refits):
        train = F.loc[:tau]
        Xtr, mu, sd = _standardize(train, train, clip)
        inits = None
        if theta_raw_prev is not None:
            inits = [((theta_raw_prev - mu.values) / sd.values).clip(-clip, clip)]
        theta, _, _ = fit_jump_model(Xtr, K, lam, n_init=n_init if inits is None else 1,
                                     seed=seed + i, init_thetas=inits)
        theta = theta[_order_by_downside(theta, cols)]
        theta_raw_prev = theta * sd.values + mu.values

        seg_end = refits[i + 1] if i + 1 < len(refits) else F.index[-1]
        hist = F.loc[:seg_end]
        Xh, _, _ = _standardize(train, hist, clip)
        V, _ = forward_pass(_loss(Xh, theta), lam)
        s_online = V.argmin(axis=1)
        mask = (hist.index > tau) & (hist.index <= seg_end)
        states.loc[hist.index[mask]] = s_online[mask]
    return states.dropna().astype(int).rename("regime")


def offline_jm(r_pct: pd.Series, lam: float, halflives=(5, 10, 21), K: int = 2, clip: float = 5.0,
               n_init: int = 10, seed: int = 0, rv: pd.Series | None = None, kind: str = "vol") -> pd.Series:
    """전체 표본으로 추정하고 역추적한 사후 판정표. 진단(online과의 비교)에만 사용."""
    F = jm_features(r_pct, halflives, rv, kind)
    X, _, _ = _standardize(F, F, clip)
    theta, _, _ = fit_jump_model(X, K, lam, n_init=n_init, seed=seed)
    order = _order_by_downside(theta, list(F.columns))
    theta = theta[order]
    s, _ = dp_assign(X, theta, lam)
    return pd.Series(s, index=F.index, name="regime_offline")


# ======================================================================
# HMM (강건성 확인용)
# ======================================================================

def hmm_features(r_pct: pd.Series, rv: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"r": r_pct, "logrv": np.log(rv.where(rv > 0))}).dropna()


def _diag_loglik(X, means, vars_):
    return -0.5 * (np.log(2 * np.pi * vars_)[None, :, :] +
                   (X[:, None, :] - means[None, :, :]) ** 2 / vars_[None, :, :]).sum(-1)


def _hamilton_filter(X, means, vars_, P, pi0):
    """필터링 확률 P(S_t | t일까지). 확률 공간에서 매 시점 정규화 (K=2이면 스칼라 연산으로 빠르게)."""
    ll = _diag_loglik(X, means, vars_)
    lik = np.exp(ll - ll.max(axis=1, keepdims=True))          # 시점별 상수배는 정규화로 사라짐
    T, K = lik.shape
    out = np.empty((T, K))
    if K == 2:
        p00, p01, p10, p11 = P[0, 0], P[0, 1], P[1, 0], P[1, 1]
        a, b = float(pi0[0]), float(pi0[1])
        l0, l1 = lik[:, 0].tolist(), lik[:, 1].tolist()
        o1 = [0.0] * T
        for t in range(T):
            pa = a * p00 + b * p10                                 # ① 예측
            pb = a * p01 + b * p11
            na, nb = pa * l0[t], pb * l1[t]                        # ② 갱신
            z = na + nb
            a, b = (na / z, nb / z) if z > 0 else (pa, pb)
            o1[t] = b
        out[:, 1] = o1
        out[:, 0] = 1.0 - out[:, 1]
        return out
    prev = np.asarray(pi0, float)
    for t in range(T):
        post = (prev @ P) * lik[t]
        prev = post / post.sum()
        out[t] = prev
    return out


def online_hmm(F: pd.DataFrame, first_fit_end: str = "2004-12-31", K: int = 2, refit: str = "M",
               n_init: int = 3, seed: int = 0) -> pd.Series:
    """
    반환: 날짜별 필터링 확률 P(혼란 | 그날까지). 혼란 = 'r' 분산이 큰 상태.
    첫 추정은 초기값 n_init개, 이후에는 직전 추정치에서 이어서(warm start) 재추정한다.
    """
    from hmmlearn.hmm import GaussianHMM

    refits = [pd.Timestamp(x) for x in _refit_dates(F.index, first_fit_end, refit)]
    prob = pd.Series(np.nan, index=F.index, name="p_turbulent")
    prev_raw = None                     # (startprob, transmat, means_raw, vars_raw)
    for i, tau in enumerate(refits):
        train = F.loc[:tau]
        mu, sd = train.mean(), train.std().replace(0.0, 1.0)
        Xtr = ((train - mu) / sd).to_numpy(float)
        cands = []
        if prev_raw is not None:
            m = GaussianHMM(n_components=K, covariance_type="diag", n_iter=100, init_params="",
                            random_state=seed + i)
            m.startprob_, m.transmat_ = prev_raw[0], prev_raw[1]
            m.means_ = (prev_raw[2] - mu.values) / sd.values
            m.covars_ = np.maximum(prev_raw[3] / sd.values ** 2, 1e-6)
            cands.append(m)
        else:
            cands += [GaussianHMM(n_components=K, covariance_type="diag", n_iter=200,
                                  random_state=seed + 97 * i + k) for k in range(n_init)]
        best, best_ll = None, -np.inf
        for m in cands:
            try:
                m.fit(Xtr)
                ll = m.score(Xtr)
                if ll > best_ll:
                    best, best_ll = m, ll
            except Exception:
                continue
        if best is None:
            continue
        vars_ = best.covars_.reshape(K, -1) if best.covars_.ndim == 2 else \
            np.stack([np.diag(c) for c in best.covars_])
        order = np.argsort(vars_[:, 0])                       # 수익률 분산 작은 쪽 = 0
        means, vars_ = best.means_[order], vars_[order]
        P = best.transmat_[np.ix_(order, order)]
        prev_raw = (best.startprob_[order], P, means * sd.values + mu.values, vars_ * sd.values ** 2)
        pi0 = np.full(K, 1.0 / K)

        seg_end = refits[i + 1] if i + 1 < len(refits) else F.index[-1]
        hist = F.loc[:seg_end]
        Xh = ((hist - mu) / sd).to_numpy(float)
        filt = _hamilton_filter(Xh, means, vars_, P, pi0)
        mask = (hist.index > tau) & (hist.index <= seg_end)
        prob.loc[hist.index[mask]] = filt[mask, 1]
    return prob.dropna()


# ======================================================================
# 진단
# ======================================================================

def run_lengths(s: pd.Series) -> pd.DataFrame:
    grp = (s != s.shift()).cumsum()
    return s.groupby(grp).agg(state="first", length="size", start=lambda x: x.index[0])


def detection_lags(online: pd.Series, offline: pd.Series, max_wait: int = 60) -> pd.Series:
    """사후 판정 기준 혼란 에피소드 시작일부터 online이 혼란으로 바뀌기까지 걸린 거래일 수"""
    both = pd.concat([online.rename("on"), offline.rename("off")], axis=1).dropna()
    runs = run_lengths(both["off"])
    lags = {}
    pos = {d: i for i, d in enumerate(both.index)}
    for _, row in runs[runs["state"] == 1].iterrows():
        i0 = pos[row["start"]]
        window = both["on"].iloc[i0:i0 + row["length"] + max_wait]
        hit = np.flatnonzero(window.to_numpy() == 1)
        lags[row["start"]] = int(hit[0]) if len(hit) else np.nan
    return pd.Series(lags, name="lag_days")


def regime_diagnostics(states: pd.Series, r_pct: pd.Series, offline: pd.Series | None = None,
                       events: dict | None = None) -> dict:
    df = pd.concat([states.rename("s"), r_pct.rename("r")], axis=1).dropna()
    stats = df.groupby("s")["r"].agg(mean="mean", std="std", n="count", skew="skew")
    stats["ann_vol_pct"] = stats["std"] * np.sqrt(252)
    stats["share"] = stats["n"] / stats["n"].sum()
    runs = run_lengths(df["s"].astype(int))
    persist = runs.groupby("state")["length"].agg(avg_duration="mean", n_episodes="count")
    years = len(df) / 252.0
    out = {"stats": stats, "persistence": persist,
           "switches_per_year": float((df["s"].diff().abs() > 0).sum() / years)}
    if offline is not None:
        both = pd.concat([states, offline], axis=1).dropna()
        out["online_offline_agreement"] = float((both.iloc[:, 0] == both.iloc[:, 1]).mean())
        lags = detection_lags(states, offline)
        out["detection_lag_median"] = float(lags.median()) if len(lags) else np.nan
        out["detection_lags"] = lags
    if events:
        ev = {}
        for name, (a, b) in events.items():
            w = df.loc[a:b, "s"]
            if len(w):
                ev[name] = float(w.mean())
        out["event_turbulent_share"] = pd.Series(ev, name="turbulent_share")
    return out


def monthly_regime(states: pd.Series) -> pd.Series:
    """월말(그달 마지막 거래일) 레짐 → 월 단위 D_t"""
    s = states.dropna()
    return s.groupby(s.index.to_period("M")).last().rename("D")
