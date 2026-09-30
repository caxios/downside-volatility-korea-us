"""
횡단면 분석 (H2, H3).

시점 규칙
  - t월의 시그널: t월 안의 일별 데이터로 계산
  - universe: t월 말 구성종목 (t+1월 구성종목을 쓰면 편입 효과가 섞임)
  - 예측 대상: t+1월 수익률 (t월 마지막 종가 → t+1월 마지막 종가)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .vol import measures


# ======================================================================
# 패널 구축
# ======================================================================

def monthly_signals(df: pd.DataFrame, mkt_r: pd.Series, min_days: int = 15) -> pd.DataFrame:
    """
    df: quality.clean_daily 결과 (r = 로그수익률, 소수)
    mkt_r: 시장 지수 일별 로그수익률 (소수)
    """
    per = df.index.to_period("M")
    adj = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    dvol = (df["Close"] * df.get("Volume", np.nan)).replace(0.0, np.nan)
    g = pd.DataFrame({"r": df["r"], "m": mkt_r.reindex(df.index), "dvol": dvol,
                      "vol0": df.get("flag_zero_vol", False), "close": df["Close"]}, index=df.index)
    rows = {}
    for p, x in g.groupby(per):
        xr = x["r"].dropna()
        if len(xr) < min_days:
            continue
        me = measures(xr.to_numpy())
        xm = x.dropna(subset=["r", "m"])
        ivol = np.nan
        if len(xm) >= min_days:
            X = np.column_stack([np.ones(len(xm)), xm["m"].to_numpy()])
            y = xm["r"].to_numpy()
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            ivol = float(np.std(y - X @ beta, ddof=2))
        rows[p] = {
            "RSJ": me["RSJ"], "RSkew": me["RSkew"], "RS_pos": me["RS_pos"], "RS_neg": me["RS_neg"],
            "RV": me["RV"], "MAX": me["MAX"], "REV": me["SUM"], "IVOL": ivol,
            "ILLIQ": float((x["r"].abs() / x["dvol"]).mean() * 1e6),
            "DVOL": float(np.log(x["dvol"].mean())) if x["dvol"].notna().any() else np.nan,
            "n_days": len(xr), "zero_vol_days": int(x["vol0"].sum()),
            "price_end": float(x["close"].iloc[-1]),
        }
    sig = pd.DataFrame(rows).T
    if sig.empty:
        return sig
    me_px = adj.groupby(per).last()
    first_px = adj.groupby(per).first()
    sig["MOM"] = (me_px.shift(1) / me_px.shift(12) - 1.0).reindex(sig.index)
    for h in (1, 2, 3, 6):
        sig[f"fwd_ret_{h}"] = (me_px.shift(-h) / me_px - 1.0).reindex(sig.index)
    sig["fwd_ret"] = sig["fwd_ret_1"]
    sig["fwd_ret_skip1"] = (me_px.shift(-1) / first_px.shift(-1) - 1.0).reindex(sig.index)
    return sig


def build_panel(daily: dict, mkt_r: pd.Series, membership: dict | None = None,
                mcap: pd.DataFrame | None = None, min_price: float = 0.0,
                max_zero_vol_days: int = 4, min_days: int = 15) -> pd.DataFrame:
    """
    daily: {ticker: clean_daily DataFrame}
    membership: {Period: set(ticker)} — t월 말 기준. None이면 필터 안 함.
    mcap: 행 Period, 열 ticker (한국). 없으면 SIZE = DVOL(거래대금) 대리변수.
    """
    frames = []
    for t, df in daily.items():
        s = monthly_signals(df, mkt_r, min_days)
        if s.empty:
            continue
        s = s.copy()
        s["ticker"] = t
        s["month"] = s.index
        frames.append(s.reset_index(drop=True))
    panel = pd.concat(frames, ignore_index=True)
    if membership is not None:
        keep = [t in membership.get(m, ()) for t, m in zip(panel["ticker"], panel["month"])]
        panel = panel[np.asarray(keep)]
    panel = panel[(panel["price_end"] >= min_price) & (panel["zero_vol_days"] <= max_zero_vol_days)]
    if mcap is not None:
        mc = mcap.stack().rename("mcap").reset_index()
        mc.columns = ["month", "ticker", "mcap"]
        panel = panel.merge(mc, on=["month", "ticker"], how="left")
        panel["SIZE"] = np.log(panel["mcap"])
    else:
        panel["SIZE"] = panel["DVOL"]
    return panel.reset_index(drop=True)


def missing_report(membership: dict, available: set) -> pd.DataFrame:
    rows = []
    for m, s in sorted(membership.items()):
        miss = s - available
        rows.append({"month": m, "n_members": len(s), "n_missing": len(miss),
                     "missing_share": len(miss) / max(len(s), 1)})
    return pd.DataFrame(rows).set_index("month")


# ======================================================================
# 기본 도구
# ======================================================================

def to_unit_rank(s: pd.Series) -> pd.Series:
    """순위를 -0.5 ~ +0.5로. 계수 = 최하위와 최상위의 수익률 차이"""
    n = s.count()
    if n < 2:
        return s * np.nan
    return (s.rank() - 1.0) / (n - 1.0) - 0.5


def nw_mean(x: pd.Series, lags: int = 3) -> tuple[float, float]:
    x = pd.Series(x).dropna()
    if len(x) < 5:
        return np.nan, np.nan
    res = sm.OLS(x.to_numpy(float), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params[0]), float(res.tvalues[0])


def restrict_months(panel: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    m = panel["month"]
    ok = pd.Series(True, index=panel.index)
    if start is not None:
        ok &= m >= pd.Period(start, "M")
    if end is not None:
        ok &= m <= pd.Period(end, "M")
    return panel[ok]


# ======================================================================
# Rank IC
# ======================================================================

def rank_ic(panel: pd.DataFrame, sig: str, ret: str = "fwd_ret", min_n: int = 30) -> pd.Series:
    def one(g):
        g = g[[sig, ret]].dropna()
        return g[sig].rank().corr(g[ret].rank()) if len(g) >= min_n else np.nan
    return panel.groupby("month")[[sig, ret]].apply(one).dropna().rename(f"IC_{sig}")


def ic_summary(panel: pd.DataFrame, sigs: list, ret: str = "fwd_ret", split: str | None = None,
               lags: int = 3, min_n: int = 30) -> pd.DataFrame:
    rows = []
    for s in sigs:
        ic = rank_ic(panel, s, ret, min_n)
        m, t = nw_mean(ic, lags)
        row = {"signal": s, "mean_IC": m, "sd_IC": ic.std(), "ICIR": ic.mean() / ic.std(),
               "t_NW": t, "pos_share": (ic > 0).mean(), "n_months": len(ic)}
        if split:
            sp = pd.Period(split, "M")
            row["IC_first_half"] = ic[ic.index <= sp].mean()
            row["IC_second_half"] = ic[ic.index > sp].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("signal")


def ic_decay(panel: pd.DataFrame, sig: str, horizons=(1, 2, 3, 6), min_n: int = 30) -> pd.DataFrame:
    rows = []
    for h in horizons:
        ic = rank_ic(panel, sig, f"fwd_ret_{h}", min_n)
        m, t = nw_mean(ic, max(3, h + 1))       # 겹치는 기간 → 시차 늘림
        rows.append({"h_months": h, "mean_IC": m, "t_NW": t})
    return pd.DataFrame(rows).set_index("h_months")


# ======================================================================
# 분위 포트폴리오
# ======================================================================

def quantile_returns(panel: pd.DataFrame, sig: str, q: int = 5, ret: str = "fwd_ret",
                     weight: str | None = None, min_n: int = 30) -> pd.DataFrame:
    def one(g):
        g = g.dropna(subset=[sig, ret])
        if len(g) < min_n:
            return pd.Series(np.nan, index=range(1, q + 1))
        b = pd.qcut(g[sig].rank(method="first"), q, labels=False) + 1
        if weight is None:
            return g[ret].groupby(b).mean()
        w = g[weight].clip(lower=0)
        return (g[ret] * w).groupby(b).sum() / w.groupby(b).sum()
    cols = [sig, ret] + ([weight] if weight else [])
    out = panel.groupby("month")[cols].apply(one)
    if isinstance(out, pd.Series):
        out = out.unstack()
    return out.dropna(how="all")


def quantile_summary(qret: pd.DataFrame, direction: int, bench: pd.Series | None = None,
                     lags: int = 3) -> pd.Series:
    """direction=-1: 1분위(낮은 시그널) 롱, 최고분위 숏"""
    q = qret.shape[1]
    lo, hi = qret.columns.min(), qret.columns.max()
    long_, short_ = (qret[lo], qret[hi]) if direction < 0 else (qret[hi], qret[lo])
    ls = long_ - short_
    m, t = nw_mean(ls, lags)
    means = qret.mean()
    out = pd.Series({f"Q{c}": means[c] for c in qret.columns})
    out["LS_mean"] = m
    out["LS_t"] = t
    out["LS_ann_sharpe"] = ls.mean() / ls.std() * np.sqrt(12) if ls.std() > 0 else np.nan
    out["monotonic_rho"] = pd.Series(range(1, q + 1)).corr(pd.Series(means.to_numpy()), method="spearman")
    if bench is not None:
        b = bench.reindex(qret.index)
        out["long_minus_bench"], out["long_minus_bench_t"] = nw_mean(long_ - b, lags)
        out["bench_minus_short"], out["bench_minus_short_t"] = nw_mean(b - short_, lags)
    return out


def ew_benchmark(panel: pd.DataFrame, ret: str = "fwd_ret") -> pd.Series:
    return panel.groupby("month")[ret].mean().rename("EW")


# ======================================================================
# Fama-MacBeth
# ======================================================================

def fama_macbeth(panel: pd.DataFrame, xcols: list, ret: str = "fwd_ret", lags: int = 3,
                 min_n: int = 30, sector_col: str | None = None):
    """매월 순위(-0.5~0.5) 회귀 → 계수 시계열의 평균과 NW t. 반환: (요약표, 월별 계수)"""
    coefs = {}
    for m, g in panel.groupby("month"):
        g = g.dropna(subset=xcols + [ret])
        if len(g) < max(min_n, len(xcols) + 5):
            continue
        X = g[xcols].apply(to_unit_rank)
        if sector_col:
            X = pd.concat([X, pd.get_dummies(g[sector_col], prefix="sec", drop_first=True, dtype=float)], axis=1)
        X = sm.add_constant(X, has_constant="add")
        coefs[m] = sm.OLS(g[ret].to_numpy(float), X.to_numpy(float)).fit().params[:len(xcols) + 1]
    B = pd.DataFrame(coefs, index=["const"] + xcols).T
    summ = pd.DataFrame({c: nw_mean(B[c], lags) for c in B.columns}, index=["coef", "t_NW"]).T
    summ["n_months"] = len(B)
    return summ, B


# ======================================================================
# 중립화, 이중정렬
# ======================================================================

def neutralize(panel: pd.DataFrame, sig: str, controls: list, sector_col: str | None = None,
               out_col: str | None = None) -> pd.DataFrame:
    out_col = out_col or f"{sig}_neu"
    parts = []
    for m, g in panel.groupby("month"):
        g = g.copy()
        ok = g[[sig] + controls].notna().all(axis=1)
        g[out_col] = np.nan
        if ok.sum() > len(controls) + 5:
            gg = g[ok]
            X = gg[controls].apply(to_unit_rank)
            if sector_col:
                X = pd.concat([X, pd.get_dummies(gg[sector_col], drop_first=True, dtype=float)], axis=1)
            X = sm.add_constant(X, has_constant="add")
            y = to_unit_rank(gg[sig])
            g.loc[ok, out_col] = sm.OLS(y.to_numpy(float), X.to_numpy(float)).fit().resid
        parts.append(g)
    return pd.concat(parts)


def dependent_double_sort(panel: pd.DataFrame, ctrl: str, sig: str, direction: int,
                          ret: str = "fwd_ret", q: int = 5, min_n: int = 50) -> pd.Series:
    """ctrl 분위 안에서 sig 분위를 나눠 롱숏 → ctrl 분위 평균"""
    def one(g):
        g = g.dropna(subset=[ctrl, sig, ret])
        if len(g) < min_n:
            return np.nan
        g = g.assign(c=pd.qcut(g[ctrl].rank(method="first"), q, labels=False))
        g["s"] = g.groupby("c")[sig].transform(lambda x: pd.qcut(x.rank(method="first"), q, labels=False))
        lo = g[g["s"] == 0].groupby("c")[ret].mean()
        hi = g[g["s"] == q - 1].groupby("c")[ret].mean()
        spread = (lo - hi) if direction < 0 else (hi - lo)
        return spread.mean()
    return panel.groupby("month")[[ctrl, sig, ret]].apply(one).dropna().rename(f"{sig}|{ctrl}")


# ======================================================================
# 레짐 조건부 (H3)
# ======================================================================

def regime_test(series: pd.Series, D: pd.Series, lags: int = 3) -> pd.Series:
    """series_t = θ0 + θ1 D_t → θ0 = 평온 평균, θ0+θ1 = 혼란 평균"""
    df = pd.concat([series.rename("y"), D.rename("D")], axis=1).dropna()
    res = sm.OLS(df["y"], sm.add_constant(df["D"])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return pd.Series({"calm_mean": res.params["const"], "diff_turb_minus_calm": res.params["D"],
                      "t_diff": res.tvalues["D"], "n_calm": int((df["D"] == 0).sum()),
                      "n_turb": int((df["D"] == 1).sum())})


def sigma_cs(panel: pd.DataFrame, ret: str = "fwd_ret") -> pd.Series:
    return panel.groupby("month")[ret].std().rename("sigma_cs")


def ic_by_episode(ic: pd.Series, D: pd.Series) -> pd.DataFrame:
    df = pd.concat([ic.rename("ic"), D.rename("D")], axis=1).dropna()
    grp = (df["D"] != df["D"].shift()).cumsum()
    rows = []
    for _, g in df.groupby(grp):
        if g["D"].iloc[0] == 1:
            rows.append({"start": g.index[0], "end": g.index[-1], "months": len(g), "mean_IC": g["ic"].mean()})
    return pd.DataFrame(rows)


def s2_signal_passes(neu_t: float, ic_h1: float, ic_h2: float, mono_rho: float,
                     long_minus_bench: float, crit: dict) -> tuple[bool, list]:
    reasons = []
    if not (abs(neu_t) >= crit["min_abs_t"]):
        reasons.append("중립화 IC t 부족")
    if crit.get("same_sign_halves") and not (np.sign(ic_h1) == np.sign(ic_h2) != 0):
        reasons.append("전반/후반 부호 불일치")
    if not (abs(mono_rho) >= crit["min_abs_monotonic"]):
        reasons.append("분위 단조성 부족")
    if crit.get("long_leg_positive") and not (long_minus_bench > 0):
        reasons.append("롱 다리 초과수익 없음")
    return len(reasons) == 0, reasons
