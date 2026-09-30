"""
7c단계: 오늘(과거)의 상승/하락 우세 → 내일(미래)의 상승/하락 우세 예측 (목표 2-b)

  python scripts/step7c_direction.py

상승 비중 s = RS⁺ / (RS⁺ + RS⁻),  상승 우세 D = 1{s > 0.5}
  - 일봉: 과거 k일 / 미래 h일 동안의 일별 수익률로 RS± 합산 (k = 1이면 s는 그날 등락 여부와 같음)
  - 장중: 하루 RS± 는 장중 봉 수익률로 계산 (vol.intraday_daily_measures, 밤사이 제외), k·h일 합산

사전 확정 설정
  일봉 지수   : k, h ∈ {1, 5, 22}. 주 분석 = 개발 구간(2005~2020, 목표 창도 2020-12 이내), 재현 = 테스트 구간(2021~)
  60분봉(3년) : k, h ∈ {1, 5}.   지수 + 50종목 패널
  5·10분봉(60일): k, h ∈ {1, 5}.  지수 + 50종목 패널 (예비)
  표준오차: 지수 Newey-West, 패널 Driscoll-Kraay. 시차 = 일봉·60분봉 max(22, 2h), 5·10분봉 max(5, 2h)

검정
  (1) 회귀  s_future = a + β s_past           β < 0 이면 방향이 뒤집힘(반전), β > 0 이면 이어짐(지속)
  (2) 적중률 반전 규칙(미래 우세 = 과거의 반대)과 지속 규칙, 그리고 기준선(표본의 다수 방향을 항상 예측)
             차이 = 반전 규칙 적중 − 기준선 적중, 월/일별 적중 차이의 평균을 NW(DK) t 로 검정
  (3) 레짐    s_future = a + β s_past + c R + d (s_past × R),  R = S1 동결 모형의 실시간 레짐(1 = 혼란)   [일봉 지수]
  (4) 큰 하락일 이후: 그날 수익률이 표본 하위 5% 인 날 다음 h일 상승 비중 − 나머지 날 이후 평균   [일봉 지수]

결과: results/step7c/*.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import _bootstrap  # noqa: F401
from src.config import DEV_START, TEST_START
from src.data_io import load_intraday
from src.firewall import FirewallError, guard, is_frozen, log_access
from src.pipeline import load_market, load_series
from src.report import out_dir, save
from step7b_intraday_semivar import INDEX, NON_STOCK, SPECS, daily_semivar, market_of


def share_frame(rsp: pd.Series, rsn: pd.Series, ks, hs) -> pd.DataFrame:
    d = pd.DataFrame(index=rsp.index)
    for k in ks:
        P, N = rsp.rolling(k).sum(), rsn.rolling(k).sum()
        d[f"x{k}"] = (P / (P + N)).where(P + N > 0)
    for h in hs:
        P, N = rsp.rolling(h).sum().shift(-h), rsn.rolling(h).sum().shift(-h)
        d[f"y{h}"] = (P / (P + N)).where(P + N > 0)
    return d


def _fit(dd, y, xs, lags, time=None):
    X = sm.add_constant(dd[xs])
    if time is None:
        return sm.OLS(dd[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    t = pd.factorize(dd[time], sort=True)[0]
    return sm.OLS(dd[y], X).fit(cov_type="hac-groupsum", cov_kwds={"time": t, "maxlags": lags})


def _mean_test(v: pd.Series, lags, time=None):
    """평균 = 0 검정 (패널이면 날짜별 평균을 먼저 낸 뒤 NW)"""
    if time is not None:
        v = v.groupby(time).mean()
    res = sm.OLS(v.to_numpy(), np.ones(len(v))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params[0]), float(res.tvalues[0]), float(res.pvalues[0])


def tests(d: pd.DataFrame, k: int, h: int, lags: int, time=None) -> dict:
    x, y = f"x{k}", f"y{h}"
    cols = [x, y] + ([time] if time else [])
    dd = d[cols].dropna().sort_values(time) if time else d[cols].dropna()
    res = _fit(dd, y, [x], lags, time)
    past_up, fut_up = dd[x] > 0.5, dd[y] > 0.5
    majority_up = fut_up.mean() >= 0.5
    hit_rev = (past_up != fut_up).astype(float)
    hit_base = (fut_up == majority_up).astype(float)
    diff, t_diff, p_diff = _mean_test(hit_rev - hit_base, lags, dd[time] if time else None)
    _, t_rev50, p_rev50 = _mean_test(hit_rev - 0.5, lags, dd[time] if time else None)
    return {"k": k, "h": h, "n": len(dd), "beta": res.params[x], "t": res.tvalues[x], "p": res.pvalues[x],
            "future_up_rate": fut_up.mean(), "hit_reversal": hit_rev.mean(), "hit_continuation": 1 - hit_rev.mean(),
            "p_reversal_vs_50": p_rev50, "hit_baseline": hit_base.mean(), "reversal_minus_baseline": diff,
            "t_vs_baseline": t_diff, "p_vs_baseline": p_diff}


def regime_test(d, reg, k, h, lags):
    x, y = f"x{k}", f"y{h}"
    dd = d[[x, y]].join(reg.rename("R"), how="inner").dropna()
    dd["xR"] = dd[x] * dd["R"]
    res = _fit(dd, y, [x, "R", "xR"], lags)
    tt = res.t_test(f"{x} + xR = 0")
    return {"k": k, "h": h, "n": len(dd), "turb_share": dd["R"].mean(),
            "beta_calm": res.params[x], "t_calm": res.tvalues[x], "p_calm": res.pvalues[x],
            "beta_turb": float(np.squeeze(tt.effect)), "t_turb": float(np.squeeze(tt.tvalue)),
            "p_turb": float(np.squeeze(tt.pvalue)),
            "diff_turb_minus_calm": res.params["xR"], "t_diff": res.tvalues["xR"], "p_diff": res.pvalues["xR"]}


def big_drop_test(r, d, h, lags):
    """그날 수익률 하위 5% (큰 하락일) 다음 h일 상승 비중 − 나머지 날 다음 h일 상승 비중"""
    y = f"y{h}"
    dd = pd.DataFrame({"r": r, y: d[y]}).dropna()
    cut = dd["r"].quantile(0.05)
    big = (dd["r"] <= cut).astype(float)
    res = _fit(dd.assign(big=big), y, ["big"], lags)
    return {"h": h, "cut_ret_pct": cut, "n_big": int(big.sum()), "share_after_big": dd.loc[big == 1, y].mean(),
            "share_after_other": dd.loc[big == 0, y].mean(), "diff": res.params["big"], "t": res.tvalues["big"],
            "p": res.pvalues["big"]}


def main():
    missing = [x for x in ("S1", "S2S3") if not is_frozen(x)]
    if missing:
        raise FirewallError(f"테스트 구간(2021~)을 쓰므로 먼저 동결할 것: {missing}")
    log_access("step7c direction", ("S1", "S2S3"))

    # ---------------- 일봉 지수
    rows, reg_rows, drop_rows = [], [], []
    for m in ("US", "KR"):
        r_all = load_market(m)["r_pct"].dropna()
        reg = load_series(out_dir("step2") / f"{m}_regime_full.parquet")
        for tag, r in (("dev", guard(r_all)), ("test", r_all)):
            rsp = pd.Series(np.where(r > 0, r ** 2, 0.0), r.index)
            rsn = pd.Series(np.where(r < 0, r ** 2, 0.0), r.index)
            d = share_frame(rsp, rsn, (1, 5, 22), (1, 5, 22))
            lo = DEV_START if tag == "dev" else TEST_START
            d, rr = d[d.index >= lo], r[r.index >= lo]
            for k in (1, 5, 22):
                for h in (1, 5, 22):
                    rows.append({"market": m, "sample": tag, **tests(d, k, h, max(22, 2 * h))})
                    reg_rows.append({"market": m, "sample": tag, **regime_test(d, reg, k, h, max(22, 2 * h))})
            for h in (1, 5, 22):
                drop_rows.append({"market": m, "sample": tag, **big_drop_test(rr, d, h, max(22, 2 * h))})
    save(pd.DataFrame(rows), "step7c", "daily_index", show=False)
    save(pd.DataFrame(reg_rows), "step7c", "daily_index_regime", show=False)
    save(pd.DataFrame(drop_rows), "step7c", "daily_index_after_big_drop", show=False)

    # ---------------- 장중 (60분 / 10분 / 5분)
    rows, cache = [], {}
    for freq, spec in SPECS.items():
        if spec["source"] not in cache:
            cache[spec["source"]] = load_intraday(spec["source"])
        sv = daily_semivar(cache[spec["source"]], spec)
        lag = (lambda h: max(22, 2 * h)) if freq == "60m" else (lambda h: max(5, 2 * h))
        for m in ("US", "KR"):
            x = sv[INDEX[m]]
            d = share_frame(x["RSp"], x["RSn"], (1, 5), (1, 5))
            frames = []
            for t, z in sv.items():
                if market_of(t) != m or t in NON_STOCK:
                    continue
                f = share_frame(z["RSp"], z["RSn"], (1, 5), (1, 5))
                f["date"] = f.index
                frames.append(f)
            P = pd.concat(frames, ignore_index=True)
            for k in (1, 5):
                for h in (1, 5):
                    rows.append({"freq": freq, "market": m, "unit": "index", **tests(d, k, h, lag(h))})
                    rows.append({"freq": freq, "market": m, "unit": "panel", **tests(P, k, h, lag(h), time="date")})
    save(pd.DataFrame(rows), "step7c", "intraday", show=False)


if __name__ == "__main__":
    main()
