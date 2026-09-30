"""
13단계: 한국 공매도 전면 금지 기간(2023-11 ~ 2025-03)을 뺀 테스트 구간 강건성 점검 (2026-09-30 추가)

  python scripts/step13_ban_robustness.py

한국 테스트 구간(2021-01 ~)의 약 4분의 1이 공매도 금지 기간이다. 금지 기간이 한국 테스트 결과를
좌우했는지 보려고, 이미 동결·평가된 규칙을 그대로 두고 금지 기간 관측치만 빼서 다시 계산한다.
미국도 같은 달력 구간을 빼서 계산한다 (대조군: 미국 결과도 비슷하게 바뀌면 금지가 아니라 그 시기 탓).
새 규칙 선택은 없다. 사후 점검이므로 결론을 바꾸는 근거가 아니라 "금지 기간이 결론을 좌우했는가"만 본다.

결과: results/step13/*.csv
"""
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import BAN_END_MONTH, BAN_START_MONTH, MARKETS, PROC_DIR, RESULTS_DIR, TEST_START
from src.backtest import perf_stats
from src.cross_section import nw_mean
from src.firewall import guard, load_frozen
from src.har import compare_models, hac_mean_t
from src.multitest import holm
from src.pipeline import load_market, s1_all  # noqa: F401  (step12 모듈이 사용)
from src.report import save
from src.s5 import nw_mean as nw_mean_s5
from src.semivar import REGRESSORS, TARGETS, equality_tests, fit, nw_lags, semivar_features

import step12_s1v2_vol_timing as s12

R = RESULTS_DIR
MK = ("US", "KR")
B0 = pd.Period(BAN_START_MONTH, "M").start_time
B1 = pd.Period(BAN_END_MONTH, "M").end_time
HORIZONS = (1, 5, 22)
KEY = {"상승 총영향: Σβ = 0": "up", "하락 총영향: Σβ = 0": "down", "합계: Σβ⁻ − Σβ⁺": "diff"}


def in_ban(ts):
    ts = pd.DatetimeIndex(ts)
    return (ts >= B0) & (ts <= B1)


# ---------------------------------------------------------------- 1. 일별 반분산 HAR (테스트)
def semivar(market, r_all):
    d = semivar_features(r_all, HORIZONS)
    d = d[d.index >= pd.Timestamp(TEST_START)]
    pos = pd.Series(np.arange(len(r_all)), index=r_all.index)
    ban_pos = pos[in_ban(pos.index)]
    lo, hi = ban_pos.iloc[0], ban_pos.iloc[-1]
    rows = []
    for tag in ("full", "excl_ban"):
        for h in HORIZONS:
            dd = d
            if tag == "excl_ban":
                # 설명변수 창(t-21 ~ t)이나 목표 창(t+1 ~ t+h)이 금지 기간에 닿는 관측치를 모두 뺀다
                p = pos.reindex(dd.index)
                dd = dd[(p + h < lo) | (p - 21 > hi)]
            for s in TARGETS:
                res = fit(dd, f"y_{s}_h{h}", REGRESSORS, nw_lags(h))
                e = equality_tests(res)
                e = e[e.index.isin(list(KEY))]
                for name, r in e.iterrows():
                    rows.append({"market": market, "sample": tag, "target": s, "h": h, "test": KEY[name],
                                 "estimate": r["diff"], "t": r["diff"] / r["se_NW"], "p": r["p"], "n": int(res.nobs)})
    out = pd.DataFrame(rows)
    out["p_holm"] = out.groupby("sample")["p"].transform(lambda p: holm(p.to_numpy()))
    return out


# ---------------------------------------------------------------- 2. 주간 실현 왜도 롱숏 (S5, 테스트)
def skew(market):
    ls = pd.read_csv(R / "step10" / f"{market}_test_week_weekly.csv", index_col=0)
    start = pd.to_datetime([w.split("/")[0] for w in ls.index])
    hold0, hold1 = start + pd.Timedelta(days=7), start + pd.Timedelta(days=13)   # 보유 주 = 형성 주 다음 주
    touch = ((hold0 <= B1) & (hold1 >= B0))
    rows = []
    for tag, x in (("full", ls), ("excl_ban", ls[~touch])):
        for c in ("LS_EW_gross", "LS_VW_gross", "LS_EW_net", "UNIV_EW"):
            m, t, p = nw_mean_s5(x[c])
            rows.append({"market": market, "sample": tag, "portfolio": c, "weeks": int(x[c].notna().sum()),
                         "mean_weekly_bps": 1e4 * m, "t_NW": t, "p": p})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. S4 월간 초과수익 (테스트)
def s4(market):
    frozen = {"US": "S4_20_rel", "KR": "S4_30_abs"}[market]
    mm = pd.read_csv(R / "step9" / f"{market}_test_monthly.csv", index_col=0)
    mm.index = pd.PeriodIndex(mm.index, freq="M")
    ban = (mm.index >= pd.Period(BAN_START_MONTH, "M")) & (mm.index <= pd.Period(BAN_END_MONTH, "M"))
    rows = []
    for tag, x in (("full", mm), ("excl_ban", mm[~ban])):
        for vs in ("EW_universe", "ETF"):
            m, t = nw_mean(x[frozen] - x[vs], 3)
            rows.append({"market": market, "sample": tag, "strategy": frozen, "vs": vs, "months": len(x),
                         "ann_diff_pct": 1200 * m, "t_NW": t})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. S1v2·S1 비중 조절 (테스트)
def timing(market):
    rl = load_frozen("S1v2")["rules"][market]
    s1 = load_frozen("S1")["rules"][market]
    mk = s12.guarded_market(market, allow_test=True)
    from src.s1v2 import all_forecasts, cand_name, HORIZONS as HZ, MODELS
    fcs = all_forecasts(mk["rv"], mk["r_pct"])
    dev_fcs = all_forecasts(mk["rv"].loc[:"2020-12-31"], mk["r_pct"].loc[:"2020-12-31"])
    sigmas = {cand_name(m, h): float(np.sqrt(fc.dropna()).mean()) for (m, h), (fc, _) in dev_fcs.items()}
    rf = s12.rf_for(market)
    w = s12.strategy_weights(mk, fcs, sigmas, s1)
    keep = {"S1_A(동결)": "S1_A", rl["candidate"]: "S1v2", "naiveVT": "naiveVT", "MA200": "MA200", "BH": "BH"}
    # step12와 같은 전략 집합으로 돌려야 공통 날짜(=전체 표본 수치)가 step12와 일치한다
    _, nets = s12.run_backtests(mk, w, rl["cost_oneway"], rf, TEST_START, None)
    # 보고된 step12 테스트 표본과 같은 날짜로 자른다 (이후 모의 운용용 갱신으로 데이터가 늘었을 수 있음)
    n_pub = int(round(pd.read_csv(R / "step12" / f"{market}_TEST_performance.csv", index_col=0).loc["BH", "years"] * 252))
    nets = {k: v.iloc[:n_pub] for k, v in nets.items()}
    rows = []
    for tag in ("full", "excl_ban"):
        idx = nets["BH"].index
        idx = idx[~in_ban(idx)] if tag == "excl_ban" else idx
        for k, lab in keep.items():
            x = nets[k].reindex(idx)
            ps = perf_stats(x, 252, rf)
            sh = float(ps["sharpe"])
            row = {"market": market, "sample": tag, "strategy": lab, "days": len(x),
                   "ann_ret_pct": 100 * 252 * x.mean(), "sharpe": sh}
            if lab != "BH":
                m, t = hac_mean_t((nets[k] - nets["BH"]).reindex(idx), 10)
                row.update({"ann_diff_vs_BH_pct": 100 * 252 * m, "t_vs_BH": t})
            rows.append(row)
    # 예측 정확도: 22일 지평 LUHAR vs LHAR (한국 테스트에서 유일하게 유의했던 비교)
    acc = []
    for h in HZ:
        dd = fcs[("LHAR", h)][1]
        dd = dd[dd.index >= pd.Timestamp(TEST_START)]
        for tag in ("full", "excl_ban"):
            idx = dd.index
            if tag == "excl_ban":
                p = pd.Series(np.arange(len(dd)), index=dd.index)
                bp = p[in_ban(p.index)]
                idx = p[(p + h < bp.iloc[0]) | (p - 22 > bp.iloc[-1])].index
            preds = {cand_name(m, h): fcs[(m, h)][0].reindex(idx) for m in MODELS}
            c = compare_models(dd.loc[idx], preds, bench=cand_name("LHAR", h), lags=max(10, 2 * h))
            c["h"], c["sample"], c["market"] = h, tag, market
            acc.append(c)
    return pd.DataFrame(rows), pd.concat(acc)


def main():
    sv, sk, s4r, tm, acc = [], [], [], [], []
    for market in MK:
        mk = load_market(market)
        r_all = guard(mk["r_pct"], allow_test=True, required=("S1",), context=f"step13 ban robustness {market}")
        sv.append(semivar(market, r_all.dropna()))
        sk.append(skew(market))
        s4r.append(s4(market))
        a, b = timing(market)
        tm.append(a)
        acc.append(b)
    sv = pd.concat(sv)
    save(sv, "step13", "semivar_test_ban", show=False)
    summ = (sv.assign(sig=sv.p_holm < 0.05).groupby(["market", "sample", "test"])["sig"].agg(["sum", "count"])
            .unstack("test"))
    save(summ, "step13", "semivar_test_ban_summary")
    save(pd.concat(sk), "step13", "skew_test_ban")
    save(pd.concat(s4r), "step13", "s4_test_ban")
    save(pd.concat(tm), "step13", "timing_test_ban")
    save(pd.concat(acc), "step13", "timing_accuracy_test_ban")


if __name__ == "__main__":
    main()
