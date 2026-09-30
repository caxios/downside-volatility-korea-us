"""
4단계: 횡단면 분석 (H2, H3) — 개발 구간(2010~2020, CS_START~DEV_END)만

  python scripts/step4_cross_section.py

입력: data/processed/panel_{시장}.parquet, results/step2/{시장}_regime_dev.parquet
결과: results/step4/{시장}_T1..T9, {시장}_s2_selection.json
"""
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src import cross_section as cs
from src.config import CONTROLS, CS_START, DEV_END, MARKETS, NW_LAGS_MONTHLY, PROC_DIR, S2_CRITERIA, SIGNALS
from src.firewall import guard
from src.pipeline import load_panel, load_series
from src.regime import monthly_regime
from src.report import out_dir, save

LAG = NW_LAGS_MONTHLY
SPLIT = "2015-06"


def analyze(market):
    path = PROC_DIR / f"panel_{market}.parquet"
    if not path.exists():
        print(f"[{market}] 패널 없음 → 3단계 먼저 실행")
        return
    panel = guard(load_panel(path), month_col="month")
    panel = cs.restrict_months(panel, CS_START, DEV_END)
    # 마지막 개발 달의 fwd_ret는 2021년 1월 수익률이므로 사용하지 않음
    panel = panel[panel["month"] < pd.Period(DEV_END, "M")]
    has_sector = "sector" in panel.columns
    sec = "sector" if has_sector else None
    reg_path = out_dir("step2") / f"{market}_regime_dev.parquet"
    D = monthly_regime(load_series(reg_path)) if reg_path.exists() else None

    # T1 기술통계
    cols = SIGNALS + ["SIZE", "ILLIQ", "MOM", "fwd_ret"]
    t1 = panel[cols].describe(percentiles=[0.05, 0.5, 0.95]).T
    t1["avg_stocks_per_month"] = panel.groupby("month").size().mean()
    save(t1, "step4", f"{market}_T1_descriptive")

    # T2 순위상관 (월별 스피어만 → 시간 평균)
    corr_cols = SIGNALS + ["SIZE", "ILLIQ", "MOM"]
    cm = panel.groupby("month")[corr_cols].apply(lambda g: g.rank().corr()).groupby(level=1).mean()
    save(cm.loc[corr_cols, corr_cols], "step4", f"{market}_T2_rank_corr")
    if cm.loc["RSJ", "REV"] > 0.7:
        print(f"[{market}] 경고: RSJ–REV 순위상관 {cm.loc['RSJ', 'REV']:.2f} — RSJ가 반전 효과의 재포장일 위험")

    # T3 IC 요약, T4 decay
    t3 = cs.ic_summary(panel, SIGNALS, split=SPLIT, lags=LAG)
    save(t3, "step4", f"{market}_T3_ic_summary")
    t4 = pd.concat({s: cs.ic_decay(panel, s) for s in SIGNALS}, axis=0)
    save(t4, "step4", f"{market}_T4_ic_decay")

    # T5 분위 포트폴리오 (동일가중 / 규모가중)
    bench = cs.ew_benchmark(panel)
    panel["w_size"] = np.exp(panel["SIZE"])
    t5 = {}
    for s in SIGNALS:
        direction = int(np.sign(t3.loc[s, "mean_IC"])) or 1
        for wname, w in (("EW", None), ("SW", "w_size")):
            q = cs.quantile_returns(panel, s, 5, weight=w)
            t5[(s, wname)] = cs.quantile_summary(q, direction, bench, LAG)
    t5 = pd.DataFrame(t5).T
    save(t5, "step4", f"{market}_T5_quintiles")

    # T6 Fama-MacBeth 경마 분석 (RSJ, RSkew 각각)
    specs = {
        "(1)": [], "(2)": ["REV"], "(3)": ["MAX", "IVOL"], "(4)": ["RSkew"],
        "(5)": ["SIZE", "ILLIQ", "MOM"], "(6)": ["REV", "MAX", "IVOL", "RSkew", "SIZE", "ILLIQ", "MOM"],
    }
    B_store = {}
    for main in ("RSJ", "RSkew"):
        rows = {}
        for k, extra in specs.items():
            xs = [main] + [c for c in extra if c != main]
            if main == "RSkew" and k == "(4)":
                xs = ["RSkew", "RSJ"]
            summ, B = cs.fama_macbeth(panel, xs, lags=LAG, sector_col=sec if k == "(6)" else None)
            rows[k] = summ["coef"].round(5).astype(str) + " (" + summ["t_NW"].round(2).astype(str) + ")"
            if k == "(6)":
                B_store[main] = B
        save(pd.DataFrame(rows), "step4", f"{market}_T6_fama_macbeth_{main}")

    # T7 이중정렬
    t7 = {}
    for s in ("RSJ", "RSkew"):
        direction = int(np.sign(t3.loc[s, "mean_IC"])) or 1
        for ctrl in ("REV", "MAX", "IVOL"):
            ds = cs.dependent_double_sort(panel, ctrl, s, direction)
            m, t = cs.nw_mean(ds, LAG)
            t7[f"{s} | {ctrl}"] = {"LS_mean": m, "t_NW": t, "n_months": len(ds)}
    save(pd.DataFrame(t7).T, "step4", f"{market}_T7_double_sort")

    # T8 중립화 IC
    t8, neu_ics = {}, {}
    for s in SIGNALS:
        ctrls = [c for c in ["REV", "SIZE"] if c != s]
        p2 = cs.neutralize(panel, s, ctrls, sector_col=sec)
        ic_n = cs.rank_ic(p2, f"{s}_neu")
        neu_ics[s] = ic_n
        m, t = cs.nw_mean(ic_n, LAG)
        t8[s] = {"raw_mean_IC": t3.loc[s, "mean_IC"], "neu_mean_IC": m, "neu_t_NW": t,
                 "shrink_pct": 100 * (1 - m / t3.loc[s, "mean_IC"]) if t3.loc[s, "mean_IC"] else np.nan}
    save(pd.DataFrame(t8).T, "step4", f"{market}_T8_neutralized_ic")

    # T9 레짐 조건부 (H3)
    if D is not None:
        rows = {}
        for s in SIGNALS:
            rows[f"IC_{s}"] = cs.regime_test(cs.rank_ic(panel, s), D, LAG)
        for main, B in B_store.items():
            rows[f"FM_{main}(6)"] = cs.regime_test(B[main], D, LAG)
        rows["sigma_cs"] = cs.regime_test(cs.sigma_cs(panel), D, LAG)
        save(pd.DataFrame(rows).T, "step4", f"{market}_T9_regime_conditional")
        save(cs.ic_by_episode(cs.rank_ic(panel, "RSJ"), D), "step4", f"{market}_T9b_rsj_ic_by_episode")
    else:
        print(f"[{market}] 레짐 파일 없음 → 2단계 먼저 실행 (T9 생략)")

    # S2 시그널 선택 (사전 기준)
    sel, report = {}, {}
    for s in SIGNALS:
        direction = int(np.sign(t3.loc[s, "mean_IC"])) or 1
        ok, reasons = cs.s2_signal_passes(
            t8[s]["neu_t_NW"], t3.loc[s, "IC_first_half"], t3.loc[s, "IC_second_half"],
            t5.loc[(s, "EW"), "monotonic_rho"], t5.loc[(s, "EW"), "long_minus_bench"], S2_CRITERIA)
        report[s] = {"pass": ok, "direction": direction, "reasons": reasons}
        if ok:
            sel[s] = direction
    save({"selected_signals": sel, "report": report, "criteria": S2_CRITERIA}, "step4", f"{market}_s2_selection")
    if not sel:
        print(f"[{market}] 기준을 통과한 시그널 없음 → 결정 관문: S2 축소/생략, 7단계(장중 주간 RSJ) 비중 확대")


def main():
    for m in MARKETS:
        analyze(m)


if __name__ == "__main__":
    main()
