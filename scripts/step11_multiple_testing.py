"""
11단계: 이미 계산된 결과표에 다중검정 보정을 적용한다 (새 추정 없음).

  python scripts/step11_multiple_testing.py

보고서마다 "가족(family)"을 결과를 보기 전의 가설 구조대로 정한다. 가족 = 같은 결론을 뒷받침하려고
함께 본 검정들의 묶음. 가족 안에서 Holm(FWER)과 Benjamini–Hochberg(FDR)를 적용하고,
Harvey–Liu–Zhu(2016)의 |t| > 3 기준도 함께 보고한다.

결과: results/step11/{보고서}_adjusted.csv, {보고서}_summary.csv
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import RESULTS_DIR
from src.multitest import adjust, p_from_t, summarize
from src.report import save

R = RESULTS_DIR
MK = ("US", "KR")


def _rows(fam, label, t=None, p=None, est=None):
    if p is None:
        p = float(p_from_t(t))
    return {"family": fam, "test": label, "estimate": est, "t": t, "p": p}


def _finish(name, rows, t_col="t"):
    d = pd.DataFrame(rows)
    d = adjust(d, p_col="p", t_col=t_col, family_col="family")
    save(d, "step11", f"{name}_adjusted", show=False)
    s = summarize(d, "family")
    save(s, "step11", f"{name}_summary")
    return d


# ---------------------------------------------------------------- REPORT.md (H1~H5)
def main_report():
    rows = []
    # H1: 표본외 DM 검정 + 레짐 항
    for m in MK:
        c = pd.read_csv(R / "step1" / f"{m}_har_vs_lhar_oos.csv", index_col=0)
        rows.append(_rows("H1", f"{m} LHAR vs HAR (DM)", t=c.loc["LHAR", "DM_t_vs_bench"]))
        v = pd.read_csv(R / "step2" / f"{m}_regime_lhar_oos_vs_LHAR.csv", index_col=0)
        rows.append(_rows("H1", f"{m} RLHAR vs LHAR (DM)", t=v.loc["RLHAR", "DM_t_vs_bench"]))
        k = pd.read_csv(R / "step2" / f"{m}_regime_lhar_coef.csv", index_col=0)
        rows.append(_rows("H1", f"{m} L×D 계수", t=k.loc["L_xD", "t_NW"], est=k.loc["L_xD", "coef"]))
    # H2: 원 IC, 중립화 IC, 동일가중 분위 롱숏
    for m in MK:
        ic = pd.read_csv(R / "step4" / f"{m}_T3_ic_summary.csv", index_col=0)
        for s, r in ic.iterrows():
            rows.append(_rows("H2", f"{m} {s} 원 IC", t=r["t_NW"], est=r["mean_IC"]))
        nu = pd.read_csv(R / "step4" / f"{m}_T8_neutralized_ic.csv", index_col=0)
        for s, r in nu.iterrows():
            rows.append(_rows("H2", f"{m} {s} 중립화 IC", t=r["neu_t_NW"], est=r["neu_mean_IC"]))
        q = pd.read_csv(R / "step4" / f"{m}_T5_quintiles.csv", index_col=[0, 1])
        for (s, w), r in q.iterrows():
            if w == "EW":
                rows.append(_rows("H2", f"{m} {s} 분위 롱숏(EW)", t=r["LS_t"], est=r["LS_mean"]))
    # H3
    for m in MK:
        t9 = pd.read_csv(R / "step4" / f"{m}_T9_regime_conditional.csv", index_col=0)
        for s, r in t9.iterrows():
            rows.append(_rows("H3", f"{m} {s} 혼란−평온", t=r["t_diff"], est=r["diff_turb_minus_calm"]))
    # H5
    h5 = pd.read_csv(R / "step5" / "H5_ic_difference_dev.csv", index_col=0)
    for s, r in h5.iterrows():
        rows.append(_rows("H5", f"{s} KR−US IC", t=r["t_NW"], est=r["mean_KR_minus_US"]))
    h5r = pd.read_csv(R / "step5" / "H5_ic_difference_by_KR_regime_dev.csv", index_col=0)
    for s, r in h5r.iterrows():
        rows.append(_rows("H5", f"{s} KR 혼란기 추가 차이", t=r["t_turb_extra"], est=r["turb_extra"]))
    # H4 국가 간 DiD (t 기반 p; 플라시보 p는 따로)
    did = pd.read_csv(R / "step5" / "H4_country_did_REAL.csv", index_col=0)
    for s, r in did.iterrows():
        rows.append(_rows("H4_country_t", f"{s} θ3", t=r["t_NW"], est=r["theta3_Ban"]))
        rows.append(_rows("H4_country_placebo", f"{s} θ3 플라시보", p=r["placebo_p"], est=r["theta3_Ban"]))
    # H4 플라시보 정정판 (과거 금지 달 제외, 2026-09-29)
    pc = R / "step5" / "H4_placebo_clean_summary.csv"
    if pc.exists():
        c = pd.read_csv(pc, index_col=0)
        for s, r in c.iterrows():
            rows.append(_rows("H4_country_placebo_clean", f"{s} θ3(과거 금지 제외) 플라시보",
                              p=r["placebo_p_excl_prior_bans"], est=r["theta3_excl_prior_bans"]))
    # H4 한국 내 DiD (있으면)
    wk = R / "step5" / "H4_within_KR_REAL.csv"
    if wk.exists():
        w = pd.read_csv(wk, index_col=[0, 1])
        for (s, term), r in w.iterrows():
            if term == "S_Ban_High":
                rows.append(_rows("H4_within_KR", f"{s} β4 (S·Ban·High)", t=r["t_cluster"], est=r["coef"]))
    return _finish("REPORT_main", rows)


# ---------------------------------------------------------------- 반분산 (1b)
KEY_TESTS = ("상승 총영향: Σβ = 0", "하락 총영향: Σβ = 0", "합계: Σβ⁻ − Σβ⁺")


def _eq_rows(path, fam):
    e = pd.read_csv(path)
    e = e[e["test"].isin(KEY_TESTS)]
    out = []
    for _, r in e.iterrows():
        blk = f"[{r['block']}] " if "block" in e.columns else ""
        t = r["diff"] / r["se_NW"] if pd.notna(r.get("se_NW")) and r.get("se_NW", 0) else np.nan
        out.append({"family": fam, "test": f"{blk}{r['test']} | 미래 {r['target']} h={int(r['h'])}",
                    "estimate": r["diff"], "t": t, "p": r["p"]})
    return out


def semivar_daily():
    rows = []
    for m in MK:
        for smp in ("dev", "test"):
            rows += _eq_rows(R / "step1b" / f"{m}_{smp}_equality.csv", f"{m}_{smp}")
    return _finish("REPORT_semivariance", rows)


def semivar_intraday():
    rows = []
    for f in sorted(glob.glob(str(R / "step7b" / "*_equality.csv"))):
        fam = Path(f).stem.replace("_equality", "")
        rows += _eq_rows(f, fam)
    return _finish("REPORT_intraday_semivariance", rows)


# ---------------------------------------------------------------- 방향 (7c)
def direction():
    rows = []
    di = pd.read_csv(R / "step7c" / "daily_index.csv", index_col=0)
    it = pd.read_csv(R / "step7c" / "intraday.csv", index_col=0)
    for _, r in di.iterrows():
        lab = f"daily {r['market']} {r['sample']} k={r['k']} h={r['h']}"
        rows.append(_rows("beta(반전·지속)", lab, t=r["t"], p=r["p"], est=r["beta"]))
        rows.append(_rows("적중률 vs 기준선", lab, t=r["t_vs_baseline"], p=r["p_vs_baseline"],
                          est=r["reversal_minus_baseline"]))
    for _, r in it.iterrows():
        lab = f"{r['freq']} {r['market']} {r['unit']} k={r['k']} h={r['h']}"
        rows.append(_rows("beta(반전·지속)", lab, t=r["t"], p=r["p"], est=r["beta"]))
        rows.append(_rows("적중률 vs 기준선", lab, t=r["t_vs_baseline"], p=r["p_vs_baseline"],
                          est=r["reversal_minus_baseline"]))
    rg = pd.read_csv(R / "step7c" / "daily_index_regime.csv", index_col=0)
    for _, r in rg.iterrows():
        rows.append(_rows("레짐 차이", f"daily {r['market']} {r['sample']} k={r['k']} h={r['h']}",
                          t=r["t_diff"], p=r["p_diff"], est=r["diff_turb_minus_calm"]))
    bd = pd.read_csv(R / "step7c" / "daily_index_after_big_drop.csv", index_col=0)
    for _, r in bd.iterrows():
        rows.append(_rows("큰 하락 다음", f"{r['market']} {r['sample']} h={r['h']}", t=r["t"], p=r["p"],
                          est=r["diff"]))
    return _finish("REPORT_direction", rows)


# ---------------------------------------------------------------- S4 (9)
def s4():
    rows = []
    ex = pd.read_csv(R / "step9" / "excess_return_tests.csv")
    for _, r in ex.iterrows():
        rows.append(_rows(f"초과수익_{r['sample']}", f"{r['market']} {r['strategy']} − {r['vs']}",
                          t=r["t_NW"], p=r["p"], est=r["ann_diff_pct"]))
    for m in MK:
        for smp in ("dev", "test"):
            ic = pd.read_csv(R / "step9" / f"{m}_{smp}_signal_ic.csv", index_col=0)
            for s, r in ic.iterrows():
                rows.append(_rows(f"신호IC_{smp}", f"{m} {s}", t=r["t_NW"], est=r["mean_IC"]))
    return _finish("REPORT_S4", rows)


# ---------------------------------------------------------------- S5 (10)
def s5():
    rows = []
    for m in MK:
        for smp in ("dev", "test"):
            for hz in ("week", "22d"):
                fam = f"{smp}"
                sm_ = pd.read_csv(R / "step10" / f"{m}_{smp}_{hz}_summary.csv", index_col=0)
                for leg in ("LS_EW_gross", "LS_VW_gross", "LONG_EW_net", "LONG_VW_net"):
                    if leg in sm_.index:
                        r = sm_.loc[leg]
                        rows.append(_rows(fam, f"{m} {hz} {leg}", t=r["t_NW"], p=r["p"], est=r["mean_weekly_bps"]))
                fu = pd.read_csv(R / "step10" / f"{m}_{smp}_{hz}_fm_univariate.csv", index_col=0)
                rows.append(_rows(fam, f"{m} {hz} FM 왜도(단독)", t=fu.loc["RSK", "t_NW"], p=fu.loc["RSK", "p"],
                                  est=fu.loc["RSK", "coef_bps_per_sd"]))
                fc = pd.read_csv(R / "step10" / f"{m}_{smp}_{hz}_fm.csv", index_col=0)
                rows.append(_rows(fam, f"{m} {hz} FM 왜도(통제)", t=fc.loc["RSK", "t_NW"], p=fc.loc["RSK", "p"],
                                  est=fc.loc["RSK", "coef_bps_per_sd"]))
    return _finish("REPORT_S5", rows)


def main():
    for f in (main_report, semivar_daily, semivar_intraday, direction, s4, s5):
        print(f"\n######## {f.__name__}")
        f()


if __name__ == "__main__":
    main()
