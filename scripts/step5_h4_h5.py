"""
5단계: 공매도 금지(H4) + 한미 비교(H5)

  python scripts/step5_h4_h5.py --phase pre    # 관문 전: 개발 구간 H5, H4 플라시보 분포, 코드 리허설
  python scripts/step5_h4_h5.py --phase post   # 관문 후(S1, S2S3 동결 필요): H4 본 분석

주의: 공매도 금지 기간(2023.11~2025.3)은 6단계 최종 테스트 구간 안에 있다.
      S2·S3 규칙을 동결하기 전에 post를 실행하면 방화벽이 막는다.
"""
import argparse

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src import cross_section as cs
from src.config import (BAN_END_MONTH, BAN_START_MONTH, CS_START, DEV_END, KR_PRIOR_BANS, NW_LAGS_MONTHLY,
                        PROC_DIR, REGIME_MODEL)
from src.firewall import guard, load_frozen
from src.pipeline import load_market, load_panel, load_series, market_regime
from src.regime import monthly_regime
from src.report import out_dir, save
from src.shortban import (country_did, high_low_short, ic_difference, months_in, placebo_ban, placebo_pvalue,
                          within_korea_did)

H4_SIGNALS = ["IVOL", "MAX", "RSJ", "RSkew"]      # 사전 확정 — 결과를 보고 바꾸지 말 것
FM_CONTROLS = ["REV", "SIZE"]
BAN_LEN = (pd.Period(BAN_END_MONTH, "M") - pd.Period(BAN_START_MONTH, "M")).n + 1
LAG = NW_LAGS_MONTHLY


def fm_series(panel, sig):
    xs = [sig] + [c for c in FM_CONTROLS if c != sig]
    _, B = cs.fama_macbeth(panel, xs, lags=LAG)
    return B[sig]


def load_panels(allow_test):
    out = {}
    for m in ("US", "KR"):
        p = load_panel(PROC_DIR / f"panel_{m}.parquet")
        if allow_test:
            p = guard(p, allow_test=True, required=("S1", "S2S3"), context=f"step5 post {m}", month_col="month")
        else:
            p = guard(p, month_col="month")
            p = p[p["month"] < pd.Period(DEV_END, "M")]
        out[m] = cs.restrict_months(p, CS_START, None)
    return out


def phase_pre():
    P = load_panels(False)
    D = {m: monthly_regime(load_series(out_dir("step2") / f"{m}_regime_dev.parquet"))
         for m in ("US", "KR") if (out_dir("step2") / f"{m}_regime_dev.parquet").exists()}

    # H5: 같은 달 IC 차이
    rows, rows_reg = {}, {}
    for s in H4_SIGNALS + ["REV"]:
        ic_kr, ic_us = cs.rank_ic(P["KR"], s), cs.rank_ic(P["US"], s)
        rows[s] = ic_difference(ic_kr, ic_us, lags=LAG)
        if "KR" in D:
            rows_reg[s] = ic_difference(ic_kr, ic_us, D["KR"], lags=LAG)
    save(pd.DataFrame(rows).T, "step5", "H5_ic_difference_dev")
    if rows_reg:
        save(pd.DataFrame(rows_reg).T, "step5", "H5_ic_difference_by_KR_regime_dev")

    # H4 플라시보: 개발 구간 안의 같은 길이 가짜 금지 기간
    plac = {}
    for s in H4_SIGNALS:
        delta = fm_series(P["KR"], s) - fm_series(P["US"], s)
        plac[s] = placebo_ban(delta, BAN_LEN, DEV_END, lags=LAG)
    plac = pd.DataFrame(plac)
    save(plac.describe().T, "step5", "H4_placebo_summary")
    plac.to_csv(out_dir("step5") / "H4_placebo_distribution.csv")

    # 코드 리허설: 가짜 High/Low 분류(2012년 평균 SIZE)와 가짜 금지 기간으로 한국 내 DiD
    kr = P["KR"]
    size12 = kr[kr["month"].astype(str).str.startswith("2012")].groupby("ticker")["SIZE"].mean()
    hi, lo = high_low_short(size12)
    rh = within_korea_did(kr, "IVOL", hi, lo, "2013-01", "2014-05", start_month="2010-01")
    save(rh, "step5", "H4_within_KR_REHEARSAL_fake_window")
    print("\n[리허설] 위 결과는 가짜 기간·가짜 분류로 코드만 점검한 것 — 해석하지 말 것")


def phase_post():
    load_frozen("S2S3")
    s1 = load_frozen("S1")["rules"]
    P = load_panels(True)
    D = {}
    for m in ("US", "KR"):
        mk = load_market(m)
        states = market_regime(mk["r_pct"], mk["rv"], s1[m]["lambda"], s1[m]["regime_model"])
        D[m] = monthly_regime(states)

    plac = pd.read_csv(out_dir("step5") / "H4_placebo_distribution.csv", index_col=0)
    rows, rob = {}, {}
    for s in H4_SIGNALS:
        b_kr, b_us = fm_series(P["KR"], s), fm_series(P["US"], s)
        res = country_did(b_kr, b_us, BAN_START_MONTH, BAN_END_MONTH, D["KR"], D["US"], LAG)
        rows[s] = {"theta3_Ban": res.loc["Ban", "coef"], "t_NW": res.loc["Ban", "t_NW"],
                   "Post": res.loc["Post", "coef"] if "Post" in res.index else np.nan,
                   "placebo_p": placebo_pvalue(res.loc["Ban", "coef"], plac[s])}
        # 교란 요인 강건성: 한국 고유 사건 월 제외
        r2 = country_did(b_kr, b_us, BAN_START_MONTH, BAN_END_MONTH, D["KR"], D["US"], LAG,
                         exclude_months=["2024-02", "2024-12"])
        rob[s] = {"theta3_excl_events": r2.loc["Ban", "coef"], "t_NW": r2.loc["Ban", "t_NW"]}
    save(pd.DataFrame(rows).T, "step5", "H4_country_did_REAL")
    save(pd.DataFrame(rob).T, "step5", "H4_country_did_excl_events")

    sr = PROC_DIR / "kr_short_ratio_preban.csv"
    if sr.exists():
        ratio = pd.read_csv(sr, index_col=0).iloc[:, 0]
        hi, lo = high_low_short(ratio)
        wk = {s: within_korea_did(P["KR"], s, hi, lo, BAN_START_MONTH, BAN_END_MONTH, start_month="2021-01")
              for s in H4_SIGNALS}
        save(pd.concat(wk), "step5", "H4_within_KR_REAL")
    else:
        print("공매도 잔고 파일 없음 → step3 --short 먼저 실행 (한국 내 DiD 생략)")


def phase_placebo_clean():
    """
    (2026-09-29 추가) 플라시보 오염 점검.
    개발 구간(2010~2020)에는 실제 한국 공매도 금지 기간(2011-08~11, 2020-03~)이 들어 있다.
    원래 플라시보는 이 달들을 "금지 없음"으로 취급했고, 일부 가짜 기간은 실제 금지와 겹쳤다.
    → 실제 금지 달을 표본에서 빼고 플라시보 분포를 다시 만든다.
    → 본 추정(θ3)도 과거 금지 달(2011, 2020-03~2021-04)을 빼고 다시 추정한다.
    """
    load_frozen("S2S3")
    s1 = load_frozen("S1")["rules"]
    P = load_panels(True)
    D = {}
    for m in ("US", "KR"):
        mk = load_market(m)
        D[m] = monthly_regime(market_regime(mk["r_pct"], mk["rv"], s1[m]["lambda"], s1[m]["regime_model"]))
    old = pd.read_csv(out_dir("step5") / "H4_placebo_distribution.csv", index_col=0)
    real = pd.read_csv(out_dir("step5") / "H4_country_did_REAL.csv", index_col=0)
    excl = [str(x) for x in sorted(months_in(KR_PRIOR_BANS))]
    rows, dist = {}, {}
    for s in H4_SIGNALS:
        b_kr, b_us = fm_series(P["KR"], s), fm_series(P["US"], s)
        delta = b_kr - b_us
        clean = placebo_ban(delta, BAN_LEN, DEV_END, lags=LAG, exclude_windows=KR_PRIOR_BANS)
        dist[s] = clean
        r2 = country_did(b_kr, b_us, BAN_START_MONTH, BAN_END_MONTH, D["KR"], D["US"], LAG, exclude_months=excl)
        th_old, th_new = real.loc[s, "theta3_Ban"], r2.loc["Ban", "coef"]
        rows[s] = {"theta3_original": th_old, "t_original": real.loc[s, "t_NW"],
                   "placebo_p_original_dist": real.loc[s, "placebo_p"],
                   "n_placebo_original": int(old[s].notna().sum()),
                   "placebo_p_clean_dist": placebo_pvalue(th_old, clean),
                   "n_placebo_clean": int(clean.notna().sum()),
                   "theta3_excl_prior_bans": th_new, "t_excl_prior_bans": r2.loc["Ban", "t_NW"],
                   "placebo_p_excl_prior_bans": placebo_pvalue(th_new, clean)}
    save(pd.DataFrame(rows).T, "step5", "H4_placebo_clean_summary")
    pd.DataFrame(dist).to_csv(out_dir("step5") / "H4_placebo_distribution_clean.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pre", "post", "placebo-clean"], required=True)
    args = ap.parse_args()
    {"pre": phase_pre, "post": phase_post, "placebo-clean": phase_placebo_clean}[args.phase]()


if __name__ == "__main__":
    main()
