"""
7단계: 장중 데이터로 결과 확인 (해상도 검증)

  python scripts/step7_intraday_validation.py

- C층(5분봉): 일간 RV·RS±·RQ → Parkinson과의 비교, HAR/LHAR/SHAR/HARQ 비교
- B층(60분봉): 주간 RSJ → 일봉 기반 지표와의 상관, 주간 횡단면 IC

주의: 장중 데이터는 전부 2021년 이후(테스트 구간)라서, S1·S2S3 동결 후에만 실행한다.
      데이터가 짧으므로 결과는 '예비 결과'로만 보고한다.
"""
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import MARKETS
from src.cross_section import nw_mean
from src.config import DATA_START
from src.data_io import daily_path, download_daily, exchange_tz, load_daily, load_intraday
from src.firewall import FirewallError, is_frozen, log_access
from src.har import compare_models, har_features, spec, walk_forward
from src.quality import clean_daily
from src.report import save
from src.vol import intraday_daily_measures, intraday_weekly_measures, parkinson


def daily_5m():
    bars = load_intraday("5m")
    if bars.empty:
        print("5분봉 없음 → step0 수집을 먼저 여러 주 돌릴 것")
        return
    rows_corr, done = {}, 0
    for t, g in bars.groupby("ticker"):
        dm = intraday_daily_measures(g, exchange_tz(t))
        dm = dm[dm["N"] >= 0.8 * dm["N"].median()]                  # 반일장·결측 많은 날 제외
        if len(dm) < 30:
            continue
        if daily_path(t).exists():
            dd = clean_daily(load_daily(t), kr=t.endswith(".KS"))
            pk = parkinson(dd).reindex(dm.index)
            both = pd.concat([dm["RV"], pk], axis=1).dropna()
            rows_corr[t] = {"days": len(both), "corr_RV5m_Parkinson": both.corr().iloc[0, 1],
                            "mean_ratio_Parkinson_over_RV5m": (both.iloc[:, 1] / both.iloc[:, 0]).mean(),
                            "mean_RSJ_daily": dm["RSJ"].mean()}
            ret_pct = dd["r"].reindex(dm.index) * 100
        else:
            ret_pct = None
        # HAR 계열 비교 (지수·ETF 위주, 표본이 짧으므로 최소 학습 60일)
        if len(dm) >= 120 and ret_pct is not None:
            d = har_features(dm["RV"], ret_pct, rs_pos=dm["RS_pos"], rs_neg=dm["RS_neg"], rq=dm["RQ"])
            mt = max(60, len(d) // 2)
            preds = {m: walk_forward(d, spec(m, True), mt, 5, log_target=True) for m in ("HAR", "LHAR", "SHAR")}
            preds["HARQ_level"] = walk_forward(d, spec("HARQ", False), mt, 5)
            preds["HAR_level"] = walk_forward(d, spec("HAR", False), mt, 5)
            save(compare_models(d, preds), "step7", f"har_family_5m_{t.replace('^', 'IDX_')}", show=False)
            done += 1
    save(pd.DataFrame(rows_corr).T, "step7", "rv5m_vs_parkinson")
    print(f"HAR 계열 비교를 저장한 종목 수: {done} (표본이 짧아 예비 결과)")


def weekly_60m():
    bars = load_intraday("60m")
    if bars.empty:
        print("60분봉 없음")
        return
    wk, fwd = {}, {}
    for t, g in bars.groupby("ticker"):
        w = intraday_weekly_measures(g, exchange_tz(t))
        w = w[w["N"] >= 20]
        if len(w) < 20 or not daily_path(t).exists():
            continue
        dd = clean_daily(load_daily(t), kr=t.endswith(".KS"))
        r = dd["r"].dropna()
        wret = r.groupby(r.index.to_period("W")).sum()
        wk[t] = w["RSJ"]
        fwd[t] = wret.shift(-1).reindex(w.index)                         # 다음 주 수익률
    rows = {}
    for mkt in ("US", "KR"):                                             # 시장별로 따로 횡단면
        names = [t for t in wk if (exchange_tz(t) == "Asia/Seoul") == (mkt == "KR") and not t.startswith("^")]
        if len(names) < 30:
            print(f"[{mkt}] 주간 횡단면 IC에 필요한 종목 수 부족 ({len(names)}개) — 수집 대상을 늘릴 것")
        S = pd.DataFrame({t: wk[t] for t in names})
        F = pd.DataFrame({t: fwd[t] for t in names})
        if S.empty:
            continue
        ic = pd.Series({p: S.loc[p].rank().corr(F.loc[p].rank()) for p in S.index
                        if S.loc[p].notna().sum() >= 10 and F.loc[p].notna().sum() >= 10}, dtype=float).dropna()
        m, t = nw_mean(ic, 4)
        rows[mkt] = {"weeks": len(ic), "mean_IC_weekly_RSJ": m, "t_NW": t, "n_tickers": len(names)}
    save(pd.DataFrame(rows).T, "step7", "weekly_rsj_60m_ic")


def main():
    missing = [r for r in ("S1", "S2S3") if not is_frozen(r)]
    if missing:
        raise FirewallError(f"장중 데이터는 테스트 구간(2021~)이므로 먼저 동결할 것: {missing}")
    log_access("step7 intraday", ("S1", "S2S3"))
    # 장중 수집 종목 중 일봉 캐시가 없는 종목(섹터 ETF 등)은 일봉을 받아둔다
    tick = set()
    for iv in ("5m", "60m"):
        b = load_intraday(iv)
        if not b.empty:
            tick |= set(b["ticker"].unique())
    need = sorted(t for t in tick if not daily_path(t).exists())
    if need:
        download_daily(need, start=DATA_START)
    daily_5m()
    weekly_60m()


if __name__ == "__main__":
    main()
