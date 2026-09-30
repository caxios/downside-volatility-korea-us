"""
10단계: S5 — 주간 실현 왜도 롱숏 (Amaya et al. 2015 방식) 검증 (src/s5.py)

  python scripts/step10_realized_skew.py --freeze    # 설계 동결 (결과를 보기 전에 1회)
  python scripts/step10_realized_skew.py             # 개발·테스트 구간 평가 + 장중(60분봉·5분봉) 버전

설계(DESIGN)는 논문을 따라 미리 고정하고, 결과를 보고 고르는 파라미터는 없다.
결과: results/step10/*.csv
"""
import argparse

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import CS_START, DEV_END, MARKETS, PROC_DIR, TEST_START
from src.data_io import exchange_tz, load_french_monthly, load_intraday
from src.firewall import freeze, guard, is_frozen, load_frozen, log_access
from src.pipeline import load_market, load_panel
from src.report import save
from src.s5 import (factor_regression, fama_macbeth, quantile_table, sort_portfolios, summarize, to_monthly,
                    weekly_from_daily, weekly_long)
from src.vol import intraday_weekly_measures

DESIGN = {
    "source": "Amaya, Christoffersen, Jacobs & Vasquez (2015, JFE) - weekly realized skewness",
    "signal_main": "이번 주(월~일) 일별 로그수익률의 실현 왜도, 주당 수익률 4개 이상",
    "signal_robust": "주 마지막 거래일 기준 최근 22거래일 실현 왜도",
    "rebalance": "매주, 그 주 마지막 거래일 종가에 결정 → 다음 주 보유",
    "portfolio": "10분위, 롱숏 = 최저 왜도(1분위) − 최고 왜도(10분위), 동일가중·가치가중(미국 거래대금, 한국 시가총액)",
    "long_only": "최저 왜도 10분위 매수 (비교: 유니버스 동일가중)",
    "universe": "결정 주 직전 월말 패널 구성원 (미국 S&P 500 과거 구성, 한국 시가총액 상위 200)",
    "costs": {m: {"oneway": MARKETS[m]["stock_cost_oneway"], "sell_tax": MARKETS[m]["sell_tax"]} for m in MARKETS},
    "dev": "보유 주 2010-01 ~ 2020-12", "test": "보유 주 2021-01 ~",
    "tests": "분위 수익률, 롱숏 NW t(시차 4), 비용 차감, Carhart 4요인(+단기반전) 알파(미국), CAPM(한국), 주간 Fama-MacBeth(REV·RVOL·SIZE 통제)",
    "intraday": "60분봉(약 3년)·5분봉(약 12주) 주간 실현 왜도, 50종목 5분위 동일가중 (예비)",
}
SIZE = {"US": "DVOL", "KR": "mcap"}


def monthly_etf(mk):
    r = mk["etf_ret"].dropna()
    return (1 + r).groupby(r.index.to_period("M")).prod() - 1


def window_filter(ls, sample):
    hold_start = pd.Series([(w + 1).start_time for w in ls.index], index=ls.index)
    hold_end = pd.Series([(w + 1).end_time for w in ls.index], index=ls.index)
    if sample == "dev":
        return ls[(hold_start >= pd.Timestamp(CS_START)) & (hold_end <= pd.Timestamp(DEV_END) + pd.Timedelta(days=1))]
    return ls[hold_start >= pd.Timestamp(TEST_START)]


def alphas(market, ls, mk, tag):
    out = {}
    if market == "US":
        F = load_french_monthly()
        for col in ("LS_EW_net", "LS_VW_net", "LS_EW_gross", "LS_VW_gross"):
            y = to_monthly(ls[col])
            out[(col, "Carhart4")] = factor_regression(y, F[["Mkt-RF", "SMB", "HML", "Mom"]])
            out[(col, "Carhart4+ST_Rev")] = factor_regression(y, F[["Mkt-RF", "SMB", "HML", "Mom", "ST_Rev"]])
    else:
        m = monthly_etf(mk).rename("MKT").to_frame()
        for col in ("LS_EW_net", "LS_VW_net", "LS_EW_gross", "LS_VW_gross"):
            out[(col, "CAPM")] = factor_regression(to_monthly(ls[col]), m)
    res = pd.concat(out, names=["portfolio", "model", "term"])
    save(res, "step10", f"{market}_{tag}_alpha", show=False)


def run_daily(market, allow_test):
    panel = load_panel(PROC_DIR / f"panel_{market}.parquet")
    daily = pd.read_parquet(PROC_DIR / f"daily_ret_{market}.parquet")
    mk = load_market(market)
    if allow_test:
        req = ("S1", "S2S3", "S4", "S5")
        panel = guard(panel, allow_test=True, required=req, context=f"step10 test {market}", month_col="month")
        daily = guard(daily, allow_test=True, required=req, context=f"step10 test {market}")
    else:
        panel, daily = guard(panel, month_col="month"), guard(daily)
        mk = {k: (guard(v) if isinstance(v, (pd.Series, pd.DataFrame)) else v) for k, v in mk.items()}
    cfg = MARKETS[market]
    tag0 = "test" if allow_test else "dev"
    for sig, window in (("week", None), ("22d", 22)):
        d = weekly_long(weekly_from_daily(daily, window), panel, SIZE[market])
        d["SIZE_log"] = np.log(d["SIZE_W"].where(d["SIZE_W"] > 0))
        q, ls = sort_portfolios(d, 10, cfg["stock_cost_oneway"], cfg["sell_tax"])
        q, ls = window_filter(q, tag0), window_filter(ls, tag0)
        tag = f"{tag0}_{sig}"
        save(summarize(ls), "step10", f"{market}_{tag}_summary", show=False)
        save(quantile_table(q, 10), "step10", f"{market}_{tag}_deciles", show=False)
        ls.to_csv(f"{_out()}/{market}_{tag}_weekly.csv")
        dd = d[d["week"].isin(ls.index)]
        save(fama_macbeth(dd, ["RSK", "REV", "RVOL", "SIZE_log"]), "step10", f"{market}_{tag}_fm", show=False)
        save(fama_macbeth(dd, ["RSK"]), "step10", f"{market}_{tag}_fm_univariate", show=False)
        alphas(market, ls, mk, tag)


def run_intraday(market, daily_all):
    """60분봉·5분봉 주간 실현 왜도 (50종목, 5분위, 동일가중). 같은 종목·같은 주의 일봉 왜도와 비교"""
    cfg = MARKETS[market]
    Wd = weekly_from_daily(daily_all)
    rows = {}
    for iv, min_n in (("60m", 20), ("5m", 200)):
        b = load_intraday(iv)
        recs = []
        for t, g in b.groupby("ticker"):
            if t.startswith("^") or (exchange_tz(t) == "Asia/Seoul") != (market == "KR") or t not in daily_all.columns:
                continue
            w = intraday_weekly_measures(g, exchange_tz(t))
            w = w[w["N"] >= min_n]
            recs.append(pd.DataFrame({"week": w.index, "ticker": t, "RSK": w["RSkew"].to_numpy(),
                                      "RVOL": np.sqrt(w["RV"].to_numpy())}))
        d = pd.concat(recs, ignore_index=True)
        d["week"] = pd.PeriodIndex(d["week"], freq="W")
        for k in ("REV", "NEXT", "RSK"):
            s = Wd[k].stack(future_stack=True).rename(k if k != "RSK" else "RSK_daily")
            s.index.names = ["week", "ticker"]
            d = d.merge(s.reset_index(), on=["week", "ticker"], how="left")
        d = d.dropna(subset=["RSK", "NEXT"])
        d["SIZE_W"] = 1.0
        for sig in ("RSK", "RSK_daily"):
            dd = d.rename(columns={"RSK": "RSK_intraday"}).rename(columns={sig if sig != "RSK" else "RSK_intraday": "RSK"})
            dd = dd.dropna(subset=["RSK"])
            q, ls = sort_portfolios(dd, 5, cfg["stock_cost_oneway"], cfg["sell_tax"], min_n=30)
            name = f"{iv}_{'intraday' if sig == 'RSK' else 'daily'}"
            s = summarize(ls[["LS_EW_gross", "LS_EW_net", "LONG_EW_net", "UNIV_EW", "turn_EW"]])
            s["weeks_range"] = f"{ls.index.min()} ~ {ls.index.max()}"
            rows[name] = s
            save(quantile_table(q, 5).loc["EW"], "step10", f"{market}_intraday_{name}_quintiles", show=False)
            if iv == "60m":
                save(fama_macbeth(dd, ["RSK", "REV", "RVOL"], min_n=30), "step10", f"{market}_intraday_{name}_fm",
                     show=False)
    save(pd.concat(rows), "step10", f"{market}_intraday_summary", show=False)


def _out():
    from src.report import out_dir
    return out_dir("step10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        p = freeze("S5", DESIGN)
        print(f"S5 설계 동결: {p['frozen_at']} sha256={p['sha256'][:12]}")
        return
    if not is_frozen("S5"):
        raise SystemExit("먼저 --freeze 로 설계를 동결할 것")
    assert load_frozen("S5")["rules"] == DESIGN, "동결된 설계와 코드의 DESIGN 이 다름"
    for market in MARKETS:
        run_daily(market, False)
        run_daily(market, True)
        log_access(f"step10 intraday {market}", ("S1", "S2S3", "S5"))
        run_intraday(market, pd.read_parquet(PROC_DIR / f"daily_ret_{market}.parquet"))


if __name__ == "__main__":
    main()
