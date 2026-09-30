"""
9단계: S4 — 예측 상승/하락 반분산 기반 종목 선택 (src/s4.py)

  python scripts/step9_s4_semivar_strategy.py --phase design   # 개발 구간(2010~2020): 4개 조합 비교 → 후보 규칙
  python scripts/step9_s4_semivar_strategy.py --freeze         # S4 동결 (design 결과, 시장별 개발 구간 순샤프 최고 조합)
  python scripts/step9_s4_semivar_strategy.py --phase test     # 동결 후 테스트 구간(2021~) 1회

사전 확정
  조합: entry_pct ∈ {0.2, 0.3} × exit ∈ {"rel", "abs"}  (S2와 같은 4개)
  선택 기준: 개발 구간 월 순수익 샤프 최고
  비교: u_only(예측 상승 변동성만), r_only(예측 비율만), 동일가중 유니버스, ETF
결과: results/step9/*.csv
"""
import argparse
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src import cross_section as cs
from src.backtest import factor_alpha, perf_table, portfolio_monthly
from src.config import CS_START, DEV_END, MARKETS, NW_LAGS_MONTHLY, PROC_DIR, S2_MAX_WEIGHT, TEST_START
from src.data_io import load_french_monthly
from src.firewall import freeze, guard, load_frozen
from src.pipeline import load_market, load_panel
from src.report import out_dir, save
from src.s4 import forecast_quality, month_end_features, rolling_forecasts, run_s4

GRID = [(0.2, "rel"), (0.2, "abs"), (0.3, "rel"), (0.3, "abs")]


def load(market, allow_test):
    panel = load_panel(PROC_DIR / f"panel_{market}.parquet")
    daily = pd.read_parquet(PROC_DIR / f"daily_ret_{market}.parquet")
    mk = load_market(market)
    if allow_test:
        req = ("S1", "S2S3", "S4")
        panel = guard(panel, allow_test=True, required=req, context=f"step9 test {market}", month_col="month")
        daily = guard(daily, allow_test=True, required=req, context=f"step9 test {market}")
    else:
        panel = guard(panel, month_col="month")
        panel = panel[panel["month"] < pd.Period(DEV_END, "M")]     # 결정 월 ≤ 2020-11 → 실현 ≤ 2020-12
        daily = guard(daily)
        mk = {k: (guard(v) if isinstance(v, (pd.Series, pd.DataFrame)) else v) for k, v in mk.items()}
    return panel, daily, mk


def monthly_etf(mk):
    r = mk["etf_ret"].dropna()
    return (1 + r).groupby(r.index.to_period("M")).prod() - 1


def realized(port_or_series, start, end):
    s = port_or_series.copy()
    s.index = pd.PeriodIndex(s.index) + 1
    sm = pd.Period(start[:7], "M")
    em = pd.Period(end[:7], "M") if end else None
    return s[(s.index >= sm) & ((s.index <= em) if em is not None else True)]


def evaluate(market, panel, fc, mk, start, end, extra=True):
    cfg = MARKETS[market]
    fwd = panel.pivot_table(index="month", columns="ticker", values="fwd_ret")
    series, ports = {}, {}
    runs = {f"S4_{int(e * 100)}_{x}": dict(entry_pct=e, exit_rule=x, mode="s4") for e, x in GRID}
    if extra:
        runs |= {"U_only_20": dict(entry_pct=0.2, mode="u_only"), "R_only_20": dict(entry_pct=0.2, mode="r_only")}
    for name, kw in runs.items():
        h = run_s4(fc, max_weight=S2_MAX_WEIGHT, **{"exit_rule": "rel", **kw})
        port = realized(portfolio_monthly(h, fwd, cfg["stock_cost_oneway"], cfg["sell_tax"]), start, end)
        ports[name], series[name] = port, port["net"]
    idx = next(iter(ports.values())).index
    ew = realized(panel.groupby("month")["fwd_ret"].mean(), start, end)
    series["EW_universe"] = ew.reindex(idx)
    series["ETF"] = monthly_etf(mk).reindex(idx)
    tab = perf_table(series, periods=12)
    for name, p in ports.items():
        tab.loc[name, "avg_monthly_turnover"] = p["turnover"].mean()
        tab.loc[name, "avg_holdings"] = p["n_holdings"].mean()
        tab.loc[name, "gross_ann_ret"] = (1 + p["gross"]).prod() ** (12 / len(p)) - 1
    return tab, ports, series


def alpha(market, net, mk, tag):
    try:
        if market == "US":
            F = load_french_monthly()
            y = net - F["RF"].reindex(net.index)
            cols = [c for c in ["Mkt-RF", "SMB", "HML", "Mom", "ST_Rev"] if c in F.columns]
            save(factor_alpha(y, F[cols]), "step9", f"{market}_{tag}_factor_alpha", show=False)
        else:
            save(factor_alpha(net, monthly_etf(mk).rename("MKT").to_frame()), "step9", f"{market}_{tag}_capm_alpha",
                 show=False)
    except Exception as e:
        print(f"[{market}] 알파 계산 실패: {e}")


def diagnostics(market, panel, fc, feat, tag, start, end):
    q = forecast_quality(fc, feat)
    q = realized(q, start, end) if len(q) else q
    rows = {}
    for c in q.columns:
        m, t = cs.nw_mean(q[c].dropna(), NW_LAGS_MONTHLY)
        rows[c] = {"mean_spearman": m, "t_NW": t, "months": int(q[c].notna().sum())}
    save(pd.DataFrame(rows).T, "step9", f"{market}_{tag}_forecast_quality", show=False)
    # 예측 U, R 의 다음 달 수익률 순위 IC (H2 방식)
    p = panel.merge(fc, on=["month", "ticker"])
    p["U"], p["R"] = p["pRSp"], p["pRSp"] / p["pRSn"]
    p = p[(p["month"] + 1 >= pd.Period(start[:7], "M")) & ((p["month"] + 1 <= pd.Period(end[:7], "M")) if end else True)]
    ic = {s: cs.nw_mean(cs.rank_ic(p, s).dropna(), NW_LAGS_MONTHLY) for s in ("U", "R")}
    save(pd.DataFrame({k: {"mean_IC": v[0], "t_NW": v[1]} for k, v in ic.items()}).T,
         "step9", f"{market}_{tag}_signal_ic", show=False)


def phase_design():
    cand = {}
    for market in MARKETS:
        panel, daily, mk = load(market, False)
        feat = month_end_features(daily)
        fc = rolling_forecasts(feat, panel, panel["month"].unique())
        tab, ports, series = evaluate(market, panel, fc, mk, CS_START, DEV_END)
        save(tab, "step9", f"{market}_design_performance")
        pd.DataFrame(series).to_csv(out_dir("step9") / f"{market}_dev_monthly.csv")
        diagnostics(market, panel, fc, feat, "dev", CS_START, DEV_END)
        grid = tab.loc[[f"S4_{int(e * 100)}_{x}" for e, x in GRID], "sharpe"]
        best = grid.idxmax()
        e, x = [(e, x) for e, x in GRID if f"S4_{int(e * 100)}_{x}" == best][0]
        cand[market] = {"entry_pct": e, "exit_rule": x, "max_weight": S2_MAX_WEIGHT,
                        "cost_oneway": MARKETS[market]["stock_cost_oneway"], "sell_tax": MARKETS[market]["sell_tax"],
                        "min_train_months": 24, "dev_sharpe": float(tab.loc[best, "sharpe"]),
                        "dev_EW_sharpe": float(tab.loc["EW_universe", "sharpe"]),
                        "dev_start_realized": str(ports[best].index.min())}
        alpha(market, ports[best]["net"], mk, "dev_best")
        print(f"[{market}] 개발 구간 선택: {best} (샤프 {grid[best]:.2f}, EW {tab.loc['EW_universe', 'sharpe']:.2f})")
    save(cand, "step9", "S4_candidate_rules")


def phase_freeze():
    rules = json.loads((out_dir("step9") / "S4_candidate_rules.json").read_text())
    payload = freeze("S4", rules)
    print(f"S4 동결: {payload['frozen_at']} sha256={payload['sha256'][:12]}")


def phase_test():
    rules = load_frozen("S4")["rules"]
    for market, rl in rules.items():
        panel, daily, mk = load(market, True)
        feat = month_end_features(daily)
        months = panel["month"].unique()
        months = months[months >= pd.Period(TEST_START[:7], "M") - 1]
        # 확장 창 학습은 전체 과거를 쓰되, 예측은 테스트 결정 월(2020-12 ~)만 만든다
        fc = rolling_forecasts(feat, panel, months)
        tab, ports, series = evaluate(market, panel, fc, mk, TEST_START, None)
        save(tab, "step9", f"{market}_FINAL_TEST_performance")
        pd.DataFrame(series).to_csv(out_dir("step9") / f"{market}_test_monthly.csv")
        diagnostics(market, panel, fc, feat, "test", TEST_START, None)
        best = f"S4_{int(rl['entry_pct'] * 100)}_{rl['exit_rule']}"
        alpha(market, ports[best]["net"], mk, "test_frozen")
        print(f"[{market}] 동결 규칙 {best}: 개발 샤프 {rl['dev_sharpe']:.2f} → 테스트 샤프 "
              f"{tab.loc[best, 'sharpe']:.2f} (EW {tab.loc['EW_universe', 'sharpe']:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["design", "test"])
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        phase_freeze()
    elif a.phase == "design":
        phase_design()
    elif a.phase == "test":
        phase_test()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
