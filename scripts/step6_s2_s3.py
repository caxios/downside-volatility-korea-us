"""
6단계: 종목 선택 전략 S2, 결합 전략 S3

  python scripts/step6_s2_s3.py --phase design     # 개발 구간에서 파라미터 선택 (4개 조합)
  python scripts/step6_s2_s3.py --freeze           # S2S3 규칙 동결 (design 결과 기반)
  python scripts/step6_s2_s3.py --phase test       # 동결 후 테스트 구간 1회 (하위 구간 포함)

S3는 2단계에서 동결한 S1 규칙을 그대로 쓴다 (S3를 위해 S1을 다시 튜닝하지 않는다).
"""
import argparse
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.backtest import (backtest_weights, factor_alpha, perf_stats, perf_table, portfolio_monthly,
                          sleeve_daily)
from src.data_io import load_french_monthly
from src.config import (BAN_END_MONTH, BAN_START_MONTH, CS_START, DEV_END, MARKETS, PROC_DIR,
                        S2_BUFFER_GRID, S2_ENTRY_GRID, S2_MAX_WEIGHT, TEST_START, VT_BUFFER)
from src.firewall import freeze, guard, load_frozen
from src.pipeline import load_market, load_panel, s1_all
from src.report import out_dir, save
from src.strategies import run_s2, run_s3


def fwd_matrix(panel):
    return panel.pivot_table(index="month", columns="ticker", values="fwd_ret")


def monthly_etf(mk):
    r = mk["etf_ret"].dropna()
    return (1 + r).groupby(r.index.to_period("M")).prod() - 1


def evaluate(market, panel, daily_ret, signals, entry, buf, s1_rules, mk, start, end):
    """S2(월별)와 S3(일별) 성과. 결정 월 m의 수익은 m+1월에 실현 → 평가 구간은 실현 월 기준."""
    cfg = MARKETS[market]
    holdings = run_s2(panel, signals, entry, buf, S2_MAX_WEIGHT)
    port = portfolio_monthly(holdings, fwd_matrix(panel), cfg["stock_cost_oneway"], cfg["sell_tax"])
    port.index = pd.PeriodIndex(port.index) + 1                       # 실현 월로 이동
    sm, em = pd.Period(start[:7], "M"), (pd.Period(end[:7], "M") if end else None)
    port = port[(port.index >= sm) & ((port.index <= em) if em is not None else True)]
    ew = panel.groupby("month")["fwd_ret"].mean()
    ew.index = pd.PeriodIndex(ew.index) + 1
    etf_m = monthly_etf(mk)
    series_m = {"S2_net": port["net"], "S2_gross": port["gross"],
                "EW_universe": ew.reindex(port.index), "ETF": etf_m.reindex(port.index)}
    tab_m = perf_table(series_m, periods=12)
    tab_m.loc["S2_net", "avg_monthly_turnover"] = port["turnover"].mean()
    tab_m.loc["S2_net", "missing_ret_per_month"] = port["n_missing_ret"].mean()

    # S3: 일별 슬리브 × S1 비중
    sl = sleeve_daily(holdings, daily_ret).loc[start:end]
    res = s1_all(mk, s1_rules["lambda"], sigma_target=s1_rules["sigma_target"],
                 end=end, turbulent_weight_c=s1_rules["turbulent_weight_c"], model=s1_rules["regime_model"])
    w1 = res["weights"][s1_rules["rule"]]
    s3 = run_s3(sl, w1, cfg["stock_cost_oneway"], cfg["sell_tax"], cfg["etf_cost_oneway"])
    etf_s1 = backtest_weights(w1, mk["etf_ret"], s1_rules["cost_oneway"], buffer=s1_rules["buffer"]).loc[start:end]
    tab_d = perf_table({"S3_net": s3["net"], "S2_daily_sleeve": sl["gross"],
                        "ETF+S1": etf_s1["net"], "ETF_BH": mk["etf_ret"].loc[start:end]})
    return tab_m, tab_d, port


def load_inputs(market, allow_test):
    panel = load_panel(PROC_DIR / f"panel_{market}.parquet")
    daily_ret = pd.read_parquet(PROC_DIR / f"daily_ret_{market}.parquet")
    mk = load_market(market)
    if allow_test:
        req = ("S1", "S2S3")
        panel = guard(panel, allow_test=True, required=req, context=f"step6 test {market}", month_col="month")
        daily_ret = guard(daily_ret, allow_test=True, required=req, context=f"step6 test {market}")
    else:
        panel = guard(panel, month_col="month")
        panel = panel[panel["month"] < pd.Period(DEV_END, "M")]
        daily_ret = guard(daily_ret)
        mk = {k: (guard(v) if isinstance(v, (pd.Series, pd.DataFrame)) else v) for k, v in mk.items()}
    return panel, daily_ret, mk


def alpha_report(market, port, mk):
    """S2 월 순수익의 팩터 알파. 미국: French 팩터, 한국: ETF 대비 CAPM (무위험 0 가정)"""
    try:
        if market == "US":
            F = load_french_monthly()
            y = port["net"] - F["RF"].reindex(port.index)
            cols = [c for c in ["Mkt-RF", "SMB", "HML", "Mom", "ST_Rev"] if c in F.columns]
            save(factor_alpha(y, F[cols]), "step6", f"{market}_FINAL_TEST_S2_factor_alpha")
        else:
            m = monthly_etf(mk).rename("MKT")
            save(factor_alpha(port["net"], m.to_frame()), "step6", f"{market}_FINAL_TEST_S2_capm_alpha")
    except Exception as e:
        print(f"[{market}] 팩터 알파 계산 실패: {e}")


def phase_design():
    s1 = load_frozen("S1")["rules"]
    chosen = {}
    for market in MARKETS:
        sel_path = out_dir("step4") / f"{market}_s2_selection.json"
        signals = json.loads(sel_path.read_text())["selected_signals"] if sel_path.exists() else {}
        if not signals:
            print(f"[{market}] 4단계에서 선택된 시그널 없음 → S2 생략 (결정 관문)")
            continue
        panel, daily_ret, mk = load_inputs(market, False)
        rows = {}
        for entry in S2_ENTRY_GRID:
            for buf in S2_BUFFER_GRID:
                tab_m, tab_d, _ = evaluate(market, panel, daily_ret, signals, entry, buf, s1[market], mk,
                                           CS_START, DEV_END)
                rows[(entry, buf)] = {"S2_net_sharpe": tab_m.loc["S2_net", "sharpe"],
                                      "S2_ann_ret": tab_m.loc["S2_net", "ann_ret"],
                                      "EW_sharpe": tab_m.loc["EW_universe", "sharpe"],
                                      "turnover": tab_m.loc["S2_net", "avg_monthly_turnover"],
                                      "S3_sharpe": tab_d.loc["S3_net", "sharpe"],
                                      "ETF+S1_sharpe": tab_d.loc["ETF+S1", "sharpe"]}
                save(tab_m, "step6", f"{market}_design_S2_{entry}_{buf}", show=False)
                save(tab_d, "step6", f"{market}_design_S3_{entry}_{buf}", show=False)
        grid = pd.DataFrame(rows).T
        grid.index.names = ["entry_pct", "buffer_pct"]
        save(grid, "step6", f"{market}_design_grid")
        best = grid["S2_net_sharpe"].idxmax()
        chosen[market] = {"signals": signals, "entry_pct": float(best[0]), "buffer_pct": float(best[1]),
                          "max_weight": S2_MAX_WEIGHT, "cost_oneway": MARKETS[market]["stock_cost_oneway"],
                          "sell_tax": MARKETS[market]["sell_tax"], "s1_rule_used": s1[market]["rule"],
                          "dev_S2_sharpe": float(grid.loc[best, "S2_net_sharpe"]),
                          "dev_S3_sharpe": float(grid.loc[best, "S3_sharpe"])}
        if grid.loc[best, "S3_sharpe"] <= grid.loc[best, "ETF+S1_sharpe"]:
            print(f"[{market}] 경고: 개발 구간에서 S3가 'ETF+S1'을 못 이김 → 종목 선택의 부가가치 의문")
    save(chosen, "step6", "S2S3_candidate_rules")


def phase_freeze():
    path = out_dir("step6") / "S2S3_candidate_rules.json"
    rules = json.loads(path.read_text())
    if not rules:
        print("동결할 S2S3 규칙 없음 (선택된 시그널 없음). 빈 규칙으로 동결해 관문만 통과시킴.")
    payload = freeze("S2S3", rules)
    print(f"S2S3 동결: {payload['frozen_at']} sha256={payload['sha256'][:12]}")


def phase_test():
    rules = load_frozen("S2S3")["rules"]
    s1 = load_frozen("S1")["rules"]
    periods = {"full_test": (TEST_START, None),
               "pre_ban": (TEST_START, "2023-10-31"),
               "ban": (f"{BAN_START_MONTH}-01", pd.Period(BAN_END_MONTH, "M").end_time.strftime("%Y-%m-%d")),
               "post_ban": (pd.Period(BAN_END_MONTH, "M").__add__(1).start_time.strftime("%Y-%m-%d"), None)}
    for market, rl in rules.items():
        panel, daily_ret, mk = load_inputs(market, True)
        for name, (a, b) in periods.items():
            tab_m, tab_d, port = evaluate(market, panel, daily_ret, rl["signals"], rl["entry_pct"],
                                          rl["buffer_pct"], s1[market], mk, a, b)
            save(tab_m, "step6", f"{market}_FINAL_TEST_S2_{name}")
            save(tab_d, "step6", f"{market}_FINAL_TEST_S3_{name}")
            if name == "full_test":
                alpha_report(market, port, mk)
        print(f"[{market}] 개발 S2 샤프 {rl['dev_S2_sharpe']:.2f} / S3 {rl['dev_S3_sharpe']:.2f} "
              "— 테스트 결과가 나빠도 규칙을 바꾸지 말 것 (바꾸려면 새 버전 동결 후 페이퍼 트레이딩)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["design", "test"], default=None)
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    if args.freeze:
        phase_freeze()
    elif args.phase == "design":
        phase_design()
    elif args.phase == "test":
        phase_test()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
