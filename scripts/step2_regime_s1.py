"""
2단계: 지수 레짐 + 레짐 LHAR + 전략 S1

  python scripts/step2_regime_s1.py                  # 개발 구간: λ 선택, 진단, 레짐 LHAR, S1 백테스트
  python scripts/step2_regime_s1.py --freeze         # S1 규칙 동결 (규칙은 개발 구간 성과로 자동 선택, --rule로 지정 가능)
  python scripts/step2_regime_s1.py --final-test     # 동결 후 테스트 구간(2021~) 1회 평가

결과: results/step2/*.csv, 레짐 시계열: results/step2/{시장}_regime_dev.parquet
"""
import argparse

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.backtest import backtest_weights, breakeven_cost, perf_by_regime, perf_stats, perf_table
from src.config import (DEV_END, DEV_START, EVENTS, FIRST_FIT_END, HAR_MIN_TRAIN, HAR_REFIT_EVERY, JM_CLIP,
                        JM_FEATURES, JM_HALFLIVES, JM_LAMBDA_GRID, MARKETS, REGIME_MODEL,
                        S1_C_TURBULENT_WEIGHT, TEST_START, VT_BUFFER)
from src.firewall import freeze, guard, load_frozen
from src.har import compare_models, full_sample_fit, har_features, spec, walk_forward
from src.pipeline import load_market, market_regime, s1_all, save_series
from src.regime import hmm_features, offline_jm, online_hmm, online_jm, regime_diagnostics
from src.report import out_dir, save
from src.strategies import buy_hold_weights, ma_rule_weights, naive_voltarget_weights
from src.data_io import load_us_rf_daily


def rf_for(market):
    if market == "US":
        try:
            return load_us_rf_daily()
        except Exception as e:
            print(f"[경고] 미국 무위험 금리 로드 실패 → 0 가정 ({e})")
    return None


def run_all_rules(mk, res, cost, rf, start, end, buffer=VT_BUFFER):
    """S1 규칙 A/B/C와 벤치마크 3개 백테스트 → (성과표, 순수익 dict, 회전율 dict)"""
    ret = mk["etf_ret"]
    sig_t = res["sigma_target"]
    weights = {
        "S1_A": (res["weights"]["A"], 0.0),
        "S1_B": (res["weights"]["B"], buffer),
        "S1_C": (res["weights"]["C"], buffer),
        "BH": (buy_hold_weights(ret.index), 0.0),
        "MA200": (ma_rule_weights(mk["close"]), 0.0),
        "naiveVT": (naive_voltarget_weights(mk["r_pct"], sig_t), buffer),
    }
    nets, tovs, gross = {}, {}, {}
    for name, (w, buf) in weights.items():
        bt = backtest_weights(w, ret, cost, rf=rf, buffer=buf).loc[start:end]
        nets[name], tovs[name], gross[name] = bt["net"], bt["turnover"], bt["gross"]
    tab = perf_table(nets, rf=rf, turnover=tovs)
    tab["breakeven_cost_oneway"] = pd.Series({k: breakeven_cost(gross[k] - (rf.reindex(gross[k].index).fillna(0)
                                                                          if rf is not None else 0), tovs[k])
                                              for k in nets})
    return tab, nets, tovs


def develop(market, mk):
    cfg = mk["cfg"]
    cost = cfg["etf_cost_oneway"]
    rf = rf_for(market)
    r_dev, rv_dev = guard(mk["r_pct"]), guard(mk["rv"])

    # ---- λ 선택 (JM일 때만): 개발 구간 규칙 A의 비용 차감 샤프
    lam_rows, states_by_lam = [], {}
    if REGIME_MODEL == "jm":
        for lam in JM_LAMBDA_GRID:
            s = online_jm(r_dev, lam, JM_HALFLIVES, FIRST_FIT_END, clip=JM_CLIP, rv=rv_dev, kind=JM_FEATURES)
            bt = backtest_weights(1.0 - s, mk["etf_ret"], cost, rf=rf).loc[DEV_START:DEV_END]
            st = perf_stats(bt["net"], rf=rf)
            lam_rows.append({"lambda": lam, "sharpe_A": st["sharpe"], "max_dd_A": st["max_dd"],
                             "switches_per_year": float((s.diff().abs() > 0).sum() / (len(s) / 252)),
                             "turbulent_share": float(s.mean())})
            states_by_lam[lam] = s
        lam_tab = pd.DataFrame(lam_rows).set_index("lambda")
        save(lam_tab, "step2", f"{market}_lambda_selection")
        lam = float(lam_tab["sharpe_A"].idxmax())
        states = states_by_lam[lam]
    else:
        lam = None
        states = market_regime(r_dev, rv_dev, None, "hmm")
    print(f"[{market}] 레짐 모형={REGIME_MODEL}, 선택된 λ={lam}")
    save_series(states, out_dir("step2") / f"{market}_regime_dev.parquet")

    # ---- 레짐 진단
    off = offline_jm(r_dev, lam or 30.0, JM_HALFLIVES, clip=JM_CLIP, rv=rv_dev, kind=JM_FEATURES)
    diag = regime_diagnostics(states, r_dev, offline=off, events=EVENTS)
    save(diag["stats"], "step2", f"{market}_regime_stats")
    save(diag["persistence"], "step2", f"{market}_regime_persistence")
    save(diag["event_turbulent_share"], "step2", f"{market}_regime_events")
    save(diag["detection_lags"], "step2", f"{market}_detection_lags", show=False)
    summary = {k: diag[k] for k in ("switches_per_year", "online_offline_agreement", "detection_lag_median")}

    # ---- 강건성: 다른 레짐 모형과의 일치
    try:
        if REGIME_MODEL == "jm":
            other = (online_hmm(hmm_features(r_dev, rv_dev), FIRST_FIT_END) > 0.5).astype(int)
        else:
            other = online_jm(r_dev, 30.0, JM_HALFLIVES, FIRST_FIT_END, clip=JM_CLIP, rv=rv_dev, kind=JM_FEATURES)
        both = pd.concat([states, other], axis=1).dropna()
        summary["agreement_with_other_model"] = float((both.iloc[:, 0] == both.iloc[:, 1]).mean())
    except Exception as e:
        summary["agreement_with_other_model"] = f"실패: {e}"
    save(summary, "step2", f"{market}_regime_summary")

    # ---- 레짐 LHAR (H1 레짐 버전)
    d = guard(har_features(mk["rv"], mk["r_pct"], regime=states)).copy()
    d.loc[d.index[-1], "y"] = np.nan
    preds = {m: walk_forward(d, spec(m, True), HAR_MIN_TRAIN, HAR_REFIT_EVERY, log_target=True)
             for m in ("HAR", "LHAR", "RLHAR")}
    ev = d.index >= pd.Timestamp(DEV_START)
    save(compare_models(d[ev], {k: v[ev] for k, v in preds.items()}), "step2", f"{market}_regime_lhar_oos")
    # 레짐 항의 추가 기여: RLHAR을 HAR이 아니라 LHAR과 직접 비교
    save(compare_models(d[ev], {k: v[ev] for k, v in preds.items() if k != "HAR"}, bench="LHAR"),
         "step2", f"{market}_regime_lhar_oos_vs_LHAR")
    save(full_sample_fit(d, spec("RLHAR", True), log_target=True), "step2", f"{market}_regime_lhar_coef")

    # ---- S1 백테스트 (개발 구간)
    res = s1_all(mk, lam, end=DEV_END, dev_end=DEV_END, turbulent_weight_c=S1_C_TURBULENT_WEIGHT)
    res["states"] = states
    tab, nets, _ = run_all_rules(mk, res, cost, rf, DEV_START, DEV_END)
    save(tab, "step2", f"{market}_S1_dev_performance")
    for name in ("S1_A", "S1_C", "BH"):
        save(perf_by_regime(nets[name], states), "step2", f"{market}_{name}_by_regime")

    cheat = backtest_weights(res["weights"]["A"], mk["etf_ret"], cost, rf=rf, delay=0)["net"].loc[DEV_START:DEV_END]
    print(f"[{market}] 누수 점검: 규칙 A 샤프 정상={tab.loc['S1_A', 'sharpe']:.2f}, "
          f"당일 체결 가정(누수)={perf_stats(cheat, rf=rf)['sharpe']:.2f}  ← 이 차이가 look-ahead의 위험")

    return {"lambda": lam, "sigma_target": res["sigma_target"], "table": tab}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--rule", choices=["A", "B", "C"], default=None, help="동결할 S1 규칙 (기본: 개발 구간 샤프 최고)")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    if args.final_test:
        final_test(args.refresh)
        return

    frozen_rules = {}
    for market in MARKETS:
        mk = load_market(market, refresh=args.refresh)
        out = develop(market, mk)
        tab = out["table"]
        rule = args.rule or tab.loc[["S1_A", "S1_B", "S1_C"], "sharpe"].idxmax().split("_")[1]
        buf = 0.0 if rule == "A" else VT_BUFFER
        frozen_rules[market] = {
            "regime_model": REGIME_MODEL, "lambda": out["lambda"], "jm_features": JM_FEATURES,
            "halflives": list(JM_HALFLIVES), "sigma_target": out["sigma_target"], "rule": rule,
            "buffer": buf, "cost_oneway": MARKETS[market]["etf_cost_oneway"],
            "turbulent_weight_c": S1_C_TURBULENT_WEIGHT,
            "expected": {"sharpe": float(tab.loc[f"S1_{rule}", "sharpe"]),
                         "max_dd": float(tab.loc[f"S1_{rule}", "max_dd"])},
        }
    save(frozen_rules, "step2", "S1_candidate_rules")
    if args.freeze:
        payload = freeze("S1", frozen_rules)
        print(f"\nS1 규칙 동결 완료: {payload['frozen_at']} sha256={payload['sha256'][:12]}")


def final_test(refresh=False):
    rules = load_frozen("S1")["rules"]
    for market, rl in rules.items():
        mk = load_market(market, refresh=refresh)
        mk_full = {k: (guard(v, allow_test=True, required=("S1",), context=f"step2 final {market}")
                       if isinstance(v, (pd.Series, pd.DataFrame)) else v) for k, v in mk.items()}
        rf = rf_for(market)
        res = s1_all(mk_full, rl["lambda"], sigma_target=rl["sigma_target"],
                     turbulent_weight_c=rl["turbulent_weight_c"], model=rl["regime_model"])
        tab, nets, _ = run_all_rules(mk_full, res, rl["cost_oneway"], rf, TEST_START, None, buffer=VT_BUFFER)
        save(tab, "step2", f"{market}_S1_FINAL_TEST_performance")
        save(perf_by_regime(nets[f"S1_{rl['rule']}"], res["states"]), "step2",
             f"{market}_S1_FINAL_TEST_by_regime")
        save_series(res["states"], out_dir("step2") / f"{market}_regime_full.parquet")
        exp = rl["expected"]["sharpe"]
        got = tab.loc[f"S1_{rl['rule']}", "sharpe"]
        se = tab.loc[f"S1_{rl['rule']}", "sharpe_se"]
        print(f"[{market}] 동결 규칙 {rl['rule']}: 개발 샤프 {exp:.2f} → 테스트 샤프 {got:.2f} (±{1.96 * se:.2f})")


if __name__ == "__main__":
    main()
