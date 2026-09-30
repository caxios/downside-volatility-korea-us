"""
12단계: S1v2 — 변동성 예측 기반 비중 조절 개선판 (반분산 대칭 모형, 22일 지평)

  python scripts/step12_s1v2_vol_timing.py --phase design    # 개발 구간(2005~2020): 후보 4개 비교, 예측 정확도
  python scripts/step12_s1v2_vol_timing.py --freeze          # 개발 샤프 최고 후보를 S1v2로 동결
  python scripts/step12_s1v2_vol_timing.py --phase test      # 동결 후 2021~ 평가 (※ 오염된 테스트, 보고서 참고)
  python scripts/step12_s1v2_vol_timing.py --phase paper     # 매 거래일 장 마감 후: 다음 거래일 목표 비중 기록

후보 (결과 보기 전에 확정): {LHAR, LUHAR} × {h = 1, 22}. LHAR·h1 = 기존 S1_B.
선택 규칙: 시장별 개발 구간 순수익 샤프 최고 후보. 버퍼 VT_BUFFER, ETF 비용, 목표 변동성 = 개발 구간 평균 예측 변동성.

결과: results/step12/*.csv, logs/paper_s1v2.csv
"""
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.backtest import backtest_weights, perf_table
from src.config import DEV_END, DEV_START, LOG_DIR, MARKETS, TEST_START, VT_BUFFER
from src.data_io import load_us_rf_daily
from src.firewall import freeze, guard, load_frozen
from src.har import compare_models, hac_mean_t
from src.multitest import adjust
from src.pipeline import load_market, s1_all
from src.report import save
from src.s1v2 import HORIZONS, MODELS, all_forecasts, cand_name, vt_weights
from src.strategies import buy_hold_weights, ma_rule_weights, naive_voltarget_weights

REQ_TEST = ("S1", "S2S3", "S1v2")
PAPER_LOG = LOG_DIR / "paper_s1v2.csv"


def rf_for(market):
    if market == "US":
        try:
            return load_us_rf_daily()
        except Exception as e:
            print(f"[경고] 미국 무위험 금리 로드 실패 → 0 가정 ({e})")
    return None


def guarded_market(market, allow_test):
    mk = load_market(market)
    if not allow_test:
        return {k: (guard(v) if isinstance(v, (pd.Series, pd.DataFrame)) else v) for k, v in mk.items()}
    return {k: (guard(v, allow_test=True, required=REQ_TEST, context=f"step12 test {market}")
                if isinstance(v, (pd.Series, pd.DataFrame)) else v) for k, v in mk.items()}


def strategy_weights(mk, fcs, sigmas, s1_frozen):
    """후보 4개 + 벤치마크 비중. sigmas: {후보: 목표 일변동성(%)}"""
    w = {}
    for (m, h), (fc, _) in fcs.items():
        c = cand_name(m, h)
        w[c] = (vt_weights(fc, sigmas[c]), VT_BUFFER)
    w["BH"] = (buy_hold_weights(mk["etf_ret"].index), 0.0)
    w["MA200"] = (ma_rule_weights(mk["close"]), 0.0)
    w["naiveVT"] = (naive_voltarget_weights(mk["r_pct"], sigmas["LHAR_h1"]), VT_BUFFER)
    res = s1_all(mk, s1_frozen["lambda"], sigma_target=s1_frozen["sigma_target"],
                 turbulent_weight_c=s1_frozen["turbulent_weight_c"], model=s1_frozen["regime_model"])
    w["S1_A(동결)"] = (res["weights"]["A"], 0.0)
    return w


def run_backtests(mk, weights, cost, rf, start, end):
    nets, tovs = {}, {}
    for name, (w, buf) in weights.items():
        bt = backtest_weights(w, mk["etf_ret"], cost, rf=rf, buffer=buf).loc[start:end]
        nets[name], tovs[name] = bt["net"], bt["turnover"]
    common = pd.concat(nets, axis=1).dropna().index            # 모든 전략이 같은 날짜로 비교되게
    nets = {k: v.reindex(common) for k, v in nets.items()}
    tovs = {k: v.reindex(common) for k, v in tovs.items()}
    return perf_table(nets, rf=rf, turnover=tovs), nets


def forecast_accuracy(fcs, start, end):
    rows = []
    for h in HORIZONS:
        d = fcs[("LHAR", h)][1]
        preds = {cand_name(m, h): fcs[(m, h)][0] for m in MODELS}
        idx = d.index[(d.index >= pd.Timestamp(start)) & (d.index <= pd.Timestamp(end))]
        cmp_ = compare_models(d.loc[idx], {k: v.loc[idx] for k, v in preds.items()},
                              bench=cand_name("LHAR", h), lags=max(10, 2 * h))
        cmp_["h"] = h
        rows.append(cmp_)
    return pd.concat(rows)


def diff_tests(nets, strat, others, lags=10):
    rows = {}
    for o in others:
        dlt = (nets[strat] - nets[o]).dropna()
        m, t = hac_mean_t(dlt, lags)
        rows[o] = {"ann_diff_pct": 100 * 252 * m, "t_NW": t, "n_days": len(dlt)}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------- design
def phase_design(do_freeze=False):
    s1 = load_frozen("S1")["rules"]
    rules = {}
    for market, cfg in MARKETS.items():
        mk = guarded_market(market, allow_test=False)
        fcs = all_forecasts(mk["rv"], mk["r_pct"])
        sigmas = {cand_name(m, h): float(np.sqrt(fc.loc[:DEV_END].dropna()).mean()) for (m, h), (fc, _) in fcs.items()}
        rf = rf_for(market)
        tab, nets = run_backtests(mk, strategy_weights(mk, fcs, sigmas, s1[market]), cfg["etf_cost_oneway"],
                                  rf, DEV_START, DEV_END)
        save(tab, "step12", f"{market}_dev_performance")
        save(forecast_accuracy(fcs, DEV_START, DEV_END), "step12", f"{market}_dev_forecast_accuracy")
        cands = [cand_name(m, h) for m in MODELS for h in HORIZONS]
        best = tab.loc[cands, "sharpe"].idxmax()
        model, h = best.rsplit("_h", 1)
        rules[market] = {"candidate": best, "model": model, "h": int(h), "features": MODELS[model],
                         "sigma_target": sigmas[best], "buffer": VT_BUFFER, "cost_oneway": cfg["etf_cost_oneway"],
                         "min_train": 1000, "refit_every": 22,
                         "expected": {"sharpe": float(tab.loc[best, "sharpe"]),
                                      "max_dd": float(tab.loc[best, "max_dd"])},
                         "dev_sharpe_all": {c: float(tab.loc[c, "sharpe"]) for c in cands}}
        print(f"[{market}] 개발 샤프 최고 후보: {best} ({tab.loc[best, 'sharpe']:.2f})")
    save(rules, "step12", "S1v2_candidate_rules")
    if do_freeze:
        payload = freeze("S1v2", rules)
        print(f"\nS1v2 동결: {payload['frozen_at']} sha256={payload['sha256'][:12]}")


# ---------------------------------------------------------------- test
def phase_test():
    rules = load_frozen("S1v2")["rules"]
    s1 = load_frozen("S1")["rules"]
    fam = []
    for market, rl in rules.items():
        mk = guarded_market(market, allow_test=True)
        fcs = all_forecasts(mk["rv"], mk["r_pct"])
        cands = [cand_name(m, h) for m in MODELS for h in HORIZONS]
        # 후보들의 목표 변동성도 개발 구간 값으로 고정 (동결 파일의 선택 후보 + 나머지는 개발 구간 재계산)
        dev_fcs = all_forecasts(mk["rv"].loc[:DEV_END], mk["r_pct"].loc[:DEV_END])
        sigmas = {cand_name(m, h): float(np.sqrt(fc.dropna()).mean()) for (m, h), (fc, _) in dev_fcs.items()}
        assert abs(sigmas[rl["candidate"]] - rl["sigma_target"]) < 1e-9, "동결 목표 변동성과 재계산 값이 다름"
        rf = rf_for(market)
        weights = strategy_weights(mk, fcs, sigmas, s1[market])
        weights["S1v2(동결)"] = weights[rl["candidate"]]
        tab, nets = run_backtests(mk, weights, rl["cost_oneway"], rf, TEST_START, None)
        save(tab, "step12", f"{market}_TEST_performance")
        save(forecast_accuracy(fcs, TEST_START, str(mk["rv"].index[-1].date())), "step12",
             f"{market}_test_forecast_accuracy")
        dt = diff_tests(nets, "S1v2(동결)", ["BH", "MA200", "naiveVT", "S1_A(동결)"])
        dt["market"] = market
        fam.append(dt)
        exp = rl["expected"]["sharpe"]
        print(f"[{market}] S1v2={rl['candidate']}: 개발 샤프 {exp:.2f} → 테스트 {tab.loc['S1v2(동결)', 'sharpe']:.2f}"
              f" (±{1.96 * tab.loc['S1v2(동결)', 'sharpe_se']:.2f})")
    d = pd.concat(fam)
    d["vs"] = d.index
    d = adjust(d.reset_index(drop=True), p_col="p", t_col="t_NW")
    save(d, "step12", "TEST_diff_tests_adjusted")


# ---------------------------------------------------------------- paper
def phase_paper():
    rules = load_frozen("S1v2")["rules"]
    df = pd.read_csv(PAPER_LOG) if PAPER_LOG.exists() else pd.DataFrame()
    for market, rl in rules.items():
        mk = load_market(market, refresh=True)
        from src.s1v2 import features, walk_forward_h
        d = features(mk["rv"], mk["r_pct"], rl["h"])
        fc = walk_forward_h(d, rl["features"], rl["h"], rl["min_train"], rl["refit_every"])
        # 마지막 행은 아직 예측이 없을 수 있음(재추정 블록 경계) → 마지막 날 특징으로 가장 최근 계수 적용
        fc = fc.dropna()
        w = vt_weights(fc, rl["sigma_target"])
        last = w.index[-1]
        row = {"run_time": datetime.now().isoformat(timespec="seconds"), "market": market,
               "data_date": str(last.date()), "candidate": rl["candidate"], "forecast_var": float(fc.iloc[-1]),
               "target_weight_raw": float(w.iloc[-1]), "buffer": rl["buffer"],
               "etf_close": float(mk["etf"]["Close"].iloc[-1]), "etf_last_date": str(mk["etf"].index[-1].date())}
        if row["data_date"] != str(mk["rv"].index[-1].date()):
            row["warning"] = f"최신 데이터({mk['rv'].index[-1].date()})의 예측이 아직 없음 — 재추정 블록 경계"
        if len(df) and ((df["market"] == market) & (df["data_date"].astype(str) == row["data_date"])).any():
            print(f"[{market}] {row['data_date']} 이미 기록됨")
            continue
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        print(row)
    df.to_csv(PAPER_LOG, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["design", "test", "paper"])
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        phase_design(do_freeze=True)
    elif a.phase == "design":
        phase_design()
    elif a.phase == "test":
        phase_test()
    elif a.phase == "paper":
        phase_paper()
    else:
        ap.error("--phase 또는 --freeze 필요")


if __name__ == "__main__":
    main()
