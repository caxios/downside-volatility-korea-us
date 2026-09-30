"""
합성 데이터로 모든 모듈을 점검한다 (네트워크 불필요).
실행: python -m pytest tests -q   또는   python tests/test_synthetic.py

합성 데이터 설계
  - 레짐: 2상태 마르코프 체인 (평온 0.8%, 혼란 2.2% 일변동성)
  - 레버리지 효과: 전날 하락이 오늘 변동성을 키움 → LHAR이 HAR보다 나아야 함
  - 하루 78개 장중 봉으로 OHLC와 5분봉 생성
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import backtest as bt
from src import cross_section as cs
from src import firewall as fw
from src import har, regime, shortban, strategies, vol
from src.quality import clean_daily, quality_report

RNG = np.random.default_rng(7)
N_DAYS = 3200
N_BARS = 78


def make_market(n_days=N_DAYS, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2000-01-03", periods=n_days)
    P = np.array([[0.99, 0.01], [0.03, 0.97]])
    s = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        s[t] = rng.choice(2, p=P[s[t - 1]])
    base = np.where(s == 0, 0.8, 2.2)
    mu = np.where(s == 0, 0.05, -0.05)
    r_prev = 0.0
    O, H, L, C, R = [], [], [], [], []
    price = 100.0
    bars = []
    for t in range(n_days):
        sig2 = 0.7 * base[t] ** 2 + 0.3 * (2.0 * r_prev ** 2 if r_prev < 0 else 0.3 * r_prev ** 2)
        sig = np.sqrt(sig2)
        steps = rng.normal(mu[t] / N_BARS, sig / np.sqrt(N_BARS), N_BARS) / 100.0
        path = price * np.exp(np.cumsum(steps))
        O.append(price); H.append(max(price, path.max())); L.append(min(price, path.min())); C.append(path[-1])
        day_r = np.log(path[-1] / price) * 100
        R.append(day_r)
        if t >= n_days - 150:
            ts = pd.date_range(dates[t] + pd.Timedelta(hours=9, minutes=35), periods=N_BARS, freq="5min",
                               tz="America/New_York").tz_convert("UTC")
            bars.append(pd.DataFrame({"Close": path}, index=ts))
        price = path[-1]
        r_prev = day_r
    df = pd.DataFrame({"Open": O, "High": H, "Low": L, "Close": C, "Adj Close": C,
                       "Volume": rng.integers(1e5, 1e6, n_days)}, index=dates)
    return df, pd.Series(s, index=dates, name="true"), pd.concat(bars)


MKT, TRUE_S, BARS = make_market()
IDX = clean_daily(MKT)
R_PCT = IDX["r"] * 100
RV = vol.parkinson(IDX)


def test_quality():
    rep = quality_report(MKT, "synthetic")
    assert rep["high_lt_low"] == 0 and rep["close_outside_HL"] == 0


def test_vol_identities():
    m = vol.measures(RNG.normal(0, 1, 78))
    assert abs(m["RS_pos"] + m["RS_neg"] - m["RV"]) < 1e-9
    assert -1 <= m["RSJ"] <= 1
    dm = vol.intraday_daily_measures(BARS, "America/New_York")
    assert len(dm) >= 140 and (dm["N"] == N_BARS - 1).mean() > 0.95
    both = pd.concat([dm["RV"], RV], axis=1).dropna()
    assert both.corr().iloc[0, 1] > 0.5, "5분봉 RV와 Parkinson이 서로 관련 있어야 함"
    pm = vol.period_measures(IDX["r"])
    assert {"RSJ", "RSkew", "MAX", "SUM"} <= set(pm.columns)


def test_har_lhar():
    d = har.har_features(RV, R_PCT)
    p = {m: har.walk_forward(d, har.spec(m, log=True), min_train=800, refit_every=22, log_target=True)
         for m in ("HAR", "LHAR")}
    p["HAR_level"] = har.walk_forward(d, har.SPECS["HAR"], min_train=800, refit_every=22)
    tab = har.compare_models(d, p)
    print("\n[HAR vs LHAR]\n", tab.round(4))
    assert tab.loc["LHAR", "QLIKE"] < tab.loc["HAR", "QLIKE"], "레버리지 효과를 심었으니 LHAR이 나아야 함"
    cheat = har.lookahead_sanity(d, har.SPECS["HAR"], min_train=800, refit_every=22)
    assert cheat < 0.5 * tab.loc["HAR", "QLIKE"], "미래값을 넣으면 손실이 크게 줄어야 함(누수 감지)"
    fit = har.full_sample_fit(d, har.SPECS["LHAR"])
    assert fit.loc["L", "coef"] > 0


def test_jm_online_offline():
    states = regime.online_jm(R_PCT, lam=30.0, first_fit_end="2003-12-31", n_init=4, rv=RV)
    both = pd.concat([states, TRUE_S], axis=1).dropna()
    acc = (both.iloc[:, 0] == both.iloc[:, 1]).mean()
    off = regime.offline_jm(R_PCT, lam=30.0, n_init=4, rv=RV)
    diag = regime.regime_diagnostics(states, R_PCT, offline=off)
    print(f"\n[JM] online 정확도={acc:.3f}, 전환/년={diag['switches_per_year']:.1f}, "
          f"online-offline 일치={diag['online_offline_agreement']:.3f}, 감지지연 중앙값={diag['detection_lag_median']}")
    assert acc > 0.75
    assert diag["stats"].loc[1, "std"] > diag["stats"].loc[0, "std"], "레짐 1이 혼란(고변동)이어야 함"
    # 작은 λ일수록 전환이 잦아야 함
    s_small = regime.online_jm(R_PCT, lam=1.0, first_fit_end="2003-12-31", n_init=2, rv=RV)
    sw_small = (s_small.diff().abs() > 0).sum()
    sw_big = (states.diff().abs() > 0).sum()
    assert sw_small >= sw_big


def test_jm_toy_example():
    """설명에 썼던 8일 예시: λ=3이면 3일째 튐은 무시, 7일째에 혼란 판정"""
    x = np.array([1, 1, 3, 1, 1, 3, 3, 3], dtype=float)[:, None]
    theta = np.array([[1.0], [3.0]])
    V, _ = regime.forward_pass(regime._loss(x, theta), 3.0)
    online = V.argmin(1).tolist()
    assert online == [0, 0, 0, 0, 0, 0, 1, 1]
    s, obj = regime.dp_assign(x, theta, 3.0)
    assert s.tolist() == [0, 0, 0, 0, 0, 1, 1, 1] and abs(obj - 5.0) < 1e-9


def test_hmm_online():
    F = regime.hmm_features(R_PCT, RV)
    p = regime.online_hmm(F, first_fit_end="2003-12-31", n_init=2)
    both = pd.concat([(p > 0.5).astype(int), TRUE_S], axis=1).dropna()
    acc = (both.iloc[:, 0] == both.iloc[:, 1]).mean()
    print(f"\n[HMM] online 정확도={acc:.3f}")
    assert acc > 0.7


def test_backtest_and_s1():
    states = regime.online_jm(R_PCT, lam=30.0, first_fit_end="2003-12-31", n_init=4, rv=RV)
    ret = np.expm1(IDX["r"])
    d = har.har_features(RV, R_PCT)
    fc = har.walk_forward(d, har.spec("LHAR"), min_train=800, refit_every=22, log_target=True)
    sig_t = float(np.sqrt(fc.dropna()).mean())
    wA = strategies.s1_weights(states, fc, sig_t, "A")
    wB = strategies.s1_weights(states, fc, sig_t, "B")
    wC = strategies.s1_weights(states, fc, sig_t, "C")
    series, tov = {}, {}
    for name, w, buf in (("A", wA, 0), ("B", wB, 0.1), ("C", wC, 0.1),
                         ("BH", strategies.buy_hold_weights(ret.index), 0),
                         ("MA200", strategies.ma_rule_weights(IDX["Close"]), 0),
                         ("naiveVT", strategies.naive_voltarget_weights(R_PCT, sig_t), 0.1)):
        res = bt.backtest_weights(w, ret, cost_oneway=0.0005, buffer=buf)
        series[name] = res["net"].loc["2004-01-01":]
        tov[name] = res["turnover"].loc["2004-01-01":]
    tab = bt.perf_table(series, turnover=tov)
    print("\n[S1 성과]\n", tab[["ann_ret", "ann_vol", "sharpe", "max_dd", "ann_turnover"]].round(3))
    assert tab.loc["A", "max_dd"] > tab.loc["BH", "max_dd"], "레짐 스위치는 낙폭을 줄여야 함"
    cheat = bt.backtest_weights(wA, ret, delay=0)["net"].loc["2004-01-01":]
    assert bt.perf_stats(cheat)["sharpe"] > tab.loc["A", "sharpe"], "delay=0(누수)이면 성과가 부풀려져야 함"
    pr = bt.perf_by_regime(series["BH"], states)
    assert len(pr) == 2


def make_panel(n_stocks=120, n_months=150, seed=3):
    """monthly_signals 로 만든 실제 시그널 + 심어둔 음의 RSJ 효과로 fwd_ret 대체"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2004-01-01", periods=n_months * 21)
    mkt = pd.Series(rng.normal(0.0003, 0.01, len(dates)), index=dates)
    daily = {}
    for i in range(n_stocks):
        beta = rng.uniform(0.5, 1.5)
        r = beta * mkt + rng.normal(0, rng.uniform(0.01, 0.03), len(dates)) \
            + rng.standard_t(3, len(dates)) * 0.003
        px = 50 * np.exp(np.cumsum(r))
        df = pd.DataFrame({"Open": px, "High": px * 1.01, "Low": px * 0.99, "Close": px, "Adj Close": px,
                           "Volume": rng.integers(1e4, 1e6, len(dates))}, index=dates)
        daily[f"S{i:03d}"] = clean_daily(df)
    months = pd.period_range(dates[0], dates[-1], freq="M")
    member = {m: set(list(daily)[: n_stocks - 10]) | ({"S119"} if m.month % 2 else set()) for m in months}
    panel = cs.build_panel(daily, mkt, member)
    # 심은 효과: RSJ 순위가 높을수록 다음 달 수익률 낮음
    rk = panel.groupby("month")["RSJ"].transform(cs.to_unit_rank)
    panel["fwd_ret"] = -0.02 * rk + rng.normal(0, 0.06, len(panel))
    for h in (1, 2, 3, 6):
        panel[f"fwd_ret_{h}"] = panel["fwd_ret"] if h == 1 else panel["fwd_ret"] * 0.5 + rng.normal(0, 0.05, len(panel))
    return panel, daily, member


PANEL, DAILY, MEMBER = make_panel()


def test_cross_section():
    p = PANEL
    assert {"RSJ", "RSkew", "MAX", "IVOL", "REV", "MOM", "SIZE", "ILLIQ"} <= set(p.columns)
    ics = cs.ic_summary(p, ["RSJ", "MAX", "IVOL"], split="2009-12")
    print("\n[IC]\n", ics.round(3))
    assert ics.loc["RSJ", "mean_IC"] < -0.05 and ics.loc["RSJ", "t_NW"] < -3
    dec = cs.ic_decay(p, "RSJ")
    assert len(dec) == 4
    q = cs.quantile_returns(p, "RSJ", 5)
    qs = cs.quantile_summary(q, direction=-1, bench=cs.ew_benchmark(p))
    assert qs["LS_mean"] > 0 and qs["monotonic_rho"] < -0.8   # 방향 -1: 분위가 높을수록 수익률 낮음
    fm, B = cs.fama_macbeth(p, ["RSJ", "REV", "MAX", "IVOL", "SIZE"])
    print("\n[FM]\n", fm.round(4))
    assert fm.loc["RSJ", "coef"] < 0 and fm.loc["RSJ", "t_NW"] < -3
    p2 = cs.neutralize(p, "RSJ", ["REV", "SIZE"])
    ic_neu = cs.rank_ic(p2, "RSJ_neu")
    rho = p.groupby("month")[["RSJ", "REV"]].apply(lambda g: g["RSJ"].rank().corr(g["REV"].rank())).mean()
    print(f"\n[월간 RSJ–REV 순위상관 평균] {rho:.2f}  (일봉 기반 월간 RSJ는 월수익률과 기계적으로 얽힘)")
    print(f"[중립화 전/후 평균 IC] {cs.rank_ic(p, 'RSJ').mean():.3f} → {ic_neu.mean():.3f}")
    assert ic_neu.mean() < -0.02
    ds = cs.dependent_double_sort(p, "REV", "RSJ", direction=-1)
    assert ds.mean() > 0
    D = pd.Series((np.arange(len(B)) // 20) % 2, index=B.index, dtype=float)
    rt = cs.regime_test(cs.rank_ic(p, "RSJ"), D)
    assert {"calm_mean", "t_diff"} <= set(rt.index)
    ep = cs.ic_by_episode(cs.rank_ic(p, "RSJ"), D)
    assert len(ep) > 0
    ok, reasons = cs.s2_signal_passes(-5, -0.1, -0.08, -0.9, 0.003,
                                      dict(min_abs_t=2, same_sign_halves=True, min_abs_monotonic=0.8,
                                           long_leg_positive=True))
    assert ok, reasons


def test_s2_s3_portfolio():
    p = PANEL
    holdings = strategies.run_s2(p, {"RSJ": -1}, entry_pct=0.2, buffer_pct=0.1)
    fwd = p.pivot_table(index="month", columns="ticker", values="fwd_ret")
    port = bt.portfolio_monthly(holdings, fwd, cost_oneway=0.001, sell_tax=0.0018)
    st = bt.perf_stats(port["net"], periods=12)
    print(f"\n[S2] 월평균 순수익={port['net'].mean():.4f}, 평균 회전율={port['turnover'].mean():.2f}, 샤프={st['sharpe']:.2f}")
    assert port["net"].mean() > 0
    # 버퍼가 회전율을 줄이는지
    h0 = strategies.run_s2(p, {"RSJ": -1}, entry_pct=0.2, buffer_pct=0.0)
    port0 = bt.portfolio_monthly(h0, fwd, 0.001)
    assert port["turnover"].mean() <= port0["turnover"].mean()
    daily_ret = pd.DataFrame({t: np.expm1(df["r"]) for t, df in DAILY.items()})
    sl = bt.sleeve_daily(holdings, daily_ret)
    w1 = pd.Series(1.0, index=daily_ret.index)
    s3 = strategies.run_s3(sl, w1, 0.001, 0.0018, 0.0002)
    assert len(s3) > 100 and s3["net"].notna().all()


def test_shortban():
    p = PANEL
    _, B1 = cs.fama_macbeth(p, ["RSJ", "REV", "SIZE"])
    b_kr = B1["RSJ"]
    b_us = B1["RSJ"] * 0.5 + RNG.normal(0, 0.01, len(B1))
    months = b_kr.index
    ban_s, ban_e = str(months[100]), str(months[116])
    b_kr = b_kr.copy()
    b_kr.iloc[100:117] -= 0.03      # 금지 기간에 한국만 효과 강화
    res = shortban.country_did(b_kr, b_us, ban_s, ban_e)
    print("\n[DiD]\n", res.round(4))
    assert res.loc["Ban", "coef"] < 0
    plac = shortban.placebo_ban(b_kr - b_us, 17, dev_end=str(months[95]))
    pv = shortban.placebo_pvalue(res.loc["Ban", "coef"], plac)
    assert 0 <= pv <= 1
    tick = sorted(p["ticker"].unique())
    ratio = pd.Series(RNG.uniform(0, 5, len(tick)), index=tick)
    hi, lo = shortban.high_low_short(ratio)
    wk = shortban.within_korea_did(p, "RSJ", hi, lo, ban_s, ban_e)
    assert "S_Ban_High" in wk.index
    ic = cs.rank_ic(p, "RSJ")
    h5 = shortban.ic_difference(ic, ic * 0.5)
    assert h5["mean_KR_minus_US"] < 0


def test_firewall():
    tmp = Path(tempfile.mkdtemp())
    fw.FROZEN_DIR, fw.LOG_DIR = tmp, tmp
    s = pd.Series(1.0, index=pd.bdate_range("2019-01-01", "2022-12-31"))
    assert s.pipe(fw.guard).index.max() < pd.Timestamp("2021-01-01")
    try:
        fw.guard(s, allow_test=True, required=("S1",))
        raise AssertionError("동결 전 테스트 접근이 막혀야 함")
    except fw.FirewallError:
        pass
    fw.freeze("S1", {"lambda": 30})
    assert len(fw.guard(s, allow_test=True, required=("S1",), context="unit")) == len(s)
    try:
        fw.freeze("S1", {"lambda": 60})
        raise AssertionError("재동결이 막혀야 함")
    except fw.FirewallError:
        pass
    assert fw.load_frozen("S1")["rules"]["lambda"] == 30
    P = pd.DataFrame({"month": pd.period_range("2019-01", "2022-12", freq="M"), "x": 1})
    assert fw.guard(P, month_col="month")["month"].max() < pd.Period("2021-01", "M")


if __name__ == "__main__":
    import inspect
    fns = [(n, f) for n, f in globals().items() if n.startswith("test_") and inspect.isfunction(f)]
    for name, fn in fns:
        fn()
        print(f"PASS {name}")
    print(f"\n모든 테스트 통과 ({len(fns)}개)")
