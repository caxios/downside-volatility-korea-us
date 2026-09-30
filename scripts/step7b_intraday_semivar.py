"""
7b단계: 장중 봉으로 만든 상승/하락 반분산 → 이후 상승/하락 반분산

  python scripts/step7b_intraday_semivar.py

하루의 RS⁺ / RS⁻ 를 장중 봉 수익률로 계산한다 (vol.intraday_daily_measures, 장중만·오버나이트 제외).

사전 확정 설정
  60분봉 (약 3년): 설명변수 일·주(5일)·월(22일), 목표 h = 1, 5, 22일, NW/DK 시차 max(22, 2h)
  5분봉 · 10분봉 (약 60거래일): 설명변수 일·주(5일), 목표 h = 1, 5일, NW/DK 시차 max(5, 2h)
      10분봉은 5분봉 종가를 10분 구간의 마지막 값으로 묶어 만든다.
  반일장·결측이 많은 날(그 종목 하루 봉 수 중앙값의 80% 미만)은 제외한다.

분석 단위
  index : 지수 단독 시계열 (^GSPC, ^KS11), 단위 %²,  Newey-West
  etf   : 지수 ETF (SPY, 069500.KS) — 지수 결과 확인용
  daily : 같은 기간·같은 날짜의 지수 일봉(종가 대 종가, 오버나이트 포함) 버전 — 기간 효과와 측정 효과를 가르는 대조군
  panel : 개별 종목을 묶은 합동 회귀, Driscoll-Kraay 표준오차.
          종목마다 변동성 수준이 달라서 각 종목의 RS±를 그 종목의 평균 일간 RV로 나눠 단위를 없앤다.

장중 데이터는 모두 2021년 이후(테스트 구간)이므로 S1·S2S3 동결 후에만 실행된다.
결과: results/step7b/*.csv
"""
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import INTRADAY_BASE
from src.data_io import exchange_tz, load_intraday
from src.firewall import FirewallError, is_frozen, log_access
from src.pipeline import load_market
from src.report import save
from src.semivar import (TARGETS, coef_table, equality_tests, fit, fit_panel, regressors, semivar_features,
                         semivar_frame)
from src.vol import intraday_daily_measures

SPECS = {
    "60m": dict(source="60m", windows={"w": 5, "m": 22}, horizons=(1, 5, 22), lags=lambda h: max(22, 2 * h)),
    "5m": dict(source="5m", windows={"w": 5}, horizons=(1, 5), lags=lambda h: max(5, 2 * h)),
    "10m": dict(source="5m", resample="10min", windows={"w": 5}, horizons=(1, 5), lags=lambda h: max(5, 2 * h)),
}
INDEX = {"US": "^GSPC", "KR": "^KS11"}
ETF = {"US": "SPY", "KR": "069500.KS"}
NON_STOCK = set(INTRADAY_BASE["us"]) | set(INTRADAY_BASE["kr"])


def market_of(t):
    return "KR" if exchange_tz(t) == "Asia/Seoul" else "US"


def resample_bars(g, rule):
    """5분봉 → rule 봉: 각 구간의 마지막 종가 (UTC 기준 구간 = 현지 시각 기준 구간, 시차가 정시 단위라서)"""
    c = g["Close"].astype(float).resample(rule).last().dropna()
    return pd.DataFrame({"Close": c})


def daily_semivar(bars, spec):
    """{ticker: DataFrame[RSp, RSn, N]} — 반일장·결측일 제외"""
    out = {}
    for t, g in bars.groupby("ticker"):
        if spec.get("resample"):
            g = resample_bars(g, spec["resample"])
        dm = intraday_daily_measures(g, exchange_tz(t))
        dm = dm[dm["N"] >= 0.8 * dm["N"].median()]
        if len(dm) < 30:
            continue
        out[t] = dm[["RS_pos", "RS_neg", "N", "overnight_pct"]].rename(
            columns={"RS_pos": "RSp", "RS_neg": "RSn", "overnight_pct": "ON"})
    return out


def run(d, spec, name, panel_time=None):
    pos, neg, xs = regressors(spec["windows"])
    coefs, tests = [], []
    for h in spec["horizons"]:
        for s in TARGETS:
            y = f"y_{s}_h{h}"
            lags = spec["lags"](h)
            if panel_time:
                res = fit_panel(d, y, xs, panel_time, lags)
                ct = _coef_panel(res, d.dropna(subset=xs + [y]), y)
            else:
                res = fit(d, y, xs, lags)
                ct = coef_table(res, d, y)
            ct.insert(0, "target", s)
            ct.insert(1, "h", h)
            ct["n"], ct["R2"] = int(res.nobs), res.rsquared
            coefs.append(ct)
            et = equality_tests(res, pos, neg)
            et.insert(0, "target", s)
            et.insert(1, "h", h)
            tests.append(et)
    coefs, tests = pd.concat(coefs), pd.concat(tests)
    coefs.index.name, tests.index.name = "term", "test"
    save(coefs, "step7b", f"{name}_coef", show=False)
    save(tests, "step7b", f"{name}_equality", show=False)
    return coefs, tests


def add_overnight(d, on, windows, scale=1.0):
    """밤사이(전일 마지막 봉 → 당일 첫 봉) 수익률의 상승/하락 제곱: ONp, ONn (일·주·월)"""
    on = on.reindex(d.index)
    d["ONp_d"] = np.where(on > 0, on ** 2, 0.0) / scale
    d["ONn_d"] = np.where(on < 0, on ** 2, 0.0) / scale
    d.loc[on.isna(), ["ONp_d", "ONn_d"]] = np.nan
    for s in ("ONp", "ONn"):
        for k, n in windows.items():
            d[f"{s}_{k}"] = d[f"{s}_d"].rolling(n).mean()
    return d


def run_decomp(d, spec, name, panel_time=None):
    """장중 RS± 와 밤사이 ON± 를 한 회귀에 넣고, 각 블록의 총영향·방향 차이를 검정"""
    pos, neg, xs = regressors(spec["windows"])
    onp, onn = [c.replace("RSp", "ONp") for c in pos], [c.replace("RSn", "ONn") for c in neg]
    xs = xs + [c for pair in zip(onp, onn) for c in pair]
    coefs, tests = [], []
    for h in spec["horizons"]:
        for s in TARGETS:
            y = f"y_{s}_h{h}"
            lags = spec["lags"](h)
            if panel_time:
                res = fit_panel(d, y, xs, panel_time, lags)
                ct = _coef_panel(res, d.dropna(subset=xs + [y]), y)
            else:
                res = fit(d, y, xs, lags)
                ct = coef_table(res, d, y)
            ct.insert(0, "target", s)
            ct.insert(1, "h", h)
            ct["n"], ct["R2"] = int(res.nobs), res.rsquared
            coefs.append(ct)
            for block, (a, b) in (("장중", (pos, neg)), ("밤사이", (onp, onn))):
                et = equality_tests(res, a, b)
                et.insert(0, "block", block)
                et.insert(1, "target", s)
                et.insert(2, "h", h)
                tests.append(et)
    coefs, tests = pd.concat(coefs), pd.concat(tests)
    coefs.index.name, tests.index.name = "term", "test"
    save(coefs, "step7b", f"{name}_coef", show=False)
    save(tests, "step7b", f"{name}_equality", show=False)


def _coef_panel(res, dd, y):
    sd_y = dd[y].std()
    rows = {}
    for k in res.params.index:
        rows[k] = {"coef": res.params[k], "se_NW": res.bse[k], "t": res.tvalues[k], "p": res.pvalues[k],
                   "std_coef": res.params[k] * dd[k].std() / sd_y if k != "const" else np.nan}
    return pd.DataFrame(rows).T


def main():
    missing = [r for r in ("S1", "S2S3") if not is_frozen(r)]
    if missing:
        raise FirewallError(f"장중 데이터는 테스트 구간(2021~)이므로 먼저 동결할 것: {missing}")
    log_access("step7b intraday semivar", ("S1", "S2S3"))

    bars_cache, summary = {}, []
    for freq, spec in SPECS.items():
        if spec["source"] not in bars_cache:
            bars_cache[spec["source"]] = load_intraday(spec["source"])
        sv = daily_semivar(bars_cache[spec["source"]], spec)

        for market in ("US", "KR"):
            # 지수·ETF 단독 시계열 (단위 %²)
            for unit, t in (("index", INDEX[market]), ("etf", ETF[market])):
                if t not in sv:
                    print(f"[{freq} {market}] {t} 데이터 없음")
                    continue
                x = sv[t]
                d = semivar_frame(x["RSp"], x["RSn"], spec["horizons"], spec["windows"])
                run(d, spec, f"{freq}_{unit}_{market}")
                if freq == "60m" and unit == "index":
                    run_decomp(add_overnight(d, x["ON"], spec["windows"]), spec, f"{freq}_decomp_index_{market}")
                summary.append({"freq": freq, "market": market, "unit": unit, "tickers": 1, "days": len(x),
                                "start": x.index.min().date(), "end": x.index.max().date(),
                                "bars_per_day_median": x["N"].median() + 1})

            # 대조군: 같은 날짜의 일봉 반분산 (설명변수의 과거 창은 그 이전 일봉까지 사용)
            if INDEX[market] in sv:
                dates = sv[INDEX[market]].index
                dly = semivar_features(load_market(market)["r_pct"], spec["horizons"], spec["windows"])
                dly = dly[(dly.index >= dates.min()) & (dly.index <= dates.max())]
                run(dly, spec, f"{freq}_daily_{market}")
                summary.append({"freq": freq, "market": market, "unit": "daily", "tickers": 1, "days": len(dly),
                                "start": dly.index.min().date(), "end": dly.index.max().date(),
                                "bars_per_day_median": np.nan})

            # 개별 종목 패널 (종목별 평균 RV로 정규화)
            frames = []
            for t, x in sv.items():
                if market_of(t) != market or t in NON_STOCK:
                    continue
                scale = (x["RSp"] + x["RSn"]).mean()
                d = semivar_frame(x["RSp"] / scale, x["RSn"] / scale, spec["horizons"], spec["windows"])
                d["ticker"], d["date"] = t, d.index
                frames.append(d)
            P = pd.concat(frames, ignore_index=True)
            run(P, spec, f"{freq}_panel_{market}", panel_time="date")
            if freq == "60m":
                frames = []
                for t, x in sv.items():
                    if market_of(t) != market or t in NON_STOCK:
                        continue
                    scale = (x["RSp"] + x["RSn"]).mean()
                    d = semivar_frame(x["RSp"] / scale, x["RSn"] / scale, spec["horizons"], spec["windows"])
                    d = add_overnight(d, x["ON"], spec["windows"], scale)
                    d["ticker"], d["date"] = t, d.index
                    frames.append(d)
                run_decomp(pd.concat(frames, ignore_index=True), spec, f"{freq}_decomp_panel_{market}",
                           panel_time="date")
            summary.append({"freq": freq, "market": market, "unit": "panel", "tickers": len(frames),
                            "days": P["date"].nunique(), "start": P["date"].min().date(),
                            "end": P["date"].max().date(), "bars_per_day_median": np.nan})
    save(pd.DataFrame(summary), "step7b", "sample_summary")


if __name__ == "__main__":
    main()
