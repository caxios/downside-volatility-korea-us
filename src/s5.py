"""
S5: 주간 실현 왜도(realized skewness) 롱숏 전략 — Amaya, Christoffersen, Jacobs & Vasquez (2015, JFE) 재현

  실현 왜도 RSK = sqrt(N) · Σ rᵢ³ / (Σ rᵢ²)^1.5          (vol.measures 와 같은 정의)
  매주 말, 이번 주 RSK 가 낮은 종목을 사고 높은 종목을 판다
  롱숏 = 최저 RSK 분위 매수 + 최고 RSK 분위 매도 (롱: 왜도가 낮은 = 왼꼬리가 긴 = 하락이 컸던 종목)

주 단위(pandas "W"), 매 주말 마감 시점 기준 리밸런싱, 다음 주 보유 후 청산.
유니버스는 결정 주의 직전 월말 패널 구성원(가짜 데이터 포함) → 미래 정보 사용 없음.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ----------------------------------------------------------------- 주간 지표 계산
def weekly_from_daily(daily_ret: pd.DataFrame, window: int | None = None, min_obs: int = 4) -> dict:
    """
    일별 수익률 DataFrame → 주간 RSK, REV, RVOL, NEXT를 계산한다.

    daily_ret: 일별 단순수익률 (행=날짜, 열=종목)
    window=None : 해당 주의 일별 수익률만으로 RSK 계산 (논문의 '이번 주' 방식, 주당 약 5일)
    window=22   : 각 주 마지막 거래일 기준 최근 22거래일로 RSK 계산 (강건성 확인용)
    반환: {"RSK","REV","RVOL","NEXT"} 각각 (주 × 종목) DataFrame
    """
    # ── 1) 단순수익률을 로그수익률로 변환: log(1+r)
    #        로그수익률은 합산이 가능해서 주간 합 = 주간 로그수익률이 됨
    r = np.log1p(daily_ret.astype(float))

    # ── 2) 각 날짜가 속하는 주(Week Period)를 구함
    wk = r.index.to_period("W")

    if window is None:
        # ── 3a) 이번 주 방식: 해당 주에 속하는 일별 수익률만 사용
        # n: 주별·종목별 유효 관측치 수 (보통 5, 공휴일이면 4 이하)
        n = r.notna().groupby(wk).sum()
        # s2: Σ rᵢ² — 실현분산의 분자 (주별 합산)
        s2 = (r ** 2).groupby(wk).sum(min_count=1)
        # s3: Σ rᵢ³ — 실현왜도의 분자 (주별 합산)
        s3 = (r ** 3).groupby(wk).sum(min_count=1)
    else:
        # ── 3b) 롤링 윈도우 방식: 각 주의 마지막 거래일 기준 최근 window일
        # last: 각 주의 마지막 거래일 날짜를 구함
        last = pd.Series(r.index, index=r.index).groupby(wk).max()
        # 롤링 윈도우로 n, s2, s3를 계산하고, 주 마지막 날의 값만 추출
        n = r.notna().rolling(window).sum().loc[last.values]
        s2 = (r ** 2).rolling(window, min_periods=int(0.8 * window)).sum().loc[last.values]
        s3 = (r ** 3).rolling(window, min_periods=int(0.8 * window)).sum().loc[last.values]
        # 인덱스를 날짜에서 주(Period)로 바꿈
        n.index = s2.index = s3.index = last.index
        # 최소 관측치 수도 윈도우의 80%로 조정
        min_obs = int(0.8 * window)

    # ── 4) 실현 왜도 계산: RSK = √N · Σr³ / (Σr²)^1.5
    #        관측치가 min_obs 미만이거나 분산이 0이면 NaN 처리
    rsk = (np.sqrt(n) * s3 / s2 ** 1.5).where((n >= min_obs) & (s2 > 0))

    # ── 5) 통제변수 계산
    # REV: 주간 로그수익률 합 = 주간 반전(reversal) 시그널
    rev = r.groupby(wk).sum(min_count=1)
    # RVOL: 주간 실현변동성 = √(Σrᵢ²)
    rvol = np.sqrt((r ** 2).groupby(wk).sum(min_count=1))

    # ── 6) 다음 주 수익률 (종속변수) 계산
    # 단순수익률을 주간으로 복리 합산: (1+r₁)(1+r₂)...(1+r₅) − 1
    # 거래일이 3일 미만인 주는 NaN 처리 (너무 짧은 주 제외)
    cnt = daily_ret.notna().groupby(wk).sum()
    nxt = ((1 + daily_ret).groupby(wk).prod(min_count=1) - 1).where(cnt >= 3).shift(-1)
    # shift(-1): 이번 주 시그널로 다음 주 수익률을 예측하므로, 다음 주 수익률을 당겨옴

    return {"RSK": rsk, "REV": rev.reindex(rsk.index), "RVOL": rvol.reindex(rsk.index), "NEXT": nxt.reindex(rsk.index)}


def weekly_long(W: dict, panel: pd.DataFrame, size_col: str) -> pd.DataFrame:
    """
    주×종목 와이드 형식(dict of DataFrames) → 롱 형식(한 행 = 한 주 × 한 종목)으로 바꾸고,
    직전 월말 패널의 유니버스 구성원만 남긴다 (size도 이 시점 값을 사용).

    W: weekly_from_daily()의 반환값 {"RSK": DataFrame, "REV": ..., ...}
    panel: step3에서 만든 월별 패널 (month, ticker, SIZE 등)
    size_col: 가치가중에 쓸 열 이름 (미국="Value", 한국="mktcap" 등)
    """
    # ── 1) dict의 각 DataFrame을 stack하여 (주, 종목)을 인덱스로 만들고 합침
    d = pd.concat({k: v.stack(future_stack=True) for k, v in W.items()}, axis=1)
    d.index.names = ["week", "ticker"]
    d = d.reset_index()

    # ── 2) 각 주가 속하는 "직전 월"을 구함
    #        예: 2020-W03 (1월 셋째 주) → 직전 월말 = 2019-12
    #        이렇게 해야 미래 정보(이번 달 패널)를 쓰지 않음
    weeks = d["week"].unique()
    d["pmonth"] = d["week"].map({w: w.end_time.to_period("M") - 1 for w in weeks})

    # ── 3) 패널에서 직전 월의 유니버스 구성원과 SIZE를 가져와 inner join
    #        유니버스에 없는 종목은 자동으로 제외됨
    u = panel[["month", "ticker", size_col]].rename(columns={"month": "pmonth", size_col: "SIZE_W"})
    return d.merge(u, on=["pmonth", "ticker"], how="inner").dropna(subset=["RSK", "NEXT"])


# ----------------------------------------------------------------- 분위 포트폴리오
def sort_portfolios(d: pd.DataFrame, n_groups: int, cost_oneway: float, sell_tax: float,
                    min_n: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    RSK 기준 분위 포트폴리오를 구성하고, 주별 수익률과 비용을 계산한다.

    d: weekly_long()의 반환값 (행=주×종목, 열에 RSK, NEXT, SIZE_W 등)
    n_groups: 분위 수 (10이면 10분위)
    cost_oneway: 편도 거래비용 비율 (예: 미국 0.0005)
    sell_tax: 매도 시 거래세 비율 (예: 한국 0.0018)
    min_n: 해당 주 종목 수가 이보다 적으면 건너뜀

    반환:
      q : 주 × 분위 (동일가중 EW_q, 가치가중 VW_q 다음 주 수익률)
      ls: 주별 롱숏(최저 − 최고) EW/VW 총수익, 비용 차감 순수익, 롱온리(최저 분위) 순수익
    비용: 각 포트폴리오 비중 변화량(매수+매도) × 편도 비용 + 매도 비중 × 거래세 (공매도 차입 비용은 반영하지 않음)
    """
    rows, prev = [], {}  # prev: 직전 주의 종목별 비중 (회전율 계산용)

    for w, g in d.groupby("week"):
        # ── 종목 수가 너무 적으면 이 주는 건너뜀
        if len(g) < min_n:
            continue

        g = g.copy()
        # ── RSK 순위를 매기고 n_groups개 분위로 나눔 (1 = 최저 왜도, n_groups = 최고 왜도)
        #    method="first"는 동점(tie) 처리: 먼저 나온 순서대로 번호를 매김
        g["q"] = pd.qcut(g["RSK"].rank(method="first"), n_groups, labels=False) + 1
        rec = {"week": w, "n": len(g)}  # 이 주의 기록용 dict

        wts = {}  # 이 주의 분위별 비중 저장 (비용 계산용)

        for q, gg in g.groupby("q"):
            # ── 동일가중 수익률: 분위 내 종목 수익률의 단순 평균
            rec[f"EW_{q}"] = gg["NEXT"].mean()

            # ── 가치가중 수익률: SIZE(시가총액 또는 거래대금)로 가중 평균
            vw = gg["SIZE_W"].clip(lower=0).fillna(0)  # 음수 SIZE 방지
            rec[f"VW_{q}"] = (gg["NEXT"] * vw).sum() / vw.sum() if vw.sum() > 0 else np.nan

            # ── 롱숏에 쓰이는 양 끝 분위(1분위, n분위)만 비중을 저장
            if q in (1, n_groups):
                # 동일가중: 각 종목 비중 = 1/종목수
                wts[("EW", q)] = pd.Series(1.0 / len(gg), index=gg["ticker"])
                # 가치가중: 각 종목 비중 = SIZE/총SIZE
                wts[("VW", q)] = pd.Series((vw / vw.sum()).to_numpy() if vw.sum() > 0 else np.nan, index=gg["ticker"])

        # ── 비용 계산: 직전 주 비중과 비교해서 변화량(회전율) 산출
        for sch in ("EW", "VW"):
            for q in (1, n_groups):
                cur = wts[(sch, q)]                                 # 이번 주 비중
                old = prev.get((sch, q), pd.Series(dtype=float))    # 직전 주 비중 (첫 주는 빈 시리즈)
                idx = cur.index.union(old.index)                    # 이번 주 + 직전 주 종목 합집합
                diff = cur.reindex(idx).fillna(0) - old.reindex(idx).fillna(0)  # 비중 변화량
                buys = diff.clip(lower=0).sum()    # 비중이 늘어난 부분 = 매수량
                sells = (-diff).clip(lower=0).sum() # 비중이 줄어든 부분 = 매도량

                # 거래세 적용 대상:
                #   1분위(롱) → 보유 종목을 파는 것이 매도이므로 sells에 세금
                #   n분위(숏) → 공매도 포지션 진입이 매도이므로 buys에 세금
                taxed = sells if q == 1 else buys
                rec[f"cost_{sch}_{q}"] = cost_oneway * (buys + sells) + sell_tax * taxed
                rec[f"turn_{sch}_{q}"] = buys + sells  # 회전율 기록
                prev[(sch, q)] = cur  # 다음 주 비용 계산을 위해 비중 저장

        rows.append(rec)

    # ── 분위별 수익률 테이블 (q)과 롱숏 요약 테이블 (ls) 생성
    q = pd.DataFrame(rows).set_index("week")
    ls = pd.DataFrame(index=q.index)

    for sch in ("EW", "VW"):
        # 롱숏 총수익 = 1분위(최저 왜도, 롱) − n분위(최고 왜도, 숏)
        ls[f"LS_{sch}_gross"] = q[f"{sch}_1"] - q[f"{sch}_{n_groups}"]
        # 롱숏 순수익 = 총수익 − 양쪽 비용
        ls[f"LS_{sch}_net"] = ls[f"LS_{sch}_gross"] - q[f"cost_{sch}_1"] - q[f"cost_{sch}_{n_groups}"]
        # 롱온리 순수익 = 1분위 수익 − 1분위 비용 (공매도 없이 최저 왜도만 매수)
        ls[f"LONG_{sch}_net"] = q[f"{sch}_1"] - q[f"cost_{sch}_1"]
        # 양쪽 합산 회전율
        ls[f"turn_{sch}"] = q[f"turn_{sch}_1"] + q[f"turn_{sch}_{n_groups}"]

    # 유니버스 동일가중 수익률 (벤치마크): 모든 종목의 단순 평균
    ls["UNIV_EW"] = d.groupby("week")["NEXT"].mean().reindex(ls.index)

    return q, ls


# ----------------------------------------------------------------- 검정 도구
def nw_mean(x: pd.Series, lags: int = 4) -> tuple[float, float, float]:
    """
    시계열 x의 평균이 0과 다른지를 Newey-West 표준오차로 검정한다.

    y = μ + ε 에서 μ̂의 t값과 p값을 구하는 것과 같다.
    HAC(Heteroskedasticity and Autocorrelation Consistent) 공분산으로
    시계열 자기상관을 보정한다 (lags=4: 최대 4주 시차까지 고려).

    반환: (평균, t값, p값)
    """
    x = x.dropna()
    # 상수항만 있는 OLS = 평균 추정. HAC 공분산으로 t값 계산
    res = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params[0]), float(res.tvalues[0]), float(res.pvalues[0])


def summarize(ls: pd.DataFrame, periods: int = 52) -> pd.DataFrame:
    """
    롱숏 요약 테이블(ls)의 각 열에 대해 성과 지표를 계산한다.

    periods: 연환산 계수 (주간 데이터이므로 52)

    산출 지표:
      - mean_weekly_bps: 주간 평균 수익률 (bps, 1 bps = 0.01%)
      - t_NW / p: Newey-West t값과 p값 (평균이 0과 다른지)
      - ann_mean_pct: 연환산 평균 수익률 (%) = 주간 평균 × 52
      - ann_sharpe: 연환산 샤프 비율 = (평균/표준편차) × √52
      - hit_rate: 양(+)의 수익률이 나온 주의 비율
      - avg_weekly_turnover: 주간 평균 회전율 (순수익 열에만 추가)
    """
    out = {}
    for c in [c for c in ls.columns if not c.startswith("turn")]:
        m, t, p = nw_mean(ls[c])
        s = ls[c].dropna()
        out[c] = {"weeks": len(s), "mean_weekly_bps": 1e4 * m, "t_NW": t, "p": p,
                  "ann_mean_pct": 100 * m * periods, "ann_sharpe": m / s.std() * np.sqrt(periods),
                  "hit_rate": (s > 0).mean()}
    # 회전율은 순수익 행에 별도 필드로 추가
    for sch in ("EW", "VW"):
        if f"turn_{sch}" in ls:
            out.setdefault(f"LS_{sch}_net", {})["avg_weekly_turnover"] = ls[f"turn_{sch}"].mean()
    return pd.DataFrame(out).T


def quantile_table(q: pd.DataFrame, n_groups: int) -> pd.DataFrame:
    """
    분위별 평균 주간 수익률(bps)과 t값을 정리한 표를 만든다.
    보고서 표 2.2에 해당: 1분위(최저 왜도)부터 n분위(최고 왜도)까지 수익률이 단조 감소하는지 확인.
    """
    rows = {}
    for sch in ("EW", "VW"):
        for g in range(1, n_groups + 1):
            m, t, p = nw_mean(q[f"{sch}_{g}"])
            rows[(sch, g)] = {"mean_weekly_bps": 1e4 * m, "t_NW": t}
    return pd.DataFrame(rows).T


def to_monthly(x: pd.Series) -> pd.Series:
    """
    주간 수익률 시리즈 → 월간 수익률 시리즈로 변환 (복리 합산).
    팩터 회귀(factor_regression)에서 월간 팩터 데이터와 주기를 맞추기 위해 사용.
    """
    x = x.dropna()
    # 각 주의 마지막 날이 속하는 월로 그룹핑
    m = pd.PeriodIndex([w.end_time.to_period("M") for w in x.index])
    # 복리 합산: (1+r₁)(1+r₂)...(1+rₖ) − 1
    return (1 + x).groupby(m).prod() - 1


def factor_regression(y: pd.Series, F: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    """
    월간 롱숏(또는 롱온리) 수익률을 팩터에 회귀하여 알파를 추정한다.

    y: 월간 포트폴리오 초과수익률 시리즈
    F: 월간 팩터 수익률 DataFrame (MKT, SMB, HML, UMD 등)
    lags: Newey-West 시차

    회귀식: y_t = α + β₁·MKT_t + β₂·SMB_t + ... + ε_t
    α가 유의하게 양(+)이면: 팩터로 설명되지 않는 초과수익이 있다
    α가 유의하지 않거나 음(−)이면: 전략에 독립적인 가치가 없다

    alpha_ann_pct: 월간 알파 × 12 × 100 = 연환산 알파 (%)
    """
    df = pd.concat([y.rename("y"), F], axis=1).dropna()
    res = sm.OLS(df["y"], sm.add_constant(df.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = pd.DataFrame({"coef": res.params, "t_NW": res.tvalues, "p": res.pvalues})
    # 알파 연환산: 월간 상수항 × 1200 (= ×12개월 × ×100%)
    out.loc["alpha_ann_pct", "coef"] = 1200 * res.params["const"]
    out.loc["n_months", "coef"] = len(df)
    return out


def fama_macbeth(d: pd.DataFrame, xs: list, lags: int = 4, min_n: int = 50) -> pd.DataFrame:
    """
    주별 Fama-MacBeth 횡단면 회귀.

    매주 모든 종목에 대해 횡단면 회귀를 돌리고, 주별 계수들의 시계열 평균을 구한다.
    이 평균이 0과 유의하게 다른지를 Newey-West t값으로 검정한다.

    회귀식 (매주 t):
      NEXT_i = α_t + β₁_t · RSK_i + β₂_t · REV_i + β₃_t · RVOL_i + ... + ε_i

    설명변수는 z-점수로 표준화하되, 극단값 영향을 줄이기 위해 1%/99% 분위로 윈저화.
    계수 단위: 1 표준편차 변화 → 다음 주 수익률 bps 변화.

    d: weekly_long()의 반환값
    xs: 설명변수 이름 리스트 (예: ["RSK"], ["RSK","REV","RVOL","SIZE"])
    lags: Newey-West 시차 (4주)
    min_n: 해당 주 종목 수가 이보다 적으면 건너뜀
    """
    coefs = []
    for w, g in d.groupby("week"):
        g = g.dropna(subset=xs + ["NEXT"])
        if len(g) < min_n:
            continue

        # ── 설명변수 표준화: 윈저화(1~99% 분위로 자름) 후 z-점수
        #    극단적 왜도/수익률 종목이 계수를 왜곡하는 것을 방지
        Z = g[xs].apply(lambda s: (s.clip(s.quantile(0.01), s.quantile(0.99)) - s.mean()) / s.std())

        # ── 이 주의 횡단면 OLS → 계수(β) 추출
        b = sm.OLS(g["NEXT"], sm.add_constant(Z)).fit().params
        coefs.append(b.rename(w))

    # ── 주별 계수를 모아 시계열 평균과 NW t값 계산
    B = pd.DataFrame(coefs)
    rows = {}
    for c in B.columns:
        m, t, p = nw_mean(B[c], lags)
        rows[c] = {"coef_bps_per_sd": 1e4 * m, "t_NW": t, "p": p, "weeks": int(B[c].notna().sum())}
    return pd.DataFrame(rows).T