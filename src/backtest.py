"""
백테스트 엔진 (Backtest Engine)

본 모듈은 정량적 자산배분 및 종목 선택 전략의 성과를 검증하고, 현실적인 시장 마찰(수수료, 슬리피지, 거래세, 비중 표류)을
반영하기 위한 핵심 백테스트 함수들을 제공합니다.

[핵심 시점 및 누수 방지 규칙]
1. t일 장 마감 시점(Close)의 정보로 결정된 포트폴리오 비중(w_t)은 t+1일 거래일부터 수익률에 반영됩니다 (delay=1).
2. delay=0으로 설정하면 당일 종가로 당일 수익률을 취하는 미래 정보 누수(Look-ahead bias)가 발생하므로,
   누수 여부를 진단하고 확인하는 비교 검정 목적으로만 사용해야 합니다.
3. 리밸런싱 간 주가 변동에 따른 포트폴리오 비중의 '표류(Drift)'를 정확히 추적하여 실제 필요한 리밸런싱 매매량과
   거래 비용(수수료, 슬리피지, 매도 증권거래세)을 엄밀하게 차감합니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ======================================================================
# 1. 단일 자산(ETF/지수) + 현금 비중 백테스트 (S1 전략용: 일별)
# ======================================================================

def apply_buffer(w: pd.Series, buffer: float) -> pd.Series:
    """
    비중 변화 완충 지대(Buffer Band / Deadband Filter)를 적용하여 불필요한 미세 매매를 방지합니다.

    [작동 원리 및 목적]
    - 일별 자산배분 모델(예: LHAR 변동성 타기팅)에서 매일 목표 비중이 0.1%~0.5%씩 미세하게 바뀔 경우,
      잦은 매매로 인해 누적 거래 비용(슬리피지, 수수료)이 전략 성과를 크게 훼손합니다.
    - 직전 실행된 포트폴리오 비중 대비 신규 목표 비중의 변화 절대값(|w_t - w_{t-1}|)이 buffer 미만이면,
      포지션을 변경하지 않고 직전 비중(w_{t-1})을 그대로 유지합니다.

    Parameters:
    ----------
    w : pd.Series
        일별 목표 주식 비중 시계열 (0.0 ~ 1.0)
    buffer : float
        비중 변경 임계값 (예: 0.05 설정 시 직전 대비 5%p 이상 변화할 때만 리밸런싱 실행)

    Returns:
    -------
    pd.Series
        완충 필터가 적용된 조정 비중 시계열
    """
    # ── 1) 버퍼가 0 이하이면 완충 필터링을 생략하고 원본 비중 그대로 반환
    if buffer <= 0:
        return w

    # ── 2) 시계열 순차 처리를 위해 넘파이 배열 복사본 생성 (연산 속도 최적화)
    #       버퍼 로직은 직전 시점(i-1)의 최종 확정 비중에 의존하므로 단순 벡터화가 아닌 순차 루프 필요
    x = w.to_numpy(float).copy()

    # ── 3) 두 번째 시점(i=1)부터 직전 확정 비중과의 차이를 비교
    for i in range(1, len(x)):
        # 직전 비중(x[i-1])과의 변화 절대값이 buffer 임계값 미만인 경우
        if abs(x[i] - x[i - 1]) < buffer:
            # 매매를 발생시키지 않고 직전 비중을 그대로 유지
            x[i] = x[i - 1]

    # ── 4) 원본 Series의 DatetimeIndex와 이름을 보존하여 Series로 복원 반환
    return pd.Series(x, index=w.index, name=w.name)


def backtest_weights(weight: pd.Series, ret: pd.Series, cost_oneway: float = 0.0,
                     rf: pd.Series | None = None, buffer: float = 0.0, delay: int = 1) -> pd.DataFrame:
    """
    단일 자산(예: KOSPI 200 ETF 또는 S&P 500 ETF)과 무위험 현금 간의 일별 동적 자산배분을 백테스트합니다.

    [포트폴리오 수익률 구조]
    - 총수익률(Gross): w_{t-1} * r_t + (1 - w_{t-1}) * rf_t
    - 회전율(Turnover): |w_t - w_{t-1}| (비중 변경에 따른 매매 체결 비율)
    - 순수익률(Net): Gross - cost_oneway * Turnover

    Parameters:
    ----------
    weight : pd.Series
        t일 장 마감 후 산출된 주식 목표 비중 (0.0 ~ 1.0)
    ret : pd.Series
        위험자산(주식/ETF)의 일별 단순수익률 시계열
    cost_oneway : float, default 0.0
        편도 거래 비용 비율 (수수료 + 슬리피지, 예: 5bps = 0.0005)
    rf : pd.Series | None, default None
        무위험 자산(현금/단기국채)의 일별 수익률 시계열 (미입력 시 0.0으로 간주)
    buffer : float, default 0.0
        비중 변화 완충값 (apply_buffer 함수로 전달)
    delay : int, default 1
        신호 반영 체결 지연일 (1 = t일 장 마감 후 신호 계산 → t+1일 장 시작부터 보유 및 수익 실현)

    Returns:
    -------
    pd.DataFrame
        보유 비중('w'), 회전율('turnover'), 총수익률('gross'), 순수익률('net')로 구성된 DataFrame
    """
    # ── 1) 위험자산 수익률 결측치 제거
    ret = ret.dropna()

    # ── 2) 목표 비중 시계열을 자산 수익률 거래일 기준으로 정렬 및 전방 결측치 채움(ffill)
    #       신호가 없는 날은 직전 비중을 유지하고, 초기 결측치는 0.0(현금 100%)으로 채우며, 롱온리(>=0) 강제
    w = weight.reindex(ret.index).ffill().fillna(0.0).clip(lower=0.0)

    # ── 3) 완충 필터 적용 (미세한 비중 변경으로 인한 잦은 거래 억제)
    w = apply_buffer(w, buffer)

    # ── 4) [핵심] 신호 지연(delay) 적용: t일 종가 결정 비중은 t+delay일부터 실제 포트폴리오에 보유됨
    #       delay=1 설정 시 shift(1)을 통해 미래 정보 누수(Look-ahead bias)를 엄격히 차단
    held = w.shift(delay).fillna(0.0)

    # ── 5) 일별 포트폴리오 비중 변경에 따른 회전율(Turnover) 계산: |w_t - w_{t-1}|
    turnover = held.diff().abs()

    # ── 6) 최초 진입일의 회전율 보정: 초기 현금 상태(0)에서 첫 비중을 매수하는 비용을 반영
    turnover.iloc[0] = held.iloc[0]

    # ── 7) 무위험 자산 수익률 정렬: 미제공 시 0.0 처리, 제공 시 동일 날짜 인덱스로 정렬
    rf_ = rf.reindex(ret.index).ffill().fillna(0.0) if rf is not None else 0.0

    # ── 8) 일별 총수익률(Gross Return) 계산: 주식 수익률과 현금 수익률의 가중합
    gross = held * ret + (1.0 - held) * rf_

    # ── 9) 일별 순수익률(Net Return) 계산: 편도 거래 비용(cost_oneway * turnover) 차감
    net = gross - cost_oneway * turnover

    # ── 10) 최종 결과 DataFrame 반환
    return pd.DataFrame({"w": held, "turnover": turnover, "gross": gross, "net": net})


# ======================================================================
# 2. 다종목 주식 포트폴리오: 월별 백테스트 (S2 종목선택 전략용)
# ======================================================================

def portfolio_monthly(holdings: dict, fwd_ret: pd.DataFrame, cost_oneway: float,
                      sell_tax: float = 0.0, missing_fill: float = 0.0) -> pd.DataFrame:
    """
    월별 다종목 주식 롱 포트폴리오(S2)를 백테스트하며, 실제 가격 변동에 따른 비중 표류(Drift)와
    매도 증권거래세를 엄밀하게 반영합니다.

    [월별 백테스트 메커니즘]
    1. t월 말 시점에 산출된 유니버스 팩터 점수로 다음 달 목표 보유 비중(w)을 결정합니다.
    2. 직전 t-1월에 보유했던 종목들은 한 달 동안 각자 주가 변동을 겪으며 실제 비중이 달라집니다('Drift').
    3. t월 말 리밸런싱 시, 목표 비중(w)과 표류된 기존 비중(drift)의 차이만큼만 실제 매수/매도 주문을 집행합니다.
    4. 매수 금액에는 편도 수수료/슬리피지가 적용되고, 매도 금액에는 편도 수수료 + 매도 거래세(sell_tax)가 적용됩니다.
    5. 다음 달(t+1월) 동안 포트폴리오를 보유하여 월간 수익률을 실현합니다.

    Parameters:
    ----------
    holdings : dict
        {Period('YYYY-MM'): pd.Series(ticker -> 비중)} 구조의 월별 목표 보유 비중 딕셔너리
    fwd_ret : pd.DataFrame
        행: 결정 월 Period(t), 열: 종목코드(ticker), 값: 다음 달(t+1월)의 실현 단순수익률
    cost_oneway : float
        편도 기본 거래 비용 (수수료 + 호가 슬리피지 비율, 예: 5bps = 0.0005)
    sell_tax : float, default 0.0
        매도 시에만 부과되는 증권거래세 비율 (예: 한국 주식 0.15%~0.20% = 0.0015~0.0020)
    missing_fill : float, default 0.0
        다음 달 수익률이 결측된 종목(거래정지, 상장폐지 등)에 부여할 대체 수익률 (상폐 손실 가정 등)

    Returns:
    -------
    pd.DataFrame
        결정 월(month)을 인덱스로 하며 총수익률(gross), 거래비용(cost), 순수익률(net),
        회전율(turnover), 보유종목수(n_holdings), 결측수익률수(n_missing_ret)를 포함하는 DataFrame
    """
    rows = []
    # drift: 직전 달 말 시점의 각 종목 실제 평가 비중 (초기에는 보유 종목 없음)
    drift = pd.Series(dtype=float)

    # ── 1) 결정 월(t)을 시간 순서대로 순회
    for m in sorted(holdings):
        w = holdings[m]

        # ── 2) 신규 목표 종목군과 기존 보유 표류 종목군의 합집합 인덱스 생성
        #       (새로 매수할 종목 + 계속 보유할 종목 + 전량 매도/청산할 종목 모두 포함)
        idx = w.index.union(drift.index)

        # ── 3) 종목별 매매 필요 비중 산출: 목표 비중(w) - 현재 표류 평가 비중(drift)
        trade = w.reindex(idx).fillna(0.0) - drift.reindex(idx).fillna(0.0)

        # ── 4) 매수 총액 및 매도 총액 분리 집계
        buys = trade.clip(lower=0).sum()        # 새로 사야 하는 비중의 합
        sells = (-trade).clip(lower=0).sum()     # 팔아서 줄여야 하는 비중의 합

        # ── 5) 거래 비용 산출: (매수 + 매도) * 편도비용 + 매도 * 매도세금
        cost = cost_oneway * (buys + sells) + sell_tax * sells

        # ── 6) 다음 달(t+1월) 실현 수익률 벡터 추출
        r = fwd_ret.loc[m].reindex(w.index) if m in fwd_ret.index else pd.Series(np.nan, index=w.index)

        # ── 7) 미래 시점이라 아직 실현 수익률 데이터가 전혀 없는 경우 루프 종료/건너뜀
        if r.notna().sum() == 0:
            continue

        # ── 8) 결측 종목 수 집계 및 결측치 대체 (상장폐지 손실 등 설정값 반영)
        n_missing = int(r.isna().sum())
        r = r.fillna(missing_fill)

        # ── 9) 포트폴리오 월간 총수익률(Gross) 계산: 목표 비중과 종목별 실현수익률의 가중합
        gross = float((w * r).sum())

        # ── 10) 한 달간 주가 변동 후 월말 평가 가치 산출: w_i * (1 + r_i)
        val = w * (1.0 + r)

        # ── 11) 다음 달 리밸런싱의 기준이 될 말일 평가 비중(drift) 갱신 (정규화)
        drift = val / val.sum() if val.sum() > 0 else val

        # ── 12) 해당 월의 백테스트 결과 행 저장
        rows.append({
            "month": m,
            "gross": gross,
            "cost": cost,
            "net": gross - cost,
            "turnover": buys + sells,
            "n_holdings": len(w),
            "n_missing_ret": n_missing
        })

    # ── 13) 결과를 월(month)을 인덱스로 하는 DataFrame으로 변환 반환
    return pd.DataFrame(rows).set_index("month")


def sleeve_daily(holdings: dict, daily_ret: pd.DataFrame) -> pd.DataFrame:
    """
    월말에 결정된 종목 포트폴리오를 다음 달 매 거래일 동안 실제로 보유했을 때의 일별 수익률 시계열을 생성합니다.
    (S3 전략: S2의 개별 종목 슬리브 일별 수익률에 S1의 시장 노출 비중을 곱하기 위한 필수 함수)

    [일별 추적 및 리밸런싱 규칙]
    1. 월의 첫 거래일(k=0)에만 월간 정기 리밸런싱에 따른 매수/매도 거래가 발생하고 회전율이 기록됩니다.
    2. 월중(k>0)에는 추가 매매 없이 주가 등락에 따라 포트폴리오 내 종목 비중이 매일 자연스럽게 표류(drift)합니다.
    3. 일별 포트폴리오 비중 업데이트 공식: w_{t, i} = w_{t-1, i} * (1 + r_{t, i}) / (1 + p_{port, t})

    Parameters:
    ----------
    holdings : dict
        {Period('YYYY-MM'): pd.Series(ticker -> 비중)} 형태의 월별 목표 비중
    daily_ret : pd.DataFrame
        행: 거래일(date), 열: 종목코드(ticker), 값: 일별 단순수익률

    Returns:
    -------
    pd.DataFrame
        날짜(date)를 인덱스로 하며 일별 총수익률('gross'), 첫날 회전율('turnover'), 첫날 매도비중('sells')을 포함하는 DataFrame
    """
    out = []
    # drift: 직전 월 말 최종 평가 비중 (새로운 달 첫날 리밸런싱 거래량 계산에 사용)
    drift = pd.Series(dtype=float)

    # ── 1) 결정 월(m)을 순차적으로 순회
    for m in sorted(holdings):
        nxt = m + 1  # 실제 포트폴리오를 보유할 다음 달 Period

        # ── 2) 다음 달(nxt)에 해당하는 일별 거래일 목록 추출
        days = daily_ret.index[daily_ret.index.to_period("M") == nxt]
        if len(days) == 0:
            continue

        # ── 3) 이번 달 목표 비중(w0)과 전월 말 표류 비중(drift) 비교를 통한 첫날 리밸런싱 매매량 산출
        w0 = holdings[m]
        idx = w0.index.union(drift.index)
        trade = w0.reindex(idx).fillna(0.0) - drift.reindex(idx).fillna(0.0)
        buys, sells = trade.clip(lower=0).sum(), (-trade).clip(lower=0).sum()

        # ── 4) 이번 달 거래일들에 대한 보유 종목 수익률 행렬 추출
        R = daily_ret.loc[days, :].reindex(columns=w0.index).fillna(0.0)

        # ── 5) 첫날 목표 비중으로 시작하여 일별로 보유하며 주가 변동 반영
        w = w0.copy()
        for k, d in enumerate(days):
            r = R.loc[d]
            # 당일 포트폴리오 수익률: 각 종목 당일 비중과 수익률의 가중합
            pr = float((w * r).sum())

            # 월 첫 거래일(k=0)에만 월간 리밸런싱 회전율 및 매도량을 기록 (나머지 날은 0.0)
            out.append({
                "date": d,
                "gross": pr,
                "turnover": buys + sells if k == 0 else 0.0,
                "sells": sells if k == 0 else 0.0
            })

            # ── 6) 당일 장 마감 후 종목별 비중 표류(Daily Drift) 업데이트:
            #       수익률이 높았던 종목은 비중이 증가하고, 낮았던 종목은 비중이 감소함
            w = w * (1.0 + r) / (1.0 + pr) if (1.0 + pr) != 0 else w

        # ── 7) 이번 달 마지막 날의 최종 표류 비중을 다음 달 리밸런싱 비교용 drift로 보관
        drift = w

    # ── 8) 일별 결과를 날짜 인덱스 DataFrame으로 변환 반환
    return pd.DataFrame(out).set_index("date")


# ======================================================================
# 3. 정량적 성과 지표 산출 (Performance Metrics)
# ======================================================================

def perf_stats(r: pd.Series, periods: int = 252, rf: pd.Series | None = None) -> pd.Series:
    """
    수익률 시계열에 대한 핵심 정량적 성과 및 위험 지표들을 계산합니다.

    [계산 지표 목록]
    - years      : 분석 대상 기간 (연수)
    - ann_ret    : 기하 연평균 복리 수익률 (CAGR)
    - ann_vol    : 연환산 표준편차 (변동성)
    - sharpe     : 연환산 샤프 지수 (초과수익률 / 변동성)
    - sharpe_se  : Andrew Lo (2002) 공식 기반 샤프 지수 표준오차
    - sortino    : 연환산 소르티노 지수 (초과수익률 / 하방 변동성)
    - max_dd     : 최대 낙폭 (Maximum Drawdown, MDD)
    - calmar     : 칼마 지수 (CAGR / |MDD|)
    - skew       : 수익률 왜도 (비대칭성)

    Parameters:
    ----------
    r : pd.Series
        전략의 단순수익률 시계열 (일별, 주별, 또는 월별)
    periods : int, default 252
        연간 주기 수 (일별=252, 주별=52, 월별=12)
    rf : pd.Series | None, default None
        무위험 이자율 시계열 (미입력 시 0.0으로 간주)

    Returns:
    -------
    pd.Series
        주요 성과 지표가 정리된 Series
    """
    # ── 1) 결측치 제거
    r = r.dropna()

    # ── 2) 표본 수가 5개 미만으로 너무 적으면 지표 계산 불가하므로 빈 Series 반환
    if len(r) < 5:
        return pd.Series(dtype=float)

    # ── 3) 무위험 이자율을 차감한 초과수익률(Excess Return) 계산
    ex = r - (rf.reindex(r.index).ffill().fillna(0.0) if rf is not None else 0.0)

    # ── 4) 데이터의 총 기간(연 단위) 계산
    yrs = len(r) / periods

    # ── 5) 누적 자산 가치 곡선(NAV / Cumulative Wealth Path, 시작점=1.0) 계산
    wealth = (1.0 + r).cumprod()

    # ── 6) 기하 연평균 복리 수익률 (CAGR = (최종가치 / 시작가치)^(1/연수) - 1)
    ann_ret = wealth.iloc[-1] ** (1.0 / yrs) - 1.0

    # ── 7) 초과수익률의 표본 표준편차
    sd = ex.std()

    # ── 8) 연환산 샤프 지수 (Sharpe Ratio = 평균 초과수익 / 표준편차 * sqrt(periods))
    sr = ex.mean() / sd * np.sqrt(periods) if sd > 0 else np.nan

    # ── 9) Andrew Lo (2002, Financial Analysts Journal) 논문 기반 샤프 지수 점근 표준오차:
    #       SE(SR) = sqrt((1 + 0.5 * SR^2) / T)
    sharpe_se = np.sqrt((1.0 + 0.5 * sr ** 2) / yrs) if np.isfinite(sr) else np.nan

    # ── 10) 하방 위험(Downside Deviation / 하방 반변동성): 0 미만의 음수 초과수익률 제곱평균의 제곱근
    down = np.sqrt((np.minimum(ex, 0.0) ** 2).mean()) * np.sqrt(periods)

    # ── 11) 연환산 소르티노 지수 (Sortino Ratio): 상승 변동성은 무시하고 하락 리스크 대비 초과수익만 평가
    sortino = ex.mean() * periods / down if down > 0 else np.nan

    # ── 12) 역사적 고점 대비 최대 낙폭 (Maximum Drawdown = min(NAV / 최고NAV - 1))
    mdd = (wealth / wealth.cummax() - 1.0).min()

    # ── 13) 칼마 지수 (Calmar Ratio = CAGR / |MDD|): 극단적 하락 위험 단위당 복리 수익률
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan

    # ── 14) 수익률 분포 왜도 (Skewness): 양수일수록 우측 꼬리(급등), 음수일수록 좌측 꼬리(급락/폭락) 위험 큼
    skew = r.skew()

    # ── 15) 결과를 딕셔너리 기반 Series로 묶어 반환
    return pd.Series({
        "years": yrs,
        "ann_ret": ann_ret,
        "ann_vol": r.std() * np.sqrt(periods),
        "sharpe": sr,
        "sharpe_se": sharpe_se,
        "sortino": sortino,
        "max_dd": mdd,
        "calmar": calmar,
        "skew": skew,
    })


def perf_table(series: dict, periods: int = 252, rf: pd.Series | None = None,
               turnover: dict | None = None) -> pd.DataFrame:
    """
    여러 전략의 수익률 시계열들을 취합하여 비교 성과 요약표(DataFrame)를 생성합니다.

    Parameters:
    ----------
    series : dict
        {전략명: pd.Series(수익률)} 형태의 딕셔너리
    periods : int, default 252
        연간 주기 수 (일별 252, 주별 52, 월별 12)
    rf : pd.Series | None, default None
        무위험 이자율
    turnover : dict | None, default None
        {전략명: pd.Series(회전율)} 형태의 회전율 딕셔너리 (제공 시 연환산 회전율 추가)

    Returns:
    -------
    pd.DataFrame
        각 전략(행)별 성과 및 위험 지표가 정리된 표
    """
    # 각 전략별로 perf_stats를 호출하여 전치(T)한 DataFrame 생성
    tab = pd.DataFrame({k: perf_stats(v, periods, rf) for k, v in series.items()}).T

    # 회전율 딕셔너리가 전달된 경우, 평균 회전율 * periods 로 연환산 회전율 열 추가
    if turnover:
        tab["ann_turnover"] = pd.Series({k: v.mean() * periods for k, v in turnover.items()})

    return tab


def perf_by_regime(r: pd.Series, regime: pd.Series, periods: int = 252) -> pd.DataFrame:
    """
    시장 변동성 레짐(예: 0=평온 레짐, 1=혼란/고변동성 레짐)별로 전략의 조건부 성과를 분리 평가합니다.

    [핵심 누수 방지 원칙]
    - 당일(t)의 수익률(r_t)에 매칭되는 레짐은 반드시 '전일 장 마감 시점(t-1)에 판정된 레짐'이어야 합니다.
    - regime.shift(1)을 통해 t-1일에 관측된 시장 상태에 따라 t일 포지션을 잡았다는 인과관계를 엄격히 유지합니다.

    Parameters:
    ----------
    r : pd.Series
        전략 수익률 시계열
    regime : pd.Series
        시장 레짐 시계열 (0 또는 1 등 정수 라벨)
    periods : int, default 252
        연간 주기 수

    Returns:
    -------
    pd.DataFrame
        각 레짐(regime_0, regime_1 등)별 성과 지표 비교 테이블
    """
    # 수익률(r)과 1시점 지연된 레짐(s = regime.shift(1))을 날짜 인덱스로 결합하고 결측치 제거
    d = pd.concat([r.rename("r"), regime.shift(1).reindex(r.index).rename("s")], axis=1).dropna()

    # 각 레짐 상태(k)별로 그룹화하여 각각의 성과 지표(perf_stats)를 산출한 뒤 표로 결합
    return pd.DataFrame({f"regime_{int(k)}": perf_stats(g["r"], periods)
                         for k, g in d.groupby("s")}).T


def breakeven_cost(gross: pd.Series, turnover: pd.Series) -> float:
    """
    전략의 순수익(Net Profit)이 정확히 0이 되는 '손익분기 편도 거래 비용(Breakeven One-Way Cost)'을 계산합니다.

    [수학적 원리]
    - 순수익 = 총수익 - (비용 * 회전율) = 0
    - 편도 손익분기 비용 = E[Gross] / E[Turnover]
    - 이 값이 실제 발생하는 매매 비용(예: 5bps~10bps)보다 충분히 커야 실제 시장 마찰 속에서도 유의미한 이익을 남깁니다.

    Parameters:
    ----------
    gross : pd.Series
        전략 총수익률 시계열
    turnover : pd.Series
        전략 회전율 시계열

    Returns:
    -------
    float
        손익분기 편도 비용 (예: 0.0035 = 35 bps)
    """
    t = turnover.mean()
    return float(gross.mean() / t) if t > 0 else np.nan


def factor_alpha(r: pd.Series, factors: pd.DataFrame, lags: int = 3, periods: int = 12) -> pd.DataFrame:
    """
    전략 초과수익률을 기저 위험 팩터(Fama-French 팩터 등)에 다변량 시계열 회귀하여 '팩터 조정 알파(Alpha)'를 검정합니다.

    [회귀 모형식]
      R_{p, t} - R_{f, t} = alpha + beta_1 * F_{1, t} + beta_2 * F_{2, t} + ... + epsilon_t

    [경제학적 의미]
    - 전략이 높은 수익률을 기록하더라도, 시장(Mkt-RF), 규모(SMB), 가치(HML), 단기반전(REV) 등의 기존 위험 프리미엄을
      단순히 레버리지하여 떠안은 결과라면 통계적으로 유의한 알파(alpha)가 0이 됩니다.
    - Newey-West HAC(이분산 및 자기상관 강건) 공분산 추정량을 적용하여 시계열 자기상관에 왜곡되지 않은 t-통계량을 산출합니다.

    Parameters:
    ----------
    r : pd.Series
        전략 수익률 시계열 (일반적으로 월별 초과수익률)
    factors : pd.DataFrame
        통제할 팩터 수익률 행렬 (예: MKT, SMB, HML, UMD, REV 등)
    lags : int, default 3
        Newey-West HAC 표준오차 계산 시 반영할 최대 시차(lags)
    periods : int, default 12
        연환산 승수 (월별 데이터면 12, 일별이면 252)

    Returns:
    -------
    pd.DataFrame
        각 팩터별 추정 계수(coef), Newey-West t-통계량(t_NW), 연환산 알파(alpha_annualized), 표본 수(n_obs)
    """
    import statsmodels.api as sm

    # ── 1) 종속변수(y = 전략 수익률)와 독립변수(factors = 팩터 행렬)를 병합하고 결측치 제거
    df = pd.concat([r.rename("y"), factors], axis=1).dropna()

    # ── 2) 상수항(alpha)을 추가하여 OLS 회귀모형 적합 (Newey-West HAC 강건 공분산 지정)
    res = sm.OLS(df["y"], sm.add_constant(df.drop(columns="y"))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )

    # ── 3) 기본 계수(coef)와 HAC t-값(t_NW) 표 생성
    out = pd.DataFrame({"coef": res.params, "t_NW": res.tvalues})

    # ── 4) 상수항(const)을 연환산(periods 배)하여 연간 팩터 알파(alpha_annualized) 기록
    out.loc["alpha_annualized", "coef"] = res.params["const"] * periods

    # ── 5) 회귀분석에 사용된 유효 관측치(개월 수) 기록
    out.loc["n_obs", "coef"] = len(df)

    return out
