"""
변동성 측정.

단위 규칙: 일별 변동성(분산)은 '퍼센트 제곱(%²)'으로 통일한다.
  - 레인지 추정치: 소수 기준 값 × 1e4
  - 장중 RV: 수익률을 % 단위(×100)로 바꿔서 제곱합
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_4LN2 = 4.0 * np.log(2.0)


# ----------------------------------------------------------------- 일봉 레인지 추정치
def parkinson(df: pd.DataFrame, scale: float = 1e4) -> pd.Series:
    hl = np.log(df["High"] / df["Low"])
    out = (hl ** 2) / _4LN2 * scale
    if "flag_bad_hl" in df.columns:
        out[df["flag_bad_hl"]] = np.nan
    return out.rename("RV_pk")


def garman_klass(df: pd.DataFrame, scale: float = 1e4) -> pd.Series:
    hl = np.log(df["High"] / df["Low"])
    co = np.log(df["Close"] / df["Open"])
    out = (0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2) * scale
    if "flag_bad_hl" in df.columns:
        out[df["flag_bad_hl"]] = np.nan
    return out.clip(lower=0).rename("RV_gk")


# ----------------------------------------------------------------- realized 지표 (한 묶음의 수익률)
_KEYS = ["N", "RV", "RS_pos", "RS_neg", "dJ", "RSJ", "RSkew", "RQ", "BV", "JV", "MAX", "SUM"]


def measures(r: np.ndarray) -> dict:
    """
    r: 한 기간(하루, 한 주, 한 달) 안의 잘게 쪼갠 수익률들
    반환: RV, 상승/하락 반분산, signed jump, RSJ, 실현왜도, quarticity, bipower 등
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2:
        return {k: np.nan for k in _KEYS} | {"N": n}
    rs_pos = float(np.sum(r[r > 0] ** 2))
    rs_neg = float(np.sum(r[r < 0] ** 2))
    rv = rs_pos + rs_neg
    bv = float(np.pi / 2 * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
    return {
        "N": n,
        "RV": rv,
        "RS_pos": rs_pos,
        "RS_neg": rs_neg,
        "dJ": rs_pos - rs_neg,
        "RSJ": (rs_pos - rs_neg) / rv if rv > 0 else np.nan,
        "RSkew": float(np.sqrt(n) * np.sum(r ** 3) / rv ** 1.5) if rv > 0 else np.nan,
        "RQ": float(n / 3.0 * np.sum(r ** 4)),
        "BV": bv,
        "JV": max(rv - bv, 0.0),
        "MAX": float(r.max()),
        "SUM": float(r.sum()),
    }


def realized_by_group(r: pd.Series, key) -> pd.DataFrame:
    """r을 key(날짜, 주, 월)로 묶어 measures 계산"""
    r = r.dropna()
    key = pd.Series(key, index=r.index) if not isinstance(key, pd.Series) else key.reindex(r.index)
    res = {k: measures(v.values) for k, v in r.groupby(key)}
    return pd.DataFrame(res).T.reindex(columns=_KEYS)


# ----------------------------------------------------------------- 장중 봉 → 일별 / 주별 지표
def intraday_returns(bars: pd.DataFrame, tz: str, skip_first: int = 0) -> pd.DataFrame:
    """
    bars: 한 종목의 장중 봉 (index: UTC 시각, 'Close' 컬럼)
    반환: DataFrame[r_pct, date, week] — 같은 날 안에서의 봉 간 로그수익률(%)
    오버나이트(전날 마지막 봉 → 오늘 첫 봉)는 제외한다.
    skip_first: 장 시작 직후 수익률 몇 개를 추가로 버릴지 (동시호가 영향)
    """
    close = bars["Close"].astype(float).dropna()
    local = close.index.tz_convert(tz)
    close.index = local
    date = pd.Series(local.tz_localize(None).normalize(), index=local)
    logc = np.log(close)
    r = logc.groupby(date.values).diff()
    df = pd.DataFrame({"r_pct": r * 100.0, "date": date.values}, index=local)
    df["k"] = df.groupby("date").cumcount()
    df = df[df["k"] > skip_first].dropna(subset=["r_pct"])
    df["week"] = pd.DatetimeIndex(df["date"]).to_period("W")
    return df.drop(columns="k")


def overnight_returns(bars: pd.DataFrame, tz: str) -> pd.Series:
    close = bars["Close"].astype(float).dropna()
    close.index = close.index.tz_convert(tz)
    date = close.index.tz_localize(None).normalize()
    first = close.groupby(date).first()
    last = close.groupby(date).last()
    return (np.log(first / last.shift(1)) * 100.0).rename("overnight_pct")


def intraday_daily_measures(bars: pd.DataFrame, tz: str, skip_first: int = 0) -> pd.DataFrame:
    ir = intraday_returns(bars, tz, skip_first)
    out = realized_by_group(ir["r_pct"], ir["date"])
    out.index = pd.DatetimeIndex(out.index)
    return out.join(overnight_returns(bars, tz), how="left")


def intraday_weekly_measures(bars: pd.DataFrame, tz: str, skip_first: int = 0) -> pd.DataFrame:
    ir = intraday_returns(bars, tz, skip_first)
    return realized_by_group(ir["r_pct"], ir["week"])


# ----------------------------------------------------------------- 일별 수익률 → 월별 지표 (A층)
def period_measures(r_daily: pd.Series, freq: str = "M") -> pd.DataFrame:
    """월 안의 일별 로그수익률(소수)로 월간 RS±, RSJ, RSkew, MAX, REV(=SUM)"""
    r = r_daily.dropna()
    return realized_by_group(r, r.index.to_period(freq))
