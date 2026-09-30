"""일봉 품질 점검과 정제."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _adj(df: pd.DataFrame) -> pd.Series:
    return df["Adj Close"] if "Adj Close" in df.columns else df["Close"]


def quality_report(df: pd.DataFrame, name: str) -> dict:
    r = np.log(_adj(df)).diff()
    ohlc = [c for c in ("Open", "High", "Low", "Close") if c in df.columns]
    gaps = df.index.to_series().diff().dt.days
    return {
        "name": name,
        "start": df.index.min().date() if len(df) else None,
        "end": df.index.max().date() if len(df) else None,
        "n_days": len(df),
        "missing_OHLC": int(df[ohlc].isna().any(axis=1).sum()),
        "high_lt_low": int((df["High"] < df["Low"]).sum()),
        "close_outside_HL": int(((df["Close"] > df["High"] * 1.0001) |
                                 (df["Close"] < df["Low"] * 0.9999)).sum()),
        "zero_range_days": int((df["High"] == df["Low"]).sum()),
        "zero_volume_days": int((df.get("Volume", pd.Series(1, index=df.index)) == 0).sum()),
        "abs_ret_gt_10pct": int((r.abs() > 0.10).sum()),
        "abs_ret_gt_30pct": int((r.abs() > 0.30).sum()),
        "max_gap_days": int(gaps.max()) if len(gaps.dropna()) else 0,
    }


def clean_daily(df: pd.DataFrame, kr: bool = False) -> pd.DataFrame:
    """
    반환 컬럼: Open High Low Close Adj Close Volume + r(로그수익률, 소수) + 플래그
    - 중복 날짜 제거, 종가 없는 날 제거
    - High<Low, 종가가 고저 범위 밖 → 레인지 변동성 계산에서 제외(flag_bad_hl)
    - 한국: |로그수익률| > 0.31 은 가격제한폭(±30%) 위반 → 데이터 오류로 보고 수익률 NaN
      미국: |로그수익률| > 0.5 는 액면분할 조정 오류 의심 → NaN
    """
    out = df.copy().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(subset=["Close"])
    out = out[out["Close"] > 0]
    adj = _adj(out)
    r = np.log(adj).diff()
    thr = 0.31 if kr else 0.5
    bad_ret = r.abs() > thr
    r[bad_ret] = np.nan
    out["r"] = r
    out["flag_bad_ret"] = bad_ret
    out["flag_bad_hl"] = ((out["High"] < out["Low"]) | (out["Low"] <= 0) |
                          (out["Close"] > out["High"] * 1.0001) | (out["Close"] < out["Low"] * 0.9999))
    out["flag_zero_vol"] = out.get("Volume", pd.Series(1, index=out.index)) == 0
    return out
