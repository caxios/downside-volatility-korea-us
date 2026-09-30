"""
다중검정 보정.

- Holm (1979): 가족별 오류율(FWER) 통제. 본페로니보다 항상 덜 보수적이면서 같은 보장을 준다.
- Benjamini–Hochberg (1995): 거짓 발견 비율(FDR) 통제.
- Harvey, Liu & Zhu (2016) 기준: |t| > 3.0 (새 팩터 주장의 관례적 문턱).

p값이 없고 t값만 있는 결과(NW t)는 양측 정규 근사 p = 2·(1 − Φ(|t|))로 바꾼다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

HLZ_T = 3.0


def p_from_t(t) -> np.ndarray:
    return 2 * stats.norm.sf(np.abs(np.asarray(t, float)))


def t_from_p(p) -> np.ndarray:
    """양측 p에 대응하는 |t| (정규 근사). F·χ² 검정처럼 t가 없는 결과의 HLZ 비교용"""
    return stats.norm.isf(np.asarray(p, float) / 2)


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    m = len(q)
    if m == 0:
        return out
    order = np.argsort(q)
    adj = np.maximum.accumulate((m - np.arange(m)) * q[order])
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def bh(p) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    m = len(q)
    if m == 0:
        return out
    order = np.argsort(q)
    ranked = q[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def adjust(df: pd.DataFrame, p_col: str = "p", t_col: str | None = None, family_col: str | None = None,
           alpha: float = 0.05) -> pd.DataFrame:
    """
    df에 p_holm, p_bh, sig_raw, sig_holm, sig_bh, pass_HLZ(|t|>3) 열을 붙인다.
    family_col이 있으면 가족별로 따로 보정한다.
    """
    d = df.copy()
    if p_col not in d and t_col is not None:
        d[p_col] = p_from_t(d[t_col])
    tt = np.abs(d[t_col].astype(float)) if t_col is not None and t_col in d else pd.Series(
        t_from_p(d[p_col]), index=d.index)
    d["abs_t_equiv"] = tt
    groups = [(None, d.index)] if family_col is None else list(d.groupby(family_col).groups.items())
    d["m_family"] = np.nan
    d["p_holm"] = np.nan
    d["p_bh"] = np.nan
    for _, idx in groups:
        d.loc[idx, "m_family"] = int(np.isfinite(d.loc[idx, p_col].astype(float)).sum())
        d.loc[idx, "p_holm"] = holm(d.loc[idx, p_col])
        d.loc[idx, "p_bh"] = bh(d.loc[idx, p_col])
    d["sig_raw"] = d[p_col] < alpha
    d["sig_holm"] = d["p_holm"] < alpha
    d["sig_bh"] = d["p_bh"] < alpha
    d["pass_HLZ_t3"] = d["abs_t_equiv"] > HLZ_T
    return d


def summarize(d: pd.DataFrame, family_col: str) -> pd.DataFrame:
    g = d.groupby(family_col)
    return pd.DataFrame({
        "m": g.size(),
        "raw_p<0.05": g["sig_raw"].sum(),
        "Holm<0.05": g["sig_holm"].sum(),
        "BH<0.05": g["sig_bh"].sum(),
        "|t|>3": g["pass_HLZ_t3"].sum(),
    })
