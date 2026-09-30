"""
S4: 예측 상승/하락 반분산 기반 종목 선택 (롱온리, 월 1회 리밸런싱)

아이디어 (사용자 제안 + 상대 비율 보완)
  - 앞으로 상승 변동성이 클 것 같은 종목을 담는다.
  - 담은 종목의 앞으로의 하락 변동성이 (상승 변동성에 비해) 커질 것 같으면 판다.
  - 상승·하락 변동성 예측은 서로 강하게 같이 움직이므로(과거 하락 반분산이 둘 다를 예측),
    절대 크기가 아니라 상대 비율 R = 예측 RS⁺ / 예측 RS⁻ 로 매도 여부를 판단한다.

1) 예측 (종목별, 매월 말 t)
   설명변수: 그 종목의 RS⁺, RS⁻ 의 일(t일)·주(최근 5일 평균)·월(최근 22일 평균)   [일별 로그수익률 %]
   목표    : 다음 달(t+1월) 일평균 RS⁺, RS⁻
   모형    : 유니버스 종목을 묶은 합동 OLS. t월 말 결정에는 목표가 이미 실현된 (s ≤ t−1) 표본만 사용,
             매월 확장 창으로 다시 추정. 학습 표본은 99.5% 분위로 절단. 최소 24개월 학습.
   예측값은 0 이하가 되지 않도록 학습 목표의 1% 분위로 하한.

2) 매매 규칙 (t월 말 결정, t+1월 보유, 동일가중, 종목당 최대 5%)
   U = 예측 RS⁺ (상승 변동성 크기),  R = 예측 RS⁺ / 예측 RS⁻ (상대 비율)
   매도(보유 종목): 유니버스에서 빠졌거나 매도 조건에 해당
       exit="rel": R 이 유니버스 중앙값 미만
       exit="abs": R < 1  (예측 하락 반분산 > 예측 상승 반분산)
   매수(빈자리): U 가 유니버스 상위 절반이고 매도 조건에 해당하지 않는 종목 중 R 이 높은 순
   보유 목표 수: entry_pct × 유니버스 종목 수
"""
from __future__ import annotations

import numpy as np
import pandas as pd

X_COLS = ["RSp_d", "RSn_d", "RSp_w", "RSn_w", "RSp_m", "RSn_m"]
MIN_TRAIN_MONTHS = 24


def month_end_features(daily_ret: pd.DataFrame) -> pd.DataFrame:
    """
    daily_ret: 일별 단순수익률 (행 날짜, 열 종목) → 월말 (month, ticker) 설명변수 + 다음 달 목표
    """
    r = np.log1p(daily_ret.astype(float)) * 100.0
    rsp = r.where(r.isna(), np.where(r > 0, r ** 2, 0.0))
    rsn = r.where(r.isna(), np.where(r < 0, r ** 2, 0.0))
    month = r.index.to_period("M")
    last_day = pd.Series(r.index, index=r.index).groupby(month).max()

    def at_month_end(df):
        out = df.loc[last_day.values]
        out.index = last_day.index
        return out.stack().rename_axis(["month", "ticker"])

    feats = {
        "RSp_d": at_month_end(rsp), "RSn_d": at_month_end(rsn),
        "RSp_w": at_month_end(rsp.rolling(5, min_periods=4).mean()),
        "RSn_w": at_month_end(rsn.rolling(5, min_periods=4).mean()),
        "RSp_m": at_month_end(rsp.rolling(22, min_periods=18).mean()),
        "RSn_m": at_month_end(rsn.rolling(22, min_periods=18).mean()),
    }
    # 목표: 다음 달 일평균 (그 달 거래일이 15일 이상일 때만)
    cnt = r.notna().groupby(month).sum()
    for s, df in (("RSp", rsp), ("RSn", rsn)):
        mm = df.groupby(month).mean().where(cnt >= 15)
        feats[f"y_{s}"] = mm.shift(-1).stack().rename_axis(["month", "ticker"])
    return pd.DataFrame(feats).reset_index()


def rolling_forecasts(feat: pd.DataFrame, universe: pd.DataFrame, months) -> pd.DataFrame:
    """
    universe: 패널의 (month, ticker) — 그 달 투자 가능 종목
    months  : 예측을 만들 결정 월들
    반환: (month, ticker, pRSp, pRSn)
    """
    f = feat.merge(universe[["month", "ticker"]], on=["month", "ticker"], how="inner")
    out = []
    for t in sorted(months):
        train = f[(f["month"] <= t - 1)].dropna(subset=X_COLS + ["y_RSp", "y_RSn"])
        if train["month"].nunique() < MIN_TRAIN_MONTHS:
            continue
        now = f[(f["month"] == t)].dropna(subset=X_COLS)
        if now.empty:
            continue
        caps = train[X_COLS + ["y_RSp", "y_RSn"]].quantile(0.995)
        tr = train[X_COLS + ["y_RSp", "y_RSn"]].clip(upper=caps, axis=1)
        X = np.column_stack([np.ones(len(tr)), tr[X_COLS].to_numpy()])
        Xn = np.column_stack([np.ones(len(now)), now[X_COLS].clip(upper=caps[X_COLS], axis=1).to_numpy()])
        pred = {}
        for s in ("RSp", "RSn"):
            beta, *_ = np.linalg.lstsq(X, tr[f"y_{s}"].to_numpy(), rcond=None)
            floor = tr[f"y_{s}"].quantile(0.01)
            pred[s] = np.maximum(Xn @ beta, max(floor, 1e-6))
        out.append(pd.DataFrame({"month": t, "ticker": now["ticker"].to_numpy(),
                                 "pRSp": pred["RSp"], "pRSn": pred["RSn"]}))
    return pd.concat(out, ignore_index=True)


def _sell_flag(R: pd.Series, exit_rule: str) -> pd.Series:
    if exit_rule == "rel":
        return R < R.median()
    if exit_rule == "abs":
        return R < 1.0
    raise ValueError(exit_rule)


def s4_select(g: pd.DataFrame, entry_pct: float, exit_rule: str, prev: set, max_weight: float = 0.05,
              mode: str = "s4") -> pd.Series:
    """
    g: 한 달의 (ticker, pRSp, pRSn)
    mode="s4"     : 제안 전략 (U 상위 절반 & 매도 조건 아님 → R 순으로 매수, 보유 종목은 매도 조건일 때 매도)
    mode="u_only" : 비교용 — U(예측 상승 변동성) 상위 entry_pct, 보유 종목은 U 상위 (entry_pct + 10%p) 이면 유지
    mode="r_only" : 비교용 — R 상위 entry_pct, 보유 종목은 R 상위 (entry_pct + 10%p) 이면 유지
    """
    g = g.set_index("ticker")
    U, R = g["pRSp"], g["pRSp"] / g["pRSn"]
    n_target = max(1, int(round(entry_pct * len(g))))
    if mode in ("u_only", "r_only"):
        sc = U if mode == "u_only" else R
        pct = sc.rank(pct=True)
        keep = [t for t in prev if t in pct.index and pct[t] >= 1.0 - (entry_pct + 0.10)]
        keep = sorted(keep, key=lambda t: -sc[t])[:n_target]
        new = [t for t in sc.sort_values(ascending=False).index if t not in keep][: n_target - len(keep)]
        names = keep + new
    else:
        sell = _sell_flag(R, exit_rule)
        keep = [t for t in prev if t in g.index and not sell[t]]
        keep = sorted(keep, key=lambda t: -R[t])[:n_target]
        eligible = (U.rank(pct=True) >= 0.5) & ~sell
        cand = R[eligible].sort_values(ascending=False).index
        new = [t for t in cand if t not in keep][: max(0, n_target - len(keep))]
        names = keep + new
    if not names:
        return pd.Series(dtype=float)
    w = pd.Series(min(1.0 / len(names), max_weight), index=names)
    return w / w.sum() if w.sum() > 1 else w


def run_s4(fc: pd.DataFrame, entry_pct: float, exit_rule: str = "rel", max_weight: float = 0.05,
           mode: str = "s4") -> dict:
    holdings, prev = {}, set()
    for m, g in fc.groupby("month"):
        w = s4_select(g, entry_pct, exit_rule, prev, max_weight, mode)
        if len(w):
            holdings[m] = w
            prev = set(w.index)
    return holdings


def forecast_quality(fc: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """월별 횡단면 스피어만 상관: 예측 vs 실현 (RS⁺, RS⁻, 상승 비중 RS⁺/(RS⁺+RS⁻))"""
    d = fc.merge(feat[["month", "ticker", "y_RSp", "y_RSn"]], on=["month", "ticker"]).dropna()
    d["pShare"] = d["pRSp"] / (d["pRSp"] + d["pRSn"])
    d["yShare"] = d["y_RSp"] / (d["y_RSp"] + d["y_RSn"])
    rows = []
    for m, g in d.groupby("month"):
        if len(g) < 30:
            continue
        rows.append({"month": m,
                     "RSp": g["pRSp"].corr(g["y_RSp"], method="spearman"),
                     "RSn": g["pRSn"].corr(g["y_RSn"], method="spearman"),
                     "share": g["pShare"].corr(g["yShare"], method="spearman"),
                     "pRSp_vs_pRSn": g["pRSp"].corr(g["pRSn"], method="spearman")})
    return pd.DataFrame(rows).set_index("month")
