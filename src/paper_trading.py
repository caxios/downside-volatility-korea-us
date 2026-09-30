"""
페이퍼 트레이딩 (8단계).

- 동결된 규칙만 사용한다 (load_frozen). 규칙을 여기서 바꾸지 않는다.
- 매 거래일 장 마감 후 실행 → 다음 거래일 목표 비중을 기록한다.
- 기록은 logs/ 아래 CSV에 누적되며, 같은 (시장, 데이터 날짜) 행은 중복 기록하지 않는다.
- evaluate()는 기록된 비중과 실제 이후 수익률로 페이퍼 성과를 계산하고 중단 규칙을 점검한다.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .backtest import backtest_weights, perf_stats
from .config import LOG_DIR
from .firewall import load_frozen
from .pipeline import load_market, s1_all

S1_LOG = LOG_DIR / "paper_s1.csv"
S2_LOG = LOG_DIR / "paper_s2.csv"


def _append(path, row: dict, keys: tuple):
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(df) and ((df[list(keys)].astype(str) == pd.Series({k: str(row[k]) for k in keys})).all(axis=1)).any():
        return False
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    return True


def update_s1(market: str) -> dict:
    frozen = load_frozen("S1")["rules"][market]
    mk = load_market(market, refresh=True)
    res = s1_all(mk, frozen["lambda"], sigma_target=frozen["sigma_target"],
                 turbulent_weight_c=frozen.get("turbulent_weight_c", 0.0))
    rule = frozen["rule"]
    w = res["weights"][rule].dropna()
    last = w.index[-1]
    row = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "market": market,
        "data_date": str(last.date()),
        "regime": int(res["states"].get(last, -1)),
        "rv_forecast": float(res["forecast"].get(last, np.nan)),
        "rule": rule,
        "target_weight": float(w.iloc[-1]),
        "etf_close": float(mk["etf"]["Close"].iloc[-1]),
        "etf_last_date": str(mk["etf"].index[-1].date()),
    }
    if row["data_date"] != row["etf_last_date"]:
        row["warning"] = "지수와 ETF의 마지막 날짜가 다름 — 데이터 갱신 지연 확인"
    _append(S1_LOG, row, ("market", "data_date"))
    return row


def evaluate_s1(market: str) -> dict:
    """기록된 목표 비중(버퍼 포함 규칙 적용) → 다음 날 ETF 수익률로 페이퍼 성과, 중단 규칙 점검"""
    frozen = load_frozen("S1")["rules"][market]
    log = pd.read_csv(S1_LOG)
    log = log[log["market"] == market]
    if log.empty:
        return {"message": "기록 없음"}
    w = pd.Series(log["target_weight"].to_numpy(float), index=pd.to_datetime(log["data_date"])).sort_index()
    w = w[~w.index.duplicated(keep="last")]
    mk = load_market(market, refresh=False)
    ret = mk["etf_ret"].loc[w.index.min():]
    bt = backtest_weights(w, ret, frozen["cost_oneway"], buffer=frozen.get("buffer", 0.0))
    bt = bt.loc[bt.index > w.index.min()]
    if len(bt) < 5:
        return {"message": f"평가하기엔 기록이 너무 짧음 ({len(bt)}일)", "n_days": len(bt)}
    stats = perf_stats(bt["net"])
    exp_sr, exp_mdd = frozen["expected"]["sharpe"], frozen["expected"]["max_dd"]
    yrs = max(len(bt) / 252.0, 1e-9)
    se = np.sqrt((1 + 0.5 * exp_sr ** 2) / yrs)
    flags = []
    if len(stats) and stats["max_dd"] < 1.5 * exp_mdd:
        flags.append("STOP: 최대 낙폭이 백테스트의 1.5배 초과")
    if len(stats) and yrs >= 0.5 and stats["sharpe"] < exp_sr - 1.96 * se:
        flags.append("REVIEW: 샤프가 백테스트 기대치 95% 하한 미만")
    return {"stats": stats, "flags": flags, "n_days": len(bt), "expected": frozen["expected"]}


def update_s2(market: str, month_panel: pd.DataFrame) -> pd.Series:
    """
    month_panel: 가장 최근 완결된 달의 시그널 패널 (cross_section.build_panel 결과 중 한 달)
    동결된 S2 규칙으로 보유 종목을 정하고 기록한다.
    """
    from .strategies import s2_select

    rules = load_frozen("S2S3")["rules"][market]
    prev = set()
    if S2_LOG.exists():
        lg = pd.read_csv(S2_LOG)
        lg = lg[lg["market"] == market]
        if len(lg):
            last_m = lg["month"].max()
            prev = set(lg.loc[lg["month"] == last_m, "ticker"])
    m = month_panel["month"].max()
    g = month_panel[month_panel["month"] == m]
    w = s2_select(g, rules["signals"], rules["entry_pct"], rules["buffer_pct"], prev, rules["max_weight"])
    rows = pd.DataFrame({"run_time": datetime.now().isoformat(timespec="seconds"), "market": market,
                         "month": str(m), "ticker": w.index, "weight": w.to_numpy()})
    if S2_LOG.exists():
        old = pd.read_csv(S2_LOG)
        old = old[~((old["market"] == market) & (old["month"] == str(m)))]
        rows = pd.concat([old, rows], ignore_index=True)
    rows.to_csv(S2_LOG, index=False)
    return w
