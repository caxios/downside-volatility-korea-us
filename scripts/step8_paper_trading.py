"""
8단계: 페이퍼 트레이딩 (동결된 규칙만 사용)

  python scripts/step8_paper_trading.py --mode s1-update      # 매 거래일 장 마감 후 (한국 16시 이후 / 미국 장 마감 후)
  python scripts/step8_paper_trading.py --mode s1-eval        # 주 1회: 페이퍼 성과와 중단 규칙 점검
  python scripts/step8_paper_trading.py --mode s2-update --sp500-csv ...   # 매월 첫 거래일

기록: logs/paper_s1.csv, logs/paper_s2.csv
실전 전환은 3~6개월 페이퍼 기록이 백테스트 기대 범위 안에 있을 때만, 잃어도 되는 소액으로.
"""
import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import MARKETS
from src.cross_section import monthly_signals
from src.data_io import kospi200_membership, load_sp500_membership, yf_download
from src.firewall import load_frozen
from src.paper_trading import evaluate_s1, update_s1, update_s2
from src.pipeline import load_market
from src.quality import clean_daily


def s2_month_panel(market, sp500_csv=None):
    """가장 최근 완결된 달의 시그널 패널 (최근 15개월 데이터만 새로 받음 → 캐시를 덮어쓰지 않음)"""
    last_month = pd.Period(date.today(), "M") - 1
    months = pd.PeriodIndex([last_month])
    member = (kospi200_membership(months) if market == "KR" else load_sp500_membership(months, sp500_csv))[last_month]
    start = (date.today() - timedelta(days=460)).isoformat()
    raw = yf_download(sorted(member), start=start, interval="1d")
    mk = load_market(market, refresh=True)
    rows = []
    for t, df in raw.items():
        df = df.copy()
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize() if df.index.tz is not None \
            else pd.DatetimeIndex(df.index).normalize()
        s = monthly_signals(clean_daily(df, kr=market == "KR"), mk["idx"]["r"])
        if last_month in s.index:
            row = s.loc[last_month].copy()
            row["ticker"], row["month"] = t, last_month
            rows.append(row)
    p = pd.DataFrame(rows)
    p = p[(p["price_end"] >= MARKETS[market]["min_price"]) & (p["zero_vol_days"] <= 4)]
    p["SIZE"] = p["DVOL"]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["s1-update", "s1-eval", "s2-update"], required=True)
    ap.add_argument("--market", choices=["US", "KR", "both"], default="both")
    ap.add_argument("--sp500-csv", default=None)
    args = ap.parse_args()
    markets = ["US", "KR"] if args.market == "both" else [args.market]

    if args.mode == "s1-update":
        for m in markets:
            row = update_s1(m)
            print(f"[{m}] {row['data_date']} 레짐={row['regime']} 예측분산={row['rv_forecast']:.3f} "
                  f"→ 다음 거래일 목표 비중 {row['target_weight']:.2f} {row.get('warning', '')}")
    elif args.mode == "s1-eval":
        for m in markets:
            out = evaluate_s1(m)
            print(f"\n[{m}] 페이퍼 {out.get('n_days', 0)}일 {out.get('message', '')}")
            if "stats" in out:
                print(out["stats"].round(3).to_string())
                print("백테스트 기대:", out["expected"])
                print("점검:", out["flags"] or "이상 없음")
    else:
        rules = load_frozen("S2S3")["rules"]
        for m in markets:
            if m not in rules:
                print(f"[{m}] 동결된 S2 규칙 없음 (4단계에서 시그널 미선택)")
                continue
            w = update_s2(m, s2_month_panel(m, args.sp500_csv))
            print(f"[{m}] 이번 달 보유 {len(w)}종목, 종목당 {w.mean():.3f}")


if __name__ == "__main__":
    main()
