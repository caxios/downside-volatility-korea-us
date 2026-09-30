"""
3단계: 종목 데이터 구축

  python scripts/step3_build_universe.py --market US --sp500-csv path/to/sp500_history.csv
  python scripts/step3_build_universe.py --market KR                # KRX Open API (기본)
  python scripts/step3_build_universe.py --market KR --short        # + 공매도 잔고 (pykrx, KRX 로그인 필요)
  python scripts/step3_build_universe.py --market both --sectors    # 업종(현재 기준)까지

산출물 (data/processed):
  panel_{US,KR}.parquet        종목×월 시그널 패널 (전체 기간 저장, 분석 단계에서 방화벽 적용)
  daily_ret_{US,KR}.parquet    일별 단순수익률 행렬 (S3용)
  kr_short_ratio_preban.csv    금지 직전 공매도 잔고 비중 평균 (H4용)
결과 (results/step3): 누락 종목 보고서
"""
import argparse
from datetime import date

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import DATA_START, MARKETS, PRE_BAN_SHORT_WINDOW, PROC_DIR
from src.cross_section import build_panel, missing_report
from src.data_io import (download_daily, fetch_sectors, kospi200_membership, kr_market_cap,
                         kr_short_balance_ratio, load_sp500_membership)
from src.pipeline import load_market, save_panel
from src.quality import clean_daily, quality_report
from src.report import save


KR_API_START = "2010-01-01"      # KRX Open API 제공 시작


def build(market, args):
    kr = market == "KR"
    if kr and args.kr_source == "openapi":
        # KRX Open API: 날짜별 전종목 → 상장폐지 종목 포함 (생존편향 없음), 2010년 이후
        from src.krx_api import download_range, load_range, monthly_mcap, to_daily_dict, top_mcap_membership
        stat = download_range(KR_API_START)
        print(f"[KR] Open API 호출 {stat['calls']}회, 남은 미수집 {stat['remaining_days']}일")
        if stat["remaining_days"] > 5:
            print("[KR] 아직 다 받지 못함 (하루 호출 한도) → 내일 같은 명령을 다시 실행할 것. 패널 구축은 보류")
            return
        long = load_range(KR_API_START)
        membership = top_mcap_membership(long, 200)
        mcap = monthly_mcap(long)
        raw = to_daily_dict(long)
        print(f"[KR] 월말 시가총액 상위 200 보통주를 KOSPI200 대리로 사용 (종목 합집합 {len(raw)}개)")
    else:
        months = pd.period_range("2004-01" if kr else DATA_START[:7], date.today().strftime("%Y-%m"), freq="M")
        if kr:
            membership = kospi200_membership(months)
            mcap = kr_market_cap(pd.PeriodIndex(sorted(membership)))
        else:
            membership = load_sp500_membership(months, args.sp500_csv)
            mcap = None
        tickers = sorted(set().union(*membership.values()))
        print(f"[{market}] 구성종목 합집합 {len(tickers)}개 다운로드")
        raw = download_daily(tickers, start=DATA_START, refresh=args.refresh)

    daily, qrows = {}, []
    for t, df in raw.items():
        if len(df) < 60:
            continue
        qrows.append(quality_report(df, t))
        daily[t] = clean_daily(df, kr=kr)
    save(pd.DataFrame(qrows).set_index("name"), "step3", f"{market}_quality", show=False)
    save(missing_report(membership, set(daily)), "step3", f"{market}_missing_by_month", show=False)
    miss = missing_report(membership, set(daily))
    print(f"[{market}] 구성종목 중 데이터 누락 비율 평균 {miss['missing_share'].mean():.1%} "
          f"(최대 {miss['missing_share'].max():.1%}) — 생존편향 크기로 보고할 것")

    mk = load_market(market)
    mkt_r = mk["idx"]["r"]
    panel = build_panel(daily, mkt_r, membership, mcap, min_price=MARKETS[market]["min_price"])
    if args.sectors:
        sec = fetch_sectors(sorted(panel["ticker"].unique()))
        panel["sector"] = panel["ticker"].map(sec)
    save_panel(panel, PROC_DIR / f"panel_{market}.parquet")
    dr = pd.DataFrame({t: np.expm1(df["r"]) for t, df in daily.items()})
    dr.to_parquet(PROC_DIR / f"daily_ret_{market}.parquet")
    print(f"[{market}] 패널 {len(panel):,}행, 종목 {panel['ticker'].nunique()}개, "
          f"기간 {panel['month'].min()} ~ {panel['month'].max()}")

    if kr and args.short:
        ratio = kr_short_balance_ratio(*PRE_BAN_SHORT_WINDOW)
        ratio.rename("short_ratio").to_csv(PROC_DIR / "kr_short_ratio_preban.csv")
        print(f"[KR] 금지 직전 공매도 잔고 비중 {ratio.notna().sum()}종목 저장")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["US", "KR", "both"], default="both")
    ap.add_argument("--sp500-csv", default=None, help="S&P 500 과거 구성종목 CSV (date,tickers)")
    ap.add_argument("--short", action="store_true", help="한국 공매도 잔고 수집 (H4)")
    ap.add_argument("--sectors", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--kr-source", choices=["openapi", "pykrx"], default="openapi",
                    help="한국 종목 데이터 출처: KRX Open API(기본, 2010년~, 생존편향 없음) 또는 pykrx+yfinance")
    args = ap.parse_args()
    for m in (["US", "KR"] if args.market == "both" else [args.market]):
        build(m, args)


if __name__ == "__main__":
    main()