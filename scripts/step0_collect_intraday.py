"""
0단계: 장중 봉 수집 (매주 1회 실행 권장 — cron / 작업 스케줄러에 등록)

  python scripts/step0_collect_intraday.py              # 5분봉(60일) + 60분봉(730일)
  python scripts/step0_collect_intraday.py --interval 5m

5분봉은 최근 60일만 조회되므로, 매주 실행해 겹치게 쌓아야 빠지는 날이 없다.
수집 대상: config.INTRADAY_BASE + config/intraday_extra_{us,kr}.txt (없으면 아래 기본 대형주 목록)
"""
import argparse

import _bootstrap  # noqa: F401
from src.config import INTRADAY_BASE, INTRADAY_TOP_N, ROOT
from src.data_io import collect_intraday

# 기본 대형주 목록 (시가총액 순위는 바뀌므로 config/intraday_extra_*.txt 로 교체 권장)
DEFAULT_US = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "AVGO", "TSLA", "JPM",
              "LLY", "V", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "COST", "ABBV", "WMT", "NFLX", "BAC",
              "KO", "MRK", "CRM", "CVX", "ORCL", "AMD", "PEP", "ADBE", "TMO", "LIN", "ACN", "MCD",
              "CSCO", "WFC", "ABT", "DIS", "IBM", "GE", "QCOM", "TXN", "INTU", "CAT", "AMGN", "VZ",
              "PM", "NOW", "GS"]
DEFAULT_KR = [f"{c}.KS" for c in [
    "005930", "000660", "373220", "207940", "005380", "000270", "068270", "005490", "035420",
    "051910", "006400", "105560", "055550", "012330", "028260", "035720", "003670", "066570",
    "032830", "086790", "015760", "034730", "096770", "017670", "033780", "011200", "138040",
    "010130", "003550", "018260", "024110", "316140", "010950", "259960", "034020", "000810",
    "011170", "047050", "267250", "009540", "042660", "012450", "329180", "030200", "000100",
    "051900", "161390", "009150", "090430", "021240"]]


def extra(tag, default):
    f = ROOT / "config" / f"intraday_extra_{tag}.txt"
    if f.exists():
        return [x.strip() for x in f.read_text().splitlines() if x.strip()]
    return default[:INTRADAY_TOP_N]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", choices=["5m", "60m", "both"], default="both")
    args = ap.parse_args()
    lists = {"us": INTRADAY_BASE["us"] + extra("us", DEFAULT_US),
             "kr": INTRADAY_BASE["kr"] + extra("kr", DEFAULT_KR)}
    plan = {"5m": "60d", "60m": "730d"}
    for interval, period in plan.items():
        if args.interval not in (interval, "both"):
            continue
        for tag, tickers in lists.items():
            n = collect_intraday(tickers, interval=interval, period=period, tag=tag)
            print(f"[{interval}] {tag}: {len(tickers)}종목, {n}행 저장")


if __name__ == "__main__":
    main()
