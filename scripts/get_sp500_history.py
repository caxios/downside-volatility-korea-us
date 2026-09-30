"""
S&P 500 과거 구성종목 CSV를 받아 프로젝트 루트에 sp500_history.csv로 저장한다.
실행: python scripts/get_sp500_history.py
"""
import io
import urllib.request

import pandas as pd

import _bootstrap  # noqa: F401
from src.config import ROOT

CANDIDATES = [
    "https://raw.githubusercontent.com/hanshof/sp500_constituents/main/sp_500_historical_components.csv",
    "https://raw.githubusercontent.com/hanshof/sp500_constituents/master/sp_500_historical_components.csv",
]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def main():
    for url in CANDIDATES:
        try:
            df = pd.read_csv(io.StringIO(fetch(url)))
        except Exception as e:
            print(f"실패: {url} ({e})")
            continue
        date_cols = [c for c in df.columns if "date" in c.lower()]
        tick_cols = [c for c in df.columns if "ticker" in c.lower()]
        if not date_cols or not tick_cols:
            print(f"형식이 다름 (컬럼: {list(df.columns)}): {url}")
            continue
        out = df[[date_cols[0], tick_cols[0]]]
        out.columns = ["date", "tickers"]
        path = ROOT / "sp500_history.csv"
        out.to_csv(path, index=False)
        n_last = len(str(out["tickers"].iloc[-1]).split(","))
        print(f"저장 완료: {path}")
        print(f"기간 {out['date'].iloc[0]} ~ {out['date'].iloc[-1]}, {len(out)}행, 최신 구성종목 수 {n_last}")
        return
    print("자동 다운로드 실패 → 아래 '수동으로 받기' 방법을 쓸 것")


if __name__ == "__main__":
    main()