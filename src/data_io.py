"""
데이터 입출력.

- 가격: yfinance (일봉, 60분봉, 5분봉)
- 한국 보조 데이터: pykrx (지수 구성종목, 시가총액, 공매도 잔고)
- 미국 구성종목: 사용자가 내려받은 과거 구성종목 CSV (없으면 현재 구성종목 + 생존편향 경고)
- 팩터: Kenneth French 데이터 라이브러리
- 무위험 금리: FRED (미국). 한국은 사용자 CSV 또는 0.

주의: yfinance와 pykrx는 비공식 라이브러리라 사이트 구조가 바뀌면 작동하지 않을 수 있다.
모든 다운로드는 data/raw 에 캐시하고, 분석은 캐시에서 읽는다.
"""
from __future__ import annotations

import io
import json
import time
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DIR, PROC_DIR

# ======================================================================
# 공통 유틸
# ======================================================================

def safe_name(ticker: str) -> str:
    return ticker.replace("^", "IDX_").replace("/", "_").replace("=", "_")


def exchange_tz(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith(".KS") or t.endswith(".KQ") or t.startswith("^KS") or t.startswith("^KQ"):
        return "Asia/Seoul"
    return "America/New_York"


def _naive_dates(idx: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _extract(df: pd.DataFrame | None, ticker: str) -> pd.DataFrame | None:
    """yf.download 결과(단일/멀티 컬럼 모두)에서 한 종목만 꺼낸다."""
    if df is None or df.empty:
        return None
    if not isinstance(df.columns, pd.MultiIndex):
        out = df.copy()
    else:
        lv0 = set(df.columns.get_level_values(0))
        lv1 = set(df.columns.get_level_values(1))
        if ticker in lv0:
            out = df[ticker].copy()
        elif ticker in lv1:
            out = df.xs(ticker, axis=1, level=1).copy()
        else:
            return None
    out = out.dropna(how="all")
    return out if not out.empty else None


def yf_download(tickers, chunk: int = 25, pause: float = 3.0, retries: int = 3, **kw) -> dict:
    """여러 종목을 나눠서 받는다. 반환: {ticker: DataFrame}"""
    import yfinance as yf

    tickers = list(dict.fromkeys(tickers))
    out = {}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        df = None
        for attempt in range(retries):
            try:
                df = yf.download(batch, group_by="ticker", auto_adjust=False,
                                 threads=True, progress=False, **kw)
                break
            except Exception as e:  # 네트워크/요청 제한
                warnings.warn(f"yfinance 실패 ({batch[:3]}...): {e} / 재시도 {attempt + 1}")
                time.sleep(pause * (attempt + 2))
        for t in batch:
            sub = _extract(df, t)
            if sub is not None:
                out[t] = sub
        time.sleep(pause)
    return out


# ======================================================================
# 일봉
# ======================================================================

def daily_path(ticker: str):
    folder = RAW_DIR / "daily"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{safe_name(ticker)}.parquet"


def download_daily(tickers, start: str, end: str | None = None, refresh: bool = False) -> dict:
    """캐시에 없는 종목만 받는다. refresh=True면 전부 다시 받는다."""
    need = [t for t in tickers if refresh or not daily_path(t).exists()]
    if need:
        got = yf_download(need, start=start, end=end, interval="1d")
        for t, df in got.items():
            df = df.copy()
            df.index = _naive_dates(df.index)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.to_parquet(daily_path(t))
        missing = sorted(set(need) - set(got))
        if missing:
            pd.Series(missing, name="ticker").to_csv(
                RAW_DIR / f"daily_missing_{datetime.now():%Y%m%d_%H%M}.csv", index=False)
            warnings.warn(f"데이터를 받지 못한 종목 {len(missing)}개 (생존편향 가능). 목록을 저장했음.")
    return {t: load_daily(t) for t in tickers if daily_path(t).exists()}


def load_daily(ticker: str) -> pd.DataFrame:
    return pd.read_parquet(daily_path(ticker))


# ======================================================================
# 장중 (60분봉 2년 / 5분봉 60일 → 매주 누적 수집)
# ======================================================================

def collect_intraday(tickers, interval: str = "5m", period: str = "60d", tag: str = "us") -> int:
    folder = RAW_DIR / f"intraday_{interval}"
    folder.mkdir(parents=True, exist_ok=True)
    got = yf_download(tickers, interval=interval, period=period)
    frames = []
    for t, df in got.items():
        df = df.copy()
        idx = pd.DatetimeIndex(df.index)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        df.index = idx
        df["ticker"] = t
        frames.append(df)
    if not frames:
        warnings.warn("장중 데이터를 하나도 받지 못했음")
        return 0
    long = pd.concat(frames)
    long.index.name = "datetime_utc"
    long.to_parquet(folder / f"{tag}_{datetime.now():%Y%m%d_%H%M}.parquet")
    return len(long)


def load_intraday(interval: str = "5m", tickers=None) -> pd.DataFrame:
    """수집한 파일을 모두 합치고 (ticker, 시각) 중복을 제거한다."""
    folder = RAW_DIR / f"intraday_{interval}"
    files = sorted(folder.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files]).reset_index()
    df = df.drop_duplicates(["ticker", "datetime_utc"], keep="last")
    if tickers is not None:
        df = df[df["ticker"].isin(tickers)]
    return df.set_index("datetime_utc").sort_index()


# ======================================================================
# 구성종목 (시점별)
# ======================================================================

def _to_yf_us(t: str) -> str:
    return t.strip().replace(".", "-")


def load_sp500_membership(months: pd.PeriodIndex, csv_path: str | None = None) -> dict:
    """
    months 각각의 월말 기준 S&P 500 구성종목 {Period: set(ticker)}.

    csv_path: 과거 구성종목 파일 (예: GitHub fja05680/sp500 의 'date,tickers' 형식).
    없으면 현재 구성종목을 모든 달에 적용한다 → 생존편향이 크다는 경고를 반드시 보고할 것.
    """
    if csv_path:
        raw = pd.read_csv(csv_path)
        date_col = [c for c in raw.columns if "date" in c.lower()][0]
        tick_col = [c for c in raw.columns if "ticker" in c.lower()][0]
        raw[date_col] = pd.to_datetime(raw[date_col])
        raw = raw.sort_values(date_col)
        snaps = [(d, {_to_yf_us(x) for x in str(s).split(",") if x.strip()})
                 for d, s in zip(raw[date_col], raw[tick_col])]
        out, j = {}, 0
        for m in months:
            end = m.end_time
            while j + 1 < len(snaps) and snaps[j + 1][0] <= end:
                j += 1
            if snaps[j][0] <= end:
                out[m] = snaps[j][1]
        return out

    warnings.warn("S&P 500 과거 구성종목 CSV가 없어 현재 구성종목을 씀: 생존편향이 큼")
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    cur = {_to_yf_us(t) for t in tables[0]["Symbol"]}
    return {m: cur for m in months}


def _pykrx():
    # pykrx는 import 시점에 KRX_ID·KRX_PW 환경 변수로 KRX에 로그인한다 (공매도 잔고 등 로그인 필요 자료).
    # 프로젝트 .env에 두 값을 넣어 두면 여기서 먼저 읽어 온다.
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass
    try:
        from pykrx import stock
        return stock
    except ImportError as e:
        raise ImportError("pykrx가 필요함: pip install pykrx") from e


def _krx_business_day(ts: pd.Timestamp) -> str:
    stock = _pykrx()
    d = ts.strftime("%Y%m%d")
    try:
        return stock.get_nearest_business_day_in_a_week(date=d)
    except Exception:
        return d


def kospi200_membership(months: pd.PeriodIndex, pause: float = 0.5) -> dict:
    """월말 기준 KOSPI200 구성종목 {Period: set('005930.KS', ...)}"""
    stock = _pykrx()
    cache = PROC_DIR / "kospi200_membership.json"
    saved = json.loads(cache.read_text()) if cache.exists() else {}
    out = {}
    for m in months:
        key = str(m)
        if key not in saved:
            day = _krx_business_day(m.end_time)
            try:
                tickers = stock.get_index_portfolio_deposit_file("1028", day)
                saved[key] = [f"{t}.KS" for t in tickers]
            except Exception as e:
                warnings.warn(f"KOSPI200 구성종목 조회 실패 {key}: {e}")
                continue
            time.sleep(pause)
        out[m] = set(saved[key])
    cache.write_text(json.dumps(saved))
    return out


def kr_market_cap(months: pd.PeriodIndex, pause: float = 0.5) -> pd.DataFrame:
    """월말 시가총액 (행: Period, 열: 'xxxxxx.KS')"""
    stock = _pykrx()
    rows = {}
    for m in months:
        day = _krx_business_day(m.end_time)
        try:
            df = stock.get_market_cap(day, market="KOSPI")
            col = "시가총액" if "시가총액" in df.columns else df.columns[1]
            rows[m] = pd.Series(df[col].values, index=[f"{t}.KS" for t in df.index])
        except Exception as e:
            warnings.warn(f"시가총액 조회 실패 {m}: {e}")
        time.sleep(pause)
    return pd.DataFrame(rows).T


def kr_short_balance_ratio(start: str, end: str, step_days: int = 7, pause: float = 0.5) -> pd.Series:
    """
    구간 [start, end] 동안 종목별 공매도 잔고 비중(%)의 평균. H4의 HighShort 분류용.
    pykrx 함수명·컬럼명이 버전마다 다를 수 있어 방어적으로 처리한다.
    """
    stock = _pykrx()
    fn = getattr(stock, "get_shorting_balance_by_ticker", None)
    if fn is None:
        raise AttributeError("pykrx에 get_shorting_balance_by_ticker가 없음: 버전을 확인할 것")
    vals = []
    for d in pd.date_range(start, end, freq=f"{step_days}D"):
        day = _krx_business_day(d)
        try:
            df = fn(day, market="KOSPI")
            col = "비중" if "비중" in df.columns else df.columns[-1]
            vals.append(pd.Series(df[col].astype(float).values, index=[f"{t}.KS" for t in df.index]))
        except Exception as e:
            warnings.warn(f"공매도 잔고 조회 실패 {day}: {e}")
        time.sleep(pause)
    if not vals:
        raise RuntimeError("공매도 잔고 데이터를 하나도 받지 못함")
    return pd.concat(vals, axis=1).mean(axis=1)


# ======================================================================
# 팩터, 금리, 업종
# ======================================================================

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {
    "ff3": "F-F_Research_Data_Factors_CSV.zip",
    "mom": "F-F_Momentum_Factor_CSV.zip",
    "strev": "F-F_ST_Reversal_Factor_CSV.zip",
}


def _parse_french_monthly(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    header, rows, started = None, [], False
    for i, ln in enumerate(lines):
        parts = [p.strip() for p in ln.split(",")]
        first = parts[0]
        if first.isdigit() and len(first) == 6:
            if not started:
                j = i - 1
                while j >= 0 and not lines[j].strip():
                    j -= 1
                header = [p.strip() for p in lines[j].split(",")]
                started = True
            rows.append(parts)
        elif started:
            break
    cols = header[1:] if header and len(header) == len(rows[0]) else [f"f{k}" for k in range(len(rows[0]) - 1)]
    df = pd.DataFrame([r[1:] for r in rows], columns=cols).astype(float) / 100.0
    df.index = pd.PeriodIndex([f"{r[0][:4]}-{r[0][4:]}" for r in rows], freq="M")
    return df


def load_french_monthly() -> pd.DataFrame:
    """미국 월별 팩터 (Mkt-RF, SMB, HML, RF, Mom, ST_Rev) — 소수 단위"""
    import urllib.request

    cache = RAW_DIR / "french_monthly.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frames = []
    for name, fname in FRENCH_FILES.items():
        with urllib.request.urlopen(FRENCH_BASE + fname, timeout=60) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            text = z.read(z.namelist()[0]).decode("latin-1")
        frames.append(_parse_french_monthly(text))
    out = pd.concat(frames, axis=1)
    out.columns = [c.strip() for c in out.columns]
    out.to_parquet(cache)
    return out


def load_us_rf_daily() -> pd.Series:
    """FRED 3개월 T-bill(DTB3) → 일별 수익률(소수)"""
    cache = RAW_DIR / "fred_dtb3.parquet"
    if cache.exists():
        s = pd.read_parquet(cache).iloc[:, 0]
    else:
        df = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3")
        date_col = df.columns[0]
        s = pd.to_numeric(df["DTB3"], errors="coerce")
        s.index = pd.to_datetime(df[date_col])
        s.to_frame("DTB3").to_parquet(cache)
    return (s.ffill() / 100.0 / 252.0).rename("rf")


def load_kr_rf_daily(csv_path: str | None = None) -> pd.Series | None:
    """
    한국 무위험 금리: ECOS 등에서 받은 CSV(date, rate_annual_percent)를 넣으면 일별로 변환.
    없으면 None (현금 수익률 0으로 계산됨 → 보고서에 명시).
    """
    if not csv_path:
        warnings.warn("한국 무위험 금리 CSV 없음: 현금 수익률 0으로 가정")
        return None
    df = pd.read_csv(csv_path)
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return (s.ffill() / 100.0 / 252.0).rename("rf")


def fetch_sectors(tickers, pause: float = 0.3) -> pd.Series:
    """yfinance 종목 정보에서 현재 섹터 (과거 섹터 변경은 반영 못 함)"""
    import yfinance as yf

    cache = PROC_DIR / "sectors.json"
    saved = json.loads(cache.read_text()) if cache.exists() else {}
    for t in tickers:
        if t in saved:
            continue
        try:
            saved[t] = yf.Ticker(t).info.get("sector") or "Unknown"
        except Exception:
            saved[t] = "Unknown"
        time.sleep(pause)
    cache.write_text(json.dumps(saved))
    return pd.Series({t: saved.get(t, "Unknown") for t in tickers}, name="sector")
