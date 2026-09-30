"""
한국거래소(KRX) Open API 클라이언트 — 한국 종목 데이터용

준비
  1. https://openapi.krx.co.kr 회원가입 → 인증키 신청 (관리자 승인 필요)
  2. '서비스 이용' 메뉴에서 쓸 API를 개별 신청 → 승인될 때까지 401 오류
       - 주식 > 유가증권 일별매매정보   (필수)
  3. 프로젝트 폴더의 .env 에  KRX_OPENAPI_KEY=발급받은키  저장 (git에 올리지 말 것)

제약 (호출 전에 알아둘 것)
  - 제공 기간: 2010년 이후
  - 호출 한도: 인증키당 하루 10,000회 (하루치 전종목 = 1회). 초과 시 다음 날 이어서 실행
  - 비상업적 목적만 허용. 받은 데이터를 재배포(예: GitHub 업로드)하지 말 것

이 모듈이 하는 일
  - 날짜별 전종목 일별매매정보를 받아 data/raw/krx_openapi/ 에 날짜별로 캐시 (재실행 시 추가 호출 없음)
  - 그날 상장돼 있던 모든 종목이 들어 있으므로, 이후 상장폐지된 종목도 포함된다 → 생존편향 제거
  - 수익률은 KRX 등락률(액면분할 등 권리 조정된 기준가 대비)로 계산 → 분할일 수익률 왜곡 없음
    (단, 배당은 포함되지 않은 가격수익률)
  - KOSPI200 과거 구성종목은 제공되지 않으므로, 월말 시가총액 상위 200개 보통주로 대리한다
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd

from .config import RAW_DIR, ROOT

BASE = "https://data-dbg.krx.co.kr/svc/apis"
ENDPOINTS = {
    "KOSPI": "sto/stk_bydd_trd",      # 유가증권 일별매매정보
    "KOSDAQ": "sto/ksq_bydd_trd",     # 코스닥 일별매매정보 (별도 신청 필요)
}
FIELD_MAP = {
    "BAS_DD": "date", "ISU_CD": "code", "ISU_NM": "name", "MKT_NM": "market",
    "SECT_TP_NM": "section", "TDD_CLSPRC": "close", "CMPPREVDD_PRC": "chg",
    "FLUC_RT": "fluc_rt", "TDD_OPNPRC": "open", "TDD_HGPRC": "high", "TDD_LWPRC": "low",
    "ACC_TRDVOL": "volume", "ACC_TRDVAL": "value", "MKTCAP": "mktcap", "LIST_SHRS": "shares",
}
NUMERIC = ["close", "chg", "fluc_rt", "open", "high", "low", "volume", "value", "mktcap", "shares"]


class KRXApiError(RuntimeError):
    pass


class KRXDailyLimit(KRXApiError):
    pass


# ----------------------------------------------------------------- 인증키
def api_key() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = os.environ.get("KRX_OPENAPI_KEY", "").strip()
    if not key:
        raise KRXApiError("KRX_OPENAPI_KEY 없음: openapi.krx.co.kr 에서 인증키를 받아 .env 에 "
                          "KRX_OPENAPI_KEY=키 로 저장할 것 (pip install python-dotenv)")
    return key


# ----------------------------------------------------------------- 호출
def _post(path: str, bas_dd: str, key: str, retries: int = 3, timeout: int = 30) -> list:
    import requests

    url = f"{BASE}/{path}"
    headers = {"AUTH_KEY": key, "Content-Type": "application/json", "Accept": "application/json"}
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json={"basDd": bas_dd}, timeout=timeout)
        except Exception as e:
            warnings.warn(f"KRX 호출 네트워크 오류 {bas_dd}: {e}")
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 200:
            try:
                return r.json().get("OutBlock_1", []) or []
            except ValueError:                        # 가끔 빈 본문이 옴 → 일시 오류로 보고 재시도
                warnings.warn(f"KRX 응답이 JSON이 아님 ({bas_dd}) 재시도 {attempt + 1}: {r.text[:200]!r}")
                time.sleep(10 * (attempt + 1))
                continue
        if r.status_code == 401:
            raise KRXApiError("401 인증 실패: 인증키가 틀렸거나, 이 API('유가증권 일별매매정보')의 "
                              "이용 신청이 아직 승인되지 않았음 (마이페이지 > 이용현황 확인)")
        if r.status_code == 429:
            raise KRXDailyLimit("하루 호출 한도 초과 (인증키당 10,000회). 내일 같은 명령을 다시 실행하면 이어서 받음")
        warnings.warn(f"KRX HTTP {r.status_code} ({bas_dd}) 재시도 {attempt + 1}")
        time.sleep(3 * (attempt + 1))
    raise KRXApiError(f"KRX 호출 실패: {path} {bas_dd}")


def _clean(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=list(FIELD_MAP.values()))
    df = df.rename(columns=FIELD_MAP)
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False)
                                  .replace({"-": np.nan, "": np.nan}), errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df[[c for c in FIELD_MAP.values() if c in df.columns]]


def _cache_dir(market: str):
    d = RAW_DIR / "krx_openapi" / market
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_range(start: str, end: str | None = None, market: str = "KOSPI",
                   max_calls: int = 9500, pause: float = 0.2) -> dict:
    """
    [start, end] 평일마다 전종목 일별매매정보를 받아 날짜별 parquet로 캐시한다.
    휴장일은 빈 파일로 기록해 다시 호출하지 않는다. 한도에 걸리면 받은 데까지 저장하고 멈춘다.
    """
    key = api_key()
    folder = _cache_dir(market)
    days = pd.bdate_range(start, end or pd.Timestamp.today().normalize())
    todo = [d for d in days if not (folder / f"{d:%Y%m%d}.parquet").exists()]
    calls, got = 0, 0
    print(f"[KRX Open API] {market}: 전체 {len(days)}일 중 미수집 {len(todo)}일")
    for d in todo:
        if calls >= max_calls:
            print(f"오늘 호출 상한({max_calls})에 도달 → 내일 다시 실행하면 이어서 받음")
            break
        try:
            df = _clean(_post(ENDPOINTS[market], f"{d:%Y%m%d}", key))
        except KRXDailyLimit as e:
            print(e)
            break
        calls += 1
        if d.normalize() == pd.Timestamp.today().normalize() and df.empty:
            continue                                  # 오늘 자료는 아직 없을 수 있음 → 기록하지 않음
        df.to_parquet(folder / f"{d:%Y%m%d}.parquet")
        got += int(not df.empty)
        if calls % 250 == 0:
            print(f"  {calls}회 호출, 거래일 {got}일 저장 (마지막 {d:%Y-%m-%d})")
        time.sleep(pause)
    remaining = sum(1 for d in days if not (folder / f"{d:%Y%m%d}.parquet").exists())
    return {"calls": calls, "trading_days_saved": got, "remaining_days": remaining}


def load_range(start: str, end: str | None = None, market: str = "KOSPI") -> pd.DataFrame:
    folder = _cache_dir(market)
    files = sorted(f for f in folder.glob("*.parquet")
                   if start.replace("-", "") <= f.stem <= (end or "99999999").replace("-", ""))
    frames = [pd.read_parquet(f) for f in files]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=list(FIELD_MAP.values()))
    return pd.concat(frames, ignore_index=True).sort_values(["code", "date"])


# ----------------------------------------------------------------- 프로젝트 형식으로 변환
def is_common_stock(code: pd.Series, name: pd.Series) -> pd.Series:
    """보통주 판별 (근사): 단축코드 끝자리 0, 스팩·리츠·인프라펀드 제외"""
    code = code.astype(str)
    excl = name.astype(str).str.contains("스팩|리츠|인프라|기업인수목적", regex=True)
    return code.str[-1].eq("0") & ~excl


def to_daily_dict(long: pd.DataFrame, suffix: str = ".KS") -> dict:
    """
    {ticker: DataFrame[Open High Low Close Adj Close Volume Value MktCap]} — quality.clean_daily 입력 형식
    Adj Close 는 등락률을 누적한 가격지수(시작=첫 종가)라서 액면분할 등이 반영돼 있다 (배당 미포함).
    거래가 없던 날(시가 0)은 OHLC를 NaN으로 두어 레인지 변동성 계산에서 제외된다.
    """
    out = {}
    for code, g in long.groupby("code"):
        g = g.sort_values("date").set_index("date")
        r = (g["fluc_rt"] / 100.0).fillna(0.0)
        adj = g["close"].iloc[0] * (1.0 + r).cumprod()
        no_trade = g["open"].fillna(0) <= 0
        df = pd.DataFrame({
            "Open": g["open"].where(~no_trade), "High": g["high"].where(~no_trade),
            "Low": g["low"].where(~no_trade), "Close": g["close"], "Adj Close": adj,
            "Volume": g["volume"], "Value": g["value"], "MktCap": g["mktcap"],
        })
        df.index.name = None
        out[f"{code}{suffix}"] = df
    return out


def top_mcap_membership(long: pd.DataFrame, n: int = 200, suffix: str = ".KS") -> dict:
    """KOSPI200 대리: 각 월 마지막 거래일 기준 시가총액 상위 n개 보통주 {Period: set}"""
    x = long[is_common_stock(long["code"], long["name"])].copy()
    x["month"] = x["date"].dt.to_period("M")
    last = x.groupby("month")["date"].transform("max")
    x = x[x["date"] == last]
    out = {}
    for m, g in x.groupby("month"):
        top = g.nlargest(n, "mktcap")["code"]
        out[m] = {f"{c}{suffix}" for c in top}
    return out


def monthly_mcap(long: pd.DataFrame, suffix: str = ".KS") -> pd.DataFrame:
    """월말 시가총액 (행 Period, 열 ticker)"""
    x = long.copy()
    x["month"] = x["date"].dt.to_period("M")
    last = x.groupby("month")["date"].transform("max")
    x = x[x["date"] == last]
    x["ticker"] = x["code"].astype(str) + suffix
    return x.pivot_table(index="month", columns="ticker", values="mktcap")