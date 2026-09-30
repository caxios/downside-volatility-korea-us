"""
프로젝트 전역 설정.

원칙: 이 파일의 값은 결과를 보기 전에 확정한다. 바꿀 때는 README의
'설정 변경 기록'에 날짜와 이유를 남긴다. (바꾼 뒤 개발 구간 결과가 좋아졌다는
이유로 바꾸는 것은 선택 편향이다.)
"""
from pathlib import Path

# ---------------------------------------------------------------- 경로
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
LOG_DIR = ROOT / "logs"
FROZEN_DIR = ROOT / "config" / "frozen"
for _d in (RAW_DIR, PROC_DIR, RESULTS_DIR, LOG_DIR, FROZEN_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 기간
DATA_START = "2000-01-01"        # 레짐 모형이 닷컴 붕괴·2008년을 학습하도록
FIRST_FIT_END = "2004-12-31"     # 이 날짜까지는 학습만(burn-in), 평가하지 않음
DEV_START = "2005-01-01"
DEV_END = "2020-12-31"
CS_START = "2010-01-01"          # 횡단면 분석(4~6단계) 시작: 한국 Open API 제공 기간·미국 생존편향 고려
TEST_START = "2021-01-01"        # 이 날짜 이후는 규칙 동결 전 열람 금지 (src/firewall.py)

# 한국 공매도 전면 금지 (월 단위)
BAN_START_MONTH = "2023-11"
BAN_END_MONTH = "2025-03"
PRE_BAN_SHORT_WINDOW = ("2023-07-01", "2023-10-31")   # HighShort 분류용 (금지 직전)
# 개발 구간 안의 과거 한국 공매도 금지 기간 (월 단위). H4 플라시보에서 제외한다 (2026-09-29 추가).
#   2011-08-10 ~ 2011-11-09 전 종목 금지 (이후 금융주만 연장)
#   2020-03-16 ~ 2021-05-02 전 종목 금지 (2021-05-03 KOSPI200·KOSDAQ150 재개)
#   (2026-09-30 확인) 2021-05-03 이후에도 KOSPI200·KOSDAQ150 밖 종목은 2023-11 전면 금지까지 계속 금지.
#   한국 표본은 시총 상위 200(KOSPI200 대리)이라 2021-04를 끝으로 둔다. 2008-10 금지는 금융주만 2013-11-14까지 이어짐.
KR_PRIOR_BANS = (("2011-08", "2011-11"), ("2020-03", "2021-04"))

# ---------------------------------------------------------------- 시장
# 비용·세금 값은 가정이다. 연도별 세율·수수료는 실제 적용 전에 확인할 것.
MARKETS = {
    "US": dict(
        index="^GSPC", etf="SPY", vix="^VIX", tz="America/New_York",
        etf_cost_oneway=0.0002,     # ETF 편도 (스프레드+수수료 가정)
        stock_cost_oneway=0.0005,   # 대형주 편도
        sell_tax=0.0,               # 매도 세금 (거래세)
        min_price=5.0,
    ),
    "KR": dict(
        index="^KS11", etf="069500.KS", vix=None, tz="Asia/Seoul",
        etf_cost_oneway=0.0008,     # 국내 주식형 ETF는 증권거래세 면제 (가정)
        stock_cost_oneway=0.0010,
        sell_tax=0.0018,            # 증권거래세 가정치 — 기간별 실제 세율로 교체 권장
        min_price=1000.0,
    ),
}

# ---------------------------------------------------------------- HAR 계열
HAR_MIN_TRAIN = 1000       # 첫 학습에 필요한 최소 일수
HAR_REFIT_EVERY = 22       # 재추정 주기(거래일)

# ---------------------------------------------------------------- 레짐 (JM, HMM)
JM_HALFLIVES = (5, 10, 21)
REGIME_MODEL = "jm"         # 주 레짐 모형: "jm" 또는 "hmm" (다른 하나는 강건성 확인용)
JM_FEATURES = "vol"         # "vol": 변동성 크기 레짐(프로젝트 정의) / "shu": Shu-Yu-Mulvey 피처
JM_LAMBDA_GRID = (5.0, 15.0, 30.0, 60.0, 120.0)   # 후보는 적게 (선택 편향 억제)
JM_K = 2
JM_CLIP = 5.0              # 표준화 피처 절단 (극단값이 거리를 지배하지 않게)
REFIT_FREQ = "M"           # 월말 재추정

# ---------------------------------------------------------------- 전략
VT_BUFFER = 0.10           # 변동성 타기팅: 비중 변화가 10%p 미만이면 거래 안 함
S1_C_TURBULENT_WEIGHT = 0.0
S2_ENTRY_GRID = (0.20, 0.30)       # 상위 몇 %를 살지
S2_BUFFER_GRID = (0.10, 0.20)      # 보유 종목은 (entry + buffer) 안이면 유지
S2_MAX_WEIGHT = 0.05

# ---------------------------------------------------------------- 횡단면 분석
SIGNALS = ["RSJ", "RSkew", "MAX", "IVOL", "REV"]
CONTROLS = ["REV", "MAX", "IVOL", "RSkew", "SIZE", "ILLIQ", "MOM"]
MIN_STOCKS_PER_MONTH = 30
NW_LAGS_MONTHLY = 3

# S2 편입 기준 (결과 보기 전에 확정)
S2_CRITERIA = dict(
    min_abs_t=2.0,               # 중립화 IC의 NW t
    same_sign_halves=True,       # 개발 구간 전반·후반 IC 부호 일치
    min_abs_monotonic=0.8,       # 분위 번호와 분위 평균수익률의 순위상관 절댓값
    long_leg_positive=True,      # 롱 다리 - 동일가중 벤치마크 > 0
)

# ---------------------------------------------------------------- 레짐 진단용 사건
EVENTS = {
    "GFC": ("2008-09-15", "2009-03-31"),
    "EU_crisis": ("2011-08-01", "2011-12-31"),
    "China_2015": ("2015-08-10", "2016-02-29"),
    "Q4_2018": ("2018-10-01", "2018-12-31"),
    "COVID": ("2020-02-20", "2020-04-30"),
    "Tightening_2022": ("2022-01-01", "2022-10-31"),
    "Aug_2024": ("2024-08-01", "2024-08-15"),
    "KR_martial_law": ("2024-12-03", "2024-12-31"),
    "Tariff_2025": ("2025-04-01", "2025-04-30"),
}

# ---------------------------------------------------------------- 장중 수집 대상 (C층)
US_SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
INTRADAY_BASE = {
    "us": ["^GSPC", "SPY"] + US_SECTOR_ETFS,
    "kr": ["^KS11", "^KS200", "069500.KS"],
}
INTRADAY_TOP_N = 50        # 각 시장 대형주 상위 N개를 추가 수집
