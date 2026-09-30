"""
개발/테스트 방화벽.

- 규칙을 동결(freeze)하기 전에는 TEST_START 이후 데이터로 어떤 결과도 계산하지 않는다.
- 동결은 JSON + SHA-256 + (가능하면) git 태그로 기록된다. 같은 이름으로 두 번 동결할 수 없다.
- 테스트 구간을 열 때마다 logs/test_access.log 에 남긴다.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime

import pandas as pd

from .config import FROZEN_DIR, LOG_DIR, ROOT, TEST_START


class FirewallError(RuntimeError):
    pass


def _sha(rules: dict) -> str:
    return hashlib.sha256(json.dumps(rules, sort_keys=True, default=str).encode()).hexdigest()


def freeze(name: str, rules: dict) -> dict:
    path = FROZEN_DIR / f"{name}.json"
    if path.exists():
        raise FirewallError(f"'{name}'은 이미 동결됨. 규칙을 바꾸려면 새 이름(예: {name}_v2)으로 동결하고 "
                            "과거 테스트 구간이 아니라 페이퍼 트레이딩으로 검증할 것.")
    payload = {"name": name, "frozen_at": datetime.now().isoformat(timespec="seconds"),
               "sha256": _sha(rules), "rules": rules}
    # UTF-8로, 임시 파일에 먼저 쓰고 이름을 바꾼다 (쓰기 도중 실패해도 빈 동결 파일이 남지 않게)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)
    try:
        subprocess.run(["git", "add", str(path)], cwd=ROOT, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", f"freeze {name}"], cwd=ROOT, check=True, capture_output=True)
        subprocess.run(["git", "tag", f"frozen-{name}"], cwd=ROOT, check=True, capture_output=True)
    except Exception:
        pass   # git 저장소가 아니어도 JSON 기록은 남는다
    return payload


def load_frozen(name: str) -> dict:
    path = FROZEN_DIR / f"{name}.json"
    if not path.exists():
        raise FirewallError(f"동결된 규칙 '{name}'이 없음")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if _sha(payload["rules"]) != payload["sha256"]:
        raise FirewallError(f"'{name}' 동결 파일이 수정됨 (해시 불일치)")
    return payload


def is_frozen(name: str) -> bool:
    return (FROZEN_DIR / f"{name}.json").exists()


def log_access(context: str, required: tuple):
    with open(LOG_DIR / "test_access.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{context}\trequired={','.join(required)}\n")


def guard(obj, allow_test: bool = False, required: tuple = (), context: str = "", month_col: str | None = None):
    """
    allow_test=False → TEST_START 이전만 반환.
    allow_test=True  → required 규칙이 모두 동결돼 있어야 하고, 접근 기록을 남긴다.
    obj: DatetimeIndex/PeriodIndex를 가진 Series/DataFrame, 또는 month_col이 있는 DataFrame
    """
    if allow_test:
        missing = [r for r in required if not is_frozen(r)]
        if missing:
            raise FirewallError(f"테스트 구간 접근 거부: 먼저 동결할 것 → {missing}")
        log_access(context, required)
        return obj
    cut = pd.Timestamp(TEST_START)
    if month_col is not None:
        return obj[obj[month_col] < pd.Period(cut, "M")]
    idx = obj.index
    if isinstance(idx, pd.PeriodIndex):
        return obj[idx < pd.Period(cut, "M")]
    return obj[idx < cut]
