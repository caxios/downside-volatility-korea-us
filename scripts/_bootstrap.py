"""scripts/*.py 를 프로젝트 루트에서 `python scripts/stepX.py` 로 실행할 수 있게 경로를 잡는다."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows 콘솔(cp949)에서 '—' 같은 문자를 출력하다 멈추지 않도록
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
