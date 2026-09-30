"""
1b단계: 상승/하락 반분산 → 미래 상승/하락 반분산 (H1 직접 검정)

  python scripts/step1b_semivariance.py

- 주 분석: 개발 구간 (DEV_START ~ 2020-12, 목표 창도 2020-12를 넘지 않음)
- 재현 확인: 테스트 구간 (2021-01 ~), S1 동결 후에만 (접근 기록 남음)
- 강건성: 개발 구간에서 설명변수·목표를 99.5% 분위로 절단(winsorize)

결과: results/step1b/*.csv
"""
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import DEV_START, MARKETS, TEST_START
from src.firewall import guard
from src.pipeline import load_market
from src.report import save
from src.semivar import (NEG, POS, REGRESSORS, TARGETS, coef_table, equality_tests, fit, nw_lags,
                         semivar_features, univariate_assoc, winsorize)

HORIZONS = (1, 5, 22)


def run_sample(d, tag, market):
    coefs, tests = [], []
    for h in HORIZONS:
        for s in TARGETS:
            y = f"y_{s}_h{h}"
            res = fit(d, y, REGRESSORS, nw_lags(h))
            ct = coef_table(res, d, y)
            ct.insert(0, "target", s)
            ct.insert(1, "h", h)
            ct["n"], ct["R2"] = int(res.nobs), res.rsquared
            coefs.append(ct)
            et = equality_tests(res)
            et.insert(0, "target", s)
            et.insert(1, "h", h)
            tests.append(et)
    coefs, tests = pd.concat(coefs), pd.concat(tests)
    coefs.index.name, tests.index.name = "term", "test"
    save(coefs, "step1b", f"{market}_{tag}_coef", show=False)
    save(tests, "step1b", f"{market}_{tag}_equality", show=False)
    return coefs, tests


def main():
    for market in MARKETS:
        mk = load_market(market)

        # 개발 구간: 2020-12까지의 수익률만으로 특징·목표를 만든다 → 목표 창이 2021년으로 넘어가지 않음
        d_dev = semivar_features(guard(mk["r_pct"]), HORIZONS)
        d_dev = d_dev[d_dev.index >= pd.Timestamp(DEV_START)]
        c_dev, t_dev = run_sample(d_dev, "dev", market)

        ys = [(f"y_{s}_h{h}", h) for h in HORIZONS for s in ("RSp", "RSn")]
        save(univariate_assoc(d_dev, REGRESSORS, ys, nw_lags), "step1b", f"{market}_dev_assoc", show=False)

        # 강건성: 극단값 절단
        cols = REGRESSORS + [c for c in d_dev if c.startswith("y_")]
        run_sample(winsorize(d_dev, cols), "dev_winsor", market)

        # 재현 확인: 테스트 구간 (설명변수에 2020년 말 값이 들어가는 것은 허용)
        r_all = guard(mk["r_pct"], allow_test=True, required=("S1",), context=f"step1b semivar {market}")
        d_test = semivar_features(r_all, HORIZONS)
        d_test = d_test[d_test.index >= pd.Timestamp(TEST_START)]
        c_test, t_test = run_sample(d_test, "test", market)

        # 콘솔 요약: 5일 목표, 하락-상승 합계 차이
        for tag, t in (("dev", t_dev), ("test", t_test)):
            s = t[t.index == "합계: Σβ⁻ − Σβ⁺"].set_index("target")
            s = s[s["h"] == 5][["diff", "stat", "p"]].round(4)
            print(f"\n[{market} {tag}] h=5 총영향 차이 (β⁻ − β⁺)\n{s.to_string()}")


if __name__ == "__main__":
    main()
