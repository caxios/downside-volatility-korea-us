"""
1단계: 지수 일봉 품질 점검 + HAR vs LHAR 표본 외 비교 (H1의 첫 결과)

  python scripts/step1_index_har.py            # 개발 구간(~2020)만
  python scripts/step1_index_har.py --refresh  # 데이터 다시 받기

결과: results/step1/*.csv
"""
import argparse

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.config import DEV_START, HAR_MIN_TRAIN, HAR_REFIT_EVERY, MARKETS
from src.firewall import guard
from src.har import compare_models, full_sample_fit, har_features, lookahead_sanity, spec, walk_forward
from src.pipeline import load_market
from src.quality import quality_report
from src.report import save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    q_rows = []
    for market in MARKETS:
        mk = load_market(market, refresh=args.refresh)
        q_rows += [quality_report(mk["idx"], mk["cfg"]["index"]), quality_report(mk["etf"], mk["cfg"]["etf"])]

        d = har_features(mk["rv"], mk["r_pct"])
        d = guard(d).copy()                             # 2020년까지만
        d.loc[d.index[-1], "y"] = np.nan                # 경계 하루(2021년 첫날 RV)도 쓰지 않음

        preds = {}
        for name in ("HAR", "LHAR"):
            preds[name] = walk_forward(d, spec(name, log=True), HAR_MIN_TRAIN, HAR_REFIT_EVERY, log_target=True)
            preds[f"{name}_level"] = walk_forward(d, spec(name, log=False), HAR_MIN_TRAIN, HAR_REFIT_EVERY)
        eval_idx = d.index[d.index >= pd.Timestamp(DEV_START)]
        cmp_ = compare_models(d.loc[eval_idx], {k: v.loc[eval_idx] for k, v in preds.items()}, bench="HAR")
        save(cmp_, "step1", f"{market}_har_vs_lhar_oos")

        save(full_sample_fit(d, spec("LHAR", log=True), log_target=True), "step1", f"{market}_lhar_coef_dev")

        cheat = lookahead_sanity(d, spec("HAR", log=True), min_train=HAR_MIN_TRAIN,
                                 refit_every=HAR_REFIT_EVERY, log_target=True)
        ok = cheat < 0.5 * cmp_.loc["HAR", "QLIKE"]
        print(f"[{market}] 누수 감지 점검: 미래값 투입 시 QLIKE={cheat:.4f} "
              f"(정상 HAR={cmp_.loc['HAR', 'QLIKE']:.4f}) → {'정상' if ok else '이상: 파이프라인 확인 필요'}")

    save(pd.DataFrame(q_rows).set_index("name"), "step1", "quality_report")


if __name__ == "__main__":
    main()
