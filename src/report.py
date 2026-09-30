"""결과 저장 도우미: results/<단계>/<이름>.csv"""
from __future__ import annotations

import json

import pandas as pd

from .config import RESULTS_DIR


def out_dir(stage: str):
    d = RESULTS_DIR / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(obj, stage: str, name: str, show: bool = True):
    d = out_dir(stage)
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        obj.to_csv(d / f"{name}.csv")
        if show:
            print(f"\n=== {stage}/{name} ===")
            with pd.option_context("display.width", 160, "display.max_columns", 30):
                print(obj.round(4) if hasattr(obj, "round") else obj)
    else:
        (d / f"{name}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
        if show:
            print(f"\n=== {stage}/{name} ===\n{json.dumps(obj, indent=2, ensure_ascii=False, default=str)}")
