# Replication package

Code and results for *Downside Volatility Predicts Risk, Not Returns: Out-of-Sample Evidence from Korea and the U.S.*

The comments in the code are in Korean. This file is the English guide.

## What is included

| Path | Content |
|---|---|
| `src/`, `scripts/` | Data download, variable construction, every analysis, rule freezing |
| `config/frozen/*.json` | The five frozen rule sets (S1, S2S3, S4, S5, S1v2), each with its freezing time and SHA-256 hash |
| `logs/test_access.log` | Every access to test-period data (2021 onward) that passed the software firewall |
| `logs/paper_*.csv` | Paper-trading records of the frozen allocation rules |
| `results/` | All result tables (CSV) that the paper's tables and figures are built from |
| `paper/` | LaTeX source; `paper/build_assets.py` rebuilds every table and figure from `results/` |
| `tests/` | Synthetic-data tests of the estimators and the firewall (`pytest tests`) |

## What is not included

Raw and processed price data (`data/`) are not redistributed.

- Yahoo Finance (via `yfinance`): indices, ETFs, U.S. stocks, intraday bars.
- Korea Exchange Open API (<https://openapi.krx.co.kr>): Korean stock prices and market capitalization. Non-commercial use only; redistribution is not permitted. You need your own authentication key.
- Kenneth French's data library (factors) and FRED (T-bill rate): downloaded by the scripts.
- `sp500_history.csv`: historical S&P 500 constituents, from <https://github.com/hanshof/sp500_constituents> (`scripts/get_sp500_history.py`).

Yahoo Finance revises adjusted prices over time, and intraday bars are only available for recent dates. A fresh download will therefore not match our data exactly, and the intraday results (Appendix C of the paper) cannot be re-created for the original dates.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then put your KRX Open API key in .env
```

Tested with Python 3.10 on Windows.

## Checking the frozen rules

```bash
python scripts/verify_frozen.py
```

This recomputes each rule set's hash and lists the test-period accesses. The frozen files and the log were stored locally and have no external timestamp. Their publication in this repository is the first external record. The paper therefore describes the procedure as *internal freezing*, not pre-registration.

`src/firewall.py` refuses to overwrite a rule set that is already frozen. Running a script with `--freeze` therefore raises an error instead of changing the published rules.

## Reproducing the results

Run the commands from the project root, in this order. Each script writes to `results/stepN/`.

```bash
# data
python scripts/get_sp500_history.py
python scripts/step3_build_universe.py --market US --sp500-csv sp500_history.csv
python scripts/step3_build_universe.py --market KR
python scripts/step0_collect_intraday.py              # recent intraday bars only (see above)

# volatility (Section 4)
python scripts/step1_index_har.py
python scripts/step1b_semivariance.py
python scripts/step2_regime_s1.py
python scripts/step2_regime_s1.py --final-test

# cross-section (Section 5) and short-sale ban (Section 7)
python scripts/step4_cross_section.py
python scripts/step5_h4_h5.py --phase pre
python scripts/step6_s2_s3.py --phase test
python scripts/step5_h4_h5.py --phase placebo-clean
python scripts/step5_h4_h5.py --phase post
python scripts/step10_realized_skew.py

# direction and stock-level semivariance forecasts (Section 6)
python scripts/step7c_direction.py
python scripts/step9_s4_semivar_strategy.py --phase design
python scripts/step9_s4_semivar_strategy.py --phase test

# allocation (Section 8), exploratory intraday (Appendix C)
python scripts/step12_s1v2_vol_timing.py --phase design
python scripts/step12_s1v2_vol_timing.py --phase test
python scripts/step7_intraday_validation.py
python scripts/step7b_intraday_semivar.py

# multiple testing, short-sale-ban robustness (Appendices B and D)
python scripts/step11_multiple_testing.py
python scripts/step13_ban_robustness.py

# tables and figures
python paper/build_assets.py
python paper/check_tex.py
```

Every run that reads test-period data appends a line to `logs/test_access.log`. The published log records our original runs, so run the package on a copy if you want to keep it unchanged.

## Where each table and figure comes from

| Paper | Results |
|---|---|
| Sample summary | `step3`, `step4/*_T1_descriptive.csv` |
| Forecast comparison, LHAR coefficients | `step1`, `step2` |
| Semivariance total effects (figure) | `step1b`, `step11/REPORT_semivariance_*` |
| RSJ | `step4` |
| Weekly realized skewness (decile figure) | `step10`, `step11/REPORT_S5_*` |
| Direction | `step7c`, `step11/REPORT_direction_*` |
| Stock-level forecasts (S4) | `step9`, `step11/REPORT_S4_*` |
| Short-sale ban (placebo figure) | `step5` |
| Allocation, turbulent days | `step12`, `step2` |
| Appendix B (multiple testing) | `step11` |
| Appendix C (intraday) | `step7b` |
| Appendix D (robustness) | `step2`, `step1b`, `step10`, `step13` |

## How the code was written

The code and the analyses were produced with an AI coding assistant (Claude Code, Anthropic). See the declaration of generative AI use at the end of the paper for the division of work between the author and the assistant.
