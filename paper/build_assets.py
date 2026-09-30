"""
Build LaTeX tables and PDF figures for the paper directly from results/*.csv.

    python paper/build_assets.py

Outputs: paper/tables/*.tex, paper/figures/*.pdf
Re-run after any change to results/ so the paper never carries hand-copied numbers
(except the few noted as "from report text" below).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R = ROOT / "results"
TAB = Path(__file__).resolve().parent / "tables"
FIG = Path(__file__).resolve().parent / "figures"
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
MK = ("US", "KR")
MKN = {"US": "U.S.", "KR": "Korea"}


# ---------------------------------------------------------------- helpers
def f(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return f"{x:.{d}f}".replace("-", "$-$")


def ft(t, d=2):
    return "" if t is None or not np.isfinite(t) else f"({f(t, d)})"


def stars(p):
    if p is None or not np.isfinite(p):
        return ""
    return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


def dag(holm_p):
    return "$^{\\dagger}$" if holm_p is not None and np.isfinite(holm_p) and holm_p < 0.05 else ""


def write(name, body):
    (TAB / f"{name}.tex").write_text(body, encoding="utf-8")
    print("wrote", name)


def adj(report):
    return pd.read_csv(R / "step11" / f"{report}_adjusted.csv", index_col=0)


# ---------------------------------------------------------------- T2 forecasting
def t_forecast():
    lines = []
    for m in MK:
        a = pd.read_csv(R / "step1" / f"{m}_har_vs_lhar_oos.csv", index_col=0)
        b = pd.read_csv(R / "step2" / f"{m}_regime_lhar_oos.csv", index_col=0)
        c = pd.read_csv(R / "step2" / f"{m}_regime_lhar_oos_vs_LHAR.csv", index_col=0)
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{MKN[m]}}}}}\\\\")
        lines.append(f"\\quad Sample A: HAR & {int(a.loc['HAR','n_oos']):,} & {f(a.loc['HAR','QLIKE'],4)} & -- & -- \\\\")
        lines.append(f"\\quad Sample A: LHAR & {int(a.loc['LHAR','n_oos']):,} & {f(a.loc['LHAR','QLIKE'],4)} & "
                     f"{f(a.loc['LHAR','QLIKE_improve_pct_vs_bench'],2)} & {f(a.loc['LHAR','DM_t_vs_bench'],2)} \\\\")
        for mod in ("HAR", "LHAR", "RLHAR"):
            imp = "--" if mod == "HAR" else f(b.loc[mod, "QLIKE_improve_pct_vs_bench"], 2)
            dm = "--" if mod == "HAR" else f(b.loc[mod, "DM_t_vs_bench"], 2)
            lines.append(f"\\quad Sample B: {mod} & {int(b.loc[mod,'n_oos']):,} & {f(b.loc[mod,'QLIKE'],4)} & {imp} & {dm} \\\\")
        lines.append(f"\\quad Sample B: RLHAR vs.\\ LHAR & & & {f(c.loc['RLHAR','QLIKE_improve_pct_vs_bench'],2)} & "
                     f"{f(c.loc['RLHAR','DM_t_vs_bench'],2)} \\\\")
        if m == "US":
            lines.append("\\addlinespace")
    d = adj("REPORT_main").set_index("test")
    holm = {k: d.loc[k, "p_holm"] for k in d.index if "(DM)" in k}
    note = ("Holm-adjusted $p$ (family H1, six tests): LHAR vs.\\ HAR U.S.\\ "
            f"{holm['US LHAR vs HAR (DM)']:.3f}, Korea {holm['KR LHAR vs HAR (DM)']:.3f}; "
            f"RLHAR vs.\\ LHAR U.S.\\ {holm['US RLHAR vs LHAR (DM)']:.3f}, Korea {holm['KR RLHAR vs LHAR (DM)']:.3f}.")
    write("t_forecast", "\n".join(lines) + f"\n%NOTE {note}\n")
    (TAB / "t_forecast_note.tex").write_text(note, encoding="utf-8")


# ---------------------------------------------------------------- T3 coefficients
def t_coef():
    names = {"const": "Constant", "lRV": "$\\log RV_t$", "lRV_w": "$\\log RV^{(w)}_t$", "lRV_m": "$\\log RV^{(m)}_t$",
             "L": "$L_t$", "L_w": "$L^{(w)}_t$", "L_m": "$L^{(m)}_t$", "D": "$D_t$ (turbulent)",
             "lRV_xD": "$\\log RV_t \\times D_t$", "L_xD": "$L_t \\times D_t$"}
    cols = {}
    for m in MK:
        cols[(m, "LHAR")] = pd.read_csv(R / "step1" / f"{m}_lhar_coef_dev.csv", index_col=0)
        cols[(m, "RLHAR")] = pd.read_csv(R / "step2" / f"{m}_regime_lhar_coef.csv", index_col=0)
    lines = []
    for k, lab in names.items():
        row = [lab]
        for m in MK:
            for mod in ("LHAR", "RLHAR"):
                c = cols[(m, mod)]
                row.append(f"{f(c.loc[k,'coef'])}{stars(c.loc[k,'p'])} {ft(c.loc[k,'t_NW'])}" if k in c.index else "")
        lines.append(", ".join(row) + " \\\\")
    write("t_coef", "\n".join(lines) + "\n")


# ---------------------------------------------------------------- T4 semivariance
def t_semivar():
    d = adj("REPORT_semivariance")
    out = []
    tgt = {"RSp": "$RS^{+}$", "RSn": "$RS^{-}$", "RV": "$RV$"}
    for smp, lab in (("dev", "Panel A: Development sample (2005--2020)"), ("test", "Panel B: Test sample (2021--2026)")):
        out.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{lab}}}}}\\\\")
        for t_, tl in tgt.items():
            for h in (1, 5, 22):
                row = [f"{tl}, $h={h}$"]
                for m in MK:
                    x = d[d.family == f"{m}_{smp}"]
                    for key in ("상승 총영향", "하락 총영향", "합계"):
                        r = x[x.test.str.startswith(key) & x.test.str.contains(f"미래 {t_} h={h}$", regex=True)]
                        r = r.iloc[0]
                        row.append(f"{f(r.estimate,2)}{dag(r.p_holm)} {ft(r.t,1)}")
                out.append(", ".join(row) + " \\\\")
        if smp == "dev":
            out.append("\\addlinespace")
    write("t_semivar", "\n".join(out) + "\n")
    s = pd.read_csv(R / "step11" / "REPORT_semivariance_summary.csv", index_col=0)
    write("t_semivar_counts", ", ".join(
        f"{int(s.loc[k,'Holm<0.05'])}/{int(s.loc[k,'m'])}" for k in ("US_dev", "KR_dev", "US_test", "KR_test")))


# ---------------------------------------------------------------- T5 RSJ vs reversal
def t_rsj():
    rows = {k: [] for k in ("rc_rev", "rc_rskew", "ic_raw", "ic_neu", "fm1", "fm2", "fm6", "ds_rev", "ds_ivol")}
    for m in MK:
        rc = pd.read_csv(R / "step4" / f"{m}_T2_rank_corr.csv", index_col=0)
        ic = pd.read_csv(R / "step4" / f"{m}_T3_ic_summary.csv", index_col=0)
        nu = pd.read_csv(R / "step4" / f"{m}_T8_neutralized_ic.csv", index_col=0)
        fm = pd.read_csv(R / "step4" / f"{m}_T6_fama_macbeth_RSJ.csv", index_col=0)
        ds = pd.read_csv(R / "step4" / f"{m}_T7_double_sort.csv", index_col=0)
        rows["rc_rev"].append(f(rc.loc["RSJ", "REV"], 2))
        rows["rc_rskew"].append(f(rc.loc["RSJ", "RSkew"], 2))
        rows["ic_raw"].append(f"{f(ic.loc['RSJ','mean_IC'])} {ft(ic.loc['RSJ','t_NW'])}")
        rows["ic_neu"].append(f"{f(nu.loc['RSJ','neu_mean_IC'])} {ft(nu.loc['RSJ','neu_t_NW'])}")

        def fmcell(s):
            c, t = s.split(" (")
            return f"{f(float(c)*100, 2)} ({f(float(t.rstrip(')')), 2)})"
        rows["fm1"].append(fmcell(fm.loc["RSJ", "(1)"]))
        rows["fm2"].append(fmcell(fm.loc["RSJ", "(2)"]))
        rows["fm6"].append(fmcell(fm.loc["RSJ", "(6)"]))
        rows["ds_rev"].append(f"{f(ds.loc['RSJ | REV','LS_mean']*100,2)} {ft(ds.loc['RSJ | REV','t_NW'])}")
        rows["ds_ivol"].append(f"{f(ds.loc['RSJ | IVOL','LS_mean']*100,2)} {ft(ds.loc['RSJ | IVOL','t_NW'])}")
    lab = {"rc_rev": "Rank correlation, RSJ vs.\\ REV", "rc_rskew": "Rank correlation, RSJ vs.\\ RSkew",
           "ic_raw": "Rank IC of RSJ, raw", "ic_neu": "Rank IC of RSJ, neutralized",
           "fm1": "FM coefficient on RSJ, RSJ only (\\%)", "fm2": "FM coefficient on RSJ, + REV (\\%)",
           "fm6": "FM coefficient on RSJ, all controls (\\%)",
           "ds_rev": "Double sort: RSJ L--S within REV (\\%/month)", "ds_ivol": "Double sort: RSJ L--S within IVOL (\\%/month)"}
    write("t_rsj", "\n".join(f"{lab[k]} & {' & '.join(v)} \\\\" for k, v in rows.items()) + "\n")


# ---------------------------------------------------------------- T6 weekly skewness
def t_skew():
    d = adj("REPORT_S5").set_index("test")
    legs = [("LS_EW_gross", "Long--short, EW, gross"), ("LS_VW_gross", "Long--short, VW, gross"),
            ("LS_EW_net", "Long--short, EW, net"), ("LONG_EW_net", "Long-only D1, EW, net"),
            ("UNIV_EW", "Universe, EW")]
    out = ["\\multicolumn{5}{l}{\\textit{Panel A: Portfolio returns (bps/week)}}\\\\"]
    for leg, lab in legs:
        row = [lab]
        for m in MK:
            for smp in ("dev", "test"):
                s = pd.read_csv(R / "step10" / f"{m}_{smp}_week_summary.csv", index_col=0)
                r = s.loc[leg]
                key = f"{m} week {leg}"
                h = None
                if key in d.index:
                    sub = d[(d.index == key) & (d.family == smp)]
                    h = sub.p_holm.iloc[0] if len(sub) else None
                row.append(f"{f(r['mean_weekly_bps'],1)}{dag(h)} {ft(r['t_NW'])}")
        out.append(", ".join(row) + " \\\\")
    out.append("\\addlinespace")
    out.append("\\multicolumn{5}{l}{\\textit{Panel B: Fama--MacBeth coefficient on skewness (bps per s.d.)}}\\\\")
    for kind, lab in (("fm_univariate", "Skewness only"), ("fm", "With REV, RVOL, SIZE")):
        row = [lab]
        for m in MK:
            for smp in ("dev", "test"):
                s = pd.read_csv(R / "step10" / f"{m}_{smp}_week_{kind}.csv", index_col=0)
                nm = "FM 왜도(단독)" if kind == "fm_univariate" else "FM 왜도(통제)"
                sub = d[(d.index == f"{m} week {nm}") & (d.family == smp)]
                row.append(f"{f(s.loc['RSK','coef_bps_per_sd'],2)}{dag(sub.p_holm.iloc[0])} {ft(s.loc['RSK','t_NW'])}")
        out.append(", ".join(row) + " \\\\")
    row = ["REV coefficient (with controls)"]
    for m in MK:
        for smp in ("dev", "test"):
            s = pd.read_csv(R / "step10" / f"{m}_{smp}_week_fm.csv", index_col=0)
            row.append(f"{f(s.loc['REV','coef_bps_per_sd'],2)} {ft(s.loc['REV','t_NW'])}")
    out.append(", ".join(row) + " \\\\")
    write("t_skew", "\n".join(out) + "\n")


def f_deciles():
    import matplotlib.pyplot as plt
    setup_mpl(plt)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=False)
    for ax, m in zip(axes, MK):
        for smp, col, lab in (("dev", C1, "Development"), ("test", C2, "Test")):
            dd = pd.read_csv(R / "step10" / f"{m}_{smp}_week_deciles.csv", index_col=[0, 1])
            y = dd.loc["EW", "mean_weekly_bps"].to_numpy()
            ax.plot(range(1, 11), y, marker="o", ms=4, lw=1.5, color=col, label=lab)
        ax.axhline(0, color=GRID, lw=0.8)
        ax.set_title(MKN[m], fontsize=9)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Skewness decile (1 = lowest)")
    axes[0].set_ylabel("Next-week return (bps)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_skew_deciles.pdf"); fig.savefig(FIG / "fig_skew_deciles.png", dpi=150)
    plt.close(fig)
    print("wrote fig_skew_deciles")


# ---------------------------------------------------------------- T7 direction
def t_direction():
    s = pd.read_csv(R / "step11" / "REPORT_direction_summary.csv", index_col=0)
    lab = {"beta(반전·지속)": "Persistence coefficient $\\beta$", "적중률 vs 기준선": "Hit rate: reversal rule $-$ baseline",
           "레짐 차이": "Turbulent $-$ calm difference in $\\beta$", "큰 하락 다음": "After large down days"}
    out = []
    for k, l in lab.items():
        r = s.loc[k]
        out.append(f"{l} & {int(r['m'])} & {int(r['raw_p<0.05'])} & {int(r['Holm<0.05'])} & {int(r['BH<0.05'])} & {int(r['|t|>3'])} \\\\")
    write("t_direction", "\n".join(out) + "\n")
    di = pd.read_csv(R / "step7c" / "daily_index.csv", index_col=0)
    r = di[(di.market == "US") & (di["sample"] == "dev") & (di.k == 1) & (di.h == 1)].iloc[0]
    rt = di[(di.market == "US") & (di["sample"] == "test") & (di.k == 1) & (di.h == 1)].iloc[0]
    macro = (f"\\newcommand{{\\dirBeta}}{{{r.beta:.3f}}}\\newcommand{{\\dirHit}}{{{100*r.hit_reversal:.1f}}}"
             f"\\newcommand{{\\dirBase}}{{{100*r.hit_baseline:.1f}}}\\newcommand{{\\dirBetaTest}}{{{rt.beta:.3f}}}"
             f"\\newcommand{{\\dirPTest}}{{{rt.p:.3f}}}")
    write("m_direction", macro.replace("-", "$-$") if False else macro)


# ---------------------------------------------------------------- T8 S4
def t_s4():
    out = []
    lab = {"RSp": "Predicted vs.\\ realized $RS^{+}$", "RSn": "Predicted vs.\\ realized $RS^{-}$",
           "share": "Predicted vs.\\ realized upside share", "pRSp_vs_pRSn": "Predicted $RS^{+}$ vs.\\ predicted $RS^{-}$"}
    for k, l in lab.items():
        row = [l]
        for m in MK:
            for smp in ("dev", "test"):
                q = pd.read_csv(R / "step9" / f"{m}_{smp}_forecast_quality.csv", index_col=0)
                row.append(f(q.loc[k, "mean_spearman"]) + ("" if k == "pRSp_vs_pRSn" else f" {ft(q.loc[k,'t_NW'],1)}"))
        out.append(", ".join(row) + " \\\\")
    write("t_s4_forecast", "\n".join(out) + "\n")
    d = adj("REPORT_S4")
    ex = pd.read_csv(R / "step9" / "excess_return_tests.csv")
    frozen = {"US": "S4_20_rel", "KR": "S4_30_abs"}
    out = []
    for m in MK:
        for vs, vl in (("EW_universe", "equal-weighted universe"), ("ETF", "index ETF")):
            row = [f"{MKN[m]} S4 $-$ {vl}"]
            for smp in ("dev", "test"):
                r = ex[(ex.market == m) & (ex["sample"] == smp) & (ex.strategy == frozen[m]) & (ex.vs == vs)].iloc[0]
                a = d[(d.family == f"초과수익_{smp}") & (d.test == f"{m} {frozen[m]} − {vs}")].iloc[0]
                row.append(f"{f(r.ann_diff_pct,2)} {ft(r.t_NW)} & {f(a.p_holm,2)}")
            out.append(", ".join(row) + " \\\\")
    write("t_s4_excess", "\n".join(out) + "\n")


# ---------------------------------------------------------------- T9 short-sale ban
def t_ban():
    real = pd.read_csv(R / "step5" / "H4_country_did_REAL.csv", index_col=0)
    exe = pd.read_csv(R / "step5" / "H4_country_did_excl_events.csv", index_col=0)
    cl = pd.read_csv(R / "step5" / "H4_placebo_clean_summary.csv", index_col=0)
    out = []
    for s in ("RSJ", "RSkew", "IVOL", "MAX"):
        out.append(f"{s} & {f(real.loc[s,'theta3_Ban'])} {ft(real.loc[s,'t_NW'])} & {f(real.loc[s,'placebo_p'],2)} & "
                   f"{f(cl.loc[s,'placebo_p_clean_dist'],2)} & {f(cl.loc[s,'theta3_excl_prior_bans'])} {ft(cl.loc[s,'t_excl_prior_bans'])} & "
                   f"{f(cl.loc[s,'placebo_p_excl_prior_bans'],2)} & {f(exe.loc[s,'theta3_excl_events'])} {ft(exe.loc[s,'t_NW'])} \\\\")
    write("t_ban", "\n".join(out) + "\n")
    n_old = int(cl["n_placebo_original"].iloc[0])
    n_new = int(cl["n_placebo_clean"].iloc[0])
    write("m_ban", f"\\newcommand{{\\nPlaceboOld}}{{{n_old}}}\\newcommand{{\\nPlaceboNew}}{{{n_new}}}")


def f_placebo():
    import matplotlib.pyplot as plt
    setup_mpl(plt)
    dist = pd.read_csv(R / "step5" / "H4_placebo_distribution_clean.csv", index_col=0)
    cl = pd.read_csv(R / "step5" / "H4_placebo_clean_summary.csv", index_col=0)
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.0), sharey=True)
    for ax, s in zip(axes, ("RSJ", "RSkew", "IVOL", "MAX")):
        x = dist[s].dropna()
        ax.hist(x, bins=15, color=C1, alpha=0.85, edgecolor="white", linewidth=0.8)
        ax.axvline(cl.loc[s, "theta3_excl_prior_bans"], color=C2, lw=2)
        ax.set_title(s, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Placebo windows")
    from matplotlib.ticker import MaxNLocator
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.text(0.5, 0.01, "Placebo ban coefficient (KR $-$ U.S. FM slope)", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "fig_placebo.pdf"); fig.savefig(FIG / "fig_placebo.png", dpi=150)
    plt.close(fig)
    print("wrote fig_placebo")


# ---------------------------------------------------------------- T10 multiple testing
def t_mt():
    rows = [
        ("Main hypotheses (H1--H5)", "REPORT_main"),
        ("Daily semivariance", "REPORT_semivariance"),
        ("Intraday semivariance", "REPORT_intraday_semivariance"),
        ("Direction", "REPORT_direction"),
        ("Semivariance stock selection (S4)", "REPORT_S4"),
        ("Weekly skewness (S5)", "REPORT_S5"),
    ]
    out = []
    for lab, k in rows:
        s = pd.read_csv(R / "step11" / f"{k}_summary.csv", index_col=0)
        out.append(f"{lab} & {len(s)} & {int(s['m'].sum())} & {int(s['raw_p<0.05'].sum())} & "
                   f"{int(s['Holm<0.05'].sum())} & {int(s['BH<0.05'].sum())} & {int(s['|t|>3'].sum())} \\\\")
    write("t_mt", "\n".join(out) + "\n")


# ---------------------------------------------------------------- appendix: overnight
def t_overnight():
    d = adj("REPORT_intraday_semivariance")
    x = d[d.family == "60m_decomp_index_US"]
    out = []
    for blk, bl in (("장중", "Intraday"), ("밤사이", "Overnight")):
        for h in (1, 5, 22):
            r = x[x.test.str.contains(f"\\[{blk}\\] 합계", regex=True) & x.test.str.endswith(f"미래 RV h={h}")].iloc[0]
            out.append(f"{bl} & {h} & {f(r.estimate,2)} & {f(r.t,2)} & {f(r.p,3)} & {f(r.p_holm,2)} & {f(r.p_bh,3)} \\\\")
    write("t_overnight", "\n".join(out) + "\n")


# ---------------------------------------------------------------- appendix: timing strategies
def t_timing():
    out = []
    for m in MK:
        dev = pd.read_csv(R / "step12" / f"{m}_dev_performance.csv", index_col=0)
        tst = pd.read_csv(R / "step12" / f"{m}_TEST_performance.csv", index_col=0)
        rule = json.loads((ROOT / "config" / "frozen" / "S1v2.json").read_text(encoding="utf-8"))["rules"][m]["candidate"]
        out.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{MKN[m]} (S1v2 = {rule.replace('_', chr(92)+'_')})}}}}\\\\")
        for k, lab in (("S1_A(동결)", "S1\\_A regime switch (frozen)"), (rule, "S1v2 vol.\\ targeting (frozen)"),
                       ("naiveVT", "Naive vol.\\ targeting (22-day)"), ("MA200", "200-day moving average"),
                       ("BH", "Buy and hold")):
            a, b = dev.loc[k], tst.loc[k]
            out.append(f"\\quad {lab} & {f(a.sharpe,2)} & {f(100*a.max_dd,1)} & {f(100*b.ann_ret,1)} & {f(b.sharpe,2)} & "
                       f"{f(b.sharpe_se,2)} & {f(100*b.max_dd,1)} \\\\")
        if m == "US":
            out.append("\\addlinespace")
    write("t_timing", "\n".join(out) + "\n")


def t_turbulent_days():
    from src.pipeline import load_market, load_series
    out = []
    for m in MK:
        r = load_market(m)["etf_ret"].dropna()
        st = load_series(R / "step2" / f"{m}_regime_full.parquet").shift(1).reindex(r.index)
        for a, b, lab in (("2005", "2020", "2005--2020"), ("2021", "2026", "2021--2026")):
            x, s = r.loc[a:b], st.loc[a:b]
            t, c = x[s == 1], x[s == 0]
            out.append(f"{MKN[m]} & {lab} & {100*len(t)/len(x):.0f} & {f(100*252*t.mean(),1)} & {f(100*252*c.mean(),1)} & "
                       f"{f(100*((1+t).prod()-1),0)} \\\\")
    write("t_turbulent", "\n".join(out) + "\n")


# ---------------------------------------------------------------- figure: semivariance total effects
C1, C2, C3, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b3"


def setup_mpl(plt):
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "pdf.fonttype": 42})


def f_semivar():
    import matplotlib.pyplot as plt
    setup_mpl(plt)
    d = adj("REPORT_semivariance")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
    labels = []
    for ax, m in zip(axes, MK):
        x = d[d.family == f"{m}_dev"]
        pos = 0
        ticks, tl = [], []
        for t_, tn in (("RSp", "RS$^+$"), ("RSn", "RS$^-$"), ("RV", "RV")):
            for h in (1, 5, 22):
                for key, col, off, lab in (("상승 총영향", C1, -0.15, "Past upside $\\Sigma\\beta^{+}$"),
                                           ("하락 총영향", C2, 0.15, "Past downside $\\Sigma\\beta^{-}$")):
                    r = x[x.test.str.startswith(key) & x.test.str.endswith(f"미래 {t_} h={h}")].iloc[0]
                    se = abs(r.estimate / r.t) if r.t else np.nan
                    ax.errorbar(pos + off, r.estimate, yerr=1.96 * se, fmt="o", ms=4, color=col, lw=1.2, capsize=0,
                                label=lab if (pos == 0) else None)
                ticks.append(pos)
                tl.append(f"{tn}\n{h}d")
                pos += 1
            pos += 0.5
        ax.axhline(0, color=GRID, lw=0.8)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tl, fontsize=6.5)
        ax.set_title(f"{MKN[m]}, 2005--2020".replace("--", "–"), fontsize=9)
    axes[0].set_ylabel("Total effect on future semivariance")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "fig_semivar.pdf"); fig.savefig(FIG / "fig_semivar.png", dpi=150)
    plt.close(fig)
    print("wrote fig_semivar")


def main():
    for fn in (t_forecast, t_coef, t_semivar, t_rsj, t_skew, t_direction, t_s4, t_ban, t_mt, t_overnight, t_timing,
               t_turbulent_days, f_semivar, f_deciles, f_placebo):
        fn()


# ================================================================ added: sample summary & robustness
def t_sample():
    from src.pipeline import load_market, load_panel
    out = []
    # Panel A: index daily log returns
    stats = {}
    for m in MK:
        r = load_market(m)["r_pct"].dropna() / 100.0
        for a, b, lab in (("2005", "2020", "dev"), ("2021", "2026", "test")):
            x = r.loc[a:b]
            stats[(m, lab)] = {"n": len(x), "ret": 100 * 252 * x.mean(), "vol": 100 * np.sqrt(252) * x.std(),
                               "skew": x.skew(), "kurt": x.kurt()}
    rowsA = [("Trading days", "n", 0), ("Mean return (\% p.a.)", "ret", 1), ("Volatility (\% p.a.)", "vol", 1),
             ("Skewness", "skew", 2), ("Excess kurtosis", "kurt", 1)]
    out.append(r"\multicolumn{5}{l}{\textit{Panel A: Index daily log returns}}\\")
    for lab, k, d in rowsA:
        cells = []
        for m in MK:
            for smp in ("dev", "test"):
                v = stats[(m, smp)][k]
                cells.append(f"{int(v):,}" if k == "n" else f(v, d))
        out.append(f"{lab} & " + ", ".join(cells) + " \\\\")
    out.append(r"\addlinespace")
    # Panel B: cross-sectional panels
    out.append(r"\multicolumn{5}{l}{\textit{Panel B: Stock panels (cross-sectional sample from 2010)}}\\")
    cnt = {}
    for m in MK:
        p = load_panel(ROOT / "data" / "processed" / f"panel_{m}.parquet")
        p = p[p["month"] >= pd.Period("2010-01", "M")]
        p = p.dropna(subset=["RSJ", "fwd_ret_1"])
        dev = p[p["month"] <= pd.Period("2020-12", "M")]
        tst = p[p["month"] >= pd.Period("2021-01", "M")]
        mm = pd.read_csv(R / "step3" / f"{m}_missing_by_month.csv")
        mm_dev = mm[(mm.month >= "2010-01") & (mm.month <= "2020-12")]
        mm_tst = mm[mm.month >= "2021-01"]
        cnt[m] = {"months": (dev["month"].nunique(), tst["month"].nunique()),
                  "stocks": (dev.groupby("month").size().mean(), tst.groupby("month").size().mean()),
                  "obs": (len(dev), len(tst)),
                  "miss": (100 * mm_dev.missing_share.mean(), 100 * mm_tst.missing_share.mean()),
                  "missmax": (100 * mm_dev.missing_share.max(), 100 * mm_tst.missing_share.max())}
    for lab, k, d in (("Months", "months", 0), ("Stocks per month (average)", "stocks", 0), ("Stock-months", "obs", 0),
                      ("Constituents missing (\%, mean)", "miss", 1), ("Constituents missing (\%, max)", "missmax", 1)):
        cells = []
        for m in MK:
            for v in cnt[m][k]:
                cells.append(f"{int(round(v)):,}" if d == 0 else f(v, d))
        out.append(f"{lab} & " + ", ".join(cells) + " \\\\")
    out.append(r"\addlinespace")
    # Panel C: signal descriptives (development sample, from step 4)
    out.append(r"\multicolumn{5}{l}{\textit{Panel C: Monthly stock characteristics, development sample}}\\")
    out.append(r" & Mean & S.d. & Mean & S.d. \\")
    desc = {m: pd.read_csv(R / "step4" / f"{m}_T1_descriptive.csv", index_col=0) for m in MK}
    for k, lab, sc in (("RSJ", "RSJ", 1), ("RSkew", "Realized skewness", 1), ("MAX", "MAX (\%)", 100),
                       ("IVOL", "IVOL (daily \%)", 100), ("REV", "Monthly return REV (\%)", 100)):
        cells = []
        for m in MK:
            cells += [f(desc[m].loc[k, "mean"] * sc, 3 if sc == 1 else 2), f(desc[m].loc[k, "std"] * sc, 3 if sc == 1 else 2)]
        out.append(f"{lab} & " + ", ".join(cells) + " \\\\")
    write("t_sample", "\n".join(out) + "\n")


def t_regime_diag():
    out = []
    lab = {}
    for m in MK:
        st = pd.read_csv(R / "step2" / f"{m}_regime_stats.csv", index_col=0)
        su = json.loads((R / "step2" / f"{m}_regime_summary.json").read_text(encoding="utf-8"))
        pe = pd.read_csv(R / "step2" / f"{m}_regime_persistence.csv", index_col=0)
        ev = pd.read_csv(R / "step2" / f"{m}_regime_events.csv", index_col=0)
        lam = json.loads((ROOT / "config" / "frozen" / "S1.json").read_text(encoding="utf-8"))["rules"][m]["lambda"]
        lab[m] = [f"{lam:g}", f(100 * st.loc[1.0, "share"], 1), f(st.loc[0.0, "ann_vol_pct"], 1), f(st.loc[1.0, "ann_vol_pct"], 1),
                  f(pe.loc[0, "avg_duration"], 0), f(pe.loc[1, "avg_duration"], 0), f(su["switches_per_year"], 2),
                  f(100 * su["online_offline_agreement"], 1), f(100 * su["agreement_with_other_model"], 1),
                  f(su["detection_lag_median"], 0)] + [f(100 * ev.loc[e, "turbulent_share"], 0) for e in
                                                        ("GFC", "EU_crisis", "China_2015", "Q4_2018", "COVID")]
    names = ["Penalty $\lambda$ (selected)", "Turbulent share of days (\%)", "Volatility, calm (\% p.a.)",
             "Volatility, turbulent (\% p.a.)", "Mean duration, calm (days)", "Mean duration, turbulent (days)",
             "Switches per year", "Online vs.\ full-sample agreement (\%)", "Agreement with HMM (\%)",
             "Median detection lag (days)", "Turbulent share: GFC 2008--09 (\%)", "Turbulent share: euro crisis 2011 (\%)",
             "Turbulent share: China 2015--16 (\%)", "Turbulent share: 2018Q4 (\%)", "Turbulent share: COVID 2020 (\%)"]
    for i, n in enumerate(names):
        out.append(f"{n} & {lab['US'][i]} & {lab['KR'][i]} \\\\")
    write("t_regime", "\n".join(out) + "\n")


def t_winsor():
    out = []
    tgt = {"RSp": "$RS^{+}$", "RSn": "$RS^{-}$", "RV": "$RV$"}
    keys = {"up": "상승 총영향", "dn": "하락 총영향", "df": "합계"}
    eq = {m: pd.read_csv(R / "step1b" / f"{m}_dev_winsor_equality.csv") for m in MK}
    for t_, tl in tgt.items():
        for h in (1, 5, 22):
            row = [f"{tl}, $h={h}$"]
            for m in MK:
                e = eq[m]
                for k in ("up", "dn", "df"):
                    r = e[e.test.str.startswith(keys[k]) & (e.target == t_) & (e.h == h)].iloc[0]
                    row.append(f"{f(r['diff'],2)}{stars(r['p'])}")
            out.append(", ".join(row) + " \\\\")
    write("t_winsor", "\n".join(out) + "\n")


def t_skew22():
    legs = [("LS_EW_gross", "Long--short, EW, gross"), ("LS_VW_gross", "Long--short, VW, gross"),
            ("LS_EW_net", "Long--short, EW, net"), ("LONG_EW_net", "Long-only D1, EW, net")]
    out = []
    for leg, lab in legs:
        row = [lab]
        for m in MK:
            for smp in ("dev", "test"):
                s = pd.read_csv(R / "step10" / f"{m}_{smp}_22d_summary.csv", index_col=0)
                row.append(f"{f(s.loc[leg,'mean_weekly_bps'],1)} {ft(s.loc[leg,'t_NW'])}")
        out.append(", ".join(row) + " \\\\")
    for kind, lab in (("fm_univariate", "FM skewness, alone"), ("fm", "FM skewness, with controls")):
        row = [lab]
        for m in MK:
            for smp in ("dev", "test"):
                s = pd.read_csv(R / "step10" / f"{m}_{smp}_22d_{kind}.csv", index_col=0)
                row.append(f"{f(s.loc['RSK','coef_bps_per_sd'],2)} {ft(s.loc['RSK','t_NW'])}")
        out.append(", ".join(row) + " \\\\")
    write("t_skew22", "\n".join(out) + "\n")


# ================================================================ added: test sample excluding the 2023-25 Korean short-sale ban
def t_ban_robust():
    S = R / "step13"
    cols = [(m, smp) for m in MK for smp in ("full", "excl_ban")]
    out = [r"\multicolumn{5}{l}{\textit{Panel A: Daily semivariance HAR, Holm-significant total effects (of 9)}}\\"]
    sv = pd.read_csv(S / "semivar_test_ban.csv", index_col=0)
    for test, lab in (("down", r"$\Sigma\beta^{-}$"), ("up", r"$\Sigma\beta^{+}$")):
        row = [lab]
        for m, smp in cols:
            x = sv[(sv.market == m) & (sv["sample"] == smp) & (sv.test == test)]
            row.append(f"{int((x.p_holm < 0.05).sum())}")
        out.append(", ".join(row) + " \\\\")
    out.append(r"\addlinespace")
    out.append(r"\multicolumn{5}{l}{\textit{Panel B: Weekly realized skewness, long--short (bps/week)}}\\")
    sk = pd.read_csv(S / "skew_test_ban.csv", index_col=0)
    for leg, lab in (("LS_EW_gross", "EW, gross"), ("LS_VW_gross", "VW, gross"), ("LS_EW_net", "EW, net")):
        row = [lab]
        for m, smp in cols:
            r = sk[(sk.market == m) & (sk["sample"] == smp) & (sk.portfolio == leg)].iloc[0]
            row.append(f"{f(r.mean_weekly_bps,1)} {ft(r.t_NW)}")
        out.append(", ".join(row) + " \\\\")
    out.append(r"\addlinespace")
    out.append(r"\multicolumn{5}{l}{\textit{Panel C: Semivariance stock selection (S4), excess return (\% p.a.)}}\\")
    s4 = pd.read_csv(S / "s4_test_ban.csv", index_col=0)
    for vs, lab in (("EW_universe", r"vs.\ equal-weighted universe"), ("ETF", r"vs.\ index ETF")):
        row = [lab]
        for m, smp in cols:
            r = s4[(s4.market == m) & (s4["sample"] == smp) & (s4.vs == vs)].iloc[0]
            row.append(f"{f(r.ann_diff_pct,1)} {ft(r.t_NW)}")
        out.append(", ".join(row) + " \\\\")
    out.append(r"\addlinespace")
    out.append(r"\multicolumn{5}{l}{\textit{Panel D: Allocation rules on the index ETF}}\\")
    tm = pd.read_csv(S / "timing_test_ban.csv", index_col=0)
    for st, lab in (("S1v2", r"Sharpe ratio, S1v2 vol.\ targeting"), ("S1_A", r"Sharpe ratio, S1\_A regime switch"),
                    ("MA200", "Sharpe ratio, 200-day MA"), ("BH", "Sharpe ratio, buy and hold")):
        row = [lab]
        for m, smp in cols:
            r = tm[(tm.market == m) & (tm["sample"] == smp) & (tm.strategy == st)].iloc[0]
            row.append(f(r.sharpe, 2))
        out.append(", ".join(row) + " \\\\")
    row = [r"S1v2 $-$ buy and hold (\% p.a.)"]
    for m, smp in cols:
        r = tm[(tm.market == m) & (tm["sample"] == smp) & (tm.strategy == "S1v2")].iloc[0]
        row.append(f"{f(r.ann_diff_vs_BH_pct,1)} {ft(r.t_vs_BH)}")
    out.append(", ".join(row) + " \\\\")
    ac = pd.read_csv(S / "timing_accuracy_test_ban.csv", index_col=0)
    row = [r"QLIKE gain, LUHAR vs.\ LHAR, $h=22$ (\%)"]
    for m, smp in cols:
        r = ac[(ac.market == m) & (ac["sample"] == smp) & (ac.index == "LUHAR_h22")].iloc[0]
        row.append(f"{f(r.QLIKE_improve_pct_vs_bench,1)} {ft(r.DM_t_vs_bench)}")
    out.append(", ".join(row) + " \\\\")
    out.append(r"\addlinespace")
    row = ["Observations (days)"]
    for m, smp in cols:
        row.append(f"{int(tm[(tm.market == m) & (tm['sample'] == smp) & (tm.strategy == 'BH')].days.iloc[0]):,}")
    out.append(", ".join(row) + " \\\\")
    write("t_ban_robust", "\n".join(out) + "\n")


if __name__ == "__main__":
    main()
    for fn in (t_sample, t_regime_diag, t_winsor, t_skew22, t_ban_robust):
        fn()
