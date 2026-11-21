"""Figures and summary tables for Step 7 (reads uncertainty_per_run.csv)."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"
FAMILIES = {"Feature models (LogReg, RF, XGBoost)": ["LogReg", "RandomForest", "XGBoost"],
            "Neural models (CNN, SensorAwareNet, +LS)": ["CNN", "SensorAwareNet", "SensorAwareNet-LS"]}
METHODS = {"MaxProb": "MaxProb (fixed threshold)", "Conformal": "Conformal",
           "Conformal-GM": "Conformal, gap-matched", "Conformal-GX": "Conformal, gap-mismatched"}
COLORS = dict(zip(METHODS, ["#7f7f7f", "#d62728", "#2ca02c", "#ff7f0e"]))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    runs = pd.read_csv(TABLES / "uncertainty_per_run.csv")
    runs = runs[runs.method.isin(METHODS)]
    clean = runs[runs.kind == "clean"]
    panels = [("set_cov_normal", "Normal segments kept in set\n(target ≥ 0.95)"),
              ("sel_risk", "Error rate on automated decisions"),
              ("coverage", "Fraction decided automatically")]
    for kind in ("block", "random"):
        d = pd.concat([clean.assign(kind=kind), runs[runs.kind == kind]])
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
        for r, (fam, models) in enumerate(FAMILIES.items()):
            f = d[d.model.isin(models)]
            for c, (col, label) in enumerate(panels):
                ax = axes[r, c]
                for m in METHODS:
                    s = f[f.method == m].groupby("rate")[col]
                    if s.mean().isna().all():
                        continue
                    ax.errorbar(s.mean().index * 100, s.mean(), yerr=s.std(), marker="o", ms=3, capsize=2,
                                color=COLORS[m], label=METHODS[m])
                if col == "set_cov_normal":
                    ax.axhline(0.95, color="black", lw=0.8, ls="--")
                ax.set_ylim(-0.02, 1.02)
                if r == 0:
                    ax.set_title(label, fontsize=10)
                if r == 1:
                    ax.set_xlabel("Observations removed (%)")
            axes[r, 0].set_ylabel(fam, fontsize=9)
        axes[0, 1].legend(frameon=False, fontsize=8)
        fig.suptitle(f"Selective decisions under {kind} observation loss "
                     "(mean ± sd over models, 5 seeds, 3 draws; thresholds fixed on clean calibration data)")
        fig.tight_layout()
        fig.savefig(FIGS / f"uncertainty_{kind}.png", dpi=200)
        plt.close(fig)

    # compact table: block 20% and random 20%, per model and method
    sel = runs[((runs.kind.isin(["block", "random"])) & (runs.rate == 0.2)) | (runs.kind == "clean")]
    cols = ["coverage", "sel_risk", "FA_auto", "miss_auto", "set_cov_normal", "set_cov_anomaly"]
    tab = sel.groupby(["kind", "model", "method"], sort=False)[cols].mean().round(3)
    tab.to_csv(TABLES / "uncertainty_key_table.csv")
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(tab.to_string())

    # label smoothing and temperature scaling: calibration error
    ece = pd.read_csv(TABLES / "uncertainty_per_run.csv")
    ece = ece[ece.method.isin(["MaxProb", "TempScale"])]
    ece = ece.assign(method=ece.method.replace({"MaxProb": "raw"}))
    et = ece.pivot_table(index=["kind", "rate"], columns=["model", "method"], values="ECE", sort=False).round(3)
    et.to_csv(TABLES / "uncertainty_ece.csv")
    with pd.option_context("display.width", 250):
        print(et.to_string())


if __name__ == "__main__":
    main()
