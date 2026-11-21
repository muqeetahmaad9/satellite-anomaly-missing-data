"""Publication figures and tables from the saved results.

Nothing is retrained: tables come from results/tables, reliability diagrams from the
saved stress-test predictions, and the example figure runs two saved models on two segments.

Outputs: results/paper/fig*.pdf|png and results/paper/table*.csv|md
"""
from pathlib import Path
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from degrade import degrade_all
from evidence import evidence_deleted, zero_anomaly_channels
from features import load_segments
from models import feature_matrix

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "results" / "paper"
# test segments chosen to illustrate the two failure modes; in 209 the gap leaves the anomalous region intact
EXAMPLES = {"normal": 449, "anomaly": 209}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.linewidth": 1.3,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})
FULL_W = 7.1  # double-column width, inches

MODELS = ["LogReg", "RandomForest", "XGBoost", "IsolationForest", "CNN", "SensorAwareNet"]
LABEL = {"LogReg": "LogReg", "RandomForest": "Random Forest", "XGBoost": "XGBoost",
         "IsolationForest": "Isolation Forest", "CNN": "1D-CNN", "SensorAwareNet": "SensorAwareNet"}
COLOR = {"LogReg": "#4C72B0", "RandomForest": "#DD8452", "XGBoost": "#55A868",
         "IsolationForest": "#8C8C8C", "CNN": "#8172B3", "SensorAwareNet": "#C44E52"}
STYLE = {m: ("--" if m == "IsolationForest" else "-") for m in MODELS}
FEATURE = ["LogReg", "RandomForest", "XGBoost"]
NEURAL = ["CNN", "SensorAwareNet", "SensorAwareNet-LS"]
METHOD_LABEL = {"MaxProb": "Max. probability", "Conformal": "Conformal (standard)",
                "Conformal-GM": "Conformal, gap-matched", "Conformal-GX": "Conformal, gap-mismatched",
                "ErrPred-conf": "Error predictor (scores)", "ErrPred-quality": "Error predictor (+ gap features)"}
METHOD_COLOR = {"MaxProb": "#7f7f7f", "Conformal": "#d62728", "Conformal-GM": "#2ca02c",
                "Conformal-GX": "#ff7f0e", "ErrPred-conf": "#9467bd", "ErrPred-quality": "#1f77b4"}


def legend_below(fig, ax, ncol, y=-0.02):
    """One shared legend centred under the panels, outside the data area."""
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)


def with_clean(runs, kind):
    return pd.concat([runs[runs.kind == "clean"].assign(kind=kind), runs[runs.kind == kind]])


# ---------------------------------------------------------------- overview
def fig_overview():
    fig, ax = plt.subplots(figsize=(FULL_W, 2.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")
    w, gap, y0, h = 17.2, 2.9, 11, 23
    titles = ["Telemetry", "Observation loss", "Detectors", "Decision rules", "Outcome"]
    bodies = ["OPSSAT-AD\n2,123 labelled\nsegments, 9 channels\nofficial split",
              "none\nscattered points\ncontiguous block\ntiming jitter\n5-30% of a segment",
              "feature-based:\nLogReg, RF,\nXGBoost, iForest\nneural: 1D-CNN,\nSensorAwareNet",
              "max. probability\nconformal: standard,\ngap-matched,\ngap-mismatched\nerror predictor",
              "automate or defer\nfalse alarms\nmissed anomalies\nvalidity, workload"]
    xs = [1 + i * (w + gap) for i in range(5)]
    for x, title, body in zip(xs, titles, bodies):
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc="#f4f6fa", ec="#4C72B0", lw=0.9))
        ax.text(x + w / 2, y0 + h - 1.8, title, ha="center", va="top", fontweight="bold", fontsize=7.2)
        ax.text(x + w / 2, y0 + h - 6.2, body, ha="center", va="top", fontsize=6.2, linespacing=1.35)
    for x in xs[:-1]:
        ax.add_patch(FancyArrowPatch((x + w + 0.4, y0 + h / 2), (x + w + gap - 0.4, y0 + h / 2),
                                     arrowstyle="-|>", mutation_scale=9, color="#333333", lw=0.9))
    notes = [(xs[2] + w / 2, "remedy:\ngap-augmented training"),
             (xs[3] + w / 2, "held-out calibration set\n(319 segments), clean or\ngap-matched copies")]
    for x, text in notes:
        ax.add_patch(FancyArrowPatch((x, 6.2), (x, y0 - 0.4), arrowstyle="-|>", mutation_scale=8,
                                     color="#C44E52", lw=0.8))
        ax.text(x, 5.8, text, ha="center", va="top", fontsize=5.9, color="#C44E52", linespacing=1.25)
    save(fig, "fig1_overview")


# ---------------------------------------------------------------- examples
def fig_examples():
    segs = load_segments()
    rf = pickle.load(open(ROOT / "results" / "models" / "RandomForest_s0.pkl", "rb"))
    san = pickle.load(open(ROOT / "results" / "models" / "SensorAwareNet_s0.pkl", "rb"))
    te_ids = sorted(k for k, s in segs.items() if s["train"] == 0)
    conditions = [("clean", 0.0, "Clean"), ("block", 0.2, "20% block removed"), ("random", 0.2, "20% scattered removed")]
    # degrade the whole test set with the stress-test draw so each panel shows exactly the evaluated copy
    copies = {kind: degrade_all(segs, te_ids, kind, rate, 100) for kind, rate, _ in conditions}
    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 3.8), sharey="row")
    for r, (label, sid) in enumerate(EXAMPLES.items()):
        s = segs[sid]
        for c, (kind, rate, title) in enumerate(conditions):
            ax = axes[r, c]
            d = copies[kind][sid]
            removed = ~np.isin(s["t"], d["t"])
            ax.plot(s["t"][removed], s["x"][removed] * 1e5, ".", ms=2, color="#bbbbbb", label="removed observations")
            ax.plot(d["t"], d["x"] * 1e5, "-", lw=0.8, color="#4C72B0", label="observed telemetry")
            p_rf = rf.predict_proba([d], X=feature_matrix([d]))[0]
            p_san = san.predict_proba([d])[0]
            ax.text(0.02, 0.97, f"P(anomaly): RF {p_rf:.2f}, SensorAwareNet {p_san:.2f}",
                    transform=ax.transAxes, va="top", fontsize=6.2)
            lo, hi = s["x"].min() * 1e5, s["x"].max() * 1e5
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.3 * (hi - lo))  # headroom for the text
            if r == 0:
                ax.set_title(title)
            if r == 1:
                ax.set_xlabel("Time from segment start (s)")
        axes[r, 0].set_ylabel(f"{label.capitalize()} segment\n{s['channel']} (×10$^{{-5}}$)")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    legend_below(fig, axes[0, 1], ncol=2)
    for ax, letter in zip(axes.flat, "abcdef"):
        ax.text(-0.12, 1.04, f"({letter})", transform=ax.transAxes, fontweight="bold")
    save(fig, "fig2_examples")


# ---------------------------------------------------------------- reliability
def fig_reliability():
    pred_dir = ROOT / "results" / "predictions" / "stress"
    sets = {"Clean": ["clean_0.0_100"],
            "20% scattered loss": [f"random_0.2_{d}" for d in (100, 101, 102)],
            "20% block loss": [f"block_0.2_{d}" for d in (100, 101, 102)]}
    colors = {"Clean": "#333333", "20% scattered loss": "#DD8452", "20% block loss": "#C44E52"}
    bins = np.linspace(0, 1, 11)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.5), sharey=True)
    for ax, model in zip(axes, ["RandomForest", "XGBoost", "SensorAwareNet"]):
        ax.plot([0, 1], [0, 1], color="#999999", lw=0.7, ls="--", label="perfect calibration")
        ece_lines = []
        for cond, files in sets.items():
            ps, ys = [], []
            for f in files:
                z = np.load(pred_dir / f"{f}.npz")
                for seed in range(5):
                    ps.append(z[f"{model}_s{seed}"])
                    ys.append(z["y"])
            p, y = np.concatenate(ps), np.concatenate(ys)
            idx = np.minimum(np.digitize(p, bins) - 1, 9)
            mp = [p[idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
            fy = [y[idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
            ece = sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(10) if (idx == b).any())
            ax.plot(mp, fy, "o-", ms=3, color=colors[cond], label=cond)
            ece_lines.append(f"{cond.replace(' loss', '')}: {ece:.2f}")
        ax.text(0.03, 0.97, "ECE\n" + "\n".join(ece_lines), transform=ax.transAxes, va="top", fontsize=6.2)
        ax.set_title(LABEL[model])
        ax.set_xlabel("Predicted P(anomaly)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed anomaly fraction")
    fig.tight_layout()
    legend_below(fig, axes[0], ncol=4)
    for ax, letter in zip(axes, "abc"):
        ax.text(-0.12, 1.04, f"({letter})", transform=ax.transAxes, fontweight="bold")
    save(fig, "fig4_reliability")


# ---------------------------------------------------------------- stress
def stress_visible_recall():
    """Recall on anomalies whose evidence survived each degraded copy, from saved stress-test predictions."""
    out = TABLES / "stress_visible.csv"
    if out.exists():
        return pd.read_csv(out)
    segs = load_segments()
    te_ids = sorted(k for k, s in segs.items() if s["train"] == 0)
    zero_chs = zero_anomaly_channels(segs)
    rows = []
    for f in sorted((ROOT / "results" / "predictions" / "stress").glob("*.npz")):
        kind, rate, draw = f.stem.split("_")
        z = np.load(f)
        assert list(z["segment"]) == te_ids
        deg = degrade_all(segs, te_ids, kind, float(rate), int(draw))
        vis = (z["y"] == 1) & ~evidence_deleted(segs, te_ids, deg, zero_chs)
        for key in z.files:
            if key in ("segment", "y"):
                continue
            model, seed = key.rsplit("_s", 1)
            rows.append({"kind": kind, "rate": float(rate), "draw": int(draw), "model": model, "seed": int(seed),
                         "Recall_vis": (z[key][vis] >= 0.5).mean(), "n_visible": int(vis.sum())})
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


def fig_stress():
    runs = pd.read_csv(TABLES / "stress_per_run.csv").merge(
        stress_visible_recall(), on=["kind", "rate", "draw", "model", "seed"])
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 4.4), sharex=True)
    panels = [("FPR", "False-alarm rate\n(normal segments)"),
              ("Recall_vis", "Recall on anomalies\nstill visible after loss")]
    for c, kind in enumerate(("block", "random")):
        d = with_clean(runs, kind)
        for r, (col, ylabel) in enumerate(panels):
            ax = axes[r, c]
            for m in MODELS:
                s = d[d.model == m].groupby("rate")[col]
                ax.errorbar(s.mean().index * 100, s.mean(), yerr=s.std(), color=COLOR[m], ls=STYLE[m],
                            marker="o", ms=2.5, capsize=1.5, elinewidth=0.6, label=LABEL[m])
            ax.set_ylim(-0.02, 1.02)
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == 0:
                ax.set_title(f"{'Contiguous block' if kind == 'block' else 'Scattered'} observation loss")
            if r == 1:
                ax.set_xlabel("Observations removed (%)")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    legend_below(fig, axes[0, 0], ncol=6)
    for ax, letter in zip(axes.flat, "abcd"):
        ax.text(-0.13, 1.04, f"({letter})", transform=ax.transAxes, fontweight="bold")
    save(fig, "fig3_stress")


# ---------------------------------------------------------------- figure 2
def fig_uncertainty():
    runs = pd.read_csv(TABLES / "uncertainty_per_run.csv")
    methods = ["MaxProb", "Conformal", "Conformal-GM", "Conformal-GX"]
    runs = runs[runs.method.isin(methods)]
    d = with_clean(runs, "block")
    panels = [("set_cov_normal", "Normal segments covered\nby the prediction set"),
              ("sel_risk", "Error rate on\nautomated decisions"),
              ("coverage", "Fraction decided\nautomatically")]
    fams = [("Feature-based detectors", FEATURE), ("Neural detectors", NEURAL)]
    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 4.4), sharex=True)
    for r, (fam, models) in enumerate(fams):
        f = d[d.model.isin(models)]
        for c, (col, ylabel) in enumerate(panels):
            ax = axes[r, c]
            for m in methods:
                s = f[f.method == m].groupby("rate")[col]
                if s.mean().isna().all():
                    continue
                ax.errorbar(s.mean().index * 100, s.mean(), yerr=s.std(), color=METHOD_COLOR[m], marker="o",
                            ms=2.5, capsize=1.5, elinewidth=0.6, label=METHOD_LABEL[m])
            if col == "set_cov_normal":
                ax.axhline(0.95, color="black", lw=0.7, ls="--")
                ax.text(0, 0.90, "target 0.95", ha="left", va="top", fontsize=6.5)
            ax.set_ylim(-0.02, 1.02)
            if r == 0:
                ax.set_title(ylabel)
            if r == 1:
                ax.set_xlabel("Block observation loss (%)")
        axes[r, 0].set_ylabel(fam)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    legend_below(fig, axes[0, 1], ncol=4)
    for ax, letter in zip(axes.flat, "abcdef"):
        ax.text(-0.15, 1.04, f"({letter})", transform=ax.transAxes, fontweight="bold")
    save(fig, "fig5_uncertainty_block")


# ---------------------------------------------------------------- figure 3
def fig_selective():
    runs = pd.read_csv(TABLES / "selective_curves_per_run.csv")
    fam = {m: "Feature-based detectors" for m in FEATURE} | {m: "Neural detectors" for m in NEURAL}
    runs["family"] = runs.model.map(fam)
    methods = ["MaxProb", "Conformal-GM", "ErrPred-conf", "ErrPred-quality"]
    cov = np.array([50, 70, 80])
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4))
    for ax, f in zip(axes, ["Feature-based detectors", "Neural detectors"]):
        d = runs[(runs.family == f) & (runs.kind == "block")]
        width = 0.2
        for i, m in enumerate(methods):
            vals = [d[d.method == m][f"risk@{c}"] for c in cov]
            ax.bar(np.arange(3) + (i - 1.5) * width, [v.mean() for v in vals], width,
                   yerr=[v.std() for v in vals], color=METHOD_COLOR[m], label=METHOD_LABEL[m],
                   error_kw={"elinewidth": 0.6, "capsize": 1.5})
        ax.set_xticks(range(3), [f"{c}%" for c in cov])
        ax.set_xlabel("Fraction decided automatically")
        ax.set_title(f"{f}, 20% block loss")
    axes[0].set_ylabel("Error rate on automated decisions")
    fig.tight_layout()
    legend_below(fig, axes[0], ncol=4)
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.12, 1.04, f"({letter})", transform=ax.transAxes, fontweight="bold")
    save(fig, "fig6_equal_automation")


# ---------------------------------------------------------------- figure 4
def fig_remedies():
    s = pd.concat([pd.read_csv(TABLES / f) for f in
                   ("uncertainty_summary.csv", "uncertainty_summary_aug1.csv", "uncertainty_summary_aug2.csv")])
    s = s[(s.kind == "block") & (s.rate == 0.2) & s.method.isin(["MaxProb", "Conformal", "Conformal-GM"])]
    bases = [("RandomForest", "Random Forest"), ("XGBoost", "XGBoost"), ("SensorAwareNet", "SensorAwareNet")]
    training = {"": ("standard", "o"), "-AugR": ("augmented, scattered gaps", "s"), "-Aug": ("augmented, both gap types", "D")}
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.6), sharey=True)
    for ax, (b, title) in zip(axes, bases):
        for suf, (tlabel, marker) in training.items():
            for m in ("MaxProb", "Conformal", "Conformal-GM"):
                row = s[(s.model == b + suf) & (s.method == m)]
                if row.empty:
                    continue
                ax.scatter(row.coverage, row.sel_risk, marker=marker, s=28, color=METHOD_COLOR[m],
                           edgecolor="black", linewidth=0.4)
        ax.set_title(title)
        ax.set_xlabel("Fraction decided automatically")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 0.7)
    axes[0].set_ylabel("Error rate on automated decisions")
    h = [plt.Line2D([], [], ls="", marker="o", color=METHOD_COLOR[m], label=METHOD_LABEL[m])
         for m in ("MaxProb", "Conformal", "Conformal-GM")]
    h += [plt.Line2D([], [], ls="", marker=mk, color="white", markeredgecolor="black", label=f"Training: {t}")
          for t, mk in training.values()]
    fig.tight_layout()
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    for ax, letter in zip(axes, "abc"):
        ax.text(-0.12, 1.04, f"({letter})", transform=ax.transAxes, fontweight="bold")
    fig.text(0.5, 1.02, "20% block observation loss; better = lower right", ha="center", fontsize=7.5)
    save(fig, "fig7_remedies")


# ---------------------------------------------------------------- tables
def write_table(df, name, caption):
    df.to_csv(OUT / f"{name}.csv")
    (OUT / f"{name}.md").write_text(f"**{caption}**\n\n" + df.to_markdown() + "\n", encoding="utf-8")


def tables():
    counts = pd.read_csv(TABLES / "audit_counts.csv", index_col=0)
    write_table(counts, "table1_dataset", "Table 1. OPSSAT-AD segments by split and label (official split).")

    b = pd.read_csv(TABLES / "baselines_per_seed.csv")
    cols = ["F1", "MCC", "AUC_ROC", "AUC_PR", "FPR", "ECE", "AURC"]
    g = b.groupby("model", sort=False)[cols]
    t2 = (g.mean().map(lambda v: f"{v:.3f}") + " ± " + g.std().map(lambda v: f"{v:.3f}")).reindex(MODELS)
    t2.index = [LABEL[m] for m in t2.index]
    t2.loc["Isolation Forest", ["ECE", "AURC"]] = "n/a"
    write_table(t2, "table2_baselines", "Table 2. Clean test performance, mean ± sd over 5 seeds (threshold 0.5).")

    st = pd.read_csv(TABLES / "stress_per_run.csv").merge(
        stress_visible_recall(), on=["kind", "rate", "draw", "model", "seed"])
    st = st[st.model.isin(MODELS) & ((st.kind == "clean") | (st.rate == 0.2))]
    t3 = st.pivot_table(index="model", columns="kind", values=["FPR", "Recall", "Recall_vis", "ECE"],
                        aggfunc="mean", sort=False)
    t3 = t3.reindex(MODELS).round(3)
    t3.index = [LABEL[m] for m in t3.index]
    write_table(t3, "table3_stress", "Table 3. Clean vs 20% observation loss (mean over 5 seeds x 3 draws).")

    u = pd.read_csv(TABLES / "uncertainty_summary.csv")
    u = u[(u.kind == "block") & (u.rate == 0.2) & u.method.isin(["MaxProb", "Conformal", "Conformal-GM", "Conformal-GX"])]
    u["family"] = np.where(u.model.isin(FEATURE), "Feature-based", "Neural")
    t4 = u.groupby(["family", "method"], sort=False)[["coverage", "sel_risk", "FA_auto", "miss_auto", "miss_auto_vis",
                                                       "set_cov_normal", "set_cov_anomaly"]].mean().round(3)
    t4 = t4.rename(columns={"coverage": "Automated", "sel_risk": "Error on automated", "FA_auto": "False alarms automated",
                            "miss_auto": "Misses automated (all anomalies)",
                            "miss_auto_vis": "Misses automated (anomaly still visible)",
                            "set_cov_normal": "Set coverage (normal)",
                            "set_cov_anomaly": "Set coverage (anomaly)"})
    write_table(t4, "table4_uncertainty", "Table 4. Decision rules under 20% block loss, thresholds fixed on clean calibration data.")

    t5 = pd.read_csv(TABLES / "stats_bootstrap.csv").set_index(["family", "statistic"])
    write_table(t5, "table5_bootstrap", "Table 5. Bootstrap 95% confidence intervals (2,000 resamples of test segments).")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("fig*"):  # figure numbers change as figures are added
        old.unlink()
    fig_overview()
    fig_examples()
    fig_stress()
    fig_reliability()
    fig_uncertainty()
    fig_selective()
    fig_remedies()
    tables()
    print("\n".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
