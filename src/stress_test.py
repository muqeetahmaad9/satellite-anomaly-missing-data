"""Step 6 - controlled degradation of the official test set.

Every saved baseline (6 models x 5 seeds, from run_baselines.py) is evaluated on
degraded copies of the test segments. Degradation is applied with 3 independent
random draws per condition. Outputs:
    results/tables/stress_per_run.csv     one row per condition x draw x model x seed
    results/tables/stress_summary.csv     mean +- std per condition x model
    results/figures/stress_*.png
"""
from pathlib import Path
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from degrade import degrade_all
from features import load_segments
from metrics import all_metrics
from models import MODEL_NAMES, FeatureModel, feature_matrix
from run_baselines import MODELS_DIR, SEEDS, TABLES

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"
PRED_DIR = ROOT / "results" / "predictions" / "stress"
CONDITIONS = ([("clean", 0.0)]
              + [(k, r) for k in ("random", "block") for r in (0.05, 0.1, 0.2, 0.3)]
              + [("jitter", r) for r in (0.1, 0.25)])
DRAWS = [100, 101, 102]


def main():
    torch.set_num_threads(8)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    segs = load_segments()
    te_ids = sorted(k for k, s in segs.items() if s["train"] == 0)
    y = np.array([segs[k]["anomaly"] for k in te_ids])
    models = {(n, s): pickle.load(open(MODELS_DIR / f"{n}_s{s}.pkl", "rb")) for n in MODEL_NAMES for s in SEEDS}

    rows = []
    for kind, rate in CONDITIONS:
        for draw in (DRAWS[:1] if kind == "clean" else DRAWS):
            deg = degrade_all(segs, te_ids, kind, rate, draw)
            te = [deg[k] for k in te_ids]
            X = feature_matrix(te)  # shared by all feature-based models
            preds = {}
            for (name, seed), m in models.items():
                p = m.predict_proba(te, X=X)
                preds[f"{name}_s{seed}"] = p
                yhat = (p >= 0.5).astype(int)
                conf = np.maximum(p, 1 - p)
                accepted = conf >= np.quantile(conf, 0.2)  # 80% most confident decided automatically
                fa = (yhat == 1) & (y == 0)
                miss = (yhat == 0) & (y == 1)
                rows.append({"kind": kind, "rate": rate, "draw": draw, "model": name, "seed": seed,
                             **all_metrics(y, p, calibrated=m.calibrated),
                             "mean_conf": conf.mean(),
                             "false_alarms": int(fa.sum()), "missed": int(miss.sum()),
                             "FA_auto@80": int((fa & accepted).sum()),
                             "miss_auto@80": int((miss & accepted).sum())})
            np.savez(PRED_DIR / f"{kind}_{rate}_{draw}.npz", segment=np.array(te_ids), y=y, **preds)
            print(f"{kind:7s} {rate:.2f} draw {draw} done", flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(TABLES / "stress_per_run.csv", index=False)
    cols = ["F1", "MCC", "Recall", "FPR", "AUC_ROC", "ECE", "AURC", "Risk@80", "mean_conf",
            "false_alarms", "FA_auto@80", "missed", "miss_auto@80"]
    g = runs.groupby(["kind", "rate", "model"], sort=False)[cols]
    summary = g.mean().round(3)
    summary.to_csv(TABLES / "stress_summary.csv")
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(summary.to_string())
    plot(runs)


def plot(runs):
    clean = runs[runs.kind == "clean"]
    metrics = [("FPR", "False-alarm rate (normal segments)"), ("Recall", "Anomaly recall"),
               ("ECE", "Expected calibration error")]
    colors = dict(zip(MODEL_NAMES, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for kind in ("random", "block", "jitter"):
        d = pd.concat([clean.assign(kind=kind), runs[runs.kind == kind]])
        fig, axes = plt.subplots(1, len(metrics), figsize=(14, 3.8))
        for ax, (col, label) in zip(axes, metrics):
            for name in MODEL_NAMES:
                s = d[d.model == name].groupby("rate")[col]
                if s.mean().isna().all():
                    continue
                ax.errorbar(s.mean().index * 100, s.mean(), yerr=s.std().fillna(0), marker="o", ms=3,
                            capsize=2, label=name, color=colors[name])
            ax.set_xlabel("Jitter (% of sampling period)" if kind == "jitter" else "Observations removed (%)")
            ax.set_title(label)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"OPSSAT-AD test set under {kind} degradation (mean ± sd, 5 model seeds × 3 draws)")
        fig.tight_layout()
        fig.savefig(FIGS / f"stress_{kind}.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
