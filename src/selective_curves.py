"""Step 8C - risk-coverage comparison of deferral rankings at equal automation.

All methods keep the detector's own label (p >= 0.5) and differ only in WHICH
segments they automate first, so the comparison isolates ranking quality:
  MaxProb          max(p, 1-p)
  Conformal-GM     conformal confidence 1 - min p-value, gap-matched calibration
  ErrPred-conf     -g(scores)            (error_predictor.py)
  ErrPred-quality  -g(scores + quality features)

Outputs: results/tables/selective_curves.csv, results/figures/selective_curves.png
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
from error_predictor import BASE_MODELS, calibration_copies, fit_error_predictors, meta_features, quality_matrix
from features import load_segments
from metrics import risk_coverage
from models import feature_matrix
from run_baselines import SEEDS, TABLES
from run_calibrated import MODELS_CAL, calibration_split
from uncertainty import TEST_DRAWS, calibration_predictions, conformal_pvalues, detect_stratum

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"
CONDITIONS = [("clean", 0.0), ("block", 0.2), ("random", 0.2)]
METHODS = ["MaxProb", "Conformal-GM", "ErrPred-conf", "ErrPred-quality"]
COLORS = dict(zip(METHODS, ["#7f7f7f", "#2ca02c", "#9467bd", "#1f77b4"]))
GRID = np.linspace(0.05, 1.0, 96)
FAMILY = {"LogReg": "Feature models", "RandomForest": "Feature models", "XGBoost": "Feature models",
          "CNN": "Neural models", "SensorAwareNet": "Neural models"}


def at_coverage(score, y, pred, c):
    accept = np.zeros(len(y), dtype=bool)
    accept[np.argsort(-score, kind="stable")[:int(np.ceil(c * len(y)))]] = True
    return {"risk": (accept & (pred != y)).sum() / accept.sum(),
            "miss_auto": (accept & (pred == 0) & (y == 1)).sum() / (y == 1).sum(),
            "FA_auto": (accept & (pred == 1) & (y == 0)).sum() / (y == 0).sum()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(8)
    segs = load_segments()
    _, cal_ids, te_ids = calibration_split(segs)
    y_cal = np.array([segs[k]["anomaly"] for k in cal_ids])
    y = np.array([segs[k]["anomaly"] for k in te_ids])
    cal_sets, cal_X, cal_Q = calibration_copies(segs, cal_ids)
    tests = []
    for kind, rate in CONDITIONS:
        for draw in (TEST_DRAWS[:1] if kind == "clean" else TEST_DRAWS):
            d = degrade_all(segs, te_ids, kind, rate, draw)
            l = [d[k] for k in te_ids]
            tests.append((kind, rate, draw, l, feature_matrix(l), quality_matrix(l), [detect_stratum(s) for s in l]))

    rows, curves = [], []
    for name in BASE_MODELS:
        for seed in SEEDS:
            m = pickle.load(open(MODELS_CAL / f"{name}_s{seed}.pkl", "rb"))
            cal_pred = calibration_predictions({(name, seed): m}, segs, cal_ids)
            fitted = fit_error_predictors(m, cal_sets, cal_X, cal_Q, y_cal, seed)
            for kind, rate, draw, l, X, Q, strata in tests:
                p = m.predict_proba(l, X=X)
                pred = (p >= 0.5).astype(int)
                pv = np.zeros((len(p), 2))
                for st in set(strata):
                    idx = np.array([s == st for s in strata])
                    pv[idx] = conformal_pvalues(*cal_pred[(name, seed, st)], p[idx])
                scores = {"MaxProb": np.maximum(p, 1 - p),
                          "Conformal-GM": 1 - pv.min(1) + 1e-6 * pv.max(1)}
                for v in ("ErrPred-conf", "ErrPred-quality"):
                    g, _, use_q = fitted[v]
                    scores[v] = -g.predict_proba(meta_features(p, Q, use_q))[:, 1]
                err = pred != y
                for method, sc in scores.items():
                    cov, risk, aurc = risk_coverage(sc, err)
                    base = {"kind": kind, "rate": rate, "draw": draw, "model": name, "seed": seed, "method": method}
                    row = {**base, "AURC": aurc}
                    for c in (0.5, 0.7, 0.8):
                        for k, v in at_coverage(sc, y, pred, c).items():
                            row[f"{k}@{int(c * 100)}"] = v
                    rows.append(row)
                    curves.append({**base, "risk": np.interp(GRID, cov, risk)})
            print(f"{name:15s} seed {seed} done", flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(TABLES / "selective_curves_per_run.csv", index=False)
    runs["family"] = runs.model.map(FAMILY)
    cols = ["AURC", "risk@50", "risk@70", "risk@80", "miss_auto@80", "FA_auto@80"]
    summary = runs.groupby(["kind", "family", "method"], sort=False)[cols].agg(["mean", "std"]).round(4)
    summary.to_csv(TABLES / "selective_curves.csv")
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(runs.groupby(["kind", "family", "method"], sort=False)[cols].mean().round(4).to_string())

    cdf = pd.DataFrame(curves)
    cdf["family"] = cdf.model.map(FAMILY)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for r, fam in enumerate(["Feature models", "Neural models"]):
        for c, (kind, rate) in enumerate(CONDITIONS):
            ax = axes[r, c]
            for method in METHODS:
                sel = cdf[(cdf.family == fam) & (cdf.kind == kind) & (cdf.method == method)]
                curve = np.vstack(sel.risk.to_numpy())
                ax.plot(GRID, curve.mean(0), color=COLORS[method], label=method)
                ax.fill_between(GRID, np.quantile(curve, 0.1, 0), np.quantile(curve, 0.9, 0),
                                color=COLORS[method], alpha=0.12, lw=0)
            ax.axvline(0.8, color="black", lw=0.6, ls=":")
            if r == 0:
                ax.set_title("clean" if kind == "clean" else f"{kind} loss, {int(rate * 100)}%")
            if r == 1:
                ax.set_xlabel("Coverage (fraction decided automatically)")
        axes[r, 0].set_ylabel(f"{fam}\nerror rate on automated")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Risk-coverage at equal automation (mean, 10-90% band over models, seeds, draws)")
    fig.tight_layout()
    fig.savefig(FIGS / "selective_curves.png", dpi=200)


if __name__ == "__main__":
    main()
