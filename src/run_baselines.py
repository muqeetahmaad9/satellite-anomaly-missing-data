"""Step 5 - full baselines on the official OPSSAT-AD split, 5 seeds per model.

Saves fitted models to results/models/ (reused by later steps), clean test
predictions to results/predictions/clean/, per-seed metrics and a mean +- std
summary to results/tables/.
"""
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import load_segments
from metrics import all_metrics
from models import MODEL_NAMES, make_model

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
MODELS_DIR = ROOT / "results" / "models"
PRED_DIR = ROOT / "results" / "predictions" / "clean"
SEEDS = [0, 1, 2, 3, 4]


def main(names=MODEL_NAMES):
    sys.stdout.reconfigure(encoding="utf-8")  # "±" in the summary on Windows consoles
    torch.set_num_threads(8)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    segs = load_segments()
    tr_ids = sorted(k for k, s in segs.items() if s["train"] == 1)
    te_ids = sorted(k for k, s in segs.items() if s["train"] == 0)
    tr, te = [segs[k] for k in tr_ids], [segs[k] for k in te_ids]
    ytr = np.array([s["anomaly"] for s in tr])
    yte = np.array([s["anomaly"] for s in te])

    rows = []
    for name in names:
        for seed in SEEDS:
            t0 = time.time()
            m = make_model(name, seed).fit(tr, ytr)
            fit_s = time.time() - t0
            p = m.predict_proba(te)
            row = {"model": name, "seed": seed, "fit_seconds": fit_s,
                   **all_metrics(yte, p, calibrated=m.calibrated)}
            for attr in ("n_params", "epochs_run"):
                if hasattr(m, attr):
                    row[attr] = getattr(m, attr)
            rows.append(row)
            np.savez(PRED_DIR / f"{name}_s{seed}.npz", segment=np.array(te_ids), y=yte, p=p)
            with open(MODELS_DIR / f"{name}_s{seed}.pkl", "wb") as f:
                pickle.dump(m, f)
            print(f"{name:15s} seed {seed}  F1 {row['F1']:.3f}  MCC {row['MCC']:.3f}  "
                  f"AUC-ROC {row['AUC_ROC']:.3f}  ({fit_s:.0f}s)", flush=True)

    per_seed = pd.DataFrame(rows)
    out = TABLES / "baselines_per_seed.csv"
    if out.exists() and set(names) != set(MODEL_NAMES):  # partial re-run: merge with earlier results
        old = pd.read_csv(out)
        per_seed = pd.concat([old[~old.model.isin(names)], per_seed], ignore_index=True)
    per_seed.to_csv(out, index=False)

    cols = ["F1", "MCC", "Precision", "Recall", "AUC_PR", "AUC_ROC", "FPR",
            "Brier", "ECE", "AURC", "Risk@80"]
    g = per_seed.groupby("model", sort=False)[cols]
    summary = g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)
    summary = summary.reindex([n for n in MODEL_NAMES if n in summary.index])
    summary.to_csv(TABLES / "baselines_summary.csv")
    print("\nmean ± std over 5 seeds (official test set, threshold 0.5)")
    print(summary.to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or MODEL_NAMES)
