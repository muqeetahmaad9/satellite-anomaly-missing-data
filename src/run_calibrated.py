"""Step 7a - retrain detectors with a held-out calibration set.

The official training set (1,594 segments) is split once, stratified, into
80% proper training (1,275) and 20% calibration (319). Every model seed uses the
same split, so differences between seeds come from model randomness only. The
official test set is untouched.

Outputs: results/models_cal/<model>_s<seed>.pkl, results/tables/calibration_split.csv
"""
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import load_segments
from models import make_model
from run_baselines import SEEDS, TABLES

ROOT = Path(__file__).resolve().parents[1]
MODELS_CAL = ROOT / "results" / "models_cal"
CAL_MODELS = ["LogReg", "RandomForest", "XGBoost", "CNN", "SensorAwareNet", "SensorAwareNet-LS"]
SPLIT_SEED = 2137


def calibration_split(segs):
    """Return (proper_train_ids, calibration_ids, test_ids)."""
    tr_ids = np.array(sorted(k for k, s in segs.items() if s["train"] == 1))
    y = np.array([segs[k]["anomaly"] for k in tr_ids])
    fit_ids, cal_ids = train_test_split(tr_ids, test_size=0.2, stratify=y, random_state=SPLIT_SEED)
    te_ids = sorted(k for k, s in segs.items() if s["train"] == 0)
    return sorted(fit_ids), sorted(cal_ids), te_ids


def main(names=CAL_MODELS, seeds=SEEDS, threads=8):
    torch.set_num_threads(threads)
    MODELS_CAL.mkdir(parents=True, exist_ok=True)
    segs = load_segments()
    fit_ids, cal_ids, _ = calibration_split(segs)
    pd.DataFrame({"segment": fit_ids + cal_ids,
                  "role": ["fit"] * len(fit_ids) + ["calibration"] * len(cal_ids)}
                 ).to_csv(TABLES / "calibration_split.csv", index=False)
    fit = [segs[k] for k in fit_ids]
    y = np.array([s["anomaly"] for s in fit])
    print(f"fit {len(fit_ids)} ({y.sum()} anomalies) | calibration {len(cal_ids)} "
          f"({sum(segs[k]['anomaly'] for k in cal_ids)} anomalies)", flush=True)
    for name in names:
        for seed in seeds:
            t0 = time.time()
            m = make_model(name, seed).fit(fit, y)
            with open(MODELS_CAL / f"{name}_s{seed}.pkl", "wb") as f:
                pickle.dump(m, f)
            print(f"{name:18s} seed {seed} fitted ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    # usage: run_calibrated.py [model ...] [--seeds 2,3] [--threads 3]
    args, opts = [], {}
    it = iter(sys.argv[1:])
    for a in it:
        if a.startswith("--"):
            opts[a[2:]] = next(it)
        else:
            args.append(a)
    main(args or CAL_MODELS,
         seeds=[int(s) for s in opts["seeds"].split(",")] if "seeds" in opts else SEEDS,
         threads=int(opts.get("threads", 8)))
