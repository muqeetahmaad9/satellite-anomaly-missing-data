"""Step 8B - learned error predictor with observation-quality features.

A second-stage model g(z) estimates P(detector is wrong | z), in the meta-model
tradition of Chen et al. (2018) and Corbiere et al. (2019). It is trained on the
held-out calibration set, clean plus degraded copies, so the detector's errors
are honest (never seen in training).

  ErrPred-conf      z = (p, max(p, 1-p))                    scores only
  ErrPred-quality   z = scores + observation-quality features (gaps, missing fraction, jitter, length)
  ErrPred-quality-R as ErrPred-quality, but degraded calibration copies use scattered gaps only,
                    to test whether it transfers to unseen block gaps

Rule: accept (decide automatically) if g(z) < tau, tau set so 80% of clean calibration
segments are accepted, using 5-fold cross-fitted scores grouped by segment.

Outputs: results/tables/errpred_per_run.csv, errpred_summary.csv
"""
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from degrade import degrade_all
from features import QUALITY_FEATURES, load_segments, quality_features
from models import feature_matrix
from run_baselines import SEEDS, TABLES
from run_calibrated import MODELS_CAL, calibration_split
from uncertainty import CAL_DRAWS, RATES, TEST_CONDITIONS, TEST_DRAWS, decision_metrics

BASE_MODELS = ["LogReg", "RandomForest", "XGBoost", "CNN", "SensorAwareNet"]
VARIANTS = {"ErrPred-conf": (False, ("random", "block")),
            "ErrPred-quality": (True, ("random", "block")),
            "ErrPred-quality-R": (True, ("random",))}


def quality_matrix(segments):
    return np.array([[quality_features(s["t"], s["sampling"])[f] for f in QUALITY_FEATURES] for s in segments])


def meta_features(p, Q, use_quality):
    Z = np.c_[p, np.maximum(p, 1 - p)]
    return np.c_[Z, Q] if use_quality else Z


def new_g(seed):
    return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=seed)


def calibration_copies(segs, cal_ids):
    """Clean calibration set + one degraded copy per gap type x rate (draws differ from test draws).

    Returns (cal_sets [(kind, rate, segment list)], feature matrices, quality matrices).
    """
    cal_sets = [("none", 0.0, [segs[k] for k in cal_ids])]
    for i, kind in enumerate(("random", "block")):
        for j, r in enumerate(RATES):
            d = degrade_all(segs, cal_ids, kind, r, CAL_DRAWS[0] + 10 * i + j)
            cal_sets.append((kind, r, [d[k] for k in cal_ids]))
    return cal_sets, [feature_matrix(l) for _, _, l in cal_sets], [quality_matrix(l) for _, _, l in cal_sets]


def fit_error_predictors(m, cal_sets, cal_X, cal_Q, y_cal, seed):
    """Fit every VARIANT for detector m. Returns {variant: (g, tau, use_quality)}."""
    n = len(y_cal)
    cal_p = [m.predict_proba(l, X=X) for (_, _, l), X in zip(cal_sets, cal_X)]
    fitted = {}
    for vname, (use_q, kinds) in VARIANTS.items():
        keep = [i for i, (k, _, _) in enumerate(cal_sets) if k == "none" or k in kinds]
        Z = np.vstack([meta_features(cal_p[i], cal_Q[i], use_q) for i in keep])
        err = np.concatenate([((cal_p[i] >= 0.5).astype(int) != y_cal) for i in keep])
        groups = np.tile(np.arange(n), len(keep))
        is_clean = np.concatenate([np.full(n, cal_sets[i][0] == "none") for i in keep])
        oof = np.zeros(len(err))
        for f_tr, f_va in GroupKFold(n_splits=5).split(Z, err, groups):
            oof[f_va] = new_g(seed).fit(Z[f_tr], err[f_tr]).predict_proba(Z[f_va])[:, 1]
        tau = np.quantile(oof[is_clean], 0.8)
        fitted[vname] = (new_g(seed).fit(Z, err), tau, use_q)
    return fitted


def main(names=BASE_MODELS):
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(8)
    segs = load_segments()
    _, cal_ids, te_ids = calibration_split(segs)
    y_cal = np.array([segs[k]["anomaly"] for k in cal_ids])
    y_te = np.array([segs[k]["anomaly"] for k in te_ids])
    cal_sets, cal_X, cal_Q = calibration_copies(segs, cal_ids)

    # degraded test sets (same draws as uncertainty.py)
    test_sets = []
    for kind, rate in TEST_CONDITIONS:
        for draw in (TEST_DRAWS[:1] if kind == "clean" else TEST_DRAWS):
            d = degrade_all(segs, te_ids, kind, rate, draw)
            l = [d[k] for k in te_ids]
            test_sets.append((kind, rate, draw, l, feature_matrix(l), quality_matrix(l)))
    print("data prepared", flush=True)

    rows = []
    for name in names:
        for seed in SEEDS:
            m = pickle.load(open(MODELS_CAL / f"{name}_s{seed}.pkl", "rb"))
            fitted = fit_error_predictors(m, cal_sets, cal_X, cal_Q, y_cal, seed)
            for kind, rate, draw, l, X, Q in test_sets:
                p = m.predict_proba(l, X=X)
                pred = (p >= 0.5).astype(int)
                err = pred != y_te
                base = {"kind": kind, "rate": rate, "draw": draw, "model": name, "seed": seed}
                rows.append({**base, "method": "MaxProb-score",
                             "err_AUROC": roc_auc_score(err, -np.maximum(p, 1 - p)) if 0 < err.sum() < len(err) else np.nan})
                for vname, (g, tau, use_q) in fitted.items():
                    r = g.predict_proba(meta_features(p, Q, use_q))[:, 1]
                    rows.append({**base, "method": vname, **decision_metrics(y_te, pred, r < tau),
                                 "err_AUROC": roc_auc_score(err, r) if 0 < err.sum() < len(err) else np.nan})
            print(f"{name:15s} seed {seed} done", flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(TABLES / "errpred_per_run.csv", index=False)
    cols = ["coverage", "sel_risk", "FA_auto", "miss_auto", "err_AUROC"]
    summary = runs.groupby(["kind", "rate", "model", "method"], sort=False)[cols].mean().round(3)
    summary.to_csv(TABLES / "errpred_summary.csv")
    key = summary.reset_index()
    key = key[(key.rate == 0.2) | (key.kind == "clean")]
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(key.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or BASE_MODELS)
