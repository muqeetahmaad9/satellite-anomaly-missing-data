"""Step 7b - do uncertainty methods catch the errors caused by missing data?

All thresholds are fixed on the held-out calibration set (run_calibrated.py) and
then applied unchanged to clean and degraded test data, as in deployment.

Decision rules ("accept" = decided automatically, otherwise deferred to an operator):
  MaxProb        accept if max(p, 1-p) >= tau, tau = 20th percentile of calibration
                 confidence (80% automated on clean calibration data)
  Conformal      Mondrian (class-conditional) split conformal, eps = 0.05; accept
                 only singleton prediction sets (Hallberg Szabadvary et al., 2025)
  Conformal-GM   gap-matched: the gap pattern of each test segment (none / block /
                 scattered, and missing fraction) is detected from its timestamps;
                 calibration segments are degraded with that pattern and used as its
                 calibration set (mask-conditional calibration, Zaffran et al., 2023)
  Conformal-GX   gap-mismatched: as GM but calibrated with the other gap type
Temperature scaling is monotone, so it accepts the same segments as MaxProb; only
its calibration error is reported.

Outputs: results/tables/uncertainty_per_run.csv, uncertainty_summary.csv,
gap_detection.csv
"""
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from degrade import degrade_all
from evidence import evidence_deleted, zero_anomaly_channels
from features import load_segments, quality_features
from metrics import ece
from models import feature_matrix
from run_baselines import SEEDS, TABLES
from run_calibrated import CAL_MODELS, MODELS_CAL, calibration_split

EPS = 0.05
RATES = (0.05, 0.1, 0.2, 0.3)
TEST_CONDITIONS = ([("clean", 0.0)] + [(k, r) for k in ("random", "block") for r in (0.1, 0.2, 0.3)]
                   + [("jitter", 0.25)])
TEST_DRAWS = [100, 101, 102]
CAL_DRAWS = [200, 201, 202]


# ------------------------------------------------------------------ gap pattern
def detect_stratum(s):
    """Observable gap pattern of a segment: ('none', 0) or (kind, nearest rate)."""
    q = quality_features(s["t"], s["sampling"])
    if q["n_gaps"] == 0:
        return ("none", 0.0)
    kind = "block" if q["n_gaps"] == 1 else "random"
    rate = min(RATES, key=lambda r: abs(r - q["missing_frac"]))
    return (kind, rate)


# ------------------------------------------------------------------ conformal
def conformal_pvalues(p_cal, y_cal, p_test):
    """Mondrian split-conformal p-values for both labels, shape (n, 2)."""
    pv = np.zeros((len(p_test), 2))
    for c in (0, 1):
        a_cal = np.sort(1 - np.where(c == 1, p_cal, 1 - p_cal)[y_cal == c])
        a_test = 1 - (p_test if c == 1 else 1 - p_test)
        n_ge = len(a_cal) - np.searchsorted(a_cal, a_test, side="left")
        pv[:, c] = (n_ge + 1) / (len(a_cal) + 1)
    return pv


def conformal_sets(p_cal, y_cal, p_test, eps=EPS):
    """Mondrian split conformal for binary labels. Returns boolean array (n, 2): label in set."""
    return conformal_pvalues(p_cal, y_cal, p_test) > eps


def temperature(p_cal, y_cal):
    z = np.log(np.clip(p_cal, 1e-6, 1 - 1e-6) / np.clip(1 - p_cal, 1e-6, 1))
    nll = lambda T: -np.mean(y_cal * -np.logaddexp(0, -z / T) + (1 - y_cal) * -np.logaddexp(0, z / T))
    return minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x


def apply_temperature(p, T):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    return 1 / (1 + np.exp(-z / T))


def decision_metrics(y, pred, accept, visible=None):
    """pred: automated label; accept: decided automatically; visible: anomalies whose evidence survived
    the degradation (evidence.py) - adds miss_auto_vis, the miss rate among those only."""
    err = accept & (pred != y)
    extra = {}
    if visible is not None:
        av = (y == 1) & visible
        extra["miss_auto_vis"] = (accept & (pred == 0) & av).sum() / max(av.sum(), 1)
    return {**extra, "coverage": accept.mean(),
            "sel_risk": err.sum() / max(accept.sum(), 1),
            "FA_auto": (accept & (pred == 1) & (y == 0)).sum() / (y == 0).sum(),
            "miss_auto": (accept & (pred == 0) & (y == 1)).sum() / (y == 1).sum(),
            "defer_anom": (~accept & (y == 1)).sum() / (y == 1).sum()}


CONFORMAL_VARIANTS = {
    "Conformal": lambda st: ("none", 0.0),
    "Conformal-GM": lambda st: st,
    "Conformal-GX": lambda st: st if st[0] == "none" else ({"block": "random", "random": "block"}[st[0]], st[1]),
}


def load_models(names):
    return {(n, s): pickle.load(open(MODELS_CAL / f"{n}_s{s}.pkl", "rb")) for n in names for s in SEEDS}


def calibration_predictions(models, segs, cal_ids):
    """Clean and gap-augmented calibration sets (pooled over draws), predicted once per model.

    Returns {(model, seed, stratum): (p, y)}; augmented arrays hold len(CAL_DRAWS) blocks of
    len(cal_ids) entries, in cal_ids order.
    """
    y_cal = np.array([segs[k]["anomaly"] for k in cal_ids])
    cal_sets = {("none", 0.0): [{k: segs[k] for k in cal_ids}]}
    for kind in ("random", "block"):
        for r in RATES:
            cal_sets[(kind, r)] = [degrade_all(segs, cal_ids, kind, r, d) for d in CAL_DRAWS]
    cal_pred = {}
    for stratum, draws in cal_sets.items():
        lists = [[d[k] for k in cal_ids] for d in draws]
        Xs = [feature_matrix(l) for l in lists]
        for key, m in models.items():
            p = np.concatenate([m.predict_proba(l, X=X) for l, X in zip(lists, Xs)])
            cal_pred[key + (stratum,)] = (p, np.tile(y_cal, len(draws)))
    return cal_pred


def stratified_sets(p, strata, cal_for, cal_lookup):
    """Conformal sets where each test segment is calibrated on the stratum cal_for(its stratum).

    cal_lookup(stratum) -> (p_cal, y_cal).
    """
    sets = np.zeros((len(p), 2), dtype=bool)
    for st in set(cal_for(s) for s in strata):
        idx = np.array([cal_for(s) == st for s in strata])
        sets[idx] = conformal_sets(*cal_lookup(st), p[idx])
    return sets


def main(names=CAL_MODELS, tag=""):
    """Evaluate models `names` from results/models_cal; outputs carry the suffix `tag`."""
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(8)
    segs = load_segments()
    _, cal_ids, te_ids = calibration_split(segs)
    y_te = np.array([segs[k]["anomaly"] for k in te_ids])
    models = load_models(names)
    print("scoring calibration sets ...", flush=True)
    cal_pred = calibration_predictions(models, segs, cal_ids)
    zero_chs = zero_anomaly_channels(segs)

    rows, detect_rows = [], []
    for kind, rate in TEST_CONDITIONS:
        for draw in (TEST_DRAWS[:1] if kind == "clean" else TEST_DRAWS):
            deg = degrade_all(segs, te_ids, kind, rate, draw)
            te = [deg[k] for k in te_ids]
            X = feature_matrix(te)
            strata = [detect_stratum(s) for s in te]
            visible = ~evidence_deleted(segs, te_ids, deg, zero_chs)
            detect_rows += [{"kind": kind, "rate": rate, "draw": draw, "detected_kind": st[0],
                             "detected_rate": st[1]} for st in strata]
            for (name, seed), m in models.items():
                p = m.predict_proba(te, X=X)
                p_c, y_c = cal_pred[(name, seed, ("none", 0.0))]
                base = {"kind": kind, "rate": rate, "draw": draw, "model": name, "seed": seed}
                pred = (p >= 0.5).astype(int)

                # MaxProb with calibration-fixed threshold
                tau = np.quantile(np.maximum(p_c, 1 - p_c), 0.2)
                rows.append({**base, "method": "MaxProb", **decision_metrics(y_te, pred, np.maximum(p, 1 - p) >= tau, visible),
                             "ECE": ece(y_te, p), "FPR": pred[y_te == 0].mean(), "Recall": pred[y_te == 1].mean(),
                             "Recall_vis": pred[(y_te == 1) & visible].mean()})
                T = temperature(p_c, y_c)
                rows.append({**base, "method": "TempScale", "ECE": ece(y_te, apply_temperature(p, T)), "T": T})

                for method, cal_for in CONFORMAL_VARIANTS.items():
                    sets = stratified_sets(p, strata, cal_for, lambda st: cal_pred[(name, seed, st)])
                    single = sets.sum(1) == 1
                    cpred = np.where(sets[:, 1] & ~sets[:, 0], 1, 0)
                    rows.append({**base, "method": method, **decision_metrics(y_te, cpred, single, visible),
                                 "set_cov_normal": sets[y_te == 0, 0].mean(),
                                 "set_cov_anomaly": sets[y_te == 1, 1].mean()})
            print(f"{kind:7s} {rate:.2f} draw {draw} done", flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(TABLES / f"uncertainty_per_run{tag}.csv", index=False)
    det = pd.DataFrame(detect_rows)
    det_tab = det.groupby(["kind", "rate"])[["detected_kind"]].value_counts(normalize=True).unstack().round(3)
    det_tab.to_csv(TABLES / "gap_detection.csv")
    cols = ["coverage", "sel_risk", "FA_auto", "miss_auto", "miss_auto_vis", "defer_anom", "set_cov_normal",
            "set_cov_anomaly", "ECE", "FPR", "Recall", "Recall_vis"]
    summary = runs.groupby(["kind", "rate", "model", "method"], sort=False)[cols].mean().round(3)
    summary.to_csv(TABLES / f"uncertainty_summary{tag}.csv")
    print(det_tab.to_string())


if __name__ == "__main__":
    # usage: uncertainty.py [tag model1 model2 ...]
    if len(sys.argv) > 2:
        main(sys.argv[2:], tag=sys.argv[1])
    else:
        main()
