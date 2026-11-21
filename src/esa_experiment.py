"""External validation on ESA Mission 1 (light protocol).

Random forest, XGBoost and 1D-CNN, 3 seeds each, trained on 80% of the 2000-2006
windows; the other 20% is the calibration set. Test windows (2007-2013) are evaluated
clean and under block / scattered loss of 10% and 20% (3 draws), with
  - false-alarm rate, recall over all anomalies and over anomalies whose evidence survived
    (exact: anomaly samples are known from ESA's interval labels), ECE;
  - max-probability, standard conformal and gap-matched conformal decision rules.
Bootstrap CIs (2,000 resamples of test windows) for the main claims.

Outputs: results/tables/esa_runs.csv, esa_summary.csv, esa_bootstrap.csv, results/esa_log.txt
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
from degrade import degrade_all
from esa_windows import load_windows
from metrics import ece
from models import feature_matrix, make_model
from uncertainty import (CONFORMAL_VARIANTS, calibration_predictions, decision_metrics, detect_stratum,
                         stratified_sets)

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
MODELS_DIR = ROOT / "results" / "models_esa"
MODELS = ["RandomForest", "XGBoost", "CNN"]
SEEDS = [0, 1, 2]
CONDITIONS = [("clean", 0.0), ("block", 0.1), ("block", 0.2), ("random", 0.1), ("random", 0.2)]
DRAWS = [100, 101, 102]
B = 2000


def evidence_visible(segs, ids, deg):
    """True for normal windows and for anomalous windows that kept >= 50% of their anomalous samples."""
    out = np.ones(len(ids), dtype=bool)
    for i, k in enumerate(ids):
        s = segs[k]
        if s["anomaly"] == 1 and s["anom_mask"].any():
            dropped = ~np.isin(s["t"], deg[k]["t"])
            out[i] = dropped[s["anom_mask"]].mean() < 0.5
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(8)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    segs = load_windows()
    tr_ids = np.array([k for k, s in segs.items() if s["train"] == 1])
    te_ids = [k for k, s in segs.items() if s["train"] == 0]
    y_tr = np.array([segs[k]["anomaly"] for k in tr_ids])
    fit_ids, cal_ids = train_test_split(tr_ids, test_size=0.2, stratify=y_tr, random_state=2137)
    fit_ids, cal_ids = sorted(fit_ids), sorted(cal_ids)
    y_fit = np.array([segs[k]["anomaly"] for k in fit_ids])
    y = np.array([segs[k]["anomaly"] for k in te_ids])
    print(f"fit {len(fit_ids)} ({y_fit.sum()} anom) | cal {len(cal_ids)} | test {len(te_ids)} ({y.sum()} anom)", flush=True)

    models = {}
    for name in MODELS:
        for seed in SEEDS:
            path = MODELS_DIR / f"{name}_s{seed}.pkl"
            if path.exists():
                models[(name, seed)] = pickle.load(open(path, "rb"))
                continue
            t0 = time.time()
            m = make_model(name, seed).fit([segs[k] for k in fit_ids], y_fit)
            pickle.dump(m, open(path, "wb"))
            models[(name, seed)] = m
            print(f"trained {name} seed {seed} ({time.time() - t0:.0f}s)", flush=True)

    print("scoring calibration sets ...", flush=True)
    cal_pred = calibration_predictions(models, segs, cal_ids)

    rows, per_seg = [], []
    for kind, rate in CONDITIONS:
        for draw in (DRAWS[:1] if kind == "clean" else DRAWS):
            deg = degrade_all(segs, te_ids, kind, rate, draw)
            te = [deg[k] for k in te_ids]
            X = feature_matrix(te)
            strata = [detect_stratum(s) for s in te]
            visible = evidence_visible(segs, te_ids, deg)
            for (name, seed), m in models.items():
                p = m.predict_proba(te, X=X)
                pred = (p >= 0.5).astype(int)
                base = {"kind": kind, "rate": rate, "draw": draw, "model": name, "seed": seed}
                p_c, _ = cal_pred[(name, seed, ("none", 0.0))]
                tau = np.quantile(np.maximum(p_c, 1 - p_c), 0.2)
                av = (y == 1) & visible
                rows.append({**base, "method": "MaxProb",
                             **decision_metrics(y, pred, np.maximum(p, 1 - p) >= tau, visible),
                             "FPR": pred[y == 0].mean(), "Recall": pred[y == 1].mean(),
                             "Recall_vis": pred[av].mean(), "deleted_frac": 1 - av.sum() / (y == 1).sum(),
                             "ECE": ece(y, p)})
                rec = {**base, "alarm": pred}
                for method in ("Conformal", "Conformal-GM"):
                    sets = stratified_sets(p, strata, CONFORMAL_VARIANTS[method], lambda st: cal_pred[(name, seed, st)])
                    single = sets.sum(1) == 1
                    cpred = np.where(sets[:, 1] & ~sets[:, 0], 1, 0)
                    rows.append({**base, "method": method, **decision_metrics(y, cpred, single, visible),
                                 "set_cov_normal": sets[y == 0, 0].mean(), "set_cov_anomaly": sets[y == 1, 1].mean()})
                    rec[f"in_set_{method}"] = sets[np.arange(len(y)), y]
                per_seg.append(rec)
            print(f"{kind:6s} {rate:.2f} draw {draw} done", flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(TABLES / "esa_runs.csv", index=False)
    cols = ["FPR", "Recall", "Recall_vis", "deleted_frac", "ECE", "coverage", "sel_risk", "FA_auto",
            "miss_auto_vis", "set_cov_normal", "set_cov_anomaly"]
    summary = runs.groupby(["kind", "rate", "model", "method"], sort=False)[cols].mean().round(3)
    summary.to_csv(TABLES / "esa_summary.csv")

    # bootstrap over test windows (paired), pooled over seeds and draws
    rng = np.random.default_rng(2137)
    normal = np.where(y == 0)[0]

    def pooled(field, kind, model_list):
        return np.mean([r[field] for r in per_seg if r["kind"] == kind and r["rate"] in (0.0, 0.2)
                        and r["model"] in model_list], axis=0).astype(float)

    stats = []
    for fam, ml in (("Feature models (RF, XGBoost)", ["RandomForest", "XGBoost"]), ("1D-CNN", ["CNN"])):
        d_fpr = pooled("alarm", "block", ml) - pooled("alarm", "clean", ml)
        d_cov = pooled("in_set_Conformal-GM", "block", ml) - pooled("in_set_Conformal", "block", ml)
        for label, v in (("FPR increase, 20% block vs clean", d_fpr),
                         ("Normal coverage, gap-matched minus standard conformal, 20% block", d_cov)):
            boots = [v[rng.choice(normal, len(normal))].mean() for _ in range(B)]
            stats.append({"family": fam, "statistic": label, "estimate": v[normal].mean(),
                          "ci_low": np.quantile(boots, 0.025), "ci_high": np.quantile(boots, 0.975)})
    boot = pd.DataFrame(stats).round(3)
    boot.to_csv(TABLES / "esa_bootstrap.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_rows", 200):
        m = runs[runs.method == "MaxProb"].groupby(["kind", "rate", "model"], sort=False)[
            ["FPR", "Recall", "Recall_vis", "deleted_frac", "ECE"]].mean().round(3)
        print("\n=== Detection (threshold 0.5) ===\n" + m.to_string())
        c = summary.reset_index()
        c = c[(c.method != "MaxProb") & (c.rate.isin([0.0, 0.2]))]
        print("\n=== Conformal at 20% loss ===\n" + c[["kind", "model", "method", "coverage", "sel_risk",
                                                       "set_cov_normal", "set_cov_anomaly"]].to_string(index=False))
        print("\n=== Bootstrap ===\n" + boot.to_string(index=False))


if __name__ == "__main__":
    main()
