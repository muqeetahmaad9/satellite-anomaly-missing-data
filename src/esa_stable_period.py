"""ESA gap effect on the low-drift test years (2007-2008).

From 2009 the test-period normal behaviour of channels 41-46 shifts away from 2000-2006
(clean false-alarm rate rises from <= 8% to 31-85%), which masks any gap effect. This repeats
the gap comparison of esa_experiment.py on test windows from 2007-2008 only, reusing its
trained models and calibration set, and adds per-year clean false-alarm rates.

Outputs: results/tables/esa_stable.csv, esa_fpr_by_year.csv, esa_stable_bootstrap.csv
"""
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from degrade import degrade_all
from esa_experiment import DRAWS, MODELS, MODELS_DIR, SEEDS, TABLES, evidence_visible
from esa_windows import OUT, load_windows
from metrics import ece
from models import feature_matrix
from uncertainty import CONFORMAL_VARIANTS, calibration_predictions, detect_stratum, stratified_sets

YEARS = (2007, 2008)
CONDITIONS = [("clean", 0.0), ("block", 0.1), ("block", 0.2), ("random", 0.1), ("random", 0.2)]
B = 2000


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(8)
    segs = load_windows()
    with np.load(OUT) as z:
        year = pd.to_datetime(z["t"][z["offsets"][:-1]], utc=True).year.to_numpy()
    tr_ids = np.array([k for k, s in segs.items() if s["train"] == 1])
    y_tr = np.array([segs[k]["anomaly"] for k in tr_ids])
    _, cal_ids = train_test_split(tr_ids, test_size=0.2, stratify=y_tr, random_state=2137)
    cal_ids = sorted(cal_ids)
    all_te = [k for k, s in segs.items() if s["train"] == 0]
    te_ids = [k for k in all_te if year[k] in YEARS]
    y = np.array([segs[k]["anomaly"] for k in te_ids])
    models = {(n, s): pickle.load(open(MODELS_DIR / f"{n}_s{s}.pkl", "rb")) for n in MODELS for s in SEEDS}
    print(f"test windows {YEARS}: {len(te_ids)} ({y.sum()} anomalous)", flush=True)

    # clean false-alarm rate per test year (documents the drift)
    te_all = [segs[k] for k in all_te]
    X_all = feature_matrix(te_all)
    y_all = np.array([s["anomaly"] for s in te_all])
    yr = year[all_te]
    by_year = []
    for (name, seed), m in models.items():
        alarm = m.predict_proba(te_all, X=X_all) >= 0.5
        for yv in sorted(set(yr)):
            sel = (yr == yv) & (y_all == 0)
            by_year.append({"model": name, "seed": seed, "year": yv, "FPR": alarm[sel].mean()})
    fy = pd.DataFrame(by_year).groupby(["model", "year"]).FPR.mean().unstack().round(3)
    fy.to_csv(TABLES / "esa_fpr_by_year.csv")

    cal_pred = calibration_predictions(models, segs, cal_ids)
    rows, per_seg = [], []
    for kind, rate in CONDITIONS:
        for draw in (DRAWS[:1] if kind == "clean" else DRAWS):
            deg = degrade_all(segs, all_te, kind, rate, draw)  # same copies as the full-period run
            te = [deg[k] for k in te_ids]
            X = feature_matrix(te)
            strata = [detect_stratum(s) for s in te]
            visible = evidence_visible(segs, te_ids, deg)
            for (name, seed), m in models.items():
                p = m.predict_proba(te, X=X)
                pred = (p >= 0.5).astype(int)
                rec = {"kind": kind, "rate": rate, "draw": draw, "model": name, "seed": seed, "alarm": pred}
                row = {k: rec[k] for k in ("kind", "rate", "draw", "model", "seed")}
                row.update(FPR=pred[y == 0].mean(), Recall_vis=pred[(y == 1) & visible].mean(), ECE=ece(y, p))
                for method in ("Conformal", "Conformal-GM"):
                    sets = stratified_sets(p, strata, CONFORMAL_VARIANTS[method], lambda st: cal_pred[(name, seed, st)])
                    row[f"cov_normal_{method}"] = sets[y == 0, 0].mean()
                    rec[f"in_set_{method}"] = sets[np.arange(len(y)), y]
                rows.append(row)
                per_seg.append(rec)
            print(f"{kind:6s} {rate:.2f} draw {draw} done", flush=True)

    runs = pd.DataFrame(rows)
    summ = runs.groupby(["kind", "rate", "model"], sort=False)[
        ["FPR", "Recall_vis", "ECE", "cov_normal_Conformal", "cov_normal_Conformal-GM"]].mean().round(3)
    summ.to_csv(TABLES / "esa_stable.csv")

    rng = np.random.default_rng(2137)
    normal = np.where(y == 0)[0]

    def pooled(field, kind, ml):
        return np.mean([r[field] for r in per_seg if r["kind"] == kind and r["rate"] in (0.0, 0.2)
                        and r["model"] in ml], axis=0).astype(float)

    stats = []
    for fam, ml in (("Feature models (RF, XGBoost)", ["RandomForest", "XGBoost"]), ("1D-CNN", ["CNN"])):
        for label, v in (("FPR increase, 20% block vs clean", pooled("alarm", "block", ml) - pooled("alarm", "clean", ml)),
                         ("Standard conformal normal coverage, 20% block minus clean",
                          pooled("in_set_Conformal", "block", ml) - pooled("in_set_Conformal", "clean", ml)),
                         ("Normal coverage, gap-matched minus standard, 20% block",
                          pooled("in_set_Conformal-GM", "block", ml) - pooled("in_set_Conformal", "block", ml))):
            boots = [v[rng.choice(normal, len(normal))].mean() for _ in range(B)]
            stats.append({"family": fam, "statistic": label, "estimate": v[normal].mean(),
                          "ci_low": np.quantile(boots, 0.025), "ci_high": np.quantile(boots, 0.975)})
    boot = pd.DataFrame(stats).round(3)
    boot.to_csv(TABLES / "esa_stable_bootstrap.csv", index=False)
    with pd.option_context("display.width", 250):
        print("\n=== Clean false-alarm rate by test year ===\n" + fy.to_string())
        print(f"\n=== Test years {YEARS} ===\n" + summ.to_string())
        print("\n=== Bootstrap ===\n" + boot.to_string(index=False))


if __name__ == "__main__":
    main()
