"""Step 9 - statistical robustness of the key claims.

(a) Bootstrap 95% CIs over test segments (2,000 resamples, stratified by class),
    pooled over 5 model seeds x 3 degradation draws, for:
      - false-alarm rate (threshold 0.5) clean vs 20% block loss
      - normal-class conformal set coverage, standard vs gap-matched, 20% block loss
      - share of anomalies missed AND automated (MaxProb), clean vs 20% block loss
    Differences are paired (same resampled segments).
(b) Calibration-size sensitivity: conformal set coverage when the calibration set is
    subsampled to 25 / 50 / 100% (20 stratified repeats).
(c) Anomaly-deletion check: under 20% block loss, recall for anomalous segments whose anomalous
    region (evidence.py) was removed vs kept, under three region definitions.

Outputs: results/tables/stats_bootstrap.csv, stats_calsize.csv, stats_deletion.csv
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from degrade import degrade_all
from evidence import anomalous_region, evidence_deleted, zero_anomaly_channels
from features import load_segments
from models import feature_matrix
from run_baselines import SEEDS, TABLES
from run_calibrated import calibration_split
from uncertainty import (CAL_DRAWS, CONFORMAL_VARIANTS, TEST_DRAWS, calibration_predictions, detect_stratum,
                         load_models, stratified_sets)

MODELS = ["LogReg", "RandomForest", "XGBoost", "CNN", "SensorAwareNet"]
B = 2000
CONDITIONS = [("clean", 0.0), ("block", 0.2), ("random", 0.2)]


def per_segment(models, cal_pred, segs, te_ids, y_te):
    """Per-segment outcomes for every model x seed x condition x draw."""
    out = []
    zero_chs = zero_anomaly_channels(segs)
    for kind, rate in CONDITIONS:
        for draw in (TEST_DRAWS[:1] if kind == "clean" else TEST_DRAWS):
            deg = degrade_all(segs, te_ids, kind, rate, draw)
            te = [deg[k] for k in te_ids]
            X = feature_matrix(te)
            strata = [detect_stratum(s) for s in te]
            visible = ~evidence_deleted(segs, te_ids, deg, zero_chs)
            for (name, seed), m in models.items():
                p = m.predict_proba(te, X=X)
                pred = (p >= 0.5).astype(int)
                p_c, _ = cal_pred[(name, seed, ("none", 0.0))]
                tau = np.quantile(np.maximum(p_c, 1 - p_c), 0.2)
                rec = {"kind": kind, "rate": rate, "draw": draw, "model": name, "seed": seed,
                       "alarm": pred, "miss_auto": (pred == 0) & (y_te == 1) & (np.maximum(p, 1 - p) >= tau)}
                rec["miss_auto_vis"] = np.where(visible, rec["miss_auto"], np.nan)
                for method in ("Conformal", "Conformal-GM"):
                    sets = stratified_sets(p, strata, CONFORMAL_VARIANTS[method],
                                           lambda st: cal_pred[(name, seed, st)])
                    rec[f"in_set_{method}"] = sets[np.arange(len(y_te)), y_te]
                out.append(rec)
    return out


def bootstrap(recs, y_te, rng):
    """CIs of pooled per-segment rates; each statistic is (label, selector, field, class)."""
    normal, anom = np.where(y_te == 0)[0], np.where(y_te == 1)[0]
    families = {"Feature models": ["LogReg", "RandomForest", "XGBoost"], "Neural models": ["CNN", "SensorAwareNet"]}

    def pooled(field, kind, models, cls):
        rows = [r[field] for r in recs if r["kind"] == kind and r["rate"] in (0.0, 0.2) and r["model"] in models]
        return np.nanmean(np.array(rows, dtype=float), axis=0)  # per-segment average over runs

    stats = []
    for fam, models in families.items():
        stats += [
            (fam, "False-alarm rate, clean", pooled("alarm", "clean", models, 0), 0),
            (fam, "False-alarm rate, 20% block", pooled("alarm", "block", models, 0), 0),
            (fam, "Normal set coverage, conformal, 20% block", pooled("in_set_Conformal", "block", models, 0), 0),
            (fam, "Normal set coverage, gap-matched, 20% block", pooled("in_set_Conformal-GM", "block", models, 0), 0),
            (fam, "Anomalies missed and automated, clean", pooled("miss_auto", "clean", models, 1), 1),
            (fam, "Anomalies missed and automated, 20% block", pooled("miss_auto", "block", models, 1), 1),
            (fam, "Anomalies missed and automated, 20% block, evidence visible",
             pooled("miss_auto_vis", "block", models, 1), 1),
        ]
    idx_b = [(rng.choice(normal, len(normal)), rng.choice(anom, len(anom))) for _ in range(B)]
    rows = []
    for fam, label, v, cls in stats:
        boots = np.array([np.nanmean(v[nb if cls == 0 else ab]) for nb, ab in idx_b])
        sel = normal if cls == 0 else anom
        rows.append({"family": fam, "statistic": label, "estimate": np.nanmean(v[sel]),
                     "ci_low": np.quantile(boots, 0.025), "ci_high": np.quantile(boots, 0.975)})
    # paired differences
    for fam in families:
        for a, b in (("False-alarm rate, 20% block", "False-alarm rate, clean"),
                     ("Normal set coverage, gap-matched, 20% block", "Normal set coverage, conformal, 20% block"),
                     ("Anomalies missed and automated, 20% block", "Anomalies missed and automated, clean"),
                     ("Anomalies missed and automated, 20% block, evidence visible",
                      "Anomalies missed and automated, clean")):
            va = next(s for s in stats if s[0] == fam and s[1] == a)
            vb = next(s for s in stats if s[0] == fam and s[1] == b)
            d = va[2] - vb[2]
            cls = va[3]
            sel = normal if cls == 0 else anom
            boots = np.array([np.nanmean(d[nb if cls == 0 else ab]) for nb, ab in idx_b])
            rows.append({"family": fam, "statistic": f"DIFF: {a} minus {b}", "estimate": np.nanmean(d[sel]),
                         "ci_low": np.quantile(boots, 0.025), "ci_high": np.quantile(boots, 0.975)})
    return pd.DataFrame(rows)


def calibration_size(models, cal_pred, segs, cal_ids, te_ids, y_te, rng):
    """Conformal set coverage with subsampled calibration sets (clean test, and 20% block with GM)."""
    y_cal = np.array([segs[k]["anomaly"] for k in cal_ids])
    n = len(cal_ids)
    tests = {}
    for kind, rate in (("clean", 0.0), ("block", 0.2)):
        deg = degrade_all(segs, te_ids, kind, rate, TEST_DRAWS[0])
        te = [deg[k] for k in te_ids]
        tests[kind] = (te, feature_matrix(te), [detect_stratum(s) for s in te])
    rows = []
    for (name, seed), m in models.items():
        preds = {k: m.predict_proba(te, X=X) for k, (te, X, _) in tests.items()}
        for frac in (0.25, 0.5, 1.0):
            for rep in range(20 if frac < 1 else 1):
                sub = np.concatenate([rng.choice(np.where(y_cal == c)[0], int(round(frac * (y_cal == c).sum())),
                                                 replace=False) for c in (0, 1)])

                def lookup(st, sub=sub):
                    p, y = cal_pred[(name, seed, st)]
                    reps = len(p) // n
                    ix = np.concatenate([sub + r * n for r in range(reps)])
                    return p[ix], y[ix]

                for kind, method in (("clean", "Conformal"), ("block", "Conformal-GM")):
                    _, _, strata = tests[kind]
                    sets = stratified_sets(preds[kind], strata, CONFORMAL_VARIANTS[method], lookup)
                    rows.append({"model": name, "seed": seed, "cal_frac": frac, "cal_n": len(sub), "rep": rep,
                                 "test": f"{kind} / {method}",
                                 "cov_normal": sets[y_te == 0, 0].mean(), "cov_anomaly": sets[y_te == 1, 1].mean()})
    raw = pd.DataFrame(rows)
    return raw.groupby(["test", "cal_frac"])[["cal_n", "cov_normal", "cov_anomaly"]].agg(
        ["mean", "std", lambda s: s.quantile(0.05)]).round(3)


def deletion_check(models, segs, te_ids):
    """Recall on anomalous test segments under 20% block loss (the stress-test draws), split by
    whether the removed block held at least half of the segment's anomalous region."""
    anom_ids = [k for k in te_ids if segs[k]["anomaly"] == 1]
    zero_chs = zero_anomaly_channels(segs)
    print(f"zero-run anomaly channels (from training data): {zero_chs}")
    rows = []
    for draw in TEST_DRAWS:
        deg = degrade_all(segs, te_ids, "block", 0.2, draw)  # same copies as the stress test
        removed = {c: [] for c in ("extreme", "residual", "combined")}
        for k in anom_ids:
            s = segs[k]
            dropped = ~np.isin(s["t"], deg[k]["t"])
            for c in removed:
                region = anomalous_region(s["x"], c, zero_channel=s["channel"] in zero_chs)
                removed[c].append(dropped[region].mean() >= 0.5)
        te_clean = [segs[k] for k in anom_ids]
        te_deg = [deg[k] for k in anom_ids]
        Xc, Xd = feature_matrix(te_clean), feature_matrix(te_deg)
        for (name, seed), m in models.items():
            hit_c = m.predict_proba(te_clean, X=Xc) >= 0.5
            hit_d = m.predict_proba(te_deg, X=Xd) >= 0.5
            for c, flags in removed.items():
                flags = np.array(flags)
                for grp, mask in (("anomaly region removed", flags), ("anomaly region kept", ~flags)):
                    rows.append({"criterion": c, "model": name, "seed": seed, "draw": draw, "group": grp,
                                 "n": int(mask.sum()), "recall_clean": hit_c[mask].mean(),
                                 "recall_block20": hit_d[mask].mean()})
    raw = pd.DataFrame(rows)
    out = raw.groupby(["criterion", "model", "group"])[["n", "recall_clean", "recall_block20"]].mean()
    out["drop"] = out.recall_clean - out.recall_block20
    return out.round(3)


def main(only=None):
    """only='deletion' re-runs just the anomaly-deletion check."""
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(8)
    rng = np.random.default_rng(2137)
    segs = load_segments()
    _, cal_ids, te_ids = calibration_split(segs)
    y_te = np.array([segs[k]["anomaly"] for k in te_ids])
    models = load_models(MODELS)
    if only == "deletion":
        with pd.option_context("display.width", 250, "display.max_rows", 200):
            dl = deletion_check(models, segs, te_ids)
            dl.to_csv(TABLES / "stats_deletion.csv")
            print("anomaly-deletion check, 20% block loss\n" + dl.to_string())
        return
    cal_pred = calibration_predictions(models, segs, cal_ids)

    with pd.option_context("display.width", 250, "display.max_rows", 200):
        boot = bootstrap(per_segment(models, cal_pred, segs, te_ids, y_te), y_te, rng).round(3)
        boot.to_csv(TABLES / "stats_bootstrap.csv", index=False)
        print("(a) bootstrap 95% CIs\n" + boot.to_string(index=False))

        cs = calibration_size(models, cal_pred, segs, cal_ids, te_ids, y_te, rng)
        cs.to_csv(TABLES / "stats_calsize.csv")
        print("\n(b) calibration-size sensitivity (mean, sd, 5th percentile)\n" + cs.to_string())

        dl = deletion_check(models, segs, te_ids)
        dl.to_csv(TABLES / "stats_deletion.csv")
        print("\n(c) anomaly-deletion check, 20% block loss\n" + dl.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
