"""Step 4 - pilot (go / no-go).

Q1  Do feature-based baselines reproduce published OPSSAT-AD performance?
Q2  Are model errors linked to natural timing irregularity, and does adding
    quality features to confidence improve error ranking?
Q3  Shortcut test: when observations are removed from NORMAL test segments,
    do the detectors start raising false alarms?

Outputs go to results/tables/pilot_*.csv, results/figures/pilot_*.png and
results/pilot_report.txt.
"""
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from degrade import degrade
from features import (BENCHMARK_FEATURES, QUALITY_FEATURES, benchmark_features, build_table,
                      load_segments)
from metrics import risk_at, risk_coverage

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"
SEED = 2137  # seed used by the official OPSSAT-AD examples
COVERAGES = (0.7, 0.8, 0.9)

LINES = []


def log(s=""):
    print(s)
    LINES.append(str(s))


def make_models():
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED)),
        "RandomForest": RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.9, random_state=SEED, n_jobs=-1,
                                 eval_metric="logloss"),
    }


def clean(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {"Accuracy": accuracy_score(y, yhat), "Precision": precision_score(y, yhat, zero_division=0),
            "Recall": recall_score(y, yhat), "F1": f1_score(y, yhat), "MCC": matthews_corrcoef(y, yhat),
            "AUC_PR": average_precision_score(y, p), "AUC_ROC": roc_auc_score(y, p)}


def degraded_features(segments, ids, kind, rate, seed):
    rng = np.random.default_rng(seed)
    rows = {}
    for sid in ids:
        s = segments[sid]
        t, x = degrade(s, kind, rate, rng)
        rows[sid] = benchmark_features(t, x, s["sampling"])
    return clean(pd.DataFrame.from_dict(rows, orient="index")[BENCHMARK_FEATURES])


def main():
    segments = load_segments()
    df = build_table(segments)
    df.to_csv(TABLES / "features_recomputed.csv")
    tr, te = df[df.train == 1], df[df.train == 0]
    Xtr, ytr = clean(tr[BENCHMARK_FEATURES]), tr.anomaly.to_numpy()
    Xte, yte = clean(te[BENCHMARK_FEATURES]), te.anomaly.to_numpy()
    log(f"train {len(tr)} ({ytr.sum()} anomalies) | test {len(te)} ({yte.sum()} anomalies)")

    # ------------------------------------------------------------ Q1 baselines
    log("\n=== Q1. Baselines on the official split (18 benchmark features, threshold 0.5) ===")
    models, p_test, res = {}, {}, {}
    for name, m in make_models().items():
        m.fit(Xtr, ytr)
        models[name] = m
        p_test[name] = m.predict_proba(Xte)[:, 1]
        res[name] = metrics(yte, p_test[name])
    q1 = pd.DataFrame(res).T.round(3)
    log(q1.to_string())
    log("published best (Ruszczak et al. 2025): F1 0.946, MCC 0.932 (FCNN); AUC-ROC 0.992 (XGBOD)")
    q1.to_csv(TABLES / "pilot_baselines.csv")

    # ------------------------------------------------------------ Q2 errors vs quality
    log("\n=== Q2. Are errors linked to natural timing irregularity? ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    irregular_tr = (tr.dt_cv > 0).to_numpy()
    irregular_te = (te.dt_cv > 0).to_numpy()
    log(f"irregular segments (any timing irregularity): train {irregular_tr.sum()}, test {irregular_te.sum()}")
    q2_rows, rc_curves = [], {}
    for name in models:
        p_oof = cross_val_predict(make_models()[name], Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
        for split, p, y, irr, qd in (("train-OOF", p_oof, ytr, irregular_tr, tr), ("test", p_test[name], yte, irregular_te, te)):
            err = ((p >= 0.5).astype(int) != y)
            conf = np.maximum(p, 1 - p)
            e_irr, e_reg = err[irr].mean(), err[~irr].mean()
            _, pval = fisher_exact([[err[irr].sum(), (~err[irr]).sum()], [err[~irr].sum(), (~err[~irr]).sum()]])
            row = {"model": name, "split": split, "n_errors": int(err.sum()),
                   "err_rate_irregular": e_irr, "err_rate_regular": e_reg, "fisher_p": pval,
                   "AURC_conf": risk_coverage(conf, err)[2]}
            for c in COVERAGES:
                row[f"risk@{int(c*100)}"] = risk_at(conf, err, c)
            # does adding quality features to the confidence improve error ranking? (5-fold CV within split)
            if err.sum() >= 10:
                Z_conf = np.c_[conf]
                Z_full = np.c_[conf, clean(qd[QUALITY_FEATURES]).to_numpy()]
                cv2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
                r_conf = cross_val_predict(lr, Z_conf, err, cv=cv2, method="predict_proba")[:, 1]
                r_full = cross_val_predict(lr, Z_full, err, cv=cv2, method="predict_proba")[:, 1]
                row["errAUROC_conf"] = roc_auc_score(err, r_conf)
                row["errAUROC_conf+quality"] = roc_auc_score(err, r_full)
            q2_rows.append(row)
            if split == "test":
                rc_curves[name] = risk_coverage(conf, err)
    q2 = pd.DataFrame(q2_rows).round(4)
    log(q2.to_string(index=False))
    q2.to_csv(TABLES / "pilot_errors_vs_quality.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (cov, risk, aurc) in rc_curves.items():
        ax.plot(cov, risk, label=f"{name} (AURC {aurc:.4f})")
    for c in COVERAGES:
        ax.axvline(c, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("Coverage (fraction decided automatically)")
    ax.set_ylabel("Selective risk (error rate on accepted)")
    ax.set_title("Risk-coverage on OPSSAT-AD test set (clean)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "pilot_risk_coverage.png", dpi=200)

    # ------------------------------------------------------------ Q3 shortcut test
    log("\n=== Q3. Shortcut test: remove observations from test segments ===")
    ids = te.index.to_numpy()
    base_feats = Xte  # recomputed with our code, so only degradation differs
    rows = []
    for kind in ("random", "block"):
        for rate in (0.0, 0.1, 0.2, 0.3):
            for seed in range(5):
                Xd = base_feats if rate == 0 else degraded_features(segments, ids, kind, rate, SEED + seed)
                for name, m in models.items():
                    p = m.predict_proba(Xd)[:, 1]
                    yhat = (p >= 0.5).astype(int)
                    rows.append({"kind": kind, "rate": rate, "seed": seed, "model": name,
                                 "FPR_normal": yhat[yte == 0].mean(), "Recall_anomaly": yhat[yte == 1].mean(),
                                 "F1": f1_score(yte, yhat), "mean_conf": np.maximum(p, 1 - p).mean()})
                if rate == 0:
                    break
    q3 = pd.DataFrame(rows)
    q3.to_csv(TABLES / "pilot_shortcut_raw.csv", index=False)
    summ = q3.groupby(["kind", "rate", "model"])[["FPR_normal", "Recall_anomaly", "F1", "mean_conf"]].agg(["mean", "std"]).round(3)
    log(summ.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, kind in zip(axes, ("random", "block")):
        d = q3[(q3.kind == kind) | (q3.rate == 0)]
        for name in models:
            g = d[d.model == name].groupby("rate").FPR_normal
            ax.errorbar(g.mean().index * 100, g.mean(), yerr=g.std().fillna(0), marker="o", capsize=3, label=name)
        ax.set_title(f"{kind} removal")
        ax.set_xlabel("Observations removed (%)")
    axes[0].set_ylabel("False-alarm rate on normal test segments")
    axes[0].legend(frameon=False)
    fig.suptitle("Do detectors raise false alarms when normal telemetry loses data?")
    fig.tight_layout()
    fig.savefig(FIGS / "pilot_shortcut_fpr.png", dpi=200)

    # feature importance of gap-related features (XGBoost gain)
    imp = pd.Series(models["XGBoost"].get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    imp = (imp / imp.sum()).round(3)
    log("\nXGBoost feature importance (gain share):")
    log(imp.to_string())

    (ROOT / "results" / "pilot_report.txt").write_text("\n".join(LINES), encoding="utf-8")


if __name__ == "__main__":
    main()
