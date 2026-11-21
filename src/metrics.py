"""Classification, calibration and selective-prediction metrics."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)


def ece(y, p, n_bins=10):
    """Expected calibration error of the anomaly probability (equal-width bins)."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return total


def risk_coverage(conf, err):
    """Selective risk when accepting the most confident fraction. Returns coverage, risk, AURC."""
    order = np.argsort(-conf, kind="stable")
    e = err[order].astype(float)
    n = len(e)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(e) / np.arange(1, n + 1)
    return cov, risk, float(risk.mean())


def risk_at(conf, err, coverage):
    _, risk, _ = risk_coverage(conf, err)
    return float(risk[int(np.ceil(coverage * len(err))) - 1])


def all_metrics(y, p, thr=0.5, calibrated=True):
    """Official OPSSAT-AD metrics plus calibration and selective risk (confidence = max(p, 1-p))."""
    y = np.asarray(y)
    yhat = (p >= thr).astype(int)
    err = yhat != y
    conf = np.maximum(p, 1 - p)
    out = {
        "Accuracy": accuracy_score(y, yhat), "Precision": precision_score(y, yhat, zero_division=0),
        "Recall": recall_score(y, yhat), "F1": f1_score(y, yhat), "MCC": matthews_corrcoef(y, yhat),
        "AUC_PR": average_precision_score(y, p), "AUC_ROC": roc_auc_score(y, p),
        "FPR": float(yhat[y == 0].mean()),
    }
    if calibrated:
        out.update(Brier=brier_score_loss(y, p), ECE=ece(y, p), AURC=risk_coverage(conf, err)[2],
                   **{f"Risk@{int(c * 100)}": risk_at(conf, err, c) for c in (0.7, 0.8, 0.9)})
    return out
