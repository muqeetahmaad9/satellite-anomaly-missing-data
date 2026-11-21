"""Did observation loss delete the anomaly itself?

OPSSAT-AD labels whole segments, so the anomalous region is estimated: in channels where
runs of exact zeros never occur in normal training segments (the magnetometers), the zero-run
is the anomaly (a stuck-at-zero dropout); otherwise it is the 10% of points deviating most from a
wide rolling-median trend. An anomaly counts as deleted when >= 50% of that region was removed.
"""
import numpy as np
import pandas as pd


def zero_runs(x, min_len=3):
    """Mask of points inside runs of >= min_len consecutive exact zeros."""
    z = np.r_[False, x == 0, False]
    edges = np.flatnonzero(np.diff(z.astype(int)))
    mask = np.zeros(len(x), dtype=bool)
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= min_len:
            mask[a:b] = True
    return mask


def zero_anomaly_channels(segs, max_normal_rate=0.01):
    """Channels where zero-runs essentially never occur in NORMAL training segments."""
    out = []
    for ch in sorted({s["channel"] for s in segs.values()}):
        normal = [s for s in segs.values() if s["channel"] == ch and s["train"] == 1 and s["anomaly"] == 0]
        if normal and np.mean([zero_runs(s["x"]).any() for s in normal]) <= max_normal_rate:
            out.append(ch)
    return out


def anomalous_region(x, criterion, zero_channel=False):
    """Boolean mask of the points most likely to carry the anomaly.

    'extreme':  top 10% of |x - median|  (misses anomalies near the median, e.g. a drop to zero)
    'residual': top 10% deviation from a wide rolling median (window = 25% of the segment, >= 15);
                misses zero plateaus where the trend itself crosses zero
    'combined': zero-runs in channels where zeros never occur normally, else 'residual'
    """
    if criterion == "combined":
        zr = zero_runs(x)
        return zr if zero_channel and zr.any() else anomalous_region(x, "residual")
    if criterion == "extreme":
        dev = np.abs(x - np.median(x))
    else:
        w = max(15, len(x) // 4) | 1
        dev = np.abs(x - pd.Series(x).rolling(w, center=True, min_periods=1).median().to_numpy())
    return dev >= np.quantile(dev, 0.9)


def evidence_deleted(segs, ids, degraded, zero_chs, criterion="combined"):
    """Per segment in `ids`: True if it is anomalous and its anomalous region was mostly removed."""
    out = np.zeros(len(ids), dtype=bool)
    for i, k in enumerate(ids):
        s = segs[k]
        if s["anomaly"] != 1 or len(degraded[k]["t"]) == len(s["t"]):  # jitter moves, never removes
            continue
        dropped = ~np.isin(s["t"], degraded[k]["t"])
        region = anomalous_region(s["x"], criterion, zero_channel=s["channel"] in zero_chs)
        out[i] = dropped[region].mean() >= 0.5
    return out
