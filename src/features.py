"""Feature extraction for OPSSAT-AD segments.

`benchmark_features` re-implements the 18 features of the official
dataset_generator.ipynb (Zenodo record 15108715) so that features can be
recomputed after controlled degradation of a segment. `check_against_official`
verifies the re-implementation against the published dataset.csv.

`quality_features` adds observation-quality descriptors (gaps, jitter) that
the benchmark does not expose directly.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sig
from scipy.stats import kurtosis, skew

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

BENCHMARK_FEATURES = [
    "mean", "var", "std", "len", "duration", "len_weighted", "gaps_squared", "n_peaks",
    "smooth10_n_peaks", "smooth20_n_peaks", "var_div_duration", "var_div_len",
    "diff_peaks", "diff2_peaks", "diff_var", "diff2_var", "kurtosis", "skew",
]

QUALITY_FEATURES = [
    "missing_frac", "n_gaps", "longest_gap_rel", "dt_cv", "jitter_frac", "log_n_obs",
]


def _n_peaks(x: np.ndarray) -> int:
    prominence = 0.1 * (np.max(x) - np.min(x))
    return len(sig.find_peaks(x, prominence=prominence)[0])


def _smooth_peaks(x: np.ndarray, k: int) -> int:
    return _n_peaks(np.convolve(x, np.ones(k) / k, mode="same"))


def benchmark_features(t: np.ndarray, x: np.ndarray, sampling: float) -> dict:
    """The 18 official features. `t` in seconds, `x` values, `sampling` nominal period.

    Follows the notebook exactly, including its use of whole seconds
    (timedelta.seconds) for duration and time steps.
    """
    dt = np.floor(np.diff(t)).astype(np.int64)
    duration = int(np.floor(t[-1] - t[0]))
    f = {
        "duration": duration,
        "len": len(x),
        "mean": np.mean(x),
        "var": np.var(x),
        "std": np.std(x),
        "kurtosis": kurtosis(x),
        "skew": skew(x),
        "n_peaks": _n_peaks(x),
        "smooth10_n_peaks": _smooth_peaks(x, 10),
        "smooth20_n_peaks": _smooth_peaks(x, 20),
        "diff_peaks": _n_peaks(np.diff(x)),
        "diff2_peaks": _n_peaks(np.diff(x, n=2)),
        "diff_var": np.var(np.diff(x)),
        "diff2_var": np.var(np.diff(x, n=2)),
        "gaps_squared": int((dt ** 2).sum()),
    }
    f["len_weighted"] = sampling * f["len"]
    f["var_div_duration"] = f["var"] / f["duration"] if f["duration"] else np.inf
    f["var_div_len"] = f["var"] / f["len"]
    return f


def quality_features(t: np.ndarray, sampling: float) -> dict:
    """Observation-quality descriptors. A step > 1.5x the nominal period is a gap."""
    dt = np.diff(t)
    n = len(t)
    expected = (t[-1] - t[0]) / sampling + 1
    return {
        "missing_frac": max(0.0, 1 - n / expected),
        "n_gaps": int((dt > 1.5 * sampling).sum()),
        "longest_gap_rel": float(dt.max() / sampling),
        "dt_cv": float(dt.std() / dt.mean()) if n > 2 and dt.mean() > 0 else 0.0,
        "jitter_frac": float(np.mean(np.abs(dt - sampling) > 0.1 * sampling)),
        "log_n_obs": float(np.log(n)),
    }


def load_segments() -> dict:
    """Return {segment_id: dict(t, x, sampling, channel, anomaly, train)}, t in seconds from start."""
    df = pd.read_csv(RAW / "segments.csv", parse_dates=["timestamp"])
    df = df.sort_values(["segment", "timestamp"])
    out = {}
    for sid, g in df.groupby("segment"):
        t = (g.timestamp - g.timestamp.iloc[0]).dt.total_seconds().to_numpy()
        out[int(sid)] = dict(
            t=t, x=g.value.to_numpy(dtype=float), sampling=float(g.sampling.iloc[0]),
            channel=g.channel.iloc[0], anomaly=int(g.anomaly.iloc[0]), train=int(g.train.iloc[0]))
    return out


def build_table(segments: dict) -> pd.DataFrame:
    rows = []
    for sid, s in segments.items():
        r = benchmark_features(s["t"], s["x"], s["sampling"])
        r.update(quality_features(s["t"], s["sampling"]))
        r.update(segment=sid, channel=s["channel"], anomaly=s["anomaly"], train=s["train"],
                 sampling=s["sampling"])
        rows.append(r)
    return pd.DataFrame(rows).set_index("segment").sort_index()


def check_against_official() -> None:
    ours = build_table(load_segments())
    official = pd.read_csv(RAW / "dataset.csv", index_col="segment").sort_index()
    assert (ours.index == official.index).all()
    assert (ours.anomaly == official.anomaly).all() and (ours.train == official.train).all()
    worst = {}
    for c in BENCHMARK_FEATURES:
        a, b = ours[c].to_numpy(float), official[c].to_numpy(float)
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
        rel[(a == b)] = 0.0
        worst[c] = float(np.nanmax(rel))
    report = pd.Series(worst).sort_values(ascending=False)
    print("max relative difference vs official dataset.csv:")
    print(report.to_string())
    bad = report[report > 1e-6]
    print("ALL 18 FEATURES MATCH" if bad.empty else f"MISMATCH in: {list(bad.index)}")


if __name__ == "__main__":
    check_against_official()
