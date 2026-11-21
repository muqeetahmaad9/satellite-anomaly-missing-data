"""Step 3 - data audit of OPSSAT-AD.

Confirms dataset counts against the published benchmark and measures the
natural timing irregularity of every segment (the signal this study is about).

Outputs:
    results/tables/audit_counts.csv
    results/tables/segment_quality.csv   (one row per segment, used by later steps)
    results/audit_report.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TABLES = ROOT / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def quality_features(seg: pd.DataFrame) -> dict:
    """Timing / data-quality descriptors for one segment.

    `sampling` is the nominal sampling period in seconds. Any time step larger
    than 1.5x the nominal period is counted as a gap (one or more missing
    observations).
    """
    # total_seconds() is independent of the datetime64 resolution (ns/us/ms)
    t = (seg["timestamp"] - seg["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    nominal = float(seg["sampling"].iloc[0])
    dt = np.diff(t)
    n = len(t)
    expected = (t[-1] - t[0]) / nominal + 1 if n > 1 else 1
    gap_mask = dt > 1.5 * nominal
    missing_per_gap = np.clip(np.round(dt / nominal) - 1, 0, None)
    return {
        "n_obs": n,
        "nominal_sampling_s": nominal,
        "duration_s": t[-1] - t[0],
        "expected_obs": expected,
        "missing_frac": max(0.0, 1 - n / expected),
        "n_gaps": int(gap_mask.sum()),
        "longest_gap_s": float(dt.max()) if n > 1 else 0.0,
        "longest_gap_rel": float(dt.max() / nominal) if n > 1 else 0.0,
        "missing_obs_est": float(missing_per_gap.sum()),
        "dt_median_s": float(np.median(dt)) if n > 1 else 0.0,
        "dt_cv": float(dt.std() / dt.mean()) if n > 2 and dt.mean() > 0 else 0.0,
        "jitter_frac": float(np.mean(np.abs(dt - nominal) > 0.1 * nominal)) if n > 1 else 0.0,
        "nonpositive_dt": int((dt <= 0).sum()),
    }


def main() -> None:
    seg = pd.read_csv(RAW / "segments.csv", parse_dates=["timestamp"])
    ds = pd.read_csv(RAW / "dataset.csv")
    lines = []
    log = lambda s="": (print(s), lines.append(s))

    log("=== Raw file: segments.csv ===")
    log(f"rows (raw measurements): {len(seg):,}")
    log(f"segments: {seg.segment.nunique():,}")
    log(f"channels: {seg.channel.nunique()} -> {sorted(seg.channel.unique())}")
    log(f"nominal sampling periods (s): {sorted(seg.sampling.unique())}")
    log(f"missing values in any column: {int(seg.isna().sum().sum())}")

    # every segment must have one label, one split flag, one channel
    per = seg.groupby("segment").agg(
        anomaly=("anomaly", "nunique"), train=("train", "nunique"), channel=("channel", "nunique"))
    log(f"segments with mixed label/split/channel: {int((per > 1).any(axis=1).sum())}")

    log("\n=== Feature file: dataset.csv ===")
    log(f"rows: {len(ds):,}  columns: {ds.shape[1]}")
    counts = ds.groupby(["train", "anomaly"]).segment.count().unstack()
    counts.index = counts.index.map({1: "train", 0: "test"})
    counts.columns = counts.columns.map({0: "normal", 1: "anomaly"})
    counts["total"] = counts.sum(axis=1)
    counts.loc["all"] = counts.sum()
    log(counts.to_string())
    counts.to_csv(TABLES / "audit_counts.csv")
    log(f"anomaly fraction: {ds.anomaly.mean():.3f}")

    log("\n=== Per-channel counts (train/test x normal/anomaly) ===")
    pc = ds.groupby(["channel", "train", "anomaly"]).segment.count().unstack([1, 2]).fillna(0).astype(int)
    log(pc.to_string())

    log("\n=== Timing quality per segment ===")
    rows = []
    for sid, g in seg.sort_values(["segment", "timestamp"]).groupby("segment"):
        q = quality_features(g)
        q.update(segment=sid, channel=g.channel.iloc[0],
                 anomaly=int(g.anomaly.iloc[0]), train=int(g.train.iloc[0]))
        rows.append(q)
    qdf = pd.DataFrame(rows).set_index("segment").sort_index()
    qdf.to_csv(TABLES / "segment_quality.csv")

    cols = ["n_obs", "missing_frac", "n_gaps", "longest_gap_rel", "dt_cv", "jitter_frac"]
    log(qdf[cols].describe(percentiles=[.5, .9, .99]).round(3).to_string())
    log(f"\nsegments with >=1 gap: {(qdf.n_gaps > 0).sum()} ({(qdf.n_gaps > 0).mean():.1%})")
    log(f"segments with non-positive time steps: {(qdf.nonpositive_dt > 0).sum()}")

    log("\n=== Does timing quality differ between normal and anomalous segments? ===")
    log(qdf.groupby("anomaly")[cols].median().round(4).to_string())
    log("\nshare with >=1 gap, by label:")
    log(qdf.groupby("anomaly").apply(lambda d: (d.n_gaps > 0).mean(), include_groups=False).round(3).to_string())

    (ROOT / "results" / "audit_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
