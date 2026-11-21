"""ESA Mission 1 -> labelled 1-hour windows in the OPSSAT-AD segment format.

Channels 41-46 (the ESA-ADB lightweight subset) are cut into non-overlapping
clock-hour windows (120 samples at 30 s). A window is anomalous if it overlaps an
'Anomaly' interval of its channel; rare nominal events count as normal. Windows
overlapping a labelled communication gap are dropped (they are data-loss events,
not behaviour). Split as in ESA-ADB: 2000-2006 train, 2007-2013 test. All anomalous
windows are kept; normal windows are subsampled (N_NORMAL per split) to keep CPU
training practical.

Each window stores which of its samples fall inside the anomaly interval, so
evidence deletion can be measured exactly.

Output: $ESA_ROOT/windows.npz (no pickles; flat arrays + offsets)
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esa_data import ESA_ROOT, load_channel, load_labels

CHANNELS = [f"channel_{i}" for i in range(41, 47)]
SPLIT_NS = pd.Timestamp("2007-01-01", tz="UTC").value
HOUR_NS = 3600 * 10**9
MIN_SAMPLES = 8
N_NORMAL = {"train": 12000, "test": 6000}
SEED = 2137
OUT = ESA_ROOT / "windows.npz"


def intervals(labels, channel, category):
    d = labels[(labels.Channel == channel) & (labels.Category == category)]
    ns = lambda col: col.dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()  # match sample timestamps
    return ns(d.StartTime), ns(d.EndTime)


def in_any(t, starts, ends):
    """Boolean mask: t inside any closed interval [start, end]."""
    mask = np.zeros(len(t), dtype=bool)
    for a, b in zip(starts, ends):
        lo, hi = np.searchsorted(t, a, "left"), np.searchsorted(t, b, "right")
        mask[lo:hi] = True
    return mask


def build():
    labels = load_labels()
    rng = np.random.default_rng(SEED)
    chosen = []  # (channel, hour, label, train)
    store = {}
    for ch in CHANNELS:
        s = load_channel(ch)
        t, v = s.index.asi8, s.to_numpy()
        anom = in_any(t, *intervals(labels, ch, "Anomaly"))
        comm = in_any(t, *intervals(labels, ch, "Communication Gap"))
        hour = t // HOUR_NS
        cuts = np.flatnonzero(np.diff(hour)) + 1
        starts = np.r_[0, cuts]
        ends = np.r_[cuts, len(t)]
        n = ends - starts
        is_anom = np.add.reduceat(anom, starts) > 0
        has_comm = np.add.reduceat(comm, starts) > 0
        ok = (n >= MIN_SAMPLES) & ~has_comm
        train = t[starts] < SPLIT_NS
        store[ch] = (t, v, anom, starts, ends)
        for i in np.flatnonzero(ok):
            chosen.append((ch, i, int(is_anom[i]), int(train[i])))
        print(f"{ch}: {ok.sum():,} windows, {int((is_anom & ok).sum()):,} anomalous", flush=True)

    df = pd.DataFrame(chosen, columns=["channel", "w", "anomaly", "train"])
    keep = [df[df.anomaly == 1]]
    for split, flag in (("train", 1), ("test", 0)):
        normal = df[(df.anomaly == 0) & (df.train == flag)]
        keep.append(normal.sample(n=min(N_NORMAL[split], len(normal)), random_state=SEED))
    df = pd.concat(keep).sort_values(["channel", "w"]).reset_index(drop=True)

    T, V, A, offsets = [], [], [], [0]
    for ch, w in zip(df.channel, df.w):
        t, v, anom, starts, ends = store[ch]
        sl = slice(starts[w], ends[w])
        T.append(t[sl]); V.append(v[sl]); A.append(anom[sl])
        offsets.append(offsets[-1] + ends[w] - starts[w])
    np.savez(OUT, t=np.concatenate(T), v=np.concatenate(V), anom=np.concatenate(A),
             offsets=np.array(offsets), channel=df.channel.to_numpy(dtype="U12"),
             anomaly=df.anomaly.to_numpy(), train=df.train.to_numpy())
    print("\nkept windows (split x label):")
    print(pd.crosstab(df.train.map({1: "train 2000-2006", 0: "test 2007-2013"}),
                      df.anomaly.map({0: "normal", 1: "anomaly"})).to_string())
    print(f"saved {OUT}")


def load_windows() -> dict:
    """{segment_id: dict(t, x, sampling, channel, anomaly, train, anom_mask)} like features.load_segments()."""
    with np.load(OUT) as z:  # read each array once; indexing the NpzFile re-reads it from disk
        T, V, A, off = z["t"], z["v"].astype(float), z["anom"], z["offsets"]
        chans, labels, train = z["channel"], z["anomaly"], z["train"]
    out = {}
    for i in range(len(labels)):
        a, b = off[i], off[i + 1]
        out[i] = dict(t=(T[a:b] - T[a]) / 1e9, x=V[a:b], sampling=30.0, channel=str(chans[i]),
                      anomaly=int(labels[i]), train=int(train[i]), anom_mask=A[a:b])
    return out


if __name__ == "__main__":
    build()
