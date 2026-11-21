"""Controlled observation loss for OPSSAT-AD segments.

A segment is a dict with keys t (seconds from start), x (values), sampling,
channel, anomaly, train. Degradation removes observations but always keeps the
first and last sample, so the segment's time span is unchanged.
"""
import numpy as np

MIN_OBS = 8  # never degrade a segment below this many observations


def degrade(s: dict, kind: str, rate: float, rng: np.random.Generator):
    """Return (t, x) with a fraction `rate` of observations removed.

    kind = "random": isolated points removed uniformly at random.
    kind = "block":  one contiguous run of points removed.
    kind = "jitter": nothing removed; inner timestamps shifted by uniform noise of
                     +- rate x nominal sampling period (rate < 0.5 keeps the order).
    """
    n = len(s["t"])
    if kind == "jitter":
        t = s["t"].copy()
        t[1:-1] += rng.uniform(-rate, rate, n - 2) * s["sampling"]
        return t, s["x"]
    k = int(round(rate * n))
    if k == 0 or n - k < MIN_OBS:
        return s["t"], s["x"]
    inner = np.arange(1, n - 1)
    if kind == "random":
        drop = rng.choice(inner, size=min(k, len(inner)), replace=False)
    elif kind == "block":
        start = rng.integers(1, max(2, n - 1 - k))
        drop = np.arange(start, min(start + k, n - 1))
    else:
        raise ValueError(kind)
    keep = np.setdiff1d(np.arange(n), drop)
    return s["t"][keep], s["x"][keep]


def degrade_all(segments: dict, ids, kind: str, rate: float, seed: int) -> dict:
    """Degraded copies of the segments `ids` (same keys, new t and x)."""
    rng = np.random.default_rng(seed)
    out = {}
    for sid in ids:
        s = segments[sid]
        t, x = (s["t"], s["x"]) if rate == 0 else degrade(s, kind, rate, rng)
        out[sid] = {**s, "t": t, "x": x}
    return out
