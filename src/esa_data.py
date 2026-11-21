"""Safe loading of the ESA Anomaly Dataset (ESA-AD), Mission 1.

Channel files are zip-compressed pandas pickles (pandas 1.5.3) stored inside the
mission zip. Unpickling can execute code, so `RestrictedUnpickler` only resolves
pandas and numpy classes (plus builtins.slice); anything else raises an error.
Loaded channels are cached as .npz (timestamps in ns, float values), so each
pickle is opened once.

usage: esa_data.py [channel ...]   -> loads and caches the given channels
"""
import io
import os
from pathlib import Path
import pickle
import sys
import zipfile

import numpy as np
import pandas as pd

ESA_ROOT = Path(os.environ.get("ESA_ROOT", Path(__file__).resolve().parents[1] / "data" / "esa"))  # holds ESA-Mission1.zip
MISSION_ZIP = ESA_ROOT / "ESA-Mission1.zip"
MISSION_DIR = ESA_ROOT / "ESA-Mission1"
CACHE = ESA_ROOT / "cache"

ALLOWED_PREFIXES = ("pandas.", "numpy.", "numpy")
ALLOWED_BUILTINS = {("builtins", "slice")}

try:  # pandas maps renamed classes from older pickles here
    from pandas.compat.pickle_compat import Unpickler as _Base
except ImportError:  # pragma: no cover
    _Base = pickle.Unpickler


class RestrictedUnpickler(_Base):
    def find_class(self, module, name):
        if (module, name) in ALLOWED_BUILTINS or module.startswith(ALLOWED_PREFIXES):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked: {module}.{name}")


def _read_pickle_bytes(channel: str) -> bytes:
    with zipfile.ZipFile(MISSION_ZIP) as outer:
        inner = zipfile.ZipFile(io.BytesIO(outer.read(f"ESA-Mission1/channels/{channel}.zip")))
        return inner.read(inner.infolist()[0])


def load_channel(channel: str) -> pd.Series:
    """Raw telemetry of one channel as a float Series indexed by UTC timestamp."""
    cached = CACHE / f"{channel}.npz"
    if cached.exists():
        z = np.load(cached)
        return pd.Series(z["v"], index=pd.to_datetime(z["t"], utc=True), name=channel)
    obj = RestrictedUnpickler(io.BytesIO(_read_pickle_bytes(channel))).load()
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"{channel}: expected a DataFrame, got {type(obj).__name__}")
    s = obj.iloc[:, 0].astype(float)
    idx = pd.DatetimeIndex(s.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    s.index, s.name = idx, channel
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cached, t=idx.asi8, v=s.to_numpy())
    return s


def load_labels() -> pd.DataFrame:
    labels = pd.read_csv(MISSION_DIR / "labels.csv", parse_dates=["StartTime", "EndTime"])
    types = pd.read_csv(MISSION_DIR / "anomaly_types.csv")
    return labels.merge(types[["ID", "Category"]], on="ID", how="left")


def load_channel_info() -> pd.DataFrame:
    return pd.read_csv(MISSION_DIR / "channels.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for ch in sys.argv[1:]:
        s = load_channel(ch)
        print(f"{ch}: {len(s):,} samples, {s.index[0]} to {s.index[-1]}, dtype {s.dtype}", flush=True)
