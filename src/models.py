"""Detectors with a common interface: fit(segments, y) and predict_proba(segments) -> P(anomaly).

`segments` is a list of segment dicts (see features.load_segments / degrade.degrade_all),
so every detector can be re-run on degraded copies of the same segments.

Feature models use the 18 OPSSAT-AD benchmark features. Neural models use the
raw signal resampled to a fixed length by linear interpolation over time, the
input representation used by OpsSatNet (Peng et al., 2026).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from degrade import degrade
from features import BENCHMARK_FEATURES, benchmark_features

CHANNELS = ["CADC0872", "CADC0873", "CADC0874", "CADC0884", "CADC0886",
            "CADC0888", "CADC0890", "CADC0892", "CADC0894"]
SEQ_LEN = 256


# ------------------------------------------------------------------ features
def feature_matrix(segments) -> np.ndarray:
    X = pd.DataFrame([benchmark_features(s["t"], s["x"], s["sampling"]) for s in segments])
    return X[BENCHMARK_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()


class FeatureModel:
    calibrated = True

    def __init__(self, kind: str, seed: int):
        self.kind, self.seed = kind, seed
        if kind == "LogReg":
            self.est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed))
        elif kind == "RandomForest":
            self.est = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
        elif kind == "XGBoost":
            self.est = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
                                     colsample_bytree=0.9, random_state=seed, n_jobs=-1, eval_metric="logloss")
        else:
            raise ValueError(kind)

    def fit(self, segments, y):
        self.est.fit(feature_matrix(segments), y)
        return self

    def predict_proba(self, segments, X=None):
        return self.est.predict_proba(feature_matrix(segments) if X is None else X)[:, 1]


class IForestModel:
    """Unsupervised reference, trained on normal training segments only.

    Scores are mapped to [0, 1] by their rank among training scores, and the
    0.5 threshold is placed at the training anomaly rate (contamination); they
    are not calibrated probabilities.
    """
    calibrated = False

    def __init__(self, seed: int):
        self.scaler = StandardScaler()
        self.est = IsolationForest(n_estimators=500, random_state=seed, n_jobs=-1)

    def fit(self, segments, y):
        X = feature_matrix(segments)
        Xn = self.scaler.fit_transform(X[np.asarray(y) == 0])
        self.est.fit(Xn)
        self.ref = np.sort(-self.est.score_samples(self.scaler.transform(X)))
        self.cut = np.quantile(self.ref, 1 - np.mean(y))
        return self

    def predict_proba(self, segments, X=None):
        s = -self.est.score_samples(self.scaler.transform(feature_matrix(segments) if X is None else X))
        below = s < self.cut
        # piecewise-linear map: training-score rank below the cut -> [0, 0.5), above -> [0.5, 1]
        r = np.searchsorted(self.ref, s) / len(self.ref)
        rc = np.searchsorted(self.ref, self.cut) / len(self.ref)
        return np.where(below, 0.5 * r / max(rc, 1e-9), 0.5 + 0.5 * (r - rc) / max(1 - rc, 1e-9))


# ------------------------------------------------------------------ sequences
def to_sequence(s: dict, length: int = SEQ_LEN) -> np.ndarray:
    grid = np.linspace(s["t"][0], s["t"][-1], length)
    return np.interp(grid, s["t"], s["x"])


class _PlainCNN(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(width, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(width, 1))

    def forward(self, x, ch):
        return self.head(self.net(x).mean(-1)).squeeze(-1)


class _ResBlock(nn.Module):
    def __init__(self, c, dilation, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(c, c, 3, padding=dilation, dilation=dilation)
        self.bn1, self.bn2 = nn.BatchNorm1d(c), nn.BatchNorm1d(c)
        self.film = nn.Linear(emb_dim, 2 * c)  # feature-wise linear modulation by channel

    def forward(self, h, e):
        z = F.gelu(self.bn1(self.conv1(h)))
        z = self.bn2(self.conv2(z))
        gamma, beta = self.film(e).unsqueeze(-1).chunk(2, dim=1)
        return F.gelu(h + (1 + gamma) * z + beta)


class _SensorAwareNet(nn.Module):
    """OpsSatNet-style: dilated residual CNN + channel FiLM + additive attention pooling.

    Re-implemented from the paper's description; conditioning is on the 9
    telemetry channels rather than the 2 sensor families.
    """

    def __init__(self, width=64, emb_dim=16, dilations=(1, 2, 4, 8), n_channels=len(CHANNELS)):
        super().__init__()
        self.emb = nn.Embedding(n_channels, emb_dim)
        self.stem = nn.Sequential(nn.Conv1d(1, width, 7, padding=3), nn.BatchNorm1d(width), nn.GELU())
        self.blocks = nn.ModuleList(_ResBlock(width, d, emb_dim) for d in dilations)
        self.att = nn.Sequential(nn.Conv1d(width, 32, 1), nn.Tanh(), nn.Conv1d(32, 1, 1))
        self.head = nn.Sequential(nn.Linear(width + emb_dim, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x, ch):
        e = self.emb(ch)
        h = self.stem(x)
        for b in self.blocks:
            h = b(h, e)
        w = torch.softmax(self.att(h), dim=-1)
        pooled = (h * w).sum(-1)
        return self.head(torch.cat([pooled, e], dim=1)).squeeze(-1)


class SequenceModel:
    calibrated = True

    def __init__(self, kind: str, seed: int, epochs=150, patience=20, lr=1e-3, batch=64, label_smoothing=0.0):
        self.kind, self.seed = kind, seed
        self.epochs, self.patience, self.lr, self.batch = epochs, patience, lr, batch
        self.label_smoothing = label_smoothing  # remedy for gap-induced overconfidence (Nanopoulos & Buza, 2025)

    def _tensors(self, segments):
        channels = getattr(self, "channels", CHANNELS)  # models saved before ESA support used the OPS-SAT list
        X = np.stack([to_sequence(s) for s in segments])
        ch = np.array([channels.index(s["channel"]) for s in segments])
        X = (X - self.mu[ch, None]) / self.sd[ch, None]
        return torch.tensor(X[:, None, :], dtype=torch.float32), torch.tensor(ch)

    def fit(self, segments, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        y = np.asarray(y)
        self.channels = sorted({s["channel"] for s in segments})
        # per-channel normalisation statistics from training values only
        self.mu, self.sd = np.zeros(len(self.channels)), np.ones(len(self.channels))
        for i, c in enumerate(self.channels):
            v = np.concatenate([s["x"] for s in segments if s["channel"] == c])
            self.mu[i], self.sd[i] = v.mean(), v.std() + 1e-12
        idx_tr, idx_va = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=self.seed)
        X, ch = self._tensors(segments)
        yt = torch.tensor(y, dtype=torch.float32)
        yt_train = yt * (1 - self.label_smoothing) + self.label_smoothing / 2
        self.net = _PlainCNN() if self.kind == "CNN" else _SensorAwareNet(n_channels=len(self.channels))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        best, best_state, wait = np.inf, None, 0
        g = torch.Generator().manual_seed(self.seed)
        for epoch in range(self.epochs):
            self.net.train()
            perm = idx_tr[torch.randperm(len(idx_tr), generator=g).numpy()]
            for i in range(0, len(perm), self.batch):
                b = perm[i:i + self.batch]
                loss = F.binary_cross_entropy_with_logits(self.net(X[b], ch[b]), yt_train[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = F.binary_cross_entropy_with_logits(self.net(X[idx_va], ch[idx_va]), yt[idx_va]).item()
            if vl < best - 1e-4:
                best, wait = vl, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.epochs_run, self.best_val_loss = epoch + 1, best
        self.n_params = sum(p.numel() for p in self.net.parameters())
        return self

    def predict_proba(self, segments, X=None):
        Xs, ch = self._tensors(segments)
        self.net.eval()
        with torch.no_grad():
            return torch.sigmoid(self.net(Xs, ch)).numpy()


class AugmentedModel:
    """Gap-augmented training: each training segment is joined by `copies` degraded
    copies (observation loss of a random fraction in [0.05, 0.3], label unchanged).

    kinds = ("random", "block") alternates the gap type across copies;
    kinds = ("random",) trains on scattered gaps only (to test unseen block gaps).
    """

    def __init__(self, base, kinds, seed: int, copies=2, rates=(0.05, 0.3)):
        self.base, self.kinds, self.seed, self.copies, self.rates = base, kinds, seed, copies, rates
        self.calibrated = base.calibrated

    def fit(self, segments, y):
        rng = np.random.default_rng(10_000 + self.seed)
        aug, y_aug = list(segments), list(y)
        for c in range(self.copies):
            kind = self.kinds[c % len(self.kinds)]
            for s, label in zip(segments, y):
                t, x = degrade(s, kind, rng.uniform(*self.rates), rng)
                aug.append({**s, "t": t, "x": x})
                y_aug.append(label)
        self.base.fit(aug, np.array(y_aug))
        for attr in ("n_params", "epochs_run"):
            if hasattr(self.base, attr):
                setattr(self, attr, getattr(self.base, attr))
        return self

    def predict_proba(self, segments, X=None):
        return self.base.predict_proba(segments, X=X)


def make_model(name: str, seed: int):
    if name.endswith("-AugR"):
        return AugmentedModel(make_model(name[:-5], seed), ("random",), seed)
    if name.endswith("-Aug"):
        return AugmentedModel(make_model(name[:-4], seed), ("random", "block"), seed)
    if name in ("LogReg", "RandomForest", "XGBoost"):
        return FeatureModel(name, seed)
    if name == "IsolationForest":
        return IForestModel(seed)
    if name in ("CNN", "SensorAwareNet"):
        return SequenceModel(name, seed)
    if name == "SensorAwareNet-LS":
        return SequenceModel("SensorAwareNet", seed, label_smoothing=0.1)
    raise ValueError(name)


MODEL_NAMES = ["LogReg", "RandomForest", "XGBoost", "IsolationForest", "CNN", "SensorAwareNet"]
