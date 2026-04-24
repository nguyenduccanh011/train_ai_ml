"""GRU sequence classifier — sklearn-compatible API.

Used for V37d: sequence model on tabular features over a sliding window.
Architecture: input (B, W, F) -> GRU(hidden, layers) -> mean pool -> Linear -> n_classes.
"""
from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.base import BaseEstimator, ClassifierMixin


class _GRUNet(nn.Module if HAS_TORCH else object):
    def __init__(self, n_features, hidden=64, n_layers=2, n_classes=3, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)


class GRUClassifier(ClassifierMixin, BaseEstimator):
    """sklearn-compat: fit/predict/predict_proba for 3-class sequence classification.

    Expects flat 2D input (n_samples, n_features). Internally builds sliding
    windows of length `window` per sample order. NOTE: assumes rows are
    chronologically sorted within a single symbol — caller must pass per-symbol
    blocks. For fit(): we treat the entire matrix as one stream (acceptable for
    our use-case where the LightGBM path also pools across symbols).
    """

    def __init__(self, window=20, hidden=64, n_layers=2, dropout=0.2,
                 epochs=8, batch_size=512, lr=1e-3, n_classes=3,
                 device="auto", random_state=42, verbose=False):
        if not HAS_TORCH:
            raise ImportError("torch is required for GRUClassifier")
        self.window = window
        self.hidden = hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_classes = n_classes
        self.random_state = random_state
        self.verbose = verbose
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_ = None
        self.classes_ = None

    def _build_windows(self, X):
        """Build sliding windows of length `window` over X (n, F).

        Returns (n, window, F). Pads first (window-1) rows by repeating row[0].
        """
        n, f = X.shape
        w = self.window
        if n < w:
            pad = np.tile(X[0:1], (w - n, 1)) if n > 0 else np.zeros((w, f), dtype=np.float32)
            X = np.concatenate([pad, X], axis=0)
            n = X.shape[0]
        idx = np.arange(w)[None, :] + np.arange(n - w + 1)[:, None]
        windows = X[idx]
        # Pad start: first w-1 rows reuse the first window to keep n alignment
        first = np.repeat(windows[0:1], w - 1, axis=0)
        return np.concatenate([first, windows], axis=0).astype(np.float32)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64)
        # Map y to 0..n_classes-1 (handle -1, 0, 1 -> 0, 1, 2)
        self.classes_ = np.array(sorted(np.unique(y)))
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[v] for v in y], dtype=np.int64)

        Xw = self._build_windows(X)
        # Class balance weights
        counts = np.bincount(y_idx, minlength=len(self.classes_))
        weights = (counts.sum() / np.maximum(counts, 1)) ** 0.5
        weights = weights / weights.sum() * len(self.classes_)
        cw = torch.tensor(weights, dtype=torch.float32, device=self.device)

        self.model_ = _GRUNet(X.shape[1], self.hidden, self.n_layers,
                              len(self.classes_), self.dropout).to(self.device)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss(weight=cw)

        ds = TensorDataset(torch.from_numpy(Xw), torch.from_numpy(y_idx))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model_.train()
        for epoch in range(self.epochs):
            tot = 0.0
            n_batches = 0
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad()
                logits = self.model_(xb)
                loss = crit(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step()
                tot += float(loss.item())
                n_batches += 1
            if self.verbose:
                print(f"  GRU epoch {epoch+1}/{self.epochs} loss={tot/max(n_batches,1):.4f}")
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xw = self._build_windows(X)
        self.model_.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(Xw), self.batch_size):
                xb = torch.from_numpy(Xw[i:i + self.batch_size]).to(self.device)
                logits = self.model_(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                outs.append(probs)
        return np.concatenate(outs, axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]
