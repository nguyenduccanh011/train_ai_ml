"""PyTorch sequence classifier for the SMAC single-model action line.

Unlike the LightGBM entry model (which sees only a per-bar feature *snapshot*), this
model consumes a causal WINDOW of the last `window` bars per symbol and runs it through
a GRU/LSTM. The hypothesis (the user's "model thấy chuỗi thời gian"): the immediate-
bleeder losers and sub-pivot timing carry a TEMPORAL signature that a snapshot model
cannot see but a recurrent model can. Drop-in for the SMAC branch: exposes `fit`,
`predict_proba`, and a `classes_` attribute so the branch's label-indexed proba mapping
works unchanged.

Deterministic (torch.manual_seed + single-thread) so multi-seed scoring is reproducible,
matching the project's deterministic-LightGBM discipline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_seq_windows(df: pd.DataFrame, feat_cols: list[str], window: int) -> np.ndarray:
    """Build per-symbol CAUSAL feature windows aligned to `df` row order.

    Returns (n_rows, window, n_feats). For each row, the window is the last `window`
    bars of that symbol up to and including the row (sorted by date); short histories
    are left-padded by repeating the symbol's earliest available bar (edge-pad, no
    look-ahead). Output rows align positionally to `df` so they pair with y_full /
    test_use exactly as the 2D path does.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    mat = df[feat_cols].to_numpy(dtype=np.float32)
    syms = df["symbol"].to_numpy()
    dates = df["date"].to_numpy()
    n, f = mat.shape
    out = np.zeros((n, window, f), dtype=np.float32)
    for sym in pd.unique(syms):
        pos = np.where(syms == sym)[0]
        # order this symbol's rows by date (df may not be globally date-sorted)
        order = np.argsort(dates[pos], kind="stable")
        pos_sorted = pos[order]
        sub = mat[pos_sorted]  # (T_sym, f) in date order
        pad = np.repeat(sub[:1], window - 1, axis=0)  # edge-pad start, no look-ahead
        padded = np.vstack([pad, sub])  # (T+window-1, f)
        win = sliding_window_view(padded, window, axis=0)  # (T, f, window)
        out[pos_sorted] = win.transpose(0, 2, 1)  # (T, window, f)
    return out


class TorchGRUClassifier:
    """Small recurrent multiclass classifier (GRU or LSTM) over feature windows."""

    def __init__(self, params: dict | None = None, seed: int = 42, rnn: str = "gru"):
        self.params = dict(params or {})
        # LightGBM-only knobs may leak in via the SMAC branch; ignore them.
        for k in ("class_weight", "n_estimators", "num_leaves", "learning_rate",
                  "min_child_samples", "reg_lambda", "subsample", "colsample_bytree",
                  "deterministic", "force_col_wise", "verbosity", "n_jobs"):
            self.params.pop(k, None)
        self.seed = int(seed)
        self.rnn = rnn
        self.hidden = int(self.params.get("hidden", 32))
        self.layers = int(self.params.get("layers", 1))
        self.dropout = float(self.params.get("dropout", 0.10))
        self.epochs = int(self.params.get("epochs", 20))
        self.batch = int(self.params.get("batch", 512))
        self.lr = float(self.params.get("lr", 1e-3))
        self.classes_ = None
        self._mu = None
        self._sd = None
        self._net = None
        self._clf = self  # so the SMAC branch's getattr(model,"_clf").classes_ works

    def _standardize_fit(self, X3: np.ndarray):
        flat = X3.reshape(-1, X3.shape[-1])
        self._mu = flat.mean(axis=0).astype(np.float32)
        self._sd = flat.std(axis=0).astype(np.float32)
        self._sd[self._sd < 1e-6] = 1.0

    def _standardize(self, X3: np.ndarray) -> np.ndarray:
        return (X3 - self._mu) / self._sd

    def fit(self, X3: np.ndarray, y: np.ndarray) -> "TorchGRUClassifier":
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        y = np.asarray(y).astype(np.int64)
        self.classes_ = np.array(sorted(set(y.tolist())))
        cls_to_idx = {int(c): i for i, c in enumerate(self.classes_)}
        y_idx = np.array([cls_to_idx[int(v)] for v in y], dtype=np.int64)
        n_cls = len(self.classes_)

        self._standardize_fit(X3)
        Xs = self._standardize(X3)

        # balanced class weights (OUT/HOLD dominate; up-weight rare ENTER/EXIT pivots)
        counts = np.bincount(y_idx, minlength=n_cls).astype(np.float64)
        counts[counts == 0] = 1.0
        w = (len(y_idx) / (n_cls * counts)).astype(np.float32)

        device = torch.device("cpu")
        Xt = torch.from_numpy(Xs)
        yt = torch.from_numpy(y_idx)
        wt = torch.from_numpy(w).to(device)

        f = X3.shape[-1]
        rnn_cls = nn.LSTM if self.rnn == "lstm" else nn.GRU
        net = nn.Sequential()  # placeholder; build a custom module below
        net = _RNNNet(rnn_cls, f, self.hidden, self.layers, self.dropout, n_cls).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        lossf = nn.CrossEntropyLoss(weight=wt)

        g = torch.Generator()
        g.manual_seed(self.seed)
        n = len(yt)
        net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for b in range(0, n, self.batch):
                idx = perm[b : b + self.batch]
                xb = Xt[idx].to(device)
                yb = yt[idx].to(device)
                opt.zero_grad()
                out = net(xb)
                loss = lossf(out, yb)
                loss.backward()
                opt.step()
        net.eval()
        self._net = net
        return self

    def predict_proba(self, X3: np.ndarray) -> np.ndarray:
        import torch

        Xs = self._standardize(X3)
        Xt = torch.from_numpy(Xs)
        outs = []
        self._net.eval()
        with torch.no_grad():
            for b in range(0, len(Xt), 4096):
                logits = self._net(Xt[b : b + 4096])
                outs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float64)


def _make_rnn_net(rnn_cls, n_feat, hidden, layers, dropout, n_cls):
    import torch.nn as nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = rnn_cls(
                input_size=n_feat,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=(dropout if layers > 1 else 0.0),
            )
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_cls))

        def forward(self, x):
            out, _ = self.rnn(x)
            last = out[:, -1, :]  # final timestep representation
            return self.head(last)

    return _Net()


def _RNNNet(rnn_cls, n_feat, hidden, layers, dropout, n_cls):
    return _make_rnn_net(rnn_cls, n_feat, hidden, layers, dropout, n_cls)
