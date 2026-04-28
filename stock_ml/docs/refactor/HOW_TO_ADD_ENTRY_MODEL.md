# HOW TO: Add an Entry Model

## Khi nào dùng

Thêm entry model khi muốn thử model architecture mới (Transformer, TabNet, XGBoost v2, ...) làm Model A (entry classifier).

## Checklist

1. Implement model wrapper (kế thừa Protocol `EntryModel`)
2. Register vào `src/components/models/registry.py`
3. Smoke test: fit + predict
4. Dùng trong YAML experiment

---

## Bước 1 — Implement wrapper

```python
# src/components/models/transformer.py
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import RobustScaler


class TransformerEntryModel:
    """Transformer-based entry classifier. Wraps a PyTorch transformer for tabular data."""

    name: str = "transformer"

    def __init__(
        self,
        *,
        n_layers: int = 2,
        d_model: int = 64,
        random_state: int = 42,
    ) -> None:
        self.n_layers = n_layers
        self.d_model = d_model
        self.random_state = random_state
        self._model = None
        self._scaler = RobustScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch
        torch.manual_seed(self.random_state)
        X_scaled = self._scaler.fit_transform(X)
        # ... build and train transformer ...
        self._model = ...  # trained model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels {-1, 0, 1}."""
        X_scaled = self._scaler.transform(X)
        # ... inference ...
        return predictions  # np.ndarray of int

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability array, shape (n_samples, n_classes)."""
        X_scaled = self._scaler.transform(X)
        # ... inference ...
        return probas  # shape (n, 3) for 3 classes
```

**Protocol `EntryModel` yêu cầu**:
- `fit(X, y) -> None`
- `predict(X) -> np.ndarray`  
- `predict_proba(X) -> np.ndarray`
- `name: str` attribute

## Bước 2 — Register

```python
# src/components/models/registry.py
from src.components.models.transformer import TransformerEntryModel

_MODEL_REGISTRY: dict[str, type] = {
    "lightgbm": LightGBMClassifier,
    "xgboost": XGBoostClassifier,
    "catboost": CatBoostClassifier,
    "random_forest": RandomForestModel,
    "gru": GRUSequenceModel,
    "transformer": TransformerEntryModel,   # <-- thêm mới
}


def get_model(name: str, **kwargs) -> EntryModel:
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name!r}. Available: {sorted(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](**kwargs)
```

## Bước 3 — Smoke test

```python
# tests/components/test_transformer_smoke.py
import numpy as np
import pytest
from src.components.models.transformer import TransformerEntryModel


@pytest.fixture
def tiny_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 20))
    y = rng.choice([-1, 0, 1], size=200)
    return X, y


def test_transformer_fit_predict(tiny_data):
    X, y = tiny_data
    model = TransformerEntryModel(n_layers=1, d_model=16, random_state=42)
    model.fit(X[:150], y[:150])
    preds = model.predict(X[150:])
    assert preds.shape == (50,)
    assert set(preds).issubset({-1, 0, 1})


def test_transformer_predict_proba(tiny_data):
    X, y = tiny_data
    model = TransformerEntryModel(n_layers=1, d_model=16, random_state=42)
    model.fit(X[:150], y[:150])
    proba = model.predict_proba(X[150:])
    assert proba.shape == (50, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
```

## Bước 4 — Dùng trong YAML

```yaml
# config/experiments/champions/v50_transformer.yaml
name: v50_transformer
strategy: v22
feature_set: leading_v4
target:
  type: trend_regime
model:
  type: transformer       # <-- model mới
  params:
    n_layers: 3
    d_model: 128
split:
  train_years: 3
  test_years: 1
```

Chạy:

```bash
python -m stock_ml validate champions/v50_transformer
python -m stock_ml run champions/v50_transformer --device cpu
```

## Verify

```bash
python -m ruff check src/components/models/transformer.py
python -m mypy src/components/models/
python -m pytest tests/components/test_transformer_smoke.py -v
```

## Lưu ý với GPU models

- Luôn set `torch.manual_seed(random_state)` + `torch.use_deterministic_algorithms(True, warn_only=True)`
- Regression tests phải chạy `device=cpu` để deterministic (xem Phase 0.1 diary)
- GRU wrapper ở `src/components/models/gru_seq.py` là mẫu tham khảo cho PyTorch models

## Khi nào KHÔNG thêm model mới

- Muốn thay hyperparams → dùng `params:` trong YAML, không cần class mới
- Muốn ensemble → dùng `EnsembleEntryModel` đã có skeleton ở `src/components/models/ensemble.py`
