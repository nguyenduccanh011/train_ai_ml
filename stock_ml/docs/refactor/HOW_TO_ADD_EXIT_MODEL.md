# HOW TO: Add an Exit Model

## Khi nào dùng

Thêm exit model khi muốn model hóa tín hiệu thoát độc lập với entry model, ví dụ dự đoán drawdown hoặc xác suất đảo chiều trong vài ngày tới.

## Checklist

1. Implement wrapper `ExitModel`
2. Register vào `src/components/exit_models/registry.py`
3. Khai báo trong YAML `components.exit_model` / `signals.exit_model`
4. Wire exit rule nếu strategy cần đọc tín hiệu model
5. Smoke test + regression test

---

## Bước 1 — Implement wrapper

```python
# src/components/exit_models/my_exit.py
from __future__ import annotations

import numpy as np


class MyExitModel:
    name = "my_exit"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ...  # binary labels: 1 = exit, 0 = hold

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return ...
```

Protocol chính nằm ở `src/components/exit_models/base.py`.

## Bước 2 — Register

```python
# src/components/exit_models/registry.py
from src.components.exit_models.my_exit import MyExitModel


def _build_my_exit(**kwargs):
    return MyExitModel(**kwargs)


_BUILDERS = {
    "null": _build_null,
    "lightgbm": _build_lightgbm,
    "xgboost": _build_xgboost,
    "catboost": _build_catboost,
    "my_exit": _build_my_exit,
}
```

`python -m stock_ml list-components --type exit_models` phải thấy model mới.

## Bước 3 — Khai báo YAML

Schema hiện tại vẫn support `components.exit_model`; schema V3 đọc qua `signals.exit_model`.

```yaml
name: v50_exit_model
strategy: v22_with_exit_model

components:
  features: leading_v4
  target:
    type: early_wave_dual
    forward_window: 15
    gain_threshold: 0.06
    loss_threshold: 0.03
  entry_model:
    type: lightgbm
  exit_model:
    enabled: true
    type: my_exit
    forward_window: 15
    loss_threshold: 0.05
    extras:
      random_state: 42

strategy_v3:
  exit_rules:
    - hard_stop_exit
    - exit_model
```

## Bước 4 — Wire strategy rule

Nếu model chỉ tạo `y_pred_exit`, runner cần một exit rule đọc tín hiệu đó. Rule chuẩn hiện tại là `exit_model` trong `src/components/fusion/strategies/core/exit_model_exit.py` (alias cũ `exit_model_exit` vẫn được hỗ trợ để tương thích ngược).

Thêm rule vào `strategy_v3.exit_rules`, hoặc vào section legacy `fusion.force_exit` / `fusion.active_exit` tùy runner đang dùng.

## Bước 5 — Test

```python
# tests/strategy/test_my_exit_model.py
import numpy as np
from src.components.exit_models.registry import get_exit_model


def test_my_exit_model_fit_predict_smoke():
    X = np.random.default_rng(42).normal(size=(100, 8))
    y = np.random.default_rng(43).integers(0, 2, size=100)
    model = get_exit_model("my_exit")
    model.fit(X[:80], y[:80])
    pred = model.predict(X[80:])
    assert pred.shape == (20,)
    assert set(pred).issubset({0, 1})
```

## Verify

```bash
python -m stock_ml validate champions/v50_exit_model
python -m pytest stock_ml/tests/strategy/test_my_exit_model.py -q
python -m pytest stock_ml/tests/regression/test_champions.py -q
```

## Khi nào KHÔNG thêm exit model mới

- Chỉ đổi `forward_window` hoặc `loss_threshold` → sửa YAML.
- Chỉ đổi điều kiện thoát thủ công → thêm fusion exit strategy, không thêm model.
- Muốn tắt model thoát → dùng `type: null` hoặc `enabled: false`.
