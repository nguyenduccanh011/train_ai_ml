# Extensibility Guide — Thêm Model, Feature, Target, Direction

**Cho nhà phát triển muốn mở rộng Stock ML.**

Hướng dẫn này cung cấp các quy trình rõ ràng, từng bước để thêm:
1. Model type mới (ML hoặc Rule)
2. Feature set mới
3. Target type mới
4. Direction support (long/short)

---

## 📋 Table of Contents

1. [Thêm Model Type Mới](#1-thêm-model-type-mới)
2. [Thêm Feature Set Mới](#2-thêm-feature-set-mới)
3. [Thêm Target Type Mới](#3-thêm-target-type-mới)
4. [Direction Support (Long/Short)](#4-direction-support)
5. [Testing & Validation](#5-testing--validation)
6. [Debugging Checklist](#6-debugging-checklist)

---

## 1. Thêm Model Type Mới

### 1.1 Hiểu Cấu Trúc Registry Hiện Tại

**File**: `stock_ml/src/models/registry.py`

```python
# Ba registries riêng biệt
_ENTRY_REGISTRY = {"lightgbm": LGBMEntryModel, "xgboost": XGBEntryModel, "rule": RuleModel}
_EXIT_REGISTRY = {"lightgbm": LGBMExitModel, "rule": RuleModel}
_REGRESSION_REGISTRY = {"lightgbm": LGBMRegressionModel, "xgboost": XGBRegressionModel}

# Factory functions
def build_entry_model(model_type: str, params: dict, seed: int = 42) -> Any:
    """Build entry model from registry."""
    return _ENTRY_REGISTRY[model_type](params, seed)

def build_exit_model(model_type: str, params: dict, seed: int = 42) -> Any:
    """Build exit model from registry."""
    return _EXIT_REGISTRY[model_type](params, seed)
```

### 1.2 Implement Model Class

**Scenario**: Thêm **XGBoost** làm entry model (chưa có).

**File**: `stock_ml/src/models/xgboost.py` (tạo mới)

```python
"""XGBoost entry/exit models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from xgboost import XGBClassifier, XGBRegressor


@dataclass
class XGBEntryModel:
    """XGBoost classifier for binary entry signal (buy = 1)."""

    params: dict
    seed: int = 42
    _clf: XGBClassifier | None = None

    def __post_init__(self):
        """Initialize XGBClassifier."""
        self._clf = XGBClassifier(
            n_estimators=self.params.get("n_estimators", 300),
            max_depth=self.params.get("max_depth", 5),
            learning_rate=self.params.get("learning_rate", 0.05),
            random_state=self.seed,
            verbosity=0,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBEntryModel":
        """Train on binary labels: 1 (buy), 0 (not buy)."""
        self._clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions {0, 1}."""
        return self._clf.predict(X).astype(np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates shape (n, 2)."""
        return self._clf.predict_proba(X)


@dataclass
class XGBRegressionModel:
    """XGBoost regressor for forward return prediction."""

    params: dict
    seed: int = 42
    _reg: XGBRegressor | None = None

    def __post_init__(self):
        """Initialize XGBRegressor."""
        self._reg = XGBRegressor(
            n_estimators=self.params.get("n_estimators", 300),
            max_depth=self.params.get("max_depth", 5),
            learning_rate=self.params.get("learning_rate", 0.05),
            random_state=self.seed,
            verbosity=0,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBRegressionModel":
        """Train on continuous return targets."""
        self._reg.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return continuous return predictions."""
        return self._reg.predict(X).astype(np.float32)
```

### 1.3 Đăng Ký Model Vào Registry

**File**: `stock_ml/src/models/registry.py`

```python
# Ở đầu file
from src.models.xgboost import XGBEntryModel, XGBRegressionModel

# Update registries
_ENTRY_REGISTRY = {
    "lightgbm": LGBMEntryModel,
    "xgboost": XGBEntryModel,  # ← NEW
    "rule": RuleModel,
}

_REGRESSION_REGISTRY = {
    "lightgbm": LGBMRegressionModel,
    "xgboost": XGBRegressionModel,  # ← NEW
}
```

### 1.4 Sử Dụng Trong YAML Config

```yaml
name: xgboost_test
strategy: trend_following
market: US_large_cap
direction: long

components:
  features: basic_v1
  
  target:
    type: forward_return_regression
    horizon: 5
  
  entry_model:
    type: xgboost  # ← NEW
    params:
      n_estimators: 500
      max_depth: 7
      learning_rate: 0.05

split:
  type: walk_forward_year
  gap_bars: 10
  test_bars: 252

engine:
  max_hold_bars: 20
  hard_stop_pct: -0.08
```

### 1.5 Test Model Được Đăng Ký

```bash
# Chạy backtest với model mới
python -m stock_ml.scripts.run_v2 \
    --symbols AAPL,MSFT \
    --config-path stock_ml/config/experiments/xgboost_test.yaml \
    --out results/xgb_test

# Verify
python -c "
import json
with open('results/xgb_test/summary_xgboost_test.json') as f:
    data = json.load(f)
    print(f\"Audit: {data['audit']['overall']}\")
    print(f\"Trades: {data['aggregate']['n_trades']}\")
"
```

### 1.6 Thêm Unit Test (Optional Nhưng Nên Làm)

**File**: `stock_ml/tests/test_xgboost_model.py`

```python
import numpy as np
import pytest

from src.models.registry import build_entry_model


def test_xgboost_entry_model():
    """Test XGBoost entry model can train and predict."""
    
    # Setup
    X_train = np.random.randn(100, 10).astype(np.float32)
    y_train = np.random.randint(0, 2, 100).astype(np.int8)
    X_test = np.random.randn(20, 10).astype(np.float32)
    
    # Train
    model = build_entry_model("xgboost", {"n_estimators": 50, "max_depth": 3})
    model.fit(X_train, y_train)
    
    # Predict
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)
    
    # Validate
    assert preds.shape == (20,)
    assert preds.dtype == np.int8
    assert np.all(np.isin(preds, [0, 1]))
    
    assert probas.shape == (20, 2)
    assert np.all(probas >= 0) and np.all(probas <= 1)
```

---

## 2. Thêm Feature Set Mới

> ⚠️ **SẼ THAY ĐỔI**: Cách dưới đây (Python registry `stock_ml/src/features/registry.py` + list cột cứng) đang được thay bằng **Expression DSL + Feature Store** (feature = công thức lưu DB, set = tham chiếu). Sau khi triển khai, thêm feature mới = tạo `feature_def` qua UI/API, không sửa code. Xem [Feature Store + DSL Design](FEATURE_STORE_DSL_DESIGN.md). Nội dung §2 này còn đúng cho tới khi cutover (Phase 5).

### 2.1 Hiểu Registry Feature Hiện Tại

**File**: `stock_ml/src/features/registry.py`

```python
_FEATURE_REGISTRIES = {
    "basic_v1": {"add_features": add_basic_features_v1, "requires": ["ohlcv"]},
    "leading_v2": {"add_features": add_leading_v2_features, "requires": ["ohlcv"]},
    "leading_v3": {"add_features": leading_v3_features, "requires": ["ohlcv"]},
}

def apply_features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Apply feature set to DataFrame."""
    entry = _FEATURE_REGISTRIES[feature_set]
    return entry["add_features"](df)

def get_feature_cols(feature_set: str) -> list[str]:
    """Get feature column names for feature set."""
    entry = _FEATURE_REGISTRIES[feature_set]
    return entry["columns"]
```

### 2.2 Implement Feature Function

**Scenario**: Thêm **momentum_v1** feature set.

**File**: `stock_ml/src/features/momentum_v1.py` (tạo mới)

```python
"""Momentum-based features (RSI, MACD, CCI)."""

from __future__ import annotations

import pandas as pd
import talib


def add_momentum_v1_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum indicators for each symbol group.
    
    Features:
    - rsi_14: Relative Strength Index (14-period)
    - macd_line: MACD line (12, 26, 9)
    - macd_signal: MACD signal line
    - macd_hist: MACD histogram
    - cci_20: Commodity Channel Index (20-period)
    
    Args:
        df: DataFrame with OHLCV + possibly existing features
    
    Returns:
        DataFrame with momentum features added
    """
    out = df.copy()
    
    def _add_momentum(group):
        """Add momentum features per symbol."""
        g = group.copy()
        close = g["close"].values
        
        # RSI
        g["rsi_14"] = talib.RSI(close, timeperiod=14)
        
        # MACD
        macd_line, signal_line, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        g["macd_line"] = macd_line
        g["macd_signal"] = signal_line
        g["macd_hist"] = hist
        
        # CCI
        high = g["high"].values
        low = g["low"].values
        g["cci_20"] = talib.CCI(high, low, close, timeperiod=20)
        
        return g
    
    out = out.groupby("symbol", group_keys=False).apply(_add_momentum)
    return out


# Feature columns exported
MOMENTUM_V1_COLUMNS = [
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "cci_20",
]
```

### 2.3 Đăng Ký Feature Set

**File**: `stock_ml/src/features/registry.py`

```python
from src.features.momentum_v1 import add_momentum_v1_features, MOMENTUM_V1_COLUMNS

_FEATURE_REGISTRIES = {
    "basic_v1": {...},
    "leading_v2": {...},
    "leading_v3": {...},
    "momentum_v1": {  # ← NEW
        "add_features": add_momentum_v1_features,
        "columns": MOMENTUM_V1_COLUMNS,
        "requires": ["ohlcv"],
    },
}
```

### 2.4 Sử Dụng Trong YAML

```yaml
components:
  features: momentum_v1  # ← NEW
  
  target:
    type: forward_return_regression
    horizon: 5
```

### 2.5 Test Feature Set

```bash
# Quick test
python -c "
import pandas as pd
from stock_ml.src.features.momentum_v1 import add_momentum_v1_features

# Sample OHLCV
df = pd.DataFrame({
    'symbol': ['AAPL'] * 100,
    'date': pd.date_range('2024-01-01', periods=100),
    'open': 150 + (pd.Series(range(100)) * 0.1),
    'high': 151 + (pd.Series(range(100)) * 0.1),
    'low': 149 + (pd.Series(range(100)) * 0.1),
    'close': 150.5 + (pd.Series(range(100)) * 0.1),
    'volume': 1000000,
})

# Apply features
result = add_momentum_v1_features(df)

# Check
print(f'Columns: {result.columns.tolist()}')
print(f'NaN count: {result.isna().sum()}')
print(f'Shape: {result.shape}')
"
```

---

## 3. Thêm Target Type Mới

### 3.1 Hiểu Registry Target Hiện Tại

**File**: `stock_ml/src/targets/registry.py`

```python
_REGISTRY = {
    "forward_return": ForwardReturnTarget,
    "forward_return_regression": ForwardReturnRegressionTarget,
    "trend_regime": TrendRegimeTarget,
}

def build_target(config: dict) -> TargetProtocol:
    """Build target by type and config."""
    config = dict(config)
    target_type = config.pop("type")
    target_class = _REGISTRY[target_type]
    return target_class(**config)
```

### 3.2 Implement Target Class

**Scenario**: Thêm **volatility_regime** target (vào/ra khi volatility cao).

**File**: `stock_ml/src/targets/volatility_regime.py` (tạo mới)

```python
"""Volatility regime target — predict when vol will be high."""

from __future__ import annotations

import pandas as pd


class VolatilityRegimeTarget:
    """
    Volatility regime target — labels high (1) vs low (0) volatility periods.
    
    Uses rolling std(returns) to label volatility regime.
    """
    
    def __init__(self, horizon: int = 5, vol_threshold: float = 0.02):
        """
        Args:
            horizon: Forward window to check future volatility
            vol_threshold: Return std threshold for "high" label
        """
        self.horizon = horizon
        self.vol_threshold = vol_threshold
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 'target' column with volatility regime labels.
        
        1: High vol in next `horizon` bars
        0: Low vol in next `horizon` bars
        NaN: Not enough future data
        """
        out = df.copy()
        
        def _volatility_regime(group):
            g = group.copy()
            close = g["close"].values
            returns = pd.Series(close).pct_change().values
            
            # Forward rolling vol
            fwd_vol = pd.Series(returns).rolling(window=self.horizon).std()
            fwd_vol = fwd_vol.shift(-self.horizon)  # Shift forward
            
            target = (fwd_vol > self.vol_threshold).astype("int8")
            g["target"] = target
            return g
        
        out = out.groupby("symbol", group_keys=False).apply(_volatility_regime)
        return out
```

### 3.3 Đăng Ký Target Type

**File**: `stock_ml/src/targets/registry.py`

```python
from src.targets.volatility_regime import VolatilityRegimeTarget

_REGISTRY = {
    "forward_return": ForwardReturnTarget,
    "forward_return_regression": ForwardReturnRegressionTarget,
    "trend_regime": TrendRegimeTarget,
    "volatility_regime": VolatilityRegimeTarget,  # ← NEW
}
```

### 3.4 Sử Dụng Trong YAML

```yaml
components:
  target:
    type: volatility_regime  # ← NEW
    horizon: 10
    vol_threshold: 0.025
```

---

## 4. Direction Support

### 4.1 Config Direction Trong YAML

```yaml
name: short_mean_reversion
direction: short  # ← NEW (default: long)

components:
  entry_model:
    type: rule
    params:
      conditions:
        - {feature: rsi, op: ">", value: 70}  # Overbought
```

### 4.2 Cách Direction Hoạt Động

```python
# Signal generation
signals = compute_signals(...)  # Long logic: 1=buy, -1=sell

# Apply direction flip
if direction == "short":
    signals = signals * -1  # Now: -1=short entry, 1=cover exit

# Backtest (unchanged)
# signal=-1 → entry (short), signal=1 → exit (cover)
```

### 4.3 Verify Direction Hỗ Trợ

```bash
# Test short config
python -m stock_ml.scripts.run_v2 \
    --symbols AAPL \
    --config-path stock_ml/config/experiments/short_meanreversion.yaml \
    --out results/short_test

# Check leaderboard
python -c "
import json
with open('results/leaderboard.json') as f:
    data = json.load(f)
    for model in data['models']:
        print(f\"{model['name']}: direction={model['direction']}\")
"
```

---

## 5. Testing & Validation

### 5.1 Unit Test Template

```python
import numpy as np
import pytest

def test_my_new_model():
    """Test new model trains and predicts."""
    from src.models.registry import build_entry_model
    
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randint(0, 2, 100).astype(np.int8)
    
    model = build_entry_model("my_model_type", {})
    model.fit(X, y)
    preds = model.predict(X[50:60])
    
    assert preds.shape == (10,)
    assert np.all(np.isin(preds, [0, 1]))

def test_my_new_features():
    """Test new feature set generates columns."""
    from src.features.registry import apply_features
    import pandas as pd
    
    df = pd.DataFrame({
        "symbol": ["AAPL"] * 50,
        "close": np.random.randn(50).cumsum(),
        "high": np.random.randn(50).cumsum() + 1,
        "low": np.random.randn(50).cumsum() - 1,
        "volume": np.ones(50) * 1000000,
    })
    
    result = apply_features(df, "my_feature_set")
    assert result.shape[0] == 50
    assert not result.isna().all().any()  # Not all NaN

def test_my_new_target():
    """Test new target generates labels."""
    from src.targets.registry import build_target
    import pandas as pd
    
    df = pd.DataFrame({
        "symbol": ["AAPL"] * 50,
        "close": np.random.randn(50).cumsum() + 100,
    })
    
    target = build_target({"type": "my_target_type", "horizon": 5})
    result = target.apply(df)
    assert "target" in result.columns
```

### 5.2 Integration Test (Full Backtest)

```bash
# Create test config
cat > /tmp/test_new_stuff.yaml << 'EOF'
name: test_new_everything
strategy: test
market: test

components:
  features: my_feature_set  # ← NEW
  target:
    type: my_target_type    # ← NEW
    horizon: 5
  entry_model:
    type: my_model_type     # ← NEW
    params: {}
  signal_mode: entry_first

split:
  type: walk_forward_year
  gap_bars: 10
  test_bars: 252

engine:
  max_hold_bars: 20
  hard_stop_pct: -0.08
EOF

# Run backtest
python -m stock_ml.scripts.run_v2 \
    --symbols AAPL,MSFT \
    --config-path /tmp/test_new_stuff.yaml \
    --out /tmp/test_results

# Verify audit passed
python -c "
import json
with open('/tmp/test_results/summary_test_new_everything.json') as f:
    data = json.load(f)
    audit = data['audit']['overall']
    if audit == 'PASS':
        print('✓ Integration test PASSED')
    else:
        print(f'✗ Integration test FAILED: {audit}')
        print(data['audit'])
"
```

---

## 6. Debugging Checklist

| Issue | Solution |
|-------|----------|
| `KeyError: 'my_model_type' not in registry` | ✓ Đăng ký model vào `_ENTRY_REGISTRY` hoặc `_REGRESSION_REGISTRY` |
| `KeyError: 'my_feature_set'` | ✓ Đăng ký feature vào `_FEATURE_REGISTRIES` |
| `KeyError: 'my_target_type'` | ✓ Đăng ký target vào `_REGISTRY` |
| `ValueError: Invalid direction: xyz` | ✓ direction phải là "long" hoặc "short" |
| `NaN in features` | ✓ Features được generate với NaN đầu (rolling windows) — normal, config dùng dropna |
| Model train nhưng không predict | ✓ Check `fit()` trả về `self` (chainable) |
| Audit FAIL | ✓ Check `audit.checks` JSON, look for detail + examples |
| All signals = 0 (no trades) | ✓ Check signal threshold, feature values, conditions |

---

## 📚 Related Documentation

- [WORKFLOW.md](WORKFLOW.md) — Complete workflow guide
- [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) — Architecture & phases
- [Phase 0.3 Implementation](../PHASE_0.3_IMPLEMENTATION.md) — Direction support details
- [Direction Explanation](../examples/DIRECTION_EXPLANATION.md) — User guide for direction

---

**Happy Extending! 🚀**
