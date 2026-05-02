# Kiến trúc mới — Stock ML Trading System v2.0

## 1. Mục tiêu thiết kế

Kiến trúc này thay thế hoàn toàn pipeline hiện tại để giải quyết 3 vấn đề:

1. **`**kwargs` túi rác** → mỗi component có interface explicit, lỗi typo bị catch ngay
2. **60 strategy file rời rạc** → chuyển sang composable component, version mới = 1 file YAML
3. **Khó thử tổ hợp** → grid-search Entry × Exit × Feature × Fusion qua matrix YAML

Mục tiêu nghiên cứu: thử nhanh 100+ tổ hợp (entry model × exit model × feature set × fusion strategy) mà KHÔNG phải viết Python file mới cho từng tổ hợp.

## 2. Nguyên tắc kiến trúc (lock-in trước khi code)

### 2.1 Component-based, không Inheritance-based

Hiện tại: 60 function/class kế thừa "ngầm" qua copy-paste + flag dictionary. Phải đọc cả 60 file mới hiểu.

Mới: 6 loại component độc lập, mỗi loại có interface chuẩn (Protocol/ABC). Một experiment = compose các component lại.

### 2.2 Explicit > Implicit

- Không dùng `**kwargs` ở interface chính. Mọi tham số phải khai báo rõ.
- Dùng `dataclass` cho input/output thay vì dict.
- Component phải khai báo capability (vd: `SUPPORTS_EXIT_MODEL = True/False`).
- Validation chạy ở thời điểm load config, KHÔNG đợi đến runtime.

### 2.3 Pure Function khi có thể

- Feature engine: pure (DataFrame → DataFrame)
- Target generator: pure
- Fusion strategy: pure (state in → state out)
- Backtester: stateful nhưng tách bạch khỏi strategy logic

→ Test dễ, debug dễ, parallelize dễ.

### 2.4 Configuration as Code

Mọi experiment được định nghĩa hoàn toàn trong YAML. Không có Python file cho 1 experiment.

→ Thử nghiệm = sửa YAML → re-run. Không cần đụng vào codebase.

### 2.5 No Backwards Compatibility Burden

Schema mới được thiết kế cho tương lai, không ràng buộc bởi format cũ. 49 version cũ ở `legacy/` chạy qua adapter, KHÔNG kéo schema cũ vào kiến trúc mới.

## 3. Sơ đồ component

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Experiment Definition (YAML)                       │
│                                                                           │
│   name, components: {features, target, entry_model, exit_model},         │
│   fusion: {pre_filters, exit_overrides, hold_managers}, params           │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ load + validate
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Pipeline Orchestrator                                │
│   (resolves components from registry, manages caches, runs walk-forward)│
└──────────┬─────────┬─────────┬─────────┬──────────┬────────────────────┘
           │         │         │         │          │
           ▼         ▼         ▼         ▼          ▼
       ┌───────┐ ┌──────┐ ┌──────┐ ┌──────┐  ┌──────────┐
       │ Data  │ │Feat. │ │Target│ │Entry │  │ Exit     │
       │Loader │ │Engine│ │ Gen  │ │Model │  │ Model    │
       │       │ │      │ │      │ │      │  │ (option) │
       └───┬───┘ └──┬───┘ └──┬───┘ └──┬───┘  └────┬─────┘
           │        │        │        │           │
           └────────┴────────┴────────┴───────────┤
                                                   ▼
                          ┌────────────────────────────────────┐
                          │        Signal Pipeline              │
                          │  (entry_signals + exit_signals)     │
                          └─────────────────┬───────────────────┘
                                            ▼
                          ┌────────────────────────────────────┐
                          │      Fusion Strategy Stack          │
                          │  pre_filter → enter → hold_mgr →    │
                          │  exit_override → backtester action  │
                          └─────────────────┬───────────────────┘
                                            ▼
                          ┌────────────────────────────────────┐
                          │     Backtester (position mgr)       │
                          │  Action stream → Trade list         │
                          └─────────────────┬───────────────────┘
                                            ▼
                          ┌────────────────────────────────────┐
                          │   Evaluator (metrics + score)       │
                          └─────────────────────────────────────┘
```

## 4. Component Specifications

### 4.1 Data Loader (giữ gần như cũ)

```python
class DataLoader:
    """Load OHLCV from Hive-partitioned CSV."""
    def load_all(self, symbols: list[str]) -> pd.DataFrame: ...
```

Không thay đổi nhiều. Chỉ extract khỏi pipeline để testable hơn.

### 4.2 Feature Engine (composable blocks)

**Concept**: Một feature_set = composition của các "feature block" độc lập.

```python
# src/components/features/base.py
from typing import Protocol
import pandas as pd

class FeatureBlock(Protocol):
    """One unit of feature computation. Pure function on DataFrame."""
    name: str
    requires: list[str]  # required input columns (vd: ["close", "volume"])
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature columns. Return enriched df."""
    
    def get_feature_names(self) -> list[str]:
        """Names of columns this block adds."""

class ComposableFeatureEngine:
    """Composes multiple FeatureBlocks into a single feature set."""
    
    def __init__(self, blocks: list[FeatureBlock]):
        self.blocks = blocks
        self._validate_dependencies()
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        for block in self.blocks:
            df = block.compute(df)
        return df
    
    def get_feature_names(self) -> list[str]:
        return [name for block in self.blocks for name in block.get_feature_names()]
    
    def signature(self) -> str:
        """Stable hash for cache key."""
        return hash_blocks(self.blocks)
```

**Block library**:
```
src/components/features/blocks/
├── ohlcv_basic.py        # SMA, EMA, returns, ATR
├── momentum.py           # RSI, MACD, ROC, ret_5/10/20
├── volume.py             # OBV, volume_ratio, vol_z
├── volatility.py         # std_20, garman_klass, parkinson
├── heikin_ashi.py        # HA candles + derived features
├── regime.py             # trend_strength, choppy_index
├── pattern.py            # candle patterns (doji, hammer, ...)
├── orderflow.py          # placeholder for future
└── sentiment.py          # placeholder for future
```

**Mỗi block** chỉ làm 1 việc, dependency tối thiểu:
```python
# src/components/features/blocks/heikin_ashi.py
class HeikinAshiBlock(FeatureBlock):
    name = "heikin_ashi"
    requires = ["open", "high", "low", "close"]
    
    def compute(self, df):
        df = df.copy()
        df["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        # ...
        return df
    
    def get_feature_names(self):
        return ["ha_close", "ha_open", "ha_dist", "ha_streak"]
```

**Mapping ngược về feature_set cũ** (cho regression):
```yaml
# config/feature_sets/leading_v4.yaml
name: leading_v4
description: V34 features = leading_v3 + Heikin-Ashi
blocks:
  - ohlcv_basic
  - momentum
  - volume
  - regime
  - heikin_ashi
```

→ `leading`, `leading_v2`, `leading_v3`, `leading_v4` đều thành 4 file YAML đơn giản.

### 4.3 Target Generator

```python
# src/components/targets/base.py
class TargetGenerator(Protocol):
    name: str
    
    def generate_entry_labels(self, df: pd.DataFrame) -> pd.Series:
        """Return labels {-1, 0, 1} aligned with df index."""
    
    def generate_exit_labels(
        self, df: pd.DataFrame, forward_window: int, loss_threshold: float
    ) -> pd.Series | None:
        """Return labels {0, 1}. None if not supported."""
    
    @property
    def num_classes(self) -> int: ...
    
    @property
    def supports_exit_labels(self) -> bool: ...
```

**Implementations**:
```
src/components/targets/
├── base.py
├── trend_regime.py          # dual MA crossover (legacy)
├── early_wave.py            # accumulation + forward gain
├── early_wave_v2.py         # relaxed + rule trigger
├── early_wave_dual.py       # 3 classes for buy AND sell head
├── return_regression.py     # placeholder cho future
└── registry.py
```

### 4.4 Entry Model

```python
# src/components/models/base.py
class EntryModel(Protocol):
    name: str
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray): ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return canonical signals: -1 / 0 / 1."""
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Return probabilities, None if not supported."""
    
    @property
    def classes_(self) -> list[int]: ...
```

**Implementations**:
```
src/components/models/
├── base.py
├── lightgbm_classifier.py
├── xgboost_classifier.py
├── catboost_classifier.py
├── random_forest.py
├── gru_seq.py               # PyTorch sequential model
├── transformer_seq.py       # placeholder
├── ensemble.py              # voting/stacking ensemble
└── registry.py
```

**Hyperparameter contract**:
```python
@dataclass
class ModelHyperparams:
    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.05
    random_state: int = 42
    device: str = "auto"
    extras: dict = field(default_factory=dict)  # model-specific
```

### 4.5 Exit Model

```python
# src/components/exit_models/base.py
class ExitModel(Protocol):
    name: str
    
    def fit(self, X_train: np.ndarray, y_exit_train: np.ndarray): ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...  # {0, 1}
    def predict_proba(self, X: np.ndarray) -> np.ndarray | None: ...
```

**Implementations**:
```
src/components/exit_models/
├── base.py
├── ml_binary.py             # LightGBM/XGBoost binary classifier
├── rule_based.py            # If-then rules (no training)
├── composite.py             # Combine ML + rule
├── trailing_stop.py         # Pure trailing stop logic
└── registry.py
```

**Tách hẳn khỏi EntryModel** vì:
- Có thể là model khác (không bắt buộc cùng loại Entry)
- Có thể không phải ML (rule-based, trailing stop)
- Có thể optional (None)
- Có thể compose nhiều exit logic (ML + trailing + max-loss)

### 4.6 Fusion Strategy Stack

**Đây là phần KHÓ NHẤT**. Chứa toàn bộ "magic" của 60 version cũ.

**Concept**: Logic xử lý tín hiệu được chia thành các "lớp" theo lifecycle của vị thế:

```
┌─────────────────────────────────────────────────────┐
│  Pre-Entry Filter Layer                              │
│  (skip entry if condition matches)                   │
│  Examples: skip_choppy, sma200_filter, anti_fomo    │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Entry Decision Layer                                │
│  (combine signals + rule ensemble + sizing)          │
│  Examples: ml_only, ml_or_rule, ml_and_rule_2of3,   │
│            hybrid_entry (rule trigger when ml=0)     │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Hold Management Layer                               │
│  (modify hold duration / position size during hold)  │
│  Examples: trend_persistence_hold, time_decay,      │
│            cooldown_after_loss                       │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Exit Rules Layer                                    │
│  (force exit before natural signal/end)              │
│  Ordered by priority:                                │
│  1. emergency_exit (max_loss hit)                    │
│  2. hap_preempt (hardcap after profit)              │
│  3. early_loss_cut                                   │
│  4. signal_exit_defer (defer exit-model signal)      │
│  5. exit_model (exit-model prediction)               │
│  6. trailing_stop                                    │
└──────────────────────┬──────────────────────────────┘
                       ▼
              ┌──────────────────┐
              │  Final Action     │
              │  HOLD / EXIT      │
              └──────────────────┘
```

**Interface**:
```python
# src/components/fusion/base.py
@dataclass
class BarContext:
    """Everything a fusion strategy needs at one bar."""
    bar_idx: int
    df_test: pd.DataFrame             # full test data
    entry_signal: int                 # from entry_model (-1, 0, 1)
    entry_proba: np.ndarray | None    # from entry_model
    exit_signal: int | None           # from exit_model (0, 1)
    exit_proba: np.ndarray | None
    position: Position | None         # current position state
    config: dict                      # strategy-specific params
    symbol_profile: str               # bank/momentum/etc.

class FusionStrategy(Protocol):
    name: str
    layer: Literal["pre_entry", "entry", "hold", "exit_rules"]
    priority: int  # within layer, lower = first
    
    def apply(self, ctx: BarContext) -> FusionResult:
        """Decide action at this layer."""

@dataclass
class FusionResult:
    action: Literal["pass", "skip_entry", "enter", "exit", "modify_hold"]
    reason: str  # for trade journal
    metadata: dict = field(default_factory=dict)
```

**Strategy library** (port từ 40+ flag cũ):
```
src/components/fusion/
├── base.py
├── pre_entry/
│   ├── skip_choppy.py
│   ├── sma200_filter.py
│   ├── anti_fomo.py
│   ├── early_wave_filter.py
│   ├── recovery_peak_filter.py
│   └── ...
├── entry/
│   ├── ml_only.py
│   ├── rule_ensemble.py
│   ├── hybrid_entry.py
│   └── ...
├── hold/
│   ├── trend_persistence_hold.py
│   ├── time_decay.py
│   ├── cooldown_after_loss.py
│   └── ...
├── exit_rules/
│   ├── hap_preempt.py
│   ├── early_loss_cut.py
│   ├── short_hold_exit_filter.py
│   ├── signal_exit_defer.py
│   ├── exit_model.py           # Wraps exit_model signal into strategy
│   ├── trailing_stop.py
│   ├── weak_oversold_exit.py
│   └── ...
└── registry.py
```

**Composition**:
```yaml
strategy:
  entry_rules:
    - {name: skip_choppy, params: {threshold: 0.3}}
    - {name: early_wave_filter, params: {min_score: 1}}
    - {name: ml_or_rule_ensemble, params: {min_score: 1}}
  
  hold_rules:
    - {name: trend_persistence_hold, params: {max_extra_bars: 5}}
  
  exit_rules:
    - {name: emergency_exit, params: {max_loss: -0.20}, priority: 1}
    - {name: hap_preempt, params: {trigger: 0.04, floor: -0.07}, priority: 2}
    - {name: early_loss_cut, params: {threshold: -0.04, days: 5}, priority: 3}
    - {name: exit_model, params: {min_hold: 3}, priority: 5}
```

### 4.7 Backtester (pure position management)

```python
# src/components/backtest/engine.py
class Backtester:
    """Pure position management. NO strategy logic."""
    
    def run(
        self,
        actions: Iterator[Action],
        df_test: pd.DataFrame,
        initial_cash: float = 100.0,
        fee_pct: float = 0.001,
    ) -> list[Trade]:
        """Convert action stream to trades."""

@dataclass
class Action:
    bar_idx: int
    type: Literal["enter_long", "exit", "hold"]
    size: float = 1.0  # 0.0-1.0 of available cash
    reason: str = ""

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl_pct: float
    holding_days: int
    entry_reason: str
    exit_reason: str
    symbol: str
```

→ Backtester chỉ biết: nhận lệnh BUY/SELL → tính PnL. Không quyết định khi nào BUY/SELL — đó là việc của fusion stack.

### 4.8 Evaluator

```python
# src/components/evaluation/scorer.py
class Evaluator:
    def calc_metrics(self, trades: list[Trade]) -> dict: ...
    def composite_score(self, metrics: dict, trades: list[Trade]) -> float: ...
    def per_symbol_breakdown(self, trades: list[Trade]) -> pd.DataFrame: ...
    def per_year_breakdown(self, trades: list[Trade]) -> pd.DataFrame: ...
```

Giữ nguyên logic từ `src/evaluation/scoring.py`, chỉ refactor thành class.

## 5. Experiment Schema (mới, gọn)

### 5.1 Single experiment

```yaml
# config/experiments/v22.yaml
name: V22
description: V19.1 + SMA200 + hybrid entry + fast_exit + signal_hard_cap
active: true
order: 5
color: '#66BB6A'

feature_set: leading_v2
target: trend_regime

signals:
  entry_model:
    type: lightgbm
    params:
      n_estimators: 500
      max_depth: 8
  exit_model:
    type: null

strategy:
  entry_rules:
    - sma200_filter
    - hybrid_entry
    - ml_only
  exit_rules:
    - fast_exit_loss
    - signal_hard_cap
    - time_decay
  params:
    fast_exit_loss:
      threshold_hb: -0.07
      threshold_std: -0.05
    signal_hard_cap:
      floor_hb: 0.15
      floor_std: 0.12
      mult_hb: 3.0
    time_decay:
      bars: 20
      mult: 0.5

execution:
  backtester: simple_long
  capital: 100_000_000

# Symbol profile dispatch (override params per profile)
profile_overrides:
  high_beta:
    strategy.exit_rules.fast_exit_loss.threshold_hb: -0.05
```

So với cũ (~30 dòng dày params không rõ thuộc về đâu), schema này **rõ ai làm gì**.

### 5.2 Matrix experiment (grid search)

```yaml
# config/experiments/_matrix_q2.yaml
matrix_name: entry_exit_grid_q2_2026
description: Grid search Entry × Exit × Feature

base:
  components:
    target: early_wave
  strategy:
    entry_rules: [skip_choppy, early_wave_filter, ml_only]

axes:
  features:
    - leading_v4
    - leading_v5_sentiment       # mới (khi có)
  
  entry_model:
    - {type: lightgbm}
    - {type: xgboost}
    - {type: gru, hyperparams: {hidden: 64, window: 30}}
  
  exit_model:
    - null
    - {type: ml_binary, fw: 15, loss: 0.05}
    - {type: rule_based, strategy: hap_preempt}
  
  fusion_exit:
    - []  # no override
    - [hap_preempt, early_loss_cut]
    - [hap_preempt, early_loss_cut, exit_model]

# Generates 2 × 3 × 3 × 3 = 54 experiments
naming_pattern: "matrix_{features}_{entry_model.type}_{exit_model.type}_{fusion_exit_idx}"

output_dir: experiments/q2_2026_grid
```

→ 1 file YAML định nghĩa 54 experiments. Pipeline auto-expand thành 54 runs, share cache cho phù hợp.

## 6. Cấu trúc thư mục đích

```
stock_ml/
├── config/
│   ├── pipeline.yaml                    # global pipeline settings (data_dir, walk-forward)
│   ├── feature_sets/
│   │   ├── leading.yaml
│   │   ├── leading_v2.yaml
│   │   ├── leading_v3.yaml
│   │   └── leading_v4.yaml
│   ├── experiments/
│   │   ├── champions/                   # 11 versions ported
│   │   │   ├── v22.yaml
│   │   │   ├── v32.yaml
│   │   │   ├── v34.yaml
│   │   │   ├── v37a.yaml
│   │   │   └── ...
│   │   ├── matrix/
│   │   │   └── q2_2026_grid.yaml
│   │   └── _index.yaml
│   └── profiles/
│       └── symbol_profiles.yaml         # symbol → bank/momentum/...
│
├── src/
│   ├── components/
│   │   ├── features/
│   │   │   ├── base.py
│   │   │   ├── engine.py
│   │   │   ├── blocks/
│   │   │   │   ├── ohlcv_basic.py
│   │   │   │   ├── momentum.py
│   │   │   │   ├── heikin_ashi.py
│   │   │   │   └── ...
│   │   │   └── registry.py
│   │   ├── targets/
│   │   ├── models/
│   │   ├── exit_models/
│   │   ├── fusion/
│   │   │   ├── base.py
│   │   │   ├── pre_entry/
│   │   │   ├── entry/
│   │   │   ├── hold/
│   │   │   ├── exit_override/
│   │   │   └── registry.py
│   │   ├── backtest/
│   │   │   └── engine.py
│   │   └── evaluation/
│   │       └── scorer.py
│   │
│   ├── pipeline/
│   │   ├── orchestrator.py             # main runner
│   │   ├── trainer.py                   # train Model A + B
│   │   ├── walker.py                    # walk-forward split
│   │   ├── cache.py                     # shared prediction cache
│   │   ├── matrix_expander.py           # expand matrix YAML to single experiments
│   │   └── exporter.py                  # CSV → JSON for dashboard
│   │
│   ├── data/
│   │   ├── loader.py
│   │   └── splitter.py
│   │
│   └── env.py                           # local/Colab path resolution
│
├── legacy/                              # 49 retired versions
│   ├── README.md                        # explains what's here
│   ├── adapter.py                       # bridges old code to new pipeline
│   ├── strategies/                      # original v11-v40 functions
│   └── configs/                         # original models.yaml entries (frozen)
│
├── tests/
│   ├── regression/
│   │   ├── golden/                      # golden CSVs
│   │   └── test_champion_versions.py
│   ├── components/
│   │   ├── test_features.py
│   │   ├── test_fusion.py
│   │   └── ...
│   └── integration/
│       └── test_pipeline_e2e.py
│
├── scripts/
│   ├── run_experiment.py                # python -m stock_ml run <name>
│   ├── run_matrix.py
│   ├── migrate_old_yaml.py              # convert old → new schema
│   └── compare_results.py
│
├── results/
│   └── ... (giữ nguyên)
│
├── archive/                             # GIỮ LẠI configs cũ
│   └── configs_legacy/                  # original models.yaml entries (read-only ref)
│
├── docs/
│   └── refactor/
│       ├── ARCHITECTURE.md              # ← file này
│       ├── CHAMPION_VERSIONS.md
│       ├── REFACTOR_ROADMAP.md
│       ├── CLEANUP_PLAN.md
│       ├── HOW_TO_ADD_FEATURE_BLOCK.md
│       ├── HOW_TO_ADD_FUSION_STRATEGY.md
│       └── HOW_TO_PORT_LEGACY_VERSION.md
│
└── visualization/                       # giữ nguyên (hoặc cập nhật manifest format)
```

## 7. Data Flow chi tiết (1 experiment chạy như thế nào)

```
1. CLI: python -m stock_ml run champions/v22
        ↓
2. Load config:
   - parse v22.yaml
   - resolve component references (features=leading_v2, target=trend_regime, ...)
   - validate: kiểm tra fusion strategy có support exit_model match không
        ↓
3. Component instantiation:
   - features = ComposableFeatureEngine([OhlcvBasicBlock(), MomentumBlock(), ...])
   - target = TrendRegimeTarget()
   - entry_model = LightGBMEntryModel(hyperparams)
   - exit_model = None  (v22 không có)
   - fusion_stack = [SMA200Filter(), HybridEntry(), MlOnly(), FastExitLoss(...), ...]
        ↓
4. Data loading:
   - DataLoader.load_all(symbols)
   - Cache key = hash(data_dir + symbols + feature_blocks_signature)
   - Hit → load parquet; Miss → compute features → save parquet
        ↓
5. Walk-forward loop (6 folds):
   For each fold:
     a. Split train (4yr) / test (1yr)
     b. Generate target labels (entry + exit if applicable)
     c. Train entry_model on (X_train, y_train)
     d. Train exit_model on (X_train, y_exit_train) if not None
     e. For each symbol in test:
        - Predict y_pred (entry signals)
        - Predict y_pred_exit (if exit_model)
        - For each bar:
          * Build BarContext
          * Run fusion_stack.apply(ctx) → Action
        - Backtester.run(actions) → Trades
        ↓
6. Aggregate trades across folds + symbols
        ↓
7. Evaluator.calc_metrics + composite_score
        ↓
8. Save results/trades_v22.csv + meta.json
        ↓
9. Export JSON for dashboard
```

## 8. Validation rules (catch bugs at config load)

Khi load experiment YAML, runner check:

| Rule | Error |
|------|-------|
| Component name không tồn tại trong registry | `UnknownComponentError: features 'leading_v9' not registered` |
| `exit_model` bật nhưng không có `exit_model` trong `strategy.exit_rules` | `ConfigError: exit_model defined but no strategy exit rule uses it` |
| Exit rule yêu cầu `y_pred_exit` nhưng `signals.exit_model.type=null` | `ConfigError: exit_model rule requires signals.exit_model` |
| Target không support exit nhưng exit_model bật | `ConfigError: target 'trend_regime' does not support exit labels` |
| Hyperparam không hợp lệ | Validate via dataclass + Pydantic schema |
| Tên duplicate giữa các experiments | `DuplicateNameError` |

→ Tất cả check ở **load time**, không đợi train xong 5 phút mới phát hiện.

## 9. Caching strategy

Cache theo signature, không theo version name:

```
cache/
├── data/                              # raw OHLCV (rarely changes)
│   └── {data_dir_hash}.parquet
├── features/                          # feature engine output
│   └── {data_hash}_{feature_blocks_signature}.parquet
├── targets/                           # target labels
│   └── {feature_hash}_{target_signature}.parquet
└── predictions/                       # model predictions per fold
    └── {feature_hash}_{target_hash}_{model_signature}_{fold_idx}.npz
```

**Lợi ích**: 50 experiment dùng cùng feature_set + target → chỉ tốn 1 lần feature compute + 1 lần target gen. Chỉ model fit là khác.

→ Matrix với 54 experiments có thể chỉ tốn ~10x thời gian của 1 experiment đơn (vì share cache).

## 10. CLI design

```bash
# Single experiment
python -m stock_ml run champions/v22
python -m stock_ml run champions/v22 --device gpu --force

# Multiple experiments
python -m stock_ml run champions/v22 champions/v34 champions/v37a

# Matrix
python -m stock_ml run-matrix matrix/q2_2026_grid
python -m stock_ml run-matrix matrix/q2_2026_grid --resume

# Validate config without running
python -m stock_ml validate champions/v22

# List registered components
python -m stock_ml list-components
python -m stock_ml list-components --type fusion
python -m stock_ml list-experiments --active

# Export only
python -m stock_ml export --all
python -m stock_ml export champions/v22

# Compare runs
python -m stock_ml compare champions/v22 champions/v34 --metric score

# Migration helper
python -m stock_ml migrate-legacy v22  # convert old → new YAML
```

## 11. Testing strategy

### 11.1 Unit tests (per component)
- Each FeatureBlock has test: known input → known output
- Each FusionStrategy has test: BarContext → expected FusionResult
- Each EntryModel has test: smoke test (fits + predicts)

### 11.2 Integration tests (per layer)
- ComposableFeatureEngine: combine 3 blocks, check output columns
- Fusion stack: 5 strategies in order, check final action

### 11.3 Regression tests (champion versions)
- 11 champion experiments, each compared with golden CSV
- Allow tolerance: 0.0% (must match exactly)
- Run on PR / pre-commit (subset) + nightly (full)

### 11.4 Property-based tests
- Random fusion stacks → backtester always produces valid trades (entry < exit, etc.)
- Random feature blocks → no NaN/Inf in output

## 12. Migration path (high-level)

| Phase | Goal | Verification |
|-------|------|--------------|
| 0 | Golden baseline cho 11 champion | Hash CSVs |
| 1 | Build component framework | Unit tests pass |
| 2 | Port v22 (simple) | Regression match v22 golden |
| 3 | Port v34, v37a | Regression match goldens |
| 4 | Port remaining 8 champion | Regression match all goldens |
| 5 | Wrap legacy 49 versions in adapter | Smoke run, basic metrics match |
| 6 | Build matrix expander | Matrix YAML → list of experiments |
| 7 | Documentation + tooling | Doc reviewable, CLI works |

Chi tiết ở `REFACTOR_ROADMAP.md`.

## 13. Quyết định kỹ thuật cần lock-in

Trước khi viết dòng code đầu tiên, lock các quyết định sau:

### 13.1 Python version + tooling
- **Python**: 3.11+ (cho `Self` type, exception groups, perf)
- **Type checker**: `mypy --strict` cho `src/components/`, lỏng hơn cho `legacy/`
- **Formatter**: `ruff format` (thay black, faster)
- **Linter**: `ruff check` (thay flake8)
- **Test runner**: `pytest` + `pytest-xdist` cho parallel

### 13.2 Dependency injection
- KHÔNG dùng framework DI (như `dependency-injector`). Quá phức tạp.
- Dùng explicit constructor: `Pipeline(features=..., target=..., entry_model=...)`.
- Registry pattern cho component discovery: `register("lightgbm", LightGBMEntryModel)`.

### 13.3 Configuration loading
- **Pydantic v2** cho schema validation. Dataclass đơn giản dùng cho internal.
- YAML library: `pyyaml` (đã dùng) — không cần đổi.
- Reference syntax: `features: leading_v2` resolves to `config/feature_sets/leading_v2.yaml`.

### 13.4 Logging
- `logging` stdlib, không dùng `loguru` (overkill).
- Log format: `{timestamp} {level} [{component}] {message}`.
- Verbose levels: pipeline DEBUG, component INFO, errors ERROR.

### 13.5 Random seeds
- Master seed = 42, set globally tại pipeline start
- Mỗi component có `seed_offset` riêng (vd: lightgbm seed = 42, gru seed = 43)
- Document trong `pipeline.yaml`

### 13.6 Naming convention
- Module: `lower_snake_case.py`
- Class: `UpperCamelCase`
- Component name (string identifier): `lower_snake_case` (vd: `"leading_v4"`, `"hap_preempt"`)
- Experiment name: same as filename without `.yaml`

### 13.7 Error handling
- Validation errors: `ConfigError` (subclass of `ValueError`) — fail at load
- Runtime errors: `PipelineError` — bubble up, don't swallow
- KHÔNG dùng `try: ... except: pass`. Mọi exception phải được handle hoặc propagate.

### 13.8 File I/O
- Cache: `parquet` cho DataFrame, `npz` cho ndarray, `json` cho metadata
- Atomic writes: write to `.tmp` → rename (tránh corrupt nếu crash)
- Path: dùng `pathlib.Path`, không string concat

## 14. Backwards compatibility với legacy

### 14.1 Legacy adapter

Legacy versions vẫn chạy được qua adapter trong giai đoạn migration; sau khi validate Phase 6 xong sẽ xóa legacy path:

```python
# src/pipeline/legacy_adapter.py
class LegacyAdapter:
    """Wraps old backtest_vXX functions to fit new pipeline interface."""
    
    def __init__(self, version_key: str):
        self.version_key = version_key
        self.old_fn = self._import_old_function(version_key)
        self.old_config = self._load_old_yaml(version_key)
    
    def run(self, prediction_cache, df_test, ...):
        # Translate new pipeline state → old function args
        # Call old function
        # Translate result back
        ...
```

→ Trong migration window vẫn run được `python -m stock_ml run legacy/v25`.

### 14.2 Future port

Format `legacy/configs/v25.yaml` (frozen old schema) cho phép sau này dễ port nếu cần:

```bash
python -m stock_ml migrate-legacy v25  # tự động translate sang champion-style YAML
```

→ Nếu sau này v25 trở nên quan trọng, port qua kiến trúc mới mất ~30 phút.

## 15. Architecture decisions locked (2026-04-26 to 2026-04-28)

1. **Symbol profile dispatch** (v37a pattern): xử lý ở **outer layer**.
   - Quyết định: orchestrator merge `profile_overrides` vào fusion params trước khi chạy theo symbol/fold.
   - Lý do: giữ fusion strategy thuần theo `BarContext`, không trộn concern "profile routing" vào từng strategy.
   
2. **Per-symbol model** (future): **support by design, defer implementation**.
   - Quyết định: base contract cho phép wrapper `EntryModel` kiểu `dict[symbol, model]`, chưa build ở Phase 1.
   - Lý do: không chặn mở rộng sau này nhưng tránh tăng scope giai đoạn foundation.
   
3. **Ensemble entry model**: **native skeleton từ đầu**.
   - Quyết định: có `EnsembleEntryModel` skeleton trong `src/components/models/` với voting cơ bản.
   - Lý do: nhiều experiment cần combine model; skeleton sớm giúp schema/registry ổn định.
   
4. **Dashboard compatibility**: **giữ manifest format cũ trong transition**.
   - Quyết định: pipeline mới export JSON/manifest tương thích dashboard hiện tại ít nhất đến hết Phase 6.1.
   - Lý do: giảm risk gãy workflow quan sát kết quả trong lúc refactor.

5. **Real-time inference** (paper trading future): **có hỗ trợ kiến trúc**.
   - Quyết định: tách rõ training mode và inference mode; inference dùng model đã lưu + fusion stack với single-bar context.
   - Lý do: tái sử dụng fusion/backtest contracts, tránh làm lại logic khi chuyển sang online inference.

## 16. Tổng kết

Kiến trúc này biến hệ thống từ **"60 file Python rời rạc"** thành **"6 loại component, mỗi loại có thư viện implementations, compose qua YAML"**.

Lợi ích cụ thể cho nghiên cứu sắp tới:
- Thử entry model mới: thêm 1 file `src/components/models/<name>.py`, đăng ký, dùng trong YAML
- Thử feature mới: thêm 1 `FeatureBlock`, kết hợp vào feature_set qua YAML
- Thử exit logic mới: thêm 1 fusion strategy
- Grid search 100+ tổ hợp: 1 file matrix YAML

Đầu tư: 6-8 tuần. Trả lại: tốc độ research 10x trong vài năm tới.
