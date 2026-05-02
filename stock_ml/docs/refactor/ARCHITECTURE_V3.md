# Architecture V3 — Signal / Strategy / Execution

## Mục tiêu

V3 tách trading system thành 3 tầng rõ ràng để version mới chủ yếu là YAML, không phải thêm runner Python mới.

```
YAML Experiment
  → Signal Layer
  → Strategy Layer
  → Execution Layer
  → Trades + Metrics
```

## 1. Signal Layer

Signal layer chỉ dự đoán, không quyết định trade.

Nguồn cấu hình:

```yaml
signals:
  entry_model:
    type: lightgbm
    params: {}
  exit_model:
    type: lightgbm
    params:
      forward_window: 15
      loss_threshold: 0.05
```

Trong schema hiện tại, YAML cũ vẫn dùng `components.entry_model` và `components.exit_model`; loader normalize về config nội bộ.

Outputs:

- `y_pred`: entry signal (`-1`, `0`, `1`)
- `y_pred_exit`: exit signal (`0`, `1`) nếu bật exit model
- metadata/cache: feature set, target config, split, model type

Code chính:

- `src/pipeline/build_predictions.py`
- `src/pipeline/trainer.py`
- `src/components/models/`
- `src/components/exit_models/`

## 2. Strategy Layer

Strategy layer quyết định action dựa trên signal + market context + position state.

```yaml
strategy_v3:
  entry_rules:
    - v19_entry_cascade
    - rule_signal_entry
  hold_rules:
    - min_hold_protection
    - long_horizon_carry
  exit_rules:
    - hard_stop_exit
    - exit_model_exit
    - hap_preempt
```

Rule categories:

| Category | Khi chạy | Output |
|---|---|---|
| `entry_rules` | Flat + có entry signal | enter / skip |
| `hold_rules` | Đang giữ position | keep / force exit |
| `exit_rules` | Đang giữ position | exit / pass |

Code chính:

- `src/components/fusion/registry.py`
- `src/components/fusion/strategies/`
- `src/components/runners/generic_fusion.py`
- `src/components/runners/lineage_backtests.py`

## 3. Execution Layer

Execution layer thực thi action và tạo trade records.

Nhiệm vụ:

- Quản lý position state
- Tính commission/tax/capital
- Ghi entry/exit date, PnL, holding days
- Chuẩn hóa output DataFrame/golden CSV

Code chính:

- `src/backtest/engine.py`
- `src/components/runners/rule_runner.py`
- `src/components/runners/v34_runner.py`
- `src/components/runners/runner_registry.py`

## Data flow

```
ExperimentConfig.from_yaml()
        ↓
validate_config()
        ↓
Pipeline.run()
        ↓
build prediction cache / reuse cache
        ↓
runner registry resolves champion runner
        ↓
strategy rules evaluate bar-by-bar
        ↓
backtester emits trades
        ↓
metrics + artifacts
```

## Extension points

| Muốn thêm | Chỉnh ở đâu |
|---|---|
| Feature set mới | `src/features/` + YAML `components.features` |
| Entry model mới | `src/components/models/registry.py` |
| Exit model mới | `src/components/exit_models/registry.py` |
| Entry/hold/exit rule mới | `src/components/fusion/strategies/` + registry |
| Champion config mới | `config/experiments/champions/*.yaml` |
| Matrix experiment mới | `config/experiments/matrix/*.yaml` |

## Tests layout

```
tests/
├── signals/      # features, target, signal/rule helpers
├── strategy/     # fusion stack, entry/hold/exit rules, model smoke tests
├── execution/    # backtest engine and execution behavior
└── regression/   # champion parity + golden checksums
```

## Legacy status

Phase 6 removed legacy migration/runtime paths:

- `src/pipeline/legacy_adapter.py`
- `scripts/migrate_legacy.py`
- `config/models.yaml`
- `legacy/`
- `experiments/`

Active champion runners now resolve through V3 config + runner registries. Special-case runners remain only where useful (`rule_runner.py`, v34 lineage helpers).
