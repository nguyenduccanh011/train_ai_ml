# Direction Support (Phase 0.3)

## Overview

Direction support allows single strategy definition to work in both **long** and **short** modes by automatically flipping signals.

## Two Directions

### Long (default)
- Entry condition met → signal = 1 (buy)
- Exit condition met → signal = -1 (sell)
- Backtest: Buy to enter position, sell to exit

### Short
- Entry condition met → signal = 1 → **flipped to -1** (short)
- Exit condition met → signal = -1 → **flipped to 1** (cover)
- Backtest: Short to enter position, cover to exit

## How It Works

### Config Example

```yaml
# LONG Strategy
direction: long
entry_model:
  type: rule
  params:
    conditions:
      - {feature: macd_line, op: ">", value: 0}

# SHORT Strategy (same conditions, opposite trades)
direction: short
entry_model:
  type: rule
  params:
    conditions:
      - {feature: macd_line, op: ">", value: 0}  # Same condition!
```

### Signal Generation Flow

#### Long Direction
```
Bar T: macd_line = 0.05
  └─ Condition met (macd > 0) → raw_signal = 1
  └─ direction = long → signal = 1 (no flip)
  └─ Backtest: BUY at open[T+1]

Bar T+3: macd_line = -0.01
  └─ Condition not met → raw_signal = 0
  └─ direction = long → signal = 0 (hold)
```

#### Short Direction
```
Bar T: macd_line = 0.05
  └─ Condition met (macd > 0) → raw_signal = 1
  └─ direction = short → signal = 1 * -1 = -1 (flipped!)
  └─ Backtest: SHORT at open[T+1]

Bar T+3: macd_line = -0.01
  └─ Exit condition met → raw_signal = -1
  └─ direction = short → signal = -1 * -1 = 1 (flipped!)
  └─ Backtest: COVER at open[T+1]
```

## Code Changes

### 1. ExperimentConfig
```python
@dataclass
class ExperimentConfig:
    direction: str = "long"  # "long" | "short"
    ...
```

### 2. Signal Generation
```python
def generate_signals_from_predictions(..., direction: str = "long"):
    # Compute signals normally
    signals = compute_signals(...)
    
    # Apply direction flip
    if direction == "short":
        signals = signals * -1
    
    return signals
```

### 3. YAML Config
```yaml
experiment:
  direction: long  # or "short"
  entry_model: {...}
  exit_model: {...}
```

## Use Cases

### Long Trend Following
```yaml
direction: long
entry_model:
  type: rule
  params:
    conditions:
      - {feature: sma_20, op: ">", feature: sma_50}
      - {feature: rsi, op: "<", value: 70}
    logic: AND
```
→ Buy when trend up + not overbought

### Short Mean Reversion
```yaml
direction: short
entry_model:
  type: rule
  params:
    conditions:
      - {feature: rsi, op: ">", value: 70}  # Overbought
      - {feature: bb_position, op: ">", value: 0.8}
    logic: AND
```
→ Short when overbought (signals flipped)
→ Cover when oversold (signals flipped)

## Testing

### Test Long Config
```bash
python -m src.pipeline.experiment examples/experiment_long_trend.yaml
```

### Test Short Config
```bash
python -m src.pipeline.experiment examples/experiment_short_meanreversion.yaml
```

### Verify Direction in Leaderboard
```python
import json
with open("results/leaderboard.json") as f:
    data = json.load(f)
    for model in data["models"]:
        print(f"{model['name']}: direction={model['direction']}")
```

## Migration Notes

- Existing configs without `direction` field default to `long` (backward compatible)
- `direction` is stored in LeaderboardRow + DB model
- YAML validation enforces direction ∈ {"long", "short"}

## Advanced: Multi-Direction Experiments

Run same strategy in both directions:

```bash
# Generate both configs
python scripts/generate_variants.py \
  --base experiment_base.yaml \
  --dimensions direction:long,direction:short

# Run experiment suite
python -m src.pipeline.experiment experiment_base_dir_long.yaml
python -m src.pipeline.experiment experiment_base_dir_short.yaml

# Compare leaderboard results
python scripts/compare_leaderboard.py --filter-by-strategy trend_following
```

## Performance Implications

- Zero overhead: direction flip is a simple `signals * -1` operation
- No retraining needed: same models used, signals inverted
- Enables portfolio of long + short strategies

---

**Phase 0.3 complete**: Direction support fully integrated into experiment pipeline, signal generation, leaderboard, and DB models.
