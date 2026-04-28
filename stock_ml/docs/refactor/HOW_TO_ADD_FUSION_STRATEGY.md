# HOW TO: Add a Fusion Strategy

## Khi nào dùng

Thêm fusion strategy khi cần logic entry filter, exit trigger, hoặc position hold guard mới.

## 4 Layers

| Layer | Runs khi | Tác dụng |
|-------|----------|----------|
| `pre_entry` | Flat, có entry signal | Filter ngăn entry (`skip_entry`) |
| `entry` | Flat | Trigger entry (`enter`) |
| `hold` | In position | Block exit (`keep_position`) hoặc force exit ngay |
| `exit_override` | In position, sau hold | Trigger exit (`exit`) |

## Checklist

1. Implement strategy class
2. Register với `register_strategy()`
3. Thêm vào YAML champion spec
4. Unit test
5. Regression test vẫn pass

---

## Bước 1 — Implement

```python
# src/components/fusion/strategies/pre_entry/vix_filter.py
from __future__ import annotations

from dataclasses import dataclass

from src.components.base import BarContext, FusionResult


@dataclass
class VixFilter:
    """Skip entry when VIX > threshold (high fear)."""

    name: str = "vix_filter"
    layer: str = "pre_entry"
    priority: int = 5
    threshold: float = 30.0

    def apply(self, ctx: BarContext) -> FusionResult:
        vix = ctx.config.get("vix_today", 0.0)
        if vix > self.threshold:
            return FusionResult(
                action="skip_entry",
                reason="vix_too_high",
                metadata={"counter": "n_vix_skip", "vix": vix},
            )
        return FusionResult(action="pass", reason="vix_ok")
```

**Rules**:
- `name`, `layer`, `priority` là required attributes (Protocol `FusionStrategy`)
- Luôn return `FusionResult` — không raise exception
- `metadata["counter"]` → được đếm vào `StackOutcome.counters`
- Không mutate `ctx` hay `ctx.position`

## Bước 2 — Register

```python
# src/components/fusion/strategies/__init__.py — thêm vào cuối
from src.components.fusion.strategies.pre_entry.vix_filter import VixFilter
from src.components.fusion.registry import register_strategy

register_strategy(
    "vix_filter",
    "pre_entry",
    lambda threshold=30.0, **_: VixFilter(threshold=threshold),
)
```

Hoặc dùng `always_on=True` nếu strategy này nên auto-prepend cho tất cả champions:

```python
register_strategy(
    "vix_filter",
    "pre_entry",
    lambda **kw: VixFilter(**kw),
    always_on=True,   # orchestrator tự thêm vào mọi stack
)
```

## Bước 3 — Thêm vào YAML

```yaml
# config/experiments/champions/v22.yaml
fusion:
  entry:
    - name: v19_entry_cascade
  force_exit:
    - name: hard_stop_exit
  active_exit:
    - name: atr_stop_loss
    - name: peak_protect_dist
  pre_entry:
    - name: vix_filter        # <-- thêm mới
      params:
        threshold: 25.0
  hold:
    - name: long_horizon_carry
```

## Bước 4 — Unit test

```python
# tests/components/fusion/test_vix_filter.py
import pandas as pd
import pytest
from src.components.base import BarContext, Position
from src.components.fusion.strategies.pre_entry.vix_filter import VixFilter


def _ctx(vix: float, in_position: bool = False) -> BarContext:
    df = pd.DataFrame({"close": [100.0] * 5})
    return BarContext(
        bar_idx=1,
        df_test=df,
        entry_signal=1,
        entry_proba=None,
        exit_signal=None,
        exit_proba=None,
        position=None if not in_position else Position(
            symbol="X", entry_idx=0,
            entry_date=pd.Timestamp("2024-01-01"), entry_price=100.0,
        ),
        config={"vix_today": vix},
    )


def test_vix_filter_blocks_high_vix():
    f = VixFilter(threshold=30.0)
    out = f.apply(_ctx(vix=35.0))
    assert out.action == "skip_entry"
    assert out.metadata["counter"] == "n_vix_skip"


def test_vix_filter_passes_normal_vix():
    f = VixFilter(threshold=30.0)
    out = f.apply(_ctx(vix=20.0))
    assert out.action == "pass"


def test_vix_filter_boundary():
    f = VixFilter(threshold=30.0)
    assert f.apply(_ctx(vix=30.0)).action == "pass"   # equal → not triggered
    assert f.apply(_ctx(vix=30.1)).action == "skip_entry"
```

## Bước 5 — Verify regression vẫn pass

```bash
python -m ruff check src/components/fusion/strategies/pre_entry/vix_filter.py
python -m mypy src/components/
python -m pytest tests/components/fusion/test_vix_filter.py -v
# Quan trọng: champion parity không đổi
python -m pytest tests/regression/test_v22_parity.py -v
```

## Priority conventions

- `0-9`: Core exits (hard stop, ATR stop) — chạy trước
- `10-19`: Profit protection (peak protect, trailing)
- `20-29`: Signal-based exits
- `30+`: Filters và guards ít quan trọng hơn

## State carry across bars

Nếu strategy cần nhớ state qua bars (vd: dem số bars đã hold):

```python
def apply(self, ctx: BarContext) -> FusionResult:
    state = ctx.position.strategy_state  # dict shared với bars trước
    hold_count = state.get("vix_hold_count", 0)
    state["vix_hold_count"] = hold_count + 1
    # ...
```

Convention: key `"<strategy_name>:<field>"` để tránh collision.
