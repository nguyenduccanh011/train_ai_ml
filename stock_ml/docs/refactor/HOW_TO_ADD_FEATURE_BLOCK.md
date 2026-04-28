# HOW TO: Add a Feature Block

## Khi nào dùng

Thêm feature block khi cần feature set mới hoặc muốn tính thêm indicator chưa có trong 14 block hiện tại.

## Checklist

1. Tạo block class ở `src/components/features/blocks/`
2. Register vào `src/components/features/registry.py`
3. Dùng trong YAML feature set
4. Test equivalence

---

## Bước 1 — Implement block

```python
# src/components/features/blocks/sentiment.py
from __future__ import annotations

import pandas as pd

from src.components.features.base import FeatureBlock


class SentimentBlock(FeatureBlock):
    name = "sentiment"
    requires = ["close", "volume"]  # cột raw cần có trong df

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Tính features — dùng .shift() thay vì look-ahead
        df["sent_vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        df["sent_price_acc"] = df["close"].diff(5) / df["close"].shift(5)
        return df

    def get_feature_names(self) -> list[str]:
        return ["sent_vol_ratio", "sent_price_acc"]
```

**Lưu ý quan trọng**:
- `df = df.copy()` bắt buộc — block không được mutate input
- Không dùng future data (`.shift(-1)`, `.rolling(n).mean().shift(-k)`)
- `get_feature_names()` phải liệt kê đúng tên column block tạo ra

## Bước 2 — Register

```python
# src/components/features/registry.py — thêm vào _BLOCK_REGISTRY
from src.components.features.blocks.sentiment import SentimentBlock

_BLOCK_REGISTRY: dict[str, type[FeatureBlock]] = {
    # ... existing blocks ...
    "sentiment": SentimentBlock,
}
```

## Bước 3 — Dùng trong YAML

```yaml
# config/feature_sets/leading_v5_sentiment.yaml
blocks:
  - ohlcv_basic
  - momentum
  - volume_advanced
  - regime
  - sentiment          # <-- block mới
```

Dùng trong experiment:

```yaml
# config/experiments/champions/v50_sentiment.yaml
name: v50_sentiment
strategy: v22
feature_set: leading_v5_sentiment
target:
  type: trend_regime
model:
  type: lightgbm
split:
  train_years: 3
  test_years: 1
```

## Bước 4 — Test

```python
# tests/components/test_sentiment_block.py
import pandas as pd
from src.components.features.blocks.sentiment import SentimentBlock


def _make_df(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame({
        "close": [100.0 + i * 0.1 for i in range(n)],
        "volume": [1_000_000.0] * n,
    })


def test_sentiment_block_columns():
    block = SentimentBlock()
    df = _make_df()
    out = block.compute(df)
    assert "sent_vol_ratio" in out.columns
    assert "sent_price_acc" in out.columns


def test_sentiment_block_no_lookahead():
    """Feature tại row i chỉ phụ thuộc vào row ≤ i."""
    block = SentimentBlock()
    df = _make_df(50)
    out_full = block.compute(df)
    out_trunc = block.compute(df.iloc[:30])
    # Row 20 phải bằng nhau dù độ dài df khác nhau (chừng đủ warmup)
    pd.testing.assert_series_equal(
        out_full["sent_vol_ratio"].iloc[:20],
        out_trunc["sent_vol_ratio"].iloc[:20],
    )
```

Thêm test equivalence nếu block thay thế logic cũ:

```python
def test_sentiment_block_matches_legacy():
    from src.features.engine import FeatureEngine
    df = load_test_data()
    old = FeatureEngine(feature_set="leading_v5_sentiment").compute_for_all_symbols(df)
    new_engine = build_engine_from_yaml("config/feature_sets/leading_v5_sentiment.yaml")
    new = new_engine.compute_for_all_symbols(df)
    pd.testing.assert_frame_equal(old[["sent_vol_ratio"]], new[["sent_vol_ratio"]])
```

## Verify

```bash
python -m ruff check src/components/features/blocks/sentiment.py
python -m mypy src/components/features/
python -m pytest tests/components/test_sentiment_block.py -v
```

## Khi nào KHÔNG thêm block mới

- Feature đã có trong block khác → dùng block đó
- Feature cần dữ liệu ngoài (news, macro) → cần adapter riêng trước
- Feature phụ thuộc cross-symbol phức tạp → xem `RelativeStrengthBlock` làm mẫu (tách `compute_cross_sectional`)
