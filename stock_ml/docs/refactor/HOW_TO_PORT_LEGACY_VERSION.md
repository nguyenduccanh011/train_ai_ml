# HOW TO: Port a Legacy Version to Champion

## Khi nào promote

Promote legacy version lên champion khi:
- Version đang được theo dõi live (active production)
- Muốn có exact-parity regression test (thay vì approximate adapter)
- Muốn version chạy qua `python -m stock_ml run champions/vXX`

## Phân biệt adapter vs dedicated runner

| | Legacy Adapter | Dedicated Runner (Champion) |
|--|--|--|
| Parity | Approximate | Exact golden |
| Regression test | Smoke only | `assert_frame_equal(check_exact=True)` |
| Runtime | Via LegacyVersionAdapter | Direct backtest function |
| Dùng cho | 49 non-champion | 11 champion |

## Checklist

1. Confirm version chưa phải champion
2. Tạo dedicated runner
3. Export từ `runners/__init__.py`
4. Tạo exact-parity regression test
5. Verify golden match
6. Thêm vào `CHAMPION_RUNNER_MAP`
7. Update `CHAMPION_VERSIONS` frozenset

---

## Bước 1 — Check version hiện tại

```bash
python -m stock_ml list-legacy  # xem version có trong đây không
python -m stock_ml list-experiments  # xem có champion YAML chưa
```

## Bước 2 — Tạo dedicated runner

Xem `src/components/runners/v34_runner.py` làm mẫu:

```python
# src/components/runners/v45_runner.py
"""Parity runner cho v45 — V37a engine với extended hold rules."""

from __future__ import annotations

from typing import Any

from src.components.runners.v34_runner import _run_v34_lineage_cache


def run_v45(
    symbols: list[str],
    data_dir: str,
    *,
    prediction_cache: list[dict[str, Any]] | None = None,
    device: str = "cpu",
    mods: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> list[Any]:
    # Identify backtest function từ experiments/
    from experiments.run_v45 import backtest_v45

    return _run_v34_lineage_cache(
        "v45",
        backtest_v45,
        symbols,
        data_dir,
        prediction_cache=prediction_cache,
        device=device,
        extra_mods=mods or {},
        extra_params=params or {},
    )


def trades_to_v45_dataframe(trades: list[Any]):
    from src.components.runners.v34_runner import trades_to_v34_dataframe
    return trades_to_v34_dataframe(trades)  # reuse nếu schema giống
```

Nếu version có schema khác V34 lineage → implement riêng (xem `v22_runner.py` hoặc `v19_3_runner.py`).

## Bước 3 — Export

```python
# src/components/runners/__init__.py
from src.components.runners.v45_runner import run_v45, trades_to_v45_dataframe

__all__ = [
    # ... existing ...
    "run_v45",
    "trades_to_v45_dataframe",
]
```

## Bước 4 — Regression test

```python
# tests/regression/test_v45_parity.py
"""Regression: v45 component runner phải match golden trades_v45.csv exact."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = REPO_ROOT / "stock_ml" / "tests" / "regression" / "golden"


@pytest.mark.regression
def test_v45_matches_golden() -> None:
    from run_pipeline import _build_predictions
    from src.components.runners import run_v45, trades_to_v45_dataframe
    from src.env import resolve_data_dir

    golden_csv = GOLDEN_DIR / "trades_v45.csv"
    golden_meta = GOLDEN_DIR / "trades_v45.meta.json"

    if not golden_csv.exists():
        pytest.fail(f"Golden không có: {golden_csv}. Chạy golden baseline trước.")

    meta = json.loads(golden_meta.read_text())
    symbols = meta["symbols"]
    data_dir = resolve_data_dir("../portable_data/vn_stock_ai_dataset_cleaned")
    if not Path(data_dir).is_dir():
        pytest.skip(f"Data dir không có: {data_dir}")

    cache = _build_predictions(
        symbols,
        meta["feature_set"],
        meta["target_config"],
        "cpu",
        model_type=meta["model_type"],
        exit_model_cfg=meta.get("exit_model_config"),
    )

    trades = run_v45(symbols, data_dir, prediction_cache=cache, device="cpu")
    df_raw = trades_to_v45_dataframe(trades).reset_index(drop=True)
    df_new = pd.read_csv(io.StringIO(df_raw.to_csv(index=False))).reset_index(drop=True)
    df_gold = pd.read_csv(golden_csv).reset_index(drop=True)

    assert len(trades) == meta["n_trades"]
    pd.testing.assert_frame_equal(df_new, df_gold, check_exact=True)
```

## Bước 5 — Generate golden baseline

```bash
PYTHONHASHSEED=42 python run_pipeline.py \
    --version v45 --device cpu --force --no-export

cp results/trades_v45.csv tests/regression/golden/
cp results/trades_v45.meta.json tests/regression/golden/

cd tests/regression/golden
sha256sum trades_v45.csv >> checksums.txt
```

## Bước 6 — Thêm vào orchestrator

```python
# src/pipeline/orchestrator.py
CHAMPION_RUNNER_MAP: dict[str, str] = {
    # ... existing ...
    "v45": "src.components.runners.v45_runner",
}
CHAMPION_DF_CONVERTER_MAP: dict[str, str] = {
    # ... existing ...
    "v45": "trades_to_v45_dataframe",
}
```

## Bước 7 — Update adapter

```python
# src/pipeline/legacy_adapter.py
CHAMPION_VERSIONS = frozenset([
    "rule", "v19_3", "v22", "v32", "v34", "v35b",
    "v37a", "v37a_exit", "v37d", "v39d", "v42_a",
    "v45",  # <-- thêm mới
])
```

## Verify final

```bash
PYTHONHASHSEED=42 python -m pytest tests/regression/test_v45_parity.py -v
python -m pytest tests/regression/test_champions.py -v  # hash check pass
python -m stock_ml validate champions/v45
python -m stock_ml run champions/v45 --device cpu
```
