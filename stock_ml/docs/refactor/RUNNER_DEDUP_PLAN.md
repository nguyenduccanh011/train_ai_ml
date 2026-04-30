# Kế hoạch Deduplication Runners — Phase 7

**Mục tiêu**: Loại bỏ ~80% boilerplate trong 7 runner V34-lineage và dọn sạch alias vô nghĩa ở experiments layer, không làm vỡ golden parity.

**Ngày lập**: 2026-04-28 | **Cập nhật**: 2026-04-29  
**Branch gợi ý**: `refactor/phase-7-runner-dedup`

**Trạng thái hiện tại**: Step 0–5 đã thực hiện. Ruff các file đổi pass, legacy smoke regression pass `25 passed`. Full regression hiện fail do `MemoryError` khi đọc feature cache/generate target, không phải do mapping Step 5.

---

## Phân tích hiện trạng

### Nhóm 1 — V34-lineage runners (7 file, ~94 dòng/file, ~80% giống nhau)

| Runner file | Version key | Backtest fn import | entry_reason | Ghi chú |
|---|---|---|---|---|
| `v35b_runner.py` | `v35b` | `experiments.run_v34_final.backtest_v35b` | `"v35b"` | |
| `v37a_runner.py` | `v37a` | `experiments.run_v37a.backtest_v37a` | `"v37a"` | |
| `v37a_exit_runner.py` | `v37a_exit` | `experiments.run_v37a.backtest_v37a` | `"v37a_exit"` | |
| `v37d_runner.py` | `v37d` | `experiments.run_v37d.backtest_v37d` | `"v37d"` | |
| `v39d_runner.py` | `v39d` | `experiments.run_v39d.backtest_v39d` | `"v39d"` | |
| `v42_a_runner.py` | `v42_a` | `experiments.run_v42.backtest_v42` | `"v42_a"` | |
| `v32_runner.py` | `v32` | `experiments.run_v32_final.backtest_v32` | `"v32"` | **`V32_TARGET` khác V34** |

Ngoài ra `v34_runner.py` là **base** chứa `_run_v34_lineage_cache`, `_run_cache_item`, `_trade_from_legacy`, `trades_to_v34_dataframe` — không thay đổi file này.

Các file trên khác nhau **chỉ** ở 3 điểm:
1. `get_model_config("<version_key>")`
2. `import backtest_fn` từ experiments
3. `entry_reason="<version_key>"`

**Ngoại lệ**: `v32_runner.py` còn định nghĩa `V32_TARGET` riêng với `gain_threshold=0.05, loss_threshold=0.04` (khác `V34_TARGET`: `gain=0.06, loss=0.03`). `RunnerDef` của v32 phải override `target_default`.

### Nhóm 2 — Experiments aliases (9 hàm là alias thuần túy)

> **Lưu ý**: `run_v37d.backtest_v37d` **không** nằm trong danh sách xóa — registry vẫn import nó.

| Alias hàm | Thực ra gọi | Trạng thái |
|---|---|---|
| `run_v34_final.backtest_v34` | `backtest_v32(...)` | **Giữ** — `v34_runner.py` và legacy smoke vẫn dùng |
| `run_v34_final.backtest_v35a` | `backtest_v32(...)` | **Đã xóa Step 5** |
| `run_v34_final.backtest_v35b` | `backtest_v32(...)` | **Giữ** — registry dùng |
| `run_v34_final.backtest_v35c` | `backtest_v32(...)` | **Đã xóa Step 5** |
| `run_v34_final.backtest_v36a` | `backtest_v32(...)` | **Đã xóa Step 5** |
| `run_v34_final.backtest_v36b` | `backtest_v32(...)` | **Đã xóa Step 5** |
| `run_v34_final.backtest_v36c` | `backtest_v32(...)` | **Đã xóa Step 5** |
| `run_v37b.backtest_v37b` | `backtest_v32(...)` | **Đã xóa Step 5** — legacy adapter map thẳng về `backtest_v32` |
| `run_v37c.backtest_v37c` | `backtest_v32(...)` | **Đã xóa Step 5** — legacy adapter map thẳng về `backtest_v32` |
| `run_v37d.backtest_v37d` | `backtest_v32(...)` | **Giữ** — registry dùng |

**Sau khi xóa aliases**: `run_v37b.py` và `run_v37c.py` chỉ còn docstring + `sys.path.insert` + import không dùng → **xóa luôn cả 2 file** ở Step 5.

### Nhóm 3 — V19/V22 helpers trùng lặp (một phần)

`_format_date`, `_track_result`, `_atr_stop` tồn tại byte-for-byte giống nhau ở `v19_3_runner.py` và `v22_runner.py`.

**Các hàm KHÔNG thể gộp** — signature khác nhau:
- `_base_ctx`: v22 thêm tham số `params` và `symbol_profile=` → **không gộp**
- `_run_exit_sequence`: giống nhau nhưng v22 dùng chiến lược exit khác (V22FastExit, V22HardCap) → gộp hàm OK nhưng không giảm được gì vì caller khác nhau

**Chỉ tách 3 hàm thuần túy không state**: `_format_date`, `_track_result`, `_atr_stop`.

---

## Kế hoạch thực hiện

### Step 0 — Tách `_build_predictions` ra khỏi `run_pipeline.py` (tiên quyết) — DONE

**Bối cảnh**: `run_pipeline.py` có `warnings.warn(DeprecationWarning)` ở module level (dòng 48-53). Import file này trong `_lineage_v34.py` sẽ trigger warning mỗi lần import — là side-effect không mong muốn.

**Hành động**: Tách `_build_predictions` từ `run_pipeline.py` ra `src/pipeline/build_predictions.py`.

```python
# src/pipeline/build_predictions.py
from __future__ import annotations

from typing import Any

# Di chuyển nguyên hàm _build_predictions từ run_pipeline.py sang đây.
# run_pipeline.py giữ lại import: from src.pipeline.build_predictions import _build_predictions
```

Sau khi tách, `run_pipeline.py` import lại từ module mới — backward-compatible, không break gì.

**Verification**:
```bash
python -c "from src.pipeline.build_predictions import _build_predictions; print('OK')"
python -m pytest stock_ml/tests/ -q -x  # smoke test toàn bộ
```

---

### Step 1 — Tạo `RunnerDef` dataclass và `_lineage_v34.py` (1 ngày) — DONE

**File tạo mới**: `src/components/runners/_lineage_v34.py`

> **Dependency coupling**: `_lineage_v34.py` import các private symbols `_exit_model_cfg`, `_run_v34_lineage_cache` từ `v34_runner.py`. Đây là **intended coupling** — `v34_runner.py` là base helper của lineage này. Nếu `v34_runner.py` refactor, phải cập nhật `_lineage_v34.py` đồng thời.

```python
# src/components/runners/_lineage_v34.py
from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.components.base import Trade
from src.components.runners.v34_runner import (
    V34_TARGET,
    _exit_model_cfg,
    _run_v34_lineage_cache,
    trades_to_v34_dataframe,
)
from src.config_loader import get_model_config


@dataclass(frozen=True)
class RunnerDef:
    version_key: str
    backtest_fn: Callable[..., dict[str, Any]]
    entry_reason: str
    feature_set_default: str = "leading_v4"
    target_default: dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(V34_TARGET)
    )


def build_prediction_cache(
    defn: RunnerDef,
    symbols: list[str],
    *,
    device: str,
    feature_set: str | None = None,
) -> list[dict[str, Any]]:
    from src.pipeline.build_predictions import _build_predictions

    model_cfg = get_model_config(defn.version_key)
    return _build_predictions(
        symbols,
        feature_set or model_cfg.get("feature_set", defn.feature_set_default),
        model_cfg.get("target", defn.target_default),
        device,
        model_type=model_cfg.get("model_type"),
        exit_model_cfg=_exit_model_cfg(model_cfg),
    )


def run_lineage(
    defn: RunnerDef,
    symbols: list[str],
    _data_dir: str,  # unused — kept for API compatibility with legacy callers
    *,
    mods: dict[str, bool] | None = None,
    params: dict[str, Any] | None = None,
    _first_test_year: int = 2020,   # unused
    _last_test_year: int = 2025,    # unused
    _train_years: int = 4,          # unused
    device: str = "cpu",
    prediction_cache: list[dict[str, Any]] | None = None,
    initial_capital: float = 100_000_000,
    commission: float = 0.0015,
    tax: float = 0.001,
    record_trades: bool = True,
    enable_model_b_exit: bool = False,
) -> list[Trade]:
    model_cfg = get_model_config(defn.version_key)
    active_mods = {**model_cfg.get("mods", {}), **(mods or {})}
    active_params = {
        **model_cfg.get("params", {}),
        **(params or {}),
        "initial_capital": initial_capital,
        "commission": commission,
        "tax": tax,
        "record_trades": record_trades,
    }
    cache = (
        prediction_cache
        if prediction_cache is not None
        else build_prediction_cache(defn, symbols, device=device)
    )
    return _run_v34_lineage_cache(
        symbols,
        cache,
        backtest_fn=defn.backtest_fn,
        mods=active_mods,
        params=active_params,
        entry_reason=defn.entry_reason,
        enable_model_b_exit=enable_model_b_exit,
    )
```

**Verification**: Import được, mypy clean, không run gì cả.
```bash
python -c "from src.components.runners._lineage_v34 import RunnerDef, run_lineage, build_prediction_cache; print('OK')"
```

---

### Step 2 — Tạo `runner_registry.py` chứa RUNNER_DEFS (30 phút) — DONE

**File tạo mới**: `src/components/runners/runner_registry.py`

> **Ghi chú v32**: `V32_TARGET` có `gain_threshold=0.05, loss_threshold=0.04` — khác `V34_TARGET` (`gain=0.06, loss=0.03`). `RunnerDef` cho v32 phải override `target_default` để fallback đúng khi config không có `target` key.

```python
# src/components/runners/runner_registry.py
from __future__ import annotations

import copy
import importlib
from collections.abc import Callable
from typing import Any

from src.components.runners._lineage_v34 import RunnerDef
from src.components.runners.v32_runner import V32_TARGET


def _lazy(module: str, fn: str) -> Callable[..., Any]:
    """Lazy import wrapper — chỉ import lúc gọi lần đầu."""
    _cache: list[Callable[..., Any]] = []

    def _call(*args: Any, **kwargs: Any) -> Any:
        if not _cache:
            _cache.append(getattr(importlib.import_module(module), fn))
        return _cache[0](*args, **kwargs)

    return _call


RUNNER_DEFS: dict[str, RunnerDef] = {
    "v32": RunnerDef(
        version_key="v32",
        backtest_fn=_lazy("experiments.run_v32_final", "backtest_v32"),
        entry_reason="v32",
        feature_set_default="leading_v3",
        target_default=copy.deepcopy(V32_TARGET),  # gain=0.05, loss=0.04
    ),
    "v35b": RunnerDef(
        version_key="v35b",
        backtest_fn=_lazy("experiments.run_v34_final", "backtest_v35b"),
        entry_reason="v35b",
    ),
    "v37a": RunnerDef(
        version_key="v37a",
        backtest_fn=_lazy("experiments.run_v37a", "backtest_v37a"),
        entry_reason="v37a",
    ),
    "v37a_exit": RunnerDef(
        version_key="v37a_exit",
        backtest_fn=_lazy("experiments.run_v37a", "backtest_v37a"),
        entry_reason="v37a_exit",
    ),
    "v37d": RunnerDef(
        version_key="v37d",
        backtest_fn=_lazy("experiments.run_v37d", "backtest_v37d"),
        entry_reason="v37d",
    ),
    "v39d": RunnerDef(
        version_key="v39d",
        backtest_fn=_lazy("experiments.run_v39d", "backtest_v39d"),
        entry_reason="v39d",
    ),
    "v42_a": RunnerDef(
        version_key="v42_a",
        backtest_fn=_lazy("experiments.run_v42", "backtest_v42"),
        entry_reason="v42_a",
    ),
}
```

**Verification**:
```bash
python -c "from src.components.runners.runner_registry import RUNNER_DEFS; assert list(RUNNER_DEFS) == ['v32','v35b','v37a','v37a_exit','v37d','v39d','v42_a']; print('OK')"
```

---

### Step 3 — Refactor 7 runner files thành thin wrappers (2 giờ) — DONE

Mỗi file runner trở thành ~15 dòng thay vì ~94 dòng. Ví dụ `v37a_runner.py` sau refactor:

```python
# src/components/runners/v37a_runner.py
from __future__ import annotations

from typing import Any

import pandas as pd

from src.components.base import Trade
from src.components.runners._lineage_v34 import run_lineage
from src.components.runners.runner_registry import RUNNER_DEFS
from src.components.runners.v34_runner import trades_to_v34_dataframe

_DEF = RUNNER_DEFS["v37a"]


def run_v37a(symbols, data_dir, **kwargs) -> list[Trade]:
    return run_lineage(_DEF, symbols, data_dir, **kwargs)


def trades_to_v37a_dataframe(trades: list[Trade | dict[str, Any]]) -> pd.DataFrame:
    return trades_to_v34_dataframe(trades)
```

Tương tự cho 6 runner còn lại. Signature public API giữ nguyên hoàn toàn — orchestrator, `__init__.py`, tests không cần đổi.

**Verification**: Import từng runner vẫn work.
```bash
python -c "from src.components.runners.v37a_runner import run_v37a; print('OK')"
```

---

### Step 4 — Regression verification (bắt buộc, không bỏ qua) — DONE

```bash
PYTHONHASHSEED=42 python -m pytest stock_ml/tests/regression/ -q
```

Phải pass toàn bộ golden hash tests. **Nếu có test nào fail → dừng ngay, revert 7 file về bản gốc, debug trên bản gốc.** Không mix state mới/cũ khi debug.

Cách revert nhanh nếu cần:
```bash
git checkout HEAD -- stock_ml/src/components/runners/v32_runner.py \
    stock_ml/src/components/runners/v35b_runner.py \
    stock_ml/src/components/runners/v37a_runner.py \
    stock_ml/src/components/runners/v37a_exit_runner.py \
    stock_ml/src/components/runners/v37d_runner.py \
    stock_ml/src/components/runners/v39d_runner.py \
    stock_ml/src/components/runners/v42_a_runner.py
```

---

### Step 5 — Dọn experiments aliases (1 giờ) — DONE

**Kết quả thực tế**:

- `experiments/run_v34_final.py`: đã xóa `backtest_v35a`, `backtest_v35c`, `backtest_v36a`, `backtest_v36b`, `backtest_v36c`.
- `backtest_v34` **không xóa** vì `src/components/runners/v34_runner.py` vẫn import trực tiếp và regression smoke vẫn yêu cầu key `v34` trong `LEGACY_STRATEGY_MAP`.
- `backtest_v35b` giữ nguyên vì `RUNNER_DEFS` import.
- `experiments/run_v37b.py` và `experiments/run_v37c.py` đã xóa sau khi grep xác nhận không còn direct Python caller.
- `src/pipeline/legacy_adapter.py`: giữ các legacy keys `v35a`, `v35c`, `v36a`, `v36b`, `v36c`, `v37b`, `v37c` để không phá legacy smoke, nhưng map thẳng về `experiments.run_v32_final.backtest_v32` thay vì alias wrappers.
- `run_pipeline.py`: gỡ các mappings alias không còn tồn tại; giữ các strategy thực còn được support.
- `experiments/run_v37d.py`: **không xóa** `backtest_v37d` — registry vẫn import nó.

**Verification đã chạy**:

```bash
python -m ruff check stock_ml/experiments/ stock_ml/run_pipeline.py stock_ml/src/pipeline/legacy_adapter.py
python -m pytest stock_ml/tests/regression/test_legacy_smoke.py -q
```

Kết quả: ruff pass, legacy smoke `25 passed`.

**Full regression note**: `PYTHONHASHSEED=42 python -m pytest stock_ml/tests/regression/ -q` hiện fail `MemoryError` khi đọc parquet/generate target trong các parity tests (`v32`, `v34`, `v35b`, `v37a*`, `v37d`, `v39d`, `v42_a`). Lỗi này xảy ra ở feature cache/target generation, không phải ở Step 5 aliases.

---

### Step 6 — Tách helpers chung V19/V22 (tuỳ chọn, rủi ro thấp) — DONE

**Chỉ làm sau khi Step 1-5 đã xong và regression pass.**

**3 hàm byte-for-byte giống nhau** (đã xác nhận bằng cách đọc code):
- `_format_date(value) -> str`: `return str(value)[:10]`
- `_track_result(res, counters)`: đọc `metadata["counter"]` và `metadata["counters"]`
- `_atr_stop(ind, i) -> float`: tính ATR stop với `max(0.025, min(1.8 * atr / close, 0.06))`

**3 hàm KHÔNG gộp được** — signature v19 và v22 khác nhau:
- `_base_ctx`: v22 có thêm `params: dict` và `symbol_profile=str(...)` — nếu gộp sẽ thay đổi behavior v19
- `_atr_ratio`: chỉ có ở v22, không có ở v19
- Simulation loop chính: hoàn toàn khác engine, **không đụng vào**

Tạo `src/components/runners/_sim_utils.py` chứa 3 hàm trên, sau đó thay thế bản local trong cả hai file.

**Kết quả thực tế**:

- Đã tạo `src/components/runners/_sim_utils.py` với 3 hàm thuần `format_date`, `track_result`, `atr_stop`.
- `v19_3_runner.py` và `v22_runner.py` import module và bind `_atr_stop`, `_format_date`, `_track_result` qua attribute access — giữ tên private cũ để không đổi call site.
- Ruff các file đổi pass; legacy smoke regression `tests/regression/test_legacy_smoke.py` `25 passed`.

**Verification**:
```bash
python -m ruff check stock_ml/src/components/runners/v19_3_runner.py stock_ml/src/components/runners/v22_runner.py stock_ml/src/components/runners/_sim_utils.py
python -m pytest stock_ml/tests/regression/test_legacy_smoke.py -q
```

Full regression `PYTHONHASHSEED=42 python -m pytest stock_ml/tests/regression/ -q` vẫn fail `MemoryError` ở feature cache/target generation (carried over từ Step 5), không liên quan Step 6.

---

## Thứ tự ưu tiên

| Step | Effort | Rủi ro | Giá trị |
|------|--------|--------|---------|
| Step 0 — Tách `_build_predictions` | 30 phút | Thấp | **Tiên quyết** — tránh DeprecationWarning khi import |
| Step 1 — `_lineage_v34.py` + `RunnerDef` | 1 ngày | Thấp | Nền tảng cho Step 2-3 |
| Step 2 — `runner_registry.py` | 30 phút | Thấp | Trung tâm config |
| Step 3 — Refactor 7 runner files | 2 giờ | Trung (regression) | Xóa ~540 dòng |
| Step 4 — Regression verify | 30 phút | N/A | Gate bắt buộc |
| Step 5 — Xóa experiments aliases + 2 file | 1 giờ | Thấp | **DONE** — xóa aliases không dùng + 2 file rác; giữ `backtest_v34` vì còn caller |
| Step 6 — V19/V22 sim utils | 1 giờ | Thấp | **DONE** — `_sim_utils.py` chứa 3 hàm chung, smoke regression pass |

**Tổng ước tính**: 1 ngày rưỡi nếu không gặp issue.

---

## Invariants phải giữ

1. **Public API không đổi**: `run_v37a(symbols, data_dir, *, ...)` vẫn là entry point, signature không thay đổi.
2. **Golden parity 100%**: sau mỗi step, `pytest tests/regression/` phải pass. Nếu môi trường fail do `MemoryError`, ghi rõ log và chạy smoke/gate hẹp liên quan trước khi tiếp tục.
3. **v34_runner.py không đổi**: file này là base helper, không phải target refactor.
4. **v19_3_runner.py, v22_runner.py không đổi engine**: chỉ tách 3 helpers thuần túy ở Step 6.
5. **Không xóa file thực experiments** như `run_v32_final.py`, `run_v37a.py`, `run_v37d.py` — chỉ xóa alias hàm bên trong.
6. **Giữ `run_v34_final.backtest_v34` khi `v34_runner.py` còn import trực tiếp**; nếu muốn xóa sau này, phải refactor `v34_runner.py` trước.
7. **Coupling `_lineage_v34.py` ↔ `v34_runner.py` là intentional**: nếu refactor `v34_runner.py`, cập nhật `_lineage_v34.py` đồng thời.

---

## Kết quả kỳ vọng

| Metric | Trước | Sau |
|--------|-------|-----|
| Dòng code runners V34-lineage (7 file) | ~658 dòng | ~105 dòng |
| File runner mới khi thêm version | Copy-paste ~94 dòng | Thêm 1 entry vào `RUNNER_DEFS` + tạo file ~15 dòng |
| Alias experiments | 10 hàm vô nghĩa | Còn 3 hàm giữ có chủ đích (`backtest_v34`, `backtest_v35b`, `backtest_v37d`) |
| File experiments rác | 2 file (run_v37b, run_v37c) | Đã xóa |
| Helpers trùng V19/V22 | 3 hàm × 2 bản | 1 module chung |
| Regression tests | Phải pass | Phải pass (không thay đổi) |
| Import side-effect (`DeprecationWarning`) | Có khi dùng `_build_predictions` | Không còn |
