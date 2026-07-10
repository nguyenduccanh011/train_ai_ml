# Kế hoạch triển khai fix live inference parity-safe

## Mục tiêu

Fix live signal cho các model `pooled_global_rerun`, đặc biệt:

```text
top1_exit_model_65_no_bear_no_chop_raw0_minhold3
```

Yêu cầu:

- Giữ nguyên hành vi backtest/golden parity.
- Live năm 2026 dùng đúng fold tiếp theo:

```text
train_2022-2025_test_2026
```

- Train vẫn dùng dataframe có `target` / `target_sell` đầy đủ.
- Predict live dùng feature-only rows tới raw latest date, ví dụ `2026-05-15`.
- Không bị double truncate do entry target horizon 21 + exit label horizon 21.
- Kiểm chứng bằng cách force 2025 và so với kết quả đã lưu.

## Hiện trạng

### Raw data

Raw data không lỗi:

```text
raw_global_max: 2026-05-15
64/65 mã tới 2026-05-15
AAV tới 2026-05-14
```

### Backtest label truncation

Luồng backtest hiện tại:

```text
raw data max                 2026-05-15
sau entry target horizon 21  2026-04-13
sau exit label horizon 21    2026-03-13
```

Lý do:

1. `early_wave_v2` cần 21 bar tương lai để tạo `target`.
2. `generate_exit_labels()` lại cần thêm 21 bar tương lai để tạo `target_sell`.
3. Backtest dùng dataframe đã labeled/dropna nên bị cắt 2 lần.

Đây là đúng cho backtest, vì backtest không được dùng row chưa có label thật.

### Live inference bug

Live signal đang reuse dataframe/cache backtest-safe cho inference:

```text
live signal -> prediction cache từ trainer/backtest -> sym_test_df đã labeled/dropna
```

Kết quả: live signal dừng ở `2026-03-13`, dù feature/raw data tới `2026-05-15`.

## Bài học từ thử nghiệm trước

Đã thử tự dựng helper live mới:

```text
train_df = labeled/dropna
infer_df = feature-only latest
fit model manually
predict manually
```

Kết quả không đạt parity:

```text
Saved baseline riêng 2025:
trades=344, WR=53.20%, total_pnl=856.29, avg=2.4892

Current live-style 2025:
trades=419, WR=54.18%, total_pnl=468.90, avg=1.1191

Standard trainer 2025:
trades=369, WR=82.66%, total_pnl=3286.39, avg=8.9062
```

Nguyên nhân:

- Helper mới clone logic trainer thủ công.
- Có rủi ro lệch `run_context`, feature set, model stack, target config, cache, split, strategy params.
- Không nên duplicate trainer logic.

Kết luận:

```text
Trainer/backtest path phải là source of truth.
Live chỉ được mở rộng inference rows, không được tự dựng lại training logic.
```

## Nguyên tắc thiết kế

### Không đổi

Không đổi các phần sau:

- `src/pipeline/trainer.py` behavior cho backtest.
- `src/pipeline/Pipeline` behavior.
- Strategy config.
- Horizon/target/exit model config.
- Golden/backtest parity.

### Được đổi

Chỉ đổi live runtime path:

- `app/serve_train61_model.py`
- `tools/run_model_backtest.py` nếu cần runner/debug report
- Có thể thêm helper nội bộ, nhưng phải reuse trainer source-of-truth.

### Invariant bắt buộc

Với active fold cũ đã có label đầy đủ, live path phải match standard trainer:

```text
force active_year=2025
standard trainer active cache == live parity cache
entry_diff_total == 0
exit_diff_total == 0
trades match
```

Sau đó mới mở rộng 2026 latest feature rows.

## Thiết kế đúng

### 1. Build standard trainer cache trước

Dùng lại source-of-truth:

```python
from src.pipeline.trainer import build_prediction_cache

standard_cache = build_prediction_cache(cfg, symbols, device="cpu")
```

Filter active fold:

```text
active_year = cfg.split.last_test_year
active_items = items có sym_test_df thuộc active_year
```

Với 2026 hiện tại, cache này có thể chỉ tới `2026-03-13` vì labeled/dropna.

### 2. Cần trained model trong cache

Hiện `build_prediction_cache()` chỉ trả:

```text
symbol
y_pred
y_pred_exit
y_proba
classes
returns
sym_test_df
feature_cols
```

Không trả trained `entry_model` / `exit_model`.

Để predict feature-only latest mà vẫn reuse trainer, cần thêm option nội bộ:

```python
build_prediction_cache(..., include_models=False)
```

Khi `include_models=True`, mỗi cache item active fold có thêm:

```text
entry_model
exit_model
window
train_rows
train_end_date
```

Default `False` để không ảnh hưởng backtest/golden/cache serialization.

Quan trọng:

- Không serialize model vào persistent `PredictionCacheManager` mặc định.
- Chỉ dùng trong live runtime path.
- Nếu `PredictionCacheManager` đang pickle cache, không bật `include_models` qua path đó.

### 3. Build feature-only inference dataframe

Dùng cùng feature engine/config resolved bởi trainer.

Cách an toàn nhất:

- Tách helper trong `src/pipeline/trainer.py` hoặc helper nội bộ trả thêm metadata cần thiết:

```text
feature_df raw computed
feature_cols
run_context
engine config
active_window
```

Nhưng tránh refactor lớn. Phương án gọn:

- Trong live function, dùng `feature_cols` từ standard active cache.
- Compute feature_df bằng cùng `cfg.feature_set()` và data_dir.
- Drop NaN chỉ trên `feature_cols`, không yêu cầu `target` / `target_sell`.
- Filter active year:

```text
active_window.test_start <= timestamp <= active_window.test_end
```

### 4. Predict latest rows bằng trained fold model

Với mỗi symbol:

```text
sym_infer_df = feature_df[symbol, active year, feature_cols non-null]
X = finite_matrix(sym_infer_df[feature_cols])
y_pred = entry_model.predict(X)
y_pred_exit = exit_model.predict(X)
```

Sau đó tạo cache item giống trainer:

```text
symbol
y_pred
y_pred_exit
y_proba
classes
returns
sym_test_df = sym_infer_df
feature_cols
train_rows
train_end_date
model_mode = pooled_global_live_next_fold_feature_only
window_label
```

### 5. Backtest replay bằng Pipeline không đổi

Sau khi có live cache items:

```python
result = Pipeline(cfg, symbols=symbols, device="cpu", prediction_cache=live_cache_items).run()
```

Pipeline/backtester/strategy giữ nguyên.

## File cần sửa

### `src/pipeline/trainer.py`

Thêm optional param:

```python
def build_prediction_cache(
    cfg,
    symbols,
    *,
    device="cpu",
    include_models: bool = False,
) -> list[dict[str, Any]]:
```

Khi append result:

```python
item = {
    ... existing fields ...
}
if include_models:
    item["entry_model"] = model
    item["exit_model"] = sell_model
    item["window_label"] = window.label
    item["train_rows"] = len(train_df)
    item["train_end_date"] = str(window.train_end.date())
    item["test_start"] = str(window.test_start.date())
    item["test_end"] = str(window.test_end.date())
```

Rủi ro:

- Nếu item có model bị cache manager pickle ngoài ý muốn.
- Giảm bằng cách chỉ bật `include_models=True` trong live path không dùng persistent cache.

### `app/serve_train61_model.py`

Sửa `_build_live_prediction_cache_pooled_global()`:

Hiện sai nếu tự dựng train model thủ công.

Đúng:

1. Gọi `build_prediction_cache(cfg, symbols, device="cpu", include_models=True)`.
2. Lấy active fold items.
3. Nếu latest labeled cache đã tới raw latest thì return chuẩn.
4. Nếu raw latest > labeled latest:
   - compute feature-only dataframe
   - dùng models trong active items để predict lại sym_df tới latest
   - return cache items mở rộng.

Pseudo:

```python
def _build_live_prediction_cache_pooled_global(...):
    standard_items = build_prediction_cache(..., include_models=True)
    active_items = _filter_active_year(standard_items, cfg.split.last_test_year)
    feature_df = _load_live_feature_df(...)
    extended = []
    for item in active_items:
        sym = item["symbol"]
        sym_df = _feature_only_rows(feature_df, sym, active_window, item["feature_cols"])
        if sym_df.empty:
            extended.append(_strip_models(item))
            continue
        extended.append(_predict_with_item_models(item, sym_df, cfg.target_dict()))
    return extended
```

Các helper nên private:

```text
_active_year_items()
_strip_model_objects()
_load_pooled_live_feature_df()
_predict_cache_item_with_models()
```

### `tools/run_model_backtest.py`

Dùng để debug/report.

Cần:

- Support `pooled_global_rerun` bằng `_build_live_prediction_cache_pooled_global()`.
- Thêm flag debug:

```text
--force-last-test-year 2025
--compare-standard-cache
```

Report cần xuất:

```text
trades.csv
closed_trades.csv
daily_signals.csv
yearly_summary.csv
benchmark_yearly.csv
summary.json
```

### Optional: `tools/compare_live_parity.py`

Nếu muốn tách test rõ ràng, thêm script debug riêng:

```text
python tools/compare_live_parity.py --model-id top1_exit_model_65_no_bear_no_chop_raw0_minhold3 --year 2025
```

Output:

```json
{
  "entry_diff_total": 0,
  "exit_diff_total": 0,
  "standard_trades": ...,
  "live_trades": ...,
  "trade_diff": 0
}
```

## Quy trình triển khai

### Phase 1 — Rollback phần sai

Rollback helper live tự dựng model thủ công trong `app/serve_train61_model.py`.

Giữ lại nếu muốn:

- report export trong `tools/run_model_backtest.py`, nhưng phải đảm bảo nó gọi live path mới đúng.

### Phase 2 — Add include_models vào trainer

Sửa `src/pipeline/trainer.py`:

- Thêm `include_models=False`.
- Khi true, attach model/window metadata.
- Default false nên backtest không đổi.

Kiểm tra compile.

### Phase 3 — Rebuild live pooled using trainer models

Sửa `app/serve_train61_model.py`:

- Gọi trainer với `include_models=True`.
- Active fold source-of-truth.
- Compute latest feature-only rows.
- Predict bằng trained fold model.
- Strip model objects trước khi trả cache cho Pipeline.

### Phase 4 — Parity test 2025

Chạy 3 bộ:

#### A. Saved baseline 2025

Đọc:

```text
results/rs_ablation_prev65/recheck2025_weighted_rs_prev65_relaxed_no_bear_no_chop_trades.csv
```

Expected riêng 2025:

```text
trades=344
WR=53.20%
total_pnl=856.29
avg_pnl=2.4892
```

#### B. Standard current path 2025 cùng config

```text
build_prediction_cache(cfg last_test_year=2025)
Pipeline(... prediction_cache=active_items)
```

#### C. Live path 2025

```text
_build_live_prediction_cache_pooled_global(cfg last_test_year=2025)
Pipeline(... prediction_cache=live_items)
```

Acceptance:

```text
B == C exact hoặc near-exact.
```

Nếu A != B:

```text
Do config/path khác saved baseline.
Không được coi là live bug.
Phải so config saved vs current riêng.
```

### Phase 5 — Live 2026 latest test

Run:

```text
python tools/run_model_backtest.py --model-id top1_exit_model_65_no_bear_no_chop_raw0_minhold3
```

Acceptance:

```text
daily_signals latest date = 2026-05-15
prediction rows > labeled-truncated rows
active_window = train_2022-2025_test_2026
train_end_date = 2025-12-31 hoặc last labeled train date <= 2025-12-31
```

### Phase 6 — Report

Xuất và đọc:

```text
summary.json
trades.csv
closed_trades.csv
daily_signals.csv
yearly_summary.csv
benchmark_yearly.csv
```

Báo cáo:

- Latest buy list.
- Latest sell list.
- Buy/sell grouped by date.
- Closed trade stats.
- Yearly stats.
- Benchmark/index if available.

## Commands đề xuất

### Compile

```powershell
python -m py_compile app/serve_train61_model.py src/pipeline/trainer.py tools/run_model_backtest.py
```

### Run model

```powershell
python tools/run_model_backtest.py --model-id top1_exit_model_65_no_bear_no_chop_raw0_minhold3
```

### Force 2025 debug

Nếu thêm flag:

```powershell
python tools/run_model_backtest.py --model-id top1_exit_model_65_no_bear_no_chop_raw0_minhold3 --force-last-test-year 2025
```

### Compare parity

Nếu thêm script:

```powershell
python tools/compare_live_parity.py --model-id top1_exit_model_65_no_bear_no_chop_raw0_minhold3 --year 2025
```

## Acceptance criteria cuối

### Bắt buộc

```text
2025 standard vs live parity:
entry_diff_total = 0
exit_diff_total = 0
trade_diff = 0
```

```text
2026 latest live:
latest_bar_date = 2026-05-15
active_window = train_2022-2025_test_2026
```

```text
Backtest/golden path không đổi:
build_prediction_cache default output không có model objects
Pipeline normal behavior không đổi
```

### Không bắt buộc

Saved baseline 2025 phải match current config.

Nếu không match, cần phân loại:

```text
config drift / runner drift / dataset drift
```

Không gộp vào live inference fix.

## Rủi ro

### Model object trong cache

Nếu model object lọt vào persistent cache sẽ nặng hoặc pickle lỗi.

Giảm thiểu:

- `include_models=False` default.
- Chỉ bật trong app live path.
- Strip model object trước khi pass tới external report/json nếu cần.

### Config drift

Saved baseline dùng config khác current model registry.

Giảm thiểu:

- Luôn ghi rõ config path trong summary.
- Khi so baseline, so cùng config trước.

### Feature context drift

Nếu live feature_df compute khác trainer feature_df, prediction lệch.

Giảm thiểu:

- Reuse feature_cols từ trainer item.
- Dùng cùng `FeatureEngine`/cache key/code paths.
- Parity test 2025 bắt buộc.

## Kết luận

Fix đúng không phải là viết lại trainer cho live.

Fix đúng là:

```text
trainer/backtest builds canonical trained fold
live reuses trained fold model
live replaces only inference dataframe bằng feature-only latest rows
Pipeline/backtester giữ nguyên
```

Chỉ khi 2025 parity pass mới tin kết quả 2026 latest.
