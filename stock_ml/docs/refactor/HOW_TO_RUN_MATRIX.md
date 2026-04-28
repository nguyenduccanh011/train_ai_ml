# HOW TO: Run a Grid Search (Matrix Experiment)

## Khi nào dùng

Matrix experiment để test nhiều tổ hợp hyperparameter hoặc component trong một lần chạy.

Ví dụ: test 3 feature set × 2 model type × 2 target = 12 experiments.

---

## Bước 1 — Tạo matrix YAML

```yaml
# config/experiments/matrix/q3_2026.yaml
name: q3_2026_feature_model_search

base:
  strategy: v22
  split:
    train_years: 3
    test_years: 1

axes:
  feature_set:
    - leading_v2
    - leading_v3
    - leading_v4

  model:
    type:
      - lightgbm
      - xgboost

  target:
    type:
      - trend_regime
      - early_wave
```

`axes` là các dimension cần search. `base` là config chung cho tất cả experiments.

Matrix expander tạo `3 × 2 × 2 = 12` ExperimentConfig objects.

## Bước 2 — Validate trước khi chạy

```bash
python -m stock_ml validate matrix/q3_2026
# → Hiển thị tất cả 12 configs, check lỗi trước khi chạy tốn thời gian
```

## Bước 3 — Chạy matrix

```bash
python -m stock_ml run-matrix matrix/q3_2026 --device cpu
```

Output: mỗi experiment in ra summary (trade count, WR, PnL).

## Bước 4 — Compare kết quả

```bash
python -m stock_ml compare \
    matrix/q3_2026_leading_v2_lightgbm_trend_regime \
    matrix/q3_2026_leading_v4_lightgbm_trend_regime \
    matrix/q3_2026_leading_v4_xgboost_early_wave
```

## Bước 5 — Promote winner

Sau khi xác định winner (vd: leading_v4 + lightgbm + early_wave):

```bash
# Copy config thành champion
cp config/experiments/matrix/q3_2026_leading_v4_lightgbm_early_wave.yaml \
   config/experiments/champions/v50_q3.yaml

# Chỉnh sửa name
vim config/experiments/champions/v50_q3.yaml
```

Sau đó generate golden và port thành dedicated runner (xem HOW_TO_PORT_LEGACY_VERSION.md).

---

## Format YAML chi tiết

### Base config

Tất cả fields của `ExperimentConfig` đều valid trong `base`:

```yaml
base:
  strategy: v22
  mods:
    v22_sma200_filter: true
  params:
    proba_threshold_entry: 0.55
  split:
    train_years: 3
    test_years: 1
    min_train_samples: 500
```

### Axes format

Axes có thể là flat list hoặc nested dict:

```yaml
axes:
  # Flat list (scalar value)
  feature_set:
    - leading_v2
    - leading_v4

  # Nested dict (thay một sub-field)
  model:
    type:
      - lightgbm
      - xgboost

  # Params (thay một param)
  params:
    proba_threshold_entry:
      - 0.50
      - 0.55
      - 0.60
```

### Ví dụ nhanh (2×2 để test)

```yaml
# config/experiments/matrix/quick_2x2.yaml
name: quick_test
base:
  strategy: v22
  split:
    train_years: 2
    test_years: 1
axes:
  feature_set:
    - leading_v2
    - leading_v4
  model:
    type:
      - lightgbm
      - xgboost
```

Chạy:

```bash
python -m stock_ml run-matrix matrix/quick_2x2 --device cpu
```

---

## Tips

- `--device cpu` cho grid search để reproducible (LightGBM GPU non-deterministic)
- Dùng `--symbols-limit 10` trong benchmark nếu muốn quick preview timing
- Matrix config tên experiment = `{name}_{axis1}_{axis2}_...` (auto-generated)
- Không cần commit matrix YAML vào git (chỉ commit winner sau khi promote)
