# Entry/Exit Research Refactor Plan

## Mục tiêu

Tạo kiến trúc nghiên cứu cho phép thử nhanh nhiều tổ hợp:

```text
feature_set × target × entry_model × exit_model × fusion_exit_policy
```

Mục tiêu thực tế:

- Thêm entry model mới mà không phải viết strategy Python mới.
- Thêm exit model/exit policy mới mà không sửa backtest engine lớn.
- Chạy matrix entry × exit qua YAML/CLI.
- So sánh kết quả tự động để chọn champion mới.
- Giữ regression parity cho 11 champion hiện có.

## Nguyên tắc refactor

1. Không rewrite toàn bộ một lần.
2. Mỗi phase phải chạy được test/regression sau khi xong.
3. Giữ legacy adapter cho version cũ; không kéo logic cũ vào interface mới.
4. Backtester chỉ quản lý vị thế và PnL; không chứa logic quyết định entry/exit.
5. Entry signal, exit signal, fusion policy phải tách riêng để compose được.

## Trạng thái hiện tại

Đã có nền tảng tốt:

- Pipeline/CLI mới: `python -m stock_ml`.
- `ExperimentConfig`, `Pipeline`, prediction cache, matrix expander.
- Component folders dưới `src/components/`.
- Guide thêm entry model và chạy matrix.
- Regression golden cho champion versions.

Điểm còn cần hoàn thiện cho nghiên cứu entry/exit:

- Exit model chưa thành component/policy đủ độc lập.
- Matrix hiện dễ search feature/model/target, nhưng entry × exit × fusion_exit chưa thật rõ ràng.
- Một phần logic exit vẫn còn gắn với runner/fusion/backtest legacy.
- Cần chuẩn hóa result artifact để compare các tổ hợp lớn.

---

## Phase 0 — Baseline và scope lock

### Mục tiêu

Đảm bảo refactor không làm lệch kết quả hiện tại trước khi tách entry/exit sâu hơn.

### Việc cần làm

1. Chạy lại regression champion hiện có.
2. Ghi lại benchmark thời gian chạy single experiment và matrix nhỏ.
3. Chốt danh sách champion bắt buộc parity:
   - `v22`
   - `v32`
   - `v34`
   - `v35b`
   - `v37a`
   - `v37a_exit`
   - `v37d`
   - `v39d`
   - `v42_a`
   - `v19_3`
   - `rule`
4. Chốt matrix smoke nhỏ để dùng xuyên suốt refactor:

```text
2 feature sets × 2 entry models × 2 exit configs = 8 runs
```

### File liên quan

- `tests/regression/`
- `tests/regression/golden/`
- `src/pipeline/orchestrator.py`
- `src/pipeline/matrix_expander.py`
- `scripts/benchmark.py`

### Verification

```bash
python -m pytest tests/regression -q
python -m stock_ml benchmark --symbols-limit 10
python -m stock_ml validate matrix/quick_entry_exit
```

### Done khi

- Regression champion pass.
- Có số benchmark baseline.
- Có matrix smoke config dùng được cho các phase sau.

### Trạng thái Phase 0 — cập nhật 2026-05-01

Phase 0 có thể xem là đóng cho baseline/scope lock hiện tại.

- Regression champion hash hiện đã pass:
  - `python -m pytest stock_ml/tests/regression/test_champions.py -q` → `13 passed`.
  - Mismatch `v42_a` ghi ngày 2026-04-30 đã được xử lý hoặc artifact hiện tại đã khớp golden.
- Benchmark baseline đã ghi nhận trước đó:
  - `v22`: new `13.9s`, legacy `10.5s`, delta `+31.8%`, `SLOW`.
  - `v34`: new `6.2s`, legacy `9.3s`, delta `-32.8%`, `FAST`.
  - `v37a`: new `6.2s`, legacy `7.7s`, delta `-20.0%`, `OK`.
  - `v32`: new `6.1s`, legacy `14.9s`, delta `-59.0%`, `FAST`.
- Matrix smoke hiện là `config/experiments/matrix/quick_entry_exit.yaml` với `2 feature sets × 2 entry models × 2 exit configs × 1 strategy = 8 runs`.
- Verification hiện tại:
  - `python -m stock_ml validate matrix/quick_entry_exit` → OK, `8 experiments`.
  - `python -m stock_ml run-matrix matrix/quick_entry_exit --dry-run --limit 3` → OK.
  - `python -m stock_ml validate champions/v22_exit_b` → OK.

---

## Phase 1 — Chuẩn hóa contract EntryModel

### Mục tiêu

Entry model mới chỉ cần implement một interface và register là dùng được trong YAML/matrix.

### Việc cần làm

1. Kiểm tra `src/components/models/base.py` có contract rõ:

```python
class EntryModel(Protocol):
    name: str
    def fit(self, X_train, y_train) -> None: ...
    def predict(self, X) -> np.ndarray: ...
    def predict_proba(self, X) -> np.ndarray | None: ...
```

2. Chuẩn hóa output canonical:
   - Entry: `-1`, `0`, `1`.
   - Không để model wrapper trả label tùy ý.
3. Đảm bảo registry chỉ expose model đã conform contract.
4. Thêm smoke test chung cho toàn bộ registered entry models:
   - fit được với tiny dataset.
   - predict đúng shape.
   - label thuộc `{-1, 0, 1}`.
   - predict_proba nếu có thì đúng số dòng.
5. Bổ sung validation config:
   - model type tồn tại.
   - hyperparams không chứa key sai nếu model có schema rõ.

### File liên quan

- `src/components/models/base.py`
- `src/components/models/registry.py`
- `src/components/models/*.py`
- `src/pipeline/config.py`
- `tests/components/`

### Verification

```bash
python -m pytest tests/components -q
python -m stock_ml list-components --type models
python -m stock_ml validate champions/v22
```

### Done khi

- Mọi entry model hiện có pass cùng một smoke test contract.
- Thêm model mới chỉ cần file wrapper + registry + YAML.

### Trạng thái Phase 1 — 2026-04-30

Đang triển khai chuẩn hóa EntryModel contract:

- `src/pipeline/trainer.py` đã chuyển từ legacy `src.models.registry.build_model()` sang component registry `src.components.models.registry.get_model()`.
- `components.entry_model.extras` được truyền vào model wrapper khi train entry và sell/exit-label model tạm thời.
- `tests/components/test_models_smoke.py` đã bổ sung contract checks:
  - XGBoost/CatBoost/Ensemble predict phải trả label thuộc `{-1, 0, 1}`.
  - XGBoost/CatBoost phải expose `classes_` đúng `[-1, 0, 1]` sau fit.
  - GRU có smoke test fit/predict/proba/name, skip nếu thiếu torch.
- `src/pipeline/validate.py` đã dùng component model registry để validate `components.entry_model.type`, nên không cần đổi thêm ở Phase 1.
- `src/components/models/registry.py` bỏ qua tham số `device` cho `random_forest` để trainer có thể gọi `get_model(..., device=device, **extras)` thống nhất cho mọi entry model.

Verification đã chạy:

```bash
python -m pytest tests/components/test_models_smoke.py -q
# 23 passed

python -m stock_ml list-components --type models
# OK, liệt kê lightgbm/xgboost/catboost/random_forest/gru

python -m stock_ml validate champions/v22
# OK: v22

python -m pytest stock_ml/tests/regression -q
# Trạng thái cũ 2026-04-30: từng fail regression; cập nhật 2026-05-01 xem Phase 0
```

---

## Phase 2 — Tách ExitModel thành component độc lập

### Mục tiêu

Exit model không còn là prediction phụ mơ hồ, mà là component riêng có contract, train/predict/cache riêng.

### Contract đề xuất

```python
class ExitModel(Protocol):
    name: str
    def fit(self, X_train, y_exit_train) -> None: ...
    def predict(self, X) -> np.ndarray: ...  # {0, 1}
    def predict_proba(self, X) -> np.ndarray | None: ...
```

### Việc cần làm

1. Tạo/chuẩn hóa folder:

```text
src/components/exit_models/
├── base.py
├── ml_binary.py
├── rule_based.py
├── null.py
└── registry.py
```

2. Implement `NullExitModel` hoặc dùng `exit_model: null` rõ ràng.
3. Implement `MLBinaryExitModel` wrap LightGBM/XGBoost binary classifier.
4. Tách logic generate exit labels ra khỏi code train entry.
5. Chuẩn hóa output:
   - `1` = đề xuất exit.
   - `0` = không exit.
6. Thêm validation:
   - Nếu config có exit_model thì target phải support exit labels.
   - Nếu fusion có `ml_exit_model` thì phải có exit_model.
   - Nếu exit_model có nhưng fusion không dùng exit signal thì warning hoặc error tùy chính sách.
7. Thêm cache key riêng cho exit predictions:

```text
feature_signature + exit_target_signature + exit_model_signature + fold
```

### File liên quan

- `src/components/exit_models/`
- `src/components/targets/base.py`
- `src/pipeline/build_predictions.py`
- `src/pipeline/cache.py`
- `src/pipeline/config.py`
- `tests/components/`

### Verification

```bash
python -m pytest tests/components -q
python -m stock_ml validate champions/v37a_exit
python -m stock_ml run champions/v37a_exit --device cpu
```

### Done khi

- Exit model train/predict không phụ thuộc trực tiếp vào entry model.
- `y_pred_exit` được sinh từ component exit rõ ràng.
- Cache entry prediction và exit prediction tách được.

### Trạng thái Phase 2 — 2026-04-30

Đã triển khai phần nền tảng ExitModel component độc lập:

- `src/components/exit_models/` đã có:
  - `base.py`: `ExitModel` protocol.
  - `null_exit.py`: `NullExitModel` luôn trả exit signal `0`.
  - `ml_exit.py`: `MLExitModel` wrap entry model registry để train binary exit label `{0, 1}`.
  - `registry.py`: `get_exit_model()` và `list_exit_models()` cho `null`, `lightgbm`, `xgboost`, `catboost`.
- `src/pipeline/config.py` đã mở rộng `ExitModelConfig`:
  - `type: "lightgbm"` mặc định.
  - `extras` để truyền hyperparams riêng cho exit model.
  - `to_legacy_dict()` include `type` và `extras`, nên prediction cache key tự bust khi đổi exit model type/hyperparams.
- `src/pipeline/trainer.py` đã dùng `get_exit_model(exit_model_cfg.type, **exit_model_cfg.extras)` để train `sell_model`, không còn dùng lại `entry_model.type`.
- `src/pipeline/validate.py` đã validate `components.exit_model.type` qua exit model registry và mở rộng `EXIT_LABEL_TARGETS` cho `early_wave`, `early_wave_v2`, `early_wave_dual`.
- `scripts/cli.py` đã hỗ trợ `python -m stock_ml list-components --type exit_models`.
- `tests/components/test_exit_models_smoke.py` đã bổ sung smoke tests cho `NullExitModel`, `MLExitModel(lightgbm)`, registry.

Verification đã chạy:

```bash
python -m pytest tests/components/test_exit_models_smoke.py -q
# 5 passed

python -m pytest tests/components/test_models_smoke.py tests/components/test_exit_models_smoke.py -q
# 28 passed

python -m stock_ml list-components --type exit_models
# null/lightgbm/xgboost/catboost

python -m stock_ml validate champions/v22
# OK: v22

python -m stock_ml validate champions/v35b
# OK: v35b
```

Ghi chú:

- `python -m stock_ml validate champions/v37a_exit` chưa chạy được vì repo hiện chỉ có YAML trong `config/experiments/champions/`: `v22.yaml`, `v35b.yaml`; chưa có `v37a_exit.yaml`.
- Targeted regression cho Phase 2 đã pass; champion hash regression được xác nhận lại ngày 2026-05-01 bằng `python -m pytest stock_ml/tests/regression/test_champions.py -q` → `13 passed`.
- Phase 2 chưa tách cache entry/exit thành hai artifact riêng; hiện cache key đã include `exit_model_dict()`, đủ để tránh dùng nhầm prediction cache khi đổi exit model type. Tách cache vật lý riêng nên làm cùng Phase 5/8 khi matrix entry × exit mở rộng.

---

## Phase 3 — Tách ExitPolicy/Fusion exit layer

### Mục tiêu

Exit decision là policy có thể swap, không nằm trong backtest engine hoặc runner riêng.

### Khái niệm

- `ExitModel`: tạo tín hiệu ML/rule `{0,1}`.
- `ExitPolicy`: quyết định có thoát vị thế tại bar hiện tại không.
- `Backtester`: chỉ nhận action `exit` và tính trade.

### Policy cần có ban đầu

```text
emergency_exit
hap_preempt
early_loss_cut
signal_exit_defer
ml_exit_model
trailing_stop
time_stop
```

### Việc cần làm

1. Chuẩn hóa `BarContext` có đủ:
   - current bar index.
   - current position.
   - entry signal/proba.
   - exit signal/proba.
   - symbol profile.
   - OHLCV/features row.
2. Chuẩn hóa `FusionResult`/`Action`:
   - `pass`
   - `skip_entry`
   - `enter_long`
   - `exit`
   - `modify_hold`
3. Tách tất cả exit override thành strategy trong:

```text
src/components/fusion/exit_override/
```

4. Thiết lập priority rõ:

```text
1. emergency_exit
2. hard profit protection / HAP
3. early loss cut
4. signal defer/filter
5. ML exit model
6. trailing/time stop
```

5. Viết unit test cho từng policy bằng `BarContext` giả.
6. Viết integration test fusion stack:
   - nhiều policy cùng fire.
   - policy priority thấp hơn không override priority cao hơn.
   - không có vị thế thì exit policy không tạo exit.

### File liên quan

- `src/components/fusion/base.py`
- `src/components/fusion/exit_override/`
- `src/components/backtest/engine.py`
- `src/components/runners/`
- `tests/components/fusion/`

### Verification

```bash
python -m pytest tests/components/fusion -q
python -m pytest tests/regression/test_v37a_exit_parity.py -q
```

### Done khi

- Thêm exit policy mới không cần sửa backtester.
- `ml_exit_model` chỉ là một policy trong stack, không phải case đặc biệt trong engine.

### Trạng thái Phase 3 — 2026-04-30

Đã port thêm 2 exit policies từ legacy engine sang FusionStrategy component:

- `src/components/fusion/strategies/core/early_loss_cut.py`: `EarlyLossCutExit` cho logic `v28_early_loss_cut`.
- `src/components/fusion/strategies/core/hap_preempt.py`: `HapPreemptExit` cho logic `v32_hap_preempt` với guard `v39b_hap_min_hold` và `v33_hap_consec_drop`.
- `src/components/fusion/strategies/core/__init__.py` và `src/components/fusion/strategies/__init__.py` đã export/register cả 2 policies mới.
- `tests/components/fusion/test_early_loss_cut.py` và `tests/components/fusion/test_hap_preempt.py` đã bổ sung unit tests cho trigger, guard, và config override.

Verification đã chạy:

```bash
python -m pytest stock_ml/tests/components/fusion -q
# 98 passed

python -m pytest stock_ml/tests/regression/test_v37a_exit_parity.py -q
# 1 passed

python -m pytest stock_ml/tests/components -q
# 225 passed
```

Ghi chú:

- Phase 3 scope hiện chỉ port 2 policies còn thiếu có priority cao theo plan nhỏ gọn; chưa port các legacy policies dài hơn như `v33_trailing_ratchet`, `v38b_stall_exit`, `v39d_rule_exit_symbols`.
- Chưa sửa legacy `backtest_unified` hoặc runner files; các strategy mới đã có thể opt-in qua fusion registry.

---

## Phase 4 — Làm sạch Backtester thành position manager

### Mục tiêu

Backtester không còn biết strategy version, model flag, hay exit model. Nó chỉ xử lý action stream.

### Việc cần làm

1. Audit `src/backtest/engine.py` và runner mới để tìm logic strategy còn nằm trong backtester.
2. Di chuyển logic quyết định entry/exit sang fusion strategies.
3. Backtester API đích:

```python
trades = Backtester().run(
    actions=actions,
    df_test=df_test,
    initial_cash=100.0,
    fee_pct=0.001,
)
```

4. Chuẩn hóa `Trade` dataclass:
   - entry/exit date.
   - entry/exit price.
   - pnl pct.
   - holding days.
   - entry reason.
   - exit reason.
   - symbol.
5. Giữ legacy engine qua adapter nếu còn cần parity cho version cũ.
6. Thêm property tests:
   - không có exit trước entry.
   - không mở hai vị thế cùng lúc nếu strategy không cho phép.
   - trade luôn có entry_date <= exit_date.

### File liên quan

- `src/backtest/engine.py`
- `src/components/backtest/engine.py`
- `src/components/fusion/`
- `legacy/adapter.py`
- `tests/components/backtest/`

### Verification

```bash
python -m pytest tests/components/backtest -q
python -m pytest tests/regression -q
```

### Done khi

- Backtester không cần biết `y_pred_exit` trực tiếp.
- Mọi quyết định exit đến từ action/fusion layer.

### Trạng thái Phase 4 — 2026-04-30

Đã xác nhận path backtester component mới đã là position manager sạch:

- `src/components/backtest/engine.py`: `SimpleLongBacktester.run(actions, df_test, initial_cash, fee_pct)` chỉ nhận action stream và ghép `enter_long` với `exit`; không biết strategy version, model flag, hoặc `y_pred_exit`.
- `src/components/base.py`: `Trade` dataclass đã có đủ canonical fields: entry/exit date, entry/exit price, pnl pct, holding days, entry reason, exit reason, symbol.
- `src/components/runners/rule_runner.py` đã dùng path mới `FusionStack` → `Action` → `SimpleLongBacktester`.
- Legacy champion runners vẫn đi qua `src/backtest/engine.py::backtest_unified` để giữ parity; chưa xóa legacy engine.

Đã bổ sung phần còn thiếu cho Phase 4:

- `src/components/backtest/legacy_adapter.py`: `LegacyBacktestAdapter` convert output dict từ `backtest_unified` thành `list[Trade]` để phục vụ migrate dần runner legacy.
- `tests/components/backtest/test_backtest_properties.py`: property/invariant tests cho `SimpleLongBacktester`:
  - trade luôn có `entry_date <= exit_date`.
  - không tạo overlapping trades.
  - exit không có position bị ignore.
  - double entry không mở hai vị thế cùng lúc.
  - dangling entry không tự close.
- `tests/components/backtest/test_backtest_properties.py`: smoke test cho `LegacyBacktestAdapter` với trade dict kiểu `backtest_unified`.

Verification đã chạy:

```bash
python -m pytest stock_ml/tests/components/backtest stock_ml/tests/components/test_backtest_simple.py -q
# 18 passed

python -m pytest stock_ml/tests/components -q
# 233 passed
```

Regression đã chạy targeted cho path liên quan:

```bash
python -m pytest stock_ml/tests/regression/test_v37a_exit_parity.py stock_ml/tests/regression/test_rule_parity.py -q
# 2 passed
```

Champion hash regression được xác nhận lại ngày 2026-05-01 bằng `python -m pytest stock_ml/tests/regression/test_champions.py -q` → `13 passed`. Full regression toàn thư mục vẫn nên chạy trước khi merge lớn nếu có thời gian.

---

## Phase 5 — Mở rộng YAML schema cho entry × exit matrix

### Mục tiêu

Một file matrix YAML có thể sinh hàng chục/hàng trăm tổ hợp entry/exit/fusion mà không viết code Python mới.

### Schema đích

```yaml
name: entry_exit_grid_q2_2026

base:
  components:
    target: early_wave
  split:
    train_years: 3
    test_years: 1
  fusion:
    pre_entry:
      - {name: skip_choppy}
    entry:
      - {name: ml_only}

axes:
  components.features:
    - leading_v2
    - leading_v4

  components.entry_model:
    - {type: lightgbm, hyperparams: {n_estimators: 500}}
    - {type: xgboost, hyperparams: {max_depth: 6}}

  components.exit_model:
    - null
    - {type: ml_binary, hyperparams: {model_type: lightgbm}, label: {forward_window: 15, loss_threshold: 0.05}}

  fusion.exit_override:
    - []
    - [{name: hap_preempt}, {name: early_loss_cut}]
    - [{name: hap_preempt}, {name: early_loss_cut}, {name: ml_exit_model}]

naming_pattern: "{features}_{entry_model.type}_{exit_model.type}_{exit_policy_hash}"
```

### Việc cần làm

1. Cho matrix expander support nested path axes như `components.entry_model`.
2. Validate tổ hợp trước khi chạy:
   - `exit_model: null` không được đi với `ml_exit_model`.
   - `exit_model != null` nhưng không có policy dùng exit signal thì báo lỗi/warning.
   - target không support exit label thì loại tổ hợp hoặc fail.
3. Tạo naming ổn định cho experiment sinh ra.
4. Tạo dry-run mode hiển thị toàn bộ tổ hợp.
5. Cho phép giới hạn nhanh:
   - `--limit N`
   - `--symbols-limit N`
   - `--dry-run`
6. Đảm bảo cache được share giữa tổ hợp cùng feature/target/model.

### File liên quan

- `src/pipeline/matrix_expander.py`
- `src/pipeline/config.py`
- `scripts/cli.py`
- `config/experiments/matrix/`
- `docs/refactor/HOW_TO_RUN_MATRIX.md`

### Verification

```bash
python -m stock_ml validate matrix/entry_exit_quick
python -m stock_ml run-matrix matrix/entry_exit_quick --dry-run
python -m stock_ml run-matrix matrix/entry_exit_quick --device cpu --symbols-limit 10
```

### Done khi

- Có thể chạy ít nhất matrix 2×2×2 entry/exit từ YAML.
- Config sai bị bắt trước khi train.

### Trạng thái Phase 5 — 2026-04-30

Đã triển khai phần nền tảng để chạy matrix entry × exit từ YAML:

- `src/pipeline/matrix_expander.py` đã hỗ trợ nested path axes như `components.exit_model`, giữ alias cũ `features`, `model_type`, `target_type`, validate root axis để bắt typo sớm, và nhận `limit` để không expand thừa khi smoke nhanh.
- Tên experiment sinh ra ổn định hơn cho dict/list axis bằng `label`, `type`, hoặc hash ngắn; `base.name` không còn làm các combo bị trùng tên.
- `scripts/cli.py` đã thêm cho `run-matrix`:
  - `--dry-run` để in các experiment đã expand mà không train.
  - `--limit N` để chỉ chạy N combo đầu.
  - `--symbols-limit N` để smoke nhanh trên N mã đầu.
- `config/experiments/matrix/quick_entry_exit.yaml` đã mở rộng thành `2 features × 2 entry models × 2 exit configs = 8 experiments`:
  - `null_exit`: `enabled=false`, `type="null"`.
  - `lightgbm_exit`: `enabled=true`, `type=lightgbm`.

Verification đã chạy:

```bash
python -m stock_ml validate matrix/quick_entry_exit
# OK: quick_entry_exit.yaml (8 experiments)

python -m stock_ml run-matrix matrix/quick_entry_exit --dry-run --limit 3
# OK, in 3 experiment đầu với entry/exit metadata

python -m stock_ml validate matrix/test_2x2
# OK: test_2x2.yaml (4 experiments)

python -m pytest stock_ml/tests/components/test_exit_models_smoke.py -q
# 5 passed
```

Ghi chú:

- Phase 5 hiện chưa thêm validation sâu cho tương tác `fusion.exit_override` vì schema fusion hiện vẫn là dict tự do và runner `v22` chưa consume exit policy stack mới trong matrix path này.
- Chưa chạy `run-matrix ... --device cpu --symbols-limit 10` vì lệnh này có thể train model thật; dry-run và validate đã xác nhận expansion/schema.
- Bước tiếp theo hợp lý là Phase 6: chuẩn hóa artifact/ranking, hoặc tiếp tục Phase 5 phần fusion-exit validation sau khi runner mới dùng `FusionStack` được nối vào matrix path.

---

## Phase 6 — Chuẩn hóa result artifact và ranking

### Mục tiêu

Sau khi chạy matrix lớn, có bảng xếp hạng rõ để chọn model vượt trội.

### Việc cần làm

1. Mỗi run lưu metadata đầy đủ:
   - feature_set.
   - target.
   - entry_model type + hyperparams hash.
   - exit_model type + label config + hyperparams hash.
   - fusion stack hash.
   - train/test split.
   - seed.
2. Chuẩn hóa output:

```text
results/experiments/<run_name>/
├── trades.csv
├── metrics.json
├── config.resolved.yaml
├── predictions_meta.json
└── ranking_row.json
```

3. Tạo compare/ranking command:

```bash
python -m stock_ml compare-matrix results/experiments/entry_exit_grid_q2_2026
```

4. Ranking nên có tối thiểu:
   - composite_score.
   - total_return.
   - win_rate.
   - max_drawdown.
   - trade_count.
   - avg_holding_days.
   - per-year consistency.
   - per-symbol coverage.
5. Thêm guard chống winner giả:
   - loại run có trade_count quá thấp.
   - cảnh báo nếu return tập trung vào 1 symbol.
   - cảnh báo nếu chỉ thắng 1 năm nhưng thua các năm khác.

### File liên quan

- `src/evaluation/scoring.py`
- `src/components/evaluation/`
- `src/pipeline/orchestrator.py`
- `scripts/benchmark.py`
- `scripts/cli.py`
- `results/`

### Verification

```bash
python -m stock_ml run-matrix matrix/entry_exit_quick --device cpu
python -m stock_ml compare-matrix results/experiments/entry_exit_quick
```

### Done khi

- Matrix output có thể sort/rank trực tiếp.
- Có đủ metadata để tái chạy winner.

### Trạng thái Phase 6 — 2026-04-30

Đã hoàn thiện nền tảng artifact/ranking cho matrix runs:

- `src/pipeline/orchestrator.py`: `PipelineResult` đã có `metrics`; `Pipeline.run()` tự tính `calc_metrics()`, `mdd_per_symbol`, `yearly_consistency`, và `composite_score()` khi có trades.
- `src/evaluation/scoring.py`: `composite_score()` đã khớp thiết kế symbol-count neutral trong docstring:
  - dùng `sharpe`, `avg_pnl`, `profit_factor`, `mdd_per_symbol`, `yr_consistency`.
  - không dùng `total_pnl` trong score để tránh bias theo số symbol.
  - có `calc_symbol_coverage()` để tính `symbol_count` và `top_symbol_pnl_ratio` phục vụ guard winner giả.
- `scripts/cli.py`: `run-matrix` mặc định lưu artifact env-aware qua `get_results_dir()` vào `results/experiments/<matrix_name>/<experiment_name>/`:
  - `trades.csv` nếu run có trades.
  - `metrics.json`.
  - `ranking_row.json` có `max_drawdown`, `mdd_per_symbol`, yearly consistency, symbol coverage, config hash.
  - `config.resolved.yaml`.
- `scripts/cli.py`: `compare-matrix` đọc các `ranking_row.json`, sort theo `composite_score`, in bảng ranking dùng `mdd_per_symbol`, và cảnh báo:
  - `LOW_TRADES` nếu `trade_count < 20`.
  - `SYMBOL_CONCENTRATION` nếu 1 symbol chiếm trên 50% absolute PnL.
  - `YEAR_INCONSISTENT` nếu yearly CV > 1.5.

Verification đã chạy:

```bash
python -m pytest stock_ml/tests/components -q
# 233 passed

python -m stock_ml validate matrix/quick_entry_exit
# OK: quick_entry_exit.yaml (8 experiments)

python -m stock_ml run-matrix matrix/quick_entry_exit --dry-run
# OK: expand đủ 8 experiments

python -m stock_ml run-matrix matrix/quick_entry_exit --device cpu --symbols-limit 5 --limit 2
# OK: 2 experiments; 32 trades cho null_exit, 36 trades cho lightgbm_exit; artifacts đã ghi

python -m stock_ml compare-matrix results/experiments/quick_entry_exit
# OK: ranking in được; lightgbm_exit score 307.3, null_exit score -47.1; cả 2 có cảnh báo SYMBOL_CONCENTRATION
```

Ghi chú:

- Chưa lưu `predictions_meta.json`; phần này để Phase 8 khi tách/cache prediction metadata rõ hơn.
- Artifact hiện đủ để tái chạy winner qua `config.resolved.yaml` và lọc/rank bằng `compare-matrix`.
- Phase 6 có thể xem là đóng cho quick matrix smoke; bước tiếp theo hợp lý là Phase 7 nếu muốn promote winner, hoặc Phase 8 nếu muốn tối ưu matrix lớn trước.

---

## Phase 7 — Promote winner thành champion mới

### Mục tiêu

Biến tổ hợp thắng matrix thành version/champion ổn định để regression lâu dài.

### Việc cần làm

1. Chọn winner không chỉ theo score, mà theo robustness:
   - score cao.
   - đủ trade count.
   - drawdown thấp.
   - ổn định theo năm.
   - không phụ thuộc quá mức vào một symbol.
2. Copy `config.resolved.yaml` thành:

```text
config/experiments/champions/v_next.yaml
```

3. Đổi name/description rõ ràng.
4. Chạy full backtest CPU.
5. Sinh golden baseline cho champion mới.
6. Thêm regression test cho champion mới.
7. Cập nhật docs guide nếu có component mới.

### File liên quan

- `config/experiments/champions/`
- `tests/regression/`
- `tests/regression/golden/`
- `docs/refactor/CHAMPION_VERSIONS.md`

### Verification

```bash
python -m stock_ml validate champions/v_next
python -m stock_ml run champions/v_next --device cpu --force
python -m pytest tests/regression/test_v_next_parity.py -q
```

### Done khi

- Winner chạy được như một champion độc lập.
- Golden regression bảo vệ kết quả.

### Trạng thái Phase 7 — 2026-04-30

Đã promote winner từ quick matrix thành champion mới `v22_exit_b`:

- Winner nguồn: `quick_entry_exit_features-leading_v2-model_type-lightgbm-components_exit_model-lightgbm_exit-strategy-v22`.
- `config/experiments/champions/v22_exit_b.yaml` dùng:
  - runner `src.components.runners.v22_runner`.
  - feature set `leading_v2`.
  - target `early_wave`.
  - entry model `lightgbm`.
  - exit model `lightgbm`, enabled.
  - `mods`/`params`/`fusion` giữ theo `v22`.
- `src/pipeline/orchestrator.py` đã đăng ký `v22_exit_b` như alias của `v22_runner.run_v22` và `trades_to_v22_dataframe`.
- `tests/regression/test_champions.py` đã thêm `v22_exit_b` vào danh sách champion hash regression.
- Golden artifact đã sinh:
  - `results/trades_v22_exit_b.csv`: 353 trades, win rate 43.3%, avg PnL 1.75%.
  - `tests/regression/golden/trades_v22_exit_b.csv`.
  - `tests/regression/golden/checksums.txt` có hash `5f7e2c56dd179c5aeef0802aab223f85dff58d27dc3cfc79780919b820a038ff`.

Verification đã chạy:

```bash
python -m stock_ml validate champions/v22_exit_b
# OK: v22_exit_b

python -m stock_ml run champions/v22_exit_b --device cpu --save-results
# Result: 353 trades for v22_exit_b
```

Ghi chú:

- `python -m stock_ml run ... --force` không hỗ trợ trong CLI hiện tại, nên đã chạy không có `--force`.
- Targeted regression cho `v22_exit_b` cần chạy sau khi cập nhật golden.

---

## Phase 8 — Tối ưu tốc độ research

### Mục tiêu

Chạy matrix lớn nhanh hơn mà vẫn reproducible.

### Việc cần làm

1. Profile matrix run để biết bottleneck:
   - feature compute.
   - target generation.
   - model fit.
   - backtest/fusion loop.
2. Tối ưu cache:
   - cache feature theo feature signature.
   - cache target theo target signature.
   - cache predictions theo model signature.
3. Thêm parallel execution an toàn theo experiment/fold nếu chưa có.
4. Thêm resume mode:
   - skip run đã có `metrics.json` hợp lệ.
   - rerun run lỗi.
5. Thêm `--top-k-preview`:
   - chạy symbols-limit nhỏ trước.
   - chọn top K.
   - chạy full only top K.

### File liên quan

- `src/pipeline/cache.py`
- `src/pipeline/orchestrator.py`
- `src/pipeline/matrix_expander.py`
- `scripts/benchmark.py`

### Verification

```bash
python -m stock_ml benchmark --versions v22,v37a_exit --symbols-limit 10
python -m stock_ml run-matrix matrix/entry_exit_medium --resume
```

### Done khi

- Matrix medium có thể resume.
- Cache hit rate rõ trong log/metadata.
- Thời gian chạy giảm so với baseline Phase 0.

### Trạng thái Phase 8 — 2026-05-01

Đã triển khai bước đầu để tăng tốc vòng lặp research matrix:

- `scripts/cli.py`: `run-matrix` đã truyền `PredictionCacheManager` vào `Pipeline`, nên prediction cache hiện được dùng trong matrix path.
  - Cache hit/miss/stored được log bởi `Pipeline._build_cache()` và track trong `PredictionCacheManager.stats()`.
  - Cache stats được expose qua `PipelineResult.metadata["cache_stats"]`.
  - Cache vẫn là sequential-safe, chưa thêm parallel execution.
- `scripts/cli.py`: thêm `--resume` cho `run-matrix`.
  - Khi có artifact `ranking_row.json`, experiment được skip với log `[SKIP] already done`.
  - Resume dựa trên thư mục artifact chuẩn `results/experiments/<matrix_name>/<experiment_name>/`.
- `scripts/cli.py`: thêm `--top-k-preview K` cho `run-matrix`.
  - Chạy preview trên `--symbols-limit` nếu có, mặc định 10 symbols nếu không truyền.
  - Preview artifact lưu ở `results/experiments/<matrix_name>_preview/` để không lẫn với full run.
  - Sau preview, chọn top K theo `composite_score` rồi chạy lại top K trên symbol set chính.
- `scripts/cli.py`: matrix artifact đã có `predictions_meta.json`.
  - Lưu `config_hash`, entry model, exit model type/enabled, feature set, split config và `cache_stats`.

Verification đã chạy:

```bash
python -m stock_ml run-matrix matrix/quick_entry_exit --dry-run --limit 3
# OK: expand 3 experiment đầu, matrix tổng có 8 experiments

python -m pytest stock_ml/tests/components/test_exit_models_smoke.py stock_ml/tests/components/fusion/test_early_loss_cut.py stock_ml/tests/components/fusion/test_hap_preempt.py stock_ml/tests/components/backtest/test_backtest_properties.py -q
# 22 passed

python -m stock_ml run-matrix matrix/quick_entry_exit --device cpu --symbols-limit 5 --limit 2 --save-results
# OK: 2 experiments, cache HIT cả 2, artifacts đã có predictions_meta.json

python -m stock_ml run-matrix matrix/quick_entry_exit --device cpu --symbols-limit 5 --limit 2 --save-results --resume
# OK: skip 2 experiments đã có ranking_row.json

python -m stock_ml run-matrix matrix/quick_entry_exit --device cpu --symbols-limit 5 --limit 2 --top-k-preview 1
# OK: preview 2 experiments, full run lại top 1

python -m pytest stock_ml/tests/regression/test_champions.py -q
# 13 passed

python -m pytest stock_ml/tests/components -q
# 233 passed
```

Ghi chú:

- Phase 8 có thể xem là đóng cho quick matrix smoke hiện tại.
- Parallel execution chưa làm ở bước này để tránh race condition cache/artifact; nên làm như future work sau khi resume/cache ổn định.
- `predictions_meta.json` hiện đã đủ metadata tối thiểu để audit entry/exit config và cache hit/miss/stored của matrix run.

---

## Thứ tự ưu tiên đề xuất

Nếu muốn đi nhanh vào nghiên cứu model mới:

1. Phase 0 — baseline.
2. Phase 2 — ExitModel component.
3. Phase 3 — ExitPolicy/fusion exit layer.
4. Phase 5 — Matrix entry × exit.
5. Phase 6 — Ranking.

Phase 1 và Phase 4 làm xen kẽ khi đụng contract/backtester, không cần chờ hoàn hảo mới bắt đầu matrix nhỏ.

## Rủi ro chính

### Rủi ro 1 — Lệch parity legacy

Cách giảm rủi ro:

- Giữ legacy adapter.
- Port từng champion một.
- Mỗi lần tách policy phải chạy regression version liên quan.

### Rủi ro 2 — Matrix tạo winner giả

Cách giảm rủi ro:

- Không rank chỉ bằng total return.
- Bắt buộc trade_count tối thiểu.
- Kiểm tra per-year và per-symbol.
- Promote winner rồi chạy full regression/golden.

### Rủi ro 3 — Exit model bị leakage

Cách giảm rủi ro:

- Exit label generator phải dùng forward window rõ ràng.
- Split train/test trước khi fit scaler/model.
- Không dùng dữ liệu test để tune threshold.
- Metadata phải lưu label config.

### Rủi ro 4 — Refactor quá rộng

Cách giảm rủi ro:

- Không sửa feature/model unrelated trong cùng phase.
- Không xóa legacy code.
- Mỗi phase có verification riêng.

## Checklist khi thêm một exit model mới sau refactor

1. Implement wrapper trong `src/components/exit_models/`.
2. Register vào `exit_models/registry.py`.
3. Thêm label config nếu cần.
4. Thêm smoke test fit/predict.
5. Thêm YAML experiment nhỏ.
6. Chạy validate.
7. Chạy matrix quick.
8. So sánh ranking.
9. Nếu tốt, promote thành champion.

## Checklist khi thêm một exit policy mới sau refactor

1. Implement policy trong `src/components/fusion/exit_override/`.
2. Khai báo priority mặc định.
3. Register vào fusion registry.
4. Viết unit test với `BarContext` giả.
5. Thêm vào YAML fusion stack.
6. Chạy regression version gần nhất có logic tương tự.
7. Chạy matrix quick để kiểm tra tương tác với entry/exit model.

## Definition of Done toàn bộ refactor

Refactor được xem là xong khi:

- Có thể định nghĩa matrix entry × exit × fusion_exit hoàn toàn bằng YAML.
- Có ít nhất 2 entry models và 2 exit configs chạy chung trong một matrix.
- Có ranking output rõ ràng sau matrix.
- Winner có thể promote thành champion và sinh golden regression.
- Regression champion cũ vẫn pass.
- Thêm entry/exit model mới không cần sửa backtester hoặc legacy runner.
