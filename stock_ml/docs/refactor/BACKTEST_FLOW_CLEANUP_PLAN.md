# Backtest Flow Cleanup Plan

**Status:** Draft 2026-05-03  
**Scope:** Hoàn thiện flow chạy backtest matrix/strategy combinations, ranking artifact, validation và hiệu năng để phục vụ model selection quy mô lớn.  
**Goal:** Biến pipeline hiện tại thành flow sạch, dễ audit, dễ resume, ít chạy thừa và đủ an toàn để xếp hạng champion/top specialist.

## 1. Bối cảnh

Dự án hiện đã có các mảnh chính để chạy model selection:

- `run-matrix` expand YAML matrix thành nhiều `ExperimentConfig`.
- `Pipeline` chạy từng experiment và dùng `PredictionCacheManager` để cache prediction.
- Artifact mỗi run có `metrics.json`, `ranking_row.json`, `predictions_meta.json`, `config.resolved.yaml`, `trades.csv`.
- `compare-matrix` đọc artifact đã lưu và in ranking.
- `compare-champions` chạy nhiều champion trên cùng split để so sánh công bằng.

Flow hiện tại chạy được, nhưng chưa đủ sạch cho sweep lớn vì còn lẫn `components`/`signals`, ranking helper phụ thuộc tên bundle thay vì path, validation research còn nhẹ, chưa có metadata phân biệt preview/full và chưa có parallel execution.

## 2. Nguyên tắc chỉnh sửa

1. `signals` là canonical runtime config; `components` chỉ dùng để backward compatibility khi load YAML cũ.
2. Backtest/ranking phải reproducible bằng artifact, không phụ thuộc nhớ tay.
3. Không thêm flag mới vào legacy backtester nếu có thể biểu diễn bằng strategy/rule/config.
4. Mọi batch dài phải resume được bằng artifact hiện có.
5. Không tối ưu song song trước khi cache key, artifact path và validation rõ ràng.
6. Preview ranking không được bị nhầm với full ranking.
7. Rule baseline phải luôn nằm trong benchmark cuối để biết ML có tạo giá trị thật không.

## 3. Vấn đề cần xử lý

### 3.1. Ranking/bundle path chưa clean

Hiện `_write_matrix_ranking(matrix_name)` tự resolve path theo `results/experiments/<matrix_name>`. `compare-champions` lại có `bundle_dir` riêng rồi gọi `_write_matrix_ranking(bundle_name)`. Cách này hoạt động với bundle name đơn giản, nhưng dễ sai khi dùng custom bundle/path.

**Cần làm:**

- Đổi helper ranking nhận trực tiếp `bundle_dir: Path`.
- Tách rõ:
  - `_write_ranking(bundle_dir: Path)`
  - `_load_ranking_rows(bundle_dir: Path)`
  - `_print_ranking_table(rows, top)`
- `run-matrix` truyền `Path(get_results_dir()) / "experiments" / yaml_path.stem`.
- `compare-champions` truyền đúng `bundle_dir` đã tính.

**Done khi:**

- `run-matrix` và `compare-champions` cùng dùng một ranking helper.
- Custom `--bundle-name` vẫn ghi `ranking.csv/json` đúng thư mục.
- `compare-matrix` đọc lại được ranking từ thư mục đó.

### 3.2. Lẫn `components` và `signals`

Config loader đã normalize `signals` và `components`, nhưng CLI/validate vẫn dùng lẫn `cfg.components.*` và `cfg.signals.*`. Điều này làm flow khó đọc và dễ bug khi thêm schema mới.

**Cần làm:**

- Trong CLI/runtime/validate, chuyển các access runtime sang `cfg.signals.*`:
  - entry model
  - exit model
  - target
  - features
- Giữ `components` trong `ExperimentConfig` chỉ để load YAML cũ và export `config.resolved.yaml` nếu còn cần parity.
- Sửa text lỗi validation từ `components.*` sang `signals.*`.

**Done khi:**

- Search runtime code không còn print/check `cfg.components.exit_model` ở CLI/orchestrator/validate.
- `validate matrix/quick_entry_exit` pass.
- Champion regression hoặc smoke run không đổi behavior.

### 3.3. Validation chưa đủ cho research sweep

Hiện validation giữ nhẹ để bảo toàn golden parity. Với model selection, cần phát hiện sớm tổ hợp vô nghĩa như bật exit model nhưng strategy không dùng exit signal, hoặc target không hỗ trợ exit label.

**Cần làm:**

- Thêm chế độ strict validation cho research:
  - CLI `validate --strict`.
  - `run-matrix --strict-validate` hoặc mặc định strict cho matrix mới nếu không phá workflow cũ.
- Rule strict đề xuất:
  - exit model enabled thì target phải thuộc nhóm hỗ trợ exit labels.
  - exit model enabled thì `strategy_v3.exit_rules`/`active_exit_rules` có rule consume exit signal, hoặc strategy được whitelist là legacy-compatible.
  - exit model disabled nhưng strategy yêu cầu exit model thì warning/error.
  - `split.first_test_year < split.last_test_year` vẫn là error.
  - entry model không build cache như `rule` thì bỏ qua model registry check.

**Done khi:**

- Config sai bị bắt ở validate, không đợi đến runtime.
- Có warning/error rõ ràng theo field canonical `signals.*` hoặc `strategy_v3.*`.
- Golden/parity flow cũ không bị phá nếu không bật strict.

### 3.4. Preview/full ranking metadata chưa rõ

`--top-k-preview` chạy matrix trên subset symbol rồi full-run top K. Nếu artifact không ghi rõ scope, preview ranking dễ bị hiểu nhầm là full ranking.

**Cần làm:**

- Ghi thêm metadata vào `predictions_meta.json` hoặc `run_meta.json`:
  - `run_scope`: `preview` hoặc `full`.
  - `symbols_count`.
  - `symbols_limit` nếu có.
  - `matrix_name`.
  - `source_matrix_yaml`.
  - `device`.
  - `created_at`.
- Với preview bundle, tên thư mục vẫn dùng `<matrix>_preview` nhưng metadata phải ghi rõ.
- Ranking table nên in cảnh báo nếu bundle là preview.

**Done khi:**

- Mở artifact bất kỳ biết ngay là preview hay full.
- `compare-matrix` có thể cảnh báo khi đọc preview bundle.

### 3.5. Matrix execution còn tuần tự

`_run_matrix_configs` chạy tuần tự từng config. Với sweep lớn, đây là điểm lãng phí hiệu năng lớn nhất sau khi cache đã ổn.

**Cần làm theo thứ tự:**

1. Đo baseline trước khi parallel:
   - thời gian build cache
   - thời gian backtest
   - cache hit/miss
   - số experiment/giờ
2. Tách giai đoạn build prediction cache và backtest nếu khả thi.
3. Thêm `--jobs N` cho phần có thể chạy song song an toàn.
4. Với GPU training:
   - mặc định serial hoặc `--jobs 1`.
   - chỉ parallel CPU backtest sau khi prediction cache đã có.
5. Đảm bảo artifact write không đụng nhau:
   - mỗi experiment ghi vào thư mục riêng.
   - ranking tổng hợp chỉ ghi sau khi worker xong.

**Done khi:**

- `run-matrix --jobs 1` giữ behavior cũ.
- `run-matrix --jobs N` nhanh hơn trên matrix CPU/backtest-heavy.
- Cache không corrupt khi nhiều worker đọc.
- Ranking tổng hợp đủ rows, không mất artifact.

### 3.6. Ranking table chưa đủ để chọn specialist

Hiện ranking sort theo `composite_score`. Để chọn model có điểm mạnh riêng, cần thêm view theo risk/consistency/coverage.

**Cần làm:**

- Thêm các mode cho `compare-matrix`:
  - `--sort composite_score` mặc định.
  - `--sort mdd_per_symbol` cho defensive.
  - `--sort yearly_consistency` hoặc consistency metric phù hợp.
  - `--sort total_pnl`.
  - `--min-trades`.
  - `--max-mdd`.
- Thêm guard warnings hiện có thành cột hoặc summary rõ hơn:
  - `LOW_TRADES`
  - `SYMBOL_CONCENTRATION`
  - `YEAR_INCONSISTENT`
  - `HIGH_DRAWDOWN`
- Nếu có rule baseline trong bundle, in delta so với rule:
  - score delta
  - PnL delta
  - MDD delta
  - win-rate delta

**Done khi:**

- Có thể tìm top overall, defensive champion và specialist bằng CLI, không cần mở pandas thủ công.
- Ranking output không chỉ phục vụ top composite.

### 3.7. Runner registry còn phân tán

`Pipeline._resolve_runner` đang tra nhiều nguồn: fusion runner defs, lineage runner defs, champion runner map. Flow chạy được nhưng khó hiểu khi thêm strategy mới.

**Cần làm:**

- Chưa cần refactor lớn ngay.
- Trước mắt thêm helper nội bộ để list strategy runner source:
  - strategy name
  - runner source: fusion/lineage/champion
  - converter
- `list-components` hoặc command mới có thể hiển thị strategy registry.
- Sau khi ổn định, cân nhắc gom registry runner về một module duy nhất.

**Done khi:**

- Người thêm strategy mới biết đăng ký ở đâu.
- Validation và CLI dùng chung danh sách strategy khả dụng.

## 4. Thứ tự triển khai đề xuất

### Phase A — Sửa correctness và clarity trước — DONE 2026-05-03

1. Đổi ranking helper nhận `bundle_dir: Path`. ✅
2. Chuẩn hóa CLI/validate/orchestrator sang `cfg.signals.*`. ✅
3. Thêm metadata preview/full vào `predictions_meta.json`. ✅
4. `compare-matrix` cảnh báo khi đọc preview bundle. ✅
5. Chạy verify: ✅

**Ghi chú:** Metadata mới chỉ được ghi khi artifact được tạo/ghi lại. Artifact cũ bị skip bởi `--resume` sẽ chưa có các field mới cho đến khi rerun không skip. Sau pass bổ sung 2026-05-03, trainer/cache cũng đọc runtime qua `cfg.signals.*`; `components` chỉ còn trong normalization/backward compatibility.

```bash
python -m stock_ml validate matrix/quick_entry_exit
python -m stock_ml run-matrix matrix/quick_entry_exit --symbols-limit 5 --device cpu --resume
python -m stock_ml compare-matrix results/experiments/quick_entry_exit --top 10
python -m stock_ml compare-champions --first-test-year 2023 --last-test-year 2024 --symbols-limit 5 --device cpu --resume
```

### Phase B — Strict validation cho research — DONE 2026-05-03

1. Thêm strict mode vào `validate_config`. ✅
2. Wire CLI `validate --strict`. ✅
3. Wire optional `run-matrix --strict-validate`. ✅
4. Chạy validate trên các matrix hiện có. ✅
5. Sửa matrix nào đang tạo tổ hợp vô nghĩa hoặc ghi rõ whitelist legacy. ✅ Không cần sửa `quick_entry_exit` vì target đã là `early_wave`.

### Phase C — Ranking phục vụ chọn champion/specialist — DONE 2026-05-03

1. Thêm sort/filter options cho `compare-matrix`. ✅
2. Thêm baseline delta nếu bundle có `rule`. ✅
3. Thêm cột flags guardrail trực tiếp trong ranking table. ✅
4. Chạy lại trên artifact hiện có để kiểm tra không cần rerun backtest. ✅

**Ghi chú:** `compare-matrix` hiện hỗ trợ `--sort composite_score|mdd_per_symbol|yearly_consistency|total_pnl`, `--min-trades`, `--max-mdd`; nếu bundle có baseline `rule` thì in delta score/PnL/MDD/win-rate so với rule.

### Phase D — Hiệu năng batch lớn — DONE 2026-05-03

1. Ghi cache stats rõ hơn vào metadata và in summary cuối `run-matrix`. ✅
2. Ghi `elapsed_seconds` vào `predictions_meta.json` để đo baseline runtime từng experiment. ✅
3. Thêm `run-matrix --jobs N` cho parallel CPU execution. ✅
4. Parallel hóa bằng `ProcessPoolExecutor`, giữ GPU/non-CPU chạy serial để tránh phá cache/GPU workflow. ✅
5. Verify resume và cache hit trên `quick_entry_exit`. ✅

**Ghi chú:** `--jobs 1` giữ behavior serial cũ. `--jobs N` chỉ parallel trên `--device cpu`; mỗi worker có `PredictionCacheManager` riêng, artifact vẫn ghi theo thư mục experiment và ranking tổng hợp chỉ ghi sau khi worker xong. Bundle `quick_entry_exit` hiện có artifact legacy còn sót nên ranking có 10 rows cho 8 config; cần rerun vào bundle sạch hoặc dọn artifact cũ có xác nhận trước khi dùng row count làm guard.

### Phase E — Runner registry cleanup — DONE 2026-05-03

1. Tạo view/list strategy registry thống nhất. ✅
2. Dùng registry đó trong validation. ✅
3. Chỉ refactor `_resolve_runner` nếu duplication gây lỗi thực tế. ✅ Không refactor lớn; chỉ thêm docstring nguồn resolution.

**Ghi chú:** `list-components --type runners` hiện in strategy runners cùng source `champion|fusion|lineage`. `validate_config` dùng chung `list_runners()` để tránh reconstruct runner set phân tán.

## 5. File dự kiến chỉnh sửa

- `stock_ml/scripts/cli.py`
  - ranking helper
  - compare-matrix options
  - compare-champions path
  - preview/full metadata
  - optional jobs/strict validate

- `stock_ml/src/pipeline/config.py`
  - giữ canonical normalization
  - nếu cần thêm helper accessors cho `signals`

- `stock_ml/src/pipeline/validate.py`
  - strict validation
  - field names canonical
  - strategy/exit compatibility rules

- `stock_ml/src/pipeline/orchestrator.py`
  - giảm dùng `components`
  - có thể expose/list runner registry source

- `stock_ml/src/pipeline/matrix_expander.py`
  - giữ alias hiện tại
  - chỉ sửa nếu cần metadata/source matrix name từ expand context

- `stock_ml/config/experiments/matrix/*.yaml`
  - sửa matrix tạo tổ hợp không hợp lệ nếu strict mode phát hiện

## 6. Rủi ro và cách kiểm soát

| Rủi ro | Cách kiểm soát |
|---|---|
| Sửa canonical `signals` làm lệch golden | Chạy validate/smoke/champion regression sau từng phase |
| Strict validation phá config legacy | Mặc định non-strict, chỉ bật strict cho research sweep |
| Parallel làm corrupt cache/artifact | Ranking chỉ ghi cuối, mỗi experiment thư mục riêng, bắt đầu với read-only cache hoặc jobs=1 default |
| Preview ranking bị chọn nhầm làm champion | Metadata `run_scope`, cảnh báo trong compare output |
| Refactor runner registry quá rộng | Chỉ thêm list/view trước, chưa gom registry nếu chưa cần |

## 7. Verification checklist

Sau mỗi phase tối thiểu chạy:

```bash
python -m stock_ml validate champions/rule
python -m stock_ml validate matrix/quick_entry_exit
python -m stock_ml run-matrix matrix/quick_entry_exit --dry-run
python -m stock_ml run-matrix matrix/quick_entry_exit --symbols-limit 5 --device cpu --resume
python -m stock_ml compare-matrix results/experiments/quick_entry_exit --top 10
```

Nếu chạm runner/config normalization, chạy thêm champion subset:

```bash
python -m stock_ml compare-champions --first-test-year 2023 --last-test-year 2024 --champions rule,v22,v34 --symbols-limit 5 --device cpu --resume
```

Nếu chạm cache/parallel, kiểm tra:

- run đầu có cache miss/store hợp lý.
- run lại với `--resume` skip đúng.
- run lại không `--resume` có cache hit.
- ranking row count bằng số experiment đã chạy.

## 8. Definition of Done

Flow được xem là clean hơn khi đạt đủ:

- ✅ `signals` là nguồn đọc runtime duy nhất trong CLI/validate/orchestrator/trainer.
- ✅ `run-matrix`, `compare-matrix`, `compare-champions` dùng chung artifact/ranking helper.
- ✅ Artifact mới phân biệt preview/full rõ ràng trong `predictions_meta.json`.
- ✅ Strict validation bắt được tổ hợp strategy/exit/target vô nghĩa.
- ✅ Ranking CLI hỗ trợ chọn overall champion, defensive champion và specialist.
- ✅ Batch run có resume ổn định và cache stats rõ ràng; ranking row-count guard cần bundle sạch vì `quick_entry_exit` còn artifact legacy.
- ✅ Có đường mở rộng sang parallel execution mà không phá behavior serial hiện tại.
- ✅ Có view strategy runner registry thống nhất cho CLI và validation.
