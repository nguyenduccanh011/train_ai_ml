# Clean Architecture Next Steps

**Status:** Draft 2026-05-02  
**Scope:** Dọn sạch nền V3 sau refactor lớn, trước khi thêm thuật toán/feature mới.  
**Goal:** Repo sạch, config dễ hiểu, matrix comparison tiện, legacy backtest được cô lập rõ.

## Tóm tắt hiện trạng

V3 đã đi đúng hướng: trading system được tách thành Signal / Strategy / Execution, runner đã generic hơn, entry model và exit model có registry riêng, matrix experiment đã chạy được nhiều tổ hợp.

Nhưng dự án chưa thật sự clean vì:

1. Working tree đang rất rộng: code, docs, golden, test rename, legacy deletion, artifact results cùng tồn tại trong một thay đổi lớn.
2. Config còn hai nguồn dễ gây nhầm: `components.entry_model` và `signals.entry_model`; `components.exit_model` và `signals.exit_model`.
3. Matrix workflow đã có artifact từng run nhưng chưa có ranking tổng hợp tự sinh sau mỗi lần chạy.
4. `src/backtest/engine.py` vẫn là legacy monolith chứa nhiều flag version.
5. Một số tài liệu cũ còn mô tả kế hoạch đã qua, dễ gây nhiễu với trạng thái V3 hiện tại.

## Nguyên tắc cleanup

- Không thêm thuật toán mới trong phase cleanup.
- Không refactor đồng thời nhiều tầng nếu không có regression test bảo vệ.
- Không xóa thêm legacy nếu chưa có parity/golden rõ ràng.
- Mọi thay đổi nên nhỏ, dễ review, có verification cụ thể.
- Logic mới không thêm flag vào `backtest_unified`; nếu cần rule mới thì thêm strategy riêng.

## Phase Clean-1 — Repo hygiene

**Mục tiêu:** working tree dễ hiểu, artifact không lẫn vào source, thay đổi có thể review/commit theo cụm.

### Việc cần làm

1. Phân loại toàn bộ thay đổi hiện tại thành 5 nhóm:
   - Source code runtime.
   - Config YAML.
   - Tests/golden.
   - Docs.
   - Generated artifacts / cache / output.
2. Đảm bảo các thư mục output không bị track nhầm:
   - `results/`
   - `stock_ml/results/`
   - `**/__pycache__/`
   - `stock_ml/visualization/node_modules/` nếu visualization không cần vendor dependency trong repo.
3. Kiểm tra `.gitignore` và chỉ thêm pattern thật sự cần.
4. Không xóa file generated đang có nếu chưa xác nhận; trước tiên chỉ ignore hoặc tách khỏi commit.
5. Chia thay đổi thành các cụm commit/PR logic nếu sau này commit:
   - config/schema
   - runner/backtest
   - tests/golden
   - docs
   - cleanup artifact

### Kết quả 2026-05-02

Working tree đã được phân loại:

- Source code runtime: chưa có thay đổi runtime trong Clean-1.
- Config YAML: chưa có thay đổi config trong Clean-1.
- Tests/golden: chưa có thay đổi tests/golden trong Clean-1.
- Docs: `stock_ml/docs/refactor/README.md`, `stock_ml/docs/refactor/CLEAN_ARCH_NEXT_STEPS.md`.
- Generated artifacts / cache / output: `results/`, `stock_ml/results/experiments/`.

`.gitignore` đã thêm ignore cho artifact root `results/` và matrix output `stock_ml/results/experiments/`; không xóa generated file hiện có.

### Verification

```bash
git status --short
python -m pytest stock_ml/tests/regression/test_champions.py -q
```

### Done khi

- `git status` không còn artifact ngoài ý muốn.
- Người review nhìn vào diff biết mỗi cụm thay đổi làm gì.
- Champion regression pass hoặc có note rõ nếu mismatch là pre-existing.

## Phase Clean-2 — Canonical config cho Signal / Strategy / Execution

**Mục tiêu:** chỉ có một cách hiểu config entry/exit model.

### Vấn đề hiện tại

Schema đang có cả:

```yaml
components:
  entry_model: ...
  exit_model: ...

signals:
  entry_model: ...
  exit_model: ...
```

Trainer hiện dùng `signals.entry_model` để chọn entry model, còn exit model lại phụ thuộc `components.exit_model` ở nhiều chỗ. Điều này chạy được nhưng dễ sai khi matrix override một bên mà bên kia không đổi.

### Quyết định đề xuất

Chọn `signals` là canonical cho model signal:

```yaml
signals:
  features: leading_v2
  target:
    type: early_wave
  entry_model:
    type: lightgbm
    extras: {}
  exit_model:
    enabled: false
    type: null
    forward_window: 15
    loss_threshold: 0.05
    extras: {}
```

`components` chỉ nên là backward-compat input tạm thời, được normalize sang `signals` khi load config.

### Việc cần làm

1. Cập nhật `ExperimentConfig` để helper public lấy từ `signals`:
   - `entry_model_type()`
   - `feature_set()`
   - `target_dict()`
   - `exit_model_dict()`
2. Matrix alias nên map trực tiếp vào canonical path:
   - `features` → `signals.features`
   - `model_type` → `signals.entry_model.type`
   - `target_type` → `signals.target.type`
   - `exit_model` → `signals.exit_model`
3. Khi đọc YAML cũ, nếu chỉ có `components`, auto-fill `signals` từ `components`.
4. Khi cả hai cùng tồn tại nhưng khác nhau, nên fail validation hoặc log lỗi rõ, không silently merge.
5. Cập nhật champion YAML dần sang schema canonical.

### Kết quả 2026-05-02

`signals` đã được chọn làm canonical runtime section cho signal model config:

- `ExperimentConfig` auto-fill `signals` từ `components` cho YAML cũ.
- Nếu YAML có cả `components` và `signals` nhưng khác nhau, config load fail fast thay vì silently merge.
- Helper public `entry_model_type()`, `feature_set()`, `target_dict()`, `exit_model_dict()` đọc từ `signals`.
- Matrix aliases đã map vào canonical path: `features`, `model_type`, `target_type`, `exit_model` → `signals.*`.
- `quick_entry_exit` đã đổi axis/base sang canonical `signals` path.
- Sau khi normalize, `components` vẫn được backfill từ `signals` để legacy caller còn đọc được trong giai đoạn chuyển tiếp.

### Verification

```bash
python -m stock_ml validate champions/v22
python -m stock_ml validate champions/v42_a
python -m stock_ml validate matrix/quick_entry_exit
python -m pytest stock_ml/tests/regression/test_champions.py -q
```

### Done khi

- Matrix entry/exit không còn phụ thuộc vào field cũ.
- Một experiment resolved config nhìn vào là biết model nào được train.
- Không còn trường hợp `signals.entry_model` khác `components.entry_model` mà vẫn chạy âm thầm.

## Phase Clean-3 — Matrix ranking chuẩn hóa

**Mục tiêu:** chạy matrix xong có bảng so sánh rõ, không cần thao tác thủ công nhiều.

### Việc cần làm

1. Sau `run-matrix`, tự ghi file tổng hợp trong matrix result dir:
   - `ranking.csv`
   - `ranking.json`
2. Ranking row nên gồm tối thiểu:
   - `rank`
   - `name`
   - `composite_score`
   - `total_pnl`
   - `win_rate`
   - `mdd_per_symbol`
   - `yearly_consistency`
   - `trade_count`
   - `avg_holding_days`
   - `feature_set`
   - `entry_model`
   - `exit_model_type`
   - `exit_model_enabled`
   - `config_hash`
3. `compare-matrix` nên đọc được trực tiếp matrix dir và in top N.
4. Thêm winner guards:
   - trade count quá thấp
   - PnL tập trung vào một symbol
   - yearly consistency quá xấu
   - drawdown quá cao
5. Nếu dùng `--top-k-preview`, artifact preview và full-run phải tách thư mục rõ.

### Kết quả 2026-05-02

Matrix ranking đã được chuẩn hóa ở CLI:

- `run-matrix` tự sinh `ranking.csv` và `ranking.json` ở root của matrix result dir sau khi chạy xong.
- `ranking_row.json` từng run đã có đủ field cấu hình để so sánh: `feature_set`, `entry_model`, `exit_model_type`, `exit_model_enabled`, `config_hash`.
- `compare-matrix` ưu tiên đọc `ranking.json`, fallback sang `*/ranking_row.json` nếu artifact tổng hợp chưa tồn tại.
- Output `compare-matrix` có `rank` và hỗ trợ `--top N`.
- Winner guards đã có: trade count thấp, PnL tập trung vào một symbol, yearly consistency xấu, drawdown cao.
- `--top-k-preview` tiếp tục tách artifact preview vào `<matrix>_preview`, full-run vào `<matrix>`.

### Verification

```bash
python -m stock_ml run-matrix matrix/quick_entry_exit --symbols-limit 10 --device cpu
python -m stock_ml compare-matrix results/experiments/quick_entry_exit
python -m stock_ml compare-matrix results/experiments/quick_entry_exit --top 3
python -m pytest stock_ml/tests/regression/test_champions.py -q
```

### Done khi

- Sau một lệnh matrix có thể thấy winner ngay.
- Có file ranking để lưu lịch sử quyết định.
- Promote champion dựa trên `config_hash` và `config.resolved.yaml`, không dựa vào nhớ tay.

## Phase Clean-4 — Cô lập legacy backtest monolith

**Mục tiêu:** không để `backtest_unified` tiếp tục phình ra, nhưng cũng không phá parity.

### Vấn đề hiện tại

`src/backtest/engine.py` vẫn chứa nhiều flag version (`v26`, `v27`, `v28`, ...). Đây là nơi rủi ro cao nhất vì vừa là execution vừa chứa strategy legacy.

### Chiến lược đề xuất

Không refactor lớn ngay. Trước mắt coi `backtest_unified` là legacy execution kernel có contract rõ:

- Input: prediction cache + params.
- Output: trades dict/dataframe theo format regression.
- Không thêm behavior mới nếu có thể viết thành strategy V3.

### Việc cần làm

1. Tạo boundary rõ trong docs/code comments ngắn:
   - legacy behavior được giữ vì parity.
   - feature mới không thêm vào đây.
2. Viết characterization tests cho các behavior quan trọng trước khi tách:
   - entry creation
   - hard stop
   - fast exit
   - HAP/preempt
   - min hold
3. Tách dần các cụm flag đã có strategy tương ứng ra helper/strategy, mỗi lần chỉ một cụm.
4. Sau mỗi cụm, chạy parity champion liên quan.
5. Không đổi CSV/golden format trừ khi có chủ ý regenerate.

### Verification

```bash
python -m pytest stock_ml/tests/execution -q
python -m pytest stock_ml/tests/strategy -q
python -m pytest stock_ml/tests/regression/test_v22_parity.py -q
```

### Done khi

- `engine.py` không nhận thêm flag version mới.
- Logic mới đi qua strategy registry.
- Những phần legacy còn lại có lý do tồn tại rõ.

## Phase Clean-5 — Docs consolidation

**Mục tiêu:** tài liệu phản ánh trạng thái hiện tại, không khiến người đọc theo kế hoạch cũ.

### Việc cần làm

1. Chọn 3 tài liệu canonical:
   - `ARCHITECTURE_V3.md` — kiến trúc hiện tại.
   - `REFACTOR_V3_PLAN.md` — lịch sử phase đã làm.
   - `CLEAN_ARCH_NEXT_STEPS.md` — việc cleanup tiếp theo.
2. Các tài liệu cũ như `ARCHITECTURE.md`, `REFACTOR_ROADMAP.md`, `CLEANUP_PLAN.md` nên ghi rõ là historical nếu không còn đúng.
3. Update `docs/refactor/README.md` để người đọc bắt đầu từ tài liệu mới.
4. Tránh tạo thêm roadmap mới nếu chỉ là checklist ngắn; cập nhật file này thay vì phân mảnh.

### Kết quả 2026-05-02

Docs refactor đã được gom lại quanh 3 tài liệu canonical:

- `docs/refactor/README.md` bắt đầu từ `ARCHITECTURE_V3.md`, `REFACTOR_V3_PLAN.md`, `CLEAN_ARCH_NEXT_STEPS.md`.
- `ARCHITECTURE.md`, `REFACTOR_ROADMAP.md`, `CLEANUP_PLAN.md` đã được đánh dấu historical ở đầu file.
- Status trong README đã cập nhật sang V3 hiện tại: canonical `signals`, matrix ranking tổng hợp, và legacy backtest còn cần cô lập.
- Checklist trước khi thêm feature mới trong README không còn trỏ theo roadmap V2 cũ.

### Verification

- Mở `docs/refactor/README.md` và đọc theo thứ tự vẫn hiểu trạng thái mới.
- Không có hướng dẫn cũ bảo chạy command đã bị xóa.

### Done khi

- Người mới đọc docs không bị lẫn giữa V2 roadmap, V3 done state, và cleanup next steps.

## Thứ tự thực hiện đề xuất

```text
1. Clean-1 Repo hygiene
2. Clean-2 Canonical config
3. Clean-3 Matrix ranking
4. Clean-5 Docs consolidation
5. Clean-4 Legacy backtest isolation
```

Lý do: làm sạch git và config trước sẽ giảm rủi ro khi đụng matrix/backtest. Legacy backtest nên để sau vì rủi ro parity cao nhất.

## Rủi ro chính

| Rủi ro | Cách giảm |
|---|---|
| Artifact bị commit nhầm | `.gitignore` + review `git status --short` trước commit |
| Config silently mismatch | validation fail khi `components` và `signals` khác nhau |
| Matrix winner ảo | winner guards + symbol/year consistency |
| Backtest parity vỡ | characterization test + chỉ tách từng cụm nhỏ |
| Docs cũ gây hiểu nhầm | đánh dấu historical và update index |

## Checklist trước khi thêm feature mới

- [x] Working tree không còn generated artifacts ngoài ý muốn.
- [x] Config canonical cho entry/exit đã rõ.
- [ ] `validate matrix/quick_entry_exit` pass.
- [x] `run-matrix` có ranking tổng hợp.
- [ ] Champion regression pass hoặc mismatch được ghi rõ.
- [x] Không còn docs chính nào hướng dẫn theo schema/CLI cũ.

## Definition of Clean

Dự án được coi là clean khi:

1. Một champion có thể được mô tả gần như hoàn toàn bằng YAML.
2. Entry model, exit model, strategy rules, execution backtester có trách nhiệm riêng.
3. Matrix comparison chạy được, lưu artifact đầy đủ, tự xếp hạng được.
4. Legacy backtest còn tồn tại nhưng bị cô lập, không là nơi thêm feature mới.
5. Test layout khớp với kiến trúc: signals / strategy / execution / regression.
6. Git diff trước commit nhỏ, có chủ đề rõ, không lẫn output/cache.
