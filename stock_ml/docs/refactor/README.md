# Refactor Documentation Index

Bộ tài liệu refactor cho dự án Stock ML Trading System V3.

## Đọc theo thứ tự

1. **[ARCHITECTURE_V3.md](ARCHITECTURE_V3.md)** — Kiến trúc hiện tại: Signal / Strategy / Execution
   - Signal layer chỉ dự đoán entry/exit model
   - Strategy layer quyết định entry/hold/exit rules
   - Execution layer chạy backtest và sinh trades/metrics

2. **[REFACTOR_V3_PLAN.md](REFACTOR_V3_PLAN.md)** — Lịch sử phase V3 đã triển khai
   - Schema YAML 3 tầng
   - Champion runners và strategy registry
   - Generic V3 flow, regression/parity status

3. **[CLEAN_ARCH_NEXT_STEPS.md](CLEAN_ARCH_NEXT_STEPS.md)** — Cleanup tiếp theo sau V3
   - Repo hygiene, canonical config, matrix ranking
   - Docs consolidation, legacy backtest isolation
   - Definition of Clean trước khi thêm feature mới

4. **[CHAMPION_VERSIONS.md](CHAMPION_VERSIONS.md)** — Champion versions và coverage nghiên cứu
   - Lý do chọn từng champion
   - Coverage matrix (features × targets × models)
   - Promotion path từ legacy → champion

5. **[ENTRY_EXIT_RESEARCH_REFACTOR_PLAN.md](ENTRY_EXIT_RESEARCH_REFACTOR_PLAN.md)** — Lịch sử refactor entry/exit research
   - ExitModel component, fusion exit policies, matrix entry × exit
   - Artifact/ranking để promote champion
   - Trạng thái Phase 0–8 và việc còn lại

6. **[ARCHITECTURE.md](ARCHITECTURE.md)**, **[REFACTOR_ROADMAP.md](REFACTOR_ROADMAP.md)**, **[CLEANUP_PLAN.md](CLEANUP_PLAN.md)** — Historical V2 docs
   - Chỉ dùng để tra quyết định cũ
   - Không dùng làm hướng dẫn triển khai hiện tại

## Quick reference

### Mục tiêu V3 hiện tại
- Signal layer chỉ dự đoán, không quyết định trade.
- Strategy layer chứa entry/hold/exit rules.
- Execution layer chạy backtest và giữ parity với legacy khi cần.
- Matrix so sánh Entry × Exit × Feature × Strategy qua YAML và ranking artifact.

### Status hiện tại (2026-05-02)
- V3 hiện dùng 3 tầng: Signal / Strategy / Execution.
- `signals` là canonical runtime config cho entry/exit model; `components` chỉ còn là backward-compat input được normalize khi load.
- Matrix `quick_entry_exit` expand `2 features × 2 entry models × 2 exit configs = 8 experiments` bằng canonical `signals.*` paths.
- `run-matrix` sinh `ranking.csv` và `ranking.json`; `compare-matrix` đọc ranking tổng hợp và hỗ trợ `--top N`.
- Docs chính hiện là `ARCHITECTURE_V3.md`, `REFACTOR_V3_PLAN.md`, `CLEAN_ARCH_NEXT_STEPS.md`.
- Partial/còn lại:
  - `src/backtest/engine.py` vẫn là legacy monolith cần characterization tests trước khi tách tiếp.
  - `predictions_meta.json`, cache hit-rate metadata và parallel matrix execution chưa có.
  - Một phần exit logic legacy vẫn nằm trong champion runners/backtest legacy để giữ parity.

### Exit terminology

| Thuật ngữ | Ý nghĩa |
|-----------|---------|
| `exit_model` | Model supervised dự đoán exit signal riêng (`signals.exit_model`). |
| `strategy.exit_rules` | Rule thoát chạy trong strategy layer, ví dụ `hard_stop`, `v22_fast_exit`, `ma_cross_hybrid_exit`. |
| `v22` | Champion baseline không train/consume exit model, nhưng vẫn dùng rule exits. |
| `v22_with_exit_model` | Champion có exit model cộng với rule exits an toàn. |

Không dùng `hybrid` đơn lẻ cho artifact mới; tên này dễ nhầm với `ma_cross_hybrid_exit` legacy. Nếu cần mô tả combo exit model + rule, dùng `exit_model_plus_rules`.

### Champion versions
v22, v22_with_exit_model, v32, v34, v35b, v37a, v37a_exit, v37d, v39d, v42_a, v19_3, rule

### Timeline
8 tuần, full-time, solo dev.

### Trước khi thêm feature mới

Bắt buộc đọc:
1. ARCHITECTURE_V3.md
2. REFACTOR_V3_PLAN.md
3. CLEAN_ARCH_NEXT_STEPS.md

Bắt buộc kiểm tra:
- Working tree không lẫn generated artifacts ngoài ý muốn.
- `python -m stock_ml validate matrix/quick_entry_exit` pass.
- Matrix ranking tổng hợp sinh được từ `run-matrix` hoặc đã có artifact hợp lệ.
- Champion regression pass hoặc mismatch được ghi rõ.
- Feature mới không thêm flag vào legacy `backtest_unified`; nếu cần rule mới thì thêm strategy riêng.

## Bugs phát hiện

- **[EXIT_MODEL_BUG.md](EXIT_MODEL_BUG.md)** — Exit model trained nhưng pipeline drop output (2026-04-27). Phase 2.4 preserve behavior để match golden; exit-model fix tách sau parity.

## Support docs

- ✅ `FUSION_STRATEGY_INVENTORY.md` — Mapping từ flag cũ → fusion strategy mới.
- ✅ `HOW_TO_ADD_FEATURE_BLOCK.md` — Guide thêm feature block.
- ✅ `HOW_TO_ADD_FUSION_STRATEGY.md` — Guide thêm fusion strategy.
- ✅ `HOW_TO_ADD_ENTRY_MODEL.md` — Guide thêm entry model.
- ✅ `HOW_TO_PORT_LEGACY_VERSION.md` — Guide promote legacy version.
- ✅ `HOW_TO_RUN_MATRIX.md` — Guide grid search.
- ✅ `ENTRY_EXIT_RESEARCH_REFACTOR_PLAN.md` — Historical entry/exit research refactor.
- ✅ `CLEAN_ARCH_NEXT_STEPS.md` — Kế hoạch cleanup tiếp theo: repo hygiene, canonical config, matrix ranking, legacy isolation.
- `BENCHMARK.md` — Performance benchmarks (chạy bằng `python -m stock_ml benchmark`)

## Quy ước hiện tại

- Cleanup/refactor tiếp theo theo [CLEAN_ARCH_NEXT_STEPS.md](CLEAN_ARCH_NEXT_STEPS.md), không tạo thêm roadmap mới nếu chỉ là checklist ngắn.
- Logic mới đi qua Signal / Strategy / Execution; không thêm flag mới vào legacy `backtest_unified`.
- Promote champion dựa trên `config_hash` và `config.resolved.yaml`, không dựa vào nhớ tay.

## Liên kết tài liệu khác

- `../README.md` — Project README chính.
- `../FIRST_TRAINING_REPORT.md` — Báo cáo đầu tiên.
- `../SMART_EXIT_PROPOSAL.md` — Đề xuất exit model đã được partial implement.
- `../MODEL_EVALUATION_REPORT.md` — Đánh giá model.
