# Refactor Documentation Index

Bộ tài liệu refactor cho dự án Stock ML Trading System v2.0.

## Đọc theo thứ tự

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** — Kiến trúc đích (component-based, composable)
   - 6 loại component: Features, Targets, Models, Exit Models, Fusion, Backtester
   - Schema YAML mới
   - Validation rules
   - Caching strategy
   - CLI design

2. **[CHAMPION_VERSIONS.md](CHAMPION_VERSIONS.md)** — 11 version sẽ port qua kiến trúc mới
   - Lý do chọn từng champion
   - Coverage matrix (features × targets × models)
   - 49 retired versions strategy
   - Promotion path từ legacy → champion

3. **[REFACTOR_ROADMAP.md](REFACTOR_ROADMAP.md)** — 8 phase chi tiết, 8 tuần
   - Phase 0: Chuẩn bị (golden baseline, tooling)
   - Phase 1: Foundation (component framework)
   - Phase 2: Fusion stack (KHÓ NHẤT)
   - Phase 3: Pipeline orchestrator
   - Phase 4: Migration & legacy adapter
   - Phase 5: Testing & verification
   - Phase 6: Tooling enhancements
   - Phase 7+: Tận dụng kiến trúc mới

4. **[CLEANUP_PLAN.md](CLEANUP_PLAN.md)** — Dọn dẹp archive/ và codebase
   - Quyết định xóa/giữ từng folder
   - Execution plan từng bước (commit-by-commit)
   - Rollback plan

5. **[ENTRY_EXIT_RESEARCH_REFACTOR_PLAN.md](ENTRY_EXIT_RESEARCH_REFACTOR_PLAN.md)** — Refactor entry/exit research sau v2 foundation
   - ExitModel component, fusion exit policies, matrix entry × exit
   - Artifact/ranking để promote champion
   - Trạng thái Phase 0–8 và việc còn lại

## Quick reference

### Mục tiêu refactor
- Giải quyết `**kwargs` túi rác → explicit interfaces
- 60 strategy file rời rạc → composable components
- Khó thử tổ hợp → grid-search Entry × Exit × Feature × Fusion qua YAML

### Status hiện tại (2026-05-01)
- Foundation v2 đã có: component folders, champion runners, `Pipeline`, `ExperimentConfig`, prediction cache, matrix expander, CLI `python -m stock_ml`.
- Entry/exit research refactor đã triển khai đến quick matrix:
  - EntryModel contract/registry smoke đã có.
  - ExitModel component có `null`, `lightgbm`, `xgboost`, `catboost`.
  - Matrix `quick_entry_exit` hiện expand `2 features × 2 entry models × 2 exit configs = 8 experiments`.
  - Artifact/ranking matrix có `metrics.json`, `ranking_row.json`, `config.resolved.yaml`, `compare-matrix`.
  - Winner quick matrix đã promote thành champion `v22_with_exit_model`; champion hash regression hiện pass.
- Partial/còn lại:
  - Exit-rule validation sâu (`signals.exit_model.type: null` + exit-model rule, exit model bật nhưng strategy không consume signal) chưa hoàn chỉnh.
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

### Trước khi bắt đầu

Bắt buộc đọc:
1. ARCHITECTURE.md đầy đủ
2. CHAMPION_VERSIONS.md
3. REFACTOR_ROADMAP.md Phase 0 (chuẩn bị)
4. CLEANUP_PLAN.md (Step 1 — Backup)

Bắt buộc làm trước Phase 1:
- ✅ Golden baseline cho 11 champions (Phase 0.2 — done 2026-04-27, CPU mode)
- ✅ Git tag `pre-cleanup-snapshot`
- ✅ Set random seeds (Phase 0.1 — done 2026-04-26)
- ✅ Tooling setup (ruff, mypy, pytest, pre-commit) — Phase 0.3 done 2026-04-28
- ✅ Architecture lock + branch policy documented — Phase 0.4/0.5 status updated 2026-04-29

## Bugs phát hiện

- **[EXIT_MODEL_BUG.md](EXIT_MODEL_BUG.md)** — Exit model trained nhưng pipeline drop output (2026-04-27). Phase 2.4 preserve behavior để match golden; exit-model fix tách sau parity.

## Support docs

- ✅ `FUSION_STRATEGY_INVENTORY.md` — Mapping từ flag cũ → fusion strategy mới (Phase 2.1)
- ✅ `HOW_TO_ADD_FEATURE_BLOCK.md` — Guide thêm feature block (Phase 5.3)
- ✅ `HOW_TO_ADD_FUSION_STRATEGY.md` — Guide thêm fusion strategy (Phase 5.3)
- ✅ `HOW_TO_ADD_ENTRY_MODEL.md` — Guide thêm entry model (Phase 5.3)
- ✅ `HOW_TO_PORT_LEGACY_VERSION.md` — Guide promote legacy version (Phase 5.3)
- ✅ `HOW_TO_RUN_MATRIX.md` — Guide grid search (Phase 5.3)
- ✅ `ENTRY_EXIT_RESEARCH_REFACTOR_PLAN.md` — Kế hoạch refactor entry/exit research; Phase 0/1/2/4/5/6/7 done cho quick-matrix scope, Phase 3/8 partial
- `BENCHMARK.md` — Performance benchmarks (chạy bằng `python -m stock_ml benchmark`)

## Diary template

Solo dev cần ghi diary để track. Format đề xuất:

```markdown
## 2026-04-26 — Phase 0.1 Lock seeds

### Done
- Set random_state=42 cho LightGBM, XGBoost, CatBoost trong src/models/registry.py
- torch.manual_seed(42) cho GRU
- np.random.seed(42) ở pipeline start

### Stuck
- LightGBM GPU mode kết quả vẫn khác CPU 0.001% → suspect floating point. Acceptable cho regression với tolerance.

### Decision
- Regression test cho GRU: chỉ run CPU để deterministic. GPU smoke test only.

### Next
- Phase 0.2: Run 11 champions với --force, save golden CSVs
```

Lưu vào `docs/refactor/diary/YYYY-MM-DD.md`.

## Quy ước

### Branch
- Long-running: `refactor/v2-clean-arch`
- Sub-phase: `refactor/phase-N-<name>`
- Mỗi sub-phase ≤ 1 tuần

### Commit message
```
[refactor/phase-2.3] Implement SMA200Filter fusion strategy

- New file: src/components/fusion/pre_entry/sma200_filter.py
- Unit test: tests/components/fusion/test_sma200_filter.py
- v22 still passes regression
```

### PR (nếu có)
Solo: skip PR, dùng `git merge --no-ff` để giữ history.

## Câu hỏi thường gặp

### Q: Nếu Phase 2 stuck, có thể skip sang Phase 3 không?
A: Không. Phase 3 (orchestrator) cần fusion stack hoạt động. Stuck > 2 ngày → đọc lại ARCHITECTURE.md, có thể design có flaw → revise.

### Q: Có cần convert tất cả legacy versions không?
A: Không. Legacy adapter là đủ. Chỉ port champion (11). Migrate tool có sẵn để promote legacy nếu cần sau.

### Q: Performance regression > 20% — phải làm gì?
A: Profile (`python -m cProfile`). 99% là cache miss hoặc validation overhead. Tune cache key generation hoặc lazy validation.

### Q: Có thay đổi YAML schema giữa chừng được không?
A: KHÔNG. Schema lock-in ở Phase 0.5. Đổi sau Phase 0 = restart Phase 1.

### Q: Khi nào pause refactor?
A:
- Stuck 1 phase > 1 tuần
- Phát hiện flaw fundamental → revise ARCHITECTURE.md
- Burnout
- Có incident production cần fix

### Q: Solo dev sợ confirmation bias — làm sao?
A: 
- 1 tháng/lần explain kiến trúc cho người ngoài (gia đình OK)
- Hoặc viết blog post mô tả → tự đặt câu hỏi
- Diary daily — đọc lại diary cũ để check decision có nhất quán không

## Liên kết tài liệu khác

- `../README.md` — Project README chính (sẽ update sau Phase 5)
- `../FIRST_TRAINING_REPORT.md` — Báo cáo đầu tiên
- `../SMART_EXIT_PROPOSAL.md` — Đề xuất exit model (đã partial implement)
- `../MODEL_EVALUATION_REPORT.md` — Đánh giá model
