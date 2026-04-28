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

## Quick reference

### Mục tiêu refactor
- Giải quyết `**kwargs` túi rác → explicit interfaces
- 60 strategy file rời rạc → composable components
- Khó thử tổ hợp → grid-search Entry × Exit × Feature × Fusion qua YAML

### Status hiện tại (2026-04-28)
- Phase 0: seeds, CPU golden baseline, tooling, architecture lock và branch policy đã chốt.
- Phase 1: base interfaces, feature blocks, targets và model wrappers đã port.
- Phase 2.1-2.4f: fusion inventory/interface + parity runners cho 11 champion đã xong.
- Phase 3: pipeline orchestrator (`Pipeline`, `ExperimentConfig`, `PredictionCacheManager`, `expand_matrix`, CLI `python -m stock_ml`) đã hoàn thành.
- Phase 4: Legacy adapter (`LegacyVersionAdapter`), migration tool (`migrate-legacy`), deprecate `run_pipeline.py` — đã hoàn thành.
- Phase 5: Smoke tests (10 legacy + property-based fusion), benchmark script, 5 HOW_TO guides — đã hoàn thành.
- Next: Phase 6 — CI/CD + cleanup + tag v2.0.

### Champion versions (11)
v22, v32, v34, v35b, v37a, v37a_exit, v37d, v39d, v42_a, v19_3, rule

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

- **[EXIT_MODEL_BUG.md](EXIT_MODEL_BUG.md)** — Exit model trained nhưng pipeline drop output (2026-04-27). Phase 2.4 preserve behavior để match golden; Model B fix tách sau parity.

## Support docs

- ✅ `FUSION_STRATEGY_INVENTORY.md` — Mapping từ flag cũ → fusion strategy mới (Phase 2.1)
- ✅ `HOW_TO_ADD_FEATURE_BLOCK.md` — Guide thêm feature block (Phase 5.3)
- ✅ `HOW_TO_ADD_FUSION_STRATEGY.md` — Guide thêm fusion strategy (Phase 5.3)
- ✅ `HOW_TO_ADD_ENTRY_MODEL.md` — Guide thêm entry model (Phase 5.3)
- ✅ `HOW_TO_PORT_LEGACY_VERSION.md` — Guide promote legacy version (Phase 5.3)
- ✅ `HOW_TO_RUN_MATRIX.md` — Guide grid search (Phase 5.3)
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
