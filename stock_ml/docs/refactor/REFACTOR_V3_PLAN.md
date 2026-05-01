# Refactor V3 Plan — Signal / Strategy / Execution

**Status:** Approved 2026-05-01
**Goal:** Tách kiến trúc thành 3 tầng rõ ràng, xóa legacy sau khi migrate xong.

## Quyết định nền

1. **Format CSV / golden checksum có thể regenerate** nếu cần cho cấu trúc sạch hơn. Vẫn giữ parity *hành vi* (PnL, trade count tương đương).
2. **Xóa hoàn toàn legacy** sau khi tất cả logic đã migrate và kết quả mới validate là tốt.
3. **Cho phép giữ vài runner đặc biệt** (vd `rule_runner`) — không bắt buộc 100% generic.

## Kiến trúc đích

```
Trading System
├── Signal Layer        ← chỉ dự đoán
│   ├── entry_model
│   └── exit_model
├── Strategy Layer      ← quyết định hành động
│   ├── entry_rules
│   ├── hold_rules
│   └── exit_rules
└── Execution Layer     ← thực thi
    └── backtester
```

YAML schema đích:

```yaml
name: v22
feature_set: leading_v2
target: trend_regime

signals:
  entry_model:
    type: lightgbm
    params: {...}
  exit_model:
    type: null

strategy:
  entry_rules:
    - skip_choppy
    - sma200_filter
  hold_rules:
    - trend_persistence
  exit_rules:
    - hard_stop
    - trailing_stop

execution:
  backtester: simple_long
  capital: 100_000_000
```

## Phases

### Phase 1 — Audit & inventory (~1 ngày)

**Status:** Done 2026-05-01 — output: [COMPONENT_INVENTORY.md](COMPONENT_INVENTORY.md).

- [x] Phân loại từng strategy trong [src/components/fusion/strategies/](../../src/components/fusion/strategies/) thành: `entry_filter`, `entry_signal`, `hold_rule`, `exit_rule`.
- [x] Liệt kê quirk của từng champion runner trong [src/components/runners/](../../src/components/runners/).
- [x] Map từng key YAML champion sang tầng tương ứng.
- [x] **Output:** `docs/refactor/COMPONENT_INVENTORY.md`.

**Verification:** review tài liệu.

### Phase 2 — Schema YAML 3 tầng (backward compat)

**Status:** Done 2026-05-01 — implemented typed V3 sections in [src/pipeline/config.py](../../src/pipeline/config.py) and fusion validation in [src/pipeline/validate.py](../../src/pipeline/validate.py).

- [x] Mở rộng [src/pipeline/config.py](../../src/pipeline/config.py): `SignalsConfig`, `StrategyV3Config`, `ExecutionConfig`.
- [x] Auto-migrator đọc YAML cũ → translate in-memory sang schema mới.
- [x] Chưa rewrite file YAML trên disk.

**Verification:**
```bash
pytest stock_ml/tests/regression/test_champions.py -q  # 13 passed
pytest stock_ml/tests/components/ -q                   # 233 passed
```

### Phase 3 — Tách rule exit khỏi entry model

**Status:** In progress 2026-05-01 — FusionStack group (`v22`, `v22_exit_b`, `v19_3`) wired to V3 strategy config.

- [x] Add explicit `force_exit_rules` / `active_exit_rules` to `StrategyV3Config` while keeping flat `exit_rules`.
- [x] Forward `strategy_v3` from orchestrator to runners that accept it (signature-based kwarg filtering so legacy runners are unaffected).
- [x] Build exit strategies from registered rule names instead of hardcoded lists for FusionStack runners.
- [x] Add validation for V3 strategy rule names.
- [x] Register missing strategies: `model_b_exit`, `signal_hard_cap`, `fast_exit_loss`, `hap_preempt`.
- [x] Move strategy construction out of per-symbol loop (build once per run).
- [ ] Lineage runners (v32, v35b, v37a, v37d, v39d, v42_a) — exit logic vẫn trong `engine.backtest_unified()`.
- [ ] **Làm từng champion một, một commit / champion.**

**Verification:**
```bash
python -m pytest stock_ml/tests/regression/test_champions.py -q  # 13 passed
python -m pytest stock_ml/tests/components/ -q                  # 246 passed
```

**Risk:** cao — dễ vỡ parity. Phải pass golden trước khi sang champion tiếp theo. Nếu format CSV đổi, regenerate golden + verify PnL tương đương.

### Phase 4 — Unified Generic Runner

- Implement `GenericRunner` driven bởi `signals + strategy + execution`.
- Migrate từng champion: xóa `vXX_runner.py` sau khi golden pass.
- Cho phép giữ `rule_runner.py` nếu thực sự không có ML.

**Verification:** golden parity sau mỗi champion migrate.

### Phase 5 — Chuẩn hóa terminology

| Cũ | Mới |
|---|---|
| `Model B`, `model_b`, `exit_b` | `exit_model` |
| `null_exit`, `*_exit_b` (YAML name) | `v22`, `v22_with_exit_model` |
| `fusion.exit_override` | `strategy.exit_rules` |
| `components.entry_model` | `signals.entry_model` |
| `LegacyVersionAdapter` | `LegacyAdapter` |

- Rename code: 1 loại / commit.
- Rename YAML files.
- Cập nhật README + ARCHITECTURE.md.

### Phase 6 — Migrate & XÓA legacy

- `python -m stock_ml migrate-legacy --all` → tất cả legacy có YAML mới.
- Validate kết quả mới == kết quả cũ (PnL, trade count).
- **Sau khi validate OK:**
  - Xóa [src/pipeline/legacy_adapter.py](../../src/pipeline/legacy_adapter.py).
  - Xóa [stock_ml/experiments/](../../experiments/) (legacy backtest functions).
  - Xóa [config/models.yaml](../../config/models.yaml).
  - Xóa [stock_ml/legacy/](../../legacy/).
  - Xóa archive scripts.

**Verification:**
```bash
python -m stock_ml run champions/v22   # đường chính duy nhất
pytest tests/regression/ -q
```

### Phase 7 — Tests & docs cleanup

- Tách test theo tầng: `tests/signals/`, `tests/strategy/`, `tests/execution/`.
- Cập nhật HOW_TO guides, thêm `HOW_TO_ADD_EXIT_MODEL.md`.
- Viết `ARCHITECTURE_V3.md`.
- Regenerate golden hash final, note trong [tests/regression/golden/README.md](../../tests/regression/golden/README.md).

### Phase 8 — Feature mới (sau khi nền sạch)

- Multiple exit models (ensemble exit).
- Position sizing component tách khỏi backtester.
- Portfolio-level strategy (cross-symbol).
- Live paper trading executor.

## Thứ tự & lý do

```
1. Inventory          → biết phải làm gì
2. Schema mới         → ngôn ngữ chung trước khi đổi code
3. Tách exit rule     → thiết lập trách nhiệm đúng
4. Unified runner     → xóa duplication
5. Đổi tên            → khi cấu trúc đã ổn
6. Xóa legacy         → còn đường lùi đến đây
7. Test/docs          → đóng gói
8. Feature mới        → trên nền sạch
```

## Ước lượng

- Phase 1-2: 1-2 ngày mỗi phase.
- Phase 3-4: 3-5 ngày mỗi phase (nặng nhất, parity-sensitive).
- Phase 5-7: 1-2 ngày mỗi phase.
- Phase 8: ngoài scope refactor.
