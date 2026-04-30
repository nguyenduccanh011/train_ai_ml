# 2026-04-28 — Phase 6.3 Cleanup (complete)

## Done

### Phase 6.3 hoàn thành — cleanup legacy structure + experiments

1. **Tạo `legacy/` structure**:
   - `legacy/scripts_reference/` — 17 historical scripts với model configs (v10–v22)
   - `legacy/strategies/`, `legacy/configs/`, `legacy/docs/` — placeholder cho future port
   - `legacy/README.md` — giải thích mục đích + promotion path

2. **`archive/scripts/` cleanup**:
   - Move 17 scripts có config vào `legacy/scripts_reference/` (giữ reference)
   - Xóa 27 one-off/duplicate scripts (~500KB)
   - `archive/scripts/` nay rỗng

3. **Nén `archive/results_legacy/`**:
   - 19MB → `archive/results_legacy.tar.gz` (~12MB)
   - Lấy lại được khi cần: `tar xzf archive/results_legacy.tar.gz`

4. **`experiments/` cleanup** — từ 50+ files xuống 12:
   - Giữ: `run_v22_final.py`, `run_v31_final.py`, `run_v32_final.py`, `run_v33_final.py`, `run_v34_final.py`, `run_v37a.py`, `run_v37b.py`, `run_v37c.py`, `run_v37d.py`, `run_v39d.py`, `run_v42.py`, `__init__.py`
   - Xóa: v23–v30, v38 variants, v39 non-champion, v40/v41 dashboard-only, one-off experiments, intermediate CSVs

5. **Root level cleanup**:
   - `visualize_v42.py` → `visualization/scripts_v42.py`
   - `compare_rule_vs_model.py` → `analysis/compare_rule_vs_model.py`
   - Xóa `run_exit_29.log`, `run_exit_remaining.log`

## Verification

- `pytest tests/regression/test_champions.py -q` → **12/12 passed**
- `pytest tests/components/ -q -k "not integration"` → **203 passed**

## Stuck

- Không.

## Decision

- `experiments/` giữ `run_v37b.py`, `run_v37c.py` dù không phải champion active — các runners (`v37d_runner.py`) delegate sang `backtest_v37d → backtest_v32` nên không dùng trực tiếp, nhưng giữ lại vì là phần của lineage history.
- `archive/scripts/` không xóa được folder bằng `rm -rf` (Device or resource busy) nhưng xóa được files bên trong — folder rỗng, tương đương xóa.

## Next

- Phase 6.3 DONE. Toàn bộ Phase 0–6 hoàn thành.
- Còn 1 action optional: tag `v2.0` — thực hiện khi user yêu cầu.
- Refactor roadmap đã lock. Phase 7+ là research iterations với kiến trúc mới.
