# Cleanup Plan — dọn dẹp archive/ và codebase

> Historical: tài liệu này là kế hoạch cleanup archive/codebase đời đầu. Đọc [CLEAN_ARCH_NEXT_STEPS.md](CLEAN_ARCH_NEXT_STEPS.md) cho trạng thái cleanup hiện tại.

## Triết lý

**Giữ lại**: cấu hình model cũ (vì có thể dùng để promote sau này) + lịch sử quan trọng.

**Xóa**: code có thể tái tạo, output có thể re-generate, scripts trùng lặp, file lớn không có giá trị nghiên cứu.

**Lưu ý**: tất cả những gì xóa đều có trong git history → **lấy lại được**. Không sợ mất.

## Tổng quan archive/ hiện tại

| Folder | Size | Quyết định |
|--------|------|-----------|
| `analysis/` | 312K | **XÓA** — analysis scripts có thể viết lại |
| `exports/` | 144K | **XÓA** — export scripts có thể viết lại |
| `misc/` | 78K | **XÓA** — debug/cleanup scripts |
| `results_legacy/` | 19M | **GIỮ** + nén — historical results value |
| `scripts/` | 1.2M | **GIỮ MỘT PHẦN** — chỉ scripts chứa cấu hình model |
| `src/` | 48K | **XÓA** — pipeline cũ + evaluation cũ, đã có thay thế |
| `visualization/` | 127M | **XÓA** — chiếm dung lượng lớn, regenerate được |

**Trước cleanup**: ~148MB  
**Sau cleanup**: ~20MB (giữ results_legacy + 1 phần scripts)  
**Tiết kiệm**: ~128MB

## Chi tiết quyết định

### 1. archive/analysis/ — **XÓA TOÀN BỘ**

**Nội dung hiện tại** (24 files, 312K):
```
analyze_trades.py             5.5K
analyze_v10_trades.py        24.8K
analyze_v12_vs_v11.py         8.3K
analyze_v18_rule_deep.py     17.4K
analyze_v19_1_deep.py        32.3K
analyze_v19_3_deep.py         9.6K
analyze_v23_diffs.py          5.7K
analyze_v23_vs_all.py         5.1K
analyze_v25_deep.py          18.5K
... (15 files khác)
```

**Lý do xóa**:
- Đều là one-off scripts cho phân tích cụ thể từng version
- Logic phân tích cơ bản (trade aggregation, win-rate breakdown) sẽ có trong `src/components/evaluation/` mới
- Không có cấu hình model unique
- Output đã được capture trong `docs/V*_*ANALYSIS*.md`

**Action**:
```bash
rm -rf stock_ml/archive/analysis/
```

### 2. archive/exports/ — **XÓA TOÀN BỘ**

**Nội dung** (12 files, 144K):
```
export_base_ohlcv.py           6.2K
export_comparison.py          10.4K
export_v10_comparison.py       9.8K
export_v11_comparison.py      10.2K
... (8 files tương tự cho v12-v24)
```

**Lý do xóa**:
- Logic export đã được unify ở `src/export/unified_export.py`
- Mỗi file là 1 variant cho 1 version cụ thể
- Format CSV → JSON cho dashboard không thay đổi → không cần reference

**Action**:
```bash
rm -rf stock_ml/archive/exports/
```

### 3. archive/misc/ — **XÓA TOÀN BỘ**

**Nội dung** (8 files, 78K):
```
_cleanup_v24.py            724B
_gen_v24.py                428B
fix_index.py               329B
generate_v7_viz.py         6.1K
visualize_v12_trades.py    8.9K
visualize_v22_trades.py   23.4K
visualize_v23_trades.py   20.9K
```

**Lý do xóa**:
- Underscore-prefix files = throwaway scripts
- Visualization scripts cũ thay bằng dashboard mới
- Helper utilities có thể recreate

**Action**:
```bash
rm -rf stock_ml/archive/misc/
```

### 4. archive/results_legacy/ — **GIỮ + NÉN**

**Size**: 19MB

**Nội dung**: Kết quả backtest historical từ v2-v22 (CSV + JSON)

**Lý do giữ**:
- Có giá trị lịch sử cho so sánh longitudinal
- Một số file có format đặc biệt (rule_trades_20260420_detailed.csv, trade_analysis files) khó tái tạo
- Reference khi promote legacy version qua kiến trúc mới

**Action — nén**:
```bash
cd stock_ml/archive/
tar czf results_legacy.tar.gz results_legacy/
rm -rf results_legacy/
# Còn lại 1 file ~5MB nén
```

**Hoặc dọn theo selectivity**:
- Giữ chỉ `results_*_compare_full.txt` (so sánh đẹp)
- Xóa intermediate CSV (trades_*.csv duplicate với `results/` chính)

→ Đề xuất: **nén toàn bộ** để giảm size mà vẫn lấy lại được khi cần.

### 5. archive/scripts/ — **GIỮ MỘT PHẦN**

**Size**: 1.2MB. Đây là phần **quan trọng** — chứa các script `run_vXX_compare.py` có cấu hình model cũ.

**Phân loại**:

#### 5a. GIỮ — Scripts chứa cấu hình model cũ (target: ~700KB)

Scripts này có inline params/flags cho version legacy mà yaml hiện tại không capture được hết:

```
run_v10_compare.py        27.8K   ✓ giữ — v10 config reference
run_v11_compare.py        29.1K   ✓ giữ — v11 config reference
run_v12_compare.py        25.8K   ✓ giữ — v12 config
run_v13_compare.py        28.1K   ✓ giữ — v13 config
run_v14_compare.py        30.5K   ✓ giữ — v14 config
run_v15_compare.py        33.9K   ✓ giữ — v15 config
run_v16_compare.py        33.9K   ✓ giữ — v16 config
run_v17_compare.py        38.0K   ✓ giữ — v17 config
run_v18_compare.py        39.7K   ✓ giữ — v18 config
run_v19_compare.py        43.9K   ✓ giữ — v19 config
run_v19_1_compare.py      45.8K   ✓ giữ — v19.1 config
run_v19_2_compare.py      46.8K   ✓ giữ — v19.2 config
run_v19_3_compare.py      47.2K   ✓ giữ — v19.3 config
run_v19_4_compare.py      49.0K   ✓ giữ — v19.4 config
run_v20_compare.py        33.8K   ✓ giữ — v20 config
run_v21_experiment.py     47.6K   ✓ giữ — v21 config
run_v22_compare.py        52.8K   ✓ giữ — v22 config
```

→ **17 files, ~700KB** — chứa "lịch sử di truyền" của project.

**Action**: Move vào `legacy/scripts_reference/` (read-only):
```bash
mkdir -p stock_ml/legacy/scripts_reference/
mv stock_ml/archive/scripts/run_v10_compare.py stock_ml/legacy/scripts_reference/
# ... (17 files)
# Add README giải thích "đây là reference, không chạy nữa"
```

#### 5b. XÓA — Scripts trùng lặp / experimental

```
run.py                              2.7K   ✗ generic, trùng pipeline mới
run_backtest.py                    14.2K   ✗ early version
run_backtest_v2.py                 18.3K   ✗ early version
run_deep_analysis.py               14.3K   ✗ analysis tool, recreate được
run_deep_compare_v19_v22.py         9.0K   ✗ duplicate logic
run_experiment_p2_p3_combo.py      10.2K   ✗ one-off experiment
run_experiment_proposals.py        55.5K   ✗ proposal scripts
run_export_signals.py               9.3K   ✗ duplicate export
run_feature_upgrade_test.py         6.9K   ✗ one-off test
run_full_comparison.py             14.4K   ✗ comparison tool
run_optimized_v3.py                19.8K   ✗ early version
run_trade_analysis.py              24.6K   ✗ analysis, recreate
run_v2.py                          10.3K   ✗ very old
run_v20_oos_experiment.py          11.0K   ✗ one-off
run_v20_train_size_experiment.py   13.0K   ✗ one-off
run_v22_tune.py                    12.0K   ✗ tuning script
run_v26_feature_compare.py         15.6K   ✗ feature compare
run_v27_experiments.py              9.1K   ✗ one-off
run_v2_tuned.py                     9.2K   ✗ very old
run_v4_analysis.py                 28.7K   ✗ analysis
run_v5_analysis.py                 31.4K   ✗ analysis
run_v6_backtest.py                 31.6K   ✗ early version
run_v7_compare.py                  21.1K   ✗ early version
run_v8_compare.py                  19.7K   ✗ early version
run_v8b_compare.py                 20.1K   ✗ early version
run_v9_compare.py                  22.1K   ✗ early version
v13_output.txt                      2.3K   ✗ output file
```

→ **27 files, ~500KB** — xóa.

**Action**:
```bash
# Tách: giữ files có config legacy, xóa rest
cd stock_ml/archive/scripts
# (delete one-off scripts here)
rm run.py run_backtest.py run_backtest_v2.py ...
```

### 6. archive/src/ — **XÓA TOÀN BỘ**

**Nội dung** (48K):
```
evaluation/
experiment_runner.py    5.1K
pipeline.py             7.3K
```

**Lý do xóa**:
- Đã được thay thế hoàn toàn bởi `src/` chính
- pipeline.py cũ là tiền thân của `run_pipeline.py`
- evaluation/ cũ replaced bởi `src/evaluation/`

**Action**:
```bash
rm -rf stock_ml/archive/src/
```

### 7. archive/visualization/ — **XÓA TOÀN BỘ**

**Size**: 127MB (lớn nhất)

**Nội dung**:
```
data/             # OHLCV duplicate
data_v27_v2/      # Old per-symbol JSON
```

**Lý do xóa**:
- 127MB là duplicate của OHLCV data đã có trong `portable_data/`
- `data_v27_v2/` là old version của visualization data, có thể regenerate qua `unified_export.py`
- Không có insight unique

**Action**:
```bash
rm -rf stock_ml/archive/visualization/
```

→ Tiết kiệm 127MB chỉ với 1 lệnh.

## Cleanup ngoài archive/

### Stock_ml root level

```
stock_ml/
├── analyze_v29_timing.py  10.5K   ← experiments/
├── analyze_v37a_vs_rule.py 2.0K   ← experiments/
├── _analyze_bigloss.py     3.6K   ← XÓA (underscore = throwaway)
├── _analyze_timing.py      3.8K   ← XÓA
├── _analyze_v37a_vs_rule.py 2.0K  ← XÓA
├── _simulate_v38.py        5.5K   ← XÓA
├── _v37a_*.csv             ~50K   ← XÓA (intermediate analysis)
├── visualize_v42.py        7.4K   ← visualization/scripts/
├── compare_rule_vs_model.py 11K   ← src/components/evaluation/
```

**Action plan**:
- 4 files `_analyze_*.py`, `_simulate_*.py` → XÓA
- `_v37a_*.csv` → XÓA (intermediate results)
- `analyze_v29_timing.py`, `analyze_v37a_vs_rule.py` → MOVE vào `analysis/` (đã có folder này)
- `visualize_v42.py` → MOVE vào `visualization/scripts/`
- `compare_rule_vs_model.py` → giữ tạm (pipeline mới sẽ thay thế)

### experiments/ folder

Hiện có 50+ scripts. Sau refactor:

**Giữ** (champion versions):
```
run_v22_final.py         # → giữ trong experiments/legacy/ làm reference
run_v34_final.py         # → giữ
run_v37a.py              # → giữ
run_v37b.py, run_v37c.py, run_v37d.py  # → giữ (champion v37d)
run_v39d.py              # → giữ
run_v42.py               # → giữ (v42_a champion)
```

**Move sang legacy/strategies/** (49 retired versions):
```
run_v23_optimal.py, run_v24.py, run_v25.py, run_v26.py, run_v27.py, run_v28.py
run_v29.py, run_v30.py, run_v31_final.py, run_v32_final.py, run_v33_final.py
run_v35*, run_v36*, run_v38*, run_v39a, v39a2, v39b, v39e, v39f, v39g
```

**XÓA** (one-off variants):
```
run_v22_final.py.bak (nếu có)
run_v26_experiments.py        # ablation, không champion
run_v28_experiments.py        # ablation
run_v29_experiments.py        # ablation
run_v29_retrain*.py           # retrain experiments
run_v29_round2.py             # round 2 experiment
run_v30_experiments.py        # experiments
run_v31.py (old)              # superseded by run_v31_final.py
run_v32.py (old)              # superseded by run_v32_final.py
run_v33.py, run_v33_phase2.py # phase experiments
run_v34.py (old)              # superseded by run_v34_final.py
run_v37*.py (variant scripts) # smaller variants
run_v38_combos.py             # combo experiments
run_v38b2.py, v38b3.py, v38c2.py, v38d2.py  # variants
run_v40.py, v41.py            # dashboard-only versions
run_feature_ablation.py       # ablation tool
analyze_v29_timing.py         # MOVE → analysis/
run_v29.py vs run_v29_retrain # duplicates
```

**Plus**:
```
_analyze_bigloss.py           ✗ XÓA
_analyze_timing.py            ✗ XÓA
_analyze_v37a_vs_rule.py      ✗ XÓA
_simulate_v38.py              ✗ XÓA
_v37a_*.csv                   ✗ XÓA
trades_*.csv (intermediate)   ✗ XÓA (kết quả ở results/)
trades_*.meta.json            ✗ XÓA (duplicate)
v28_timing_audit.csv          ✗ XÓA
v29_timing_audit.csv          ✗ XÓA
v32_timing_analysis.csv       ✗ XÓA
trades_hybrid_*.csv           ✗ XÓA (intermediate)
trades_tri_*.csv              ✗ XÓA (intermediate)
trades_rule.csv (here)        ✗ XÓA (chính ở results/)
```

**Sau cleanup**: experiments/ chỉ chứa ~10 file champion + a few utilities.

### docs/ folder

**Hiện**: 23 files, mix giữa report cũ và proposal mới.

**Đề xuất**:
- Giữ: PROJECT_DOCUMENTATION.md, ROOT_CAUSE_ANALYSIS.md, SMART_EXIT_PROPOSAL.md, MODEL_EVALUATION_REPORT.md, FIRST_TRAINING_REPORT.md
- Move sang `docs/legacy_reports/`: V*_DEEP_ANALYSIS.md, V*_VALIDATION.md (đã được sumarize)
- XÓA: `*_20260420.md` cũ nếu đã có report tổng hợp mới

### visualization/data_v*/ folders

**Hiện**: data_v22 → data_v49, mỗi folder ~10MB cho per-symbol JSON.

**Quyết định**:
- Sau refactor, dashboard mới regenerate từ `results/trades_*.csv`
- Giữ folder này tạm thời cho dashboard hoạt động
- Sau Phase 6 (dashboard compatible với pipeline mới) → cleanup

**Action sau Phase 6**:
```bash
# Chỉ giữ 11 champion + rule
cd stock_ml/visualization
ls data_*/ | grep -v -E "data_(v22|v32|v34|v35b|v37a|v37a_exit|v37d|v39d|v42_a|v19_3|rule)" | xargs -I{} rm -rf {}
```

## Cleanup execution plan

### Step 1: Backup trước khi xóa (1 commit)

```bash
cd stock_ml
git add -A
git commit -m "Pre-cleanup snapshot — full archive content"
git tag pre-cleanup-snapshot
```

→ Mọi thứ xóa sau bước này đều lấy lại được qua `git checkout pre-cleanup-snapshot`.

### Step 2: Tạo legacy/ structure (1 commit)

```bash
mkdir -p stock_ml/legacy/{scripts_reference,strategies,configs,docs}

# README giải thích
cat > stock_ml/legacy/README.md << 'EOF'
# Legacy code — pre-refactor reference

This folder contains:
- `scripts_reference/`: original `run_vXX_compare.py` scripts (read-only)
- `strategies/`: retired backtest functions (49 versions)
- `configs/`: frozen YAML entries from old `models.yaml`
- `docs/`: legacy analysis reports

## Status: read-only

Code here is not actively maintained. To re-run a legacy version:
```bash
python -m stock_ml run legacy/v25
```

## Promotion path

If a legacy version becomes important:
```bash
python -m stock_ml migrate-legacy v25
# → creates config/experiments/champions/v25.yaml
```

Then port any version-specific fusion logic into new architecture components.
EOF

git add stock_ml/legacy/
git commit -m "Create legacy/ structure"
```

### Step 3: Move scripts có config (1 commit)

```bash
cd stock_ml/archive/scripts/
mv run_v{10,11,12,13,14,15,16,17,18,19,19_1,19_2,19_3,19_4,20,21,22}_*.py ../../legacy/scripts_reference/
mv run_v21_experiment.py ../../legacy/scripts_reference/

cd ../../..
git add stock_ml/legacy/scripts_reference/
git rm -r stock_ml/archive/scripts/  # those moved
git commit -m "Move legacy scripts with model configs to legacy/"
```

### Step 4: Xóa archive/ folders không cần (1 commit)

```bash
rm -rf stock_ml/archive/analysis/
rm -rf stock_ml/archive/exports/
rm -rf stock_ml/archive/misc/
rm -rf stock_ml/archive/src/
rm -rf stock_ml/archive/visualization/

git rm -r stock_ml/archive/{analysis,exports,misc,src,visualization}
git commit -m "Remove archive folders that can be regenerated"
```

### Step 5: Nén results_legacy (1 commit)

```bash
cd stock_ml/archive/
tar czf results_legacy.tar.gz results_legacy/
rm -rf results_legacy/

cd ../..
git add stock_ml/archive/results_legacy.tar.gz
git rm -r stock_ml/archive/results_legacy/
git commit -m "Compress historical results"
```

### Step 6: Cleanup root + experiments/ (1 commit)

```bash
cd stock_ml/

# Root level
rm -f _analyze_*.py _simulate_*.py _v37a_*.csv

# experiments/ — xóa intermediate CSVs
rm -f experiments/_*.py experiments/_*.csv
rm -f experiments/trades_*.csv experiments/trades_*.meta.json
rm -f experiments/v*_timing_*.csv

# experiments/ — xóa one-off variants
rm -f experiments/run_v{26,28,29,30}_experiments.py
rm -f experiments/run_v29_retrain*.py experiments/run_v29_round2.py
rm -f experiments/run_v31.py experiments/run_v32.py experiments/run_v34.py  # superseded by *_final.py
rm -f experiments/run_v33_phase2.py
rm -f experiments/run_v40.py experiments/run_v41.py  # dashboard-only
rm -f experiments/run_v37{a,b,c,d}.py  # ⚠️ thực ra giữ champion v37a, v37d. Loại bỏ chỉ v37b, v37c

# Move analysis files
mkdir -p analysis/v29 analysis/v37
mv experiments/analyze_v29_timing.py analysis/v29/
mv experiments/_analyze_v37a_vs_rule.py analysis/v37/  # nếu chưa xóa

git add -A
git commit -m "Clean up root + experiments/ — remove intermediate files and one-off scripts"
```

### Step 7: Final tree verification (1 commit)

```bash
cd stock_ml/
find . -type f | wc -l   # đếm files trước/sau
du -sh .                  # đếm size

# Update README để reflect cấu trúc mới
# Update .gitignore nếu cần

git add -A
git commit -m "Update structure docs after cleanup"
```

## Trước khi cleanup — checklist

Bắt buộc xong trước:

- ✅ Golden baseline đã tạo (Phase 0.2)
- ✅ Git status clean (mọi thứ đã commit)
- ✅ Đã backup branch hiện tại: `git tag backup/pre-refactor-2026-04-26`
- ✅ Đã đọc lại CLEANUP_PLAN.md này
- ✅ Có disk space free > 1GB (cho safety)

## Sau cleanup — verification

Phải pass tất cả:

```bash
# 1. Champion versions vẫn chạy được
python run_pipeline.py --version v22 --device gpu

# 2. Dashboard vẫn mở được
# (open visualization/dashboard.html)

# 3. Test regression vẫn pass
pytest tests/regression/

# 4. Disk size giảm đáng kể
du -sh stock_ml/  # mục tiêu: <100MB (từ ~250MB)
```

Nếu có 1 fail → revert bằng `git checkout pre-cleanup-snapshot`.

## Rollback plan

Nếu cleanup gây vấn đề:

```bash
# Lấy lại 1 file cụ thể
git checkout pre-cleanup-snapshot -- path/to/file

# Lấy lại 1 folder
git checkout pre-cleanup-snapshot -- stock_ml/archive/visualization/

# Hoàn toàn revert
git reset --hard pre-cleanup-snapshot  # ⚠️ cẩn thận, mất các commit sau
```

## Sau khi cleanup — folder structure đích

```
stock_ml/
├── archive/
│   └── results_legacy.tar.gz       # nén ~5MB
├── config/                          # YAML configs
├── src/                             # Source code (active)
│   ├── components/                  # Refactor target
│   ├── pipeline/                    # Refactor target
│   ├── data/, features/, models/    # → migrate to components/
│   └── ...
├── experiments/                     # ~10 files (champion only)
├── legacy/                          # NEW — retired versions
│   ├── README.md
│   ├── scripts_reference/           # 17 historical scripts (~700KB)
│   ├── strategies/                  # 49 retired backtest functions
│   ├── configs/                     # frozen old YAMLs
│   └── adapter.py                   # bridge to new pipeline
├── tests/                           # NEW — regression + unit
├── docs/
│   ├── refactor/                    # ARCHITECTURE.md, etc.
│   └── legacy_reports/              # historical analysis
├── analysis/                        # ad-hoc analysis scripts
├── results/                         # backtest CSVs
├── visualization/                   # dashboard
├── scripts/                         # CLI entry points
├── pyproject.toml
├── requirements.txt
└── README.md
```

**Total size estimated**: ~30-40MB (giảm từ ~250MB).

## Anti-patterns cần tránh khi cleanup

❌ **Xóa rồi mới commit** — không có recovery point  
✅ **Commit trước khi xóa** — tag, recovery dễ

❌ **Xóa hàng loạt 1 commit** — khó tìm bug nếu có  
✅ **Mỗi loại xóa = 1 commit riêng**

❌ **Xóa trước khi có golden baseline**  
✅ **Phase 0.2 phải xong trước**

❌ **Xóa file mà không kiểm tra import**  
✅ **`grep -r "from xxx import"` trước khi xóa module**

❌ **Xóa với `rm -rf` mà không backup tag**  
✅ **`git tag pre-cleanup-snapshot` trước khi xóa**

## Lifecycle of legacy/ folder

```
Today:                          legacy/ chứa 49 retired versions, full code
6 tháng sau:                    Một số versions được promote → đẹp về 30 versions  
1 năm sau (nếu stable):         Compress legacy/strategies/ thành .tar.gz
2 năm sau:                      Xóa hẳn nếu không ai touch (vẫn có git history)
```

→ legacy không phải dump vĩnh viễn. Có lifecycle.
