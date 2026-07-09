# Kế Hoạch Thực Hiện Fix Leakage - Chi Tiết

**Ngày tạo:** 2026-05-18  
**Trạng thái:** Draft - Chờ phê duyệt  
**Ước tính thời gian:** 3-5 ngày (tùy batch size)

---

## 1. TỔNG QUAN

### 1.1. Vấn đề

Toàn bộ 888 models trong leaderboard hiện tại bị ảnh hưởng bởi 1-4 loại leakage:

| Loại Leakage | Số models bị ảnh hưởng | Ảnh hưởng ước tính | Trạng thái code |
|--------------|-------------------------|-------------------|-----------------|
| **Pivot leak** (market_structure forward-looking) | 317 models | WR -8.5%, PF -5.7 | ✅ Đã fix |
| **Boundary leak** (gap_days=0) | 888 models (100%) | WR -0.1% | ✅ Đã fix |
| **Target sell leak** (exit signal không shift) | 888 models (100%) | Chưa đo | ✅ Đã fix |
| **Target buy leak** (early_wave không shift) | 433 models | Chưa đo | ✅ Đã fix |

**Kết luận:** Code đã được fix đúng, nhưng TẤT CẢ models trong leaderboard đều được train bằng code cũ (leaky). Cần retrain toàn bộ.

### 1.2. Mục tiêu

1. **Retrain toàn bộ models** với code đã fix
2. **Rebuild leaderboard** với metrics chính xác
3. **Validate** không còn leakage
4. **Document** quy trình và kết quả

---

## 2. PHÂN TÍCH HIỆN TRẠNG

### 2.1. Leaderboard hiện tại

```
Tổng models: 888
├─ Có pivot leak (leading_v2/v3/v4/deriv/leading): 317
├─ Có boundary leak (gap_days=0): 888 (100%)
├─ Có target_sell leak (exit_model enabled): 888 (100%)
└─ Có target buy leak (early_wave/v2): 433
```

**Top 1 vn_stock (leaky):**
- Model: `v22_exit_ablation_round25_signals_target-earlyv2_fw15_g04_l02-exit_model-exit_fw20_l0345`
- WR: 77.9%, PF: 14.76, Composite: 655.7
- Leaks: pivot + boundary + target_sell + target_buy (4 leaks)

**Top 1 vn_derivatives (leaky):**
- Model: `deriv_p259_30m_exit_model_algo_micro`
- WR: 69.46%, PF: 11.55, Composite: 268.2
- Leaks: boundary + target_sell (2 leaks)

### 2.2. Code đã fix

**Files đã sửa:**

1. **Pivot leak** - [src/features/engine.py:442-507](src/features/engine.py#L442-L507)
   - Đã chuyển pivot detection từ forward-looking sang backward-only

2. **Pivot leak** - [src/components/features/blocks/market_structure.py:20-72](src/components/features/blocks/market_structure.py#L20-L72)
   - Component class cũng đã fix

3. **Boundary leak** - [src/data/splitter.py](src/data/splitter.py)
   - Default `gap_days=25` (thay vì 0)

4. **Boundary leak** - [src/pipeline/config.py](src/pipeline/config.py)
   - `SplitConfig.gap_days=25`

5. **Boundary leak** - [config/base.yaml](config/base.yaml)
   - `gap_days: 25`

6. **Target buy leak** - [src/data/target.py:279](src/data/target.py#L279)
   - `_early_wave()` đã có `.shift(-1)`

7. **Target buy leak** - [src/data/target.py:368](src/data/target.py#L368)
   - `_early_wave_v2()` đã có `.shift(-1)`

8. **Target sell leak** - [src/data/target.py:398](src/data/target.py#L398)
   - `_early_exit_signal()` đã có `.shift(-1)`

**Xác nhận:** Đã kiểm tra code, tất cả fixes đều đúng.

### 2.3. Scripts có sẵn

1. **[batch_retrain_leaky_models.py](batch_retrain_leaky_models.py)** - Retrain hàng loạt
2. **[cleanup_leaky_leaderboard.py](cleanup_leaky_leaderboard.py)** - Xóa entries cũ
3. **[retrain_top1_check_leakage.py](retrain_top1_check_leakage.py)** - Retrain model đơn lẻ
4. **[compare_3_versions.py](compare_3_versions.py)** - So sánh metrics
5. **[tests/signals/test_leakage.py](tests/signals/test_leakage.py)** - Leakage tests

---

## 3. KẾ HOẠCH THỰC HIỆN

### Phase 1: Validation & Pilot (Ngày 1)

**Mục tiêu:** Xác nhận fix hoạt động đúng trước khi retrain hàng loạt

#### 3.1.1. Chạy leakage tests

```bash
cd stock_ml
pytest tests/signals/test_leakage.py -v
```

**Kỳ vọng:** Tất cả tests pass

**Nếu fail:**
- Kiểm tra lại code fix
- Fix bugs
- Re-run tests

#### 3.1.2. Retrain top 5 models (verify)

```bash
python batch_retrain_leaky_models.py --verify-only --device cpu
```

**Output:**
- `results/leakage_check/batch_retrain_log.json`
- `results/experiments/{bundle}/{run_name}/trades.csv` (mới)
- `results/experiments/{bundle}/{run_name}/metrics.json` (mới)

**Kiểm tra:**
1. WR giảm ~8-9% so với leaderboard (do pivot leak fix)
2. Trades tăng ~6-7x (do boundary leak fix)
3. PF giảm ~5-6 (do pivot leak fix)
4. Không có errors trong log

**Nếu metrics không khớp pattern:**
- Review code fix
- Check config overrides
- Debug với 1 model đơn lẻ

#### 3.1.3. So sánh metrics

```bash
python compare_3_versions.py
```

**Kỳ vọng:**
- Leaderboard (leaky): WR ~78%, PF ~15
- Retrained (fixed): WR ~69%, PF ~9
- Delta: WR -8-9%, PF -5-6

**Decision point:** Nếu validation pass → tiếp tục Phase 2. Nếu fail → dừng và debug.

---

### Phase 2: Batch Retrain (Ngày 2-4)

**Mục tiêu:** Retrain toàn bộ 888 models với code đã fix

#### 3.2.1. Ước tính thời gian

**Giả định:**
- Avg time per model: 60s (từ batch_retrain_leaky_models.py log)
- Total models: 888
- Total time: 888 × 60s = 14.8 giờ

**Chiến lược:**
- Chạy batch 200 models/lần (3.3 giờ/batch)
- 5 batches × 3.3 giờ = 16.5 giờ
- Dự phòng: 20 giờ (2.5 ngày)

#### 3.2.2. Batch 1: Top 200 models

```bash
python batch_retrain_leaky_models.py \
  --batch-size 200 \
  --device cpu \
  --skip-existing \
  --skip-cutoff 2026-05-18T08:00:00
```

**Monitoring:**
```bash
# Terminal 2: Monitor progress
watch -n 60 'tail -20 results/leakage_check/batch_retrain_log.json'
```

**Output:**
- `results/leakage_check/batch_retrain_log.json` (incremental)
- `results/experiments/{bundle}/{run_name}/` (artifacts mới)

**Kiểm tra sau mỗi batch:**
1. Success rate > 95%
2. Failed models < 10
3. Avg time per model < 90s

**Nếu fail rate > 5%:**
- Review error logs
- Fix common issues
- Re-run failed models

#### 3.2.3. Batch 2-5: Remaining models

Lặp lại 3.2.2 cho:
- Batch 2: models 201-400
- Batch 3: models 401-600
- Batch 4: models 601-800
- Batch 5: models 801-888

**Parallel option (nếu có nhiều CPU):**
```bash
# Split models thành 4 groups, chạy song song
python batch_retrain_leaky_models.py --batch-size 222 --device cpu &
python batch_retrain_leaky_models.py --batch-size 222 --device cpu &
python batch_retrain_leaky_models.py --batch-size 222 --device cpu &
python batch_retrain_leaky_models.py --batch-size 222 --device cpu &
```

**Lưu ý:** Cần đủ RAM (ước tính 4GB/process × 4 = 16GB)

---

### Phase 3: Rebuild Leaderboard (Ngày 5)

**Mục tiêu:** Tạo leaderboard mới từ models đã retrain

#### 3.3.1. Backup leaderboard cũ

```bash
cd stock_ml
cp results/leaderboard/leaderboard.csv \
   results/leaderboard/leaderboard_pre_fix_backup_$(date +%Y%m%d_%H%M%S).csv
cp results/leaderboard/leaderboard.json \
   results/leaderboard/leaderboard_pre_fix_backup_$(date +%Y%m%d_%H%M%S).json
```

#### 3.3.2. Rebuild leaderboard

```bash
python -c "
from src.leaderboard import rebuild_leaderboard
from pathlib import Path
ROOT = Path('.')
experiments_dir = ROOT / 'results/experiments'
output_dir = ROOT / 'results/leaderboard'
rows = rebuild_leaderboard(experiments_dir, output_dir)
print(f'Rebuilt: {len(rows)} models')
"
```

**Output:**
- `results/leaderboard/leaderboard.csv` (mới)
- `results/leaderboard/leaderboard.json` (mới)

#### 3.3.3. Validate leaderboard mới

```bash
python -c "
import pandas as pd
df = pd.read_csv('results/leaderboard/leaderboard.csv')
print(f'Total models: {len(df)}')
print(f'Top 1 WR: {df.iloc[0][\"wr\"]}')
print(f'Top 1 PF: {df.iloc[0][\"pf\"]}')
print(f'Top 1 composite: {df.iloc[0][\"composite_score\"]}')
print()
print('Top 5:')
print(df[['run_name','wr','pf','trades','composite_score']].head())
"
```

**Kỳ vọng:**
- Total models: 888
- Top 1 WR: ~69-70% (giảm từ 78%)
- Top 1 PF: ~9-10 (giảm từ 15)
- Top 1 composite: ~450-500 (giảm từ 655)

**Nếu metrics vẫn cao (WR > 75%):**
- Kiểm tra lại code fix có được apply không
- Verify config overrides
- Re-run một vài models với debug mode

---

### Phase 4: Validation & Documentation (Ngày 5)

**Mục tiêu:** Xác nhận không còn leakage và document kết quả

#### 3.4.1. Chạy lại leakage tests với models mới

```bash
# Test với top 1 model mới
python -c "
import pandas as pd
df = pd.read_csv('results/leaderboard/leaderboard.csv')
top1 = df.iloc[0]
print(f'Testing: {top1[\"run_name\"]}')
print(f'Bundle: {top1[\"bundle\"]}')
# Load config và verify gap_days=25
import yaml
from pathlib import Path
config_path = Path('results/experiments') / top1['bundle'] / top1['run_name'] / 'config.resolved.yaml'
cfg = yaml.safe_load(config_path.read_text())
print(f'gap_days: {cfg[\"split\"][\"gap_days\"]}')
assert cfg['split']['gap_days'] == 25, 'gap_days not fixed!'
print('✓ gap_days verified')
"
```

#### 3.4.2. So sánh metrics trước/sau

```bash
python -c "
import pandas as pd
old = pd.read_csv('results/leaderboard/leaderboard_pre_fix_backup_*.csv')
new = pd.read_csv('results/leaderboard/leaderboard.csv')
print('=== METRICS COMPARISON ===')
print(f'Old top 1: WR={old.iloc[0][\"wr\"]:.2f}% PF={old.iloc[0][\"pf\"]:.2f}')
print(f'New top 1: WR={new.iloc[0][\"wr\"]:.2f}% PF={new.iloc[0][\"pf\"]:.2f}')
print(f'Delta: WR={new.iloc[0][\"wr\"]-old.iloc[0][\"wr\"]:.2f}% PF={new.iloc[0][\"pf\"]-old.iloc[0][\"pf\"]:.2f}')
"
```

#### 3.4.3. Update documentation

Tạo file `LEAKAGE_FIX_RESULTS.md`:

```markdown
# Kết Quả Fix Leakage

**Ngày hoàn thành:** 2026-05-XX
**Models retrained:** 888/888

## Metrics Comparison

| Metric | Old (leaky) | New (fixed) | Delta |
|--------|-------------|-------------|-------|
| Top 1 WR | 77.9% | XX.X% | -X.X% |
| Top 1 PF | 14.76 | X.XX | -X.XX |
| Top 1 Trades | 1,231 | X,XXX | +X,XXX |
| Top 1 Composite | 655.7 | XXX.X | -XXX.X |

## Leakage Fixes Applied

1. ✅ Pivot leak - market_structure backward-only
2. ✅ Boundary leak - gap_days=25
3. ✅ Target sell leak - shift(-1)
4. ✅ Target buy leak - shift(-1)

## Validation

- ✅ All leakage tests pass
- ✅ Config verified (gap_days=25)
- ✅ Metrics match expected pattern
- ✅ No errors in retrain logs

## Next Steps

1. Deploy top 1 model to production
2. Monitor live performance
3. Compare live vs backtest metrics
```

---

## 4. RISK MANAGEMENT

### 4.1. Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Retrain fails (bugs) | Medium | High | Pilot với 5 models trước |
| Metrics không khớp pattern | Low | High | Validate với synthetic data |
| Hết disk space | Low | Medium | Monitor disk usage, cleanup old artifacts |
| Retrain quá lâu | Medium | Low | Parallel batches, optimize code |
| Models mới kém hơn expected | Low | Medium | Review code fix, re-audit |

### 4.2. Rollback Plan

**Nếu cần rollback:**

1. Restore leaderboard cũ:
```bash
cp results/leaderboard/leaderboard_pre_fix_backup_*.csv \
   results/leaderboard/leaderboard.csv
```

2. Xóa artifacts mới:
```bash
# Backup trước khi xóa
mv results/experiments results/experiments_fixed_backup
# Restore từ backup cũ (nếu có)
```

3. Revert code changes:
```bash
git checkout HEAD~1 src/data/target.py
git checkout HEAD~1 src/features/engine.py
# ... (revert các files khác)
```

**Lưu ý:** Chỉ rollback nếu phát hiện bug nghiêm trọng trong code fix. Metrics thấp hơn là expected behavior.

---

## 5. SUCCESS CRITERIA

### 5.1. Phase 1 (Validation)

- ✅ All leakage tests pass
- ✅ Top 5 models retrained successfully
- ✅ Metrics match expected pattern (WR -8%, PF -5)

### 5.2. Phase 2 (Batch Retrain)

- ✅ 888/888 models retrained
- ✅ Success rate > 95%
- ✅ No critical errors

### 5.3. Phase 3 (Rebuild)

- ✅ Leaderboard rebuilt with 888 models
- ✅ Top 1 metrics: WR ~69%, PF ~9
- ✅ No warnings in leaderboard

### 5.4. Phase 4 (Validation)

- ✅ Config verified (gap_days=25)
- ✅ Leakage tests pass với models mới
- ✅ Documentation complete

---

## 6. TIMELINE & RESOURCES

### 6.1. Timeline

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| Phase 1: Validation | 4 giờ | Ngày 1 sáng | Ngày 1 chiều |
| Phase 2: Batch Retrain | 20 giờ | Ngày 2 sáng | Ngày 4 chiều |
| Phase 3: Rebuild | 2 giờ | Ngày 5 sáng | Ngày 5 sáng |
| Phase 4: Validation | 2 giờ | Ngày 5 chiều | Ngày 5 chiều |
| **Total** | **28 giờ** | **Ngày 1** | **Ngày 5** |

### 6.2. Resources

**Compute:**
- CPU: 8+ cores (recommended)
- RAM: 16GB+ (for parallel batches)
- Disk: 50GB free space (for artifacts)

**Personnel:**
- 1 ML engineer (full-time, 5 ngày)
- 1 reviewer (part-time, 2 giờ)

---

## 7. POST-IMPLEMENTATION

### 7.1. Monitoring

**Tuần 1 sau deploy:**
- Monitor live trading metrics
- Compare live vs backtest WR/PF
- Track drawdown

**Nếu live metrics thấp hơn backtest > 5%:**
- Audit execution logic
- Check slippage/commission
- Review signal timing

### 7.2. Future Improvements

1. **Automated leakage detection:**
   - Add CI/CD checks
   - Synthetic data tests
   - Pre-commit hooks

2. **Purged K-Fold:**
   - Implement embargo period
   - Remove overlapping train/test rows

3. **Feature audit:**
   - Review `_volatility_regime`
   - Review `_leading_signals`
   - Review `_exhaustion_signals`

4. **Target audit:**
   - Review `_return_classification`
   - Review `_forward_risk_reward`
   - Add shift validation

---

## 8. APPENDIX

### 8.1. Commands Reference

```bash
# Validation
pytest tests/signals/test_leakage.py -v
python batch_retrain_leaky_models.py --verify-only

# Batch retrain
python batch_retrain_leaky_models.py --batch-size 200 --device cpu

# Rebuild leaderboard
python -c "from src.leaderboard import rebuild_leaderboard; ..."

# Monitor progress
tail -f results/leakage_check/batch_retrain_log.json
```

### 8.2. File Locations

```
stock_ml/
├── src/
│   ├── data/
│   │   ├── target.py (target fixes)
│   │   └── splitter.py (gap_days fix)
│   ├── features/
│   │   └── engine.py (pivot fix)
│   └── pipeline/
│       └── config.py (gap_days default)
├── config/
│   └── base.yaml (gap_days config)
├── tests/
│   └── signals/
│       └── test_leakage.py (validation tests)
├── results/
│   ├── leaderboard/
│   │   ├── leaderboard.csv (main leaderboard)
│   │   └── leaderboard.json
│   ├── leakage_check/
│   │   └── batch_retrain_log.json (progress log)
│   └── experiments/
│       └── {bundle}/{run_name}/ (model artifacts)
├── batch_retrain_leaky_models.py (main script)
├── cleanup_leaky_leaderboard.py (cleanup script)
└── LEAKAGE_FIX_IMPLEMENTATION_PLAN.md (this file)
```

### 8.3. Contact

**Questions/Issues:**
- Review audit reports: `LEAKAGE_AUDIT_FINAL.md`, `LEAKAGE_AUDIT_COMPLETE.md`
- Check test results: `pytest tests/signals/test_leakage.py -v`
- Review code fixes: git log src/data/target.py src/features/engine.py

---

**Phê duyệt:**
- [ ] ML Lead review
- [ ] Code review (fixes)
- [ ] Resource allocation approved
- [ ] Timeline approved

**Bắt đầu thực hiện:** Sau khi tất cả checkboxes được tick ✅
