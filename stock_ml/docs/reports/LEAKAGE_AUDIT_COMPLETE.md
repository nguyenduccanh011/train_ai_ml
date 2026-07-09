# Báo Cáo Audit Leakage HOÀN CHỈNH - Model Top 1 Leaderboard

**Ngày:** 2026-05-18  
**Model:** v22_exit_ablation_round25 (top 1 leaderboard cũ)  
**Trạng thái:** PHÁT HIỆN VÀ FIX 4 LEAKAGE

---

## 1. TỔNG QUAN 4 LEAKAGE

| # | Loại Leakage | Ảnh hưởng ước tính | Trạng thái |
|---|--------------|-------------------|-----------|
| 1 | **Pivot leak** (market_structure) | WR +8.5%, PF +5.7 | ✅ Fixed |
| 2 | **Boundary leak** (gap_days=0) | WR +0.1% | ✅ Fixed |
| 3 | **Target_sell không shift** | Chưa đo | ✅ Fixed |
| 4 | **Target buy không shift** (early_wave, early_wave_v2) | Chưa đo | ✅ Fixed |

---

## 2. CHI TIẾT TỪNG LEAKAGE

### 2.1. Pivot Leakage (Feature)

**Phát hiện:** `_market_structure()` trong `src/features/engine.py` dùng forward-looking để confirm pivot.

**Code lỗi:**
```python
# Line 442-507: src/features/engine.py
for i in range(order, n - order):
    if is_pivot_high(i, order):  # nhìn forward 'order' bars
        pivots_high[i] = 1
```

**Ảnh hưởng:**
- WR: 78.20% → 69.76% (-8.44%)
- PF: 15.16 → 9.50 (-5.66)

**Fix:** Đã sửa pivot detection thành backward-only (commit trước).

**Feature sets bị ảnh hưởng:**
- `leading_v2`, `leading_v3`, `leading_v4`, `leading_deriv`, `all_features`
- **610/886 models** trong leaderboard bị leak này

---

### 2.2. Boundary Leakage (Split)

**Phát hiện:** `gap_days=0` cho phép target tại row cuối train nhìn forward vào test set.

**Ví dụ:**
- Train end: 31/12/2020
- Target `forward_window=15` tại 31/12 → nhìn đến 15/01/2021 (test set)

**Ảnh hưởng:**
- WR: 69.76% → 69.64% (-0.12%)
- PF: 9.50 → 9.44 (-0.06)
- Ảnh hưởng nhỏ vì chỉ 6 boundary days/6 năm test

**Fix:**
- `src/data/splitter.py`: default `gap_days=25`
- `src/pipeline/config.py`: `SplitConfig.gap_days=25`
- `config/base.yaml`: `gap_days: 25`

**Models bị ảnh hưởng:** **TẤT CẢ 886 models** (100%)

---

### 2.3. Target_sell Không Shift (Target)

**Phát hiện:** `_early_exit_signal()` trong `src/data/target.py` line 392 không có `shift(-1)`.

**Code lỗi:**
```python
# Line 392: src/data/target.py (TRƯỚC KHI FIX)
df["target_sell"] = sell  # ❌ Không shift
```

**Logic sai:**
- `target_sell[i] = 1` nghĩa là "tại bar i nên exit"
- Model học từ thông tin "tại i, nhìn forward thấy giá giảm" → leak

**Logic đúng (sau fix):**
```python
# Line 396: src/data/target.py (SAU KHI FIX)
df["target_sell"] = pd.Series(sell, index=df.index).shift(-1)  # ✅ Shift
```
- `target_sell[i] = 1` nghĩa là "tại bar i+1 nên exit"
- Model dự đoán trước 1 bar

**Ảnh hưởng:** Chưa đo riêng (đang retrain).

**Models bị ảnh hưởng:** Tất cả models có `exit_model_enabled=true` (~80% leaderboard)

---

### 2.4. Target Buy Không Shift (Target)

**Phát hiện:** `_early_wave()` và `_early_wave_v2()` trong `src/data/target.py` không shift.

**Code lỗi:**
```python
# Line 278, 366: src/data/target.py (TRƯỚC KHI FIX)
df["target"] = targets  # ❌ Không shift
```

**Logic sai:**
- `target[i] = 1` nghĩa là "tại bar i là early wave"
- Model học từ thông tin "tại i, nhìn forward thấy giá tăng" → leak

**Logic đúng (sau fix):**
```python
# Line 280, 368: src/data/target.py (SAU KHI FIX)
df["target"] = pd.Series(targets, index=df.index).shift(-1)  # ✅ Shift
```
- `target[i] = 1` nghĩa là "tại bar i+1 là early wave"
- Model dự đoán trước 1 bar

**Ảnh hưởng:** Chưa đo riêng (đang retrain).

**Models bị ảnh hưởng:** Tất cả models dùng `early_wave` hoặc `early_wave_v2` target (~70% leaderboard)

**Lưu ý:** Component classes (`src/components/targets/early_wave.py`, `early_wave_v2.py`) ĐÃ CÓ shift đúng. Chỉ legacy `src/data/target.py` bị leak. Pipeline trainer dùng legacy path → tất cả models bị ảnh hưởng.

---

## 3. KẾT QUẢ RETRAIN (Đang chạy)

**Config:** v22_exit_ablation_round25 (top 1 cũ)  
**Fixes applied:**
1. ✅ Pivot fix
2. ✅ gap_days=25
3. ✅ target_sell shift
4. ✅ target buy shift

**Metrics dự kiến:**
- Leaderboard (4 leaks): WR 78.2%, PF 15.16
- Pivot fix only: WR 69.76%, PF 9.50
- **All fixes:** WR ~69.5%, PF ~9.4 (ước tính)

Target shift leak có thể gây thêm -0.2% WR (ước tính).

---

## 4. LEADERBOARD UPDATE

**Đã thực hiện:**
1. ✅ Mark superseded 610 models có pivot leak
2. ✅ Add boundary leak warnings cho 277 models active (gap_days < forward_window)
3. ✅ Add model fix vào leaderboard (rank #11/277)

**Top 5 mới (non-superseded):**
- Tất cả dùng feature_set `leading` (không có market_structure)
- **NHƯNG:** Vẫn có boundary leak (gap_days=0) và target shift leak
- Metrics của chúng cũng bị inflate, chỉ ít hơn models dùng leading_v2

**Kết luận:** **KHÔNG CÓ MODEL NÀO TRONG LEADERBOARD CŨ LÀ CLEAN** (tất cả đều có ít nhất 2-3 leakage).

---

## 5. FILES ĐÃ FIX

### Features (Pivot leak)
- `src/features/engine.py` lines 442-507
- `src/components/features/blocks/market_structure.py` lines 20-72

### Split (Boundary leak)
- `src/data/splitter.py` (default gap_days=25)
- `src/pipeline/config.py` (SplitConfig.gap_days=25)
- `config/base.yaml` (gap_days: 25)

### Target (Target shift leak)
- `src/data/target.py` line 280 (_early_wave shift)
- `src/data/target.py` line 368 (_early_wave_v2 shift)
- `src/data/target.py` line 396 (_early_exit_signal shift)

---

## 6. KHUYẾN NGHỊ

### 6.1. Khẩn cấp

1. **Đợi retrain hoàn tất** để có metrics chính xác với all fixes
2. **Retrain toàn bộ top 50** với 4 fixes
3. **Rebuild leaderboard** từ đầu với code đã fix

### 6.2. Trung hạn

4. **Audit các target types khác:**
   - `_return_classification` (line 194) — không shift
   - `_forward_risk_reward` (line 278 old) — không shift
   - `trend_regime` — đã có shift ✅

5. **Add CI tests:**
   - Unit test cho mỗi target type (verify shift)
   - Integration test: synthetic data với target đặc biệt, check không leak

6. **Code review checklist:**
   - Mọi `df["target"] =` phải có `.shift(-1)`
   - Mọi feature computation không được nhìn forward
   - `gap_days >= max(target_fw, exit_fw)`

### 6.3. Dài hạn

7. **Implement Purged K-Fold** (embargo period)
8. **Automated leakage detection** trong CI/CD
9. **Leakage audit report** cho mỗi experiment run

---

## 7. TIMELINE

- **2026-05-18 09:00:** Phát hiện pivot leak
- **2026-05-18 10:30:** Fix pivot, phát hiện boundary leak
- **2026-05-18 11:00:** Fix boundary leak (gap_days=25)
- **2026-05-18 13:00:** Phát hiện target_sell không shift
- **2026-05-18 13:15:** Phát hiện target buy không shift
- **2026-05-18 13:30:** Fix cả 2 target shift leaks
- **2026-05-18 13:40:** Retrain với all fixes (đang chạy)

---

## 8. PHỤ LỤC

**Scripts:**
- `retrain_top1_check_leakage.py` — retrain với fixes
- `update_leaderboard_with_fix.py` — mark superseded + add fix
- `add_boundary_warnings.py` — add boundary leak warnings
- `compare_3_versions.py` — so sánh metrics

**Outputs:**
- `results/leakage_check/top1_retrain_gap25_trades.csv`
- `results/leakage_check/top1_retrain_gap25_metrics.json`
- `results/leaderboard/leaderboard.json` (updated)

**Backups:**
- `results/leaderboard/leaderboard_pre_supersede_20260518_132838.json`
- `results/leaderboard/leaderboard_pre_boundary_warn_20260518_133644.json`
