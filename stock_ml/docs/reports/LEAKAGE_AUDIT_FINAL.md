# Bao cao Audit Leakage CUOI CUNG - Model Top 1 Leaderboard

**Ngay:** 2026-05-18  
**Model:** v22_exit_ablation_round25 (top 1 leaderboard)  
**Trang thai:** PHAT HIEN VA FIX 2 LEAKAGE

---

## 1. TONG QUAN 3 VERSIONS

| Version | WR | PF | Trades | Total PnL | MaxDD |
|---------|----|----|--------|-----------|-------|
| **Leaderboard (leaky)** | 78.20% | 15.16 | 1,225 | 15,977 | 149.86 |
| **Pivot fix (gap=0)** | 69.76% | 9.50 | 7,770 | 96,366 | 1525.58 |
| **Pivot + boundary fix (gap=25)** | 69.64% | 9.44 | 7,764 | 96,131 | 1531.11 |

---

## 2. PHAT HIEN

### 2.1. Pivot Leakage (DA FIX)

**Anh huong:** WR -8.56%, PF -5.72

Pivot detection trong `_market_structure()` truoc day nhin forward `order` bars
de confirm pivot. Day la forward-looking leak nghiem trong.

**Files da fix:**
- `src/features/engine.py` lines 442-507
- `src/components/features/blocks/market_structure.py` lines 20-72

### 2.2. Boundary Leakage (DA FIX)

**Anh huong:** WR -0.12%, PF -0.06 (rat nho)

Truoc day `gap_days=0` cho phep target tai row cuoi train (forward_window=15-20)
nhin vao test set.

**Files da fix:**
- `src/data/splitter.py`: default `gap_days=25`
- `src/data/splitter.py` `from_config`: default 25
- `src/pipeline/config.py` `SplitConfig`: default 25
- `config/base.yaml`: `gap_days: 25`

**Note:** Anh huong nho hon du doan vi:
- Walk-forward windows it (6 nam test) -> chi 6 boundary days bi leak
- Target shift(-1) da loai bo NaN o cuoi train -> mot phan leak da bi cat

### 2.3. Target Shift Logic (OK)

Logic `shift(-1)` trong target generation hoat dong dung. Khong co leak.

---

## 3. KET LUAN

**Model top 1 KHONG dat duoc 78.2% WR / 15.16 PF.**

Hieu suat thuc te (sau khi fix het leak):
- **WR: 69.64%**
- **PF: 9.44**
- **Trades: 7,764**

**Day van la metrics tot, nhung thap hon nhieu so voi leaderboard.**

Pivot leakage la nguyen nhan chinh (~8.5% WR inflation), boundary leak chi
gop ~0.1% WR.

---

## 4. KHUYEN NGHI TIEP THEO

### 4.1. Khan cap

1. **Mark superseded toan bo leaderboard cu**
   - 303/886 models dung `leading_v2/v3/v4/deriv` co pivot leak
   - 80% top 50 leaderboard la leaky

2. **Retrain toan bo top 50** voi:
   - Pivot fix (da co)
   - `gap_days=25` (da co)

3. **Update build_leaderboard.py**:
   - Dam bao chi tinh metrics tren models post-fix
   - Add validation: warn neu config co `gap_days < forward_window`

### 4.2. Trung han

4. **Audit cac feature blocks khac:**
   - `_volatility_regime`
   - `_leading_signals`
   - `_exhaustion_signals`
   - Cac blocks rieng le trong `src/components/features/blocks/`

5. **Add CI tests cho leakage:**
   - Synthetic data test: dat target dac biet va check no future leak
   - Unit test cho moi feature block (assert no forward access)

### 4.3. Dai han

6. **Implement Purged K-Fold:**
   - Ngoai gap_days, them embargo period
   - Loai bo train rows ma target overlap voi test rows

---

## 5. FILES LIEN QUAN

**Code da fix:**
- `src/features/engine.py` (pivot)
- `src/components/features/blocks/market_structure.py` (pivot)
- `src/data/splitter.py` (gap_days default)
- `src/pipeline/config.py` (gap_days default)
- `config/base.yaml` (gap_days)

**Outputs retrain:**
- `results/leakage_check/top1_retrain_trades.csv` (gap=25, 7764 trades)
- `results/leakage_check/top1_retrain_metrics.json`

**Scripts:**
- `retrain_top1_check_leakage.py`
- `compare_3_versions.py`

**Configs:**
- `config/experiments/matrix/v22_exit_ablation_round25.yaml`
- `config/experiments/matrix/v22_exit_ablation_round25_leakage_fix_test.yaml`
