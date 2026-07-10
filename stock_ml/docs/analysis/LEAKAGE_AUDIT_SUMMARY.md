# Leakage Audit - Summary for User

## Công việc đã hoàn thành

### 1. ✅ Audit toàn bộ pipeline (hoàn thành)

**Phát hiện**: 1 leakage NGHIÊM TRỌNG trong `_market_structure()` pivot detection

**Các component đã audit**:
- ✅ Target shift logic — OK
- ✅ Backtest execution timing — OK  
- ✅ `_leading_signals` — OK
- ✅ `_exhaustion_signals` — OK
- ✅ `_volatility_regime` — OK
- ✅ `_multi_timeframe` — OK
- ✅ `_accumulation_features` — OK
- ✅ `_heikin_ashi_features` — OK
- ✅ `_relative_strength` — OK
- ✅ `_liquidity_features` — OK
- 🚨 **`_market_structure` (pivot)** — LEAKAGE → ĐÃ FIX

### 2. ✅ Fix leakage (hoàn thành)

**Files đã sửa**:
1. `src/features/engine.py` lines 442-507
2. `src/components/features/blocks/market_structure.py` lines 20-72

**Logic mới**: Pivot tại bar `i-order` chỉ được confirm tại bar `i` (sau khi thấy `order` bars). Không còn forward-looking.

### 3. ✅ Phân tích tác động (hoàn thành)

**Models bị ảnh hưởng**:
- 303 / 886 models (34%) dùng `leading_v2/v3/v4/deriv`
- 80% top 50 leaderboard là leaky
- Top 1-19 đều là `v22_exit_ablation_round25` với `leading_v2`

**Metrics inflation** (leaky vs safe):
- WR: +1.34% (trung bình toàn bộ)
- PF: +0.38 (trung bình toàn bộ)
- **Top tier**: +3-4% WR, +1-2 PF (top 1-19 vs top 20+)

### 4. 🔄 Retrain round25 (đang chạy)

**Status**: Background task đang chạy
- 42 models cần retrain
- Mỗi model: 6 walk-forward windows
- Dự kiến: 30-60 phút

**Output**: `results/leaderboard/pivot_fix_impact_report.json`

---

## Files quan trọng đã tạo

1. **LEAKAGE_AUDIT_REPORT.md** — báo cáo audit đầy đủ
2. **fix_pivot_leakage.py** — reference implementation + test
3. **retrain_round25_with_fix.py** — script retrain + so sánh metrics
4. **config/experiments/matrix/v22_exit_ablation_round25_leakage_fix_test.yaml** — config test

---

## Kết quả dự kiến sau retrain

**Top 1 model** (5be8a6ec):
- **Trước fix**: WR 78.2%, PF 15.16, Score 662.1
- **Sau fix** (dự đoán): WR ~74-75%, PF ~12-13, Score ~620-630

**Lý do**: Model không còn học từ pivot features có future data.

---

## Bước tiếp theo (sau khi retrain xong)

1. **Xem impact report**: `results/leaderboard/pivot_fix_impact_report.json`
2. **So sánh leaderboard**: pre-fix vs post-fix
3. **Quyết định**:
   - Chấp nhận fix → retrain toàn bộ 303 leaky models
   - Rollback fix → giữ behavior cũ (không khuyến nghị)
   - Disable `_market_structure` → dùng `leading` thay `leading_v2`

---

## Monitoring retrain progress

Task ID: `bsao0f7pk`
Output: `C:\Users\DUCCAN~1\AppData\Local\Temp\claude\...\tasks\bsao0f7pk.output`

Bạn sẽ được thông báo khi retrain hoàn thành.
