# Phân tích điểm mù Model Top 1 Phase 90 - VN30F1M 1H

## Tổng quan Model Top 1

**Model**: `deriv_p90_1hlqr` (phase 90)
- **Composite Score**: 167.6 (rank #1)
- **Timeframe**: 1H
- **Entry Model**: CatBoost
- **Exit Model**: LightGBM (fw=48 bars, loss_threshold=0.0485)
- **Target**: early_wave_v2 (fw=102, gain=0.038, loss=0.026)
- **Strategy**: V22 với dist-params-s0 fusion
- **Performance**: 92 trades, WR 88.04%, avg PnL 5.88%, Sharpe 0.936

---

## Phát hiện chính: Vấn đề KHÔNG phải thiếu entry signals

### Các sóng tăng được cho là "bỏ lỡ":

| Wave | Khoảng thời gian | Điểm tăng | Trades thực tế | PnL thực tế |
|------|------------------|-----------|----------------|-------------|
| Wave 1 | 17/9/24 → 3/10/24 | +90 điểm | **2 trades** (25/9, 26/9) | **-0.69%, -1.66%** |
| Wave 2 | 20/11/24 → 27/12/24 | +100 điểm | **1 trade** (20/11) | **+4.70%** |
| Wave 3 | 1/8/25 → 5/9/25 | +250 điểm | **2 trades** (1/8, 14/8) | **+18.06%, +6.43%** |

**Kết luận**: Model ĐÃ CÓ entry signals trong các sóng này, nhưng:
1. **Wave 1**: Entry muộn (8 ngày sau khi sóng bắt đầu) và exit quá sớm → PnL âm
2. **Wave 2**: Entry đúng ngày đầu sóng nhưng exit sớm (30/12 do year-boundary)
3. **Wave 3**: Entry đúng ngày đầu sóng nhưng exit sớm → chỉ capture 5.7% của 316% max profit

---

## Vấn đề thực sự: Exit Model quá aggressive

### Thống kê profit capture:

```
Trades capturing <10% of max profit: 73 / 82 (89%)
Mean max_profit: 265% (weak trend), 85% (strong trend)
Mean realized PnL: 5.88%
```

### Top 10 trades bỏ lỡ profit nhiều nhất:

| Entry Date | Realized PnL | Max Profit | Capture % | Hold Days | Exit Reason | Trend |
|------------|--------------|------------|-----------|-----------|-------------|-------|
| 2023-03-01 | 19.45% | 317.75% | 6.1% | 519 | peak_protect_dist | weak |
| 2025-08-01 | 18.06% | 316.48% | 5.7% | 111 | exit_model | weak |
| 2021-03-25 | 12.74% | 311.04% | 4.1% | 87 | exit_model | weak |
| 2023-11-01 | 9.71% | 309.85% | 3.1% | 46 | exit_model | weak |
| 2025-11-11 | 8.81% | 309.00% | 2.9% | 83 | exit_model | weak |

**Pattern**: Weak trend entries có max_profit trung bình 265% nhưng chỉ realize 5.74%

---

## Phân tích 3 vấn đề chính

### 1. **Entry Timing - Gap dài giữa các trades**

**Thống kê**:
- 27 gaps > 30 ngày trong 92 trades
- Gap dài nhất: 139 ngày (4/2023 → 8/2023)
- Gap trước Wave 1: 104 ngày (6/2024 → 9/2024)

**Nguyên nhân**:
- Model chờ đợi điều kiện entry "hoàn hảo" (score >= 3-4)
- Không có pullback re-entry mechanism
- Sau khi exit, model cần thời gian để ML signal "warm up" lại

**Ví dụ Wave 1**:
```
Trade cuối trước Wave 1: 12/6/2024
  - Exit: 23/7/2024 (PnL: -2.44%, exit_model)
  - Features: trend=strong, ret5d=1.97%, dist_sma20=1.72%, wp=0.9615
  
Gap: 64 ngày (23/7 → 25/9)

Trade đầu Wave 1: 25/9/2024
  - Features: trend=strong, ret5d=1.52%, dist_sma20=1.13%, wp=0.9706
  - Vào muộn 8 ngày sau khi sóng bắt đầu (17/9)
```

---

### 2. **Exit Model - Cắt lỗ winners quá sớm**

**Thống kê exit reasons**:
```
exit_model:           80 trades (87%) - avg_pnl 5.69%
end (year boundary):  10 trades (11%) - avg_pnl 3.23%
peak_protect_dist:     2 trades (2%)  - avg_pnl 19.27%
```

**Vấn đề**:
- Exit model (LightGBM, fw=48 bars, loss_threshold=0.0485) fire quá sớm
- Không có trend continuation protection
- Weak trend entries có max_profit cao nhất (265%) nhưng exit sớm nhất

**Ví dụ Wave 1 Trade #74**:
```
Entry: 25/9/2024 03:00
Exit:  6/11/2024 06:00 (exit_model)
Hold: 152 days
PnL: -0.69%
Max profit: 102.31%

→ Exit ngay khi giá bắt đầu pullback, không cho phép position "breathe"
```

---

### 3. **Strong Trend Paradox**

**Thống kê**:
```
Strong trend trades: 23 trades
  - Win rate: 78.3% (THẤP NHẤT)
  - Mean PnL: 6.04%
  - Mean max_profit: 84.95%
  - Entry features: ret5d=1.01%, dist_sma20=1.83%, wp=0.937

Weak trend trades: 47 trades
  - Win rate: 89.4% (CAO NHẤT)
  - Mean PnL: 5.74%
  - Mean max_profit: 265.68% (GẤP 3 LẦN)
  - Entry features: ret5d=-0.25%, dist_sma20=-1.98%, wp=0.270
```

**Paradox**: 
- Strong trend entries có win rate THẤP HƠN weak trend
- Strong trend entries có max_profit THẤP HƠN weak trend (85% vs 265%)
- Strong trend entries thường vào muộn (wp=0.937 = gần đỉnh range)

**Nguyên nhân**:
1. **Entry quá muộn**: Strong trend được detect khi giá đã chạy xa (wp > 0.9)
2. **Exit quá sớm**: Sau khi vào gần đỉnh, pullback nhỏ trigger exit_model
3. **Weak trend entries tốt hơn**: Vào khi giá thấp (wp=0.27), có nhiều upside hơn

**Ví dụ losing trades trong Wave 1**:
```
Trade #74 (25/9): trend=strong, wp=0.9706 → PnL -0.69%
Trade #77 (26/9): trend=strong, wp=0.9413 → PnL -1.66%

Cả 2 trades đều:
- Vào gần đỉnh range (wp > 0.94)
- ret5d > 1.4% (đã "nóng")
- Exit khi pullback nhỏ → không capture được continuation
```

---

## Hot Ret5d Filter - KHÔNG phải nguyên nhân chính

**Thống kê**:
```
Trades với ret_5d > 6%:  48 trades (52%) - Mean PnL 6.02%, WR 79.2%
Trades với ret_5d 0-6%:   2 trades (2%)  - Mean PnL 9.19%, WR 100%
Trades với ret_5d < 0%:  42 trades (46%) - Mean PnL 5.56%, WR 97.6%
```

**Kết luận**: 
- Model KHÔNG block entry khi ret5d cao
- 52% trades có ret5d > 6% và vẫn profitable
- Hot ret5d filter (nếu có) không phải bottleneck chính

---

## Losing Trades Analysis

**Tổng số**: 11 / 92 trades (12%)

**Pattern chung**:
1. **Strong trend losses** (5 trades):
   - Entry gần đỉnh (wp > 0.94)
   - ret5d > 1% (đã nóng)
   - Exit nhanh khi pullback (1-2 days hold)
   - Max profit 100-234% nhưng realize âm

2. **Weak trend losses** (4 trades):
   - Entry khi giá thấp (wp < 0.35)
   - dist_sma20 < -2% (xa dưới SMA20)
   - Exit nhanh (1-2 days hold)
   - Max profit 233-236% nhưng realize âm

3. **Year-boundary losses** (2 trades):
   - Forced exit 30/12 do walk-forward split
   - Không phải lỗi của model

**Insight**: Cả strong và weak trend đều có thể loss nếu exit quá sớm

---

## Đề xuất cải tiến

### Priority 1: Exit Optimization (ảnh hưởng lớn nhất)

#### A. Trend Continuation Mode

**Mục tiêu**: Giữ position trong uptrend mạnh thay vì exit sớm

**Config**:
```yaml
params:
  trend_continuation_enabled: true
  trend_continuation_min_pnl: 0.05      # 5%
  trend_continuation_min_hold: 20       # bars
  trend_continuation_exit_relax: true
```

**Logic**:
```python
# Trong exit_model_exit.py
if current_pnl > 0.05 and hold_bars > 20:
    if trend == "strong" and close > sma20 and macd > 0:
        # Disable exit_model, chỉ dùng trailing stop
        return FusionResult(action="pass", reason="trend_continuation")
```

**Expected impact**: 
- Tăng avg_pnl từ 5.88% → 10-15%
- Giảm số trades nhưng tăng quality
- Capture được 20-30% thay vì 5-10% của max profit

---

#### B. Adaptive Exit Model Hold Time

**Vấn đề hiện tại**: Exit model có thể fire ngay sau min_hold (không rõ config hiện tại)

**Cải tiến**:
```python
# Dynamic min_hold based on entry context
if entry_trend == "strong" and entry_score >= 3:
    min_hold = 12  # ~1.5 trading days
elif entry_trend == "strong":
    min_hold = 8
elif entry_trend == "moderate":
    min_hold = 6
else:
    min_hold = 3

# Exit inhibition trong strong trend
if hold_bars >= min_hold and exit_signal == 1:
    if trend == "strong" and current_pnl > 0.02:
        if exit_confidence < 0.7:
            return FusionResult(action="pass", reason="trend_override")
```

**Expected impact**: Giảm quick exits, tăng avg_hold từ 89 bars → 120-150 bars

---

#### C. Weak Trend Exit Relaxation

**Insight**: Weak trend entries có max_profit cao nhất (265%) nhưng exit sớm nhất

**Cải tiến**:
```python
# Weak trend entries cần hold lâu hơn
if entry_trend == "weak" and current_pnl > 0.05:
    # Relax exit threshold
    exit_threshold = exit_threshold * 1.5
    
    # Hoặc: chỉ exit khi trend chuyển sang "strong" (reversal)
    if current_trend != "strong":
        return FusionResult(action="pass", reason="weak_trend_hold")
```

**Expected impact**: Capture 15-20% thay vì 5-6% của max profit trong weak trend entries

---

### Priority 2: Entry Timing (ảnh hưởng trung bình)

#### D. Pullback Re-Entry Strategy

**Mục tiêu**: Sau khi exit trong uptrend, cho phép re-entry nhanh khi pullback

**Logic**:
```python
# Thêm vào entry_state
last_exit_in_uptrend: bool
last_exit_price: float
last_exit_bar: int

# Trong v19_entry_cascade
if last_exit_in_uptrend and bars_since_exit <= 10:
    pullback_depth = (last_exit_price - close[i]) / last_exit_price
    if 0.02 < pullback_depth < 0.05:  # 2-5% pullback
        if close[i] > sma20[i] and macd_line[i] > 0:
            new_position = 1
            counters["n_pullback_reentry"] += 1
```

**Expected impact**: Giảm gap time từ 64 ngày → 20-30 ngày

---

#### E. Strong Trend Entry Timing Fix

**Vấn đề**: Strong trend entries vào quá muộn (wp > 0.9)

**Cải tiến**:
```python
# Relax entry conditions cho strong trend EARLY
if trend == "strong" and sma20 > sma50:
    # Allow entry khi giá vừa breakout SMA20
    if close > sma20 and 0.5 < wp < 0.7:  # Mid-range, not peak
        if ret_5d < 0.08:  # Chưa quá nóng
            entry_score += 1  # Boost score
```

**Expected impact**: 
- Giảm wp trung bình của strong entries từ 0.937 → 0.7-0.8
- Tăng win rate của strong entries từ 78% → 85%+

---

### Priority 3: Year Boundary Fix (ảnh hưởng nhỏ)

#### F. Position Carry-Over

**Vấn đề**: Walk-forward split force close tất cả positions vào 30/12

**Cải tiến**:
```yaml
split:
  method: walk_forward
  train_years: 4
  test_years: 1
  gap_days: 0
  allow_position_carryover: true  # NEW
  carryover_max_hold: 60
```

**Expected impact**: 
- Wave 2 trade có thể hold thêm 1-2 tuần vào tháng 1
- Tăng PnL từ 4.70% → 8-10%

---

## Ma trận thử nghiệm đề xuất

### Phase A: Exit Optimization (HIGHEST PRIORITY)

```yaml
experiments:
  - name: phase_A1_trend_continuation
    params:
      trend_continuation_enabled: true
      trend_continuation_min_pnl: 0.05
      trend_continuation_min_hold: 20
    expected_impact: "+50-100% total PnL"
  
  - name: phase_A2_adaptive_min_hold
    params:
      exit_model_min_hold_strong: 12
      exit_model_min_hold_moderate: 8
      exit_model_min_hold_weak: 4
    expected_impact: "+30-50% total PnL"
  
  - name: phase_A3_weak_trend_hold
    params:
      weak_trend_exit_relax: true
      weak_trend_min_pnl_hold: 0.05
    expected_impact: "+40-60% total PnL"
  
  - name: phase_A4_exit_confidence_gate
    params:
      exit_model_confidence_threshold: 0.65
      exit_model_trend_override: true
    expected_impact: "+20-30% total PnL"
```

### Phase B: Entry Timing

```yaml
experiments:
  - name: phase_B1_pullback_reentry
    params:
      pullback_reentry_enabled: true
      pullback_reentry_window: 10
      pullback_depth_min: 0.02
      pullback_depth_max: 0.05
    expected_impact: "+10-20 trades, +50-100 PnL"
  
  - name: phase_B2_strong_trend_early_entry
    params:
      strong_trend_early_entry: true
      strong_trend_wp_range: [0.5, 0.7]
      strong_trend_ret5d_max: 0.08
    expected_impact: "+5-10% win rate for strong entries"
```

### Phase C: Year Boundary

```yaml
experiments:
  - name: phase_C1_position_carryover
    split:
      allow_position_carryover: true
      carryover_max_hold: 60
    expected_impact: "+10-20 PnL"
```

---

## Kết luận

### Điểm mù thực sự của model:

1. **Exit quá sớm** (89% trades capture <10% max profit) - **VẤN ĐỀ CHÍNH**
2. **Strong trend paradox** (vào muộn, exit sớm, win rate thấp)
3. **Entry gaps dài** (27 gaps > 30 ngày)
4. **Year boundary artifact** (10 trades forced exit 30/12)

### Tiềm năng cải thiện:

**Conservative estimate** (chỉ Phase A):
- Tăng avg_pnl: 5.88% → 10-12%
- Tăng total_pnl: 540% → 900-1100%
- Giảm số trades: 92 → 70-80 (nhưng quality cao hơn)
- Tăng composite score: 167.6 → 200-220

**Aggressive estimate** (Phase A + B + C):
- Tăng avg_pnl: 5.88% → 15-20%
- Tăng total_pnl: 540% → 1200-1600%
- Số trades: 80-90 (thêm pullback re-entries)
- Tăng composite score: 167.6 → 250-300

### Trade-off:

**Pros**:
- Capture nhiều profit hơn từ winning trades
- Giảm số lượng trades nhưng tăng quality
- Win rate có thể tăng từ 88% → 90%+

**Cons**:
- Tăng avg hold time → tăng exposure risk
- Tăng max drawdown (do hold lâu hơn)
- Có thể giảm Sharpe ratio nếu volatility tăng

### Next steps:

1. **Implement Phase A1** (trend_continuation) - **HIGHEST PRIORITY**
2. Backtest trên full dataset 2020-2025
3. So sánh với baseline (current top 1)
4. Nếu Phase A1 thành công → Phase A2, A3, A4
5. Phase B và C là optional, tùy kết quả Phase A

---

## Appendix: Detailed Trade Examples

### Wave 1 - Trade #74 (Losing trade)

```
Entry: 2024-09-25 03:00
Exit:  2024-11-06 06:00
Hold: 152 days (6.3 trading days)
PnL: -0.69%
Max profit: 102.31%

Entry features:
  - trend: strong
  - ret_5d: 1.52% (đã nóng)
  - dist_sma20: 1.13% (trên SMA20)
  - wp: 0.9706 (gần đỉnh range)
  - score: 2 (thấp)
  - breakout_entry: False

Exit reason: exit_model

Analysis:
- Vào muộn 8 ngày sau khi sóng bắt đầu (17/9)
- Vào gần đỉnh range (wp=0.97)
- Exit khi giá pullback nhỏ
- Bỏ lỡ 102% max profit
```

### Wave 3 - Trade #82 (Best trade but still underperformed)

```
Entry: 2025-08-01 06:00
Exit:  2025-09-05 02:00
Hold: 111 days (4.6 trading days)
PnL: 18.06%
Max profit: 316.48%

Entry features:
  - trend: weak
  - ret_5d: -1.19% (pullback)
  - dist_sma20: -1.98% (dưới SMA20)
  - wp: 0.0198 (đáy range)
  - score: 4 (cao)
  - breakout_entry: False

Exit reason: exit_model

Analysis:
- Vào đúng ngày đầu sóng (1/8)
- Vào ở đáy range (wp=0.02) - PERFECT ENTRY
- Exit khi giá mới chạy được 18%
- Bỏ lỡ 298% profit còn lại (capture chỉ 5.7%)
- Đây là trade TỐT NHẤT nhưng vẫn exit quá sớm
```

### Comparison: Best vs Worst Entry Timing

**Best entry (Trade #82)**:
- Entry: đáy range (wp=0.02)
- Trend: weak (pullback)
- ret_5d: -1.19% (đã nguội)
- Result: +18.06% (nhưng bỏ lỡ 316%)

**Worst entry (Trade #74)**:
- Entry: đỉnh range (wp=0.97)
- Trend: strong (đã chạy)
- ret_5d: +1.52% (đã nóng)
- Result: -0.69% (bỏ lỡ 102%)

**Lesson**: Weak trend entries ở đáy range > Strong trend entries ở đỉnh range
