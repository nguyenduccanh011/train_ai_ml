# Phân tích điểm mù mô hình phái sinh Top 1 - Uptrend 30/10/2020 - 13/1/2021

## Tổng quan mô hình Top 1

**Model**: `deriv_p65_1hem` (phase 65)
- **Composite Score**: 163.6 (rank #1)
- **Timeframe**: 1H
- **Entry Model**: Random Forest
- **Exit Model**: LightGBM (fw=48 bars, loss_threshold=0.038)
- **Strategy**: V22 với v19_entry_cascade
- **Performance**: 96 trades, WR 86.46%, avg PnL 4.71%, Sharpe 0.84

## Vấn đề: Uptrend bị chia nhỏ

### Khoảng thời gian phân tích: 30/10/2020 7h → 13/1/2021 4h (~75 ngày)

Trong trend tăng mạnh này, mô hình chỉ bắt được **4 trades**:

| Entry Date | Exit Date | PnL% | Hold (bars) | Exit Reason | Entry Score |
|------------|-----------|------|-------------|-------------|-------------|
| 2020-11-23 07:00 | 2020-12-30 07:00 | 11.43% | 133 | **end** (year boundary) | 2 |
| 2020-12-07 07:00 | 2020-12-30 07:00 | 7.47% | 81 | **end** (year boundary) | 3 |
| 2021-01-07 06:00 | 2021-01-08 04:00 | 1.94% | 4 | **exit_model** | 2 |
| 2021-01-07 07:00 | 2021-01-08 07:00 | 2.46% | 5 | **exit_model** | 3 |

**Tổng PnL thực tế**: ~23% (11.43 + 7.47 + 1.94 + 2.46)
**Tiềm năng bỏ lỡ**: Trend tăng liên tục 75 ngày nhưng chỉ nắm giữ ~37 ngày (133+81 bars ≈ 24 trading days)

---

## Phân tích 3 điểm mù chính

### 1. **Gap Entry đầu trend: 30/10 → 23/11 (24 ngày bỏ lỡ)**

**Hiện tượng**: 
- Trend bắt đầu từ 30/10/2020
- Trade cuối trước đó exit 21/10/2020 (trade #11: +19.8%, exit_model)
- Trade đầu tiên vào trend mới: 23/11/2020 (gap **24 ngày**)

**Nguyên nhân**:

#### a) **Hot ret5 filter** (line 263-281 trong v19_entry_cascade.py)
```python
if ret_5d[i] > ret5_hot and not strong_breakout_context:
    entry_alpha_ok = False
```
- `ret5_hot` default = 0.06 (6%)
- Khi trend tăng nhanh đầu giai đoạn, ret_5d thường > 6%
- Filter này block entry để tránh "chase" giá quá nóng
- **Nhưng**: Trong uptrend mạnh, đây lại là thời điểm tốt nhất để vào

#### b) **Prev_pred continuation filter** (line 188-192)
```python
if new_position == 1 and not quick_reentry and not breakout_entry and not vshape_entry:
    if (bs >= 4 and vs > 1.2) or (trend == "strong" and rs > 0):
        pass
    elif prev_pred != 1:
        new_position = 0
```
- Yêu cầu bar trước cũng có signal = 1
- Sau khi exit 21/10, model cần thời gian để ML signal "warm up" lại
- Trong giai đoạn consolidation ngắn, signal có thể flip 0-1-0 → block entry

#### c) **Consolidation breakout pattern chưa trigger**
- Model ưu tiên breakout entry (consolidation_breakout, secondary_breakout)
- Nhưng trong uptrend "clean" không có pullback rõ ràng, pattern này không xuất hiện
- Entry thông thường bị block bởi các filter khác

**Features của trade đầu tiên (23/11)**:
- `entry_ret_5d`: 0.73% (đã nguội)
- `entry_dist_sma20`: 1.57% (xa SMA20)
- `entry_wp`: 0.948 (gần đỉnh range 20d)
- `breakout_entry`: True
- `entry_score`: 2 (thấp)

→ Chỉ vào được khi đã có breakout pattern rõ ràng + ret5 đã nguội

---

### 2. **Overlapping positions: 23/11 và 07/12 (vấn đề kỹ thuật)**

**Hiện tượng**:
- Trade #12 entry 23/11, exit 30/12
- Trade #25 entry 07/12, exit 30/12
- **Cùng symbol VN30F1M, cùng hold đến 30/12**

**Nguyên nhân**:
- Backtest engine cho phép 2 "slots" entry độc lập
- Trong thực tế futures, không thể có 2 position riêng biệt trên cùng 1 contract
- Đây là artifact của backtest design cho multi-stock, không phù hợp với single futures

**Hậu quả**:
- Phân tán vốn không cần thiết
- Trade #25 chỉ ăn được 7.47% trong khi #12 ăn 11.43% (cùng exit date)
- Nếu gộp vốn vào 1 position từ 23/11 → có thể tối ưu hơn

---

### 3. **Year-boundary forced exit + Exit model quá nhanh**

#### a) **Forced exit 30/12/2020**
- Walk-forward split: test_years=1, first_test_year=2020
- Tất cả position phải close vào 30/12/2020 (end of test period)
- **Trend vẫn tiếp tục sang tháng 1/2021** nhưng bị cắt đứt

**Trades bị ảnh hưởng**:
```
Trade #12: 11.43% (133 bars) - exit_reason: end
Trade #25: 7.47% (81 bars) - exit_reason: end
```

#### b) **Exit model trigger quá sớm 07-08/01/2021**
- Model retrain trên data 2017-2020, bắt đầu test 2021
- Entry 07/01/2021 (2 trades song song)
- **Exit ngay 08/01 sau 4-5 bars**

**Features của 2 trades này**:
```
Trade #26: entry_ret_5d=1.32%, max_profit=103.12%, realized=1.94%
Trade #36: entry_ret_5d=2.06%, max_profit=13.69%, realized=2.46%
```

**Nguyên nhân exit nhanh**:
1. **Exit model min_hold = 3 bars** (exit_model_exit.py line 11)
   - Sau 3 bars, exit_model được phép fire
   - Forward window = 48 bars (2 days) với loss_threshold = 3.8%
   - Model dự đoán reversal → exit ngay

2. **Không có trend continuation protection**
   - Entry vào đỉnh trend (entry_wp = 0.98-1.0)
   - Exit model không biết đây là continuation của uptrend từ 2020
   - Treat như 1 trade mới độc lập → exit conservative

3. **V22 fast exit có thể trigger** (config có v22_fast_exit_threshold_hb = -0.07)
   - Nếu giá pullback nhẹ sau entry → fast exit fire
   - Trong trend mạnh, pullback ngắn là bình thường nhưng model exit luôn

---

## Phân tích thống kê bổ sung

### Exit reason distribution (toàn bộ 96 trades):
```
exit_model:        86 trades (89.6%) - avg_pnl 4.69%, avg_hold 60 bars, WR 87.21%
end (year boundary): 9 trades (9.4%) - avg_pnl 3.23%, avg_hold 66 bars, WR 77.78%
peak_protect_dist:  1 trade (1.0%) - pnl 19.08%, hold 213 bars
```

### Quick exits (≤10 bars): 33 trades (34.4%)
- Nhiều trades exit sau 4-5 bars
- Avg PnL của quick exits: ~1-2%
- **Max profit bỏ lỡ**: Nhiều trades có max_profit 100-300% nhưng chỉ realize 0.5-2%

**Ví dụ điển hình**:
```
2020-02-26: pnl=0.62%, max_profit=301% (exit sau 4 bars)
2020-07-28: pnl=1.06%, max_profit=234% (exit sau 4 bars)
2021-01-07: pnl=1.94%, max_profit=103% (exit sau 4 bars)
```

→ Exit model quá aggressive, không cho phép trades "breathe"

### Trend analysis:
```
Strong trend:  34 trades, avg_pnl 4.34%, avg_hold 64 bars, WR 79.41%
Moderate:      17 trades, avg_pnl 5.37%, avg_hold 39 bars, WR 88.24%
Weak:          45 trades, avg_pnl 4.73%, avg_hold 69 bars, WR 91.11%
```

**Paradox**: Strong trend có WR thấp nhất (79.41%) và avg_hold dài nhất (64 bars)
→ Model struggle với strong trends, có thể do:
- Entry quá muộn (sau khi trend đã chạy)
- Exit quá sớm (không tin vào continuation)

---

## Đề xuất cải tiến

### 1. **Trend Continuation Mode** (ưu tiên cao)

**Mục tiêu**: Giữ position trong uptrend mạnh thay vì exit-reentry liên tục

**Cách implement**:
```python
# Thêm vào config
params:
  trend_continuation_enabled: true
  trend_continuation_min_pnl: 0.05  # 5%
  trend_continuation_min_hold: 20   # bars
  trend_continuation_exit_relax: true
```

**Logic**:
- Khi trade đang profitable > 5% và hold > 20 bars
- Nếu trend vẫn "strong" (SMA20 > SMA50, MACD > 0, close > SMA20)
- **Disable exit_model**, chỉ dùng trailing stop hoặc hard stop
- Cho phép position "ride the trend" thay vì exit sớm

**Expected impact**: 
- Giảm số lượng trades nhưng tăng avg_pnl
- Bắt được full wave thay vì nhiều đoạn nhỏ

---

### 2. **Relax Hot Ret5 Filter trong Strong Trend**

**Vấn đề hiện tại**:
```python
if ret_5d[i] > ret5_hot and not strong_breakout_context:
    entry_alpha_ok = False
```

**Cải tiến**:
```python
# Thêm exception cho strong uptrend
if ret_5d[i] > ret5_hot:
    # Allow entry nếu:
    # 1. Trend = strong
    # 2. SMA20 > SMA50 (macro uptrend)
    # 3. Close > SMA20 (đang trên support)
    # 4. Entry score >= 3
    if not (trend == "strong" and sma20 > sma50 and close > sma20 and entry_score >= 3):
        entry_alpha_ok = False
```

**Expected impact**: Giảm gap entry từ 24 ngày xuống ~10-15 ngày

---

### 3. **Adaptive Exit Model Hold Time**

**Vấn đề hiện tại**: `exit_model_min_hold = 3` bars (quá ngắn)

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
```

**Thêm exit inhibition trong strong trend**:
```python
# Nếu đang trong strong trend và profitable
if hold_days >= min_hold and ctx.exit_signal == 1:
    if trend == "strong" and current_pnl > 0.02:
        # Chỉ exit nếu exit_signal rất mạnh (confidence > 0.7)
        if exit_confidence < 0.7:
            return FusionResult(action="pass", reason="trend_continuation_override")
```

**Expected impact**: Giảm quick exits từ 33 trades xuống ~15-20 trades

---

### 4. **Year-Boundary Position Carry-Over**

**Vấn đề**: Walk-forward split force close tất cả positions vào 30/12

**Cải tiến**:
```yaml
split:
  method: walk_forward
  train_years: 4
  test_years: 1
  gap_days: 0
  allow_position_carryover: true  # NEW
  carryover_max_hold: 60  # Chỉ carry positions đã hold < 60 bars
```

**Logic**:
- Positions mở trong Q4 được phép carry sang năm sau
- Retrain model vào đầu năm mới nhưng không force close
- Exit khi exit signal fire hoặc đạt max hold

**Expected impact**: 
- Trade #12 và #25 có thể hold thêm 1-2 tuần vào tháng 1
- Tăng avg_pnl từ 11.43% → 15-20%

---

### 5. **Anti-Fragmentation: Single Position per Symbol**

**Vấn đề**: 2 trades song song trên cùng VN30F1M

**Cải tiến**:
```python
# Trong backtest engine
if symbol in active_positions:
    # Không cho phép entry mới
    # HOẶC: add to existing position
    existing_pos = active_positions[symbol]
    existing_pos.size += new_position_size
    existing_pos.avg_entry_price = weighted_avg(...)
```

**Expected impact**: 
- Tập trung vốn vào 1 position
- Giảm complexity, dễ quản lý risk

---

### 6. **Pullback Re-Entry Strategy**

**Mục tiêu**: Sau khi exit trong uptrend, cho phép re-entry nhanh khi pullback

**Logic**:
```python
# Thêm vào entry_state
last_exit_in_uptrend: bool
last_exit_price: float
last_exit_bar: int

# Trong v19_entry_cascade
if last_exit_in_uptrend and bars_since_exit <= 10:
    # Pullback re-entry conditions
    pullback_depth = (last_exit_price - close[i]) / last_exit_price
    if 0.02 < pullback_depth < 0.05:  # 2-5% pullback
        if close[i] > sma20[i] and macd_line[i] > 0:
            new_position = 1
            counters["n_pullback_reentry"] += 1
```

**Expected impact**: Bắt lại trend sau exit sớm, giảm gap time

---

## Ma trận thử nghiệm đề xuất

### Phase A: Entry Timing (giải quyết gap 24 ngày)
```yaml
experiments:
  - name: phase_A1_relax_hot_ret5
    params:
      v19_relax_hot_ret5_strong: true
      v19_hot_ret5_threshold: 0.08  # Tăng từ 0.06
  
  - name: phase_A2_trend_entry_bypass
    params:
      v19_strong_trend_bypass: true
      v19_bypass_min_score: 2
  
  - name: phase_A3_shorter_ma
    components:
      target:
        short_window: 3  # Giảm từ 5
        long_window: 15  # Giảm từ 20
```

### Phase B: Exit Optimization (giải quyết exit sớm)
```yaml
experiments:
  - name: phase_B1_adaptive_min_hold
    params:
      exit_model_min_hold_strong: 12
      exit_model_min_hold_moderate: 8
      exit_model_min_hold_weak: 4
  
  - name: phase_B2_trend_continuation
    params:
      trend_continuation_enabled: true
      trend_continuation_min_pnl: 0.05
      trend_continuation_min_hold: 20
  
  - name: phase_B3_exit_confidence_gate
    params:
      exit_model_confidence_threshold: 0.65
      exit_model_trend_override: true
```

### Phase C: Year Boundary (giải quyết forced exit)
```yaml
experiments:
  - name: phase_C1_position_carryover
    split:
      allow_position_carryover: true
      carryover_max_hold: 60
  
  - name: phase_C2_rolling_window
    split:
      method: rolling_window
      train_days: 1460  # 4 years
      test_days: 90     # 3 months
      step_days: 30     # Roll monthly
```

### Phase D: Pullback Re-Entry
```yaml
experiments:
  - name: phase_D1_pullback_reentry
    params:
      pullback_reentry_enabled: true
      pullback_reentry_window: 10
      pullback_depth_min: 0.02
      pullback_depth_max: 0.05
```

---

## Kết luận

### Điểm mù chính của mô hình:

1. **Entry quá muộn**: Gap 24 ngày do hot ret5 filter + prev_pred continuation
2. **Exit quá sớm**: Min hold 3 bars + exit model aggressive → bỏ lỡ 100-300% max profit
3. **Year boundary artifact**: Forced exit 30/12 cắt đứt trend đang chạy
4. **Overlapping positions**: Phân tán vốn không cần thiết

### Tiềm năng cải thiện:

**Conservative estimate**:
- Giảm entry gap: 24 ngày → 10 ngày (+10-15% PnL)
- Tăng hold time: 4-5 bars → 12-20 bars (+5-10% PnL)
- Position carryover: +5-10% PnL

**Tổng tiềm năng**: Từ 23% → 35-45% PnL trong uptrend này

**Trade-off**:
- Giảm số lượng trades (từ 96 → ~70-80)
- Tăng avg hold time (từ 60 bars → 80-100 bars)
- Tăng max drawdown risk (do hold lâu hơn)

### Next steps:

1. Implement Phase A experiments (entry timing) - **ưu tiên cao**
2. Backtest trên full dataset 2020-2025
3. So sánh với baseline (current top 1)
4. Nếu Phase A thành công → Phase B (exit optimization)
5. Phase C và D là optional, tùy kết quả Phase A+B

---

## Appendix: Các vùng tương tự cần kiểm tra

Để tìm pattern tương tự (uptrend bị chia nhỏ), cần scan toàn bộ 96 trades:

**Criteria**:
- Trend = "strong"
- Exit reason = "end" hoặc "exit_model" với hold < 10 bars
- Entry gap > 15 ngày so với trade trước

**Candidates** (cần verify bằng price data):
- 2020-04-21 → 2020-05-19 (trade #5: +15.34%, 72 bars)
- 2020-08-05 → 2020-10-21 (trade #11: +19.80%, 262 bars)
- 2021-04-01 → 2021-04-20 (trade #29, #39: +7.28%, +7.47%)

Cần plot price chart + trades để xác nhận.
