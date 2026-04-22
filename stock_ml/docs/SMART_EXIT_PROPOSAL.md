# 🎯 ĐỀ XUẤT CHI TIẾT: SỬA BUG & CHIẾN LƯỢC SMART EXIT

**Ngày:** 19/04/2026

---

## 🔴 PHẦN 1: PHÂN TÍCH CHI TIẾT BUG BUY & HOLD = -99.8%

### 1.1 Bug chính: Nối returns nhiều cổ phiếu thành 1 chuỗi

**Vị trí bug:** `run_backtest.py` dòng 138-140 và 152-156

```python
# Dòng 138-140: Nối returns từ TẤT CẢ symbols & windows vào 1 list
all_returns.extend(returns.tolist())   # ← BUG TẠI ĐÂY

# Dòng 156: Backtest chuỗi nối này như 1 cổ phiếu duy nhất
agg_bt = backtest_predictions(all_pred_arr, all_returns_arr, args.capital)
```

**Tại sao sai:**
- Giả sử có 10 cổ phiếu × 6 windows = 60 đoạn returns
- Mỗi đoạn ~250 ngày → tổng ~15,000 ngày returns
- Các returns này bị **nối nối tiếp** (HPG 2020 → VNM 2020 → FPT 2020 → HPG 2021 → ...)
- Buy & Hold formula: `(1 + r1) × (1 + r2) × ... × (1 + r15000)` 
- Compounding 15,000 ngày returns (thay vì 1,250 ngày) → tích lũy errors cực lớn
- Chỉ cần trung bình return hơi âm (-0.05%/ngày) → `0.9995^15000 ≈ 0.0005` → **-99.95%**

### 1.2 Bug phụ: Backtest line 93

```python
bnh_equity = initial_capital * (1 + returns).cumprod()
```

Formula này đúng cho 1 cổ phiếu, nhưng khi `returns` là chuỗi nối từ nhiều cổ phiếu, nó tính như đầu tư liên tiếp: hết HPG chuyển sang VNM rồi FPT... không phải Buy & Hold thực sự.

### 1.3 Cách sửa đúng

```python
# CÁCH 1: Backtest PER-SYMBOL rồi average
per_symbol_returns = {}
for symbol in symbols:
    symbol_bt = backtest_predictions(y_pred_symbol, returns_symbol, capital)
    per_symbol_returns[symbol] = symbol_bt

# Average/portfolio metrics
avg_return = np.mean([bt['total_return_pct'] for bt in per_symbol_returns.values()])
portfolio_return = sum([bt['total_return_pct'] for bt in per_symbol_returns.values()]) / len(symbols)

# CÁCH 2: Portfolio backtest (chia vốn đều cho N cổ phiếu)
capital_per_stock = initial_capital / n_stocks
# Backtest từng cổ phiếu với capital_per_stock, rồi sum equity curves

# CÁCH 3: Buy & Hold đúng cho từng symbol
for symbol in symbols:
    bnh = capital_per_stock * (1 + symbol_returns).cumprod()
    total_bnh += bnh[-1]
```

---

## 🧠 PHẦN 2: CHIẾN LƯỢC SMART EXIT - "CƯỠI SÓNG THÔNG MINH"

### 2.1 Ý tưởng cốt lõi

```
Sóng tăng thực tế: +10% → -2% → +10% → -5% → +8% → -3% → +15%
                          ↑ nhỏ,       ↑ lớn,            ↑ nhỏ
                        giữ lại     BÁN + mua lại      giữ lại

Mục tiêu: Phân biệt "điều chỉnh nhỏ" (noise) vs "đảo chiều thật" (trend change)
→ Giữ qua noise, chốt lãi trước đảo chiều, mua lại ở đáy
```

### 2.2 Các tín hiệu Smart Exit

#### Signal 1: Adaptive Trailing Stop (Trailing Stop thích ứng theo volatility)
```
Thay vì trailing stop cố định -5%, dùng ATR-based:
- stop_distance = 2 × ATR(14)  
- Khi volatility thấp (thị trường calm): stop chặt hơn (vd: -2%)
- Khi volatility cao (thị trường wild): stop rộng hơn (vd: -7%)
- Tự động phân biệt noise vs real reversal

Ví dụ: ATR = 1.5% → trailing stop = 3%
- Giảm -2%: < 3% → GIỮ (noise)
- Giảm -5%: > 3% → BÁN (real pullback)
```

#### Signal 2: Pullback Depth Classification (Model phân loại độ sâu pullback)
```
Thêm features mới để model dự đoán:
- pullback_depth: % giảm từ đỉnh gần nhất
- pullback_duration: số ngày giảm liên tục
- volume_on_pullback: volume giảm hay tăng khi giá giảm?
  → Volume giảm khi giá giảm = profit-taking nhẹ → GIỮ
  → Volume tăng khi giá giảm = bán tháo → BÁN
- support_distance: giá gần support bao nhiêu %?
- rsi_divergence: RSI tăng trong khi giá giảm = bullish divergence → GIỮ
```

#### Signal 3: Multi-Timeframe Trend Confirmation
```
- Trend ngắn hạn (SMA5 vs SMA10): chuyển xuống → cảnh báo
- Trend trung hạn (SMA20 vs SMA50): vẫn tăng → chưa bán
- Trend dài hạn (SMA50 vs SMA200): vẫn tăng → bullish

Rule:
- Nếu chỉ trend ngắn hạn bearish: GIỮ (pullback nhỏ)
- Nếu trend ngắn + trung hạn bearish: BÁN (pullback lớn)
- Nếu cả 3 bearish: BÁN NGAY + chờ đáy
```

#### Signal 4: Volume Profile Analysis
```
- Giảm giá + volume thấp = pullback bình thường → GIỮ
- Giảm giá + volume spike (>2x avg) = bán tháo → BÁN
- Giảm giá + volume cao + gần support = có thể reversal → CHUẨN BỊ MUA LẠI
```

### 2.3 Chiến lược tổng hợp: 4-Level Exit System

```
LEVEL 0: HOLD (Giữ vững)
├── Drawdown từ đỉnh < 1×ATR (thường ~2%)
├── Volume bình thường
├── Trend trung hạn vẫn UP
└── Action: Không làm gì, tiếp tục nắm giữ

LEVEL 1: ALERT (Cảnh giác)
├── Drawdown từ đỉnh = 1-1.5×ATR (~2-3%)
├── Bắt đầu tighten trailing stop
├── Kiểm tra volume pattern
└── Action: Di chuyển stop-loss lên, khóa một phần lợi nhuận

LEVEL 2: PARTIAL EXIT (Chốt lãi một phần)
├── Drawdown từ đỉnh = 1.5-2×ATR (~3-5%)
├── Volume tăng khi giá giảm
├── Trend ngắn hạn bearish
└── Action: BÁN 50% vị thế, giữ 50% với stop chặt

LEVEL 3: FULL EXIT (Thoát hoàn toàn)
├── Drawdown từ đỉnh > 2×ATR (~5%+)
├── Trend ngắn + trung hạn bearish
├── HOẶC volume spike + giá phá support
└── Action: BÁN 100%, chuyển sang tìm điểm mua lại
```

### 2.4 Re-Entry Strategy (Mua lại sau khi bán)

```
Sau khi FULL EXIT, tìm điểm mua lại:

Tín hiệu mua lại:
1. RSI < 30 (oversold) + bắt đầu quay lên
2. Giá chạm/phá xuống rồi bounce lại support (Bollinger Band dưới, SMA50)
3. Volume spike + giá đảo chiều (hammer candle)
4. Model predict chuyển sang UPTREND lại
5. MACD histogram chuyển từ âm sang dương

Confirmation rule:
- Cần ≥ 2/4 tín hiệu trên để mua lại
- Mua lại từng phần: 50% trước, 50% khi confirm
```

### 2.5 Ví dụ cụ thể

```
Ngày 1-20:  Giá tăng +10%     → HOLD (riding the wave) ✅
Ngày 21-23: Giá giảm -2%      → LEVEL 0-1 (noise, < 1.5×ATR) → GIỮ ✅  
Ngày 24-30: Giá tăng +10%     → HOLD (sóng tiếp) ✅
Ngày 31-35: Giá giảm -3%      → LEVEL 1 (cảnh giác, tighten stop)
Ngày 36:    Giá giảm thêm -2% → LEVEL 2 (tổng -5%, > 2×ATR)
            Volume spike 2.5x  → FULL EXIT: BÁN 100%
            
Ngày 37-42: Giá tiếp tục giảm -3% → Đang CASH, không ảnh hưởng ✅
Ngày 43:    RSI = 28, bounce từ SMA50, volume spike
            → RE-ENTRY: Mua 50%
Ngày 45:    MACD cross up, giá > SMA10
            → RE-ENTRY: Mua thêm 50%
Ngày 46-65: Giá tăng +15% → HOLD (sóng mới) ✅

Kết quả:
- Không Smart Exit: +10% -2% +10% -5% +15% = compound ~+29.7%
- Với Smart Exit:   +10% -2% +10% BÁN(-0%) MUA LẠI +12% = compound ~+33%
  (tránh được -5%, mua lại gần đáy, gain extra ~3-4%)
```

---

## 🏗️ PHẦN 3: IMPLEMENTATION PLAN

### 3.1 File mới cần tạo

```
stock_ml/
├── src/
│   ├── evaluation/
│   │   ├── backtest.py          ← SỬA: thêm Smart Exit logic
│   │   ├── backtest_v2.py       ← MỚI: Portfolio-based backtest
│   │   └── exit_signals.py      ← MỚI: Exit signal generators
│   ├── features/
│   │   └── engine.py            ← SỬA: thêm exit-related features
│   └── strategy/
│       ├── __init__.py
│       ├── smart_exit.py        ← MỚI: Smart Exit Strategy
│       └── position_manager.py  ← MỚI: Position sizing & management
├── run_backtest.py              ← SỬA: per-symbol backtest
└── run_backtest_v2.py           ← MỚI: Smart Exit backtest
```

### 3.2 Features mới cần thêm cho Smart Exit

```python
# Trong engine.py, thêm feature group "exit_signals":
new_features = {
    # Drawdown features
    'drawdown_from_peak': (price - rolling_max) / rolling_max,
    'days_since_peak': số ngày kể từ đỉnh gần nhất,
    
    # Pullback classification  
    'pullback_depth_pct': % giảm từ đỉnh local,
    'pullback_duration': số ngày giảm liên tục,
    'pullback_vol_ratio': volume_during_pullback / avg_volume,
    
    # Support/Resistance
    'dist_to_support': % distance to nearest support,
    'dist_to_resistance': % distance to nearest resistance,
    
    # Multi-timeframe trend
    'trend_short': SMA5 > SMA10,
    'trend_medium': SMA20 > SMA50,  
    'trend_long': SMA50 > SMA200,
    'trend_alignment': sum of above (0-3),
    
    # Reversal signals
    'rsi_divergence': giá giảm nhưng RSI tăng,
    'volume_price_divergence': giá giảm + volume giảm,
    'hammer_pattern': candle đảo chiều,
}
```

### 3.3 Target mới cho Exit Model

```python
# Thay vì chỉ predict UPTREND/SIDEWAYS/DOWNTREND
# Thêm predict: EXIT_SIGNAL

# Target cho exit model:
# 1 = nên GIỮ (pullback nhỏ, sẽ hồi lại)  
# 0 = nên BÁN (pullback lớn, sẽ giảm tiếp)

# Cách tạo target (look-ahead, chỉ dùng cho training):
def create_exit_target(df, threshold_pct=3):
    """
    Khi đang trong pullback:
    - Nếu sau 5 ngày giá hồi lại > giá hiện tại → target = 1 (GIỮ)
    - Nếu sau 5 ngày giá giảm thêm > threshold → target = 0 (BÁN)
    """
    future_return_5d = df['close'].shift(-5) / df['close'] - 1
    target = (future_return_5d > 0).astype(int)
    return target
```

---

## 📊 PHẦN 4: SO SÁNH CHIẾN LƯỢC

| Chiến lược | Ưu điểm | Nhược điểm | Độ phức tạp |
|---|---|---|---|
| **Hiện tại** (all-in/all-out) | Đơn giản | DD -90%, miss sóng | ⭐ |
| **Fixed Stop-Loss** (-5%) | Giảm DD | Bị stop-out bởi noise | ⭐⭐ |
| **ATR Trailing Stop** | Thích ứng volatility | Cần tune multiplier | ⭐⭐⭐ |
| **Smart Exit 4-Level** | Tối ưu nhất | Complex, cần nhiều features | ⭐⭐⭐⭐ |
| **ML Exit Model** | Adaptive, học từ data | Cần data + training riêng | ⭐⭐⭐⭐⭐ |

### Đề xuất: Implement theo 3 bước

1. **Bước 1:** Sửa bug + ATR Trailing Stop (nhanh, hiệu quả ngay)
2. **Bước 2:** Smart Exit 4-Level + Volume signals
3. **Bước 3:** ML Exit Model (train model riêng cho exit decisions)

---

## ⚡ PHẦN 5: BẮT ĐẦU NGAY - SỬA BUG & ATR TRAILING STOP

Bạn muốn tôi implement ngay không? Tôi sẽ:

1. **Sửa `run_backtest.py`**: Backtest per-symbol rồi aggregate đúng cách
2. **Tạo `backtest_v2.py`**: Thêm ATR trailing stop + partial exit
3. **Tạo `exit_signals.py`**: Compute exit signals (drawdown, volume, trend alignment)
4. **Cập nhật `engine.py`**: Thêm exit-related features
5. **Tạo `run_backtest_v2.py`**: Runner mới với Smart Exit

Kỳ vọng sau cải thiện:
- Max Drawdown: -90% → **-15% đến -25%**
- Win Rate: 47% → **55-60%**
- Sharpe: 0.28 → **0.5-1.0**
- Tận dụng được 60-70% sóng tăng thay vì 40% như hiện tại
