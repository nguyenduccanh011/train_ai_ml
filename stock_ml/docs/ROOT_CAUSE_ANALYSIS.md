# 🔬 PHÂN TÍCH GỐC RỄ VẤN ĐỀ MÔ HÌNH

## TÓM TẮT: 3 VẤN ĐỀ GỐC RỄ

### ❌ GỐC RỄ #1: TARGET SAI - "Dán nhãn quá khứ, không dự đoán tương lai"

**Hiện tại:** Target `trend_regime` dùng dual MA crossover:
```
UPTREND = SMA10 > SMA40 AND price > SMA10
```
Rồi shift(-1) để "dự đoán ngày mai".

**Vấn đề cốt lõi:**
- MA crossover là **lagging indicator** (chậm 10-20 ngày so với thực tế)
- Khi SMA10 > SMA40 → giá đã tăng 15-30% rồi → model học cách nhận ra "đã tăng" thay vì "sắp tăng"
- Shift(-1) chỉ dịch 1 ngày → label ngày mai gần như giống hôm nay → model học pattern hiện tại, không predict tương lai
- **Kết quả:** Model mua khi wave_position = 0.85 (gần đỉnh) vì MA crossover chỉ confirm uptrend khi đã tăng nhiều

**Ví dụ cụ thể:**
```
Ngày 1-20: Giá tăng từ 10→15 (+50%)
Ngày 15: SMA10 cross trên SMA40 → bắt đầu label = UPTREND
Ngày 20: Model predict UPTREND → MUA (nhưng đã tăng 50% rồi!)
Ngày 25: Giá đỉnh tại 16
Ngày 30: Giá giảm → SMA10 cross xuống SMA40 → label = 0 → BÁN (đã giảm 10%)
```

### ❌ GỐC RỄ #2: FEATURES TOÀN LAGGING - "Nhìn gương chiếu hậu để lái xe"

**Hiện tại:** Features chủ yếu là:
- Moving Averages (SMA5, 10, 20, 50) → lagging
- RSI, MACD, Bollinger → đều dựa trên MA → lagging
- Return 1d, 5d, 20d → look-back

**Vấn đề:**
- Không có features **leading** (dự báo trước): volume spike trước breakout, volatility contraction trước expansion, accumulation patterns
- Không có features về **market structure**: support/resistance, swing points, consolidation patterns
- Features và target đều lagging → model chỉ học "nhận diện uptrend đang diễn ra" → CHẬM

### ❌ GỐC RỄ #3: BÀI TOÁN SAI - "Predict trend regime hàng ngày thay vì predict điểm mua"

**Hiện tại:** Classification 3 class (UP/SIDE/DOWN) mỗi ngày
- Model predict regime hàng ngày → flip signal liên tục → 42% trades chỉ giữ 1-2 ngày (win 1.2%)
- Regime classification có noise rất cao ở daily frequency
- Không phân biệt "bắt đầu uptrend" vs "đang giữa uptrend" vs "sắp hết uptrend"

---

## 🎯 GIẢI PHÁP GỐC RỄ: THIẾT KẾ LẠI TỪ ĐẦU

### GIẢI PHÁP 1: TARGET MỚI - "Forward Risk-Reward" (Quan trọng nhất!)

Thay vì label trend regime (lagging), dùng **forward-looking target**:

```python
# Với mỗi ngày t, nhìn về TƯƠNG LAI:
max_gain_10d = max(price[t+1 : t+11]) / price[t] - 1   # max upside 10 ngày tới
max_loss_10d = min(price[t+1 : t+11]) / price[t] - 1   # max downside 10 ngày tới
risk_reward = max_gain_10d / abs(max_loss_10d)            # tỷ lệ reward/risk

# Label:
BUY = 1 nếu: max_gain >= 5% AND max_loss > -3% AND risk_reward >= 2
AVOID = 0 nếu không
```

**Tại sao tốt hơn:**
- Target trực tiếp đo "nếu mua hôm nay, 10 ngày tới có lời không?"
- Bao gồm cả upside VÀ downside → model học tránh mua khi risk cao
- Không dùng MA → không bị lag
- Forward window 10 ngày → model cần predict TRƯỚC khi giá tăng

### GIẢI PHÁP 2: FEATURES LEADING - "Tín hiệu dẫn dắt"

Thêm features phát hiện **SỚM** trước khi sóng bắt đầu:

```python
# 1. VOLUME PRECURSOR - Volume tăng trước price breakout
volume_surge = volume / volume_ma20  # >2 = bất thường
price_volume_divergence = (volume tăng mạnh nhưng giá chưa tăng nhiều)

# 2. VOLATILITY CONTRACTION → trước expansion
bb_width = (BB_upper - BB_lower) / BB_middle  # hẹp = sắp breakout
atr_ratio = ATR5 / ATR20  # <0.8 = đang co lại = sắp nổ

# 3. ACCUMULATION PATTERNS
close_position_in_range = (close - low) / (high - low)  # >0.7 liên tục = tích lũy
obv_divergence = OBV tăng trong khi giá đi ngang

# 4. SUPPORT/RESISTANCE PROXIMITY
dist_to_support = (price - recent_low_20d) / price
dist_to_resistance = (recent_high_20d - price) / price
breakout_potential = dist_to_resistance < 0.02  # gần resistance = sắp breakout

# 5. MARKET STRUCTURE
higher_lows_count = số lần low > previous low (5 bars)
consolidation_days = số ngày giá dao động < 2%
```

### GIẢI PHÁP 3: BÀI TOÁN MỚI - "Predict BUY POINT, không predict daily regime"

**Cách tiếp cận:**
1. **Binary classification**: "Hôm nay có phải BUY POINT không?" (1/0)
2. **Không predict mỗi ngày** - Chỉ trigger khi xác suất cao (>70%)
3. **Fixed holding period** - Mua và giữ 10 ngày (không depend vào daily prediction)
4. **Hoặc ATR trailing stop** - Mua và giữ cho đến khi trailing stop hit

**Tại sao tốt hơn:**
- Không bị whipsaw từ daily flip
- Model tập trung vào 1 việc: "tìm điểm mua tốt"
- Exit rule cố định/mechanical → không cần model predict exit
- Ít noise hơn nhiều so với daily regime classification

---

## 📊 SO SÁNH CŨ vs MỚI

| Khía cạnh | Cách cũ | Cách mới |
|-----------|---------|----------|
| **Target** | MA crossover (lagging) | Forward risk-reward (leading) |
| **Câu hỏi** | "Hôm nay uptrend?" | "Mua hôm nay có lời 10d tới?" |
| **Features** | RSI, MACD (lagging) | Volume precursor, volatility contraction (leading) |
| **Exit** | Daily prediction flip | Fixed 10d hoặc trailing stop |
| **Frequency** | Mỗi ngày | Chỉ khi xác suất cao |
| **Entry timing** | Gần đỉnh (0.85) | Gần đáy/giữa sóng |

## 🔧 KẾ HOẠCH TRIỂN KHAI

1. **Phase 1:** Thêm target `forward_risk_reward` vào `target.py`
2. **Phase 2:** Thêm leading features vào `engine.py`
3. **Phase 3:** Thay đổi backtest: binary buy signal + fixed hold/trailing stop
4. **Phase 4:** So sánh kết quả cũ vs mới

**Ước tính cải thiện:**
- Entry timing: wave_pos 0.85 → 0.3-0.5
- Win rate: 29% → 45-55%
- Avg return/trade: +1.8% → +3-5%
- Max DD in trade: -1.6% → -1.0% (nhờ risk filter trong target)
