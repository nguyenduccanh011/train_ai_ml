# 📊 Phân tích sâu Model V8b — Trường hợp ACB & Vấn đề tổng thể

## 1. Tổng quan kết quả

| Metric | Giá trị |
|--------|---------|
| Tổng mã test | 14 |
| Mã có lãi | 13/14 (93%) |
| Mã lỗ | 1/14 (VNM: -14.57%) |
| Top performer | DGC (+375%), SSI (+305%), AAS (+290%) |
| Bottom performer | ACB (+40%), VNM (-14.57%) |

> **Lưu ý**: Hiện chỉ test 14 mã, KHÔNG PHẢI 200+ mã. Cần mở rộng test toàn bộ thị trường.

---

## 2. Phân tích chi tiết ACB (27 trades, WR 51.9%, Avg PnL +1.50%)

### 2.1 Tổng quan trades ACB
- **Wins**: 14 lệnh, trung bình **+8.04%**/lệnh
- **Losses**: 13 lệnh, trung bình **-5.55%**/lệnh  
- **Lỗ lớn (>5%)**: 6 lệnh — chiếm **46% losses** → Vấn đề nghiêm trọng
- **Lãi nhỏ (0-3%)**: 2 lệnh — bỏ lỡ cơ hội
- **Lãi lớn (>10%)**: 4 lệnh

### 2.2 Phân tích từng trade đáng chú ý

#### 🔴 Trade #2: MUA đỉnh, BÁN muộn → Lỗ -16.37%
- **Entry**: 2020-03-05, giá ~7.3 (gần đỉnh sóng)
- **Exit**: 2020-03-20, giá ~6.1 (đáy COVID crash)
- **Vấn đề gốc**: 
  - Model predict BUY khi giá ở **range_position cao** (gần đỉnh 20 ngày)
  - RSI slope dương nhưng đã chạm đỉnh → signal misleading
  - Không nhận diện được **macro risk** (COVID black swan)
  - Stop loss ATR 2.5x quá rộng cho volatility bình thường, nhưng khi crash thì gap xuống qua stop
- **Cải thiện**: Thêm filter "VIX/market breadth" hoặc "tỷ lệ mã giảm toàn sàn" → cắt lỗ sớm hơn khi thị trường panic

#### 🟡 Trade #3: Sóng lớn 4.6 → 7.5 nhưng chỉ ăn +4.41%
- **Context**: Giá chạy từ 4.65 (1/4/2020) lên 7.65 (2/6/2020) = **+65% upside**
- **Entry**: 2020-05-27, giá 6.58 (đã chạy được ~42% rồi)
- **Exit**: 2020-06-16, giá 6.87 → Chỉ ăn **+4.41%**
- **Vấn đề gốc**:
  1. **MUA QUÁ MUỘN**: Sóng bắt đầu từ 1/4, nhưng model mua 27/5 — bỏ lỡ 80% sóng tăng
  2. **BÁN QUÁ SỚM**: Giá còn có thể chạy lên 7+ nhưng exit signal quá nhạy
  3. **Entry filter quá chặt**: Yêu cầu 2 ngày liên tiếp predict=1 + nhiều entry filters → trễ signal
  4. **Dual MA (10/40)** chỉ xác nhận uptrend khi sóng đã chạy được nửa đường

#### 🔴 Trade #7: Lỗ -14.29% (stop_loss)
- **Entry**: 2021-01-13 → **Exit**: 2021-01-28
- Giá giảm nhanh, stop loss kích hoạt nhưng đã lỗ quá sâu
- **Nguyên nhân**: ATR stop 2.5x = ~6-8% nhưng giá gap down → slippage

#### 🟢 Trade #10: +24.75% (trailing_stop) — Trade tốt nhất
- **Entry**: 2021-05-04 → **Exit**: 2021-06-07, hold 24 ngày
- Trailing stop hoạt động tốt khi trend mạnh
- Đây là **mẫu trade lý tưởng** mà model nên tái tạo nhiều hơn

#### 🔴 Trade #11-12: Liên tiếp -8.24% và -9.36% (stop_loss)
- Hai lệnh thua liên tiếp ngay sau win lớn
- **Nguyên nhân**: Model cố gắng bắt lại trend nhưng thị trường đã chuyển sang sideway/bearish
- **Cải thiện**: Cần cooldown dài hơn sau big win hoặc regime detection

---

## 3. Chẩn đoán 5 vấn đề gốc của mô hình

### Vấn đề #1: 🎯 MUA QUÁ MUỘN (Late Entry)
**Nguyên nhân**: 
- Target dùng Dual MA (short=10, long=40) → signal trễ 10-20 ngày so với đáy
- Yêu cầu 2 ngày liên tiếp predict=1 mới vào lệnh
- Entry filters (range_position, breakout_score, rsi_slope...) loại bỏ entry sớm

**Impact**: Bỏ lỡ 50-80% sóng tăng. Trade #3 ACB là ví dụ điển hình.

### Vấn đề #2: 🛑 STOP LOSS CHƯA TỐI ƯU
**Nguyên nhân**:
- ATR stop 2.5x quá rộng cho sideway, quá hẹp cho volatile market
- Không adaptive theo regime (trending vs ranging)

**Impact**: 6/13 losses ở ACB > 5%, trung bình loss -5.55% quá lớn so với avg win +8.04%

### Vấn đề #3: 📉 THIẾU MARKET REGIME DETECTION
**Nguyên nhân**:
- Model chỉ nhìn single stock, không xét market-wide condition
- Không có VN-Index trend, market breadth, sector rotation signals

**Impact**: Mua trong bear market (Trade #2 COVID), mua liên tiếp khi thị trường sideway (Trade #11-12)

### Vấn đề #4: ⏱️ EXIT SIGNAL QUÁ NHẠY TRONG UPTREND
**Nguyên nhân**:
- EXIT_CONFIRM = 3 ngày nhưng pullback nhỏ trong uptrend cũng trigger 3 ngày exit
- Trailing stop kích hoạt quá sớm khi profit chưa maximize

**Impact**: Trade #3 chỉ ăn 4.41% trong sóng 65%. Nhiều trades lãi nhỏ 1-4%.

### Vấn đề #5: 🔢 THIẾU DIVERSITY & COVERAGE
**Nguyên nhân**:
- Chỉ test 14 mã, chưa biết model hoạt động ra sao trên 200+ mã
- Có thể bị overfitting trên tập mã nhỏ

**Impact**: Kết quả có thể không representative cho thị trường thực

---

## 4. Phương án cải thiện (Roadmap)

### Phase 1: Quick Wins (1-2 ngày)
| Action | Expected Impact |
|--------|----------------|
| **Giảm ATR multiplier từ 2.5 → 1.8** cho non-trending | Giảm avg loss từ -5.5% → ~-3.5% |
| **Thêm max loss cap 8%** (hard stop) | Tránh trade #2 kiểu -16% |
| **Giảm entry confirmation từ 2 → 1 ngày** khi breakout_score ≥ 4 | Mua sớm hơn 1-2 ngày |
| **Tăng trailing stop width** khi RSI slope > 0 | Giữ winners lâu hơn |

### Phase 2: Model Enhancement (3-5 ngày)
| Action | Expected Impact |
|--------|----------------|
| **Market regime feature** (VN-Index SMA, market breadth) | Tránh mua trong bear market |
| **Target dùng EMA(5/20) thay vì SMA(10/40)** | Entry sớm hơn 5-10 ngày |
| **Adaptive stop-loss** theo regime (tight in range, wide in trend) | Tăng WR 5-8% |
| **Anti-correlation filter**: không mua khi VN-Index giảm > 2% | Giảm 50% big losses |

### Phase 3: Scale & Validation (1 tuần)
| Action | Expected Impact |
|--------|----------------|
| **Test toàn bộ 200+ mã** | Biết true coverage rate |
| **Walk-forward validation nghiêm ngặt** | Tránh overfitting |
| **Monte Carlo simulation** | Đo risk-adjusted return |
| **Sector-level analysis** | Biết model tốt cho sector nào |

---

## 5. Kỳ vọng sau cải thiện

| Metric | Hiện tại (ACB) | Mục tiêu |
|--------|---------------|----------|
| Win Rate | 51.9% | **60-65%** |
| Avg Win | +8.04% | +10-12% |
| Avg Loss | -5.55% | **-3.0 đến -3.5%** |
| Avg PnL/trade | +1.50% | **+4-6%** |
| Max Drawdown | -16.37% | **< -8%** |
| Profit Factor | ~1.3 | **> 2.0** |

**Kết luận**: Model V8b có nền tảng tốt (13/14 mã lãi), nhưng **entry quá muộn** và **stop loss quá rộng** là 2 vấn đề chính cần khắc phục. Ưu tiên Phase 1 trước để cải thiện ngay Avg PnL từ +1.5% lên +3-4%/trade.
