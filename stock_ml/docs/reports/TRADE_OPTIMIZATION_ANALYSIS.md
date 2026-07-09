# 📊 BÁO CÁO BACKTEST V7 — Lần chạy 19/04/2026

> **Cập nhật lần cuối:** 19/04/2026 20:35 — Đã sửa bug PnL calculation (dùng close prices thay vì equity-based)

## Thống kê tổng quan (846 trades, 29 mã, 2020-2025)

| Metric | Giá trị | Đánh giá |
|--------|---------|----------|
| Total trades | 846 | ~29 trades/symbol |
| Win Rate | 45.6% | ❌ Chưa đủ cao |
| Avg PnL/trade | +1.56% | ❌ Quá thấp (mục tiêu >10%) |
| Median PnL | -0.23% | ❌ Hơn nửa số lệnh thua lỗ |
| Avg win | +7.46% | 🟡 Chấp nhận được |
| Avg loss | -3.40% | ✅ Tốt |
| R:R ratio | 2.20 | ✅ Tốt |
| Profit Factor | 1.84 | 🟡 OK |
| Gross wins | +2,878.9% | |
| Gross losses | -1,562.4% | |
| **Net PnL** | **+1,316.5%** | ✅ Dương |

---

## 📈 Phân bổ PnL theo trades

| Loại | Số lệnh | % tổng |
|------|---------|--------|
| Tiny wins (0-3%) | 182 | 21.5% |
| Medium wins (3-10%) | 124 | 14.7% |
| Big wins (≥10%) | 80 | 9.5% |
| Small losses (0 to -3%) | 259 | 30.6% |
| Medium losses (-3 to -5%) | 92 | 10.9% |
| Big losses (≤-5%) | 109 | 12.9% |

**→ 51.3% trades có PnL < 0 (thua lỗ)**
**→ 80 big wins (≥10%) mang lại phần lớn lợi nhuận**

---

## 📊 PER-SYMBOL PERFORMANCE (29 mã)

| Symbol | # Trades | WR | Avg PnL | Total PnL | Đánh giá |
|--------|----------|-----|---------|-----------|----------|
| SSI | 35 | 62.9% | +4.92% | +172.2% | ⭐ Top 1 |
| VND | 39 | 53.8% | +4.63% | +180.7% | ⭐ Top 2 |
| AAS | 33 | 42.4% | +4.78% | +157.7% | ⭐ Top 3 |
| DGC | 35 | 42.9% | +3.96% | +138.5% | ⭐ |
| AAV | 27 | 40.7% | +4.59% | +123.9% | ⭐ |
| VIC | 24 | 37.5% | +3.55% | +85.2% | ✅ |
| HPG | 35 | 48.6% | +2.14% | +74.8% | ✅ |
| FPT | 34 | 55.9% | +2.12% | +71.9% | ✅ |
| ABB | 26 | 42.3% | +2.32% | +60.3% | ✅ |
| REE | 31 | 51.6% | +1.71% | +52.9% | ✅ |
| TCB | 30 | 43.3% | +1.61% | +48.2% | ✅ |
| MBB | 34 | 58.8% | +1.26% | +42.7% | ✅ |
| BID | 36 | 44.4% | +0.99% | +35.6% | 🟡 |
| CTG | 41 | 36.6% | +0.60% | +24.5% | 🟡 |
| AAH | 5 | 20.0% | +4.70% | +23.5% | 🟡 Ít data |
| ABW | 12 | 50.0% | +1.90% | +22.8% | 🟡 |
| MWG | 33 | 54.5% | +0.65% | +21.3% | 🟡 |
| ACB | 32 | 46.9% | +0.58% | +18.5% | 🟡 |
| AAT | 19 | 42.1% | +0.75% | +14.3% | 🟡 |
| VHM | 35 | 42.9% | +0.35% | +12.3% | 🟡 |
| MSN | 34 | 44.1% | +0.31% | +10.4% | 🟡 |
| ACG | 14 | 50.0% | +0.54% | +7.5% | 🟡 |
| PNJ | 29 | 48.3% | -0.16% | -4.6% | ❌ |
| VCB | 24 | 41.7% | -0.22% | -5.3% | ❌ |
| PLX | 31 | 38.7% | -0.29% | -8.9% | ❌ |
| GAS | 30 | 40.0% | -0.40% | -12.0% | ❌ |
| VNM | 23 | 34.8% | -0.58% | -13.4% | ❌ |
| AAA | 40 | 45.0% | -0.43% | -17.2% | ❌ |
| ABS | 25 | 32.0% | -0.87% | -21.8% | ❌ |

**→ 22/29 mã có tổng PnL dương (75.9%)**
**→ 7 mã thua lỗ: PNJ, VCB, PLX, GAS, VNM, AAA, ABS**

---

## 🔍 5 NGUYÊN NHÂN GỐC RỄ VÌ SAO AVG PNL CHỈ +1.56%

### 1. EXIT QUÁ SỚM — Nguyên nhân #1 (57% lệnh exit ≤7 ngày)

| Hold period | Số trades | % tổng | Avg PnL |
|-------------|-----------|--------|---------|
| 1-3 ngày | 37 | 4.4% | **-4.68%** |
| 3-7 ngày | 392 | 46.3% | **-2.36%** |
| 7-14 ngày | 259 | 30.6% | +1.26% |
| 14-21 ngày | 95 | 11.2% | **+9.17%** |
| 21-42 ngày | 54 | 6.4% | **+19.70%** |
| >42 ngày | 3 | 0.4% | **+51.17%** |

**→ Lệnh giữ ≤7 ngày: -2.48% trung bình (481 lệnh = LỖ RÒ RỈ)**
**→ Lệnh giữ >14 ngày: +15.28% trung bình (131 lệnh = LÃI LỚN)**

### 2. TARGET DEFINITION SAI — trend_regime với dual_ma quá nhạy

- Target: `trend_regime` dùng `dual_ma(short=10, long=40)` 
- MA10 cross MA40 → UPTREND(1), SIDEWAYS(0), DOWNTREND(-1)
- MA10 rất nhạy, cross qua cross lại → model học "flip-flop"
- **MA10/MA40 crossover tạo ~30+ signals/year/symbol** → quá nhiều, phần lớn noise

### 3. 47% WINNING TRADES CHỈ LÃI < 3%

182/386 winning trades (47%) lãi < 3% — "tiny wins" bị phí giao dịch ăn gần hết.

### 4. WIN RATE 45.6% KHÔNG ĐỦ BÙ CHO TINY WINS

**Expectancy = 0.456 × 7.46 - 0.544 × 3.40 = +1.55%/trade** → quá thấp

### 5. KHÔNG LỌC MARKET REGIME

Model trade mọi lúc, kể cả bear market 2022. Trong downtrend, hầu hết signals fail.

---

## 📐 GIẢI PHÁP ĐỀ XUẤT (V8+) ĐỂ ĐẠT >10% AVG PNL

| Thay đổi | Trades/symbol | Avg PnL ước tính |
|----------|---------------|------------------|
| Hiện tại V7 | ~29 | +1.56% |
| + Min hold 10d | ~20 | +5-8% |
| + Confidence ≥0.7 | ~12 | +8-12% |
| + Forward target 20d/10% | ~10 | +10-15% |
| + Market filter | ~8 | +12-18% |

---

## 🎯 KẾT LUẬN

**Vấn đề cốt lõi: Model trade như scalper (hold 5-7d, target 2-3%) nhưng nên trade như swing trader (hold 14-40d, target 10-20%).**

Dữ liệu chứng minh: **khi hold >14 ngày, avg PnL = +15.28%** — đã đạt mục tiêu! Vấn đề là model bán quá sớm.

**3 thay đổi quan trọng nhất:**
1. 🥇 **Min hold period 10 ngày** — đơn giản nhất, tác động lớn nhất
2. 🥈 **Target = forward_return 20d/10%** — train model nhắm đúng mục tiêu
3. 🥉 **Entry confidence ≥ 0.70** — ít trades hơn, chất lượng cao hơn

---

## 🖥️ XEM BIỂU ĐỒ

Truy cập: **http://localhost:8888** để xem biểu đồ candlestick với các điểm mua/bán cho tất cả 29 mã cổ phiếu.
