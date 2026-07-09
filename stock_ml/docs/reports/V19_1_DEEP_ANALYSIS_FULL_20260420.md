# V19.1 PHAN TICH SAU - BAO CAO TOAN DIEN
# Ngay: 2026-04-20

---

## 1. TONG QUAN HIEU SUAT

| Model | Trades | WR | AvgPnL | TotalPnL | PF | MaxLoss | AvgHold |
|-------|--------|-----|--------|----------|-----|---------|---------|
| V11 | 419 | 46.8% | +3.22% | +1350.8% | 2.26 | -39.0% | 17.4d |
| V17 | 418 | 45.9% | +3.96% | +1654.6% | 2.53 | -23.6% | 18.8d |
| V19 | 427 | 44.5% | +4.37% | +1864.2% | 2.64 | -26.2% | 19.5d |
| **V19.1** | **419** | **44.9%** | **+4.46%** | **+1866.8%** | **2.68** | **-28.6%** | **20.3d** |
| Rule | 585 | 40.3% | +3.39% | +1982.6% | 2.21 | -27.2% | 19.3d |

**Ket luan:** V19.1 co PF cao nhat (2.68) va AvgPnL tot nhat (+4.46%), nhung Rule thang tong TotalPnL (+1982.6% vs +1866.8%) nho co nhieu giao dich hon (585 vs 419). Chenh lech: **-115.8%**.

---

## 2. DIEM MANH CUA V19.1

### 2.1. Peak Protection - Vu khi so 1
- `peak_protect_dist`: 32 trades, **96.9% WR**, avg +32.54%, tong **+1041.2%**
- `peak_protect_ema`: 9 trades, **100% WR**, avg +14.44%, tong **+129.9%**
- Day la module sinh loi nhieu nhat, bao ve loi nhuan khi gia dat dinh

### 2.2. Hieu suat theo xu huong
- Strong trend: 203 trades, PF=3.49, tong +1196.3% (chiem 64% tong loi nhuan)
- V-shape entries: 42 trades, 50% WR, +175.0%

### 2.3. Ma V19.1 thang Rule
| Ma | V19.1 | Rule | Gap | Ly do |
|----|-------|------|-----|-------|
| VIC | +249.0% | +152.5% | **+96.5%** | Peak protect giu loi nhuan lon |
| VNM | -0.6% | -53.5% | **+52.9%** | Rule giao dich qua nhieu trong bear |
| HPG | +155.9% | +137.1% | **+18.8%** | Strong trend carry tot |
| BID | +54.2% | +42.0% | **+12.2%** | Regime adapter giup bank profile |
| AAV | +323.1% | +318.5% | +4.6% | PF=4.63 tot nhat trong tat ca |

---

## 3. PHAN TICH DIEM YEU - NGUYEN NHAN GOC RE

### 3.1. VAN DE #1: Signal Exit - "Cai chet cham" (ROOT CAUSE CHINH)

**Du lieu:**
- `signal` exit: **336/419 trades** (80% tong giao dich)
- Win rate chi **34.5%** - cuc ky thap
- Tong loi: +95.2% (trong khi tong lo la -1069.4%)
- Nhieu lenh lo lon: VND -28.6%, -26.9%; SSI -18.5%; DGC -14.3%

**Nguyen nhan ky thuat (tu source code):**
1. Bearish score threshold qua cao -> giu lenh thua qua lau
   - Base threshold: 2.0-2.35 tuy profile
   - Cong them +0.7 khi strong trend & cum_ret > 6%
   - Cong them +0.4 khi hold < 7 ngay
   - Chi tru -0.4 khi cum_ret < -3%
   => Tong threshold co the len 3.45 -> can gan nhu tat ca dieu kien bearish moi duoc thoat

2. EXIT_CONFIRM = 3 bars -> cho 3 ngay xac nhan lien tiep
   - Khi da lo, cho them 3 ngay = lo them 3-5%
   - Regime adapter chi giam xuong 2 cho cum_ret < -3%

3. Strong trend override (line 706-708): Neu cum_ret > 0 va trend == "strong" -> KHONG BAO GIO THOAT
   - Nhieu lenh vao strong trend roi trend dao chieu, nhung vi lagging indicators (SMA20/50) van bao "strong"

**Vi du cu the - VND:**
- Trade 2021-01-18 -> 2021-02-01: **-28.59%** (signal exit, moderate trend, 10 ngay)
  - Day chi la 2 tuan truoc khi VND tang manh tu 2021-02-17 (+147% theo Rule)
  - V19.1 vao sai thoi diem, giu qua lau vi bearish_score chua du threshold
- Trade 2022-06-02 -> 2022-06-16: **-26.93%** (signal exit, weak trend)
  - Vao trong weak trend voi score 4, size 0.45 -> van lo lon

### 3.2. VAN DE #2: Rule bat duoc nhieu xu huong hon

**Du lieu:** Rule co 585 trades vs V19.1 chi 419 (gap = 166 trades)

**Ma bi anh huong nang nhat:**

#### DGC: V19.1 +183.0% vs Rule +266.6% (gap = -83.6%)
- Rule co 42 trades vs V19.1 chi 31 trades
- 4 giao dich Rule lai lon ma V19.1 bi lot hoac lo:
  - 2021-02-08 -> 2021-03-22: Rule +26.8% (V19.1 underperform)
  - 2022-05-24 -> 2022-06-21: Rule +13.0% (V19.1 underperform)
  - 2024-02-06 -> 2024-04-01: Rule +25.3% (V19.1 underperform)
  - 2024-12-03 -> 2025-01-03: Rule +4.6% (V19.1 underperform)
- V19.1 co 13 lenh lo > 3%, worst: -14.28%, -11.18%, -8.89%
- **Nguyen nhan:** DGC la "momentum" profile (size_mult=0.92) nhung entry filter qua chat, bo lo nhieu diem vao som

#### AAS: V19.1 +171.1% vs Rule +253.7% (gap = -82.6%)
- 5 giao dich Rule lai ma V19.1 missed/underperform:
  - 2021-05-21 -> 2021-07-06: Rule **+58.0%** (gap cuc lon)
  - 2024-12-09 -> 2025-01-15: Rule +24.1%
  - 2025-05-05 -> 2025-06-09: Rule +15.5%
- V19.1 co 12 lenh lo > 3%
- **Nguyen nhan:** AAS la "high_beta" - V19.1 giam size (size_mult=0.98) nhung van vao nhieu lenh nhieu roi thoat bang signal

#### VND: V19.1 +253.6% vs Rule +302.1% (gap = -48.5%)
- Rule bat trend 2022-11-18 -> 2022-12-23: +25.6%
  - V19.1 vao muon (2022-12-14) va lo -9.7%
- **Dac biet:** V17 dat +278.3% voi WR=61.1% tren VND (vs V19.1 chi 46.9%)
  - V17 don gian hon (khong co regime adapter) nhung giu lenh thang lau hon

#### SSI: V19.1 +207.1% vs Rule +231.8% (gap = -24.6%)
- 3 giao dich Rule tot ma V19.1 missed
- Worst trade: 2022-01-06: **-18.48%** (weak trend, score 4, full size 0.686)
- **Nguyen nhan:** Position size 60-80% trong weak trend van qua lon

### 3.3. VAN DE #3: Trend detection lagging

**Bang chung:**
- Nhieu lenh lo nang (-10% den -28%) vao trong "strong" trend
  - FPT -9.97% (strong trend), -9.74% (strong trend)
  - HPG -9.67% (weak), -8.53% (moderate)
  - DGC -14.28% (moderate), -11.18% (strong)
- `detect_trend_strength()` dung SMA20/50, MACD, days_above_ma20 -> tat ca lagging 5-20 ngay
- Khi trend dao chieu, cac indicator nay van bao "strong" -> model khong thoat

### 3.4. VAN DE #4: Position size bucket 60-80% yeu nhat
- 67 trades, WR=34.3%, PF=1.94
- Day la cac lenh "trung binh conviction" - khong du tot de vao full, nhung van vao qua lon
- So sanh: bucket 80-100% lai tot (PF=3.87, WR=48.7%) - nghich ly

### 3.5. VAN DE #5: V11/V17 tot hon V19.1 o mot so ma

| Ma | V19.1 | Model tot hon | Gap | Nguyen nhan |
|----|-------|---------------|-----|-------------|
| ACB | +26.2% | V11: +59.8% | -33.6% | V19.1 filter qua chat, V11 tho nhung trade nhieu |
| VND | +253.6% | V17: +278.3% | -24.7% | V17 exit don gian hon, giu winner lau hon |
| REE | +19.3% | V17: +52.3% | -33.0% | V17 WR 40.7% vs V19.1 43.3% nhung avg cao hon |
| VNM | -0.6% | V17: +16.1% | -16.7% | V17 it filter hon nen bat duoc rally |
| HPG | +155.9% | V17: +182.9% | -27.0% | V17 giu lenh thang lau hon |

**Ket luan:** V19.1 over-engineer exit logic -> mat loi nhuan o cac ma don gian

---

## 4. PHAN TICH ENTRY TYPE

| Entry Type | Trades | WR | AvgPnL | TotalPnL | PF |
|------------|--------|-----|--------|----------|-----|
| quick_reentry | 3 | 66.7% | +9.69% | +29.1% | 4.81 |
| breakout_entry | 33 | 27.3% | +2.59% | +85.5% | 2.07 |
| vshape_entry | 42 | 50.0% | +4.17% | +175.0% | 2.47 |
| normal_ml | 341 | 45.7% | +4.63% | +1577.2% | 2.75 |

**Nhan xet:**
- `breakout_entry` yeu nhat: WR chi 27.3% - BO Quality Filter (mod_f) co the qua loose
- `normal_ml` chiem chu dao va hieu qua tot nhat (PF=2.75)
- `quick_reentry` tuyet voi nhung qua it (chi 3 trades) - can mo rong dieu kien

---

## 5. DE XUAT CAI TIEN CHO V20

### A. SIGNAL EXIT OVERHAUL (Uu tien #1 - Impact cao nhat)

**Hien tai:** bearish_score >= threshold (2.0-3.45) -> qua phuc tap va cham

**De xuat:**
```
1. FAST EXIT khi dang lo:
   - Neu cum_ret < -5% va hold > 3 ngay -> EXIT NGAY (khong can confirm)
   - Neu cum_ret < -3% va MACD histogram < 0 va close < EMA8 -> EXIT (1 bar confirm)

2. TIME DECAY cho signal exit:
   - Neu hold > 15 ngay va cum_ret < 3% -> giam threshold 40%
   - Neu hold > 20 ngay va cum_ret < 5% -> giam threshold 60%

3. HARD CAP cho signal losses:
   - Max loss tu signal exit: -15% (hien tai -28.6%)
   - Khi cum_ret < -12% -> exit bat ke bearish_score

4. Don gian hoa bearish confirm:
   - Primary: MACD histogram < 0 VA close < EMA8 -> exit
   - Bo confirm bars khi dang lo (cum_ret < 0)
```

### B. ENTRY TIMING - Dong bo voi Rule (Uu tien #2)

**Van de:** Rule bat 166 trades nhieu hon -> bat nhieu xu huong hon

**De xuat:**
```
1. HYBRID ENTRY: Khi Rule signal (SMA20 cross > SMA50 voi volume) VA ML score > 0.4
   -> Vao voi full size, khong can entry_alpha check

2. TREND START DETECTOR moi:
   - SMA20 vua cat len SMA50 trong 5 ngay gan nhat
   - Volume > 1.5x avg20 it nhat 1 ngay
   - Close > SMA20
   -> Vao nhu breakout_entry voi 70% size

3. Giam min_score trong macro uptrend:
   - Khi SMA20 > SMA50 lien tuc > 20 ngay -> min_score = 1 cho moi trend
   - Hien tai min_score = 3 cho weak trend -> bo lo nhieu co hoi
```

### C. SYMBOL-SPECIFIC FIXES (Uu tien #3)

```
1. VND/SSI (high_beta):
   - Hard stop tai -15% (hien tai HARD_STOP = 8% nhung signal exit cho lo den -28%)
   - Giam MIN_HOLD tu 6 -> 4 ngay khi trend != strong
   - exit_score_threshold giam 0.3 khi cum_ret < 0

2. ACB:
   - Noi long entry filter (V11 khong filter dat +59.8% vs V19.1 +26.2%)
   - Giam min_score xuong 1 cho bank profile khi macro uptrend
   - Tang so luong trades len ~40 (hien tai chi 30)

3. DGC/AAS (momentum/high_beta):
   - Cho phep nhieu entries hon bang cach:
     - Giam dp_floor tu 0.018/0.015 -> 0.012 khi strong trend
     - Bo anti-chop filter khi SMA20 > SMA50 confirmed > 10 ngay
```

### D. POSITION SIZING OVERHAUL (Uu tien #4)

**Van de:** Bucket 60-80% worst (PF=1.94), trong khi 80-100% tot (PF=3.87)

**De xuat:**
```
1. BINARY SIZING thay vi continuous:
   - High conviction (score >= 4 + strong trend): 90-100%
   - Medium conviction: 50%
   - Low conviction: 30% hoac KHONG VAO
   -> Loai bo vung 60-80% yeu

2. ATR-BASED SIZE CAP:
   - Khi ATR/price > 4%: max size = 50%
   - Khi ATR/price > 5.5%: max size = 35%
   - Hien tai chi giam size_mult 0.85-0.9 -> chua du

3. WEAK TREND SIZE REDUCTION:
   - Weak trend: max 40% (hien tai co the len 70%)
   - Moderate trend: max 70%
```

### E. MO HINH V20 - Y TUONG DOT PHA

```
1. ENSEMBLE VOTING:
   - Chay V11, V17, V19.1 song song
   - Vao khi >= 2/3 models dong y
   - Size = ty le dong y (2/3 = 67%, 3/3 = 100%)
   - Du kien: WR > 50%, PF > 3.0

2. RULE-ML HYBRID:
   - Entry: Rule signal (SMA crossover) lam trigger chinh
   - Size: ML confidence score quyet dinh
   - Exit: V19.1 peak protect + don gian hoa signal exit
   - Du kien: 500+ trades, PF > 2.5

3. REGIME-SWITCHING:
   - Train 3 model rieng: bull / bear / sideways
   - Regime classifier dua tren VN-Index SMA50 + breadth
   - Bull: aggressive entry, wide trail
   - Bear: chi V-shape + breakout, tight stop
   - Sideways: mean-reversion thay vi trend-following

4. SHORTER HOLDING IN WEAK TREND:
   - Target 5-8 ngay swing trade thay vi trend-following
   - Exit sau 8 ngay neu cum_ret < 5%
   - Giam ZOMBIE_BARS tu 14 -> 8 cho weak trend
```

---

## 6. MUC TIEU V20

| Metric | V19.1 | Target V20 | Cach dat |
|--------|-------|------------|----------|
| Total PnL | +1866.8% | >+2500% | Bat nhieu trades + giam lo |
| Win Rate | 44.9% | >50% | Ensemble voting + better entry |
| Profit Factor | 2.68 | >3.2 | Fast exit khi lo + giu winner |
| Max Single Loss | -28.6% | <-12% | Hard cap -12% cho signal exit |
| Signal Exit WR | 34.5% | >45% | Overhaul bearish score system |
| Trades | 419 | 500-550 | Hybrid entry + lower min_score |
| Avg Hold (losers) | ~10d | <6d | Fast loss cut + time decay |

---

## 7. THU TU THUC HIEN

1. **V19.2**: Signal exit hard cap -12% + fast exit khi lo (1-2 ngay)
2. **V19.3**: Position sizing binary + ATR cap (1 ngay)
3. **V19.4**: Symbol-specific fixes VND/ACB/DGC (1 ngay)
4. **V19.5**: Hybrid entry voi Rule signal (2 ngay)
5. **V20.0**: Ensemble voting 3 models (3-5 ngay)

Moi buoc chay backtest va so sanh voi V19.1 baseline.

---

## 8. TOM TAT

**V19.1 la model tot nhat hien tai** voi PF=2.68, nhung con **3 van de chinh**:
1. Signal exit qua cham -> lo lon (-28.6% max) -> Can hard cap + fast exit
2. It trades hon Rule 166 lenh -> mat loi nhuan -> Can hybrid entry
3. Over-engineering exit logic -> V17 don gian hon lai tot hon o nhieu ma -> Can don gian hoa

**Impact du kien neu fix ca 3:** +400-600% tong PnL them, dat >+2500%.
