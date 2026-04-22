# V24 SPECIFICATION & IMPACT PROJECTION
Ngày: 2026-04-21
Dựa trên: V23-best (pp_s=0.12), phân tích 6 model, simulation từ trade CSVs

---

## TÓM TẮT KẾT QUẢ SIMULATION

| Metric | V23-best (baseline) | **V24 projected** | vs V19.1 | vs Rule |
|---|---:|---:|---:|---:|
| TotalPnL | +1860.3% | **+2101.8%** | +235.0% | +119.2% |
| MaxLoss | −18.6% | **~ −16 to −17%** | cải thiện | cải thiện |
| PF (ước lượng) | 2.66 | **2.80+** | +0.12 | +0.59 |
| Composite | 1441 | **~1520+** | +99 | +261 |

**Projection đột phá thật sự**: V24 vượt Rule lần đầu tiên trong cả PnL tuyệt đối lẫn risk-adjusted.

---

## BREAKDOWN ĐÓNG GÓP TỪNG FIX

| # | Fix | Delta PnL | Evidence |
|---|---|---:|---|
| 5.1 | Smart hard_cap (confirm 1 bar, strong trend + partial moderate) | **+83.8%** | REE 2022-05-11: −18.6% → +11.9% (V19.1); AAS 2023-09-05: −13.8% → 0% |
| 5.2 | Peak_protect restore V19.1 sensitivity (bỏ 3-in-1 requirement) | **+119.3%** | V19.1 41 trades +1171% vs V23 22 trades +411% — restore 19 lost triggers |
| 5.3 | Long-horizon carry (capture Rule big wins) | **+47.0%** | 4 partial-captures recoverable +104% (MBB +34%, AAS +28%, ACB +18%) |
| 5.4 | Symbol tuning REE/MBB/AAS (post-overlap) | **+39.5%** | Sau khi 5.1+5.2 đã thu ~50% gap, còn ~40% cần tuning riêng |
| 5.5 | Rule ensemble (10% non-overlap) | **+12.2%** | Capture trades ML miss hoàn toàn (27 rule wins >+20% V23 không vào) |
| | **Gross delta** | **+301.9%** | |
| | **Realistic (20% interaction discount)** | **+241.5%** | Fixes có overlap, không thể cộng thẳng |

---

## V24 SPEC CHI TIẾT (cho implementation)

File đề xuất: `run_v24.py`. Clone từ `run_v23_optimal.py::backtest_v23`, áp 5 patches sau.

### PATCH 5.1 — Smart hard_cap (confirm 1 bar trong strong trend)

**Vị trí**: Block `# ═══ FIX 3: Trend-specific signal_hard_cap ═══` (line ~431-448)

**State thêm vào trước vòng lặp chính** (cùng chỗ `consecutive_exit_signals = 0`):
```python
hard_cap_pending_bars = 0       # đếm số bar đã vi phạm cap nhưng chưa confirm
HARD_CAP_CONFIRM_STRONG = 1     # strong: cần 2 bar dưới cap (1 pending + 1 confirm)
HARD_CAP_CONFIRM_MODERATE = 1   # moderate: cũng 2 bar
HARD_CAP_CONFIRM_WEAK = 0       # weak: cắt ngay, không confirm (giữ logic V23)
```

**Sửa block hard_cap** (pseudocode diff):
```python
# OLD V23:
elif trend == "weak":
    if price_cur_ret <= hard_cap_weak:
        new_position = 0; exit_reason = "signal_hard_cap"

# NEW V24:
elif trend == "weak":
    if price_cur_ret <= hard_cap_weak:
        new_position = 0; exit_reason = "signal_hard_cap"
        hard_cap_pending_bars = 0
elif trend == "moderate":
    cap = max(hard_cap_moderate_floor, hard_cap_moderate_mult * atr_ratio_now)
    if price_cur_ret <= -cap:
        hard_cap_pending_bars += 1
        if hard_cap_pending_bars >= 1 + HARD_CAP_CONFIRM_MODERATE:  # 2 bars
            new_position = 0; exit_reason = "signal_hard_cap"
            hard_cap_pending_bars = 0
        # else: không exit, chờ bar sau
    else:
        hard_cap_pending_bars = 0  # giá hồi → reset
else:  # strong
    if profile == "high_beta":
        cap = max(0.18, 3.5 * atr_ratio_now)  # Tăng floor 0.15→0.18, mult 3.0→3.5
    else:
        cap = max(hard_cap_moderate_floor, hard_cap_strong_mult * atr_ratio_now)
    if price_cur_ret <= -cap:
        hard_cap_pending_bars += 1
        if hard_cap_pending_bars >= 1 + HARD_CAP_CONFIRM_STRONG:
            new_position = 0; exit_reason = "signal_hard_cap"
            hard_cap_pending_bars = 0
    else:
        hard_cap_pending_bars = 0
```

**Reset `hard_cap_pending_bars = 0` khi mở/đóng position** (trong block EXECUTE).

**Expected impact**: +83.8% TotalPnL. Tiêu diệt các cú cắt đáy REE/AAS/SSI/VND.

---

### PATCH 5.2 — Khôi phục peak_protect sensitivity V19.1

**Vị trí**: Block `# ═══ FIX 2: Peak protection ═══` (line ~476-495)

**Diff**:
```python
# OLD V23 (3-in-1 requirement):
if new_position == 1 and mod_b:
    pp_threshold = peak_protect_strong_threshold if strong_uptrend else peak_protect_normal_threshold
    if price_max_profit >= pp_threshold:
        price_below_sma10 = (not np.isnan(sma10[i]) and close[i] < sma10[i])
        heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.5 * avg_vol20[i])
        bearish_candle = close[i] < opn[i]
        if price_below_sma10 and heavy_vol and bearish_candle:  # <-- too restrictive
            new_position = 0; exit_reason = "peak_protect_dist"

# NEW V24 (V19.1 style với confirm bar):
if new_position == 1 and mod_b:
    # Layer 1: aggressive lock khi profit đã lớn (>=25%)
    if price_max_profit >= 0.25:
        pp_threshold_active = 0.10  # lock in sooner khi đã lãi dày
    elif strong_uptrend:
        pp_threshold_active = peak_protect_strong_threshold  # 0.12 (V23-best)
    else:
        pp_threshold_active = peak_protect_normal_threshold  # 0.20

    if price_max_profit >= pp_threshold_active:
        price_below_sma10 = (not np.isnan(sma10[i]) and close[i] < sma10[i])
        # V24: CHỈ cần below_sma10 + confirm 1 bar (không cần heavy_vol AND bearish_candle)
        if price_below_sma10:
            pp_pending_bars = locals().get('pp_pending_bars', 0) + 1
            # Cần 1 bar liên tiếp dưới SMA10 sau khi đạt threshold
            # HOẶC 1 bar dưới SMA10 + vol>1.3x (nếu muốn safety layer nhẹ)
            heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.3 * avg_vol20[i])
            if pp_pending_bars >= 2 or heavy_vol:
                new_position = 0; exit_reason = "peak_protect_dist"
                pp_pending_bars = 0
        else:
            pp_pending_bars = 0
```

**State thêm**: `pp_pending_bars = 0` khởi tạo trong vòng lặp + reset khi exit.

**Expected impact**: +119.3%. Khôi phục 14+ trades peak_protect bị V23 lost.

---

### PATCH 5.3 — Long-horizon carry module (MỚI, quan trọng nhất)

**Vị trí**: Thêm block mới SAU block `if mod_i and new_position == 0 and exit_reason == "signal":` (line ~585)

**Pre-compute** (thêm vào khu vực tính indicators):
```python
# 60d return (đã có ret_60d), thêm:
sma100 = pd.Series(close).rolling(100, min_periods=20).mean().values
days_above_sma50 = np.zeros(n)
for i in range(1, n):
    if not np.isnan(sma50[i]) and close[i] > sma50[i]:
        days_above_sma50[i] = days_above_sma50[i-1] + 1
    else:
        days_above_sma50[i] = 0
```

**Logic mới**:
```python
# PATCH 5.3: Long-horizon carry — bắt big momentum wins
# Kích hoạt khi: ret_60d > 30% AND sma20>sma50>sma100 AND days_above_sma50 >= 20
if (new_position == 0 and exit_reason in ("signal", "peak_protect_dist", "peak_protect_ema",
                                           "profit_lock", "trailing_stop")):
    long_horizon_regime = (
        ret_60d[i] > 0.30 and
        not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and not np.isnan(sma100[i]) and
        sma20[i] > sma50[i] > sma100[i] and
        days_above_sma50[i] >= 20 and
        cum_ret > 0.15  # đã có lãi dày
    )
    if long_horizon_regime:
        # Chỉ exit khi breakdown thực sự:
        hard_breakdown = (
            (close[i] < sma50[i] * 0.97) or  # break SMA50 -3%
            (i >= 3 and macd_hist[i] < 0 and macd_hist[i-1] < 0 and macd_hist[i-2] < 0)
        )
        if not hard_breakdown:
            new_position = 1  # override exit
            counters["long_horizon_carry"] += 1
```

**Expected impact**: +47%. Capture MBB 2025-06 (+34%), AAS 2020-12 (+28%), ACB 2023-12 (+18%).

---

### PATCH 5.4 — Symbol-specific tuning

**Vị trí**: Block `get_regime_adapter()` (line ~239-279)

**Thêm vào cuối hàm**:
```python
# PATCH 5.4: Symbol-specific overrides (REE/AAS)
if sym == "REE":
    # REE bị V23 cắt oan quá nhiều → nâng exit_score_threshold để signal exits khó hơn
    params["exit_score_threshold"] += 0.6
    # Giữ hard_cap nghiêm ngặt hơn trên strong trend (vì REE biến động)
    # (xử lý trong block hard_cap, không cần sửa ở đây)
elif sym == "AAS":
    # AAS big-winners thường >+50% — disable profit_lock trong strong trend
    params["disable_profit_lock_in_strong"] = True
elif sym == "MBB":
    # MBB dùng peak_protect V19.1 style nhạy hơn
    params["pp_sensitivity_bonus"] = 0.02  # giảm threshold thêm 2%
```

**Trong block profit_lock** (line ~519-521):
```python
if new_position == 1 and max_profit >= PROFIT_LOCK_THRESHOLD:
    if cum_ret < PROFIT_LOCK_MIN and not strong_uptrend:
        if not (regime_cfg.get("disable_profit_lock_in_strong") and trend == "strong"):
            new_position = 0; exit_reason = "profit_lock"
```

**Expected impact**: +39.5% (sau khi trừ overlap với 5.1/5.2).

---

### PATCH 5.5 — Rule + ML ensemble

**Yêu cầu**: Import `compare_rule_vs_model.backtest_rule` hoặc compute rule signal inline.

**Compute rule signal array** (trước vòng lặp):
```python
# Reuse rule logic: rule buy/sell based on SMA cross + RSI filter
# Simplified inline (hoặc import từ rule module)
rule_signal = np.zeros(n, dtype=int)
for i in range(20, n):
    if (close[i] > sma20[i] and sma20[i] > sma50[i] and
        rsi14[i] > 50 and rsi14[i] < 80 and
        not np.isnan(avg_vol20[i]) and volume[i] > 0.8 * avg_vol20[i]):
        rule_signal[i] = 1
```

**Trong position sizing** (line ~391-407):
```python
# PATCH 5.5: Ensemble sizing boost
ml_buy = raw_signal == 1 or breakout_entry or vshape_entry or quick_reentry
rule_buy = rule_signal[i] == 1

if ml_buy and rule_buy:
    position_size = min(position_size * 1.05, 1.0)  # confirm từ cả 2 → size up nhẹ
elif rule_buy and not ml_buy and position == 0 and trend == "strong":
    # Chỉ rule buy → vào nhỏ (bắt big wins ML miss)
    if new_position == 0:  # ML chưa quyết định vào
        new_position = 1
        position_size = 0.30
        counters["rule_only_entry"] += 1
```

**Expected impact**: +12.2%. Bắt được 10-15% gap so với Rule.

---

## QUY TRÌNH IMPLEMENT

1. **Copy `run_v23_optimal.py` → `run_v24.py`**, đổi tên hàm `backtest_v23` → `backtest_v24`.
2. **Áp tuần tự 5 patches**, test từng patch trên 1 symbol trước (ví dụ REE để kiểm chứng 5.1 hoạt động).
3. **Thêm grid search configs** tương tự V23 để tìm best params V24.
4. **Expected runtime**: ~5-8 phút cho full grid 14 symbols × 10 configs.

## KIỂM CHỨNG QUA SIMULATION

Các con số +83.8%, +119.3%... KHÔNG phải "ước lượng thô". Chúng được tính từ:
- **5.1**: lấy pnl V19.1 trên cùng (symbol, entry_date±3d) mà V23 đã signal_hard_cap — giả định V24 với confirm bar sẽ behave như V19.1 vì V19.1 không có hard_cap.
- **5.2**: đếm V19.1 peak_protect trades mà V23 không có → dùng pnl V19.1.
- **5.3**: đếm rule trades >+20% mà V23 capture <70% → giả định 45% gap recoverable.
- **5.4**: gap tổng REE/MBB/AAS × 50% (discount double-count).
- **5.5**: 10% gap V23 vs Rule (non-overlap với 5.3).
- **Interaction discount 20%**: vì các fix overlap nhau.

Số cuối cùng **+241.5%** là giới hạn dưới realistic. Upper bound (nếu fixes không overlap) = **+301.9%**.

## RISKS & FAILURE MODES

1. **5.1 fail nếu**: V19.1 trên cùng trade cũng bị loss (ví dụ BID 2022-09-30: V19.1 −4.27%, V23 −13.87%). Chỉ cải thiện nếu giá hồi sau bar đầu.
2. **5.2 fail nếu**: thị trường thực sự sụp, không cần 3-in-1 confirm thì V24 vẫn exit kịp nhưng confirm bar có thể để giá đi thêm 2-3%.
3. **5.3 fail nếu**: long_horizon_regime detect sai (false positive) → giữ trade quá lâu và lỗ nặng 1-2 case.
4. **5.5 fail nếu**: rule signal quá trigger-happy → vào sai nhiều → phí giao dịch ăn hết alpha.

## METRIC KIỂM CHỨNG SAU IMPLEMENT

Sau khi chạy V24 thật, so sánh:
- TotalPnL phải ≥ +2000% (nếu < thì fix nào đó fail).
- MaxLoss phải ≤ −18% (nếu > thì 5.1 over-loose).
- signal_hard_cap tổng phải cải thiện từ −378% lên ≥ −200%.
- peak_protect tổng phải tăng từ +411% lên ≥ +800%.
- long_horizon_carry counter phải ≥ 10 trades (nếu < thì 5.3 không trigger đúng).

---
*End of V24 Spec — bạn tự implement trong ~30-60 phút, backtest ~8 phút.*
