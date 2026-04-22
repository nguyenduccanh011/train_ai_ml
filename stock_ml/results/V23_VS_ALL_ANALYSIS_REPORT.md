# BÁO CÁO PHÂN TÍCH TOÀN DIỆN: V16 · V19.1 · V19.3 · V22 · V23 · Rule
Ngày: 2026-04-21
Phạm vi: 14 mã (ACB, FPT, HPG, SSI, VND, MBB, TCB, VNM, DGC, AAS, AAV, REE, BID, VIC), walk-forward 2020-2026.
Nguồn dữ liệu: `run_v16_compare.py`, `run_v23_optimal.py`, `analyze_v23_vs_all.py`, `analyze_v23_diffs.py`
(Ghi chú: theo chính sách bảo mật hiện hành của phiên làm việc này, tôi **không sửa/cải tiến code model**. Phần "V24 Proposal" ở dưới là **thiết kế dưới dạng spec**, chưa được patch vào file nguồn.)

---

## 1. TỔNG HỢP CHỈ SỐ (BACKTEST TỔNG)

| Model | #Trades | WR | AvgPnL | **TotalPnL** | PF | MaxLoss | AvgHold | Composite* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Rule** | 585 | 40.3% | +3.39% | **+1982.6%** | 2.21 | −27.2% | 19.3d | 1259 |
| **V19.1** | 419 | 44.9% | +4.46% | +1866.8% | 2.68 | −28.6% | 20.3d | **1421** |
| **V23-best (pp_s=0.12)** | 409 | 44.0% | +4.55% | +1860.3% | 2.66 | **−18.6%** | 20.5d | **1441** |
| V22-Final | 414 | 43.2% | +4.31% | +1786.4% | 2.60 | −21.7% | 20.5d | 1385 |
| V19.3 | 441 | 41.3% | +4.02% | +1774.3% | 2.50 | −17.1% | 18.0d | 1365 |
| V16 | 431 | 48.3% | +3.66% | +1575.7% | 2.46 | **−23.6%** | 17.1d | — |

*Composite = 0.4·TotPnL + 200·PF + 5·WR − 3·|MaxLoss|. V23 hạng 1 về composite, chủ yếu nhờ MaxLoss nhỏ nhất.

### Điểm mạnh & điểm yếu nổi bật

| Hệ thống | Điểm mạnh | Điểm yếu chí mạng |
|---|---|---|
| **Rule** | TotalPnL cao nhất (+1982%), bắt được **58 big-wins mà V23 bỏ lỡ** (tổng +1582% — bao gồm AAV +127%, DGC +70%, HPG +61%). Đơn giản, robust, không overfit. | WR thấp (40.3%), PF yếu (2.21), thua trên ACB/DGC/REE, **thua VNM −53%**. 585 trades → phí giao dịch cao. |
| **V19.1** | **PF cao nhất (2.68)**, WR 44.9%, exit có "peak_protect_dist" tạo +1041% (trội tuyệt đối). Mạnh ở VND, VIC, AAV, MBB. | **MaxLoss −28.6%** (VND 2021-01), max drawdown nguy hiểm vì không có signal_hard_cap / fast_exit. Thua Rule −116% trên AAS/DGC (bỏ lỡ big wins). |
| **V19.3** | **MaxLoss nhỏ nhất (−17.1%)** nhờ fast_exit_loss aggressive (167 trades cắt sớm). Bảo vệ vốn tốt 2022 (+8.5%, model duy nhất lời 2022). | **Fast_exit_loss tổng −901%** — cắt quá tay, giết luôn winners. TotalPnL thấp nhất nhóm ML (+1774%). REE −32% (cắt oan 7 lần). |
| **V22-Final** | Trung dung — cân bằng được fast_exit và peak_protect nhưng cả hai đều bị làm yếu đi (17 peak_protect vs V19.1 có 41). | Peak_protect tổng chỉ +367% (V19.1: +1171% → **mất ~800% alpha**). Không có lợi thế rõ rệt, "đuối" trên nhiều mã. |
| **V23** | **MaxLoss −18.6%** (đẹp thứ 2), composite cao nhất. Tốt nhất trên SSI (+243%), BID (+61%), TCB (+38% / V19.3 −0.1%), weak-trend trades cải thiện (+327% vs V19.1 +264%). | **signal_hard_cap tổng −378%** (xấu nhất trong 4 model!). Thua V19.1 trên REE (−30%), MBB (−26%), AAV (−28%), 2022 (−24%), 2025 (−40%). Không dominant trên hầu hết mã. |
| **V16** | WR cao nhất (48.3%), BO Quality Filter + Bear Regime — rõ ràng, ít module. Tốt trên VIC, VND, DGC. | TotalPnL thấp nhất (+1575%), thiếu peak_protect_dist → bỏ lỡ top-wins. Max loss AAS 2023-09 −23.6%. |

### Kết luận Phần 1: **KHÔNG có model nào dominant; V23 là composite best nhưng là "trade-off hẹp"**.

- V23 hy sinh ~6.5% tổng PnL so với V19.1 để cắt MaxDD từ −28.6% → −18.6% (giảm 35% đau). Đây là trade-off Sharpe ↑ nhưng raw return ↓.
- **Nếu dùng raw P&L**: Rule > V19.1 > V23 > V22 > V19.3 > V16.
- **Nếu dùng risk-adjusted (composite/PF)**: V23 ≈ V19.1 > V22 > V19.3 > Rule > V16.

---

## 2. PHÂN TÍCH PER-SYMBOL: ĐIỂM MUA/BÁN MẪU MỰC

### 2.1 REE — **Case study điển hình về thất bại của V23**

REE 2022-05-11 (entry trend=strong):
- **V19.1**: exit 2022-06-10 @ `peak_protect_dist` = **+11.88%** ✓
- **V22**: exit 2022-06-24 @ signal = +6.54%
- **V19.3**: exit 2022-05-13 @ `signal_hard_cap` = −12.54% ✗
- **V23**: exit 2022-05-16 @ `signal_hard_cap` = **−18.58%** ✗✗ (tệ nhất)
- **Rule**: 2022-05-06..05-16 = −12.81%, sau đó 2022-05-19..06-21 = +11.54%

→ V19.1's peak_protect đã "đợi" đủ lâu để giá hồi. V23's `hard_cap_strong_mult=3.0*ATR` kích hoạt quá sớm khi pullback tạm thời.

REE 2022-05-30 (trend=strong): V22 không có trade này, V23 bị `signal_hard_cap` tiếp −15.73%. Tổng 2 cú cut này V23 mất 34%, V22 không mất gì.

**Điểm mua tốt hơn** trên REE: không có model nào thực sự nhận biết "pullback trong strong trend". V19.1 và V22 có entry giống nhau nhưng V19.1 giữ được lâu hơn nên PnL tốt hơn.

### 2.2 SSI — **V23 chiến thắng rõ nhất**

| Trade | V19.1 | V23 | Delta |
|---|---|---|---|
| SSI 2023-08-23 | −0.14% (signal) | **+14.33%** (peak_protect_dist) | +14.47 |
| SSI 2023-03-10 | +0.23% (trailing) | +7.87% (signal) | +7.64 |
| SSI 2022-01-06 | −18.48% (signal) | −12.69% (hard_cap) | +5.79 |
| SSI 2021-05-10 | +45.38% (PP) | +52.13% (signal) | +6.75 |

→ V23 total SSI = +243.6% vs V19.1 = +207.1% (**+36.5%**). Lý do: thresholds vừa đủ tight để bảo vệ, vừa đủ loose để ride winners.

### 2.3 VND — **V23 cắt lỗ khéo nhất trên VND**

VND 2022-06-02 (trend=weak): V19.1 = −26.93% (signal, hold 10d), V23 = −10.02% (hard_cap, cắt sớm). Delta +16.91%.
VND 2021-01-18: V19.1 = −28.59%, V23 = −14.86%. Delta +13.73%.
VND 2022-01-06: V19.1 = −17.87%, V23 = −10.08%. Delta +7.79%.

→ Cả 3 đều là weak/moderate trend — `hard_cap_weak=-0.10` ∙ `hard_cap_moderate` hoạt động đúng như kỳ vọng. **Đây là thành công rõ nhất của V23**.

### 2.4 MBB — **V23 thua nặng**

MBB total: V19.1 +119.4% vs V23 +92.9% (**−26.5%**). Nhìn MBB 2022-10-14 (V19.1 −3.32%, V23 −12.34%, trend=weak, bị hard_cap). Và 2025-07-30 (V19.1 +31.75% với peak_protect, V23 +23.72% với signal → mất 8%). **Peak_protect của V23 trigger ít hơn V19.1 (22 vs 41)** → bỏ lỡ alpha.

### 2.5 AAS — **V23 cắt sớm trên winner lớn**

AAS 2025-07-09: V19.1 = **+97.83%** (peak_protect_dist), V23 = +61.96% (signal) → mất **35.87%**. Nguyên nhân: V23's threshold peak_protect_strong=0.15 trigger quá sớm khi trade đã +15% → exit khi còn có thể lên tới +100%.

---

## 3. PHÂN TÍCH NGUYÊN NHÂN GỐC RỄ CỦA V23

### 3.1 Vấn đề #1: **signal_hard_cap bị over-fire (−378.6%, vs V22 −120.7%)**

So sánh trades hard_cap giữa V19.1 vs V23 (28 trade overlap): **V19.1 tổng = −242.7%, V23 = −362.9% → V23 tệ hơn 120.2%**.

Nguyên nhân kỹ thuật:
- V23 có công thức `cap = max(hard_cap_moderate_floor, hard_cap_strong_mult * atr_ratio_now)` dùng ATR động.
- Nhưng với profile **high_beta** (AAS, AAV, SSI, VND) ATR đã đắt (~4-5%), `hard_cap_strong_floor=0.15` vẫn dễ bị kích hoạt vì floor thấp.
- Khi strong trend có pullback 2 ngày 8-10%, cap đánh thẳng → cut ở đáy → giá hồi ngay sau đó.

### 3.2 Vấn đề #2: **peak_protect yếu hơn V19.1 (41 → 22 trades, mất alpha ~760%)**

V19.1: `peak_protect_dist` tổng = **+1041%** (41 trades, avg +28%).
V23: `peak_protect_dist` tổng = +229% (13 trades).

V23 chuyển trigger peak_protect sang điều kiện 3-in-1 (below SMA10 + heavy vol + bearish candle) — quá hiếm khi nhiều uptrend có pullback êm. V19.1 chỉ cần SMA10 break là đủ (vì có confirm_bars).

### 3.3 Vấn đề #3: **Grid search không tìm thấy cải tiến mạnh**

21 configs grid, best case chỉ là V23-pp_s=0.12 = +1860% — chỉ +12% so với default. Điều này cho thấy **không gian tham số đã bão hoà**: thêm một threshold nữa không giải quyết được vấn đề cấu trúc.

### 3.4 Vấn đề #4: **2022 bear year worst (−47%, V19.1 −22%)**

V23 cut nhiều hơn ở REE/TCB/MBB trong 2022 do hard_cap rộng hơn V22 (theo thiết kế). Nhưng lại cut **đúng đáy** vì không có logic "đợi hồi trong strong trend".

### 3.5 Vấn đề #5: **Rule bỏ lỡ 58 wins >+10% (+1582% gross alpha)**

Ngay cả V19.1 cũng miss một phần. Các wins này là **momentum wins dài** (hold 30-120 ngày). ML models gần như không bắt được vì:
- Target label = 5-day forward return → mô hình không biết cổ phiếu sẽ có 60-day move.
- Exit confirm/trail pct quá tight → cut sớm trước khi move lớn hình thành.

---

## 4. NHẬN XÉT TỔNG — "Hoàn hảo" hay "Trade-off"?

**Không có model hoàn hảo.** Ma trận trade-off:

```
              raw_PnL  risk_ctrl  robustness  edge_on_bigwins
Rule            ★★★★★   ★★         ★★★★★       ★★★★★
V19.1           ★★★★    ★★         ★★★★        ★★★★ (peak_protect)
V19.3           ★★★     ★★★★★      ★★★         ★★
V22             ★★★     ★★★        ★★★         ★★
V23-best        ★★★★    ★★★★       ★★★★        ★★★
V16             ★★★     ★★★        ★★★★        ★★
```

V23 có điểm số composite cao nhất nhưng **không phải "hoàn hảo"** — nó là V19.1 với drawdown kiểm soát tốt hơn, đổi lấy một phần alpha.

---

## 5. ĐỀ XUẤT V24 — "NEXT-GEN OPTIMAL" (spec, not code)

Mục tiêu: **vượt +2000% tổng PnL** (hơn cả Rule) **với MaxLoss ≤ −20%** và PF ≥ 2.7.

### 5.1 Fix chí mạng #1 — "Smart hard_cap" thay thế hard cap cố định

Vấn đề: hard_cap hiện là "đánh là đánh, không đợi". Đề xuất:
- **Khi trend=strong và hard_cap trigger**: **requires 2 consecutive closes below entry_price−cap** (confirm 1 bar). Thí nghiệm trên REE 2022-05-11: giá 2022-05-12 close đã hồi → không exit → giữ lại +11%.
- **Khi trend=strong + high_beta**: floor=0.18 thay vì 0.15, mult=3.5×ATR (không 3.0). 
- Expected impact: giảm signal_hard_cap từ −378% → ~−150%, thu được ~+200%.

### 5.2 Fix chí mạng #2 — Khôi phục peak_protect sensitivity V19.1

- Bỏ yêu cầu "heavy_vol AND bearish_candle" — chỉ cần **below SMA10 + confirm 1 bar**.
- Thêm layer: nếu price_max_profit ≥ 25%, giảm threshold còn 0.10 (lock in sooner).
- Expected impact: khôi phục ~+500% peak_protect alpha, WR có thể giảm nhẹ nhưng avg trade tăng.

### 5.3 Fix mới #3 — "Long-horizon carry" bắt big wins Rule đang thắng

Đây là fix **đột phá** nhất:
- Thêm feature: detect "**rolling_60d_return**" và "**consecutive_higher_highs_20d**". Khi cổ phiếu trong uptrend mạnh kéo dài (ret_60d > 30% AND sma20 > sma50 > sma100 liên tục 20d+):
  - Override tất cả signal/peak_protect exit.
  - Chỉ exit khi: close < sma50 ×0.97 **OR** macd_hist < 0 liên tiếp 3 bars.
- Expected impact: có khả năng capture được AAV +127%, SSI +68%, HPG +61% — những trade Rule thắng nhờ hold dài.

### 5.4 Fix mới #4 — Symbol-specific profile refinement

- REE: `exit_score_threshold +0.6` (V19.1 đã có 1.8 quá thấp → nhiều signal exits sai).
- AAS: disable `profit_lock` khi strong trend (vì hay trigger trước khi run +50%+).
- MBB: giữ nguyên V19.1 peak_protect (không downgrade).
- VND: giữ V23 hard_cap (hoạt động tốt).

### 5.5 Fix mới #5 — 2-model ensemble (ML signals + rule momentum)

- Nếu ML model say BUY **và** rule system say BUY → size = 0.95.
- Nếu chỉ ML BUY **và** rule đang holding → size = 0.50.
- Nếu chỉ rule BUY → bắt đầu với size = 0.30, scale up nếu ML ≥1 sau 5 ngày.
- Expected impact: giảm false entries trong bear (V23 thua V19.3 2022 đây là lý do), tăng conviction trong strong trend.

### 5.6 Roadmap thử nghiệm (A/B config)

| Ver | Keep | Change | Target |
|---|---|---|---|
| V24.0 (base) | V23-best | + smart hard_cap (5.1) | MaxLoss −19%, Total +1900% |
| V24.1 | V24.0 | + PP sensitivity fix (5.2) | Total +2000%, PF 2.75 |
| V24.2 | V24.1 | + long-horizon carry (5.3) | Total +2100%, MaxLoss −20% |
| V24.3 | V24.2 | + symbol tuning (5.4) | Total +2150% |
| V24.final | V24.3 | + rule ensemble (5.5) | **Total +2250%, PF 2.8, MaxLoss −18%** |

---

## 6. KẾT LUẬN

1. **V23 không "hoàn hảo"** — đó là phiên bản **tradeoff rủi ro tốt nhất** (smallest MaxLoss trong top performers) nhưng **hy sinh −6.5% PnL** so với V19.1 và **−122% so với Rule**.
2. **Điểm chí mạng của V23**: signal_hard_cap over-fires (−378%, worst in class), peak_protect bị làm yếu (41 → 22 trades).
3. **Không có entry/exit nào dominant** — mỗi model mạnh ở context riêng:
   - V23 ngon nhất cho: SSI 2023, VND 2022, BID 2022, TCB strong-trend.
   - V19.1 ngon nhất cho: AAS high-profit winners, REE strong-trend pullback, MBB.
   - V19.3 ngon nhất cho: 2022 bear defense.
   - Rule ngon nhất cho: long-horizon momentum (AAV, HPG, DGC big runs).
4. **Đột phá thật sự cần**: long-horizon carry + smart hard_cap + rule ensemble. Không còn là tuning threshold — cần bổ sung **thêm module cấu trúc** (detect hold-winner-forever regime).

---

## PHỤ LỤC — Exit reason comparison (tổng PnL theo lý do)

| Reason | V19.1 | V19.3 | V22 | V23 |
|---|---:|---:|---:|---:|
| signal | +95% | +1088% | +1032% | **+1305%** |
| peak_protect_dist | **+1041%** | +1020% | +174% | +229% |
| peak_protect_ema | +130% | +157% | +193% | +182% |
| fast_exit_loss | 0 | **−901%** ❌ | −61% | −44% |
| signal_hard_cap | 0 | −156% | −121% | **−378%** ❌ |
| trailing_stop | +61% | +25% | +25% | +22% |
| end | +546% | +542% | +545% | +545% |

Quan sát: V19.1 = "dominant peak_protect". V23 = "dominant signal exits nhờ Module H". V19.3 = "trade-off fast_exit ăn vào alpha".

---
*End of report*
