# HỒ SƠ PHÁP Y: PDR 2023-12-19 → 2024-04-24 (bán trailing trễ 7 phiên so với exit score)

Ngày lập: 2026-07-10. Scripts: `em08_pdr_trace.py` (trace + quét cohort), `em09_structtrail.py`
(counterfactual %-trail), output: `pdr_trace.csv`, `pdr_forensic_scan.csv`, `pdr_structtrail_cf.csv`.

## 1. Định vị lệnh — khớp phiên bản nào?

| Nguồn | Entry | Exit | PnL | Rule |
|---|---|---|---|---|
| gb_x08 s42 (run_trades, template 2783) | 2023-12-19 @21.387 (fill limit, signal 13/11) | 2024-04-24 @22.596 | **+5.25%** | trailing_stop |
| champion 2646 s42 (`signalq/st_champ2646_s42_trades.csv`) | 2023-12-19 @21.387 | 2024-04-24 @22.596 | +5.25% | trailing_stop |
| serving (signals.csv, bundle 2643 wavestruct) | signal +1 giữ suốt 12/2023→12/4/2024 | **signal lật −1 đúng 2024-04-15** (exit_score 1.27→2.41, đỉnh 2.94 ngày 19/4) | — | — |

Kết luận định vị: lệnh user nhìn **khớp cả gb_x08 lẫn champion 2646 (hai bản cho ra lệnh GIỐNG HỆT
nhau)** — tức thủ phạm KHÔNG phải khối snr_extend riêng của gb_x08 (2646 không có mà vẫn giữ tới
24/4). Tín hiệu bán 15/4 trên chart user = exit score/signal −1 của serving, và mô hình backtest
cũng force-sell đúng 15/4 (xem §2). MFE lệnh: +31.0% (close cao nhất 2/4), exit +5.25% → nhả lại
25.8 điểm từ đỉnh.

## 2. Truy vết từng bar (parity engine đã xác nhận: struct-break 23/4 → fill close 24/4 = 22.63×(1−slip) = 22.596 ✓)

Entry fill 21.387. Trailing đã ARM từ 05/4 (peak_gain 31% ≥ 27% và close < MA10) nhưng ở chế độ
**struct-Donchian** (`trailing_struct_donch_win=80`): chỉ bắn khi close < đáy-80-bar-trước (~21.3–21.8),
KHÔNG dùng băng %-giveback. snr_defer: SNR âm suốt tháng 4 → không bao giờ kích hoạt (minh oan).

| Ngày | Close | Gain | Peak | Giveback | Sell force? | mkt_drop z | Cơ chế nuốt | Trailing struct |
|---|---|---|---|---|---|---|---|---|
| 10–12/4 | 26.43–26.73 | +24–25% | +31% | 6–7.4% | KHÔNG (chưa có sell) | −1.60/−1.09/+0.40 | — | don_low 21.29, chưa vỡ |
| **15/4** | 24.89 | +16.4% | +31% | 14.6% | **CÓ** (force lowbreadth: zigzag6 đảo chiều + breadth<25%; serving −1 cùng ngày) | **−2.30 ≤ −1.75** | **exit_market_drop suppress** | chưa vỡ (24.89 ≫ 21.29) |
| 16/4 | 24.39 | +14.0% | +31% | 17.0% | CÓ (+nonbull) | −3.03 | mkt_drop suppress | chưa vỡ |
| 17/4 | 23.42 | +9.5% | +31% | 21.5% | CÓ (+downleg12) | −3.25 | mkt_drop suppress | chưa vỡ |
| 19/4 | 21.92 | +2.5% | +31% | 28.5% | CÓ | −3.75 | mkt_drop suppress | 21.92 > don_low 21.79, chưa vỡ |
| 22/4 | 22.29 | +4.2% | +31% | 26.8% | CÓ | −3.05 | mkt_drop suppress | chưa vỡ |
| **23/4** | 21.16 | −1.1% | +31% | 32.1% | CÓ | −1.43 (đã nhả) | không còn gì chặn | **21.16 < don_low 21.83 → trailing bắn** |
| 24/4 | 22.63 | +5.8% | | | | −0.06 | | fill close 24/4 → +5.25% |

**Chuỗi sự kiện:** tín hiệu bán CÓ THẬT từ 15/4 (cả force-gate backtest lẫn exit score serving).
Cơ chế nuốt nó đích danh là **`exit_market_drop` suppress (mode zscore, window 5, threshold −1.75)**:
VN-Index sập 15/4/2024 (~−4.7%), z universe rơi −2.30 → −3.75 và giữ ≤ −1.75 suốt 5 phiên 15→22/4,
mỗi ngày đều `continue` qua nhánh signal-exit ("đừng bán đáy washout, giao cho trailing").
Nhưng "trailing backstop" lúc đó là **struct-Donchian 80 bar** — trần giveback thực tế ~26%, không
phải 8% — nên lưới đỡ thứ hai cũng đứng im tới khi giá thủng đáy 80 phiên (23/4, đúng đáy sóng).
Ngày 23/4 z đã hồi lên −1.43, suppress nhả — signal cũng muốn bắn lại — nhưng trailing đứng trước
trong `exit_priority` nên nhãn exit là trailing_stop.

**Chi phí trì hoãn riêng lệnh này:**
- Bán theo signal 15/4 (fill close 16/4 = 24.39): **+13.47%** → thực tế +5.25% ⇒ suppress nuốt **−8.2 điểm**.
- Nếu trailing là %-trail thuần (không struct): bắn 11/4, fill 12/4 → **+24.4%** ⇒ struct-donch nuốt **−19.1 điểm** (đường độc lập, không bị mkt_drop chặn).
- Hai cơ chế phải CÙNG hỏng thì mới ra +5.25%: mkt_drop khóa cửa signal, struct-donch tháo lưới trailing.

## 3. Tổng quát hóa — toàn bộ 1378 lệnh gb_x08 s42

### 3a. Signal-exit bị nuốt trước decision bar (proxy sell = force gates; head-sell/s3z không tái tạo được offline — số n là cận dưới)

517/1378 lệnh có ít nhất một bar force-sell bị cơ chế nuốt. Delta = pnl_thực − pnl_bán-tại-bar-bị-nuốt
(âm = trì hoãn làm tệ hơn). **Entries ≥2022:**

| Cơ chế nuốt (theo thứ tự engine) | n | u thiệt (gross) | u cứu (gross) | **NET** | tệ hơn ≥5% | tốt hơn ≥5% |
|---|---|---|---|---|---|---|
| mkt_drop suppress | 266 | −3.35 | +9.68 | **+6.33** | 16 | 51 |
| hold_extatr (MA50+1.3ATR·rs) | 49 | −1.33 | +4.27 | **+2.94** | 7 | 10 |
| protect_band (8–99%+trend) | 4 | 0.00 | +0.44 | +0.44 | 0 | 3 |
| snr_defer (nghi phạm số 1) | 5 | −0.25 | +0.18 | **−0.08** | 2 | 2 |
| recon_mismatch | 5 | −0.13 | +0.33 | +0.19 | 1 | 1 |

Net theo năm entry (mkt_drop): 2022 +0.10 · 2023 +2.24 · 2024 +0.05 · 2025 +3.71 · 2026 +0.23 —
**không năm nào âm**. Cắt theo giveback-tại-bar-bị-nuốt: mọi bucket (kể cả giveback>20%) đều net dương.
PDR (−8.2đ) là đuôi 6% tệ nhất của một cơ chế trung bình +2.4đ/lệnh. 15/16 lệnh "tệ ≥5%" sau đó vẫn
exit bằng signal muộn hơn; chỉ PDR rơi vào combo struct-trail.

### 3b. Struct-Donchian 80 vs %-trail thuần (em09, counterfactual approximation)

293 lệnh %-trail thuần bắn sớm hơn exit thực. **≥2022: n=161, NET +9.25** (cứu +14.58 / thiệt −5.32;
44 lệnh tệ ≥5%, 61 lệnh tốt ≥5%). Theo năm: 2022 −0.34 · 2023 +3.47 · 2024 −0.49 · 2025 +6.48 ·
2026 +0.13. Cũng net dương; 2022/2024 âm nhẹ không đủ tín hiệu.

### 3c. Thử ranh giới sửa: struct-donch + trần giveback từ đỉnh (knob CHƯA có trong engine — chỉ ước u)

| CAP giveback | n ảnh hưởng ≥2022 | net Δu ≥2022 |
|---|---|---|
| 12% | 139 | **−5.63** (phá runner) |
| 15% | 77 | −2.86 |
| 18% | 40 | −0.09 (~hòa) |
| 22% | 13 | +0.03 (~hòa) |

Không mức nào dương có nghĩa. Không có ranh giới sạch ⇒ **không tạo probe fx_**.

## 4. VERDICT

1. **Thủ phạm chính: `exit_market_drop` suppress (z ≤ −1.75, w5)** nuốt tín hiệu bán 15→22/4 trong
   cú sập thị trường 4/2024; **tòng phạm: `trailing_struct_donch_win=80`** biến lưới trailing từ 8%
   thành ~26% giveback. `exit_snr_defer` (nghi phạm số 1 ban đầu) **vô tội** với lệnh này — SNR âm
   suốt tháng 4, cả trade chưa từng bị defer.
2. Trên toàn bộ dữ liệu ≥2022, CẢ HAI cơ chế đều net dương rõ (+6.3u và +9.3u): chúng tồn tại chính
   vì đa số washout ngắn hạn hồi lại (2023/2025 hưởng lớn). PDR 4/2024 là washout-không-hồi — đuôi
   phân phối mà thiết kế đã chấp nhận trả giá.
3. Mọi ranh giới thử (giveback bucket của suppress, trần giveback cho struct-trail 12–22%) đều net
   âm hoặc hòa ≥2022 ⇒ theo protocol không có gì đáng probe. **Cơ chế hoạt động đúng thiết kế; chi
   phí PDR là giá đã tính trong +3.6 của gb_x08.** Nếu user muốn giảm đuôi này thì hướng duy nhất
   còn lại là knob mới "suppress-release khi giveback-từ-đỉnh vượt X trong lúc mkt_drop" — nhưng chính
   cohort đó (giveback ≥10–20% lúc bị nuốt) hiện đang NET DƯƠNG (+2.6u ở 2025), nên chưa có bằng
   chứng để sửa.

Hạn chế ghi nhận: proxy sell = force-gates (không có head-sell/s3z per-bar offline) → n bị nuốt là
cận dưới; counterfactual per-trade không mô phỏng lại slot/re-entry sau khi exit sớm.
