# TUYẾN R2 — VÒNG 3 (đào quanh oxtrail04, mục tiêu f22 ≥2sd) — KẾT LUẬN: CÓ ỨNG VIÊN
Ngày: 2026-07-10. Prefix: `r2c_` (template id mới, DB Postgres 5433; 17 run seed-42). Thước đo DUY NHẤT: `nh_nav2.py` sim v2 (settle T+2, advance 0.08%/vòng, shuffle-mean±sd 20 perm, R0.6, K25) — delta vs gb K25 adv (×13.78±0.42 full, ×3.43±0.08 f22, ×2.464±0.063 f23, DD −15.28/worst −15.61).
Scripts: r2c_00_cfg.py, r2c_01_anatomy.py (r2c_01_out.txt), r2c_02_cascade.py (r2c_02_out.txt), r2c_10_sweep.py (logs/r2c_g1..g6.log), r2c_20_double.py (r2c_20_out.txt), r2c_21_yearly.py (r2c_21_out.txt), r2c_30_k.py (r2c_30_out.txt).

## 1. Giải phẫu r2b_oxtrail04

**Config xác nhận từ DB (t2884)**: fc_rule2 + pb 4.0/50 + snr g12 (0.8/20/gain 0.12/gb 0.08) + **duy nhất `overext_trail_pct: 0.04`**. Nền: overext_pct 0.12/ma20, trailing 8% ATR-adaptive (×2.0, floor 4%/cap 16%), activate 15%. Cơ chế engine (engine.py:2085,2312): khi giá vượt +12% trên MA20, KHÔNG bán thẳng (`overext`) — arm trailing band chặt (`overext_trail`), runner ride tiếp, fader bị cắt gần mức overext.

**Vì sao dominate c2** (paired diff symbol+entry_date, 1846 cặp chung):
- 359 trade đổi exit overext→overext_trail: ride thêm +1.65 ngày, **d_pnl +4.0đ tổng**, toàn bộ nằm ở cohort winner <10% (+4.03đ); cohort loser không đổi (−0.14đ) — đúng thiết kế "ăn continuation sau overext, không mở thêm rủi ro loser".
- Cascade entry (exit đổi ngày → reentry_cooldown/occupancy → bộ entry khác): c2-only 236 (+12.0đ) vs ox-only 189 (+9.4đ) = net **−2.6đ** — cascade hơi thiệt; lợi thế NAV thật = ride winner + fader thoát sớm quay vòng vốn.
- Exit mix: overext 468/+65.1 → overext_trail 422/+65.8; trailing/signal/dtstop nguyên vẹn.

**Lỗ hổng f22 (1.5sd)** — yearly NAV (K25 adv, mean 20 perm): ox−gb theo năm: 2020 −2.8 / 2021 **+18.2** / 2022 +3.6 / 2023 +5.7 / **2024 −1.6** / 2025 +2.0 / 2026 −0.2. d_pnl của 359 trade đổi exit: 2020-22 +1.0..+1.2/năm, 2023 +0.05, **2024 −0.11, 2026 −0.16**. Cơ chế: **ride overext thắng năm trend, hòa-thua năm chop (2023-24, 26)** — lát 2024 âm là thứ ghìm f22 ở 1.5sd.

## 2. Lưới r2c_ (17 điểm mới, chấm nh_nav2 K25 adv R0.6, delta vs gb K25 adv)

| điểm | comp s42 | full ±sd (DD mean=worst mọi perm) | vs gb (sd) | f22 ±sd | vs gb (sd) |
|---|---|---|---|---|---|
| r2b_oxtrail04 (tâm, vòng 2) | 577.3 | ×15.75±0.54 (−13.7) | +14.4%±5.3 (2.7) | ×3.60±0.07 | +5.0%±3.3 (1.5) |
| r2c_oxt02 | 571.4 | ×15.50±0.65 (−13.8) | +12.5%±5.9 (2.1) | ×3.65±0.10 | +6.4%±3.9 (1.7) |
| r2c_oxt025 | 570.3 | ×15.52±0.65 (−13.8) | +12.6%±5.9 (2.2) | ×3.67±0.08 | +6.9%±3.5 (2.0) |
| r2c_oxt03 | 570.1 | ×15.77±0.56 (−13.8) | +14.5%±5.4 (2.7) | ×3.68±0.07 | **+7.3%±3.3 (2.2)** |
| r2c_oxt05 | 569.7 | ×14.51±0.61 (−13.8) | +5.3%±5.5 (1.0) | ×3.48±0.08 | +1.4%±3.2 (0.4) |
| r2c_oxt06 | 563.7 | ×14.33±0.44 (−13.8) | +4.0%±4.5 (0.9) | ×3.58±0.10 | +4.3%±3.7 (1.2) |
| r2c_oxt04_ox10 | 568.1 | ×14.22±0.46 (−13.6) | +3.2%±4.6 (0.7) | ×3.49±0.11 | +1.7%±4.0 (0.4) |
| r2c_oxt04_ox14 | 572.1 | ×14.11±0.74 (−13.7) | +2.4%±6.2 (0.4) | ×3.33±0.07 | −3.1%±3.1 (−1.0) |
| r2c_oxt03_ox10 | 570.2 | ×14.50±0.40 (−13.5) | +5.2%±4.3 (1.2) | ×3.60±0.07 | +4.7%±3.2 (1.5) |
| r2c_oxt05_ox14 | 563.3 | ×13.41±0.42 (−13.8) | −2.6%±4.3 (−0.6) | ×3.34±0.06 | −2.6%±3.0 (−0.9) |
| r2c_oxt04_w40 | 576.7 | ×15.56±0.50 (−13.7) | +12.9%±5.0 (2.6) | ×3.63±0.07 | +5.6%±3.3 (1.7) |
| r2c_oxt03_w40 | 571.5 | ×15.54±0.67 (−13.8) | +12.8%±6.0 (2.1) | ×3.69±0.08 | +7.5%±3.4 (2.2) |
| r2c_oxt04_p39 | 579.9 | ×15.08±0.42 (−14.0) | +9.5%±4.5 (2.1) | ×3.56±0.11 | +3.8%±4.1 (0.9) |
| r2c_oxt04_p41 | 583.3 | ×16.48±0.52 (−13.5) | +19.6%±5.3 (3.7) | ×3.67±0.10 | +6.7%±3.9 (1.7) |
| **r2c_oxt04_p42** | **588.4** | **×16.84±0.75 (−13.3)** | **+22.2%±6.6 (3.4)** | **×3.76±0.09** | **+9.4%±3.6 (2.6)** |
| r2c_oxt04_p43 | 591.1 | ×16.49±0.61 (−13.2) | +19.7%±5.7 (3.4) | ×3.63±0.10 | +5.7%±3.8 (1.5) |
| r2c_oxt04_p45 | 579.8 | ×16.12±0.56 (−12.4) | +17.0%±5.4 (3.1) | ×3.55±0.08 | +3.4%±3.5 (1.0) |
| r2c_oxt03_p41 | 575.5 | ×15.56±0.54 (−13.6) | +12.9%±5.2 (2.5) | ×3.60±0.11 | +5.0%±4.0 (1.3) |

Đọc bản đồ trục:
- **Trail width (nền depth 4.0)**: plateau f22 tại **0.025-0.03** (2.0-2.2sd), rơi ở 0.04 (1.5), sập ở 0.05+ — trail chặt cắt fader năm chop, ăn thêm 2025/26 (yearly oxt03−gb: 2025 +4.1, 2026 +0.3; nhưng 2024 vẫn −1.4).
- **Arm threshold (overext_pct) chết cả hai phía** (ox10 0.7sd, ox14 −1.0sd f22): 0.12 là ridge — arm sớm nghẹt runner, arm muộn mất protection.
- **Depth dưới trail = trục vàng vòng này**: 3.9→+9.5 / 4.0→+14.4 / 4.1→+19.6 / **4.2→+22.2** / 4.3→+19.7 / 4.5→+17.0 full — **plateau rộng 4.1-4.5, mọi điểm ≥3.1sd full** (khác hẳn ridge 4.0 của vòng 1 KHÔNG-trail: trail04 đã đổi tối ưu depth về phía sâu — có trail bảo vệ, pullback sâu hơn có giá vào tốt hơn mà không tăng DD; DD còn GIẢM dần: −13.5→−12.4).
- f22 tại K25 nhô đỉnh 4.2 (2.6sd; hàng xóm 1.7/1.5) — nhìn riêng K25 thì nghi gai; **kiểm K22 (r2c_30): cả dải 4.1/4.2/4.3 f22 = ×3.86/3.96/3.86 vs gb ×3.36 = +14.9..+17.9%, đều ≥2.9sd** → mặt f22 là plateau thật khi có thêm slot, không phải may-knob.
- Combo âm: oxt03×p41 (1.3sd f22) — trail chặt và depth sâu tranh nhau cùng một alpha ride, KHÔNG cộng dồn; w40 trung tính (không cộng với oxt03).

## 3. Kiểm tra kép r2c_oxt04_p42 (điểm đề cử) + oxt03 (á quân f22)

| check | r2c_oxt04_p42 | r2c_oxt03 |
|---|---|---|
| NAV-từ-2023 (chống-ghost sâu) | ×2.61±0.05 = **+5.7%±3.5 (1.6sd)** | ×2.54±0.05 (+2.9%, 0.9sd) |
| MaxDD mean±sd / worst-perm | −13.28%±0.00 / −13.28% (gb −15.28/−15.61) | −13.78 / −13.78 |
| top-20-trade share | 5.4% (chuẩn c2 7.5%) | 5.4% |
| per-entry-year pnl | dương đủ 7 năm (2024 +4.3, 2026 +0.6) | dương đủ 7 năm |
| yearly NAV vs gb | 2020 −0.7 / 2021 +25.2 / 2022 +3.6 / 2023 +7.6 / **2024 −0.9** / 2025 +5.0 / 2026 −1.6 | 2024 −1.4, còn lại dương từ 2021 |
| K-stability (22/25/28, luôn #1 cả 2 khung) | full ×17.73/16.84/15.58; f22 ×3.96/3.76/3.53 | ×16.31/15.77/14.19; ×3.80/3.68/3.41 |
| điều kiện tồn tại | no-adv: ×13.25 vs gb ×13.03 (+1.7% noise), f22 −2.4% → **chết nếu không ứng trước** (y hệt toàn tuyến) | tương tự |

Lát 2024 (−0.9) và 2026 (−1.6) vẫn là hai lát âm mỏng vs gb — depth 4.2 vá gần hết 2024 (từ −1.6 còn −0.9, per-year +4.3 tuyệt đối dương), không lát nào âm quá 1.6đ. Full-frame tựa 2021 (+25.2) nhưng f22 (2.6sd) và f23 (1.6sd) độc lập cùng dương → không phải ghost 2020-21.

## 4. VERDICT — có ứng viên f22 ≥2sd: **r2c_oxt04_p42** (template id 2900, run template/r2c_oxt04_p42, trades r2c_oxt04_p42_s42_trades.csv)

= fc_rule2 + **pb 4.2%/50** + snr g12 (0.8/20/0.12/0.08) + **overext_trail 0.04**. Bảng hội đồng vs gb_x08 (nh_nav2 v2, R0.6, adv):

| tiêu chí | r2c_oxt04_p42 | gb_x08 | verdict |
|---|---|---|---|
| composite s42 | 588.4 (subframe ≥2022: 292.3) | 735.0 (368.5) | gb thắng — composite đo per-trade quality, KHÔNG phải thước tuyến này |
| NAV v2 full K25 | **×16.84±0.75** | ×13.78±0.42 | **+22.2%±6.6 = 3.4sd** |
| f22 K25 | **×3.76±0.09** | ×3.43±0.08 | **+9.4%±3.6 = 2.6sd — ĐẠT chuẩn ≥2sd** |
| f23 K25 | ×2.61±0.05 | ×2.464±0.063 | +5.7%±3.5 = 1.6sd (dương, khung quá ngắn để đòi 2sd) |
| MaxDD (mean/worst-perm) | **−13.28/−13.28%** | −15.28/−15.61% | nông hơn 2.0-2.3đ Ở CÙNG K |
| K22 (chấp DD ngang gb) | ×17.73±0.87, DD −15.0; f22 ×3.96 vs gb-K22 ×3.36 = +17.9%±5.2 (3.4sd) | ×13.79±0.53, −15.9 | dominate cả ở matched-DD |
| điều kiện tồn tại | retail R0.6 + **ứng trước tiền bán 0.08%/vòng BẮT BUỘC** (no-adv = noise vs gb) | không cần ứng | ghi vào cost model serving |
| độ phức tạp serving | **0-ML**, deterministic, 7 số knob (pb 4.2/50, snr 4 số, trail 4%) | ML exit-head + gb infra | R2 đơn giản hơn hẳn |

**Tuyên bố (chuẩn ≥2sd + regime ≥2022)**: trong điều kiện sống của tuyến (retail R0.6 + ứng trước), r2c_oxt04_p42 vượt gb_x08 **+22.2% full (3.4sd) và +9.4% f22 (2.6sd) tại MaxDD nông hơn 2đ**; điểm số không tựa ghost 2020-21 (f22 2.6sd, f23 +5.7% dương, đủ 7 năm entry dương, top-20 share 5.4%). Đây là lần đầu tuyến R2 có điểm qua chuẩn 2sd ở CẢ HAI khung — nâng cấp từ verdict "alternative đồng hạng risk-adjusted" (NAV_V2_FRONTIER) lên **ứng viên vượt trần thật sự của hệ 0-ML**.

Dè chừng trung thực trước khi promote:
1. f22 2.6sd là ĐỈNH của plateau depth tại K25 (hàng xóm 4.1/4.3 chỉ 1.7/1.5sd); phòng thủ đúng = tuyên bố theo dải: "depth 4.1-4.3 × trail04 cho full +19.6..+22.2% (≥3.4sd), f22 +5.7..+9.4%", và K22 xác nhận cả dải ≥2.9sd f22. Nếu hội đồng đòi số duy nhất: lấy 4.2 nhưng expected-value thực tế nên chiết khấu về giữa dải (~+7% f22).
2. Hai lát âm mỏng còn lại vs gb: 2024 (−0.9) và 2026 YTD (−1.6) — bản chất cơ chế ride kém năm chop; không có knob còn sống nào trong lưới vá được mà không trả giá năm trend (oxt03 vá 2026 nhưng mất p42-full; combo âm).
3. Phí ứng trước 0.08%/vòng là ĐIỀU KIỆN TỒN TẠI (đã tái xác nhận trên chính p42: no-adv +1.7% = noise); serving phải model phí này và giới hạn vốn (R0.6 retail).
4. Trục còn lại chưa đào (nếu cần vòng 4): tương tác depth-sâu × window (4.2/45, 4.2/55), snr threshold trên nền p42, và overext_dist_neg/vol_spike keys (chưa đụng). Trục chết cập nhật thêm: overext_pct ±0.02 (cả 2 phía), trail ≥0.05, combo trail-chặt×depth-sâu.
