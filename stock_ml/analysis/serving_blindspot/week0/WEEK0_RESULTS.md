# WEEK-0 RESULTS — Serving Blindspot (BLINDSPOT_REPORT.md Phần 4-5)

Ngày: 2026-07-09. Champion gốc: template **2646** `n2_2643_wavestruct_la05_lamp02` (KHÔNG bị sửa; hàng leaderboard seed-555 đã xác minh nguyên vẹn sau toàn bộ batch: comp 730.4, pnl 127.940, pf 5.942, trades 1386). Mọi thí nghiệm là clone config-only `w0_*`; 100% run cache-hit (30-76s/run, không có retrain nào).

## 1. Bảng screen seed-42 (toàn bộ run, Δ vs CONTROL seed-42: comp 729.6 / pnl 127.253 / pf 5.963 / mdd 0.17455 / 1384 trades)

| Run | Tmpl | Comp | ΔComp | PnL | ΔPnL | PF | MDD | ΔMDD | Trades |
|---|---|---|---|---|---|---|---|---|---|
| **LINE B — pyramid sizing** (`pyramid_add_units` u, `pyramid_add_min_ret` r, bars=3) | | | | | | | | | |
| w0_pyr_u05_r02 | 2666 | 858.9 | +129.3 | 159.715 | +32.46 | 6.405 | 0.20302 | +0.0285 | 1384 |
| w0_pyr_u05_r03 | 2667 | 825.9 | +96.3 | 151.403 | +24.15 | 6.273 | 0.19574 | +0.0212 | 1384 |
| w0_pyr_u05_r04 | 2668 | 811.8 | +82.2 | 147.101 | +19.85 | 6.286 | 0.18768 | +0.0131 | 1384 |
| **w0_pyr_u10_r02** | 2669 | **978.1** | **+248.5** | 192.178 | +64.92 | 6.642 | 0.24100 | +0.0664 | 1384 |
| w0_pyr_u10_r03 | 2670 | 912.8 | +183.2 | 175.552 | +48.30 | 6.426 | 0.22848 | +0.0539 | 1384 |
| w0_pyr_u10_r04 | 2671 | 886.1 | +156.5 | 166.949 | +39.70 | 6.464 | 0.21066 | +0.0361 | 1384 |
| **LINE C — exit levers** | | | | | | | | | |
| w0_relk_30 | 2673 | 729.5 | −0.1 | 127.240 | −0.01 | 5.963 | 0.17455 | 0 | 1383 |
| w0_relk_35 | 2674 | 729.4 | −0.2 | 127.215 | −0.04 | 5.962 | 0.17455 | 0 | 1383 |
| w0_mfeact_05 | 2675 | 727.2 | −2.4 | 127.007 | −0.25 | 5.920 | 0.17655 | +0.0020 | 1384 |
| w0_mfeact_10 | 2676 | 727.2 | −2.4 | 127.007 | −0.25 | 5.920 | 0.17655 | +0.0020 | 1384 |
| w0_snr_10 | 2677 | 731.8 | +2.2 | 127.884 | +0.63 | 6.004 | 0.17455 | 0 | 1377 |
| w0_snr_12 | 2678 | 730.1 | +0.5 | 127.502 | +0.25 | 5.983 | 0.17455 | 0 | 1379 |
| **LINE OPS** | | | | | | | | | |
| w0_cxlma20 | 2672 | 578.4 | −151.2 | 105.712 | −21.54 | 5.043 | 0.21006 | +0.0355 | 1064 |
| w0_win20 | 2679 | 688.2 | −41.4 | 121.742 | −5.51 | 5.257 | 0.19405 | +0.0195 | 1410 |
| w0_reprem04 | 2680 | 711.9 | −17.7 | 123.862 | −3.39 | 5.891 | 0.17795 | +0.0034 | 1376 |

Kết luận screen: (a) toàn bộ lưới pyramid dương mạnh, PF TĂNG ở mọi ô (add có lọc ≥min_ret tại bar 3 chọn đúng winner, không phải leverage mù); (b) `mfe_act_k` là lever chết (clamp floor 0.08/cap 0.27 làm k=0.5 và k=1.0 cho kết quả identical từng bit); (c) `release_drop_k` 2.5 của champion là tối ưu cục bộ; (d) `exit_snr_extend_threshold=1.0` là lever exit duy nhất dương (+2.2); (e) cả 3 lever OPS đều trả giá alpha, chỉ w0_reprem04 (−17.7) đủ rẻ để cân nhắc trade-off serving.

## 2. Multi-seed (seeds 42/7/99/555) — composite từng seed và mean

| Candidate | s42 | s7 | s99 | s555 | **Mean** | ΔMean vs control | Δ/seed (min..max) |
|---|---|---|---|---|---|---|---|
| CONTROL 2646 | 729.6 | 731.5 | 722.5 | 730.4 | **728.5** | — | — |
| **w0_pyr_u10_r02** | 978.1 | 985.2 | 967.8 | 986.2 | **979.3** | **+250.8 (+34.4%)** | +245.3..+255.8 |
| w0_pyr_u10_r03 | 912.8 | 919.2 | 902.5 | 922.6 | 914.3 | +185.8 | +180.0..+192.2 |
| w0_pyr_u10_r04 | 886.1 | 891.9 | 880.0 | 894.2 | 888.1 | +159.6 | +156.5..+163.8 |
| w0_pyr_u05_r02 | 858.9 | 863.6 | 850.3 | 863.2 | 859.0 | +130.5 | +127.8..+132.8 |
| w0_pyr_u05_r03 | 825.9 | 830.2 | 817.3 | 831.1 | 826.1 | +97.6 | +94.8..+100.7 |

Champion seed-stable (spread 9.0); mọi candidate pyramid thắng control ở CẢ 4 seed với spread hẹp (~2% mean).

### Khung portfolio (mean 4 seed) — vì composite KHÔNG so sánh hoàn hảo dưới sizing

**Cảnh báo comparability: CÓ bị compromised.** Pyramid add làm `pnl_pct` = base+add units (524/1384 trade của u10_r02 mang weight 2.0), nên composite của candidate pyramid không cùng đơn vị với champion. Phán quyết phải dựa khung (total_pnl, mdd_per_symbol):

| Candidate | PnL mean | MDD mean | PnL/MDD | vs control |
|---|---|---|---|---|
| CONTROL 2646 | 127.44 | 0.1789 | 712.2 | — |
| **w0_pyr_u10_r02** | **192.67** | 0.2432 | **792.2** | PnL +51.2%, MDD +35.9% (tương đối), ratio +11.2% |
| w0_pyr_u10_r03 | 176.09 | 0.2308 | 763.1 | |
| w0_pyr_u10_r04 | 167.56 | 0.2130 | 786.8 | |
| w0_pyr_u05_r02 | 160.05 | 0.2061 | 776.5 | |
| w0_pyr_u05_r03 | 151.76 | 0.1988 | 763.5 | |

Ngay cả trong khung portfolio, **w0_pyr_u10_r02** vẫn tốt nhất: PnL tuyệt đối cao nhất VÀ PnL/MDD cao nhất (792 vs 712) — trả thêm MDD nhưng được đền bù trên cả hai trục. Nếu ràng buộc MDD chặt (~0.21): u10_r04 là điểm lui hợp lý (PnL +31.5%, MDD +19%, ratio 787).

## 3. BEST CANDIDATE: w0_pyr_u10_r02 (template 2669)

Config thêm trên champion: `pyramid_add_units=1.0`, `pyramid_add_min_ret=0.02` (`pyramid_add_bars=3` default). Mean composite 979.3 vs control 728.5.

### Guardrail acceptance (Phần 3) — trade frame seed 42, candidate vs champion cùng seed
(export: `best_trades/trades_w0_pyr_u10_r02.csv` & `trades_n2_2643_wavestruct_la05_lamp02.csv`; script `guardrails_best.py`, log `guardrails_best.log`)

| Guardrail | Ngưỡng | Candidate | Champion s42 | Kết quả |
|---|---|---|---|---|
| G1 Bigwin-rate (pnl>15%) | ≥19% | **24.49%** (339/1384) | 21.03% | **PASS** |
| G2 Bigwin pnl-share | ~100% | **107.5%** | 99.7% | **PASS** (non-bigwin cohort net âm — đúng mô hình "nhà máy một sản phẩm") |
| G3 Hold>20-bar WR | ~84% | **81.21%** (612 trades, mean +34.0%) | 84.97% (mean +22.6%) | **PASS có caveat**: WR −3.8pp (23 trade lật win→loss do add-unit kéo pnl_pct gộp xuống), nhưng mean cohort +34.0% vs +22.6% và pnl-share 108.4% giữ nguyên |
| G4 Trades trong ±10% của 1386 | ±10% | **1384** (−0.1%) | 1384 | **PASS** (pyramid không tạo/chặn entry) |
| G5 PnL 2022 không tệ hơn control >5u | Δ>−5u | **14.045u** | 12.104u | **PASS** (Δ **+1.94u** — năm gấu còn TỐT HƠN) |

PnL theo năm (candidate − control, đơn vị u): 2020 +7.80, 2021 +28.14, 2022 +1.94, 2023 +8.00, 2024 +3.80, 2025 +13.13, 2026YTD +2.12 — **dương TỪNG năm**, không có năm nào bị hy sinh.

## 4. Kết luận & bước tiếp theo đề xuất

**Kết luận:** w0_pyr_u10_r02 là candidate thắng rõ ràng của Week-0: +34.4% mean composite trên 4 seed, thắng ở mọi seed, PF tăng (6.52-6.72 vs 5.82-6.00), PnL/MDD tăng, pass cả 5 guardrail, PnL dương thêm từng năm kể cả 2022. Chi phí duy nhất: MDD/symbol 0.179→0.243 (+36% tương đối) — cần phê duyệt risk-budget trước khi promote.

**Bước tiếp theo (theo thứ tự ưu tiên):**
1. **Stack thử `w0_snr_10` lên 2669** (`exit_snr_extend_threshold=1.0, window=20, min_gain=0.27`): lever exit dương duy nhất (+2.2 standalone), trục trực giao với sizing; screen seed 42 rồi multi-seed nếu dương. Đặc biệt đáng thử vì SNR-extend giữ runner lâu hơn đúng lúc position đã pyramid.
2. **Audit seed ngoài batch (vd 123)** trên 2669 + control để loại nghi ngờ seed-mining (4 seed hiện tại đều được dùng trong quá trình chọn).
3. **Portfolio sim với capital constraint thật**: 524/1384 trade mang weight 2.0 — cần xác nhận equity curve (đã export `equity_*.csv` nếu có) và margin/capital feasibility; composite hiện tại không phản ánh chi phí vốn của add-unit.
4. **Điều tra caveat G3**: 23 trade hold>20 lật win→loss do add — xem add tại r=0.02 có quá sớm với cohort chậm; thử `pyramid_add_min_ret=0.03` chỉ khi cần siết (u10_r03 vẫn +185.8 mean, MDD 0.231).
5. Nếu risk-budget từ chối MDD 0.243: promote **w0_pyr_u10_r04** (mean 888.1, MDD 0.213, ratio 787) làm phương án bảo thủ.
6. Đóng các lever chết khỏi backlog: `mfe_act_k` (clamp-saturated), `release_drop_k` ≠2.5, toàn bộ OPS levers trừ khả năng trade-off serving của `reentry_max_premium_pct=0.04` (−17.7 comp).

**Vệ sinh dữ liệu:** hàng champion seed-555 đã restore và verify EXACT sau các run export (RESTORE_CHECK PASS trong `export_best_trades.log`). Templates w0_* (2666-2680) giữ nguyên trên leaderboard, mỗi template một run_id, không đụng run_id champion.

## 5. Stack + seed-123 audit

Batch follow-up 2026-07-09 (scripts `run_stack_snr.py`, `run_seed123_audit.py`; log cùng thư mục). Clone mới duy nhất: **w0_pyr_u10_r02_snr10** (template **2681**) = clone TỪ 2669 (pyramid keys giữ nguyên, verify OK trong DB: `pyramid_add_units=1.0, pyramid_add_min_ret=0.02` + `exit_snr_extend_threshold=1.0, exit_snr_extend_window=20, exit_snr_min_gain=0.27`). 2646/2669 không bị sửa. 100% run cache-hit (37-41s).

| Run | Tmpl | Seed | Comp | PnL | PF | MDD | Trades |
|---|---|---|---|---|---|---|---|
| w0_pyr_u10_r02_snr10 | 2681 | 42 | 983.0 | 193.573 | 6.696 | 0.24100 | 1377 |
| w0_pyr_u10_r02_snr10 | 2681 | 7 | 990.7 | 195.627 | 6.772 | 0.24397 | 1374 |
| w0_pyr_u10_r02_snr10 | 2681 | 99 | 974.1 | 192.059 | 6.575 | 0.24543 | 1386 |
| w0_pyr_u10_r02_snr10 | 2681 | 555 | 990.6 | 195.346 | 6.738 | 0.24237 | 1380 |
| CONTROL 2646 | 2646 | 123 | 728.3 | 127.930 | 5.886 | 0.17959 | 1382 |
| w0_pyr_u10_r02 | 2669 | 123 | 977.6 | 193.845 | 6.545 | 0.25175 | 1382 |
| w0_pyr_u10_r02_snr10 | 2681 | 123 | 983.3 | 195.427 | 6.603 | 0.25175 | 1375 |

**Stack:** mean 4 seed **984.6** vs 2669 979.3 → **Δ +5.3**, dương ở CẢ 4 seed (+4.4..+6.3); standalone snr chỉ +2.2 trên champion → hơi super-additive khi position đã pyramid, đúng giả thuyết. PnL mean 194.15 vs 192.67, MDD mean 0.2432 = KHÔNG đổi vs 2669 → PnL/MDD 798.3 vs 792.2. Verdict: **stack thắng miễn phí (MDD không tăng), 2681 thay 2669 làm best candidate.**

**Seed-123 (ngoài batch, anti seed-mining):** Δ(2669−control) tại seed 123 = **+249.3** vs Δ mean in-batch +250.8 (lệch −0.6%, nằm sâu trong ±20%) → **edge pyramid KHÔNG phải seed-mining**. Δ(2681−2669) tại 123 = +5.7, khớp Δ stack in-batch +5.3. Control seed-123 comp 728.3 nằm trong range 4-seed (722.5-731.5).

**Anomaly (nhỏ):** MDD seed-123 của 2669/2681 = 0.2517 — cao hơn range 4-seed (0.2410-0.2454), là worst-case MDD quan sát được; 2669 và 2681 có MDD identical từng bit tại seed 123 (snr-extend chỉ kéo dài winner, không đổi đường drawdown). Trades của stack giảm nhẹ (~3-9 trade/seed), nhất quán với hành vi w0_snr_10 standalone.

## 6. AUDIT CÔNG BẰNG (gap forensics, 2026-07-09) — KẾT LUẬN ĐẢO CHIỀU

Ba agent độc lập (mechanism / decomposition / fairness-critic) mổ xẻ +248.5 composite của 2669. Composite tái lập bit-chính-xác (729.6/978.1) từ trade CSV bằng `scoring.py` thật trước khi hiệu chỉnh. Script: `tmp_gap_pyramid_mech.py`, `tmp_gap_decomp.py`, `tmp_gap_fairness.py`.

### Cơ chế thật (engine.py:2304-2327)
Pyramid là **post-pass thuần kế toán**: 1384 lệnh identical champion (entry/exit/giá/lý do khớp 1:1, 860 lệnh không-add có pnl identical từng bit). Nếu close tại bar fill+3 ≥ +2% so fill → add 1.0 unit tại close đó (+slippage), exit chung với base; `pnl_pct := base_net + add_net` (TỔNG return 2 unit, không phải return/vốn), weight=2.0. Trigger bar-3 là bộ phân loại winner lộ-sau-entry cực mạnh: cohort add có base-WR 83.2% / mean +0.200 vs 41.3% / +0.026 phần còn lại.

### Phân rã +248.5 điểm
- Term PnL tuyến tính (w 0.45, chưa chạm cap): **+261.9 điểm** (105% gap) — mỗi +1u PnL = +4.03 điểm.
- Term MDD (0.1745→0.2410): −25.0. Các term khác: +12.7. Composite **mù vốn**: không có term nào chuẩn hóa theo units.

### Phán quyết công tố: ~98% gap là artifact quy mô vốn
| Khung đo | Candidate 2669 vs Champion |
|---|---|
| Unit-days triển khai | 62,710 vs 40,713 (**×1.54**); peak units 96 vs 58 (×1.655) |
| PnL / 1000 unit-days | 3.065 vs 3.126 (**−2.0%** — hiệu quả vốn/ngày KÉM hơn) |
| Composite chuẩn hóa vốn trung bình (÷1.54) | 735.1 vs 729.6 (**+5.5**, không phải +248.5) |
| Composite khung vốn-peak (÷1.655) | 702.9 (**THUA −26.7**) |
| **PnL/MDD (bất biến scale — edge thật duy nhất)** | **797.4 vs 729.0 (+9.4%)** |
| Add fill same-bar close (lạc quan hơn convention close_next của chính engine) | −2.98u nếu sửa đúng convention (12/524 add bất khả thi) ≈ −12 điểm |
| Grid winner u10_r02 | = biên lưới cả 2 trục; Δcomp tuyến tính theo units (×1.92 khi ×2) — composite thưởng leverage |
| Seed-robustness (kể cả 123) | vô giá trị chứng minh: artifact kế toán deterministic tự tái lập mọi seed |
| Add-leg theo năm | 2020 +25.4u (39%) … **2024 +0.17u, 2026YTD −0.03u** — vốn biên ~0 hai giai đoạn gần nhất |

### Provenance
Kết quả pyramid đã tồn tại từ 2026-06-21 (`results/_research_2429/pyramid_sweep_result.json`, untracked, seed-42-only, số identical). Chưa từng bị bác — bị bỏ quên đúng vì lo ngại "pnl_pct semantics đổi khi có size", mà audit này xác nhận là **chí mạng**.

### Kết luận sửa lại
- **BÁC tuyên bố "+35% composite / +51% PnL"**. Tuyên bố đúng: 2669/2681 là **conditional-leverage overlay** — PnL/MDD +9.4% (≈+8.7% sau khi sửa add fill sang close_next), đổi lấy +54% vốn trung bình/+66% vốn peak; ở vốn bằng nhau: PnL −2.0%, MDD −10.4%.
- 2681 stack (+5.3) là exit-timing thật, MDD không đổi, nhưng cùng thang bị thổi phồng (~+3.4 sau chuẩn hóa).
- Điều kiện tiên quyết cho MỌI tuyến sizing (Line B): (1) sửa add fill → close_next; (2) thêm khung chấm điểm chuẩn-hóa-vốn / portfolio sim có ràng buộc vốn thật vào protocol promote; (3) so sánh trên PnL/MDD và %NAV, không dùng composite thô.
