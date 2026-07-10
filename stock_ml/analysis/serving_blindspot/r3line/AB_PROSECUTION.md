# HỒ SƠ CÔNG TỐ — ab_modernclean_mh16 (t2936 = gb_x08 − trailing_struct ×2 + max_hold 16)
Ngày: 2026-07-10/11. Công tố viên adversarial; tiền lệ 6 REJECT, 1 CONDITIONAL (r2c), 1 SURVIVED (t2907), 1 lật khung đo.
Thước: `nh_nav2` K25 R0.6 settle T+2, shuffle-mean±sd 20 perm, HAI chế độ (adv 0.08% / no-advance).
Scripts/bằng chứng mới: `pr5_10_plateau.py, pr5_20_inflation.py, pr5_30_friction.py, pr5_40_mech.py,
pr5_50_cfg.py, pr5_60_dyn.py, pr5_70_hybrid.py` (+ pr5_*_out.txt, pr5_noT_mh{12,20,25}_s42_trades.csv,
pr5_dynclean_mh25_s42_trades.csv). Clone DB mới: pr5_noT_mh12/20/25 = t2940/2941/2942, pr5_dynclean_mh25 = t2943
(KHÔNG đụng canonical, KHÔNG commit). Trades bị cáo: ab_noT_s{42,7,99,555} (ab_30, đã có).

## TỘI 1 — Selection inflation (~80 config NAV-scored tích lũy): GỠ PHẦN LỚN — plateau bỏ-T THẬT, held-out sống; f25-adv = noise → tripwire kế thừa
- **Held-out f21** (pr5_20a): ab **+49.1% adv (11.9sd) / +43.2% noadv (10.0sd)** vs gb — MẠNH NHẤT rổ, hơn cả t2907 (+37.9/+34.3).
- **Held-out f24**: **+22.2% (7.8sd) / +23.8% (7.2sd)** — hơn t2907 (+18.5/+20.7).
- **Held-out f25→nay**: s42 adv +3.3% (1.8sd) / noadv **+5.4% (2.6sd)**. NHƯNG 4 seed (pr5_20b): adv +0.4..+3.3% (0.2-1.8sd — noise trên 3/4 seed); noadv +2.8..+5.4% (1.5-2.6sd — thở được 3/4 seed). **DD-edge ĐẢO ở f25 y án t2907: ab −13.4/−13.5 vs gb −11.4** → tripwire recency BẮT BUỘC kế thừa nguyên văn.
- **LOYO 7/7 PASS ≥2sd CẢ HAI chế độ** (pr5_20c): min adv bỏ-2021 +39.2% (8.7sd); min noadv bỏ-2021 +42.1% (6.7sd). Yearly delta dương 13/14 ô (adv 2026 −0.4đ = noise).
- **Drop-top-10/20 cả hai bên: edge TĂNG** (pr5_20d): drop20 full adv **+84.4% (13.4sd)**, noadv +64.8% (12.0sd), f22 +36.0% (10.0sd). Bị cáo trải 2440 lệnh, không tựa top-trade.
- **Plateau bỏ-T (pr5_10, 3 run mới t2940-2942)**: mh12 ×22.88±0.55/×19.61±0.66 (DDw −12.0, f22 4.46/4.29), mh20 ×20.54/×17.77 (trũng ~−10% như mọi stack tại mh20 — hình dạng lặp lại của R3_ROUND1, không phải bệnh riêng), mh25 ×21.89/×19.40 (f22 4.45/4.47). **mh16 KHÔNG phải đỉnh cô lập** — mh12 ngang adv, cả dải 12-25 ≥ +49% vs gb. Số yếu nhất trong dải (mh20) vẫn thắng gb +49% adv.
- Trừ điểm: số công bố PHẢI theo **seed-mean 4 seeds ×22.24 adv / ×20.67 noadv** (ab_30), không quote s42 ×22.87.

## TỘI 2 — Cơ chế bỏ-T tạo rủi ro mới: GỠ (kèm 1 ghi chú trung thực)
- **Sold-then-rallied cohort overext_trail 203 lệnh** (pr5_40a): sau khi bán-vào-sức-mạnh (+17.4% mean, 8.8 phiên), max-rally 10 bar sau exit mean +8.0%, **33% lệnh chạy tiếp ≥+10%**. NHƯNG counterfactual GIỮ-TỚI-CAP bar 17 (đúng điều gb+mh16 làm) chỉ được thêm **mean +2.8% / med +2.1%, 37% ÂM**, tổng bỏ lỡ 5.6u trên 203 lệnh — đổi lấy **1659 symbol-ngày slot giải phóng**. NAV compound +13.4% vs gbmh16 nói rằng vòng quay thắng: rủi ro "bán non" là CÓ THẬT nhưng đã được trả giá thừa. Trailing_stop 10 lệnh: counterfactual +0.2%, 70% âm — vô hại.
- **Tail runner** (pr5_40b): ab n≥+30% 66 / n≥+50% 10 vs gbmh16 (giữ T) 85/15 — mất ~19 lệnh runner nhẹ, top10-share 5% vs 6%, max GIỮ NGUYÊN +85.5%. So t2907 (61/10): ab NHỈNH hơn. "Lệnh +50% bị chốt +17%" đúng ở mức 5 lệnh/6.5 năm — không phải cấu trúc.
- **Tail âm KHÔNG xấu đi** (pr5_30d/40d): p1 −13.8% / p5 −8.1% / sh<−10% 2.7% ≈ t2907 (−13.5/−8.1/2.6%); cohort overext min −3.9%, 0% <−10% (kênh chốt-lời thuần); dao vẫn nằm ở kênh signal (−1702u, 5.1%<−10%) — y hệt mọi án trước, phí bảo hiểm đã khai. Max_hold cohort mean +8.70%, 0.5%<−10%.
- **Bootstrap DD 200×80%** (pr5_40c): ab NAV p5 15.57 / med 17.47 / p95 19.83, DD med −11.6% / p95 −13.1% / worst −14.5% — phân phối NAV tách hẳn gb (p5 ab 15.57 > p95 gb 13.25), DD nhỉnh hơn t2907 (med −11.6 vs −11.2, worst −14.5 vs −14.7 ~ hòa).

## TỘI 3 — Friction: GỠ TOÀN BỘ — không kém án t2907 ở bất kỳ ô nào (pr5_30)
- **Fee sweep adv**: R0.7 +59.0% (8.9sd) / R0.8 +52.3% / R0.9 +45.8% / **R1.0 +39.7% (6.9sd)**; f22 R1.0 **+16.2% (4.6sd)** [t2907: +32.6%/+11.8%] — dư địa dày hơn.
- **Entry lag +1 cả hai bên**: full adv **+38.5% (6.6sd)**, full noadv **+18.9% (3.4sd)** [t2907: +31.3/+9.9], f22 adv +24.8% (8.4sd), f22 noadv +18.8% (4.5sd). Không lật được.
- **Exit lag +1 đúng 1046 lệnh max_hold** (gb nguyên): full +51.4% (7.9sd) / noadv +47.1% (6.0sd); f22 +18.5/+17.0%. Exit lag +1 TẤT CẢ cả hai bên: +60.7% (7.8sd).
- Per-trade: +4.58%/lệnh (med +2.15%, wr 62.7%, PF 4.02, hold 10.7) — mỏng hơn gb (+9.32%) nhưng đệm phí sống tới R1.0, hình y án t2907 đã SURVIVED.

## TỘI 4 — Era/consistency: GỠ HOÀN TOÀN (1 đoạn, pr5_50)
DB diff t2936 vs t2783: khác ĐÚNG 3 chỗ — exit_priority prepend "max_hold", max_hold_bars 16, thiếu 2 key trailing_struct; **71 engine key còn lại + 2 slot + meta/split/universe/threshold GIỐNG HỆT**. Cùng pipeline, cùng walk-forward train2/test1/gap85, cùng universe 61 mã, cùng heads — mọi kết luận era-audit của án t2907 (pr4_00/01) áp nguyên, không có bề mặt leak mới. Sanity ab_allrm tái lập t2907 bit-chính-xác (ABLATION §2) đã khóa tính hợp lệ của phân rã.

## TỘI 5 — Serving: THÀNH LẬP 1 ĐIỀU KIỆN NHẸ (không chặn) — đường promote NGẮN NHẤT rổ
Điều tra code (engine.py, serving/inference.py, stock-serving/serving/trades.py, wheel dist):
- **2 key bị bỏ là lever GATED default-off, không structural**: engine.py:346 `trailing_struct_donch_win: int|None = None` (None = tắt, về %-trail legacy), :351 `apply_overext: bool = False`. Bỏ key = engine tự về default = đúng hành vi ab_noT. **Wheel stock_ml_core 0.3.3 tại host ĐÃ CÓ đủ field** (kể cả max_hold_bars) → KHÔNG cần bump wheel.
- **Host stock-serving CHẠY ENGINE THẬT**: serving/trades.py:47-86 dựng EngineConfig từ bundle `config['engine']` và gọi `run_backtest` của wheel → max_hold 16, exit_priority, và cả kênh overext_trail (203 lệnh sống lại) đều do ENGINE trong wheel enforce, không cần host tự viết. Kênh overext KHÔNG phải blindspot mới (chuỗi trailing/overext là code engine có sẵn; live gb_x08 cũng mang các key này).
- **ĐIỀU KIỆN duy nhất**: trades.py:60-64 hiện GIẢ ĐỊNH max_hold_bars khổng lồ ("can never actually fire"). Với 16 nó SẼ fire — hành vi engine đúng nhưng CHƯA tiền lệ live → trước promote: export bundle t2936 (models = đúng dòng live hiện đại downpress, chỉ sửa engine dict 3 chỗ) + leakage-auditor 4-check + **shadow-run so trades bit-level, chú ý lệnh đóng tại bar 16 và cột open-positions**.
- So t2907: t2907 cần exit head **vol_market** (bundle riêng, đời fs cũ, phải export + kiểm parity); t2936 giữ nguyên exit-fs downpress = cùng feature pipeline champion live → **t2936 là đường promote ngắn nhất**.

## TỘI 6 — So dyn_mh25 (t2929, recent-tilt): THÀNH LẬP MỘT PHẦN — t2936 KHÔNG thống trị recent-noadv; hybrid mở tuyến mới
- Head-to-head (pr5_60a): ab thắng full +8.1%/+8.3% (1.2-1.6sd), NHƯNG **thua f22 noadv −7.8% (−3.2sd)**, f23 noadv −4.1% (−1.5sd), **f25 noadv −4.0% (−2.2sd)**; f25 DDw dyn nông hơn (−12.8/−12.9 vs −13.4/−13.5). dyn25 cũng đè gb f25 noadv +9.8% (5.2sd) — thứ t2936 không làm được.
- Overlap (pr5_60b): exact 71.4% của ab / 78.5% của dyn; fuzzy±5d 77.3% — cùng gia tộc lệnh nhưng đủ khác để blend có nghĩa; blend giấy K50: full ×23.88, f22 4.61 → má số dương → được phép 1 run.
- **Hybrid pr5_dynclean_mh25 (t2943, 1 seed 42)**: t2936-stack + entry_ensemble3 → csrank dyn_pure z0.88 (thay head mfe-breadth, đúng cấu trúc t2531 — pr5_50 xác nhận đây là THAY head, không phải thêm) + mh25: full ×21.32±0.56/×19.48±0.64; **f22 4.56/4.60, f23 2.99/3.00, f25 1.65/1.68, DDw f25 −12.9 — MẠNH NHẤT rổ ở MỌI lát ≥2022, cả hai chế độ**, đổi ~−7% full (nhường 2020-21). Đây là 1 seed, CHƯA qua công tố — KHÔNG chặn t2936; mở tuyến "kế nhiệm recent" riêng nếu user muốn.

## PHÁN QUYẾT: **SURVIVED** (kèm 4 điều kiện — cùng hạng án t2907, hồ sơ MẠNH hơn ở mọi ô đo được)
1. **Công bố theo seed-mean**: full adv **×22.24** (+61% vs gb) / noadv **×20.67** (+59%), f22 ~4.37/4.19, f23 ~2.94/2.88 — không quote s42 ×22.87. Mọi seed ≥ t2907 điểm công bố (verify ab_30).
2. **Serving trước promote**: export bundle t2936 (sửa engine dict 3 chỗ, giữ models dòng downpress) + leakage-auditor 4-check + shadow-run bit-level, tập trung hành vi max_hold-fire (chưa tiền lệ live). KHÔNG cần bump wheel (0.3.3 đủ field).
3. **Tripwire recency kế thừa nguyên án t2907**: f25 adv chỉ +0.4..+3.3% (≤1.8sd, noise 3/4 seed) và DD-edge f25 đảo (−13.4 vs gb −11.4); rolling-18-tháng vs gb < −1sd 2 quý liên tiếp → demote về gb+mh16/gb. Dự phòng tâm **mh12** vẫn hợp lệ (bỏ-T mh12 ×22.88 adv, DDw −12.0 nông nhất).
4. **t2936 THAY t2907 làm ứng viên chính** (hơn ở cả 6 ô NAV, f21/f24 held-out, LOYO min, friction, serving-path); t2907 rớt về dự phòng khảo cứu. dyn_mh25/t2943 = tuyến satellite recent-tilt, cần công tố riêng nếu theo đuổi.

## BẢNG HỘI ĐỒNG CUỐI (nh_nav2 K25 R0.6, mean±sd 20 perm; adv | noadv; delta = vs gb)
| tiêu chí | gb_x08 (t2783, live) | t2907 (t2429+mh16, SURVIVED) | **t2936 (ab_noT) — BỊ CÁO** | dyn_mh25 (t2929) |
|---|---|---|---|---|
| NAV full 2020- | ×13.78±0.42 \| ×13.03±0.42 | ×21.77±0.53 \| ×19.62±0.60 | **×22.87±0.64 \| ×20.99±0.89** | ×21.15±0.79 \| ×19.39±0.87 |
| seed-mean 4 seeds | — (s42) | ×21.06 \| ×19.38 | **×22.24 \| ×20.67** (mọi seed ≥ t2907) | — (s42 đơn) |
| f22 | 3.43 \| 3.34 | 4.27 \| 4.10 | **4.43 \| 4.22** | 4.45 \| **4.57** |
| f23 | 2.46 \| 2.44 | 2.83 \| 2.77 | **2.92 \| 2.85** | **2.95 \| 2.97** |
| f21 held-out (vs gb) | ref | +37.9 \| +34.3 | **+49.1 (11.9sd) \| +43.2 (10.0sd)** | +49.1 \| **+52.7** |
| f24 held-out | ref | +18.5 \| +20.7 | **+22.2 \| +23.8** | +21.6 \| **+30.9** |
| f25→nay | ref (DDw −11.4) | +0.7% (0.4sd) \| +2.9% (1.5sd) | +3.3% (1.8sd) \| **+5.4% (2.6sd)**; 4-seed adv ≤1.8sd → **tripwire** | +2.7% (1.4sd) \| **+9.8% (5.2sd)**, DDw −12.9 |
| LOYO | ref | 7/7 ≥2sd (min +35.1%) | **7/7 ≥2sd cả 2 chế độ (min adv +39.2% 8.7sd / noadv +42.1% 6.7sd)** | chưa chạy |
| MaxDD DDw full / bootstrap med..worst | −14.2 / −14.6..−16.5 | −13.8 / −11.2..−14.7 | −14.0 / **−11.6..−14.5**; NAV boot p5 15.57 > gb p95 13.25 | −13.7 / chưa chạy |
| phí R1.0 (vs gb) | ref | full +32.6% \| f22 +11.8% (3.5sd) | **full +39.7% (6.9sd) \| f22 +16.2% (4.6sd)** | chưa chạy |
| entry lag +1 cả 2 bên | ref | full adv +31.3% (5.5sd) | **full adv +38.5% (6.6sd) \| noadv +18.9% (3.4sd); f22 +24.8/+18.8** | chưa chạy |
| exit lag +1 | ref | mh-only +43.7% | **mh-only +51.4/+47.1%; all-2-bên +60.7% (7.8sd)** | chưa chạy |
| drop-top-20 | tựa top-trade | +81.4% (13.1sd) | **+84.4% (13.4sd) \| +64.8% (12.0sd)** | chưa chạy |
| per-trade | **+9.32%, PF 6.02**, 1378 lệnh | +4.45%, PF 3.94, 2451 | +4.58%, PF 4.02, 2440, hold 10.7 | +4.65*, PF 3.98, 2221 |
| plateau max_hold | — | 12-25 rộng (R3) | **bỏ-T: mh12 ×22.88 / mh20 ×20.54 / mh25 ×21.89 — không đỉnh cô lập** | điểm mh25 |
| serving-path | đang live, 0 việc | cần bundle exit_vol_market riêng + shadow | **ngắn nhất: bundle = dòng live downpress, chỉ sửa engine dict 3 chỗ; wheel 0.3.3 đủ; host engine tự enforce mh16/overext; cần shadow-run (mh-fire chưa tiền lệ)** | cần head csrank (xsec) — nặng hơn |
| vai trò đề xuất | fallback / demote-target | dự phòng khảo cứu (bị thay) | **ỨNG VIÊN CHÍNH promote (điều kiện §Phán quyết)** | satellite recent-tilt; hybrid t2943 (f22 4.56/4.60, f25 1.65/1.68, 1 seed) = tuyến sau |

Files: pr5_10_out.txt (plateau), pr5_20_out.txt (held-out/LOYO/droptop/f25×4seed), pr5_30_out.txt (friction),
pr5_40_out.txt (sold-then-rallied/tail/bootstrap), pr5_50 (cfg diff, in console), pr5_60_out.txt (dyn),
pr5_70_out.txt (hybrid t2943). Đối chiếu: ABLATION_8PCT.md, R3_PROSECUTION.md, XGEM_CROSSAPPLY.md.
