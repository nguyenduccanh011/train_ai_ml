# HỒ SƠ CÔNG TỐ — họ t2429 + max_hold (tâm plateau mh16, đại diện r3_mh16/t2907)
Ngày: 2026-07-10. Công tố viên adversarial, tiền lệ 6 REJECT + 1 lật khung đo + r2c CONDITIONAL.
Thước: `nh_nav2.py` (K25, R0.6, settle T+2, adv 0.08%/vòng + no-advance, shuffle 20 perm). Baseline gb_x08
tái lập đúng: ×13.78±0.42 adv / ×13.03±0.42 noadv, f22 3.43/3.34, DDw −14.2/−14.1 (full −15.3 công bố = mean-DD frame cũ).
Scripts/bằng chứng: `pr4_00_cfg.py, pr4_01_engdiff.py, pr4_10_seeds.py, pr4_20_heldout.py,
pr4_21_loyo_droptop.py, pr4_30_friction.py, pr4_40_tail.py` (+ pr4_*_out.txt, pr4_mh16_s{7,99,555}_trades.csv).
Clone mới: `pr4_mh16` = t2918 (KHÔNG đụng canonical, không commit).

## TỘI 1 — Era-artifact / leak (nặng nhất): GỠ HOÀN TOÀN
Forensics config từ DB (pr4_00/pr4_01):
- **"Đời cũ" là nguỵ danh**: t2429 tạo 2026-06-19, t2783 (gb) tạo 2026-07-09 — cách nhau 20 ngày, CÙNG pipeline.
  CÙNG strategy `regression_dual_ml_recombine_decoupled`, CÙNG entry head (ml=82 entry_plain lightgbm,
  fs `entry_lvup126_recov` — không phải "lvup126_lean" như cáo trạng, label triple_barrier pt.15/sl.08 h30),
  CÙNG exit head (ml=78 exit_lgbmreg_mono_div2, velocity_exit_regression h20), CÙNG universe `vn_stock_default`
  (61 symbol cả hai), CÙNG split_config `walk_forward_year train2/test1/gap85/2020-2025` (splitter.py:1-5:
  train trim cuối bằng gap_days — purge chuẩn, retrain từng fold, dùng chung cả hai stack → mọi khiếm khuyết nếu có là CHUNG,
  không tạo lợi thế khác biệt).
- **Khác biệt thật chỉ 2 chỗ**: (i) exit feature set `exit_vol_market` (14 feat) vs `exit_vol_downpress` — mà downpress
  = vol_market + dist-day + down-volume (catalog.py:792,1469): tập của 2429 là TẬP CON của gb → không có kênh
  feature riêng để leak; (ii) gb THÊM 16 engine key (wavestruct hold-extension + snr defer/extend + trailing_struct).
- **Phân rã nguồn thắng**: cap tự nó tái lập gần hết hiệu ứng trên stack hiện đại: gb ×13.78 → gb+mh16 ×20.16 (+46%).
  Phần dư "cũ thắng mới dưới cap" chỉ **+8.0%±4.1 (2.0sd) adv / +6.8%±5.7 (1.2sd) noadv** — cỡ hyperparameter
  (16 key hold-extension hơi âm dưới cap + khác exit-fs), KHÔNG phải bí ẩn 1.5-2.2 NAV-x cần era-leak để giải thích.
- **Seed-luck loại bỏ (pr4_10, 3 run mới t2918)**: s7 ×20.75/×19.24, s99 ×20.40/×18.76, s555 ×21.33/×19.88
  (adv/noadv full; s42 ×21.77/×19.62); comp 631-636, ~2450 lệnh, pf 3.90-3.95 — mọi seed ≥ gb +44% noadv.

## TỘI 2 — Selection inflation (~70 config NAV-scored): GỠ PHẦN LỚN — plateau THẬT ngoài tập chọn; f25 = tripwire
- **f21 held-out**: mh12/16/25 adv **+42.1/+37.9/+48.2%** (7.3-9.0sd); noadv +37.1/+34.3/+45.7%. Cả plateau sống, không riêng điểm chọn.
- **f24 held-out**: adv **+24.0/+18.5/+18.7%** (7.0-9.1sd); noadv +25.0/+20.7/+27.7% (6.7-10.3sd).
- **f25→nay (18 tháng)**: tâm mh16 adv **+0.7% (0.4sd) = noise** (giống hình r2c); nhưng mh12 +4.1% (2.5sd),
  và noadv: mh12 +6.2% (3.4sd) / mh16 +2.9% (1.5sd) / mh25 +8.0% (4.1sd) — plateau còn thở ở no-advance.
  **DD-edge ĐẢO ở f25**: gb −11.4 vs mh16 −13.5 (mh12 giữ −12.0) → tripwire bắt buộc.
- **LOYO 7/7 PASS ≥2sd CẢ HAI chế độ** (pr4_21): min adv bỏ-2021 +35.5% (8.6sd); min noadv bỏ-2022 +35.1% (6.3sd).
  Yearly delta dương 6/7 (adv 2026 −0.9đ; noadv 2024 −0.5đ — đều noise).
- **Drop-top-10/20 CẢ HAI bên: edge TĂNG** — drop20 full adv **+81.4% (13.1sd)**, noadv +63.2% (9.2sd), f22 +31.4% (8.1sd).
  gb mới là hệ tựa top-trade; bị cáo trải 2451 lệnh.
- EV plateau công bố (adv ~×21.4 / noadv ~×18.7) vs mean 4 seed mh16 (×21.06/×19.38): khớp trong sd. Công bố theo
  plateau chứ không lấy max (mh12 adv ×22.39) = mitigation hợp lệ.

## TỘI 3 — Friction/thực thi: GỠ TOÀN BỘ (khác biệt quyết định vs án r2c)
- **Fee sweep (adv)**: R0.7 +51.2% (8.7sd) / R0.8 +44.8% / R0.9 +38.5% / **R1.0 vẫn +32.6% (6.4sd)**; f22 R1.0 +11.8% (3.5sd).
  KHÔNG có trần chết trong khảo sát (r2c chết 2sd ở R0.9).
- **Entry lag +1 phiên áp CẢ HAI bên (đòn đã lật r2c)**: mh16 VẪN THẮNG — full adv +31.3% (5.5sd), full noadv +9.9% (2.1sd),
  f22 adv +19.8% (7.2sd), f22 noadv +13.5% (3.5sd). Không lật được.
- **Exit lag +1 trên đúng 1027 lệnh max_hold** (bán trễ 1 phiên, gb giữ nguyên): full +43.7% (8.1sd), f22 +14.7% (4.2sd) — sống.
  Exit lag +1 TẤT CẢ lệnh cả hai bên: +48.6% (7.9sd).
- Per-trade edge: **+4.45%/lệnh** (med +2.08%, wr 62.4%, PF 3.94, hold med 11, 2451 lệnh) vs gb +9.32%/PF 6.02/1378 lệnh —
  mỏng hơn nhưng đệm phí dày (sống tới R1.0), khác hẳn r2c (+5.19% nhưng chết lag).

## TỘI 4 — DD/tail thật: GỠ (ghi chú crash 2025-04)
- **Bootstrap subsample 80% ×200 rep**: mh16 DD med −11.2% / p95 −13.1% / worst −14.7% vs gb −14.6/−15.9/−16.5 —
  lợi thế DD bền ngoài khung shuffle-perm, phân phối NAV cũng tách hẳn (p5 mh16 14.70 > p95 gb 13.25).
- **Crash 2025-04**: mh16 −13.4% vs gb −12.8% (cùng đáy 09/04, cùng hồi 05/05) — trong crash cấp tính bị cáo SÂU HƠN 0.6đ;
  lợi thế DD đến từ phần còn lại của mẫu. Ghi chú, không tội.
- **Knife check**: cohort max_hold KHÔNG phải ổ dao — mean **+8.62%**, min −16.2%, chỉ 0.5% lệnh <−10%. Đuôi âm nằm ở kênh
  signal-exit (min −20.9%, 4.9%<−10%, tổng −15.9u) = phí bảo hiểm đã khai, cùng hình gb. Tail per-position TỐT hơn gb
  (sh<−10%: 2.6% vs 3.8%; p1 −13.5 vs −14.6). Concentration symbol top5 17% ≈ gb 16%.

## TỘI 5 — Serving/parity: THÀNH LẬP 1 ĐIỀU KIỆN (không chặn, KHÔNG cần bump wheel)
Điều tra code (agent, engine.py + serving/bundle.py + catalog.py + experiment.py):
- Wheel `stock_ml_core` đóng gói chính `stock_ml.src*` → EngineConfig repo = nguồn chân lý: có ĐỦ mọi key t2429+mh16;
  `max_hold_bars` là field required (engine.py:158), `"max_hold"` trong exit_priority hợp lệ và luôn thoả _assert_viable_exit
  (engine.py:2318-2321, 2509-2512); các key recombine-only đều được pop trước khi dựng config → không key nào crash.
- Strategy = CÙNG strategy champion live wavestruct (t2646); bundle t2429 base ĐÃ TỒN TẠI
  (`bundle_n2_consw20_conv04_vg_combo_hb_nbpbw_2027-01-01_wf`, 5 entry head + 1 exit); cả `entry_lvup126_recov` lẫn
  `exit_vol_market` đều trong catalog.py (792, 1145) → serving tính được, không cần DB/retrain.
- **LỖ HỔNG DUY NHẤT**: đường serving (`generate_signals_from_bundle` → `recombine_signals`) KHÔNG chạy engine —
  `max_hold_bars` là tham số engine-only. **Serving host (stock-serving) phải TỰ enforce "đóng sau 16 phiên"**; CHƯA có
  bundle live nào dùng max_hold hữu hạn → chưa tiền lệ live. (Đã verify DB: mọi template maxhold đều có "max_hold"
  đứng đầu exit_priority — cảnh báo "thiếu max_hold trong priority" của trinh sát là nhầm, pr4_01 bác bỏ.)
- ML head load-bearing (tắt signal-exit → ×11.78, DD −31.5%): stack cần predict-path đầy đủ — chính là stack champion đang live.

## TỘI 6 — Composite counter-view: GỠ
- comp 637.6 vs gb 735: composite phạt vòng-quay (đã thành án từ fc_rule2/NAV_GEM_SCAN) — không phải bằng chứng chống bị cáo.
- Kịch bản bào edge: phí tăng (R1.0 sống 3.5sd f22), fill trượt/lag ±1 (sống, T3), cắt top-trade (edge tăng, T2),
  settlement không ứng (sống, mọi bảng) — KHÔNG kịch bản khảo sát nào bào sạch.
- **Dominate r2c mọi mặt** (r2c chỉ CONDITIONAL): f21 +37.9 vs +24.8; f22 adv +24.3 vs +9.4; f22 noadv **+22.7 vs −2.5 (r2c chết)**;
  f23 +15.0 vs +5.7; f24 +18.5 vs +7.7; DD tốt hơn; entry-lag sống vs LẬT. Bị cáo thay thế r2c hoàn toàn.

## PHÁN QUYẾT: **SURVIVED** (kèm 3 điều kiện promote — nhẹ hơn án r2c: KHÔNG có điều-kiện-tồn-tại nào về phí/lag/settlement)
1. **Công bố theo EV plateau + seed-mean**: full adv ~×21.0-21.4 (+52..+55%), noadv ~×19.0-19.4 (+45..+49%), f22 ~4.2/4.1 —
   không quote đỉnh mh12 ×22.39.
2. **Serving**: trước promote phải (a) implement + verify enforce max_hold 16 phiên ở stock-serving host (engine-only knob,
   chưa tiền lệ live), (b) export bundle t2907 + leakage-auditor 4-check + shadow-run so trades bit-level.
3. **Tripwire recency (kế thừa r2c)**: f25→nay tâm mh16 adv chỉ +0.7% (0.4sd) và DD-edge đảo (−13.5 vs −11.4);
   nếu rolling-18-tháng vs gb < −1sd 2 quý liên tiếp → demote về gb+mh16 hoặc gb. Cân nhắc **mh12 làm tâm dự phòng**
   (f25 sống cả 2 chế độ: +4.1%/2.5sd adv, +6.2%/3.4sd noadv; DD nông nhất −12.0).

## BẢNG HỘI ĐỒNG CUỐI — 3 LỰA CHỌN (nh_nav2 K25 R0.6, mean±sd 20 perm; adv | noadv)
| tiêu chí | gb_x08 nguyên bản (t2783) | gb+mh16 hiện đại (t2910) | t2429+mh16 đời-cũ (t2907) |
|---|---|---|---|
| NAV full 2020- | ×13.78±0.42 \| ×13.03±0.42 | ×20.16±0.58 (+46%) \| ×18.38±0.80 (+41%) | **×21.77±0.53 (+58%) \| ×19.62±0.60 (+51%)** |
| seeds 7/99/555 (full) | — (s42) | chưa chạy | ×20.4-21.3 \| ×18.8-19.9 (robust) |
| f22 | 3.43 \| 3.34 | 4.14 (+20.5%) \| 4.06 (+21.4%) | **4.27 (+24.3%) \| 4.10 (+22.7%)** |
| f23 | 2.46 \| 2.44 | 2.77 (+12.5%) \| 2.73 (+12.2%) | **2.83 (+15.0%) \| 2.77 (+13.8%)** |
| f21 / f24 (held-out) | ref | +33.2/+17.5 \| +33.4/+20.2 | **+37.9/+18.5 \| +34.3/+20.7** |
| f25→nay | ref (DDw −11.4) | −1.3% (−0.6sd) \| +2.4% (1.1sd) | +0.7% (0.4sd) \| +2.9% (1.5sd) → **tripwire** |
| LOYO | ref | chưa chạy | **7/7 ≥2sd cả 2 chế độ** (min +35.1%) |
| MaxDD (DDw full / bootstrap med-worst) | −14.2 / −14.6..−16.5 | −14.1 \| −13.9 | **−13.8 / −11.2..−14.7 \| −13.6** |
| crash 2025-04 | **−12.8%** | ~ | −13.4% (sâu hơn 0.6đ) |
| per-trade | **+9.32%, PF 6.02**, 1378 lệnh, hold 14/29.7 | +4.69%, PF 4.04, 2367 lệnh, hold 11.3 | +4.45%, PF 3.94, 2451 lệnh, hold 11 |
| phí R1.0 (f22 vs gb) | ref | ~ (chưa chạy) | **+11.8% (3.5sd) sống** |
| entry lag +1 cả 2 bên | thắng r2c, thua mh16 | chưa chạy | **+31.3% full (5.5sd) sống** |
| drop-top-20 | tựa top-trade | ~ | **+81.4% (13.1sd), edge tăng** |
| serving | đang live dạng ML, 0 việc | wheel OK; cần host enforce max_hold + có 16 key snr/hold-ext | wheel OK, bundle base ĐÃ có; cần host enforce max_hold; config GỌN hơn gb |
| vai trò đề xuất | fallback an toàn / demote-target | lựa chọn "hiện đại + cap" (thua ~8% adv/2sd) | **ứng viên chính** (điều kiện §Phán quyết) |

Files: pr4_00_out.txt, pr4_01_out.txt, pr4_10_out.txt, pr4_20_out.txt, pr4_21_out.txt, pr4_30_out.txt, pr4_40_out.txt;
trades seed mới: pr4_mh16_s7/s99/s555_trades.csv; clone DB: pr4_mh16 = t2918.
