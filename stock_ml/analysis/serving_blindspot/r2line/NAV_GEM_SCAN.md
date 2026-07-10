# NAV GEM SCAN — quét NAV toàn kho template (tìm "fc_rule2 tiếp theo")
Ngày: 2026-07-10. Scripts: `navscan/nv_00_recon.py` (recon kho), `nv_01_proxy.py` (+`nv_01_candidates.csv`), `nv_02_families.py`, `nv_03_check_trades.py`, `nv_10_dump.py` (dump trades), `nv_11_nav.py` (+`nv_11_results.csv`, log `nv_11_out.txt`), `nv_20_deep.py` (đào sâu), `nv_30_rerun_maxhold.py` (re-run s42, log `nv_30_out.txt`).

Thước: `na_audit/nh_nav2.py` NGUYÊN BẢN (verify 9/9 anchor trước khi chạy) — K25, R0.6, settle T+2, advance 0.08%/vòng, shuffle-mean±sd 20 perm. Anchor cùng máy cùng ngày: **gb_x08 ×13.78±0.42 / f22 ×3.43±0.08 / DD −15.3%**; **r2c_oxt04_p42 ×16.84±0.75 / f22 ×3.76±0.09 / DD −13.3%**. Ngưỡng gem: ≥ gb+8% (×14.88).

## 1. Proxy screening (bước 1)

Kho: 3.056 rows leaderboard_runs (kể cả superseded), 2.389 template có run. Proxy: velocity = pnl/hold-day (fc_rule2 6.8 vs gb 4.3), pnl×√trades, gate trades≥1200 & pf≥2 & mdd_sym≤0.40 & hold≤30. LOẠI án cũ: pm/pm2, t903/t1058-nopb, xr_, r17/ruleexp, fc_/r2*/n3 (họ fc_rule2 đã NAV). LOẠI khung: w0_pyr_* (comp 983 nhưng là pyramid sizing overlay trên champ 2646 — K-slot sim không biểu diễn được multi-lot; tuyến sizing đã hoãn). 15 ứng viên đa dạng (mỗi gia tộc/era 1-3 đại diện):

| # | template | gia tộc/era | comp | pnl | pf | mdd_sym | tr | hold | vel |
|---|---|---|---|---|---|---|---|---|---|
| 1 | t2429 n2_consw20_conv04_vg_combo_hb_nbpbw | 2429-base (tiền thân wavestruct) | 719.1 | 125.2 | 4.77 | .19 | 1796 | 20.0 | 6.3 |
| 2 | t2516 n2_2429_maxhold20_cagr | 2429 + max_hold 20 | 645.4 | 111.1 | 3.93 | .20 | 2305 | 12.0 | 9.2 |
| 3 | t2531 n2_2429_dyncsr88 | 2429 + head cross-sectional dynamics | 712.0 | 124.6 | 4.55 | .19 | 1839 | 19.6 | 6.4 |
| 4 | t2234 n2_velov_univ150c | velov universe-150 | 656.0 | 292.3 | 4.07 | .23 | 4577 | 17.1 | 17.1 |
| 5 | t2222 sw_ox08_pb03 | unmask-era champ pb03/ox08 | 636.1 | 116.6 | 3.58 | .23 | 2087 | 17.5 | 6.7 |
| 6 | t2040 qQ_rsdrop_d15 | unmask RS-drop exit | 516.9 | 84.0 | 3.64 | .18 | 2155 | 8.3 | 10.1 |
| 7 | t1799 n2_v19_fullwave | v19 fullwave | 602.0 | 106.0 | 3.86 | .22 | 1844 | 16.5 | 6.4 |
| 8 | t1410 n2_am20_oxt03 | oxtrail siêu nhanh (era 1378) | 396.2 | 68.5 | 2.54 | .21 | 3617 | 4.2 | 16.3 |
| 9 | t1306 n2_mgz13_emgz09 | mgz market-gate (era 1258) | 509.6 | 92.1 | 2.91 | .23 | 2908 | 9.1 | 10.1 |
| 10 | t1644 n2_dz_act05_m10_fl03_cap08 | dead-zone vol-trail (era 1586) | 480.5 | 90.3 | 2.70 | .24 | 3351 | 8.0 | 11.3 |
| 11 | t971 n2_pb_xrule_bear3 | champ-958 + bear force-exit | 411.4 | 100.3 | 2.15 | .40 | 4296 | 9.9 | 10.1 |
| 12 | t1194 n2_1187_age4 | age-incubation | 472.2 | 92.8 | 2.72 | .29 | 2492 | 13.1 | 7.1 |
| 13 | t2658 n2_smac_v52_gruraw_vfast_ridepb | SMAC (chưa từng NAV) | 185.0 | 50.5 | 2.80 | .32 | 767 | 49.4 | 1.0 |
| 14 | t625 velexit_contE_h10_u3_e020_x040 | cont-era regression_dual_ml | 151.9 | 115.8 | 1.88 | .56 | 2393 | 54.6 | 2.1 |
| 15 | t200 zzf6_zzpk_p06_t5_e48_x52 | zigzag_dual_ml | 188.2 | 64.7 | 1.92 | .58 | 1101 | 71.6 | 0.9 |

Trades: TẤT CẢ còn trong run_trades (không cần re-run hàng loạt); riêng t2516 chỉ có run s555 → sau khi lộ gem đã clone re-run s42 (mục 3).

## 2. Bảng NAV-hóa (advance 0.08%, K25, R0.6, shuffle 20 perm)

| template | gia tộc | full ±sd | Δ vs gb | f22 ±sd | Δ | MaxDD (worst) | verdict nhanh |
|---|---|---|---|---|---|---|---|
| **n2_2429_maxhold20** (s42 clone) | 2429-maxhold | **×19.78±0.53** | **+43.5%** | **×4.12±0.07** | **+20.1%** | −12.4% | **GEM** |
| n2_velov_univ150c | velov-univ150 | ×22.40±1.94 | +62.6% | ×3.08±0.18 | −10.2% | −20.8% (−22.0) | REJECT regime (ghost 2020-21) |
| n2_2429_base (t2429) | 2429-base | ×17.90±0.56 | +29.9% | ×3.74±0.07 | +9.0% | −12.9% | vượt gb rõ, ngang r2c |
| n2_2429_dyncsr88 | 2429-dyncsr | ×17.42±0.59 | +26.4% | ×3.76±0.07 | +9.6% | −12.9% | như base (không cộng thêm) |
| n2_v19_fullwave | v19 | ×16.29±0.78 | +18.2% | ×3.40±0.08 | −0.9% | −12.5% | dưới r2c, f22 ngang gb |
| *(r2c_oxt04_p42 — chuẩn hiện tại)* | R2 line | *×16.84±0.75* | *+22.2%* | *×3.76±0.09* | *+9.6%* | *−13.3%* | anchor |
| n2_dz_act05_cap08 | deadzone-trail | ×13.96±0.34 | +1.3% | ×3.18±0.05 | −7.3% | −17.0% | noise |
| n2_mgz13_emgz09 | mgz-gate | ×13.60±0.41 | −1.3% | ×2.97±0.07 | −13.4% | −13.1% | noise/âm |
| qQ_rsdrop_d15 | unmask-rsdrop | ×12.92±0.24 | −6.2% | ×3.13±0.04 | −8.7% | −13.6% | âm |
| sw_ox08_pb03 | unmask-sw | ×12.84±0.48 | −6.8% | ×2.99±0.07 | −12.8% | −14.3% | âm |
| n2_am20_oxt03 | oxtrail-fast | ×9.47±0.15 | −31% | ×2.54±0.02 | −26% | −9.2% | chết (pnl/lệnh quá mỏng) |
| n2_1187_age4 | age-incub | ×8.84±0.41 | −36% | ×2.34±0.09 | −17.2% | −16.6% | chết |
| n2_pb_xrule_bear3 | champ958-bear | ×6.53±0.44 | −53% | ×1.57±0.07 | −54% | −42.2% | chết |
| smac_v52_ridepb | SMAC | ×3.21±0.15 | −77% | ×1.20±0.03 | −30% | −29.8% | họ chết |
| zzf6_zzpk_p06 | zigzag | ×2.65±0.19 | −81% | ×1.14±0.04 | −52% | −51.6% | họ chết |
| velexit_contE_h10 | cont-era | ×2.22±0.23 | −84% | ×0.78±0.04 | −77% | −50.5% | họ chết |

Bài học proxy: velocity thô overrate khi pnl/lệnh quá mỏng (am20_oxt03 vel 16.3 nhưng NAV ×9.5 — net/lệnh 1.9% không gánh nổi cost 0.6%/vòng). Vùng sống: hold 10-20 + pf ≥3.9 + selection ML.

## 3. Đào sâu top (≥ gb+8%)

5 ứng viên vượt ngưỡng: maxhold20, velov_univ150c, 2429_base, dyncsr88, v19_fullwave.

| frame | maxhold20 (s42) | 2429_base | dyncsr88 | velov150 | v19_fw | gb | r2c |
|---|---|---|---|---|---|---|---|
| f23 | **×2.77±0.06** | ×2.51 | ×2.50 | ×2.45 | ×2.60 | ×2.46 | ×2.61 |
| LOYO bỏ 2021 | **×8.58±0.23** | ×7.96 | ×7.67 | ×6.43 | ×6.83 | ×7.43 | ×7.11 |
| drop-top-20 winner | **×15.05±0.46** | ×15.01 | ×14.30 | ×19.38 | ×12.74 | ×9.23 | ×13.55 |
| **NO-ADVANCE full** | **×17.18±0.59** | ×16.76 | — | — | — | ×13.03 | ×13.25 |
| NO-ADVANCE f22 | **×3.92±0.09** | ×4.00 | — | — | — | ×3.34 | ×3.26 |

- **n2_2429_maxhold20 = GEM THẬT** (clone s42 `nv_2429_maxhold20` t2903, comp 649.1, 2309 trades, hold median 11, mdd_sym .197 — khớp s555, seed-robust; NAV s42 ×19.78 vs s555 ×19.85):
  - Thắng gb MỌI mặt cắt: full +43.5% (≈11sd), f22 +20.1%, f23 +12.6%, LOYO-2021 +15.5%, drop-top20 +63%, MaxDD thấp hơn 2.9đ (−12.4 vs −15.3).
  - Thắng cả chuẩn r2c_oxt04_p42: full +17.5%, f22 +9.6%, f23 +6.1%, DD tốt hơn.
  - **Sống ở chế độ KHÔNG ứng trước tiền bán** (khác hẳn tuyến R2 vốn chết no-adv): ×17.18 vs gb ×13.03 = +31.8% (~5.9sd), f22 +17.4%. Advance fee KHÔNG phải điều kiện tồn tại → gem bền hơn r2c về vận hành.
  - Per-year pnl (s42): 2020 31.3 / 2021 29.3 / 2022 12.5 / 2023 12.4 / 2024 7.4 / 2025 17.9 / 2026 1.1 — dương 7/7, KHÔNG ghost.
- **Cơ chế — cùng công thức fc_rule2, nguồn sức mạnh lai**: t2516 = t2429 + đúng 1 knob `max_hold_bars: 20` (exit_priority thêm max_hold trước trail/ox/signal). Hold p95 121→21 phiên; 831/2309 lệnh exit qua max_hold (+78.9u); winner p95 bị chém từ +35.5% còn +26.1% nhưng vòng quay bù dư (đúng công thức vòng-quay-nhanh của fc_rule2). Khác fc_rule2 ở chỗ selection = 4-head ML ensemble champion stack (không phải rule gate) → pf 3.94 vs 3.45. Overlap trades: vs gb 56.8% exact, vs r2c 45.0% → nửa cohort riêng, diversify MỘT PHẦN, không phải alpha nguồn hoàn toàn mới. Mô tả template gốc đã tự khai "COMPOUNDING-OPTIMIZED, composite −55 vì objective khác" — kho ĐÃ chứa lời giải, composite dìm nó xuống hạng 645, không ai NAV-hóa cho tới nay.
- **Gia tộc 2429 nói chung bị composite dìm**: t2429 (tiền thân wavestruct, comp 719 < champ 735) NAV ×17.90 vs gb ×13.78 — nhánh nâng cấp 2643-wavestruct (hold-extension, +16 composite) làm GIẢM NAV ~23%. Đây là ca thứ 2 (sau fc_rule2) composite phạt quay-vòng-nhanh, lần này ngay trong dòng champion.
- **n2_velov_univ150c: REJECT theo chuẩn regime ≥2022** — full ×22.4 nhưng pnl dồn 2020-21 (91+85/292u), f22 ×3.08 và f23 ×2.45 đều THUA gb, DD −20.8%. Overlap gb chỉ 13% → alpha nguồn khác thật (universe 150 mã) nhưng template này không dùng được; hướng "universe expansion" nên nghiên cứu riêng (trên stack 2429/maxhold), không phải qua template này.
- v19_fullwave: overlap r2c 74% (họ hàng cơ chế), NAV dưới r2c mọi frame → không mở gì mới. dyncsr88 ≈ 2429_base (head dynamics không cộng NAV).

## 4. Verdict

1. **Kho CÒN gem, và gem số 1 = n2_2429_maxhold20_cagr (t2516, clone s42 = nv_2429_maxhold20 t2903)**: NAV ×19.78±0.53 advance / ×17.18±0.59 no-advance, f22 ×4.12, DD −12.4% — vượt gb +43.5%/+31.8%, vượt cả r2c +17.5%, sống mọi mặt cắt công tố sơ bộ (f22/f23/LOYO/drop-top/no-adv/seed). Frontier NAV mới của kho.
2. Phân loại: **cùng công thức vòng-quay-nhanh với fc_rule2** (cap hold, recycle vốn) **nhưng chồng lên ML selection** — hợp nhất đúng 2 nguồn mà FAMILY_CHAMPIONS đã chẩn đoán tách rời (gap fc_rule2 = "ML selection"; gap gb = "mù vốn"). Không phải alpha nguồn mới hoàn toàn (overlap gb 57%) — giá trị diversify trung bình, giá trị NAV rất lớn.
3. Đề xuất mở tuyến (CHƯA chạy) — "R3 / maxhold line" trên t2903:
   - Sweep `max_hold_bars` 12/15/20/25/30 + exit_priority variants (đỉnh chưa chắc ở 20; khả năng cao có ridge như pullback R2).
   - Áp max_hold lên gb_x08 stack (2783, có snr-defer) và 2646 wavestruct — hỏi: snr-extend có cộng hưởng hay triệt tiêu với cap hold?
   - Kênh signal-exit của maxhold20 đang −16.2u (kênh xả rác, cùng hình fc_rule2) → thử defer/lọc.
   - Công tố đầy đủ chuẩn r2c: K-sweep 15-30, 5-seed, cost R0.7/0.9, LOYO full 7 năm, so obs no-advance (đã dương sẵn).
4. Kết quả âm trung thực: SMAC, zigzag, cont-era, champ958-bear, age-incubation, fast-exit era 1137-1644 (oxtrail/mgz/dz), unmask sw/qQ — NAV ngang hoặc thua gb, đóng ở NAV-frame. w0_pyr để lại cho tuyến sizing (khung K-slot không đo được).

## Files
- `navscan/nv_00..nv_30*.py`, `nv_01_candidates.csv`, `nv_11_results.csv`, `nv_11_out.txt`, `nv_30_out.txt`
- Trades: `navscan/nv_<name>_s42_trades.csv` (15 ứng viên) + `nv_nv_2429_maxhold20_s42_trades.csv` (clone s42)
- Clone DB: `nv_2429_maxhold20` = t2903 (KHÔNG đụng canonical; không commit git)
