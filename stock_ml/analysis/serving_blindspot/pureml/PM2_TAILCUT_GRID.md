# PM2 — LƯỚI TAIL-CUT + CHỐNG NGHẼN SỔ trên t1058 (pure-ML exit)
Ngày: 2026-07-10. Base: t1058 `n2_sx_rr_h10_thr2p5` (clone mới prefix `pm2_`, KHÔNG đụng 1058/2808 canonical). Trục hard_stop = `hard_stop_pct` (dấu ÂM) + `"hard_stop"` vào `exit_priority` (trap LINE_A §5); trục zX = `signal_threshold` (decoupled: SELL khi z(exit) > signal_threshold, experiment.py:2555). So chuẩn: pm_sxrr25 s42 644.8 (mean-5 630.5); champion 2646 mean-5 728.5; gb_x08 mean-5 733.8.

## Vòng 1 — lưới 3×3 config-only, seed 42 (composite; templates 2809–2817)

| hard_stop \ zX | 2.0 | **2.5** | 3.0 |
|---|---|---|---|
| **−10%** | 709.9 | **775.1** | 731.6 |
| −12% | 693.7 | 751.9 | 710.0 |
| −15% | 673.5 | 719.4 | 680.3 |

Chi tiết từng ô (pnl / pf / mdd_sym / trades / **n lệnh entry 2023-25** / worst):

| ô | pnl | pf | mdd | tr | n23-25 | worst |
|---|---|---|---|---|---|---|
| hs10_zx20 | 168.1 | 4.17 | .507 | 1092 | 287 | −.250 |
| **hs10_zx25** | **194.7** | **5.20** | .523 | **828** | **204** | −.241 |
| hs10_zx30 | 193.8 | 5.84 | .494 | 673 | 150 | −.241 |
| hs12_zx20 | 167.1 | 4.15 | .519 | 1036 | 270 | −.261 |
| hs12_zx25 | 193.6 | 5.20 | .528 | 775 | 189 | −.261 |
| hs12_zx30 | 193.8 | 5.91 | .491 | 617 | 137 | −.241 |
| hs15_zx20 | 165.2 | 4.13 | .537 | 972 | 252 | −.261 |
| hs15_zx25 | 190.4 | 5.06 | .562 | 717 | 177 | −.261 |
| hs15_zx30 | 190.7 | 5.73 | .514 | 569 | 129 | −.334 |

Cấu trúc RÕ: (i) stop càng NÔNG càng tốt (−10% > −12% > −15% ở mọi cột) — xác nhận hint t1053/t1052; (ii) zX 2.5 là đỉnh mọi hàng (2.0 nhả sổ quá sớm → nhiều lệnh nhỏ WR thấp; 3.0 ôm quá lâu → ít lệnh, mất confidence); (iii) baseline 644.8 → ô nhất 775.1 = **+130.3 chỉ bằng 1 key config**, đúng cỡ counterfactual clip −12% (+140) dự báo trong PURE_ML_CHAMPION §3.

### Vòng 1b — chống nghẽn bằng max_hold (trên ô nhất): FAIL dứt khoát
Engine CÓ key `max_hold_bars` + rule `"max_hold"` trong `exit_priority` (t1058 để 10000 và không đưa vào priority → chưa từng fire).

| biến thể | comp s42 | tr | n23-25 | kết luận |
|---|---|---|---|---|
| mh120 | 513.1 | 1368 | 522 | chém trọn runner (hold TB runner ~440-590d) → giết alpha |
| mh200 | 630.9 | 1187 | 397 | vẫn −144 vs không mh |

max_hold nhả sổ thật (n23-25 ×2.5) nhưng runner-riding CHÍNH LÀ alpha của họ này → mọi trần hold cứng đều phản tác dụng. Hard_stop −10% tự nó đã là cơ chế nhả sổ đủ: lệnh chết bị đá trong vài ngày thay vì treo 600-700 phiên (n23-25 = 204 vs 142 của t1058; tr 828 vs 591).

## Vòng 2 — multi-seed 5 seeds + regime + tail + runner

### Per-seed composite
| seed | pm2_hs10_zx25 | pm2_hs12_zx25 | champ 2646 | gb_x08 |
|---|---|---|---|---|
| 42 | **775.1** | 751.9 | 729.6 | 735.0 |
| 7 | 756.0 | 736.0 | 731.5 | 736.7 |
| 99 | 766.0 | 743.9 | 722.5 | 728.1 |
| 555 | 752.8 | 728.1 | 730.4 | 735.7 |
| 123 | 755.0 | 732.4 | 728.3 | 733.3 |
| **mean** | **761.0** | 738.5 | 728.5 | 733.8 |

pm2_hs10_zx25 thắng CẢ champion lẫn gb_x08 ở **từng seed một** (Δ mean +32.5 / +27.2, min-seed 752.8 vẫn > max của gb 736.7). Ô nhì hs12_zx25 mean 738.5 chỉ +4.7 vs gb, thua gb ở 3/5 seed → đúng là stop nông hơn tốt hơn, hs10 là lựa chọn duy nhất đáng promote.

### Regime test (pnl theo ENTRY-year slice, protocol gb_regime; trades per-seed)
| | 2020 | ≥2022 | ≥2023 | ≥2024 | ≥2025 | all | n |
|---|---|---|---|---|---|---|---|
| champ2646 s42 | 44.3 | 55.6 | 44.4 | 26.4 | 24.1 | 127.3 | 1384 |
| gb_x08 s42 | 45.4 | 56.6 | 45.3 | 27.2 | 24.9 | 128.5 | 1378 |
| pm2_hs10_zx25 s42 | 148.7 | 39.6 | 32.3 | 24.1 | 21.9 | 194.7 | 828 |
| s7 / s99 / s555 / s123 (≥2022) | 135.7–145.5 | 39.0–40.2 | 32.9–33.6 | 24.2–25.3 | 22.6–23.6 | 187.3–191.9 | 829–853 |

Đánh giá thẳng: mọi lát ≥2022/2023/2024/2025 DƯƠNG THỰC và ổn định qua 5 seed (khác hẳn t1058 gốc ≈0 hoặc âm) → qua sàn "regime ≥2022 bắt buộc". Nhưng pnl ≥2022 (~40) vẫn DƯỚI champ/gb (~56): lợi thế composite đến từ (a) cohort 2020 ăn gấp 3 (149 vs 45 — runner không bị clip) và (b) pf/trades. Đây KHÔNG phải ghost kiểu dsb60 (lát nào cũng dương, 2025 +22-24 sát gb), nhưng cũng chưa phải model "đều mọi năm".

### Tail-check (mục tiêu: hết lệnh −61%)
- worst per-seed −0.24..−0.25 (gap-down xuyên stop −10%, fill thực tế); n(≤−20%) = 10–15 / ~830 lệnh; p05 = −0.17 (t1058: worst −0.61, p05 −0.29, chuỗi DIG/AAV −55..−61%).
- exit mix s42: signal 395 / hard_stop 376 / open 57 — stop gánh ~46% lượt thoát, đúng vai "cắt lệnh chết sớm".

### Runner-check (ĐIỀU KIỆN SỐNG) — ĐẠT, thậm chí khuếch đại
| runner | t1058 gốc | pm2_hs10_zx25 |
|---|---|---|
| LPB 2022-11 | +469% | vài lần thử bị stop đá (−5%, +8%) rồi **re-enter 2022-11-18 → +405..+481% / 546-592d** (mọi seed) |
| VTP 2022-12 | +331% | entry 12/2022 bị stop; **re-enter 2023-02-13 → +369% / 530d** (mọi seed) |
| BSR 2025 | +188% | **+188..+208% / 204-206d** (mọi seed) |
- hold>200d: **142–145 lệnh, +182..+198u** (gốc: 35 lệnh +109.6) — stop −10% không giết ngách runner mà còn tăng số runner nhờ vốn quay vòng re-enter đúng chân sóng. Top winners 2020 (VCI/NKG/BCG/DGC +7..+9x) giữ nguyên.

## Vòng 3 — hiện đại hóa nhẹ trên pm2_hs10_zx25 (seed 42)
| biến thể | thay đổi | comp s42 | Δ vs base 775.1 |
|---|---|---|---|
| (a) `pm2_hs10_zx25_recovfs` (t2820) | entry feature set `entry_lvup126_lean` → `entry_lvup126_recov` (bộ hiện đại của champ/gb; entry head retrain) | 770.4 | −4.7 |
| (b) `pm2_hs10_zx25_snr` (t2821) | + `exit_snr_extend` 3 key non-default của gb_x08 (thr 0.8 / min_gain 0.27 / giveback 0.08; window 20 = default) — rule mềm defer-signal | 776.7 | **+1.6** |

Kết luận vòng 3: cả hai đều KHÔNG dịch kim đáng kể — feature "hiện đại" không phải nguồn alpha còn thiếu (khớp chẩn đoán (iv) trong PURE_ML_CHAMPION §3: entry đã đủ tốt); snr-extend chồng lên gần như trung tính vì họ này vốn đã ôm runner bằng chính zX (không có velocity-exit để phải defer). Base pm2_hs10_zx25 giữ nguyên là ứng viên; (b) chỉ đáng chạy multi-seed nếu cần vét +1..2 điểm.

## Verdict
**Tuyến thuần-ML + tail-cut ĐÃ VƯỢT vùng champion**: pm2_hs10_zx25 (t1058 + hard_stop −10% + zX 2.5) mean-5 = **761.0** vs champ 728.5 / gb_x08 733.8, thắng từng seed, mốc "730+ = ứng viên thật" đạt với biên +27. Một key config đóng đúng tử huyệt số 1 (tail −61% → −25%, p05 −29% → −17%) và đồng thời giải một nửa tử huyệt số 2 (591 → ~830 trades; n lệnh 2023-25 = 204 vs 142) mà KHÔNG giết ngách runner-riding (LPB/VTP/BSR sống, hold>200d +182..+198u). max_hold bị bác bỏ bằng số liệu (513/631). Điểm còn hở khi ra hội đồng promote: pnl lát ≥2022 (~40) vẫn dưới champ (~56) — alpha nghiêng về cohort 2020; nếu cần "đều mọi năm" thì bước kế tiếp là cơ chế nhả-sổ MỀM (ML risk-head / re-rank vốn theo zX đương thời) chứ không phải trần hold. Vòng 3 xác nhận thêm: alpha nằm ở cặp head t1058 + stop nông, không phải ở feature hiện đại (recovfs −4.7) hay snr-extend (+1.6).

## Files & templates
- Scripts: pm2_01_grid.py (+ pm2_01_grid.log, pm2_01_extra.log), pm2_02_seeds.py (+log), pm2_03_checks.py, pm2_04_modern.py (+log)
- Trades CSV: pm2_pm2_hs*_s42_trades.csv (9 ô), pm2_pm2_hs10_zx25_s{7,99,555,123}, pm2_pm2_hs12_zx25_s{7,99,555,123}
- Templates: 2809–2817 (lưới), pm2_hs10_zx25_mh120/mh200, pm2_hs10_zx25_recovfs / _snr (vòng 3)
