# HYBRID VERDICT — 3 mảnh ghép pm/pm2 vào tuyến gb_x08: kiểm chứng + số phận

Ngày: 2026-07-10. Bối cảnh: pm2_hs10_zx25 đã bị REJECT (PM2_PROSECUTION: ghost 2020 + NAV ×5.61 + sổ nghẹt 50%);
câu hỏi còn lại: có MẢNH nào của họ pure-ML đáng ghép vào gb_x08 (t2783, 735.0 s42) không, hay toàn bộ cho xuống.
Án đã có từ trước: reward_risk h10 làm kênh BÁN thêm = chết (EXIT2_CHANNEL_RESULTS, best −3.3, 91-98% trùng force).
Scripts: hy_01_swing.py, hy_02_ens5_offline.py, hy_03_zxgate.py, hy_04_ens5_run.py (+logs/hy_04_ens5.log).
Nguồn score offline: bundle_n2_pbfill_p0p03_w25_2026-01-01_wf (đúng cặp head t1058: continuation h6/lean +
reward_risk h10/vol_market, seed 42, WF, prediction_history 2020-01→2026-06-08). Không đụng canonical 2646/2730/2783.

## Mảnh C — swing-bỏ-phí trong lệnh ôm dài của pm2 vs champion chop cùng sóng (hy_01_swing.py)

Cohort: 208 lệnh hold>100d của pm2_hs10_zx25 s42 (chiếm +214.7u trên tổng +194.7u của run — phần còn lại âm ròng).
Sóng-ngược = zigzag hindsight 15% trên close trong đời lệnh. Oracle = bán đúng đỉnh, mua lại đúng đáy từng sóng
hoàn chỉnh, trừ cost 0.7%/vòng; missed_u = (1+pnl)·(Πfactors−1). Đối chứng: MỌI trade champ 2646 / gb_x08 s42
cùng mã có entry trong [entry, exit] của lệnh pm2 (unit-weight như composite).

- Sóng-ngược ≥15%: **337 sóng trong 148/208 lệnh**; ≥20%: **184 sóng trong 99 lệnh**. Downleg cuối (chưa recover,
  chính là giveback lúc thoát) ≥15%: 90 lệnh, ~+41.5u depth — đó là chuyện exit, không phải chop.
- **Oracle chop = +371.1u bỏ phí** (2020: 237.8; ≥2022: 130.2) — đúng như user quan sát: trong bụng các lệnh ôm
  546d có rất nhiều sóng bỏ phí. NHƯNG đây là trần thần-thánh (bắt đúng đỉnh/đáy từng sóng).
- **Champion chop THỰC TẾ trên cùng mã + cùng khoảng thu ít hơn cả ôm**:

| lát (entry-year lệnh pm2) | n lệnh pm2 | pm2 ôm (u thực) | oracle bỏ phí | champ chop thực | gb chop thực |
|---|---|---|---|---|---|
| ALL | 208 | **+214.7** | +371.1 | +110.2 (991 tr) | +111.1 (985 tr) |
| ≥2022 | 134 | **+59.6** | +130.2 | +40.9 (633 tr) | +41.6 |
| ≥2023 | 83 | **+29.8** | +35.7 | +23.4 (272 tr) | +24.1 |
| chỉ lệnh có sóng≥15% | 148 | **+197.0** | +371.1 | +98.5 | +98.8 |

- Per-window: pm2 ôm > champ chop ở **140/208 cửa sổ** (pm2 +207.6 vs champ +90.0); champ chỉ thắng 68 cửa sổ
  nhỏ (champ +20.3 vs pm2 +7.0).
- **Phân xử**: cả hai đều "sai nghề sóng" so với oracle, nhưng giữa hai nghề có thật thì **ôm > chop ~2×
  trên cùng sóng, ở mọi lát năm** (unit-weight). 371u bỏ phí KHÔNG thuộc về ai: máy chop thật của champion
  (force-gate + re-entry pullback) chỉ nhặt được ~30% giá trị sóng, và mọi cơ chế bán-tĩnh nhằm bắt đỉnh sóng
  đã bị falsify (EXIT2, TWOSIDED, QUANTILE_TRIGGER). Số này KHÔNG lật án NAV của pm2 (u unit-weight, sổ nghẹt
  vẫn nguyên) — nó chỉ nói: đừng mong "dạy gb chop giỏi hơn" bằng head pm2, vì chop thực tế vốn thu ít hơn ôm.

## Mảnh A — entry continuation h6 làm ensemble 5 của union champion (hy_02 offline + hy_04 run)

Offline (join bundle pbfill × sa_scores buy_union, 92,992 bar chung 2020-04→2026-06):
- **corr(z_cont6, z_score3 continuation-h10 sẵn có của gb) = 0.849** — head gần như trùng thông tin với slot 3.
- Buy-bars THÊM (fire mà buy_union chưa fire), ngưỡng ensemble-style:

| năm | bars | union có | z>0.7 fire/THÊM | z>0.9 fire/THÊM | z>1.2 fire/THÊM |
|---|---|---|---|---|---|
| 2023 | 15,189 | 8,597 | 3,304/212 | 1,761/77 | 551/4 |
| 2024 | 15,243 | 9,324 | 3,919/**87** | 2,256/24 | 687/5 |
| 2025 | 15,121 | 12,200 | 4,728/**111** | 3,084/46 | 1,417/14 |
| 2026H1 | 6,173 | 4,499 | 2,963/**469** | 2,441/355 | 1,704/231 |

- Vùng đói 2024-25: thêm 87-111 bar/năm (~0.6-0.7% số bar), trong đó chỉ ~20% là "fresh" (không nằm cạnh ±3 bar
  một cụm union cũ) = **24 bar fresh cho cả 2024-25**. Cụm đáng kể duy nhất là 2026-04/05 (90 bar fresh, dồn
  SHB/SSI/FRT/NLG) — 6 tuần cuối dữ liệu, không đo được pnl.
- Run thật seed-42 (clone 2783 + `entry_ensemble5` = continuation h6/pen1.0/tw70, features entry_lvup126_lean;
  templates hy_ens5_z07=2828, hy_ens5_z09=2829):

| run | tmpl | comp s42 | Δ gb 735.0 | pnl | trades | khác biệt trade-level |
|---|---|---|---|---|---|---|
| hy_ens5_z07 | 2828 | 735.4 | **+0.4** | 128.5 | 1379 | +4/−3 lệnh (VHM/ACB/SAB xê dịch entry, SBT thêm −0.04%), tất cả 2024-25, net pnl ≈ +0.04u |
| hy_ens5_z09 | 2829 | 735.0 | **+0.0** | 128.5 | 1378 | **trùng gb_x08 từng lệnh một** (join symbol+entry_date: 1378/1378 both) |

- Đúng tiền lệ "union head thêm buy-bars = fill-bound": 667 bar thêm (z0.7) → **+1 trade net**; cụm fresh
  2026-05 không chuyển thành trade nào (mã đang occupied / cooldown / pullback không fill). Cả hai mức
  << ngưỡng +2 → DỪNG theo protocol (không 5-seed).
- **Verdict A: CHẾT** — head continuation h6 lean không mang thông tin entry mới cho union (corr 0.849 với
  slot 3 sẵn có); phần "đói tín hiệu 2024-26" của gb không phải do thiếu head này. Templates 2828/2829 + run
  giữ trong DB làm chứng cứ; không nâng cấp.

## Mảnh B — zX (reward_risk h10) làm cổng-giữ cho gb_x08 (hy_03_zxgate.py)

Câu hỏi: tại decision bar (bar trước exit_date) của lệnh gb_x08 bị đóng, zX có tách "đóng đúng" khỏi
"đóng non" (rallied = post_max_c≥5%/20bar) không? Hướng defer thiết kế: zX THẤP = head nói "chưa đến lúc bán"
= sóng còn khỏe → lẽ ra nên GIỮ. Ngưỡng sống: AUC ≥0.58 ổn định ≥2022.

| cohort | AUC(−zX→rallied) ALL | ≥2022 | ≥2024 | theo năm |
|---|---|---|---|---|
| 648 dl12-closed | 0.481 | 0.461 | 0.482 | 2020: .368, 2021: .445, 2022: .357, 2023: .571, 2024: .523, 2025: .442, 2026: .591 |
| toàn bộ 1377 | 0.454 | 0.457 | 0.452 | 2020: .318, 2021: .439, 2022: .286, 2023: .546, 2024: .437, 2025: .444, 2026: .459 |

- **Hướng defer (zX thấp→giữ) DƯỚI random ở mọi lát gộp.** Chiều NGƯỢC (zX cao→rallied) chỉ đạt ~0.55 pooled,
  và chính chiều đó là vô nghĩa vận hành: nó bảo "hoãn bán khi head đang hét BÁN (zX cao)" — đúng mẫu
  anti-selective/calibration-đảo mà EXIT2 §4 đã đo (zX cao 2025 = bán trúng người thắng).
- Decile-lift xác nhận: P(rallied) TĂNG theo zX (decile 0: .500 → decile 9: .594; post_max_c .057→.093) —
  head không có tín hiệu giữ, nó có tín hiệu ngược yếu + đảo dấu theo era (quintile 2020-21 AUC-đảo .615,
  2022-23 hình chữ U, 2024-26 phẳng ~.45).
- **Verdict B: CHẾT như TWOSIDED** — không đạt 0.58 ở bất kỳ chiều/lát nào ≥2022; không thiết kế knob defer.
  Cộng với án x2_ (kênh bán thêm chết): head reward_risk h10 nay đã bị đo ở CẢ HAI chiều (bán thêm, giữ thêm)
  trên stack gb_x08 — đều không có thông tin. File: hy_03_zxgate.csv.

## PHÁN QUYẾT SỐ PHẬN

**KHÔNG mảnh nào sống. Họ pm/pm2 = superseded toàn phần đối với tuyến gb_x08.**

| mảnh | kết quả | số quyết định |
|---|---|---|
| C — swing-bỏ-phí | Quan sát của user ĐÚNG về oracle (+371u trong 208 lệnh ôm) nhưng chop thực tế của champion trên cùng sóng chỉ thu +110u < ôm +215u — không có cơ chế thật nào ăn được số bỏ phí đó | ôm > chop ở 140/208 cửa sổ, mọi lát năm |
| A — continuation h6 ens5 | chết (fill-bound + trùng thông tin slot 3) | z07 +0.4, z09 +0.0 (trùng từng lệnh); corr 0.849 |
| B — zX cổng-giữ | chết như TWOSIDED (hướng defer dưới random; hướng ngược = anti-selective đã biết) | AUC ≥2022 = 0.457-0.461, cần ≥0.58 |

Với án x2_ trước đó (reward_risk h10 kênh bán thêm: best −3.3), cả BA cách khai thác head-pair t1058 trên stack
gb_x08 (mua thêm, bán thêm, giữ thêm) đã bị đo và bác bằng số. Giá trị duy nhất còn lại của họ này vẫn là câu
trong PM2_PROSECUTION: nguyên liệu cho sleeve runner-riding vốn tách riêng — không phải mảnh ghép cho gb_x08.

### Đề nghị mark superseded=true (CHỜ USER DUYỆT, chưa update)

Các row pm2_/pr2_ đang đứng trên gb_x08 (735.0) trên leaderboard (mỗi template 1 row, row = seed chạy cuối):

| run_name | tmpl | seed | composite |
|---|---|---|---|
| pr2_hs08_zx25 | 2822 | 42 | 798.6 |
| pr2_hs09_zx25 | 2823 | 42 | 790.8 |
| pm2_hs10_zx25_snr | 2821 | 42 | 776.7 |
| pm2_hs10_zx25_recovfs | 2820 | 42 | 770.4 |
| pr2_hs11_zx25 | 2824 | 42 | 767.4 |
| pm2_hs10_zx25 | 2810 | 123 | 755.0 |

(Các row pm_/pm2_ còn lại đều < 735.0, không che gb_x08; có thể mark cùng đợt cho sạch nếu user muốn:
pm2_hs12_zx25 732.4, pm2_hs10_zx30 731.6, pm2_hs15_zx25 719.4, pm2_hs12_zx30 710.0, pm2_hs10_zx20 709.9,
pm2_hs12_zx20 693.7, pm_ctl 681.0, pm2_hs15_zx30 680.3, pm2_hs15_zx20 673.5, pm_dip035 669.8, pm_dip04_coh09
631.4, pm_dip035_coh09_trail35 631.1, pm2_hs10_zx25_mh200 630.9, pm_dip035_coh09_ox16 630.5, pm_dip035_coh09
629.2, pm_sxrr25 622.7, pm_dip03_coh09 615.6, pm_dip035_coh07 608.3, pm2_hs10_zx25_mh120 513.1.)

Lý do án: composite unit-weight của cả họ là tiền giấy hai lớp (PM2_PROSECUTION Tội 1+2 — ghost 2020 + NAV
×5.61/−50% MaxDD/sổ nghẹt 50% lệnh); ba mảnh ghép khả dĩ cuối cùng vừa bị bác ở hồ sơ này; giữ các row này
đứng trên gb_x08 chỉ gây nhiễu cho mọi vòng so sánh sau.

## Files
- hy_01_swing.py → hy_01_swing.csv (208 lệnh, per-window pm2 vs champ/gb)
- hy_02_ens5_offline.py (buy-bars thêm theo năm), hy_04_ens5_run.py → hy_hy_ens5_z07/z09_s42_trades.csv,
  logs/hy_04_ens5.log; templates hy_ens5_z07=2828, hy_ens5_z09=2829 (clone 2783, KHÔNG đụng canonical)
- hy_03_zxgate.py → hy_03_zxgate.csv (1377 lệnh gb_x08 + zX decision-bar + rallied)
