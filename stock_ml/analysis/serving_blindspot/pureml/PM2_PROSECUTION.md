# PM2 — HỒ SƠ CÔNG TỐ: pm2_hs10_zx25 (t2810, t1058 + hard_stop −10% + zX 2.5)
Ngày: 2026-07-10. Công tố độc lập, adversarial. Bị cáo: mean-5 composite 761.0 (thắng champ 728.5 / gb_x08 733.8 ở từng seed). Hồ sơ bào chữa: PM2_TAILCUT_GRID.md. Baseline NAV chuẩn: BASE gb_x08 K25 = ×14.44 / CAGR 50.67% / MaxDD −15.31% (tái lập chính xác bằng pr2_04_navsim.py trước khi xử).

## Tội 1 — Ghost 2020 (subframe composite ≥cut, sv_subframe protocol): **THÀNH LẬP**
pm2 s42 vs gb_x08 s42 (n_sym=61, cùng frame):

| cut | pm2 comp | gb comp | Δcomp | pm2 pnl | gb pnl | pm2 pf/mdd | gb pf/mdd |
|---|---|---|---|---|---|---|---|
| full | 775.1 | 735.0 | +40.1 | 194.7 | 128.5 | 5.20/.523 | 6.02/.175 |
| ≥2022 | 93.8 | 368.5 | **−274.7** | 39.6 | 56.6 | 1.90/**.456** | 3.71/.146 |
| ≥2023 | 176.2 | 306.3 | **−130.1** | 32.3 | 45.3 | 3.42/.157 | 3.95/.133 |
| ≥2024 | 142.6 | 201.9 | **−59.3** | 24.1 | 27.2 | 3.12/.146 | 3.47/.116 |

(vs champ 2646: −269.1 / −125.5 / −54.6 — cùng cấu trúc.) Per-entry-year delta pnl vs gb: **2020 +103.3**; 2021 −20.1, 2022 −4.1, 2023 −9.8, 2024 −0.2, 2025 −4.2 (2026 +1.2 nhiễu). Bỏ entries 2020-21 khỏi CẢ HAI bên → gb thắng tuyệt đối mọi chiều (pnl, pf, mdd, comp). **Toàn bộ lợi thế +27..+40 của bị cáo nằm ở cohort 2020; mọi năm khác đều THUA gb.** Khác dsb60: các lát ≥2022 vẫn dương thật (không phải ghost-âm), nhưng mdd_sym ≥2022 = 0.456 (gấp 3 gb) cho thấy tail-risk hậu-2020 vẫn nguyên — lợi thế là ghost, mô hình hậu-2022 là bản kém hơn của gb.

## Tội 2 — NAV thật với vốn thật (pr2_04_navsim.py, khung chuẩn dl_01 BASE, K25, 100% NAV, cost trong pnl CSV): **THÀNH LẬP — ÁN TỬ**

| config | final | CAGR | MaxDD | underwater dài nhất | 2022 | fills/skip_cash |
|---|---|---|---|---|---|---|
| BASE gb_x08 s42 K25 | **×14.44** | 50.67% | **−15.31%** | 120d | +31.9% | 1021/357 |
| champ 2646 s42 K25 | ×14.06 | 50.05% | −13.76% | 120d | +31.3% | 1028/356 |
| **pm2 s42 K25** | **×5.61** | 30.31% | **−50.28%** (2022-11-15) | **639d** | **−33.9%** | 418/**410** |
| pm2 s555 K25 | ×5.51 | 29.97% | −47.96% | 589d | −33.3% | 437/416 |
| pm2 s42 K50 | ×6.26 | 32.53% | −41.42% | 359d | −30.4% | 706/122 |
| pm2 s42 K15 | ×6.40 | 32.96% | −49.10% | 640d | −35.0% | 261/567 |

- **Đảo ngược hoàn toàn kết luận composite** (đúng tiền lệ pyramid): +27 composite → −61% NAV cuối, MaxDD gấp 3.3 lần, 2 năm dưới đỉnh.
- Start 2022-01-01: pm2 ×1.27 (CAGR **5.5%**, MaxDD −48.9%) vs gb ×3.34 (CAGR 30.7%, MaxDD −13.6%). Ai vào vốn từ 2022 gần như đứng yên 4.5 năm với drawdown −49%.
- **Slot KHÔNG đói mà NGHẸT**: avg 21.2/25 vị thế mở, idle cash chỉ 10.3% — sổ luôn đầy vì hold dài (546d). Hệ quả: **410/828 lệnh (50%) không có tiền để vào**. Pnl bị bỏ lỡ = **106.7u (55% tổng pnl composite 194.7u)**, gồm 31 mega trong đó CHÍNH các runner hậu-2020 mà bào chữa viện dẫn: **VTP 2023 (+369%), FRT 2023 (+270%), BSR 2025 (+208%), VIC 2025 (+193%)** + DGC/BCG/VND/FRT 2020 (+560..810%). Composite đếm unit-weight mọi lệnh → conf_mult/pnl-sum ưu ái sai một cách hệ thống cho model hold-dài: nửa alpha trên giấy không hiện thực hóa được.
- Không K nào cứu được: K50 vẫn ×6.26/−41%; K15 ×6.40/−49%. Cấu trúc lỗi là 2022 NAV −30..−35% (sổ đầy vị thế 2021-22 bị stop −10..−24% hàng loạt + MTM runner giveback) — không phải tham số khung.

## Tội 3 — Concentration/lottery (pr2_03_concentration.py): **THÀNH LẬP (concentration), GỠ (seed-fragility)**
- Top-10 lệnh = **34-35% pnl** mọi seed (gb 13%, champ 12%). Lệnh >100% (mega, 56-60 lệnh) = **85-89% tổng pnl**.
- Bỏ top-5 lệnh: comp còn 596-620 → **thua gb full ở CẢ 5 seed (−112..−137)**; so công bằng cùng bỏ top-5: pm2 620.5 vs gb 687.5 (s42) = −67. Bỏ top-10: còn 483-501.
- Seed-fragility: GỠ — mega set ổn định 50/67 chung cả 5 seed; LPB-2022, VTP-2023, BSR-2025 xuất hiện ở đủ 5 seed. Đây là lottery-by-design (runner-riding) chứ không phải lottery-by-seed. Nhưng kết hợp Tội 2: các vé số này với vốn thật **một nửa không mua được**.

## Tội 4 — Stop −10% robustness + gap-xuyên-stop: **GỠ**
- Biên quanh −10% (pr2_ clone từ t1058, seed 42, KHÔNG đụng canonical; t2822-2824): −0.08 → **798.6**, −0.09 → 790.8, −0.10 → 775.1, −0.11 → 767.4 (−0.12 → 751.9, −0.15 → 719.4 từ lưới pm2). Đường cong **đơn điệu, phẳng cục bộ** — không đỉnh nhọn, không overfit 1 tham số. (Trớ trêu: −10% còn không phải điểm tốt nhất; −8% hơn +23.5.)
- Gap-xuyên-stop ĐÃ nằm trong backtest (fill = close bar kế tiếp, engine.py:2394-2397, không giả định fill tại mức stop): n_hardstop 376-383/seed, median fill −10.5%, p25 −13.4%, p05 −19%, 66-74 lệnh <−15%, 10-15 lệnh <−20%, worst −24..−25%; tổng lỗ vượt mức −10% ≈ **−8.5u/seed đã tính vào kết quả**.
- T+ audit: chỉ 24 lệnh stop hold≤2 thời T+3 (trước 2022-08-29) là không thể bán đúng ngày backtest; bán trễ 1 phiên → chênh **+0.19u** tổng (không đáng kể; vài lệnh ăn thêm sàn −7% nhưng bù bằng lệnh hồi). Serving thật không tệ hơn backtest có ý nghĩa.

## Tội 5 — Era-drift / leak / static-model: **GỠ**
- Walk-forward CHUẨN, retrain từng fold: experiment.py:3154 loop `splitter.split` → `train_fold()` fit model MỚI mỗi fold (_train_loop.py:49-53). split walk_forward_year: train 2 năm, `train_end = test_start − 85d` (splitter.py:121) — không overlap; gap 85d >> horizon h10 (~14 ngày lịch); label `pct_change(fw).shift(-fw)` đúng chiều (target.py:141,156); features backward-only. Seed vào LGBM random_state, retrain thật → 5-seed spread là robustness thật.
- Trades 2026: fold cuối (train 2023-01-01→2024-10-08) được extend test đến data_max 2026-06-16 theo thiết kế (experiment.py:3115-3116) — OOS thật, model cuối "già" 20 tháng so với lệnh cuối. Không phải leak; áp dụng đồng đẳng cho champ/gb.
- Lưu ý chung (không phải tội riêng bị cáo): survivorship bias universe (mã còn sống đến 2026) áp cho mọi ứng viên cùng leaderboard.

## Tội 6 — Khả năng serving: **THÀNH LẬP MỘT PHẦN (blocker vận hành)**
- ĐÃ CÓ: cả 4 thành phần (`continuation_entry_regression`, `reward_risk_regression`, `entry_lvup126_lean`, `exit_vol_market`) đều trong wheel stock_ml_core (stock_ml/src/targets/continuation_entry.py, path_quality.py, features/catalog.py) — không blocker packaging. Pullback 3%/25 chạy chung code path với wavestruct 4.5%/40 (serving/trades.py:248-251). signal_threshold decoupled exit đang chạy production.
- THIẾU/MÂU THUẪN: (a) `hard_stop_pct` + `exit_priority ["hard_stop","signal"]` có trong EngineConfig nhưng **chưa từng chạy live** (bundle wavestruct để hard_stop=null, priority ["trailing_stop","overext","signal"]) — cần validate; (b) **hold 546-592 ngày × hạ tầng 1-slot = zombie position khóa account 1.5 năm** — mâu thuẫn trực diện; mô hình này chỉ có nghĩa ở kiến trúc ~20-25 slot (mà theo Tội 2, chính ở khung đó nó thua gb ×2.6 lần); (c) mdd_sym 0.52 nghĩa là account 1-slot có thể chìm >50% trên một vị thế; (d) `entry_bar_fill_type close_next` cần verify serving support.

## PHÁN QUYẾT: **REJECT**
Tội 2 là án tử, tội 1+3 đồng phạm, tội 6 chôn cất:
1. **Khung NAV vốn thật đảo ngược hoàn toàn composite** — ×5.61 vs gb ×14.44 (−61% tài sản cuối), MaxDD −50.3% vs −15.3%, 639 ngày dưới đỉnh, 2022 −33.9% vs gb +31.9%; vào vốn từ 2022 chỉ còn CAGR 5.5%. Không K slot nào cứu (K15/25/50 đều ×5.5-6.4). Đây chính xác là tiền lệ pyramid (+35% composite, chết NAV) lặp lại.
2. Lợi thế composite +27 là **tiền giấy hai lớp**: (a) 100% từ cohort 2020 (subframe ≥2022 thua −274.7 comp, mọi năm ≠2020 thua gb); (b) 55% pnl (106.7u) không hiện thực hóa được với vốn thật vì sổ nghẹt — kể cả các runner hậu-2020 dùng làm bằng chứng bào chữa (VTP/FRT/BSR/VIC đều bị MISS trong NAV frame).
3. Model không gian lận: walk-forward sạch, stop không overfit, tail đã được backtest định giá trung thực, mega ổn định qua seed. Nó là một mô hình **trung thực nhưng kém hơn gb_x08 trong mọi khung có ràng buộc vốn** — chỉ thắng ở khung composite unit-weight vốn vô hạn.
- Giá trị còn lại (không phải promote): họ zX-exit ôm runner (LPB +481%, VTP +369%, BSR +208% sống thật qua seed) là NGUYÊN LIỆU cho sleeve runner-riding vốn nhỏ/margin tách riêng (kiểu DL_MG đã bị reject trước đây vì lý do khác) — nếu muốn khai thác phải giải bài toán "sổ nghẹt + 2022 −34%" trước, không phải bài toán composite.

## So sánh tổng hợp cuối (cho hội đồng)
| khung | pm2_hs10_zx25 | gb_x08 | ai thắng |
|---|---|---|---|
| composite mean-5 (unit-weight) | **761.0** | 733.8 | pm2 +27 |
| composite ≥2022 (s42) | 93.8 | **368.5** | gb −275 |
| NAV K25 final / CAGR | ×5.61 / 30.3% | **×14.44 / 50.7%** | gb ×2.6 |
| NAV MaxDD / underwater | −50.3% / 639d | **−15.3% / 120d** | gb |
| NAV từ 2022 | ×1.27 (5.5%/y) | **×3.34 (30.7%/y)** | gb |
| pnl hiện thực hóa được K25 | ~45% số lệnh | ~74% số lệnh | gb |

## Files
- Scripts: pr2_00_dump_cfg.py, pr2_02_stopgrid.py (+log; clones t2822-2824 `pr2_hs08/09/11_zx25`), pr2_03_concentration.py, pr2_04_navsim.py (+ pr2_nav_*.csv)
- Dữ liệu đối chiếu: pm2_pm2_hs10_zx25_s{42,7,99,555,123}_trades.csv, signalq/nicheloss/gb_x08_s42_trades.csv, signalq/st_champ2646_s42_trades.csv
- Điều tra phụ: era-drift (experiment.py:3115-3116,3154; splitter.py:121; target.py:141,156), serving (pyproject.stock_ml_core.toml; stock-serving bundle wavestruct config; serving/trades.py:248)
