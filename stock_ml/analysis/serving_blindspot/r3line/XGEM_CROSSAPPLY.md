# XGEM CROSS-APPLY — max_hold lên các gem khác: phổ quát hay đặc sản t2429?

Ngày: 2026-07-10. Scripts: `xg_00_cfg.py` (+`xg_00_out.txt`), `xg_10_run.py` (+`xg_10_out.txt`, `xg_10_dyn_out.txt`), `xg_20_deep.py` (+`xg_20_*_out.txt`). Trades: `xg_{dyn,v19}_mh{12,16,25}_s42_trades.csv`. Clone DB: xg_dyn_mh12/16/25 = t2927/2928/2929, xg_v19_mh12/16/25 = t2931/2932/2934 (KHÔNG đụng canonical, không commit).

Thước: `nh_nav2` NGUYÊN BẢN K25 R0.6 settle T+2, shuffle-mean±sd 20 perm, hai chế độ (adv 0.08% / no-adv). Chuẩn so: **t2429+mh16 (r3_mh16 s42): full ×21.77±0.53 adv / ×19.62 noadv, f22 4.27/4.10, f23 2.83, LOYO-2021 ×9.64±0.25, DDw −13.8%** (verify lại cùng máy cùng ngày, khớp R3_ROUND1). Ngưỡng "đáng đào sâu" = mh16 − 5% = ×20.68 adv full.

## 1. Hai base là gì (từ config DB, `xg_00_out.txt`)

- **dyncsr88 = t2531 `n2_2429_dyncsr88`**: chính là **t2429 + head thứ 5 "PURE-dynamics csrank"** (entry_ensemble3: features `entry_dyn_pure`, norm cross-sectional rank, pct 0.88 — velocity/expansion của MA/RSI/%R/MACD-hist xếp hạng chéo toàn universe). Mọi thứ khác giống 2429 (4-head ensemble, pullback conv combo, nonbull gate). Label horizon: heads h10/h10/h10/h20, entry slot triple_barrier h30, exit slot velocity_exit h20 → trần hợp lý 12-25 nằm giữa dải horizon. `exit_priority` gốc = [trailing_stop, overext, signal], max_hold_bars 10000 → prepend "max_hold" y khuôn t2429.
- **v19_fullwave = t1799 `n2_v19_fullwave`**: champion đời 1798 + **trend-intact trail-hold (ma10) + entry_threshold −1.9** — thiết kế "bắt trọn sóng" được thưởng dưới Sortino composite. KHÔNG có ensemble heads (đời trước multi-head); sig 5.5, pullback window 50, trailing_activate 0.15 (thấp — trail bật sớm), exit head reward_risk_regression h10, entry triple_barrier h30, fs entry_lvup126_lean. Cùng exit_priority/max_hold 10000 → prepend giống hệt.

## 2. Bảng 6 ô + uplift từng stack (adv full / noadv full / f22 adv / f23 adv; base từ scan + chấm lại noadv từ CSV scan)

| stack | base | mh12 | mh16 | mh25 | uplift adv full (best) |
|---|---|---|---|---|---|
| **dyncsr88 (t2531)** | ×17.42 / 16.57 / 3.76 / 2.50 | ×20.91 / 17.67 / 4.29 / 2.92 | ×20.80 / 19.35 / 4.24 / 2.87 | **×21.15 / 19.39 / 4.45 / 2.95** | **+19..+21%** |
| **v19_fullwave (t1799)** | ×16.29 / 14.23 / 3.40 / 2.60 | ×17.66 / 16.11 / 3.58 / 2.82 | ×17.81 / 16.61 / 3.71 / 2.85 | ×17.90 / 16.35 / 3.82 / 2.83 | **+8..+10%** |

Tham chiếu các stack đã đo trước (R3_ROUND1): t2429 base ×17.90 → mh12/16/25 ×22.39/21.77/21.67 (**+21..+25%**); gb_x08 (t2783) ×13.78 → mh16 ×20.16 (**+46%**); 2643 wavestruct + mh20 ×18.54.

Ghi chú kỹ thuật (bẫy env): 3 run dyn fail lần đầu — t2531 có `exit_force_gate_lowbreadth` → `_load_market_breadth`/`_load_xsec_features` trong `stock_ml/src/pipeline/experiment.py` (dòng 1578/1594/1625) hard-code path TƯƠNG ĐỐI `market_data/market.duckdb`, bỏ qua STOCK_DATA_DIR → **phải chạy runner với cwd = repo root** cho template nào có breadth/xsec gate. (v19/t2429 không dính vì không có gate này.)

## 3. Điểm mạnh nhất vs t2429+mh16: dyn_mh25 (t2929)

| frame | dyn_mh25 | t2429+mh16 | Δ |
|---|---|---|---|
| full adv | ×21.15±0.79 | ×21.77±0.53 | −2.8% (trong ngưỡng −5%) |
| full noadv | ×19.39±0.87 | ×19.62±0.60 | −1.2% |
| f22 adv / noadv | **×4.45±0.09 / ×4.57±0.06** | ×4.27 / ×4.10 | **+4.2% / +11.5%** |
| f23 adv | **×2.95±0.05** | ×2.83±0.04 | **+4.2%** |
| LOYO-2021 adv | ×8.90±0.21 | ×9.64±0.25 | −7.7% |
| DDw | −13.7% | −13.8% | ≈ |

- Per-year (s42): 2020 31.3 / 2021 29.9 / 2022 13.4 / 2023 12.9 / 2024 7.4 / 2025 18.6 / 2026 0.9 — dương 7/7, không ghost; pf 3.98, 2221 trades, hold median 10.
- **Overlap trades vs t2429+mh16: exact 80.1%, fuzzy±5d 86.0%** (dyn_mh16 tới 96.6% — gần như CÙNG một hệ lệnh). → **KHÔNG phải ứng viên diversify**; dyncsr88 vốn là t2429+1 head, base đã overlap 86.8%. Giá trị của dyn_mh25 là **biến thể recent-tilted cùng gia tộc**: nhường ~3% ở full (nặng 2020-21) đổi lấy +4..+11% ở f22/f23 — đáng đem vào công tố R3 như một điểm trên cùng plateau họ t2429+mh, không phải gem mới.
- v19+mh: overlap thấp nhất (53-58% exact, 72-76% fuzzy) — diversify khá hơn về cohort nhưng NAV ×17.90 = −17.8% vs chuẩn, dưới ngưỡng −5% → không đạt tiêu chí "mạnh mà khác lệnh". Không mở tuyến.

## 4. Verdict phổ quát vs đặc sản

**Max_hold là cơ chế PHỔ QUÁT — dương trên cả 4/4 stack đã thử — nhưng độ lớn uplift phụ thuộc "độ hở đuôi hold" của stack:**

| stack | hold gốc (median/p95) | uplift adv full | vì sao |
|---|---|---|---|
| gb_x08 (2783) | dài, snr-extend | **+46%** | winner đuôi dài nhất, cap giải phóng vốn nhiều nhất |
| t2429 base | 9 / ~121 | **+21..25%** | đuôi hold dài không kiểm soát |
| dyncsr88 (2531) | 9 / tương tự 2429 | **+19..21%** | = t2429 + 1 head; head dynamics không đổi cấu trúc hold |
| v19_fullwave (1799) | 9 / trail-hold ma10 | **+8..10%** | thiết kế "fullwave" ĐÃ chủ động ride sóng có kiểm soát (trail bật sớm 0.15, exit head h10) → ít đuôi rác cho cap chém |

- Quy luật: uplift tỉ lệ thuận với phần NAV bị chôn ở hold-đuôi-dài; stack nào đã có cơ chế quản lý hold (trail-hold, activate sớm) thì cap chỉ cộng ~9-10%. Đúng giả thuyết "trần-giữ-lệnh là cơ chế phổ quát cho hệ label chân trời ngắn" (mọi label ở đây h10-h30) — không phải đặc sản t2429; nhưng **t2429-family + mh vẫn là combo mạnh nhất** vì selection tốt nhất (pf ~3.9-4.0) + đuôi hở nhiều nhất.
- **Không có gem-tổ-hợp nào VƯỢT t2429+mh16 ở full-frame** trong 6 ô này; dyn_mh25 ngang (−2.8%) và thắng ở f22/f23 → đưa vào rổ công tố R3 (cùng selection-inflation ~70+6 config phải trừ); v19 đóng ở NAV-frame.
- Kết quả âm trung thực: cả 6 run đều seed-42 đơn (chưa multi-seed); dyn_mh16 overlap 96.6% cho thấy head dyncsr88 gần như trung tính dưới cap — nhánh "thêm head dynamics" không đáng đầu tư thêm.
