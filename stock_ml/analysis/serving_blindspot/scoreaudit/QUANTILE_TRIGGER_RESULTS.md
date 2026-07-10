# QUANTILE TRIGGER — probe ngưỡng tự-hiệu-chỉnh cho exit head gb_x08 (NGÁCH ĐÓNG BẰNG SIM OFFLINE)

Ngày: 2026-07-10. Script: `qt_00_sim.py` (offline thuần: `sa_scores.parquet` + `run_trades`
của `template/gb_x08-32a8dfee`). **KHÔNG chạm engine, không tạo template, không run leaderboard**
— sim first-order âm rõ ở mọi q nên dừng trước bước implement knob (nhiệm vụ mục 3).

## 0. Thiết kế probe

Thay `sell_ml = zX > 2.0` (tĩnh) bằng `exit_score ≥ Q_q(rolling 252, min_periods 60)`,
per-symbol trailing chứa bar hiện tại — đúng convention `_causal_zscore_by_symbol`
(`experiment.py:1256`), trigger trong RAW-space (quantile cửa sổ raw, rank-safe). Gate
`cons2_w20` giữ nguyên. Sweep offline q ∈ {0.97, 0.98, 0.985, 0.99, 0.995} (rộng hơn protocol
vì sim rẻ). dpnl first-order 2 chiều trên 1,378 trades gb_x08 seed-42 (pnl_sum 128.4965):

- **A (bán sớm)**: bar quantile đầu tiên trong (entry, exit) → dpnl = (1+pnl)·C[t_new]/C[t_exit] − 1 − pnl.
- **B (kéo dài)**: exit thật là sell_ml-thuần (không force cùng bar) mà quantile im → trade
  kéo đến bar (force | quantile-sell) kế tiếp.
- Caveat: bỏ qua exit_snr defer của 2783 và occupancy re-entry — chỉ tin DẤU + độ lớn.

## 1. Sell-bar/năm — phát hiện cơ chế: "tự-hiệu-chỉnh" KHÔNG tự thích nghi kịp

Trigger PRE-GATE (bar ≥ quantile trượt) — về cơ học phải ≈ (1−q) số bar mỗi năm, thực tế:

| Năm | trig q985 | trig q990 | trig q995 | % bar (q990) | post-gate q990 / additive-sau-force |
|---|---|---|---|---|---|
| 2020 | 645 | 550 | 352 | 3.71 | 464 / 402 |
| 2021 | 398 | 352 | 258 | 2.37 | 268 / 163 |
| 2022 | 536 | 438 | 314 | 2.89 | 334 / 72 |
| 2023 | 835 | 686 | 525 | 4.52 | 435 / 204 |
| 2024 | 384 | 317 | 226 | 2.08 | 304 / 6 |
| **2025** | **11** | **7** | **5** | **0.05** | **7 / 3** |
| 2026H1 | 81 | 62 | 46 | 0.95 | 36 / 31 |

**2025 vẫn câm (7 bar pre-gate tại q99, = 1/20 mức cơ học)** — lý do cấu trúc: cửa sổ trailing
252 chứa giá trị regime-2024 (cao) suốt gần trọn năm sụp, nên threshold quantile tụt CHẬM HƠN
đà sụp của score; bar mới hầu như không bao giờ vượt top-1% của cửa sổ còn "nhớ" năm cũ.
Rolling quantile lag đúng bằng window — **cùng gót chân với z-norm 252/60** (SCORE_AUDIT §1),
chỉ khác dạng thống kê. Nó chỉ "bắt kịp" khi phân phối đã ổn định ở mức mới ≥ ~1 năm
(2026H1 lên 0.95% ≈ cơ học — nhưng lúc đó thì đã muộn 1 năm). Premise trung tâm của ý tưởng
("tự thích nghi khi phân phối co") **sai với drift đơn điệu** — nó chỉ đúng với co-rồi-đứng-yên.

## 2. Outcome bar bắn (fwd10 trên 40,884 open-bars của trades gb_x08)

winner% = % bar bán có fwd10 > mean năm (bán nhầm người thắng); avoided = Σ(mean năm − fwd10),
%-pt; decTB = decile fwd10 trung bình của bar bán (0 = tệ nhất → bán đúng).

| Năm | meanF10 | q985: n / winner% / avoided / decTB | q990 | q995 |
|---|---|---|---|---|
| 2020 | +4.77% | 313 / 19 / **+1300** / 2.7 | 274 / 19 / +1085 / 2.8 | 181 / 21 / +630 / 3.0 |
| 2021 | +2.40% | 164 / **64** / **−695** / 5.9 | 147 / 64 / −645 / 5.9 | 105 / 61 / −447 / 5.8 |
| 2022 | −0.34% | 46 / 15 / +281 / 2.3 | 40 / 15 / +244 / 2.4 | 27 / 22 / +143 / 2.6 |
| 2023 | +1.46% | 102 / 45 / +46 / 4.2 | 79 / 47 / +56 / 4.2 | 64 / 47 / +53 / 4.2 |
| 2024 | +0.79% | 47 / 55 / −50 / 5.1 | 40 / 55 / −33 / 5.1 | 29 / 52 / −21 / 5.0 |
| 2025 | +2.57% | 5 / 60 / −19 / 5.6 | 3 / 33 / +5 / 4.0 | 3 / 33 / +5 / 4.0 |
| 2026H1 | +0.24% | 26 / 38 / −18 / 4.2 | 20 / 35 / −4 / 3.9 | 15 / 27 / +11 / 3.5 |

Đúng mẫu calibration-đảo SCORE_AUDIT §3 dự báo: bán đúng ở 2020/2022 (decTB 2.3–3.0), **bán
nhầm winner áp đảo ở 2021 (winner% 61–64, avoided −447…−695) và nghiêng nhầm 2024/2025** —
quantile trượt không đổi được DẤU của thông tin, chỉ đổi vị trí ngưỡng.

## 3. dpnl first-order trên trades (con số quyết định)

Đơn vị = pnl như DB (baseline gb_x08 = 128.4965). Năm = năm exit.

| q | TỔNG dpnl | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026H1 | n trade đổi |
|---|---|---|---|---|---|---|---|---|---|
| 0.97 | **−19.00** | −1.00 | **−15.49** | +0.11 | −0.13 | −0.48 | +0.01 | −2.02 | 294 |
| 0.98 | −17.92 | −1.00 | −14.21 | −0.06 | +0.03 | −0.62 | +0.01 | −2.06 | 258 |
| 0.985 | −17.45 | −0.93 | −13.49 | −0.32 | +0.06 | −0.66 | +0.03 | −2.15 | 232 |
| 0.99 | −15.85 | −0.67 | −12.48 | −0.38 | +0.06 | −0.66 | +0.03 | −1.75 | 220 |
| 0.995 | −14.10 | −0.68 | −12.09 | −0.51 | +0.10 | −0.14 | +0.03 | −0.81 | 205 |

- **2021 là hố tử thần**: 46–54 trades bị bán sớm trung bình ~45–52 bar, dpnl −12…−15.5 —
  cắt cụt các sóng lớn nhất của năm bull đúng như bảng winner% cảnh báo. Không q nào thoát
  (đuôi phải của head 2021 = danh sách winner).
- **2025 — mục tiêu của cả ý tưởng — chỉ được +0.01…+0.03** (1–2 trade đổi): trigger tự-hiệu-chỉnh
  vẫn câm ở đó (§1). Ngách này không sửa được đúng cái nó sinh ra để sửa.
- 2026H1 âm −0.8…−2.2 (8–12 trade bán sớm ~22–29 bar). Vế B (sell_ml cũ biến mất → kéo dài)
  nhỏ và ~trung tính (2023 +0.24 là lớn nhất).
- Không tồn tại vùng q nào tổng ≥ 0; năm dương tốt nhất (+0.10) < ngưỡng protocol +2 cả một
  bậc độ lớn, trong khi vế âm lớn hơn 100×.

## 4. Verdict

**Ngách ngưỡng tự-hiệu-chỉnh CHẾT — đóng bằng sim offline, không tốn engine run nào.**
Hai tầng falsify độc lập: (1) rolling quantile 252/60 lag đúng bằng window nên 2025 vẫn câm
(7 bar pre-gate/năm, +0.01…0.03u — không giải được bài toán gốc); (2) ở các năm nó CÓ bắn,
calibration-đảo biến nó thành máy bán winner (2021: −12…−15.5u first-order, winner% 64).
Kết hợp với EXIT2_CHANNEL_RESULTS: **mọi họ ngưỡng "một-tham-số trên chuỗi score exit"
(z tĩnh, OR-channel tĩnh, quantile trượt) đã bị falsify bằng số** — thông tin exit đổi DẤU
theo năm, không đổi được bằng vị trí/dạng ngưỡng. Trần oracle +1–3u/năm (nếu còn) đòi
regime-conditioning đổi-dấu-sử-dụng, không phải threshold engineering.

Không implement `exit_z_quantile`, không template mới, không commit. Chứng cứ: `qt_00_sim.py`
+ bảng trong file này.
