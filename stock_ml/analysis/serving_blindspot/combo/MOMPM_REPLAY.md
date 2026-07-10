# MOMPM REPLAY — CỔNG DECISION-LAYER CHO mom_pm TRÊN TRADES gb_x08: **KILL**

Ngày: 2026-07-10. Scripts + output: `combo/mp_01_replay.py` (event study + 4 luật chiều đề bài),
`mp_02_reverse.py` (4 luật chiều ngược — đúng chiều screen P3), `mp_03_sens.py` (grid độ nhạy k×thr),
`mp_trades_qdec.csv` (1378 trades + quantile mom_pm tại decision bar). Tiếp nối
`pureml/P3_DERIV_SCREEN_V2.md` (mom_pm: IC trực giao +0.096..0.133 lát 2024+, tín hiệu sống duy nhất).
OFFLINE — không đụng engine, không leaderboard.

## 0. Phương pháp

- `mom_pm` (momentum buổi chiều VN30F1M, chốt ATC 14:45, causal) → **quantile trượt q ∈ [0,1]**:
  rank của giá trị ngày t trong 252 obs TRƯỚC đó (exclude t, min 120) — không lookahead, không threshold tĩnh.
- Trades gb_x08 s42 (1378): decision bar = bar trước exit_date (per-symbol). Match q theo NGÀY CHÍNH XÁC
  (lỗ intraday 2023 → NaN → luật không kích hoạt). Coverage q_dec: **1330/1378** (≥2022: 994/1042).
- Δpnl per trade = `(1+pnl)·c[exit_mới]/c[exit_cũ] − 1 − pnl` (tỷ lệ close, tự nhất quán slippage);
  riêng R3 dùng `cf_pnl` có sẵn của `exitmap/pdr_forensic_scan.csv`. First-order, KHÔNG mô phỏng knock-on
  slot (mọi Δ dương vì thế là CẬN TRÊN). Mega-check: các lệnh pnl ≥ +40% (gồm GEX/VCI/VIC/LPB top).

## 1. Event study — tín hiệu NGHIÊNG THẬT, đúng chiều screen

Bucket q_dec (quantile mom_pm tại decision bar), lát ≥2022 (n=994):

| bucket | n | P(rallied≥5%) | post_max_c | post_end_c | post_min_c | giveback_u |
|---|---|---|---|---|---|---|
| **q≤.10 (PM kiệt)** | 81 | **0.593** | **+0.090** | +0.002 | −0.086 | 0.152 |
| .10–.30 | 292 | 0.462 | +0.067 | −0.005 | −0.073 | 0.105 |
| .30–.70 | 315 | 0.346 | +0.042 | −0.031 | −0.085 | 0.091 |
| .70–.90 | 151 | 0.470 | +0.062 | −0.007 | −0.078 | 0.106 |
| **q>.90 (PM mạnh)** | 155 | 0.381 | +0.053 | −0.019 | −0.097 | 0.101 |

2024+ giữ nguyên dạng (q≤.10: rallied 0.516 vs .30–.70: 0.322). Cohort đau xác nhận:
sold-then-rallied q_dec mean 0.462 / P(q≤.1)=0.114 vs sold-đúng 0.510 / 0.058. Tín hiệu tại dec−1/dec−2
đã NHẠT hẳn (monotonicity biến mất) → thông tin tập trung đúng bar quyết định — khớp screen, không phải artifact.
Suppress-victims (mkt_drop, 377 lệnh có q): **delta suppression DƯƠNG ở mọi bucket mom_pm** (kể cả q≤.10:
+2.31u ≥2022) — suppress đang cứu đúng cả khi PM kiệt, không có lát nào để "un-suppress có điều kiện" cứu thêm.

**Nhưng — chìa khóa của cái chết:** forward return k-bar sau exit theo bucket (≥2022):

| bucket | fwd1 | fwd2 | fwd3 | fwd5 | post_max_c (20 bar) |
|---|---|---|---|---|---|
| q≤.1 | **−0.25%** | **−0.39%** | **−0.31%** | **−0.23%** | **+8.95%** |
| .1–.2 | +0.12% | +0.94% | +1.03% | +0.01% | +7.88% |
| .3–.7 | −0.83% | −1.12% | −1.23% | −1.99% | +4.16% |

Bucket kiệt nhất RALLY THẬT trong 20 bar (+9%) nhưng 1–5 bar đầu vẫn ÂM — cú hồi đến SAU horizon mà một
luật defer 1–3 bar với được, và đến sau khi rơi thêm (post_min −8.6%). mom_pm dự báo đúng cái nó được screen
để dự báo (giveback/rally 10–20 bar) — nhưng "giữ thêm 20 bar" đã bị chứng minh ghost-2021 từ EXIT_ATTRIBUTION §5.

## 2. Replay 4 luật × 2 chiều × ngưỡng — bảng Δu (first-order, cận trên)

Chiều đề bài (mp_01) — TẤT CẢ ÂM:

| luật | ngưỡng | n chạm | Δu ≥2022 | Δu 2024+ | mega d<−2% |
|---|---|---|---|---|---|
| R1 defer 1 bar khi PM MẠNH | q≥.8 / q≥.9 | 284 / 200 | −1.36 / −1.15 | −0.98 / −0.88 | 10 / 7 lệnh |
| R1 defer 2 bar khi PM MẠNH | q≥.8 / q≥.9 | 284 / 200 | −3.04 / −2.56 | −1.55 / −1.52 | 10 / 10 |
| R2 trail8/arm15 khi PM KIỆT | q≤.05/.1/.2 | 207–289 | −2.77 / −3.52 / −5.87 | −1.03 / −1.61 / −3.81 | 45–54 lệnh, giết VCI/VPB/HPG/LPB/VIC |
| R3 un-suppress mkt_drop khi PM KIỆT | q≤.1/.2/.3 | 180–263 | −2.31 / −2.94 / −2.48 | −1.88 / −2.41 / −2.53 | 13–33 |
| R4 un-defer snr khi PM yếu | q≤.2/.5 | 74 / 119 | −6.12 / −6.80 | −6.05 / −6.80 | 25 / 44 (GEX −1.14, VIC −1.05) |

Chiều ngược — đúng chiều screen P3 (mp_02) — cũng âm/hòa:

| luật | ngưỡng | n | Δu ≥2022 | Δu 2024+ |
|---|---|---|---|---|
| R1b defer 1–2 bar khi PM KIỆT q≤.1 | k=1/k=2 | 128 | −0.24 / −0.40 | −0.28 / −0.04 |
| R1b defer 1–2 bar q≤.2 | k=1/k=2 | 269 | −0.13 / **+0.80** | −0.33 / +0.67 |
| R2b trail8 khi PM MẠNH | q≥.8/.9/.95 | 114–220 | −3.79 / −2.17 / −0.54 | −3.39 / −1.79 / −1.18 |
| R3b un-suppress khi PM MẠNH | q≥.7/.8/.9 | 8–40 | −0.19 / −0.41 / +0.01 | +0.16 / −0.08 / +0.01 |
| R4b un-defer snr khi PM MẠNH | q≥.7/.8/.9 | 21–38 | −1.37 / −1.64 / −1.01 | −1.52 / −1.58 / −1.01 |

Grid độ nhạy luật sống sót duy nhất R1b (mp_03, thr .1/.2/.3 × k 1/2/3/5, ALL vs force-only) — ô TỐT NHẤT:
force-only q≤.2 k=3 → **+1.02u ≥2022 / +0.66u 2024+** (ALL: +0.86/+0.66). Nhưng:
(i) **< nửa chuẩn +2u**, và là cận trên chưa trừ knock-on (snr_extend từng mất 1.6u/2.5u vì knock-on);
(ii) **không đơn điệu theo ngưỡng**: q≤.1 (tín hiệu MẠNH nhất) âm ở mọi k, q≤.3 âm — chỉ dải .1–.2 dương,
đúng ô fwd2/fwd3 dương lẻ loi của bảng forward — pattern "một ô sáng giữa lưới tối" = noise khai thác ngược;
(iii) không đơn điệu theo k: k=5 lật âm (−0.72/−0.98);
(iv) vẫn cào mega: 5–9 lệnh mega Δ<−2% mỗi ô (LPB 2023 −0.10, VTP −0.24, GEX 2025 −0.07).

## 3. VERDICT: **KILL kênh mom_pm ở decision layer** — đúng kịch bản pv_corr

1. **Không luật nào đạt chuẩn GO** (Δu ≥ +2 ở ≥2022, không âm 2024+, mega nguyên vẹn): 8 dạng luật ×
   2 chiều × 2–4 ngưỡng × 2 cohort = **~40 ô, max +1.02u**, ô max là ô cô lập không đơn điệu.
2. **Cơ chế chết định lượng được, không phải screen sai**: IC của mom_pm là thật (event study tái xác nhận
   rallied 59% vs 35%, riêng bar quyết định) nhưng horizon của tín hiệu (rally/giveback 10–20 bar) không khớp
   horizon của bất kỳ van exit nào (1–5 bar). Ăn được cú rally 20-bar đòi "giữ thêm 20 bar" — họ lever đã bị
   xử ghost-2021 (EXIT_ATTRIBUTION §5); còn cửa sổ 1–5 bar thì bucket kiệt nhất vẫn âm (bán đúng đáy cú xả
   NHƯNG đáy còn rơi thêm 5 bar nữa).
3. Van có sẵn cũng không cứu được: mkt_drop suppress **đã net dương ở mọi bucket mom_pm** (không có lát điều
   kiện nào cải thiện — R3 hai chiều đều âm); snr_defer đang defer ĐÚNG bất kể PM (un-defer mọi kiểu −1..−7u).
4. Khớp án lệ pv_corr: **screen-pass ≠ decision-value**. Đây là lần thứ hai một tín hiệu qua null trung thực
   chết ở replay — củng cố cổng replay như bước bắt buộc trước mọi engine work.

**Số cho sổ**: mom_pm decision-layer: 0/40 ô đạt chuẩn; best cell +1.02u ≥2022 (cận trên first-order,
cần ≥ +2); event-study tilt có thật (Δrallied +24.7 điểm % giữa q≤.1 và .3–.7, ≥2022).

Đường còn mở cho họ PM-exhaustion (ghi để không đốt lại, KHÔNG phải đề xuất làm ngay):
- Dạng dùng khác horizon: điều kiện **entry/re-entry** (REENTRY_GAP) hoặc sizing risk-off ở tầng danh mục —
  nơi horizon 10–20 bar là bản địa; hoặc chờ kiến trúc **multi-position** (defer không còn trả knock-on).
- KHÔNG thử thêm biến thể exit-timing 1–5 bar của mom_pm/mom_60/vwap_dev — cùng họ, cùng horizon, đã đủ số.

KHÔNG commit git. Không file nào ngoài `combo/mp_*` + hồ sơ này được ghi.
