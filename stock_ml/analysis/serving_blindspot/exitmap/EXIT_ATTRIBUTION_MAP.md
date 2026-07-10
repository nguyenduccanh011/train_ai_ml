# BẢN ĐỒ ATTRIBUTION EXIT + HIỆU SUẤT MFE — gb_x08 (template 2783, seed 42)

Ngày: 2026-07-10. Run: `template/gb_x08-32a8dfee` (comp 735.0, pnl 128.497u, 1378 trades, DB run_trades khớp 100%).
Scripts + dữ liệu: `exitmap/em01_dump.py … em07_condarm.py`, `gbx08_s42_trades.csv`, `gbx08_enriched2.csv`.

## 0. Phương pháp & xác minh (không đoán)

- `run_trades.exit_reason` gần vô dụng: 1374/1378 = `signal` (3 trailing_stop, 1 open) — vì các force-gate
  (`exit_force_gate=downleg12`, `exit_force_gate_nonbull=belowma20p2`, `exit_force_gate_lowbreadth=downleg6@breadth<0.25`)
  được trộn vào exit-signal series UPSTREAM (experiment.py:2358-2399), engine chỉ thấy `sig<0` → ghi "signal".
- Attribution tái lập từ OHLCV (market.duckdb, universe_id=8 v2, 61 mã) tại **decision bar = bar trước exit_date**
  (fill close_next, engine.py:2361-2367). Công thức force-gate replicate đúng experiment.py: zigzag causal 12%/6%
  (`_causal_leg_dir`), belowMA20 persist-2 × nonbull (EW proxy < MA35 persist-2), breadth full-universe pct_above_ma50 < 0.25.
- Label theo hierarchy: downleg12 → nonbull_bma20p2 → lowbreadth_dl6 → head_signal (residual). Caveat: head có thể
  fire trùng bar với force gate (không quan sát được nếu không có model scores) → n của head_signal là **lower bound**.
- **Validation cứng**: tái lập `market_drop_dates` đúng công thức engine (zscore w5 lb60 thr −1.75, engine.py:2654) →
  **0/1374** signal-exit có decision bar rơi vào ngày drop (engine suppress đúng như thiết kế). Alignment decision-bar
  và chuỗi market của bản đồ này là chính xác cấu trúc.
- Đơn vị "u" = tổng pnl_pct per-trade (như hồ sơ SNR_EXTEND_AUTOPSY). MFE = max(high)/entry_fill − 1 trong đời lệnh;
  efficiency = realized/MFE; post-exit đo trên 20 bar sau fill.

## 1. Attribution theo rule

| rule (đóng lệnh) | n | pnl (u) | pnl/lệnh | WR | hold_med | eff_med | P(rally≥5% sau exit) |
|---|---|---|---|---|---|---|---|
| force_downleg12 | 648 | +73.8 | +0.114 | 53.7% | 17 | 0.13 | 0.57 |
| force_nonbull_bma20p2 | 471 | +32.5 | +0.069 | 56.3% | 11 | 0.17 | 0.40 |
| head_signal (velocity head) | 166 | +15.1 | +0.091 | 71.1% | 22.5 | 0.44 | 0.36 |
| force_lowbreadth_dl6 | 89 | +6.8 | +0.076 | 57.3% | 7 | 0.13 | 0.51 |
| trailing_stop | 3 | +0.3 | — | 100% | 99 | — | — |
| open | 1 | +0.0 | — | — | 40 | — | — |

Overlap flags tại decision bar (báo trung thực): 126 lệnh cả 3 flag, 127 dl12+nonbull, 137 nonbull+lowbreadth,
386 chỉ dl12, 334 chỉ nonbull, 166 không flag nào (= head organic).

Theo năm (điểm đáng chú ý):
- **Engine chính là downleg12** ở mọi năm (2021: 166 lệnh +39.3u; 2025: 84 lệnh +12.5u). Trailing stack
  (activate 27%/stop 8%/donch80/skip-above-MA10) **gần như không bao giờ là closer** — 3 lệnh/6 năm: signal+force
  luôn bắn trước.
- **Velocity head tắt dần**: 2020: 55 lệnh (WR 85%), 2023: 50, 2024: 13 (WR **8%**, −0.36u — rule duy nhất "bắn toàn
  lệnh xấu", n nhỏ), 2025: 1, 2026: 0. Từ 2024 hệ exit thực tế là **pure price-rule** (zigzag + MA), head chỉ còn
  đóng vai trò qua các veto/protect.
- head_signal là rule chất lượng cao nhất khi bắn (WR 71%, eff_med 0.44 gấp 3 lần force rules) nhưng hiếm.

## 2. MFE map — 3 con số then chốt

1. **Tổng giveback = 158.5u** (MFE 287.0u, realized 128.5u) → **efficiency danh mục 0.448** (winners-only 0.598).
2. **Slice giveback lớn nhất: downleg12 = 99.9u (63%)**, trong đó MFE≥27% chiếm 54.2u (bucket >50%: 34.6u);
   theo năm: 2021 32.8u, 2022 21.3u. Nhưng xem §5 — slice này đã được chứng minh là **structural, không thu hoạch được**
   bằng bất kỳ knob siết nào hiện có.
3. Efficiency theo hold: lệnh >40 bar eff 0.625 (pnl +116.1u/357 lệnh — toàn bộ lãi của hệ nằm ở đây); hold ≤10 bar
   eff ÂM (639 lệnh, pnl −10.7u, giveback 40.4u) — churn fast-fail là entry-problem, không phải exit-problem
   (cf giữ thêm 20 bar chỉ +2.6u toàn kỳ, ≥2022 ≈ −0.4u).

Harvest của snr_extend + giveback-guard (từ autopsy 2730 + so pnl runs): defer group +2.5u − knock-on 1.6u = **+0.9u net**,
guard 0.08 thêm **+0.32u** (128.177→128.497). Cohort defer-eligible (peak≥27%, từng chạm điều kiện defer) = 212 lệnh,
pnl 98.8u, eff 0.6–0.7 — rule đang bảo vệ đúng phần lãi lõi; phần "còn nguyên" của MFE map KHÔNG nằm ở slice này.

## 3. Cohort "bán xong thì chạy tiếp" (sold-then-rallied)

| ngưỡng rally 20-bar | n | % lệnh | cf giữ thêm 20 bar | cf oracle-đỉnh |
|---|---|---|---|---|
| ≥3% | 857 | 62% | +55.1u | +109.2u |
| **≥5%** | **663** | **48%** | **+60.0u** | **+101.1u** |
| ≥10% | 345 | 25% | +55.7u | +76.4u |

- Theo rule: downleg12 36.2u (bán ở ĐÁY pullback 12% rồi giá bật — giveback_d_med 15.7%), nonbull 17.1u.
- **KHÔNG có ranh giới sạch** (khác vụ giveback≤10%): quét gain_d/giveback_d/peak_d/SNR/hold quintiles →
  P(rallied) chỉ dao động 0.38–0.62; flag mạnh nhất là "exit 1 bar sau khi market-drop gate nhả" P=0.56–0.67.
- Mọi boundary ứng viên FAIL regime test ≥2022 (§5): phần lớn của +60u là hàng 2020–21 + phần không tách được ex-ante.

## 4. Cohort "bán muộn" (đối xứng)

| định nghĩa | n | realized | late_giveback |
|---|---|---|---|
| exit ≥3 bar sau đỉnh-close & nhả ≥5% | 780 | +114.5u | 108.7u |
| **exit ≥5 bar sau đỉnh & nhả ≥10%** | **445** | **+95.9u** | **80.1u** |
| exit ≥10 bar sau đỉnh & nhả ≥10% | 275 | +52.9u | 48.1u |

- 63.8u/80.1u nằm ở downleg12 (mfe_med 29%, hold_med 44) — đúng bản chất rule: reversal-confirm PHẢI trả ~12%+ từ đỉnh
  mới bắn. Đây là **giá vé structural** của cơ chế đã tạo ra +73.8u.
- Hai mặt trade-off: trái (bán sớm) ≈ 60u, phải (bán muộn) ≈ 80u — nhìn gross tưởng nhiều, nhưng §5 cho thấy
  **dư địa ròng khai thác được ở regime sống ≈ 0–1u** với mọi knob hiện có.

## 5. Xếp hạng lever — kết quả ÂM ghi trung thực (không đốt probe nào, 0/4)

Counterfactual first-order trên OHLCV (không knock-on/slot; số ≥2022 là tiêu chí quyết):

| lever (knob CÓ SẴN) | delta toàn kỳ | delta ≥2022 | phán quyết |
|---|---|---|---|
| downleg_skip_bull (thả downleg12 trong bull MA50p3, cf giữ 20 bar) | +4.1u | **−4.0u** (2021 +8.7) | GHOST 2021 — reject |
| defer exit 1–3 bar sau washout-release (exit_market_drop*) | +17.7u @1bar | **−5.3u** (2021 +16.9) | GHOST 2021 — reject |
| siết trail uniform 8% arm 15% (trailing_activate↓/donch↓) | **−22.8u** | âm | reject |
| vol_spike arm z≥2 trail 4% (vol_spike_z_threshold) | −13.5u | −5.5u | âm cả 7 năm — reject |
| dist_arm ad_balance≤−2 trail 5% (dist_arm_neg_thresh) | −24.3u | −11.2u | âm cả 7 năm — reject |
| trend_break_lock MA50 gain≥10% (trend_break_lock_gain) | −26.1u | −8.2u | âm cả 7 năm — reject |
| incubate fast-fail (signal_exit_min_age 5–10) | +2.6u | −0.4u | ~0 — reject |
| exit_force_suppress cho nonbull trong uptrend | +0.4u | ~0 | ~0 — reject |

**Kết luận bản đồ**: sau gb_x08, mặt exit-timing đã BÃO HÒA với thông tin price/volume. Mọi biến thể "bán sớm hơn"
âm ở cả 7 năm (đỉnh không dự đoán được bằng OHLCV-transform); mọi biến thể "giữ lâu hơn" chỉ dương nhờ sóng 2020–21
(đúng failure mode dsb60/sx_w60). Không lever config-only nào ước lượng ≥ +3u ở lát ≥2022 → **không chạy probe**
(tiết kiệm 4 lượt; probe chỉ có thể xác nhận số âm/ghost đã đo được deterministic).

Lever còn giá trị (chỉ THIẾT KẾ, cần code/kiến trúc — ngoài scope):
1. **Multi-position/slot**: bản đồ xác nhận chi phí ẩn lớn nhất của mọi defer-rule là knock-on chiếm slot
   (−1.6u/+2.5u ở snr_extend; mọi cf "giữ thêm" đều bị trừ knock-on chưa tính). 1-slot là trần cấu trúc —
   khớp hướng đã chốt trong hồ sơ wavestart.
2. **Head exit mới cho 2024+**: velocity head gần như im lặng từ 2024 (closer 13→1→0 lệnh); nếu muốn thêm alpha exit
   thì phải là SIGNAL MỚI (không phải transform giá/vol — họ đó đã âm toàn dải), ví dụ retrain velocity head trên
   dữ liệu ≥2022 hoặc feature ngoài OHLCV.
3. Knob mới `exit_snr_defer_max_gb` (trần giveback TRONG lúc hoãn) đã bị chứng minh âm toàn dải từ autopsy 2730 — không nhắc lại.

## Phụ lục: giveback theo rule × năm (u)

| rule \ năm exit | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | ALL |
|---|---|---|---|---|---|---|---|---|
| downleg12 | 5.4 | 32.8 | 21.3 | 12.0 | 6.5 | 11.9 | 9.9 | 99.9 |
| nonbull_bma20p2 | 1.9 | 5.5 | 5.9 | 5.6 | 8.3 | 8.7 | 3.1 | 38.9 |
| head_signal | 4.5 | 0.5 | 2.2 | 3.0 | 1.0 | 0.1 | — | 11.4 |
| lowbreadth_dl6 | 0.0 | 0.1 | 2.8 | 0.8 | 1.3 | 1.2 | 1.5 | 7.6 |
