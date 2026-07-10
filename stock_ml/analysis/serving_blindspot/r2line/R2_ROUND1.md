# TUYẾN R2 — VÒNG 1 (NAV-first sweep cho fc_rule2 / t2005)
Ngày: 2026-07-10.

Mục tiêu: TĂNG TRƯỞNG VƯỢT gb_x08 ở thước NAV/CAGR/MaxDD vốn thật (base đã thắng: ×14.78 vs ×14.44). Composite là phụ — xếp hạng theo NAV. Ngưỡng có nghĩa: NAV K25 > ×15.5.
Khung: clone `r2_*` từ fc_rule2 (t2835 = t2005), seed 42 (deterministic tuyệt đối — r2_base tái lập 568.3 đúng từng chữ số), leaderboard + NAV sim K25 100% NAV (`r2_nav.py` = bản sao byte-level pr2_04_navsim.py, chỉ đổi chỗ ghi output + thêm arg date_lo chống ghost).
Lưu ý env: sau refactor repo, phải override `STOCK_DATA_DIR=F:/PROJECTS/train_ai_ml/market_data/market.duckdb` (config trỏ stock_ml/market_data không còn tồn tại).
Lưu ý DB: prefix leaderboard `r2_*` đụng tên với tuyến volexp cũ (r2_01_volexp… comp 100-200) — template của tuyến này là r2_base/r2_snr*/r2_pb*/r2_dt*/r2_gate*/r2_c1*, id 2845+.

## 1. Baseline tái lập
r2_base (t2845): comp **568.3**, pnl 100.3, PF 3.45, 2012 trades — khớp canonical.
NAV K25 **×14.78 / CAGR 51.21% / MaxDD −12.25%** (2025-04-09), uw 101d, fills 1688/skip 324 — khớp FAMILY_CHAMPIONS.
K15 ×15.77/−19.34% (khớp), K20 ×**17.70**/−15.21% (điểm mới — xem §4 trục K).

## 2. Bảng sweep seed-42 (composite × NAV K25)

| variant | knob | comp | Δcomp | NAV | CAGR | MaxDD | fills |
|---|---|---|---|---|---|---|---|
| r2_base | — | 568.3 | 0 | ×14.78 | 51.2% | −12.25% | 1688 |
| **(a) snr_extend/giveback (4 key gb, rule market-level 0-ML — defer kênh signal-exit khi regime clean-trend & trade đang thắng)** |
| r2_snr_gb | thr.8/w20/gain.27/gb.08 (nguyên gb) | 568.3 | 0.0 | ×14.78 | 51.2% | −12.25% | 1688 |
| r2_snr_g12 | gain .27→.12 | 569.4 | +1.1 | ×14.95 | 51.5% | −12.25% | 1688 |
| r2_snr_g12_nogb | gain .12, gb 0 | 569.2 | +0.9 | ×14.94 | 51.5% | −12.25% | 1688 |
| r2_snr06_g12 | thr .6, gain .12 | 568.7 | +0.4 | ×14.93 | 51.5% | −12.25% | 1686 |
| **(b) pullback depth × window (base 4.5%/50)** |
| r2_pb35_30 | 3.5%/30 | 559.6 | −8.7 | ×13.97 | 49.9% | −14.45% | 1741 |
| **r2_pb35_50** | **3.5%/50** | **572.6** | **+4.3** | **×15.64** | **52.5%** | **−12.70%** | **1764** |
| r2_pb35_70 | 3.5%/70 | 556.9 | −11.4 | ×15.35 | 52.1% | −12.84% | 1774 |
| r2_pb45_30 | 4.5%/30 | 565.1 | −3.2 | ×15.24 | 51.9% | −12.94% | 1670 |
| r2_pb45_70 | 4.5%/70 | 555.8 | −12.5 | ×15.46 | 52.3% | −12.87% | 1684 |
| r2_pb55_30 | 5.5%/30 | 553.5 | −14.8 | ×13.75 | 49.5% | −12.87% | 1598 |
| r2_pb55_50 | 5.5%/50 | 549.5 | −18.8 | ×14.07 | 50.1% | −13.90% | 1607 |
| r2_pb55_70 | 5.5%/70 | 530.0 | −38.3 | ×13.94 | 49.9% | −12.81% | 1586 |
| **(c) dtstop (base −6%)** |
| r2_dt05 | −5% | 568.2 | −0.1 | ×14.77 | 51.2% | −12.25% | 1694 |
| r2_dt08 | −8% | 567.4 | −0.9 | ×14.25 | 50.4% | −12.25% | 1676 |
| **(d) entry_gate (base upleg_abovema20; gate = token-string, không có knob số)** |
| r2_gate_up | bỏ abovema20 (chỉ upleg) | 569.3 | +1.0 | ×14.85 | 51.3% | −12.25% | 1697 |
| r2_gate_ama10 | upleg_abovema10 | 565.8 | −2.5 | ×15.15 | 51.8% | −12.25% | 1687 |
| r2_gate_ama50 | upleg_abovema50 | 505.9 | −62.4 | ×11.00 | 44.5% | −12.33% | 1553 |
| **(e) vòng bồi (combo + ridge depth @ w50)** |
| r2_c1_pbsnr | pb3.5/50 + snr g12 | 572.8 | +4.5 | ×15.68 | 52.6% | −12.70% | 1764 |
| r2_pb30_50 | 3.0%/50 | 560.8 | −7.5 | ×14.49 | 50.8% | −13.36% | 1785 |
| **r2_pb40_50** | **4.0%/50** | **569.2** | **+0.9** | **×16.29** | **53.5%** | **−13.63%** | **1739** |
| **r2_c2_pb40snr** | **pb4.0/50 + snr g12** | **569.9** | **+1.6** | **×16.34** | **53.6%** | **−13.63%** | **1738** |

Ridge depth @ w50: 3.0→×14.49, 3.5→×15.64, **4.0→×16.29**, 4.5→×14.78, 5.5→×14.07 — đỉnh NAV tại **4.0%**, hai phía rơi; KHÔNG monotonic theo throughput thô (pb30 nhiều fill nhất 1785 nhưng NAV thấp — fill rác nhịp chỉnh quá nông làm bẩn cohort). Combo snr chỉ cộng +0.04 NAV trên nền pb (gần bão hòa).

Ghi chú kỹ thuật: r2_snr_gb byte-identical base — min_gain 0.27 KHÔNG BAO GIỜ bind ở hệ này (p95 winner +23.5% < 27%); phải hạ xuống thang winner của rule2 (0.12) thì mới có tác dụng (+0.17 NAV, đến từ 2022 +25.4% vs +23.9%).

## 3. Ứng viên NAV > ×15.5 + kiểm chống-ghost

### ★ r2_c2_pb40snr (ứng viên chốt vòng 1): ×16.34 / CAGR 53.55% / MaxDD −13.63% — comp 569.9
= pb 4.0%/50 + snr_extend(0.8/20/gain 0.12/giveback 0.08). So gb_x08 (×14.44/50.7%/−15.31%): **NAV +13.2%, CAGR +2.9đ, MaxDD nông hơn 1.7đ** — thắng cả 3 mặt. So base rule2: +1.56 NAV, đổi 1.4đ MaxDD.
- **Subframe ≥2022**: comp 281.7 vs base 275.5 (**+6.2**); ≥2023: 248.0 vs 249.0 (−1.0); ≥2024: 168.5 vs 166.4 (**+2.1**) — không thua bậc nào.
- **NAV-từ-2022** (chống ghost NAV): ×**3.80** vs base ×3.48 (**+9.2%**) — alpha NAV KHÔNG tựa 2020-21; yearly 2022 +31.1/2023 +44.1/2024 +25.2/2025 +48.7/2026 +8.0 — thắng base 4/5 lát.
- **Structure sanity** (từ pb40_50, c2 chỉ khác +6 trade snr-defer): per-entry-year dương cả 7 năm; exit mix cùng hình (overext 468/+65.1, trail 450/+65.4, signal 1070/−22.7, dtstop 90/−6.1); percentiles pnl gần trùng base (p05 −10.2% vs −9.8%); top-20 share 7.5%; occupancy 14.7/25, days_full 20%, skip_cash 344 — sổ khỏe.
- Cơ chế: nới depth pullback-limit 4.5→4.0% = +70 trades/+50 fills chất lượng giữ nguyên (PF 3.32) → quay vòng vốn nhanh hơn; snr g12 cộng đuôi +0.05.

### r2_pb35_50: ×15.64 / 52.5% / −12.70% — PASS chống-ghost, bị pb40 vượt
- **Subframe ≥2022** (sv_subframe): comp 279.1 vs base 275.5 (**+3.6**), d_pnl +2.78; ≥2023: 238.1 vs 249.0 (−10.9); ≥2024: 161.8 vs 166.4 (−4.6) — cùng bậc, không sập.
- **NAV-từ-2022** (chống ghost NAV): ×3.66 vs base ×3.48 — **HƠN base cả ở khung bỏ 2020-21**; yearly 2022 +34.8% (base +24.8%), 2023 +39.1/2024 +18.8/2025 +49.9/2026 +9.7.
- **Occupancy/structure sanity**: avg_open 15.1/25 (base 15.0), days_full 23% (base 19%) — không nghẹt sổ; skip_cash 397 (base 324) chấp nhận được; per-entry-year DƯƠNG cả 7 năm; exit mix cùng hình (overext 529/+69.7, trail 453/+66.8, signal 1081/−26.3); percentiles pnl gần trùng base; top-20 winner share 7.3% — không concentration; 2161 trades, hold median 8d.
- Cơ chế: nới độ sâu pullback-limit 4.5→3.5% ở window 50 = fill được nhiều nhịp chỉnh nông hơn (+149 trades, +76 fills NAV) — đúng vũ khí throughput của hệ; giá vào cao hơn một chút (p25/p05 tệ hơn ~0.4pt) nhưng vòng quay vốn thắng.

### r2_pb45_70: ×15.46 (sát ngưỡng) — giữ làm datapoint, KHÔNG tuyên bố
NAV-từ-2022 ×3.46 ≈ base ×3.48 (trung tính) — lợi thế NAV full-frame tựa vào 2020 (+91.1% vs +82.8%) và 2023; subframe comp −12.6. Nhưng ở khung K20 nó là ĐỈNH: ×18.03/−15.54% (xem §4).

## 4. Bản đồ trục: cái gì dịch NAV, cái gì chỉ dịch composite

| trục | comp | NAV | verdict |
|---|---|---|---|
| **pullback depth @ w50** | +0.9..+4.3 | **ridge đỉnh 4.0% (+1.51); 3.5 +0.86; 3.0 −0.29; 5.5 −0.71** | **trục sống chính** — nới depth vừa phải = throughput thật (qua cả NAV-từ-2022); quá nông (3.0) fill rác nhịp chỉnh cạn, NAV rơi dù fill nhiều nhất |
| pullback window (30/70) | âm mọi ô | 70 hơi dương, 30 âm | window 50 là ridge; w70 +NAV nhờ 2020/2023 (pb45_70 KHÔNG qua NAV-từ-2022: ×3.46 ≈ base) |
| snr_extend/giveback (min_gain 0.12) | +0.4..+1.6 | +0.15..+0.17 solo; +0.04..+0.05 trên nền pb | dương nhẹ, cộng được (orthogonal), lấy free; giá trị gb-parity (gain 0.27) KHÔNG BIND ở hệ winner nhỏ — phải scale theo p95 winner của hệ |
| dtstop | ~0 | 0..−0.5 | −6% đã tối ưu; trục đóng |
| gate abovema10 | −2.5 | +0.37 (NAV-từ-2022 ×3.52 ≈ base → lợi thế tựa 2020-21) | NAV-dương/comp-âm nhưng không qua chống-ghost sạch; ưu tiên thấp |
| gate bỏ hẳn abovema20 | +1.0 | +0.07 | trung tính |
| gate abovema50 | −62 | −3.8 | chết thảm — siết trend dài giết throughput (2022 +10.7% vs +23.9%) |
| **K slot (không đổi config!)** | n/a | **base K20 ×17.70/−15.21%; pb45_70 K20 ×18.03/−15.54%; pb40_50 K20 ×17.58/−16.89%; pb35_50 K15 ×17.84/−19.72%** | **trục mạnh nhất toàn vòng về NAV thô**: hệ 2k-trade thừa signal nuôi sổ K20; ở CÙNG mức MaxDD với gb (−15.3%) NAV nhảy ×17.6-18.0 — nhưng là đòn bẩy risk, phải so ở matched-DD frontier chứ không cộng dồn với knob |

Bài học ox18 lặp lại chiều ngược: composite và NAV tiếp tục lệch pha (pb35_70/pb45_70 comp −12 nhưng NAV +0.6-0.7; ama50 comp −62 NAV −3.8 thì đồng pha). Mọi biến thể trong sweep giữ MaxDD trong −12.25..−14.5% — hệ này rất khó làm hỏng DD bằng knob entry/exit (gate/dtstop giữ nguyên tail).

## 5. Đề xuất vòng 2
1. **Tinh chỉnh quanh đỉnh 4.0/50 (ưu tiên 1)**: depth {0.038, 0.040, 0.042} × window {40, 50, 60} trên nền c2 (giữ snr g12) — xem đỉnh 16.34 có phải plateau hay gai đơn (nếu 3.8-4.2 đều ≥16 → tin được; nếu chỉ 4.0 nhô → nghi overfit knob, lùi về vùng phẳng).
2. **Matched-MaxDD frontier K × depth**: quét K {18,20,22,25} cho c2/pb45_70/base, vẽ frontier NAV-vs-MaxDD; câu hỏi quyết định: ở DD trần −15.3% (mức gb), hệ rule đạt ×17.5-18 — chọn điểm vận hành theo khẩu vị DD chứ không theo K mặc định 25. Kèm NAV-từ-2022 ở từng K (pb45_70 đã trượt check này ở K25 — nghi ở K20 cũng tựa 2020-21).
3. **Winner-riding còn nguyên gap +415**: snr chỉ ăn được +0.05-0.17 vì defer xong trade vẫn bị overext 12%/trail 8% chém. Thử NỐI: snr-defer + overext_trail_pct (arm trail chặt thay vì bán thẳng khi overext, key đã có ở gb: overext_trail 0.04) — cơ chế khác ox18 thô (đã chết): chỉ nới TRONG regime clean-trend.
4. KHÔNG đi tiếp: dtstop, depth ≤3.0 và ≥5.5, window 30, abovema50, snr threshold/window tinh chỉnh, gate_ama10 (trượt NAV-từ-2022).
5. Trước khi tuyên bố: 5-seed (hình thức — deterministic đã xác nhận ở fc_rule2), full regime test ≥2022 theo chuẩn snr08, và so trực tiếp với gb_x08 cùng ngày data.

## Files
- Scripts: r2_00_dump_cfg.py, r2_01_gbsnr_peek.py, r2_10_sweep.py (groups: base/snr/pb1/pb2/dt/gate/r2b/r2c), r2_nav.py, r2_20_sanity.py
- Trades: r2_<variant>_s42_trades.csv (r2line/); NAV series: r2_nav_<label>.csv (gồm *_from2022, *_K15/K20)
- Templates DB: r2_base t2845 … r2_c2_pb40snr (id 2845+, leaderboard run_name r2_*; canonical t2005/t2835 KHÔNG bị đụng)
