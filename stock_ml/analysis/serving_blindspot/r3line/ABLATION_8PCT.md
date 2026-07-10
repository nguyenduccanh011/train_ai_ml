# ABLATION +8% — key nào tốn NAV dưới cap? (t2429+mh16 vs gb+mh16)
Ngày: 2026-07-10. Thước: `nh_nav2` K25 R0.6 settle T+2, shuffle-mean±sd 20 perm, hai chế độ (adv 0.08% / no-advance). Seed 42 (ứng viên có thêm 7/99/555). Scripts: `ab_00_diff.py, ab_10_run.py, ab_20_mech.py … ab_30_seeds.py` (+ ab_*_out.txt, ab_*_trades.csv). Mốc: gb+mh16 (t2910) ×20.16±0.58/×18.38±0.80; t2429+mh16 (t2907) ×21.77±0.53/×19.62±0.60; gb_x08 (t2783) ×13.78/×13.03.

## 1. DIFF chính xác t2783 (gb) vs t2429 (từ DB, ab_00_out.txt)
Meta/split/universe/entry-head/exit-head/threshold: **giống hệt**. Khác đúng 16 engine key (gb CÓ, t2429 KHÔNG) + exit feature-set. Nhóm 5 cụm:

| cụm | keys | giá trị gb |
|---|---|---|
| **W** wavestruct hold-ext (8) | signal_exit_hold_{ext_atr, ma, min_score3_z, profit_floor, rs_scale, legage_pct, legage_scale, legamp_scale} | 1.3 / 50 / 0.5 / −0.03 / 8.0 / 0.06 / 0.5 / 0.2 |
| **S** snr defer/extend (4) | exit_snr_{extend_threshold, extend_window, min_gain, defer_min_giveback} | 0.8 / 20 / 0.27 / 0.08 |
| **P** protect/skip (2) | signal_exit_protect_release_drop_k, signal_exit_skip_if_score3_z | 2.5 / 1.6 |
| **T** trailing struct (2) | trailing_struct_apply_overext, trailing_struct_donch_win | True / 80 |
| **F** exit-fs | exit_vol_downpress (19f) vs exit_vol_market (14f); downpress = vol_market + {dist_day_25, dist_day_vol20_25, down_vol_intensity_5, down_vol_count_10, updown_vol_20} — vol_market là TẬP CON | (retrain exit head) |

## 2. Bảng ablation từ phía gb+mh16 (bỏ/thay từng cụm; clone t2930-2937, seed 42)
Baseline gb+mh16: adv ×20.16±0.58 / noadv ×18.38±0.80; f22 4.14/4.06; f23 2.77/2.73.

| variant | full adv | Δadv | full noadv | Δnoadv | f22 adv/noadv | f23 adv/noadv | DDw |
|---|---|---|---|---|---|---|---|
| ab_noW (bỏ W) | ×19.09±0.70 | **−5.3%** | ×16.99±0.60 | **−7.6%** | 4.00/3.90 | 2.75/2.71 | −14.1 |
| ab_noS (bỏ S) | ×20.11±0.58 | −0.2% | ×18.25±0.80 | −0.7% | 4.13/4.05 | 2.77/2.73 | −14.1 |
| ab_noP (bỏ P) | ×20.42±0.45 | +1.3% | ×18.38±0.69 | 0.0% | 4.14/4.08 | 2.76/2.73 | −13.8 |
| **ab_noT (bỏ T)** | **×22.87±0.64** | **+13.4%** | **×20.99±0.89** | **+14.2%** | **4.43/4.22** | **2.92/2.85** | −14.0 |
| ab_fsvm (F→vol_market) | ×20.12±0.55 | −0.2% | ×19.07±0.86 | +3.8% | 4.13/4.10 | 2.78/2.76 | −14.1 |
| ab_allrm (bỏ hết + fs) — sanity | ×21.77±0.53 | +8.0% | ×19.62±0.60 | +6.7% | 4.27/4.10 | 2.83/2.77 | −13.8 |

- **Sanity KÍN**: ab_allrm tái lập t2907 **bit-chính-xác** (comp 637.6, 2451 lệnh, mọi NAV trùng) → phân rã hợp lệ.
- **Phân rã ~tuyến tính**: −5.3(W) −0.2(S) +1.3(P) +13.4(T) −0.2(F) ≈ +9.0 ≈ +8.0 quan sát (adv).
- **ĐÁP ÁN**: +8% KHÔNG rải đều — do MỘT cụm duy nhất: **T (trailing_struct 2 key) tốn ~13-14% NAV dưới cap**; t2429 "thắng" gb chỉ vì nó không mang T. W ngược lại CỘNG +5-8% (t2429 không có W nên mới chỉ hơn +8% chứ không phải +13%). S, P, F trung tính (F hơi cộng noadv +3.8%, không bõ đổi fs khỏi champion).

## 3. Ứng viên `ab_modernclean_mh16` ≡ **ab_noT** (t2936) = gb_x08 + max_hold 16 − 2 key trailing_struct
Giữ nguyên stack hiện đại của champion live (downpress exit-fs, đủ W/S/P) — chỉ bỏ 2 key + thêm cap. So 3 mốc (full adv | noadv):

| mốc | NAV full | ab_noT hơn | f22 | f23 |
|---|---|---|---|---|
| gb_x08 (t2783) | ×13.78 \| ×13.03 | **+66% \| +61%** | 3.43/3.34 | 2.46/2.44 |
| gb+mh16 (t2910) | ×20.16 \| ×18.38 | **+13.4% \| +14.2%** | 4.14/4.06 | 2.77/2.73 |
| t2429+mh16 (t2907, SURVIVED) | ×21.77 \| ×19.62 | **+5.1% \| +7.0%** | 4.27/4.10 | 2.83/2.77 |
| **ab_noT (t2936)** | **×22.87±0.64 \| ×20.99±0.89** | — | **4.43/4.22** | **2.92/2.85** |

Vượt chuẩn "≥ t2429+mh16 − 2%" với dư địa lớn: **ứng viên serving-friendly mới**, còn HƠN đại diện SURVIVED ở cả 6 ô đo.
- **Seed-luck loại bỏ (ab_30_out.txt)**: s7 ×22.00±0.72/×20.57±0.82, s99 ×21.82±0.63/×20.44±0.85, s555 ×22.26±0.72/×20.69±0.82 (f22 4.29-4.38/4.14-4.24; f23 2.94-2.95/2.86-2.91; comp 645-648, ~2430 lệnh, pf ~4.0). **Seed-mean 4 seeds: adv ×22.24 / noadv ×20.67** — mọi seed ≥ t2429+mh16 điểm công bố, và ≥ seed-mean t2907 (×21.06/×19.38) +5.6%/+6.7%; riêng noadv min-seed 20.44 vs 19.62 = +4.2%.
- **Stack test noT+fsvm (ab_11_out.txt)**: ×22.97±0.60/×20.91±0.77 — fsvm KHÔNG cộng gì trên nền −T (±0.4% = noise) → giữ exit_vol_downpress, parity champion nguyên vẹn, khẳng định F trung tính.

## 4. Cơ chế: cụm T làm gì mà tốn NAV? (trade-level, ab_20/ab_21/ab_22/ab_24_out)
- `trailing_struct_donch_win=80` thay tier trailing bằng "gãy đáy Donchian-80"; `apply_overext=True` áp luôn cho kênh overext — trong đời sống ≤16 phiên của cap, đáy-80-phiên gần như KHÔNG BAO GIỜ gãy ⇒ **kênh bán-vào-sức-mạnh chết hẳn**: gb+mh16 có **0 lệnh overext_trail/trailing_stop** (1191 max_hold + 1175 signal); t2429+mh16 có 199 overext_trail (+17.2%, 8.7 phiên, wr 96%) + 10 trailing_stop. (gb gốc không cap cũng vậy: 1374/1378 lệnh thoát bằng signal — T vốn là van "ride runner" chủ lực của gb_x08 khi hold 29.7 phiên.)
- Cô lập T (matched ab_noT vs gb+mh16, khác đúng 2 key): **148 lệnh** gb ôm tới cap bar 17 (+23.2%) thay vì chốt overext bar 9.2 (+18.6%) — per-trade gb HƠN +4.6đ, nhưng trả giá +7.8 phiên/lệnh; thêm 16 lệnh signal-exit muộn (bar 13.5, +8.6%) mà noT chốt overext bar 7.1 (+15.1%).
- Vận tốc vốn: kênh overext_trail sinh **1.97%/ngày-giữ** — nhanh nhất hệ — bị cuộn vào max_hold chỉ 0.63%/ngày. Slot giải phóng sớm 601 symbol-ngày → noT vào thêm 190 entry mới (+11.0u) vs gb-only 117 (+5.0u).
- Tổng: PnL thô gần hòa (111.7 vs 111.0u) nhưng NAV compound K25 **+13.4%** — T không "mất tiền trên lệnh", nó **giết vòng quay**: dưới cap 16 phiên, "ride cấu trúc" bị chặt đúng ở bar 16 nên mất cả đỉnh xa lẫn vòng quay. T là key tối ưu cho hệ hold-dài, phản tác dụng dưới hold-ngắn — đúng loại tương tác cap × trailing mà R3 đi tìm.

## 5. Verdict
1. **+8% quy được trọn** cho cụm T (2 key trailing_struct); các cụm khác: W +5-8% (PHẢI GIỮ), S/P/F ~0.
2. **ab_modernclean_mh16 = ab_noT (t2936)**: config = t2783 nguyên vẹn − {trailing_struct_apply_overext, trailing_struct_donch_win} + max_hold_bars 16 + exit_priority prepend "max_hold". Seed-42 ×22.87/×20.99, seed-mean 4 seeds ×22.24/×20.67 — hơn t2429+mh16 mọi seed, mọi lát (full/f22/f23), cả hai chế độ, VÀ giữ serving-parity hiện đại (đường live champion, exit-fs downpress nguyên bản, không cần bundle đời cũ).
3. Đề xuất: đưa ab_noT vào ghế bị cáo công tố kế tiếp (selection inflation ~80 config đã chấm NAV, held-out f21/f24/f25, LOYO, lag/phí) — số seed-42 này CHƯA phải công bố. Điều kiện serving kế thừa nguyên án t2907: host phải enforce max_hold 16 phiên (engine-only knob).
