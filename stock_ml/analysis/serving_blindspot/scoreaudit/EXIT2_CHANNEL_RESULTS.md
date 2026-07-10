# EXIT2 CHANNEL — probe "kênh bán độc lập thứ hai" trên gb_x08 (2783)

Ngày: 2026-07-10. Thực thi đúng khuyến nghị SCORE_AUDIT.md §5: head mới (reward_risk h10)
vào như KÊNH SELL ĐỘC LẬP OR-union ngưỡng riêng, không blend vào exit_score cũ.
Scripts: `x2_00_qmap.py` (+csv), `x2_grid.py`, `x2_10_attrib.py` (+`x2_attrib.txt`), log `x2_*.log`.
KHÔNG đụng 2646/2730/2783 (clone-only); không đụng run op_*.

## 1. Hạ tầng exit2 thật (xác nhận trong code)

- `engine_config.exit_ensemble = {"target": {...}, "z_threshold": Z}` — tên key đúng như audit đoán.
- `experiment.py:2960-2962`: build `target_exit2` từ `exit_ensemble.target` → `train_fold`
  (`:916-927`) train head `exit2` bằng **exit features + exit_model params của template**
  (kể cả monotone_map của head velocity) → cột `exit_score2` (giữ qua fold checkpoint, `:940`).
- `experiment.py:2489-2491`: `sell |= causal_z(exit_score2, 252/60) > z_threshold` — OR-union
  **SAU** `exit_gate` cons2_w20 (`:2225`) và mọi force-rule → kênh độc lập thật, không bị gate nuốt.
- Pop-key sạch (`:3241`), gap tự nới theo horizon target ensemble (`:3326`) — h10 < gap 85, an toàn.
- Knob của kênh chỉ có `target` + `z_threshold`. KHÔNG có min_gain/điều-kiện-lãi per-channel
  (engine-level `exit_snr_min_gain=0.27` của 2783 vẫn áp lên mọi signal-sell như cũ) — biến thể
  "chỉ bán khi lãi" không biểu đạt được config-only, không phát minh knob mới.
- Ứng viên #1 reward_risk h10 = **config-only đúng nghĩa pipeline** (thêm block exit_ensemble,
  head tự train trong run); predictions bundle n2_3h có sẵn nhưng pipeline không nhận file ngoài.
  Ứng viên #2 (swal06 swing_value h20) không cần đến — xem verdict.

## 2. Ngưỡng theo quantile (`x2_00_qmap.csv`, causal z 252/60 của reward_risk h10)

| Năm | q97 | q98 | q99 | %>2.5 | %>3.0 | %>3.5 | %>4.0 |
|---|---|---|---|---|---|---|---|
| 2020 | 0.60 | 0.87 | 1.34 | 0.16 | 0.06 | 0.04 | 0.02 |
| 2021 | 1.99 | 2.16 | 2.47 | 0.95 | 0.42 | 0.16 | 0.07 |
| 2022 | 3.94 | 4.34 | 4.87 | 13.5 | 8.3 | 4.8 | 2.8 |
| 2023 | 0.84 | 1.00 | 1.23 | 0.09 | 0.03 | 0.00 | 0.00 |
| 2024 | 1.29 | 1.52 | 1.98 | 0.35 | 0.14 | 0.05 | 0.03 |
| 2025 | 4.07 | 5.05 | 6.86 | 5.7 | 4.6 | 3.8 | 3.1 |
| 2026H1 | 2.65 | 3.08 | 3.71 | 3.7 | 2.2 | 1.3 | 0.8 |
| pooled | 2.71 | 3.17 | 4.03 | 3.6 | 2.3 | 1.5 | 1.0 |

→ sweep z ∈ {2.5, 3.0, 3.5, 4.0} ≈ q96→q99 pooled. Phân phối bắn ĐÚNG nơi cần (2022 bear +
2025-26), câm 2020-21/2023-24 — **không phải mẫu ghost**, điều kiện dừng-sớm không kích hoạt.

## 3. Parity + sweep seed-42 (baseline gb_x08 = 735.0, pnl 128.4965, pf 6.0211, mdd 0.174549, 1378 tr)

| run | tmpl | comp | Δ gb_x08 | pnl | pf | mdd | trades |
|---|---|---|---|---|---|---|---|
| x2_parity (kênh off) | 2803 | **735.0** | **+0.0** | 128.50 | 6.021 | 0.17455 | 1378 |
| x2_rr10_z25 | 2804 | 724.6 | −10.4 | 125.82 | 5.915 | 0.17081 | 1390 |
| x2_rr10_z30 | 2805 | 722.1 | −12.9 | 125.72 | 5.874 | 0.17309 | 1383 |
| x2_rr10_z35 | 2806 | **731.7** | **−3.3** (best) | 127.76 | 5.971 | 0.17283 | 1379 |
| x2_rr10_z40 | 2807 | 729.6 | −5.4 | 127.70 | 5.947 | 0.17677 | 1380 |

Parity PASS tuyệt đối (identical đến full precision, cùng config-hash 32a8dfee). Mọi mức z ÂM;
best −3.3 << ngưỡng +2 → **DỪNG theo protocol** (không 5-seed, không regime test).

## 4. Phân bố sell kênh 2 theo năm (signal-level, folds thật của run; `x2_attrib.txt`)

`ch2` = bar zX2>thr; `ch2_add` = bar kênh 2 bắn mà sell_ml/sell_force gb_x08 KHÔNG bắn (additive thật).

| Năm | z2.5 ch2/add | z3.0 ch2/add | z3.5 ch2/add | z4.0 ch2/add |
|---|---|---|---|---|
| 2020 | 12/7 | 5/4 | 4/3 | 2/2 |
| 2021 | 119/85 | 38/29 | 14/10 | 7/5 |
| 2022 | 2133/640 | 1310/338 | 777/182 | 479/76 |
| 2023 | 8/4 | 3/1 | 1/0 | 0/0 |
| 2024 | 63/45 | 21/16 | 5/3 | 1/1 |
| 2025 | 869/81 | 709/32 | 592/18 | 498/9 |
| 2026H1 | 172/26 | 94/8 | 49/1 | 15/0 |

Phân phối trained-in-stack khớp qmap dự báo (2025 z2.5: 5.75% vs 5.70 dự báo) — head train trên
exit_vol_downpress tái tạo đúng hình dạng. Kênh KHÔNG câm: 2025 có 498–869 bar sell. Nhưng:

1. **Additive 2025 rất mỏng**: 9–81 bar — 91–98% bar kênh 2 bắn trùng force-rule đang bắn.
   Premise "force chưa bắn ở đó" sai ở mức bar: downleg12/nonbull phủ gần hết đuôi phải của head.
2. **Trade-level: mọi thay đổi 2025 đều làm XẤU PnL** (dpnl 2025: −0.34/−0.26/−0.14/−0.17 theo
   z tăng dần; 2022: −1.38/−1.23/−0.64/−0.59; 2020 z≤3.0 clip một winner −1.8). Chỉ 2021 dương
   nhẹ (+1.03 ở z2.5). Bán theo đuôi phải reward_risk h10 năm 2025 = bán trúng người thắng —
   đúng mẫu calibration-đảo mà SCORE_AUDIT §3 đo được cho head velocity, lặp lại ở head mới.
3. Mega-check: z3.5/z4.0 VCI/VIC/GEX/LPB **nguyên vẹn từng số** (không giết mega); z2.5/3.0 đổi
   nhẹ theo hướng trung tính — không phải nguồn lỗ.
4. Số trade đổi rất ít (16–120/1378) — kênh gần như chỉ churn quanh force, và churn nào chạm
   2022/2025 đều âm.

## 5. Verdict — con số quyết định

**Kênh độc lập KHÔNG mở khóa được giá trị nào mà blend "đã nuốt".** Cái blend nuốt là SỐ LƯỢNG
tín hiệu (1,042 bar sell 2025), không phải GIÁ TRỊ: khi cho toàn bộ số lượng đó đi qua kênh
OR riêng, phần additive sau force chỉ còn 9–81 bar/năm 2025 và phần chạm trade là âm ở mọi
ngưỡng (best seed-42 = 731.7, −3.3 vs gb_x08; cả 4 mức âm). Ống dẫn không oan: **head
reward_risk h10 không có thông tin bán đúng hướng ở 2025** (zX2 cao → bán người thắng), nên
mọi cách wire (blend hay OR) đều không tạo alpha.

Hệ quả cho các vòng sau:
- Đóng hướng "thêm head exit mới qua kênh OR tĩnh" cho stack 2783: hạ tầng hoạt động đúng,
  parity sạch, nhưng lớp force-rule đã chiếm phần bar hữu ích và phần còn lại của head là
  anti-selective. Ứng viên #2 (swal06 swing_value) không cần chạy — cùng cơ chế, và op_grid
  vừa đo swal06 ở operating-point riêng cũng ≤ +0.4 vs verdict cũ (op_s06_st21 = 726.6).
- Trần oracle exit 2024-26 (+1–3u/năm) nếu còn tồn tại thì nằm ở "điều-hòa-liên-tục theo
  regime" (đổi DẤU sử dụng head theo năm), không phải ở kênh/ngưỡng tĩnh — mọi ngưỡng tĩnh
  (blend, OR, quantile-informed) đã bị falsify bằng số ở vòng này.
- Template x2_* (2803-2807) + run template/x2_* giữ trong DB làm chứng cứ; không commit git.
