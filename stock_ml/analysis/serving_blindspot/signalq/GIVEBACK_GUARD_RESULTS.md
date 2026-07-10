# GIVEBACK-GUARD CHO SNR_EXTEND — KẾT QUẢ (2026-07-09)

Lever sinh ra từ SNR_EXTEND_AUTOPSY.md: trong 51 lệnh bị 2730 hoãn bán, nhóm hoãn khi mới nhả ≤10% từ đỉnh → 83% tệ (−0.46u); nhóm đã nhả >10% → +2.97u (giữ cả 2 mega VCI/VIC). Knob chưa tồn tại → implement + sweep + 5-seed + regime test.

## Knob (gated, additive)
- `exit_snr_defer_min_giveback: float = 0.0` — chỉ cho phép defer signal-exit khi giveback hiện tại từ đỉnh ≥ giá trị này. Default 0.0 = hành vi 2730 y nguyên.
- Vị trí: engine.py:329 (field), :2235-2237 (điều kiện trong khối snr_extend defer). Code do agent autopsy-kế-nhiệm viết; script tại `autopsy/gb_sweep.py`, `autopsy/gb_parity_check.py`.
- **Parity PASS tuyệt đối**: clone 2730 + knob=0.0 (gb_p00) seed-42 = 733.3, trades 1376, mdd khớp từng chữ số với 2730.

## Sweep seed-42 (clone từ 2730; baseline 2730 s42 = 733.3, champion 729.6)
| run | giveback | comp | Δ vs 2730 | trades |
|---|---|---|---|---|
| gb_p00 | 0.00 | 733.3 | 0.0 (parity) | 1376 |
| gb_x05 | 0.05 | 734.0 | +0.7 | 1376 |
| **gb_x08** | **0.08** | **735.0** | **+1.7** | 1378 |
| gb_x10 | 0.10 | 734.9 | +1.6 | 1382 |
| gb_x12 | 0.12 | 734.6 | +1.3 | 1382 |

MDD/symbol bất biến 0.17455 ở mọi điểm. Đường lồi sạch, đỉnh 8-10% — đúng ranh giới autopsy dự đoán.

## Multi-seed gb_x08 (template 2783; script `gb_multiseed.py`)
| seed | comp | vs champion |
|---|---|---|
| 42 | 735.0 | +5.4 |
| 7 | 736.7 | +5.2 |
| 99 | 728.1 | +5.6 |
| 555 | 735.7 | +5.3 |
| 123 | 733.3 | +5.0 |
| **mean-5** | **733.76** | **+5.30** |

Dải delta hẹp bất thường (+5.0..+5.6) — đều hơn cả 2730 (+3.1..+4.3). Đóng góp riêng của giveback-guard trên nền 2730 ≈ +1.7 mean. **Lần đầu trong chiến dịch một ứng viên vượt +5 mean-5** (lằn ranh promote của quy luật 16 thế hệ).

## Regime test (protocol cứng, seed 42, `autopsy/gb_regime.py`)
pnl theo năm ENTRY:
| | 2020 | ≥2022 | ≥2023 | ≥2024 | ≥2025 | all |
|---|---|---|---|---|---|---|
| champ 2646 | 44.34 | 55.56 | 44.39 | 26.41 | 24.06 | 127.25 |
| 2730 | 45.33 | 56.40 | 45.30 | 27.23 | 24.88 | 128.18 |
| **gb_x08** | 45.44 | **56.61** | 45.27 | 27.23 | 24.88 | **128.50** |

- Thắng champion trên MỌI lát. Vs 2730: ≥2022 +0.21, ≥2024/2025 bằng, ≥2023 −0.03 (mức noise trên lát 45u, không phải regime-dependence). Gain rải đều 2020 (+0.11) + ≥2022 (+0.21) — KHÔNG phải ghost 2020/21 (đối chiếu: sx_w60 +24.7 toàn kỳ nhưng thua mọi lát ≥2022 → REJECT).
- Knock-on: entries bị chặn 10 → 7 lệnh (pnl_lost ~không đổi +1.58).

## VERDICT
**gb_x08 (template 2783) = ứng viên promote MỚI, thay 2730.** Điều kiện thắng đủ: mean-5 +2.26 so 2730 (5/5 seed dương), không thua lát ≥2022 nào ở mức có nghĩa, MDD bất biến, cơ chế có giải phẫu nhân quả chống lưng (không phải sweep mù). Cảnh báo giữ nguyên từ autopsy: CẤM nới `exit_snr_extend_window` >20 (ghost pump 2020/21).

Đường promote: leakage-auditor → kiểm tra wheel stock_ml_core có knob `exit_snr_extend_*` VÀ `exit_snr_defer_min_giveback` chưa → export bundle `--replicate-last-fold` → shadow serving 4-6 tuần song song champion → swap. Engine changes đang ở working tree, CHƯA commit (chờ user duyệt).
