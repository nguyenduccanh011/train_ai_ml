# DEEP STOP (hard_stop tầng sâu trên gb_x08) — KHE ĐÃ ĐÓNG Ở BƯỚC OFFLINE

**Ngày:** 2026-07-10. **Câu hỏi:** tồn tại mức hard stop X ∈ [−12%,−20%] cắt 9 lệnh tail ≤−15%
(−1.49u) mà không chạm winner? **Trả lời: KHÔNG — mọi mức đều ÂM, kể cả phần "cứu tail" cũng ≈ 0.**
Bước 2 (probe config-only `hs_`) KHÔNG chạy vì gate offline (vùng dương ≥ +0.8u) fail tuyệt đối.

## Setup
- Dữ liệu: `gbx08_enriched2.csv` (1378 trades seed-42, sum pnl 128.5u) + OHLCV duckdb.
- Mô phỏng đúng engine (`engine.py` L1976–1981, L2393–2399): trigger khi `low[i]/entry_fill−1 ≤ X`
  với `hold_bars ≥ min_hold_bars=2`; **fill = CLOSE bar i+1** (close_next) × (1−slip .0015) − RT .004.
- Parity: tái tính pnl 1377 lệnh close-fill từ OHLCV khớp `pnl_pct` max_err < 1e-6 → mechanics đúng.
- Script: `hs01_separation.py`; per-trade sim: `hs01_touch_sim.csv`. Slot-giải-phóng = 0 (bảo thủ).

## 1. Separation: winner-nhúng-sâu vs tail (bar trigger-eligible: age≥2, trước decision bar)
| X | winner chạm (n / pnl / mega≥.3) | mid-loser chạm (n / pnl) | tail chạm (n / pnl) |
|------|------------------|------------|------------|
| −10% | 6 / +0.72u / 1 mega (STB +0.53) | 78 / −6.29u | 6 / −1.02u |
| −12% | 2 / +0.02u / 0 | 46 / −4.14u | 5 / −0.85u |
| −15% | 1 / ~0u / 0 | 21 / −1.98u | **3 / −0.48u** |
| −18% | 0 | 8 / −0.74u | 3 / −0.48u |
| −20% | 0 | 3 / −0.33u | 2 / −0.32u |

Depth-separation phía winner RẤT SẠCH (MAE winner: q05 = −6.4%, min −15.7%; chỉ 1 mega từng chạm −10%).
**Nhưng phía tail cũng "sạch" theo nghĩa xấu: chỉ 3/9 lệnh tail có bar eligible chạm −15%.**

## 2. Vì sao stop sâu không cứu được tail (trace bar-by-bar 9 lệnh)
Tail gb_x08 = crash 1–3 phiên sàn liên tiếp ngay cuối đời lệnh; low sâu nhất nằm **trên chính bar
fill của exit thực** hoặc bar ngay trước nó — force-gate/signal exit hiện có đã bán ở close khả dụng
đầu tiên sau khi crash bắt đầu:
- NVL −20.5%: đường close −2%→−4%→−5%→−8%→−14%→exit −19.9%. Chạm −15% lần đầu = chính bar exit. Stop −14% trigger bar áp chót → fill cùng close exit. dpnl = 0.
- SHB −16.4% (hold 3), MSN, PC1, DCM, AAS: cùng pattern — không có bar eligible nào chạm −15% trước decision bar.
- GEX −16.2%: chạm −19% intrabar age 2, nhưng close bar 3 (= fill của stop) cũng chính là close exit thực. dpnl ≈ 0.
- KBC −15.0%: chạm −15% ở age 72 = decision bar của exit thực. dpnl = 0.
- Tuổi chạm: tail chạm −15% ở age {2,2,72}; winner-chạm−12% ở age {2,3} → **depth×age KHÔNG tách được** (cùng age 2–3), knob age-conditioned không có cửa — không thiết kế tiếp.

## 3. First-order dpnl theo mức X (fill next-bar close, đúng engine; slot = 0)
| X | nTrig | ΔTail | ΔWinner | ΔMid-loser | **ΔTotal** |
|------|----|--------|--------|--------|------------|
| −10% | 90 | −0.12u | −0.93u | −0.96u | **−2.00u** |
| −12% | 53 | −0.12u | −0.16u | −0.84u | **−1.12u** |
| −14% | 32 | −0.12u | −0.06u | −0.66u | **−0.84u** |
| −15% | 25 | −0.06u | −0.06u | −0.56u | **−0.68u** |
| −16% | 20 | −0.06u | 0 | −0.58u | **−0.64u** |
| −18% | 11 | −0.01u | 0 | −0.42u | **−0.42u** |
| −20% | 5 | −0.01u | 0 | −0.22u | **−0.23u** |

- **ΔTail ≤ 0 ở MỌI mức**: fill next-bar-close trên chuỗi sàn VN bán ĐÚNG close mà exit thực đã bán,
  hoặc thấp hơn (AAS −0.115, DPM −0.11, SSI −0.08 là các lệnh bị stop bán đáy rồi giá hồi).
- Nguồn lỗ chính = mid-loser (−5%..−15%) bị chốt đáy mất phần hồi: −0.22u đến −0.96u tùy mức.
- Theo năm (X=−15%): 2025 −0.58u, 2022 −0.15u — âm đúng vào các năm regime-test.
- MDD angle: stop không giảm được realized loss của bất kỳ lệnh nào (fill cùng/thấp hơn close exit thực)
  → không có Δ MDD/sym dương để composite thưởng. Không có phần bù.

## Verdict
**Khe "cắt tail miễn phí bằng hard stop tầng sâu" CHẾT — đóng bằng chứng cứ offline, không tốn run.**
Con số quyết định: cứu tail tối đa = +0.005u (X=−18/−20) vs thiệt hại ≥ −0.42u; mức "hợp lý" −15%
mất −0.68u. Nguyên nhân cấu trúc: (1) exit stack hiện tại (force downleg12/nonbull/lowbreadth +
signal) đã bán trong 1–2 phiên sau khi crash bắt đầu; (2) engine chỉ có fill next-bar-close cho
hard_stop — trên chuỗi giảm sàn không tồn tại giá thoát tốt hơn ở dạng config-only. Muốn cứu tail
thật sự phải có **intraday stop-limit fill tại mức stop** (engine work, và trên chuỗi sàn VN cũng
không khớp lệnh được) — không phải khe config.

Kết luận này CỘNG với EXIT_ATTRIBUTION_MAP §4 (siết trail/vol_spike/trend_break/incubate đều chết):
toàn bộ họ lever "cắt lỗ nhanh hơn" trên gb_x08 đã cạn. Tail −1.49u/1378 lệnh là chi phí cấu trúc
của at-market crash risk, không bòn được thêm bằng exit config.
