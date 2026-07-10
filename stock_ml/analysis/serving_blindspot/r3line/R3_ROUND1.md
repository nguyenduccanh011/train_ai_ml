# TUYẾN R3/MAXHOLD — VÒNG 1 (2026-07-10)

Nguồn: NAV_GEM_SCAN phát hiện t2516 = t2429 + max_hold_bars 20 (NAV ×19.78 adv / ×17.18 noadv). Vòng này: sweep trục + cross-apply + vá kênh signal-exit. Thước: nh_nav2 K25, R0.6, settle T+2, shuffle-mean±sd 20 perm, HAI chế độ (adv 0.08% / no-advance). Baseline gb_x08: ×13.78±0.42 adv / ×13.03 noadv, f22 3.43, DD −15.3/−15.6. Runner: `r3_10_run.py`; log: r3_10_A_out.txt, r3_10_B1_out.txt, r3_10_B2C_out.txt. (Agent vòng này chết giữa chừng, phần B2+C do main session chạy nốt bằng đúng runner.)

## A — Sweep max_hold trên t2429 (đời cũ, ML 4-head + pullback)

| mh | comp | NAV adv | NAV noadv | f22 adv/noadv | f23 adv | DDw |
|---|---|---|---|---|---|---|
| 12 | 624.3 | **×22.39±0.48** | ×19.05±0.66 | 4.32/4.21 | 2.93 | **−12.0%** |
| 16 | 637.6 | ×21.77±0.53 | **×19.62±0.60** | 4.27/4.10 | 2.83 | −13.8% |
| 20 (=t2516) | 645 | ×19.78 | ×17.18 | 4.12 | 2.77 | −12.4% |
| 25 | 670.0 | ×21.67±0.66 | ×18.89±0.76 | **4.34/4.40** | 2.87 | −13.7% |
| 30 | 680.4 | ×19.83±0.63 | ×18.45±0.86 | 4.19/4.20 | 2.75 | −14.6% |
| 40 | 679.4 | ×17.94±0.71 | ×17.37±0.65 | 3.87/3.89 | 2.60 | −12.9% |

**PLATEAU RỘNG 12-25** — điểm gem gốc (20) không phải đỉnh; mh12 đỉnh adv, mh16 đỉnh noadv, mh25 đỉnh f22. Composite NGƯỢC chiều NAV (tăng theo mh dài) — đúng bệnh composite thưởng hold. EV plateau (12-25): adv ~×21.4 (+55% vs gb), noadv ~×18.7 (+43%), f22 ~4.26 (+24%), DD −12..−13.8%.

## B — Cross-apply lên dòng hiện đại

| variant | NAV adv | noadv | f22 adv/noadv | DDw |
|---|---|---|---|---|
| gb(2783)+mh16 | ×20.16 | ×18.38 | 4.14/4.06 | −14.1% |
| gb+mh20 | ×18.89 | ×16.70 | 3.94/3.63 | −12.7% |
| gb+mh25 | ×18.31 | ×16.91 | 3.92/3.89 | −14.0% |
| gb+mh20 bỏ snr keys | ×18.69 | ×16.52 | 3.90/3.60 | −12.7% |
| 2643+mh20 | ×18.54 | ×16.43 | 3.84/3.55 | −12.7% |

- gb+max_hold lấy lại toàn bộ 23% NAV wavestruct làm mất **và hơn** (×13.78→×20.16 tại mh16) — NHƯNG vẫn THUA t2429+mh cùng điểm (~1.5-2.2 NAV-x): stack đời cũ + cap là combo mạnh hơn, không phải do wavestruct mà do khác head/label (cần công tố xử vì sao).
- snr keys không xung đột với max_hold (bỏ đi tệ hơn nhẹ) — van defer vẫn cộng chút giá trị dưới cap.
- 2643 ≈ 2646 khi đã cap → bước wavestruct trung tính dưới max_hold; tổn thất NAV của nó nằm ở hold-sâu không cap.

## C — Vá kênh signal-exit của t2516

| variant | NAV adv | noadv | DDw | verdict |
|---|---|---|---|---|
| tắt signal-exit | ×11.78 | ×10.10 | **−31.5%** | THẢM HỌA — kênh signal là xương sống |
| min_age 6/10 | ×14.2/×13.7 | ×12.5/×11.6 | −18..−24% | tệ |
| volhold (bộ hold-ext) | ×20.15 | ×16.92 | −12.6% | ~ngang t2516, adv nhỉnh; không đáng phức tạp thêm |

Kênh signal-exit −16.2u là "phí bảo hiểm" không cắt được (đúng mẫu suppress/struct-trail của gb). Đóng nhánh C.

## KẾT LUẬN VÒNG 1
Ứng viên vào công tố: **họ t2429+max_hold, tâm plateau mh16** (đại diện; công bố theo EV plateau 12-25, không lấy max). So gb_x08: EV +43-55% NAV (2 chế độ), f22 +24%, DD nông hơn 1.5-3.3đ. Câu hỏi công tố phải xử: selection inflation (~70 config R2+R3), vì sao stack đời cũ thắng stack hiện đại dưới cap (era-drift? label khác? cần loại trừ leak/era-artifact), lag+1, phí, serving stack đời cũ (entry_lvup126_lean era — wheel/serving parity?).
