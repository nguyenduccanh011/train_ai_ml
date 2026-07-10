# GIẢI PHẪU MẶT TỐI RULE `exit_snr_extend` (ứng viên 2730 vs champion 2646)

Ngày: 2026-07-09. Seed phân tích: 42. Scripts + dữ liệu trung gian: `signalq/autopsy/` (ax_join.py, ax_counterfactual.py, ax_regime.py, ax_rescue.py, ax_slices.py, deferred_trades.csv, sx_*_s42_trades.csv, sx_*_s42.log).

## 0. Nguồn dữ liệu (đã xác minh, không đoán)

| tập trade | file | xác minh |
|---|---|---|
| champion 2646 s42 | `signalq/st_champ2646_s42_trades.csv` (n=1384, pnl 127.253) | khớp 100% run_trades DB `template/n2_2643_wavestruct_la05_lamp02-32a8dfee` (0 lệch pnl, 1384/1384 match) |
| ứng viên 2730 s42 | `signalq/sv_snr08_s42_trades.csv` (n=1376, pnl 128.177) | khớp `RESULT xq_snr_t08_g27 tmpl=2730 seed=42 comp=733.3 pnl=128.177 tr=1376` (run_exit_family.log:810) |

Key thật trong engine (engine.py:319-321): `exit_snr_extend_threshold`, `exit_snr_extend_window`, **`exit_snr_min_gain`** (không phải `exit_snr_extend_min_gain` như tên gọi miệng). Template 2730 = 2646 + `{threshold:0.8, window:20, min_gain:0.27}` — xác nhận từ engine_config DB.

Cơ chế thực tế (engine.py:2225-2228): tại bar mà signal-exit muốn nổ, nếu SNR(20) universe ≥ 0.8 **và MFE (đỉnh so với giá vào) ≥ 27%** thì `continue` — signal-exit bị nuốt. Lưu ý: điều kiện dùng **đỉnh**, không phải lãi hiện tại → lệnh từng peak +27% nhưng đã nhả về +5% vẫn bị hoãn bán. Trail hiện hữu (stop 8%/activate 27%, skip-above-MA10, struct donch 80) **không bao giờ nổ trong lúc hoãn** — cả 51 lệnh hoãn đều kết thúc bằng chính exit `signal` khi SNR tụt < 0.8.

## 1. Giải phẫu join seed-42 (2730 vs 2646, khóa symbol+entry_date)

- Matched 1374 cặp (entry_price identical — entries không đổi, đúng thiết kế). Exit khác nhau: **51 lệnh bị hoãn** (3.7%).
- Champ-only (entry bị chặn do slot còn kẹt vì lệnh hoãn giữ chỗ): **10 lệnh, +1.579u bị mất**. 2730-only: 2 lệnh, −0.011u.
- Phân rã tổng: Δpnl toàn cục **+0.924u** = nhóm hoãn **+2.515u** + knock-on **−1.590u**. Chi phí ẩn knock-on nuốt ~63% lãi gộp của rule.

### (a) Phân phối Δ của 51 lệnh hoãn
- **21/51 = 41.2% TỆ hơn**, tổng −1.475u; 28/51 tốt hơn +3.990u; 2 hòa.
- Median Δ chỉ +0.008; p5 = −0.142, min = **−0.249** (PVS 2020-09: +70.3% → +45.4%, mất 24.9pp chỉ vì hoãn 3 bar). Tail xấu top-3: PVS −0.249, AAV −0.232, AAS 2025 −0.164.
- **Toàn bộ lãi của rule = 2 mega-runner**: VCI 2020-11 (+1.286, hoãn 91 bar) + VIC 2025-07 (+1.201, 57 bar) = +2.49u. **Loại 2 lệnh này, nhóm hoãn còn +0.027u ≈ 0.** Bucket theo số bar hoãn: 1-3 bar +0.16, 4-10 bar −0.07, 11-30 bar −0.03, >30 bar +2.46 (n=4). Rule bản chất là vé số mega-runner, phần còn lại zero-sum.
- Hoãn xong bán đúng lúc tape nguội: 84.3% exit muộn xảy ra khi SNR < 0.8 (median 0.578), SNR rơi median −0.47 từ lúc hoãn đến lúc bán — adverse selection cấu trúc (extension chỉ chấm dứt khi regime đã xấu).

### (b) Nhóm tệ tập trung ở đâu
- Theo năm entry: 2020 (11/19 lệnh tệ), 2022 (5/8, tổng −0.07); lãi nằm ở 2025 (+1.386, chỉ 2/14 tệ) và 2 mega. Năm exit tệ: 2020-2021 (14/21) — pha xoay trụ cuối sóng bull.
- Thanh khoản: tercile thấp (tval60 < ~40tr) 9/17 tệ vs tercile cao 5/17; median tval60 nhóm tệ 30.4tr vs 52.1tr nhóm tốt. Nhưng Δpnl theo tercile không tách được tiền (low +1.04 vì có mega, high +1.46).
- SNR: nhóm tệ bị hoãn khi SNR universe **cao hơn** (median 1.12 vs 0.76) và SNR riêng mã cao hơn (0.99 vs 0.60) — hoãn ở đỉnh regime nóng → chết khi nó gãy. SNR ≥ 1.1 lúc hoãn: 68.8% tệ, nhưng bucket này vẫn +1.075u (chứa mega) → không cắt được.

### (c) Pattern nhận diện trước — CÓ, và đo được
**`giveback_at_defer ≤ 0.10`** (lúc bị hoãn, giá còn nằm trong vòng 10% của đỉnh lệnh — tức exit head chủ động gọi đỉnh ngay sát đỉnh):
- n=12, **83.3% tệ**, tổng **−0.459u**;
- phần còn lại (đã nhả >10% từ đỉnh rồi mới có signal-exit): n=39, chỉ 28.2% tệ, **+2.974u**, và **giữ nguyên cả 2 mega** (VCI hoãn khi đã nhả 29.4% từ đỉnh, VIC 22.4%).

Diễn giải: khi exit head bắn tín hiệu NGAY SÁT ĐỈNH thì nó thường đúng (top-call chủ động) — hoãn là phá; khi tín hiệu đến SAU một cú nhả sâu (pullback giữa sóng) thì hoãn mới cứu được runner. **Knob này không tồn tại trong engine config** (rule chỉ có threshold/window/min_gain) → không thể rescue config-only bằng pattern này. Đề xuất lever tương lai (1 điều kiện trong khối engine.py:2225): chỉ hoãn nếu `closes[i] ≤ (1−X)·peak_high`, X≈0.10.

## 2. Counterfactual screens (first-order, trên 51 cặp — ax_counterfactual.py)

| screen | kết quả |
|---|---|
| min_gain ↑ 0.35/0.40/0.50 | 0.35 ~neutral (−0.05); 0.40+ **mất 1.0-1.1u** (nhóm tốt cũng peak thấp) → loại |
| threshold ↑ 0.9/1.0/1.1 | cắt mất mega ở SNR 0.76-0.95 (GEX, VND, PVS) → loại (leaderboard đã xác nhận: t10_g15 731.6, t12_g15 731.0 < 733.3) |
| SNR upper band [0.8,1.1..1.5) | mất 1.6-2.1u → loại |
| giveback cap TRONG lúc hoãn 5-15% | **âm toàn dải** (delta nhóm tốt −1.8..−3.1 nuốt sạch +0.4..+1.3 cứu được) — mega nhả 22-47% từ đỉnh giữa chừng rồi mới chạy tiếp; cap kiểu trailing sẽ giết đúng con gà đẻ trứng |

## 3. Rescue variants thực chạy trên leaderboard (seed 42, clone 2730 → sx_, không đụng 2646/2730)

| variant | tmpl | comp s42 | vs 2730 (733.3) | pnl | mdd | ghi chú |
|---|---|---|---|---|---|---|
| sx_g35 (min_gain .35) | 2776 | 733.3 | **+0.0** | 128.14 | 0.17455 | no-op, lát regime identical 2730 |
| sx_w40 (window 40) | 2777 | 748.1 | +14.8 | 131.99 | 0.17093 | **GHOST — xem bảng lát** |
| sx_w60 (window 60) | 2778 | 758.0 | +24.7 | 134.96 | 0.16808 | **GHOST — xem bảng lát** |

### Regime test (protocol cứng) — pnl entries theo lát, seed 42

| lát | champ 2646 | 2730 | sx_w40 | sx_w60 |
|---|---|---|---|---|
| entries 2020 | +44.34 | +45.33 | **+54.93** | **+62.26** |
| entries ≥2022 | +55.56 | **+56.40** | +56.32 | +55.28 |
| entries ≥2023 | +44.39 | **+45.30** | +45.22 | +44.11 |
| entries ≥2024 | +26.41 | **+27.23** | +26.99 | +26.18 |
| entries ≥2025 | +24.06 | **+24.88** | +24.65 | +23.83 |

- sx_w60: +17.53/+17.77u delta nằm ở entries 2020; ở mọi lát ≥2022 **thua 2730**, lát ≥2022 thua cả champion. sx_w40 cùng dạng (nhẹ hơn). Composite +24.7 được tài trợ 100% bởi sóng 2020-21 — đúng failure mode dsb60 đã bị công tố REJECT. Knock-on của w60 cũng phình to: 82 entry bị chặn, +9.97u mất.
- **REJECT sx_w40, sx_w60 ngay tại seed-42; không đốt 5 seed cho ghost** (cơ chế deterministic — SNR(60) trong melt-up 2020-21 dính ≥0.8 hàng tháng liền, seed không đổi được cấu trúc năm). Đây là ca minh họa: composite toàn kỳ có thể phồng +25 điểm trong khi mọi lát sống (≥2022) đều đi lùi.

## 4. Kết luận

1. **2730 giữ nguyên trạng thái ứng viên promote, KHÔNG đổi lấy biến thể nào.** Không biến thể config-only nào cải thiện được mặt tối mà không giết phần lãi thật: min_gain no-op/âm, threshold âm (đã có bằng chứng leaderboard), window là bẫy ghost 2020/21, giveback-cap kiểu trailing âm toàn dải.
2. Mặt tối của rule (ghi nhận trung thực, không che): 41% lệnh hoãn tệ hơn (−1.48u, tail −25pp/lệnh); knock-on −1.59u nuốt 63% lãi gộp; lãi ròng +0.9u pnl đứng trên đúng 2 mega-runner; exit sau hoãn có adverse selection cấu trúc (bán khi regime đã nguội). Composite +3.6 mean-5-seed của 2730 vẫn đứng vững vì PF/consistency cải thiện và lát ≥2022 thắng đều (+0.8..+0.9u mỗi lát) — nhưng cần hiểu đó là edge mỏng + vé số mega, không phải cải thiện phổ quát.
3. **Lever tương lai đáng giá nhất** (cần 1 knob engine mới, ngoài scope hôm nay): điều kiện giveback-at-defer — chỉ hoãn khi giá đã nhả ≥10% từ đỉnh lệnh. Trên seed-42 nó cắt đúng 12 ca (83% tệ, −0.46u) và giữ nguyên cả 2 mega (+2.97u còn lại). Ước lượng first-order ≈ +0.5u pnl + giảm tail; phải qua đủ 5 seed + regime test khi triển khai.
4. Nhắc protocol: mọi biến thể chạm `exit_snr_extend_window` > 20 phải nghi ghost trước tiên — trục window đã chứng minh là máy bơm composite bằng regime chết.
