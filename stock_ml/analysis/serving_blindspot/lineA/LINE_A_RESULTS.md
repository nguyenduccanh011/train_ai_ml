# TUYẾN A — KẾT QUẢ RESEARCH KÊNH VÀO LỆNH (2026-07-09)

3 mũi song song, ~25 run leaderboard thật (seed 42, clone 2646, champion không bị đụng). Scripts + logs cùng thư mục. Câu hỏi: có tồn tại cơ chế vào lệnh không-pullback (hoặc ít chờ) giữ được hiệu suất champion không?

## Bản đồ gap (seed 42, champion = 729.6)

| Trục | Best case | Δ | Kết luận |
|---|---|---|---|
| **A2 — bỏ pullback (at-market)** | a2_nopb 492.3 | **−237.3** | PF 5.96→3.08, MDD/symbol ×1.9. MỌI lớp bảo vệ làm TỆ THÊM: structural stop −11..−26, hard stop (đã sửa priority) −23.3, trailing sớm −1.5, min_hold 0 −0.7 |
| **A3 — limit nông / window ngắn** | pb02_w10 613.7 | −116 | Chảy máu tuyến tính theo window (w40/w20/w10: −10/−60/−141); confirm_reversal ở depth nông tệ thêm −28..−49 |
| **A3 — hybrid mua đuổi có trần** | a3_hyb_fm01 719.5 | −10.1 | Trần premium 1%: gần miễn phí, xóa 7.582 signal-drop, nhưng KHÔNG bắt runaway, KHÔNG giảm limit treo |
| **A1 — mở gate dưới MA20** | a1_gate_upleg_rsiok10 726.1 | −3.5 | Gần miễn phí nhưng chỉ +12 lệnh: tầng pullback tự vứt tín hiệu chân sóng (không có nhịp chỉnh 4.5% để khớp). Điểm mù #1 nằm ở EXECUTION, không phải gate |

## Kết luận cứng (đã xác nhận trên champion hiện hành, khớp EXP-1 engine.py:659)

1. **Alpha của champion = cái đệm giá 4.5% tại entry** (basis rẻ hơn), không phải selection (chỉ ~9% tín hiệu bị pullback loại hẳn trong khung leaderboard — trades chỉ tăng 10% khi bỏ pullback). Không cơ chế exit nào cứu được entry đắt hơn 4.5%.
2. **"Model không pullback hiệu suất ngang champion" với stack tín hiệu HIỆN TẠI: bác.** Gap −237 điểm, mọi protection làm tệ thêm. Muốn giữ hiệu suất phải giữ fill kiểu limit.
3. Pool tín hiệu dưới MA20 không độc (A1) nhưng không thể monetize qua pullback; muốn bắt chân sóng phải có **kênh fill riêng + tín hiệu xác nhận riêng** — không tồn tại config-only.
4. Đóng vĩnh viễn các trục: depth/window tuning (bác lần 3), confirm_reversal, entry gates bó hẹp, stops cơ học trên at-market.
5. Config trap ghi nhận: `hard_stop_pct` KHÔNG có `"hard_stop"` trong `exit_priority` (+ quy ước dấu âm) = no-op bit-identical — bẫy cho mọi sweep sau.

## Hệ quả cho hướng research đột phá

Execution layer đã được **map cạn kiệt** — không còn đột phá ở tầng thực thi. Con đường còn lại chuyển lên **tầng tín hiệu**:

- **Hướng chính đề xuất: "wave-start specialist head"** — train một head chuyên biệt cho cú đảo chiều chân sóng (label kiểu bottom-turn/reversal quality, không phải triple-barrier tổng quát), dùng làm điều kiện kích hoạt kênh B với fill neo khác (limit tại swing-low reclaim / MA-touch, không phải % từ close). Lý do còn hy vọng: A2 test at-market cho TOÀN BỘ tín hiệu generic; một head chuyên biệt + fill có neo giá là tổ hợp chưa từng thử. Kỳ vọng: cộng THÊM lệnh mới (bar chấp nhận = net dương + không phá MDD), không phải thay kênh chính.
- **Nhặt lộc đã xác nhận:** multi-seed `w0_snr_10` (+2.2, MDD không đổi); cân nhắc `a3_hyb_fm01` (−10.1 đổi lấy xóa drop — quyết định vận hành, không phải alpha); `a1_gate_upleg_rsiok10` (−3.5) chỉ có nghĩa khi kênh B ra đời.
- Templates a1_/a2_/a3_ (2682–2706) giữ nguyên trên leaderboard làm kết quả âm trung thực (đều dưới champion, không gây nhiễu ranking).
