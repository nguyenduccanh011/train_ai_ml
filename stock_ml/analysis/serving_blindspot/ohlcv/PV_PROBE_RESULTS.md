# PV-CHANNEL PROBE — exit head + kênh giá–khối lượng (pv_corr_10) — KẾT QUẢ ÂM

Ngày: 2026-07-09. Tiếp nối `OHLCV_VIRGIN_MAP.md`: kênh duy nhất sống sót screening trực giao
(pv_corr_10, resid-IC +0.041 phía exit label, 5/5 fold, ~5σ) được đưa vào exit head của champion
2646 qua feature-set MỚI (clone, không đụng template/set cũ).

## Setup

- Base: clone template **2646** `n2_2643_wavestruct_la05_lamp02` (champion, seed-42 = 729.6).
  Chỉ đổi `component_slots[exit].feature_set_name`; toàn bộ engine_config/threshold/slot khác giữ nguyên.
  Clone: **2779 pv_full**, **2780 pv_only** (runner `pv_probe.py`, pattern rq_/sv_probe; exit head retrain).
- Feature mới (`stock_ml/src/features/catalog.py`, additive; parity vs `screen_ic.py` xác minh bằng
  `tmp_pv_parity.py` trên 5 mã / 21 762 bar: **maxdiff = 0.0 tuyệt đối** ở mọi row hai bên cùng có giá trị):
  - `pv_corr_10` = `Fillna(Corr(Pct($close,1), Delta(Log($volume/($volume>0)),1), 10), 0)`
  - `clv` (có sẵn trong catalog; idiom h==l → 0 thay vì NaN — chỉ khác screen ở bar trần/sàn full-lock)
  - `nr_pos_7` = `#tr_1 / Max(#tr_1,7)` (helper mới `tr_1` = true range; guard tuần chết → 0)
- Set mới: `exit_vol_downpress_pv` (19+3), `exit_vol_downpress_pvonly` (19+1). Set cũ
  `exit_vol_downpress` **nguyên vẹn byte-identical** (assert pass trong tmp_pv_parity.py).
- Thay đổi engine DSL: thêm op **`Fillna`** (elementwise, additive) vào `dsl/ops.py`. Lý do: run đầu
  pv_full FAIL đúng fail-loud policy — `pv_corr_10` có 171 NaN nội tại trong test fold (BCM đầu 2020:
  volume 0 / stretch giá chết ⇒ Corr undefined; screening chỉ skip row khi tính IC, model cần hàm total).
  Fill 0 = "không có thông tin đồng chuyển động". Side-effect đã cân nhắc: engine_code_fingerprint đổi
  ⇒ feature-store recompute 1 lần (giá trị feature cũ không đổi; champion không re-run nên không rủi ro).

## Kết quả (seed 42)

Champion s42 baseline: composite **729.6**, pnl 127.253, pf 5.963, mdd/sym 0.17455, trades 1384.

| run | tmpl | exit feature set | composite | Δ vs 729.6 | pnl | pf | mdd/sym | trades |
|---|---|---|---|---|---|---|---|---|
| pv_full | 2779 | exit_vol_downpress_pv (19+pv_corr_10+clv+nr_pos_7) | **728.8** | **−0.8** | 127.24 | 5.939 | 0.17651 | 1386 |
| pv_only | 2780 | exit_vol_downpress_pvonly (19+pv_corr_10) | **728.1** | **−1.5** | 127.13 | 5.936 | 0.17660 | 1385 |
| pv_snr | — | KHÔNG chạy (điều kiện tiền đề ≥ +2 không đạt) | | | | | | |
| 5-seed / regime test | — | KHÔNG chạy (điều kiện ≥ +3 không đạt) | | | | | | |

### Chẩn đoán hành vi (trade overlap vs champion, `tmp_trade_overlap.py`)

| run | shared entries | probe-only | champ-only | same exit-date (shared) | shared pnl khác (>1e-9) |
|---|---|---|---|---|---|
| pv_full | 1369/1386 | 17 | 15 | 96.8% | 50 trade, mean\|Δ\| 0.21% |
| pv_only | 1367/1385 | 18 | 17 | 97.1% | 48 trade, mean\|Δ\| 0.12% |

⇒ Exit head hấp thụ feature mới với thay đổi hành vi gần bằng 0: ~99% trade trùng entry, ~97% trùng cả
ngày exit; khác biệt rải đều các năm (không có regime nào hưởng lợi). Đây không phải noise seed —
cùng seed 42, cùng entry stack; phần thay đổi duy nhất là exit model, và nó hầu như không đổi quyết định.

## Verdict — ĐÓNG kênh pv trên exit head hiện tại

1. **Kết quả âm trung thực**: cả 2 probe đều DƯỚI champion (−0.8 / −1.5), dưới ngưỡng tiếp tục (+2)
   → theo protocol dừng: không pv_snr, không 5-seed, không regime test.
2. **Diễn giải**: IC trực giao +0.041 là thật về mặt thống kê nhưng QUÁ NHỎ để đổi quyết định của
   exit head sau khi 19 feature cũ đã nói phần lớn câu chuyện — LightGBM gần như bỏ qua kênh mới
   (97% exit y hệt), phần còn lại là reshuffle nhiễu quanh biên z-threshold 2.0. Khớp với cảnh báo
   trong OHLCV_VIRGIN_MAP: mức 0.035–0.04 "là NHỎ so với cú nhảy composite ≥ +5 điển hình cần
   label mới + thông tin mới". Thông tin mới ở mức này KHÔNG đủ nếu chỉ nhét thêm cột vào head cũ.
3. **Không mở vòng hyperparameter/label mới trên kênh pv ở dạng feature-cho-head-cũ.** Nếu có bao giờ
   quay lại, con đường duy nhất còn logic là kênh pv làm GATE/label riêng (ví dụ điều kiện hóa
   signal_exit theo pv_corr cao) — nhưng số ở đây không tự biện minh cho việc đốt thêm run.
4. Hạ tầng để lại (đều additive, không ảnh hưởng champion): op `Fillna`, feature `pv_corr_10`/`tr_1`/
   `nr_pos_7`, 2 set `exit_vol_downpress_pv*`, template 2779/2780 + 2 row leaderboard `pv_full`/`pv_only`
   (giữ làm bằng chứng âm).

*Files: pv_probe.py (runner), pv_full_s42.log / pv_only_s42.log, tmp_pv_parity.py (parity PASS),
tmp_trade_overlap.py, tmp_champ_row.py, tmp_inspect_tmpl.py.*
