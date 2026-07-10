# RE-ENTRY GAP — cohort "bán xong chạy tiếp" gb_x08 (template 2783, seed 42)

Ngày: 2026-07-10. Câu hỏi: thay vì "giữ lâu hơn" (đã bác ở EXIT_ATTRIBUTION_MAP §5), hệ có
MUA LẠI kịp 663 lệnh sold-then-rallied (giá +≥5%/20 bar sau exit, cf giữ 20 bar +60u gross) không?
Script + dữ liệu: `re00_reentry.py`, `re00_rallied.csv`, `re00_hypo.csv`, `re00_diag.csv`.
OFFLINE hoàn toàn — 0 run leaderboard.

## 0. Phương pháp & rào cản re-entry trong engine (soi code, không đoán)

- Trades: `gbx08_enriched2.csv` (1378, pnl 128.497u khớp DB; rallied = post_max_c≥5%/20bar = 663).
  OHLCV: market.duckdb universe_id=8 v2 (61 mã). Cost model verify 20/20 trade: pnl = exit/entry −1 −0.004.
- Engine (engine.py) sau exit slot mã đó TRỐNG ngay; muốn có trade mới cần: (1) buy-signal mới của head,
  (2) qua entry_gate `upleg_abovema20` (zigzag causal 6% up-leg & close≥MA20 — áp UPSTREAM lúc sinh signal),
  (3) qua market-weak gate (z5≤−1.1 lb60 ∪ cumret5≤−4%), (4) cooldown `reentry_cooldown_bars=4` CHỈ sau exit LỖ,
  (5) pullback limit 4.5% conv-scaled khớp trong 40 bar. `reentry_max_premium_pct`=off, `resume_reentry`=off.
  → Rào cản config thuần duy nhất là cooldown 4 bar sau exit lỗ; còn lại là head/gate/fill.
- Signal proxy cho chẩn đoán: signals.csv frame serving 2643 (106k buy-bar, 79% trades trùng gb_x08;
  3/61 mã AAS AAV BCG không có proxy) + eff_depth từ depths.parquet. Caveat ghi ở §4.

## 1. Phần tự-tái-mua ĐANG CÓ — hệ đã mua lại 79% cohort trong 40 bar

| cửa sổ sau exit | n có trade mới cùng mã | % | u của re-entry |
|---|---|---|---|
| ≤10 bar | 132 | 19.9% | +16.1 |
| ≤20 bar | 316 | 47.7% | +30.2 |
| **≤40 bar** | **524** | **79.0%** | **+53.2** |
| 41–90 bar | +112 | (cộng dồn 96%) | +14.9 |
| >90 / không bao giờ | 27 | 4.1% | — |

- Median gap: signal mới sau exit **9 bar**, fill sau **17 bar** — đúng nhịp "limit kế tiếp cùng mã"
  của RUNAWAY_AUTOPSY (median 20 ngày). Re-entry ≤40 bar: WR 58.6%, **+0.1015/lệnh ≈ 1.09× pool** (+0.0932).
- Theo năm (u re-entry ≤40): 2020 +14.4 · 2021 +20.0 · 2022 +4.6 · 2023 +6.5 · 2024 +1.0 · 2025 +6.5 · 2026 +0.2.
  **≥2022: n=346/449 (77%), +18.8u.**
- Kết luận đo: cơ chế hiện tại KHÔNG bỏ rơi cohort rallied — nó tái-vào 96% trường hợp trong ≤90 bar
  và đã thu +68u qua các trade kế tiếp. "Re-entry gap" thật sự chỉ còn 139 lệnh không fill ≤40 bar.

## 2. Gap còn lại (139 lệnh, 21%) — cận trên qua pullback 4.5% chuẩn = **ÂM**

Giả lập lạc quan nhất còn đúng cơ chế: oracle-signal ngay bar exit+1 (bỏ qua head + mọi gate),
limit = close×(1−4.5%), fill thực theo low≤limit trong 40 bar, exit = force-gate replication
(dl12/nonbull-bma20p2/lowbreadth, suppress mkt-drop — đúng bộ closer 96% lệnh thật), cost engine:

- **73/139 không bao giờ fill** (giá chạy thẳng không chỉnh 4.5% — runaway con, đúng bucket (c) mở rộng).
- 66 fill: hold median **3 bar**, WR 40.9% → **pnl_h = −0.90u toàn kỳ, −0.72u ≥2022** (âm 5/7 năm, không năm nào > +0.02).
  Cơ chế: các exit này do downleg12/nonbull xác nhận; fill −4.5% rơi giữa down-leg nên force-gate
  **bắn lại ngay sau 2–3 bar** — mua lại rồi bị chính exit stack xả. Muốn giữ qua đó = đổi exit
  = chính họ lever "giữ lâu hơn" đã âm ≥2022 ở EXIT_ATTRIBUTION_MAP §5.
- **Occupancy: 0 lệnh thật bị chặn** (0.00u) — theo định nghĩa nhóm này không có trade thật trong 40 bar,
  và hold hypothetical quá ngắn để đè lên fill 41–90 bar. Slot mã trống suốt cửa sổ rally → khoản trừ
  occupancy không cứu được con số: NET = −0.90u.
- Trần oracle tuyệt đối của nhóm (mua đúng giá exit, bán đúng đỉnh close 20 bar — bất khả thi):
  Σ post_max_c = +14.2u gross / 139 lệnh, rải đều 7 năm (18–27 lệnh/năm), median rally chỉ 7.5%
  — nhỏ hơn round-trip (4.5% đệm + cost) + rủi ro force re-exit; đó là lý do cơ chế thật −0.9u.

## 3. Chẩn đoán vì sao 139 lệnh lọt lưới (proxy 2643, gate 2783 replicate) × năm

| nguyên nhân | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | ALL |
|---|---|---|---|---|---|---|---|---|
| (a1) head im, gate mở phần lớn cửa sổ | 1 | 3 | 8 | 6 | 4 | 6 | 6 | 34 |
| (b1) gate `upleg_abovema20` đóng ~cả cửa sổ (veto chân sóng) | 0 | 2 | 4 | 0 | 0 | 3 | 2 | 11 |
| (b2) có signal nhưng rơi đúng ngày gate/market-weak | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 2 |
| (c) signal + gate mở nhưng pullback không khớp (runaway con) | 4 | 0 | 0 | 2 | 1 | 0 | 6 | 13 |
| **(d) cooldown/config engine chặn** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |
| (e) proxy 2643 bắn + fill được nhưng head 2783 im (khác frame) | 12 | 12 | 8 | 9 | 14 | 14 | 4 | 73 |
| không có proxy (AAS/AAV/BCG) | 0 | 1 | 0 | 0 | 2 | 3 | 0 | 6 |

- **(d) = 0/139**: cooldown 4-bar-sau-exit-lỗ KHÔNG chặn trường hợp nào (cửa sổ 40 bar >> 4 bar,
  và phần lớn exit rallied là exit lãi). `reentry_max_premium`/`resume_reentry` đang off nên không chặn gì.
  → **Không tồn tại rào cản config nào đang giữ tiền.**
- Bucket lớn nhất (e, 53%) là chuyện HEAD (model 2783 không bắn nơi head 2643 bắn) — không sửa được bằng
  knob engine; kể cả gộp e+a1+c làm "signal mới hoàn hảo" thì §2 đã cho thấy cơ chế fill+exit chỉ thu −0.9u.
- b1+b2 = 13 lệnh: veto chân-sóng thật, nhưng thả gate trong down-leg = đúng lever `downleg_skip_bull`-họ
  đã GHOST 2021 / âm ≥2022 (EXIT map §5).

## 4. Caveat trung thực

- Proxy signal là frame 2643 (79% trùng trade): ranh giới a1↔e không tuyệt đối; nhưng verdict không phụ
  thuộc vào ranh giới đó vì cận trên §2 bỏ qua hẳn head (oracle-signal) mà vẫn âm.
- Oracle-signal bar exit+1 là bias LẠC QUAN (hệ thật còn mất thời gian chờ head+gate) — cận trên thật
  chỉ thấp hơn −0.9u.
- Occupancy first-order cùng phương pháp e_sim/rw_ (không sim domino sau exit hypothetical); với NET âm
  và 0 blocked, sai số này không đổi dấu kết luận.

## VERDICT — DỪNG, KHÔNG CÓ LEVER CONFIG, KHÔNG ĐỐT PROBE (điều kiện bước 4: gap < +2u ≥2022)

1. Re-entry đã **bão hòa tự nhiên**: 79% cohort rallied có trade mới cùng mã ≤40 bar (+53.2u; ≥2022 +18.8u,
   1.09× pool/lệnh), 96% ≤90 bar. Đây chính là mặt trade-level của cơ chế "limit kế tiếp monetize runaway".
2. Gap còn lại 139 lệnh: cận trên config-mechanism (pullback 4.5%, exit stack giữ nguyên, sau occupancy)
   = **−0.9u toàn kỳ / −0.72u ≥2022** — không những < +2u mà còn ÂM; nguyên nhân (d) = 0 lệnh.
3. Không clone 2783, không knob mới (mọi knob "vào sớm trong down-leg" hoặc "giữ qua force-gate" đều là
   lever đã bác ở EXIT_ATTRIBUTION_MAP §5 / RUNAWAY_AUTOPSY). gb_x08 giữ nguyên top-1.
4. Hướng duy nhất còn dư địa (ngoài scope config): head entry mới bắn được trong 40 bar hậu-exit nơi
   2643 bắn mà 2783 im (bucket e, 73 lệnh rải đều các năm) — nhưng giá trị bị chặn trên bởi chính
   cơ chế fill/exit (−0.9u với oracle) nên chỉ có nghĩa nếu đi kèm kiến trúc multi-position/exit khác,
   khớp kết luận wavestart + EXIT map.
