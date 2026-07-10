# CONV SCALING RESULTS — knob "conviction input mở rộng" (snr_sym + dist_MA20 ex-ante)

*2026-07-10. Lever #1 của `ENTRY_STRUCTURE_MAP.md`: điều hòa LIÊN TỤC conviction theo z(snr_sym) và
z(dist_MA20) tại bar tín hiệu. Base = gb_x08 (template 2783, seed-42 = 735.0 composite / 1378 trades).
Scripts: `convscale/cv_*.py` cùng thư mục.*

---

## 1. Cơ chế conviction hiện tại (đọc trước khi code)

Vị trí: `stock_ml/src/backtest/engine.py` — precompute `pb_conv_mult` (~:1396-1515), apply (~:1737).

**Nó scale CÁI GÌ:** chỉ scale **DEPTH của pullback-limit**, không scale threshold, không scale notional.
Mỗi bar tín hiệu có `depth = entry_pullback_pct * pb_conv_mult[i]`, với
`pb_conv_mult = clip(1 − conv_k · strength, conv_floor, 1)` — strength cao ⇒ limit NÔNG hơn
(mua gần giá tín hiệu, không đợi dip sâu); strength thấp/warmup ⇒ full depth 4.5%.

**Config 2783 (gb_x08):** `conv_scale=true, use_combo=true, k=0.4, floor=0.5,
combo_w=(rsi .25, ext .20, eff .30, rpos .25), head_w=0.5, vol_z=1.0 (lb 40)`.
⇒ `strength = 0.5·price_combo + 0.5·sigmoid(escore_z)`; mult ∈ [0.6, 1.0] thực tế
(k=0.4 ⇒ strength=1 → mult 0.6; floor 0.5 không chạm). Gate phụ: bar có realized-vol z > 1.0
⇒ ép mult=1 (full depth); warmup RSI/MA20 NaN ⇒ mult=1.

**Input hiện tại:** price_combo = trộn tuyến tính 4 kênh chuẩn hóa [0,1]: RSI14 (rescale từ
conv_rsi_lo=50), ext = close/MA20−1 (cap 0.10), efficiency 10-bar, range-position 20-bar; cộng kênh
ML head `sigmoid(escore_z)`. Các kênh mở rộng có sẵn (mfe/score3/rs/leg/vol_conf/vwap) đều theo cùng
pattern: `blend = (1 − Σw)·price_combo + Σ wᵢ·channelᵢ`, channel ∈ [0,1], **KHÔNG clip residual** —
nếu Σw > 1 thì trọng số price_combo ÂM (lưu ý cho cell snr_w=0.6: 1−0.5−0.6 = −0.1).

**Z-norm chuẩn của hệ:** rolling per-symbol `(x − mean252)/(std252+1e-9)`, `min_periods=60`
(escore_z :1140, emfe_z :1133, s3z :1164 v.v.). Kênh ML đưa z vào strength qua `sigmoid(z)`
(NaN → 0.5 trung tính).

**Knob mới (thiết kế):** `entry_pullback_conv_snr_w`, `entry_pullback_conv_dma20_w` (default 0.0 = off).
- `snr_sym` = mean(logret,20)/std(logret,20) tại bar hiện tại (đúng định nghĩa em_01/em_04, causal).
- `dma20` = close/MA20−1 (chính là `_ext2` đã có trong block conv).
- Cả hai z-norm 252/60 per-symbol → `sigmoid(z)` → blend với trọng số w theo đúng pattern kênh cũ,
  chỉ hoạt động trong nhánh `use_combo` (như mọi kênh mở rộng khác).
- Pipeline: KHÔNG cần pop key (là field EngineConfig thật, không phải key pipeline-level).

**Cảnh báo reshuffle:** knob này ĐỔI GIÁ limit của pending book (nông/sâu hơn theo snr/dma20) —
đúng họ cơ chế đã giết bot_shallow/anchor/deepen. Khác biệt duy nhất: nó đi qua phép điều hòa
conviction có sẵn (đã được walled bởi vol_z gate + floor). Mỗi probe PHẢI soi chữ ký reshuffle
(join symbol+entry_date với gb_x08 s42).

## 2. Parity

**PASS.** Implement: 2 field mới `entry_pullback_conv_snr_w` / `entry_pullback_conv_dma20_w`
(engine.py, default 0.0), precompute z trong block use_combo, blend `w·sigmoid(z)` cùng chỗ các
kênh head/rs/leg. Clone 2783 + 2 key = 0.0 (`cv_parity`, tmpl mới) seed-42:
**comp = 735.0 / 1378 trades — đúng byte**; reshuffle join: 1378/1378 lệnh trùng, 100% same-price,
dPnL = 0, n_changed = 0. Regression `test_champions.py`: 1 passed / 12 skipped (chuẩn).

## 3. Sweep seed-42

(chưa chạy)
