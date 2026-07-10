# P1_RANKING_E1 — Screen quyết định tuyến CROSS-SECTIONAL RANKING làm lõi chọn-mã

Ngày: 2026-07-10. Scripts: `p1_00_dataset.py` → `p1_01_train.py` → `p1_02_metrics.py`
→ `p1_03_sensitivity.py` → `p1_04_robust.py`. Artefacts: `p1_dataset.parquet`,
`p1_preds.parquet`, `p1_metrics.json`, `p1_ic_table.csv`, `p1_spread_table.csv`,
`p1_fill_table.csv`, `p1_sensitivity.csv`, `p1_robust.csv`, `p1_fill_monthly.csv`.

**Kill-criteria đăng ký trước (NEW_FAMILY_OPTIONS §P1):** spread top-20 net-cost ≥2022 ≤ 0
∨ Rank-IC < 0.03 ∨ fill-rate limit 4.5%/40 < ~50%.

## 0. Thiết kế

- **Universe = 61 mã champion** (manifest bundle gb_x08 model-side) — KHÔNG mở universe.
  Lưu ý: con số "488" trong NEW_FAMILY_OPTIONS là universe *breadth*; universe GIAO DỊCH
  của champion (và của leaderboard) là 61 mã. Top-20 = ~1/3 universe.
- **Features = đúng bộ champion entry** `entry_lvup126_recov` (45 per-symbol, tính bằng
  `FeatureResolver.from_catalog()` — cùng code pipeline) + 2 breadth 488-univ
  (`breadth_pct_above_ma50`, `breadth_adv_pct`, hàm `_load_market_breadth` pipeline) = 47 cột.
  (Ghi chú: kế hoạch nói "58 feature tay"; bộ thật của champion entry là 47 cột như trong
  `feature_spec.json` của bundle.)
- **Label MỚI với họ champion**: `fwd20_dm` = forward return 20-bar close→close DEMEANED
  THEO NGÀY trên universe (loại beta thị trường); sensitivity `fwd10_dm`. Máy kiểm:
  max |mean per date| = 2.9e-17.
- **Walk-forward theo năm**: train ≤ y−1 với **purge label-window** (mọi bar mà label
  chạm tới phải < test_start), test y ∈ {2021..2026H1}. Data 2017→2026-06-16
  (market.duckdb — cùng nguồn pipeline).
- **Models**: (a) LGBM regression trên fwd20_dm; (b) LGBM **lambdarank** (group=ngày,
  relevance=quintile per-date); (c) baseline momentum-rank không-ML (`ret_20d`).
  Seeds 42/7/99; "mean" = trung bình per-date pct-rank của 3 seed.
- **Cost model leaderboard**: roundtrip 0.70% (comm 2×0.15% + slip 2×0.15% + tax 0.10%).
  Thực thi lag 1 bar (vào close[t+1]) — không lookahead.

## 1. Rank-IC OOS theo năm (Spearman per-date, pred vs fwd20_dm)

| model | 2021 | 2022 | 2023 | 2024 | 2025 | 2026H1 | **pooled ≥2022** | t_cons |
|---|---|---|---|---|---|---|---|---|
| lgbm_reg s42 | .035 | .012 | .031 | .011 | .054 | .012 | **.026** | 0.85 |
| lgbm_reg s7 | .024 | .005 | .048 | .017 | .052 | .001 | **.028** | 0.92 |
| lgbm_reg s99 | .027 | .008 | .038 | .009 | .052 | .006 | **.025** | 0.84 |
| lgbm_reg mean | .029 | .008 | .040 | .013 | .054 | .008 | **.027** | 0.89 |
| **lgbm_rank s42** | .047 | .024 | .003 | .042 | .080 | .061 | **.039** | 1.37 |
| **lgbm_rank s7** | .039 | .026 | −.005 | .038 | .075 | .059 | **.036** | 1.24 |
| **lgbm_rank s99** | .052 | .042 | .015 | .031 | .080 | .055 | **.043** | 1.51 |
| **lgbm_rank mean** | .048 | .030 | .003 | .037 | .079 | .059 | **.039** | 1.36 |
| lgbm_reg fwd10_dm s42 | .022 | −.003 | .053 | .017 | .076 | .021 | .034 | 1.15 |
| momo ret_20d (không-ML) | .051 | −.032 | .028 | −.006 | .021 | −.002 | **.002** | 0.07 |

**Null shuffle-within-date ≥2022** (1086 ngày): analytic ±0.0077, empirical 200 hoán vị
[−0.0083, +0.0079].

- **Lambdarank vượt bar 0.03 ở cả 3 seed** (0.036–0.043), gấp ~5× null hi. Regression trên
  label demeaned KHÔNG vượt bar (0.025–0.028) — objective ranking thật là thứ mang tín hiệu.
- Momentum-rank thô chết hẳn (0.002) ⇒ phần ML là thật, không phải momentum trá hình.
- Điểm yếu duy nhất theo năm: 2023 của lambdarank ~0 (regression lại khỏe 2023 — hai
  objective bắt lát khác nhau).

## 2. Spread net-cost + turnover/hysteresis

### 2.1 Thiết kế gốc — rebalance TUẦN (5 bar), K=20 (pooled ≥2022, ann.)

| model \| scheme | turnover/reb | top net | uni EW | bottom | **top−bot net** | **top−uni net** |
|---|---|---|---|---|---|---|
| reg \| plain | .384 | +1.5% | +8.3% | +1.4% | +0.2% | **−6.8%** |
| reg \| hyst(20/30) | .248 | +6.0% | +8.3% | +1.4% | +4.6% | **−2.3%** |
| rank \| plain | .339 | +3.5% | +8.3% | +0.3% | +3.2% | **−4.8%** |
| rank \| hyst(20/30) | .215 | +6.7% | +8.3% | +0.3% | +6.4% | **−1.6%** |
| momo \| hyst | .200 | +9.1% | +8.3% | +0.6% | +8.5% | +0.8% |

Rebalance tuần **chết vì cost**: gross spread top−uni của rank/hyst = +6.1%/năm nhưng
turnover 0.215/tuần × 0.7% ≈ −7.8%/năm. Hysteresis (vào top-20, ra khi rank>30) cắt
turnover 0.339→0.215 và cải thiện spread ~+3.2 điểm — đúng hướng nhưng chưa đủ.

### 2.2 Sensitivity K × tần suất (p1_sensitivity.csv, ≥2022, top−uni NET)

Điểm gãy là **tần suất rebalance khớp horizon label (20 bar)**:

| rank_mean, hyst | reb=5 (tuần) | reb=10 | **reb=20 (tháng)** |
|---|---|---|---|
| K=5 | −8.5% | −3.0% | **+11.8%** |
| K=10 | −7.2% | −1.9% | **+4.3%** |
| K=15 | −3.8% | +0.4% | **+5.9%** |
| K=20 | −1.6% | +1.8% | **+5.6%** |

### 2.3 Bài kiểm tra phá: quét 20 phase bắt đầu × 3 seed (reb=20, hyst; p1_robust.csv)

| model | K | min | p25 | **median** | p75 | max | % phase > 0 |
|---|---|---|---|---|---|---|---|
| rank mean | 5 | −9.6% | −2.4% | +2.8% | +11.6% | +18.6% | 65% |
| rank mean | 10 | −0.2% | +1.8% | **+3.9%** | +6.5% | +13.8% | **95%** |
| rank mean | 20 | −0.0% | +2.1% | **+3.4%** | +5.1% | +8.6% | **95%** |
| rank s42 | 20 | +0.0% | +1.8% | +3.3% | +5.2% | +8.2% | 100% |
| rank s7 | 20 | −0.7% | +2.3% | +3.3% | +4.4% | +7.3% | 95% |
| rank s99 | 20 | +1.3% | +2.9% | +4.0% | +5.6% | +8.1% | 100% |

- **K=5 tháng (+11.8%) là phase-luck** — median chỉ +2.8%, 35% phase âm. Không tin.
- **K=10–20 tháng là thật**: median +3.3–4.9%/năm net, 95–100% phase dương, cả 3 seed.
  Per-year (K=20, offset 0): 2022 +13.2%, 2023 −3.6%, 2024 −0.6%, 2025 +11.8%,
  2026H1 +11.0% — dương ở năm gấu 2022, mỏng 2023/24.
- Kết luận spread: **top−uni net ≥2022 > 0** với thực thi đúng horizon (tháng) — tiêu chí
  kill KHÔNG kích hoạt; nhưng biên chỉ ~+3–4%/năm ở cấu hình robust.

## 3. Fill-rate pullback-limit 4.5%/40 trên cohort vào top-K (điều kiện ghép execution)

Cohort = tên MỚI vào membership (hysteresis), limit = close(ngày rank)×0.955, hiệu lực
40 bar — đúng champion. Số ≥2022, monthly (p1_fill_monthly.csv; weekly tương tự p1_fill_table.csv):

| K | n orders | **fill-rate** | ngày-đến-fill (med) | fwd20 net TỪ GIÁ FILL | fwd40 net | cohort vào thẳng close (a40) | **phần KHÔNG fill (a40)** |
|---|---|---|---|---|---|---|---|
| 5 | 187 | 64.2% | 5 | −0.3% | +0.7% | +1.9% | **+12.8%** |
| 10 | 310 | 67.7% | 4.5 | −0.6% | +0.2% | +2.0% | **+14.1%** |
| 20 | 476 | 68.1% | 7 | +0.4% | +2.4% | +2.3% | **+12.9%** |

- Fill-rate 64–68% ≥ 50% ⇒ tiêu chí kill số-học KHÔNG kích hoạt.
- **NHƯNG adverse selection nặng đúng như rủi ro (a) đã dự báo**: phần fill được limit
  −4.5% có forward return ≈ 0 (−0.6%…+2.4%/40 bar net); phần KHÔNG BAO GIỜ pullback đủ
  4.5% mới mang alpha (+12.9…+14.1%/40 bar). Limit discount đảo dấu selection alpha:
  rank-top + rơi 4.5% trong vài ngày = tín hiệu gãy, không phải chiết khấu.
- ⇒ **Ghép nguyên tầng thực thi pullback của champion vào tuyến ranking là CHẾT về kinh
  tế** (dù pass số-học). Án "bỏ pullback −237" KHÔNG áp ở đây: án đó xử tín hiệu timing
  của họ champion; với tín hiệu membership ranking, E1 đo trực tiếp chiều ngược lại.
  Tầng thực thi đúng cho tuyến này = vào thẳng close_next (đã tính đủ cost trong §2).

## 4. Overlap với champion (gb_x08 s42, 1198 entry 2021→)

| membership | % entry champion nằm trong top-K tại rebalance gần nhất | % slot top-K champion đang giữ vị thế |
|---|---|---|
| top-20 tuần (rank) | 34% (theo năm 31–38%) | 39% |
| top-20 tuần (reg) | 36% | 40% |
| top-20 tuần (momo) | 37% | 54% |
| top-10 tháng (rank) | 20% | 35% |
| top-5 tháng (rank) | 9% | 36% |

Overlap ~1/3 — **thấp hơn hẳn ngưỡng re-skin 70%**. Kết hợp spread dương ⇒ nguồn alpha
selection tương đối là THẬT và KHÁC champion (champion trả lời timing tuyệt đối; 2/3 số
mã ranking chọn champion không hề vào lệnh cùng lúc).

## 5. Verdict theo kill-criteria

| Tiêu chí | Kết quả | Trạng thái |
|---|---|---|
| Rank-IC ≥2022 < 0.03 | lambdarank .036–.043 (3 seed, null ±.008) | **PASS** (regression .025–.028 = fail riêng nó; objective phải là lambdarank) |
| Spread top-20 net ≥2022 ≤ 0 | tuần: −1.6% (fail); **tháng khớp horizon: +3.4% median, 95–100% phase, 3 seed** | **PASS có điều kiện** (cadence tháng + hysteresis) |
| Fill-rate < 50% | 64–68% | PASS số-học; **NHƯNG alpha nằm ở phần không-fill ⇒ pullback-pairing chết** |

**VERDICT: GO — có điều kiện, kỳ vọng được kiểm soát.** Không kill: nguồn alpha
selection tương đối tồn tại, trên null rõ, không re-skin champion. Ba điều kiện sống
E1 rút ra: (1) objective = **lambdarank** (không phải regression demeaned); (2) cadence
= **tháng** (khớp horizon 20-bar) + hysteresis K/1.5K, K=10–20; (3) execution = **vào
thẳng close_next, KHÔNG pullback-limit** — đây là điểm khác kiến trúc lớn nhất với dự
kiến ban đầu ("giữ nguyên tầng pullback").

**Kỳ vọng trung thực cho E2**: biên robust chỉ +3–4%/năm trên nền universe EW ~+8–12%
⇒ top-20 tháng net ~17%/năm. Champion ≥2022 = 226 trades/năm × +5.6% net/trade (hold
24d). Tuyến rank ĐỨNG MỘT MÌNH gần như chắc chắn KHÔNG chạm trần pure-ML 625.8 ngay;
giá trị thực tế hơn nằm ở E3 (rank làm selector ứng viên cho timing head, additive).
E2 vẫn đáng chạy vì rẻ và vì composite engine là thước duy nhất so công bằng.

### Thiết kế E2 (signal-adapter qua engine, so leaderboard công bằng)

1. Strategy mode mới `xsec_rank_topk` (clone template append-only, knob gated
   default-off + golden parity đường cũ): mỗi 20 bar, LGBMRanker (lambdarank, quintile
   per-date, bộ feature `entry_lvup126_recov`+breadth, WF yearly như mọi template) rank
   universe 61; signal BUY khi mã vào top-K (K=20 chính, 10 sensitivity), SELL khi rớt
   khỏi top-1.5K.
2. Engine config: `entry_pullback_pct=null` (vào close_next — theo phát hiện §3),
   giữ hard-stop/washout force-exit tối thiểu (`exit_force_gate` downleg12 giữ nguyên
   làm phanh sự cố), exit chính = signal rớt membership; cost mặc định leaderboard.
3. Chạy multi-seed 42/7/99/555 + chuẩn ≥2022 bắt buộc; đăng ký leaderboard kể cả xấu.
   Bar giai đoạn: composite ≥ 625.8 (trần pure-ML) trước khi tối ưu bất kỳ knob nào;
   nếu <400 sau multi-seed → đóng tuyến đứng-một-mình, chuyển thẳng giá trị sang E3
   (rank = nguồn ứng viên additive cho head champion, không phải gate).
4. Riêng 2023 (năm lambdarank ~0): theo dõi như lát regime bắt buộc; nếu E2 âm tập trung
   2023 thì đó là chỗ regression-objective bù (ensemble 2 objective là knob E3, không
   phải E2).

### Caveat ghi hồ sơ

- Universe 61 mã cố định (survivorship như mọi số leaderboard) — so sánh TƯƠNG ĐỐI với
  champion là công bằng; đừng đọc absolute return.
- EW top-K close→close, lag 1 bar, không model được thanh khoản/limit-lock ngày vào
  (engine E2 sẽ xử phần này — có thể ăn thêm vài chục bps).
- Spread tháng đo trên 54 rebalance × 20 phase — con số median ±~2%/năm là noise floor;
  không tối ưu K/cadence thêm trên chính dữ liệu này (đã cố định K=20/tháng cho E2 từ
  phân tích robust, không cherry-pick cell tốt nhất).
