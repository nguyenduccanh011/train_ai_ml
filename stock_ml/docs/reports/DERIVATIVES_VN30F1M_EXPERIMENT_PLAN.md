# Chuẩn hoá kế hoạch thử nghiệm phái sinh VN30F1M (1H)

Ngày chuẩn hoá: 2026-05-06  
Phạm vi: `stock_ml` + dữ liệu `portable_data/derivatives_ai_dataset`

## 1. Mục tiêu chuẩn

- Tập trung tìm champion đầu tiên cho `VN30F1M` khung `1H`.
- Dùng mô hình cổ phiếu làm baseline transfer, không trộn dữ liệu cổ phiếu vào phase đầu.
- Chuẩn hoá đầu-cuối: dữ liệu -> config market -> matrix -> leaderboard -> chọn champion.

## 2. Thuật ngữ chuẩn (để tránh lệch ngữ nghĩa)

- `bundle`: tên batch thí nghiệm, tương ứng thư mục dưới `results/experiments/<bundle>/...`.
- `run_name`: tên cấu hình đã expand đầy đủ tham số.
- `run_id`: định danh duy nhất `bundle/run_name#config_hash8`.
- `baseline`: cấu hình đối chiếu được khai báo trong `stock_ml/config/leaderboard.yaml`.

## 3. Trạng thái hiện tại (đã xác minh)

### 3.1 Leaderboard cổ phiếu (`vn_stock`)

Top hiện tại (theo `stock_ml/results/leaderboard/by_market/vn_stock/leaderboard.csv`):

| Rank | Bundle | Composite | Total PnL | Trades | Ghi chú |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `v22_exit_ablation_round42` | 637.8 | 15768.62 | 1262 | Bản ghi đầy đủ metadata (`ohlcv_daily`, `1D`) |
| 2 | `v22_exit_ablation_round38` | 637.8 | 15768.62 | 1262 | Bản ghi cũ còn thiếu metadata `schema/timeframe` |
| 3 | `v22_exit_ablation_round42` | 637.3 | 15719.12 | 1266 | - |

Baseline fairness hiện cấu hình là `v22_exit_ablation_round42` trong `stock_ml/config/leaderboard.yaml`.

### 3.2 Dữ liệu phái sinh

Từ `portable_data/derivatives_ai_dataset/dataset_manifest.json`:

- `dataset_version`: 2
- `symbols`: `VN30F1M`, `VN30F2M`
- `timeframes`: `1m`, `5m`, `15m`, `30m`, `1H`, `1D`
- `total_rows`: 1,025,032
- Trục chính để tối ưu: `VN30F1M` - `1H` (9,603 rows)

### 3.3 Market profile `vn_derivatives`

Theo `stock_ml/config/markets/vn_derivatives.yaml` (trạng thái hiện hành):

- `market_type`: `futures_contract`
- `data.schema`: `ohlcv_futures_1h`
- `data.default_timeframe`: `1H`
- `execution.pnl_mode`: `futures_contract`
- `execution.contract_multiplier`: `100000`
- `execution.commission`: `0.0004`
- `execution.slippage`: `0.0005`
- `execution.short_enabled`: `false`
- `symbols.default_list`: `VN30F1M`, `VN30F2M`

## 4. Nguyên tắc triển khai

- Không train chung toàn bộ dữ liệu cổ phiếu với phái sinh ở phase đầu.
- Phase đầu chỉ tối ưu `VN30F1M` `1H`, sau đó mới thêm `VN30F2M` phụ trợ.
- Walk-forward phải giữ time order, không shuffle, không leak tương lai.
- Champion chọn theo risk-adjusted metrics, không chọn chỉ theo PnL.

## 5. Cấu trúc thử nghiệm chuẩn

### Nhánh A: Transfer từ cổ phiếu (baseline)

- Mục tiêu: đo khả năng chuyển giao cấu trúc `v22` sang futures.
- Khởi điểm: `features=leading/leading_v2`, `entry_model=random_forest/lightgbm`, `target=early_wave_v2`.

### Nhánh B: Derivatives-only (nhánh chính)

- Train/test chỉ trên `VN30F1M` `1H`.
- Bắt đầu từ bundle nền nhỏ, sau đó mở rộng dần.

### Nhánh C: Auxiliary `VN30F2M`

- Chỉ chạy sau khi Nhánh B đã có baseline ổn định.
- Mục tiêu: cải thiện tính ổn định qua regime, không hy sinh chất lượng trên `VN30F1M`.

## 6. Walk-forward đề xuất cho VN30F1M 1H

- Split A (baseline): `train=2 năm`, `test=6 tháng`, `first_test=2020-01-01`.
- Split B (nhạy regime): `train=1 năm`, `test=3 tháng`, `first_test=2020-01-01`.
- Split C (đối chiếu style cổ phiếu): `train=4 năm`, `test=1 năm`, `first_test=2023-01-01`.

## 7. Lộ trình phase chuẩn

- Phase 0: data quality + manifest + schema sanity.
- Phase 1: baseline run 1-16 config trên `VN30F1M 1H`.
- Phase 2: transfer baseline từ cổ phiếu.
- Phase 3: derivatives-only mở rộng (medium/full).
- Phase 4: robustness (split B/C + stress cost).
- Phase 5: auxiliary `VN30F2M`.
- Phase 6: multi-timeframe (15m/5m) nếu 1H đã có edge rõ.

## 8. Trạng thái thực hiện

### Phase 0 — data quality + manifest + schema sanity

Trạng thái: hoàn tất kiểm tra tối thiểu ngày 2026-05-06.

Đã thêm test `test_vn_derivatives_phase0_manifest_and_schema_sanity` trong `stock_ml/tests/data/test_loader_layouts.py` để xác minh:

- Market profile `vn_derivatives` trỏ đúng dataset `portable_data/derivatives_ai_dataset`.
- `dataset_manifest.json` có `dataset_version=2`, symbol `VN30F1M`, timeframe mặc định `1H`.
- Entry manifest cho `VN30F1M` `1H` có `9603` rows và file `symbol=VN30F1M/timeframe=1H/data.csv`.
- Loader đọc được đúng số dòng, timestamp tăng dần, không duplicate timestamp.
- Các cột bắt buộc không null; quan hệ OHLC hợp lệ; `volume >= 0`.

Lệnh đã chạy:

```bash
PYTHONPATH=stock_ml pytest stock_ml/tests/data/test_loader_layouts.py
```

Kết quả: `2 passed`.

### Phase 1 — baseline run VN30F1M 1H

Trạng thái: hoàn tất baseline run 8 config ngày 2026-05-06.

Đã chạy bundle `derivatives_vn30f1m_phase1_smoke` với 8 cấu hình derivatives-only trên `vn_derivatives`, timeframe mặc định `1H`.

Top 3 theo `stock_ml/results/experiments/derivatives_vn30f1m_phase1_smoke/ranking.csv`:

| Rank | Features | Entry model | Composite | Total PnL | Win rate | Max DD | Trades |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `leading_v2` | `random_forest` | 56.0 | 99.43 | 65.45 | 6.82 | 55 |
| 2 | `leading` | `random_forest` | 43.1 | 95.01 | 63.16 | 8.13 | 57 |
| 3 | `leading_v3` | `random_forest` | 37.8 | 61.56 | 59.52 | 6.03 | 42 |

Baseline derivatives hiện chọn theo composite score:

- `bundle`: `derivatives_vn30f1m_phase1_smoke`
- `run_name`: `derivatives_vn30f1m_phase1_smoke_signals_features-leading_v2-signals_entry_model_type-random_forest-signals_target-earlyv2_fw21_g033125_l0165625-exit_model-exit_fw21_l03725-fusion-peak_dist_only`
- `config_hash`: `744abf05b7cf`

Lệnh đã chạy:

```bash
PYTHONPATH=stock_ml python -m stock_ml.scripts.cli run-matrix matrix/derivatives_vn30f1m_phase1_smoke --jobs 4 --resume --save-results
```

Nhận xét: 4/4 cấu hình `random_forest` có composite dương và PnL dương; nhóm `lightgbm` vẫn PnL dương nhưng composite thấp hoặc âm do win rate/drawdown kém hơn. Phase 2/3 nên lấy `leading_v2 + random_forest + early_wave_v2` làm baseline đối chiếu, chưa mở rộng `VN30F2M` hoặc multi-timeframe.

#### Xem kết quả Phase 1

Leaderboard derivatives đã được rebuild tại `stock_ml/results/leaderboard/by_market/vn_derivatives/`. Nếu mở trực tiếp `leaderboard.html` bằng `file://`, browser có thể chặn đọc JSON local do CORS và hiển thị `VN Derivatives leaderboard not found`. Cần serve bằng HTTP server:

```bash
cd stock_ml/visualization
python serve.py
```

Sau đó mở:

- `http://localhost:8080/leaderboard.html` để xem leaderboard, chọn market `VN Derivatives`.
- `http://localhost:8080/dashboard.html` để xem chart/trades.

Đã export top 3 derivatives configs sang dashboard bằng:

```bash
PYTHONPATH=stock_ml python -m stock_ml.scripts.cli export-matrix derivatives_vn30f1m_phase1_smoke --top-k 3
```

Các file dashboard đã tạo:

- `stock_ml/visualization/data_signals_features_leading_v2_signals_entr/VN30F1M.json`
- `stock_ml/visualization/data_signals_features_leading_signals_entry_m/VN30F1M.json`
- `stock_ml/visualization/data_signals_features_leading_v3_signals_entr/VN30F1M.json`

Các JSON này có `markers` buy/sell và `trades` để dashboard hiển thị điểm mua/bán, PnL, ngày entry/exit, holding days và exit reason.

Cập nhật dashboard ngày 2026-05-06:

- `dashboard.html` đã có market selector: `All Markets`, `VN Stock`, `VN Derivatives`.
- `VN Derivatives` lọc đúng 3 model top export mới: `deriv_v2_rf`, `deriv_ld_rf`, `deriv_v3_rf`.
- Đã export nến 1H đầy đủ cho `VN30F1M` vào `stock_ml/visualization/data_derivatives/VN30F1M.json` với `9603` candles.
- `manifest.json` đã có `base_data_dirs.vn_derivatives = data_derivatives` và metadata `market/schema/timeframe` cho model phái sinh.

Lệnh đã chạy để cập nhật dashboard:

```bash
python stock_ml/scripts/export_derivatives_ohlcv.py
PYTHONPATH=stock_ml python -m stock_ml.scripts.cli export-matrix derivatives_vn30f1m_phase1_smoke --top-k 3
python -m py_compile stock_ml/scripts/export_derivatives_ohlcv.py stock_ml/scripts/cli.py stock_ml/src/export/unified_export.py
node --check stock_ml/visualization/js/state.js && node --check stock_ml/visualization/js/ui.js && node --check stock_ml/visualization/js/chart.js && node --check stock_ml/visualization/js/app.js
```

### Phase 2 — transfer baseline từ cấu trúc v22 cổ phiếu

Trạng thái: hoàn tất transfer baseline 48 config ngày 2026-05-06.

Đã chạy bundle `derivatives_vn30f1m_phase2_transfer` để mở rộng grid theo cấu trúc `v22_exit_ablation_round42`, train/test trên `vn_derivatives`, timeframe mặc định `1H`.

Grid Phase 2:

- `features`: `leading`, `leading_v2`
- `entry_model`: `random_forest`, `lightgbm`
- `target`: 4 cấu hình `early_wave_v2` từ `v22_exit_ablation_round42`
- `exit_model`: 3 cấu hình `lightgbm` exit từ `v22_exit_ablation_round42`
- Tổng: `2 x 2 x 4 x 3 = 48` configs

Top 3 theo `stock_ml/results/experiments/derivatives_vn30f1m_phase2_transfer/ranking.csv`:

| Rank | Features | Entry model | Target | Exit | Composite | Total PnL | Win rate | Max DD | Trades |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `leading` | `random_forest` | `earlyv2_fw22_g033125_l0165625` | `exit_fw22_l03725` | 63.6 | 122.97 | 70.69 | 5.81 | 58 |
| 2 | `leading` | `random_forest` | `earlyv2_fw21_g0334375_l01671875` | `exit_fw22_l03725` | 63.2 | 121.01 | 67.27 | 6.82 | 55 |
| 3 | `leading` | `random_forest` | `earlyv2_fw22_g033125_l0165625` | `exit_fw21_l03725` | 58.8 | 115.47 | 68.52 | 5.81 | 54 |

Baseline derivatives mới theo composite score:

- `bundle`: `derivatives_vn30f1m_phase2_transfer`
- `run_name`: `derivatives_vn30f1m_phase2_transfer_signals_features-leading-signals_entry_model_type-random_forest-signals_target-earlyv2_fw22_g033125_l0165625-exit_model-exit_fw22_l03725-fusion-peak_dist_only`
- `config_hash`: `efff40293ca8`

So với Phase 1, top Phase 2 cải thiện rõ: composite `56.0 -> 63.6`, Total PnL `99.43 -> 122.97`, win rate `65.45 -> 70.69`, max drawdown `6.82 -> 5.81`, trade count `55 -> 58`.

Lệnh đã chạy:

```bash
PYTHONPATH=stock_ml python -m stock_ml.scripts.cli run-matrix matrix/derivatives_vn30f1m_phase2_transfer --jobs 4 --resume --save-results
PYTHONPATH=stock_ml python -m stock_ml.scripts.build_leaderboard rebuild
PYTHONPATH=stock_ml python -m stock_ml.scripts.cli export-matrix derivatives_vn30f1m_phase2_transfer --top-k 3
```

Kết quả xác minh:

- Full run tạo `48` dòng trong `ranking.csv`.
- Leaderboard rebuild có `vn_derivatives: 52` rows.
- Export dashboard tạo 3 model derivatives mới: `deriv_ld_rf`, `deriv_ld_rf_2`, `deriv_ld_rf_3`.

Nhận xét: transfer cấu trúc `v22` sang `VN30F1M 1H` có cải thiện thực tế so với baseline Phase 1. Top 10 vẫn nghiêng về `leading + random_forest`; `leading_v2` không còn là top sau khi mở rộng target/exit. Phase 3 nên lấy `leading + random_forest + earlyv2_fw22_g033125_l0165625 + exit_fw22_l03725` làm baseline chính, sau đó mở rộng derivatives-only quanh target/exit và kiểm tra robustness trước khi thêm `VN30F2M` hoặc multi-timeframe.

### Phase 3 — derivatives-only mở rộng quanh baseline

Trạng thái: hoàn tất grid mở rộng 48 config ngày 2026-05-06.

Đã chạy bundle `derivatives_vn30f1m_phase3_derivatives_only` để chỉ giữ nhánh đang thắng `leading + random_forest`, rồi quét hẹp quanh target/exit tốt nhất của Phase 2 trên `vn_derivatives`, timeframe mặc định `1H`.

Grid Phase 3:

- `features`: `leading`
- `entry_model`: `random_forest`
- `target`: 8 cấu hình `early_wave_v2` quanh forward window `21-23` và gain/loss threshold vùng `0.0325-0.0334375` / `0.01625-0.01671875`
- `exit_model`: 6 cấu hình `lightgbm` exit quanh forward window `21-22` và `loss_threshold=0.037-0.0375`
- Tổng: `1 x 1 x 8 x 6 = 48` configs

Top 3 theo `stock_ml/results/experiments/derivatives_vn30f1m_phase3_derivatives_only/ranking.csv`:

| Rank | Features | Entry model | Target | Exit | Composite | Total PnL | Win rate | Max DD | Trades |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `leading` | `random_forest` | `earlyv2_fw21_g0334375_l01671875` | `exit_fw22_l037` | 64.0 | 121.31 | 67.27 | 6.82 | 55 |
| 2 | `leading` | `random_forest` | `earlyv2_fw22_g033125_l0165625` | `exit_fw22_l03725` | 63.6 | 122.97 | 70.69 | 5.81 | 58 |
| 3 | `leading` | `random_forest` | `earlyv2_fw21_g0334375_l01671875` | `exit_fw22_l03725` | 63.2 | 121.01 | 67.27 | 6.82 | 55 |

Baseline derivatives mới theo composite score:

- `bundle`: `derivatives_vn30f1m_phase3_derivatives_only`
- `run_name`: `derivatives_vn30f1m_phase3_derivatives_only_signals_features-leading-signals_entry_model_type-random_forest-signals_target-earlyv2_fw21_g0334375_l01671875-exit_model-exit_fw22_l037-fusion-peak_dist_only`
- `config_hash`: `8b8f9d264fb5`

So với Phase 2, top Phase 3 chỉ cải thiện nhẹ composite `63.6 -> 64.0`, nhưng PnL thấp hơn một chút `122.97 -> 121.31`, win rate thấp hơn `70.69 -> 67.27`, max drawdown cao hơn `5.81 -> 6.82`. Cấu hình Phase 2 vẫn đứng rank 2 trong Phase 3 và cân bằng hơn về win rate/drawdown, nên chưa nên đổi champion vận hành chỉ dựa trên composite.

Lệnh đã chạy:

```bash
PYTHONPATH=stock_ml python -m stock_ml.scripts.cli run-matrix matrix/derivatives_vn30f1m_phase3_derivatives_only --jobs 4 --resume --save-results
PYTHONPATH=stock_ml python -m stock_ml.scripts.build_leaderboard rebuild
PYTHONPATH=stock_ml python -m stock_ml.scripts.cli compare-matrix results/experiments/derivatives_vn30f1m_phase3_derivatives_only --top 10
PYTHONPATH=stock_ml python -m stock_ml.scripts.cli export-matrix derivatives_vn30f1m_phase3_derivatives_only --top-k 3
```

Kết quả xác minh:

- Full run tạo `48` dòng trong `ranking.csv`.
- Leaderboard rebuild có `vn_derivatives: 88` rows.
- Export dashboard tạo 3 model derivatives mới: `deriv_ld_rf`, `deriv_ld_rf_2`, `deriv_ld_rf_3`.

Nhận xét: Phase 3 xác nhận vùng tốt tập trung quanh `leading + random_forest` và `exit_fw22`, nhưng cải thiện so với Phase 2 không đủ lớn để bỏ qua risk-adjusted tradeoff. Bước tiếp theo nên là Phase 4 robustness trên Split B/C và stress cost, so sánh tối thiểu top Phase 3 rank 1 với Phase 2/Phase 3 rank 2 trước khi thêm `VN30F2M` hoặc multi-timeframe.

### Phase 4 — robustness Split B/C

Trạng thái: hoàn tất robustness Split B/C ngày 2026-05-06. Stress cost đã chủ động bỏ qua ở bước này để không tạo thêm market profile hoặc thay đổi matrix expander.

Đã chạy 2 bundle robustness để kiểm tra 2 ứng viên chính từ Phase 2/3, theo cấu hình:

- `stock_ml/config/experiments/matrix/derivatives_vn30f1m_phase4_split_b.yaml`: `train_years=1`, `test_years=1`, `first_test_year=2020`, `last_test_year=2025`.
- `stock_ml/config/experiments/matrix/derivatives_vn30f1m_phase4_split_c.yaml`: `train_years=4`, `test_years=1`, `first_test_year=2023`, `last_test_year=2025`.

Lưu ý: splitter hiện tại chỉ hỗ trợ `test_years` dạng số năm nguyên, nên Split B là xấp xỉ conservative của đề xuất `test=3 tháng`.

Top kết quả Split B theo `stock_ml/results/experiments/derivatives_vn30f1m_phase4_split_b/ranking.csv`:

| Rank | Target | Exit | Composite | Total PnL | Win rate | Max DD | Trades |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `earlyv2_fw22_g033125_l0165625` | `exit_fw22_l03725` | 53.2 | 106.08 | 63.64 | 6.38 | 55 |
| 2 | `earlyv2_fw22_g033125_l0165625` | `exit_fw22_l037` | 48.9 | 103.51 | 63.64 | 8.41 | 55 |
| 3 | `earlyv2_fw21_g0334375_l01671875` | `exit_fw22_l037` | 43.8 | 98.64 | 61.11 | 9.20 | 54 |

Top kết quả Split C theo `stock_ml/results/experiments/derivatives_vn30f1m_phase4_split_c/ranking.csv`:

| Rank | Target | Exit | Composite | Total PnL | Win rate | Max DD | Trades |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `earlyv2_fw21_g0334375_l01671875` | `exit_fw22_l037` | 27.3 | 27.47 | 65.00 | 6.82 | 20 |
| 2 | `earlyv2_fw21_g0334375_l01671875` | `exit_fw22_l03725` | 27.2 | 27.26 | 65.00 | 6.82 | 20 |
| 3 | `earlyv2_fw22_g033125_l0165625` | `exit_fw22_l037` | 27.0 | 27.39 | 68.00 | 5.81 | 25 |

Lệnh đã chạy:

```bash
PYTHONPATH=stock_ml python -m stock_ml.scripts.cli run-matrix matrix/derivatives_vn30f1m_phase4_split_b --jobs 4 --save-results
PYTHONPATH=stock_ml python -m stock_ml.scripts.cli run-matrix matrix/derivatives_vn30f1m_phase4_split_c --jobs 4 --save-results
PYTHONPATH=stock_ml python -m stock_ml.scripts.build_leaderboard rebuild
PYTHONPATH=stock_ml python -m stock_ml.scripts.cli compare-matrix results/experiments/derivatives_vn30f1m_phase4_split_b --top 10
PYTHONPATH=stock_ml python -m stock_ml.scripts.cli compare-matrix results/experiments/derivatives_vn30f1m_phase4_split_c --top 10
```

Kết quả xác minh:

- Full run tạo `4` dòng trong mỗi `ranking.csv`.
- Leaderboard rebuild có `vn_derivatives: 96` rows.

Nhận xét: cấu hình Phase 2 top 1 (`earlyv2_fw22_g033125_l0165625 + exit_fw22_l03725`) vẫn là ứng viên cân bằng nhất: đứng đầu Split B, giữ drawdown thấp nhất trong Split C, win rate cao nhất trong Split C, và có trade count tốt hơn top Split C. Cấu hình Phase 3 top 1 thắng nhẹ ở composite Split C nhưng yếu hơn rõ ở Split B và có trade count thấp hơn trong giai đoạn 2023-2025. Champion vận hành hiện nên giữ cấu hình Phase 2 top 1, chưa cần mở rộng `VN30F2M` hoặc multi-timeframe trước khi xem thêm stress cost.

## 9. Lệnh CLI chuẩn (đang dùng được)

```bash
# Validate config
python -m stock_ml.scripts.cli validate champions/v22

# Chạy 1 experiment full config
python -m stock_ml.scripts.cli run champions/v22 --save-results

# Chạy matrix full config
python -m stock_ml.scripts.cli run-matrix matrix/<ten_matrix> --jobs 4 --resume --save-results

# Xem xếp hạng matrix
python -m stock_ml.scripts.cli compare-matrix stock_ml/results/experiments/<ten_matrix> --top 10

# Rebuild leaderboard
python -m stock_ml.scripts.build_leaderboard rebuild
```

## 10. Tiêu chí chọn champion phái sinh

- PnL dương sau cost profile thực tế.
- Max drawdown và `mdd_per_symbol` trong ngưỡng chấp nhận.
- Profit factor > 1.1 ở phần lớn fold.
- Không phụ thuộc một fold/năm duy nhất.
- Số trade đủ lớn, không phải kết quả may mắn.

## 11. Chuẩn quyết định hiện tại

- Tiếp tục trọng tâm `VN30F1M 1H`.
- Dùng transfer cấu trúc từ cổ phiếu, nhưng train chính phải derivatives-only.
- Chỉ mở rộng `VN30F2M`, multi-timeframe, hoặc short-selling sau khi có baseline ổn định qua Split A/B.
