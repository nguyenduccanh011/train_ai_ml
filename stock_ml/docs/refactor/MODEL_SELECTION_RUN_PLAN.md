# Model Selection Run Plan

**Status:** Draft 2026-05-02  
**Scope:** Chạy toàn diện các tổ hợp entry model, exit model, feature set và rule baseline để chọn champion cuối.  
**Goal:** Tìm model tốt nhất tổng thể, giữ lại các model có điểm mạnh riêng, loại bỏ model yếu/trùng lặp/bị model khác hơn hẳn.

## 1. Mục tiêu quyết định

Kết quả cuối không chỉ là một bảng xếp hạng, mà phải trả lời được:

1. Model nào là **best overall champion** để dùng mặc định.
2. Model nào là **defensive champion** nếu ưu tiên drawdown thấp/ổn định.
3. Model nào là **specialist** có edge riêng đáng phát triển tiếp.
4. Model nào nên **retire** vì yếu, trùng vai trò, hoặc bị model khác dominance.
5. Rule baseline đang ở đâu so với ML model, để biết ML có thật sự tạo thêm giá trị.

## 2. Hiện trạng model space

### Matrix hiện có

`config/experiments/matrix/quick_entry_exit.yaml` hiện mở rộng thành 8 tổ hợp:

| Feature set | Entry model | Exit model | Strategy |
|---|---|---|---|
| leading_v2 | lightgbm | none | v22 |
| leading_v2 | lightgbm | lightgbm | v22 |
| leading_v2 | xgboost | none | v22 |
| leading_v2 | xgboost | lightgbm | v22 |
| leading_v4 | lightgbm | none | v22 |
| leading_v4 | lightgbm | lightgbm | v22 |
| leading_v4 | xgboost | none | v22 |
| leading_v4 | xgboost | lightgbm | v22 |

### Registry hiện có

Entry models:

- `lightgbm`
- `xgboost`
- `catboost`
- `random_forest`
- `gru`

Exit models:

- `null`
- `lightgbm`
- `xgboost`
- `catboost`

Rule baseline:

- Có runner `rule` trong pipeline.
- Cần đưa rule baseline vào benchmark để làm mốc so sánh.

## 3. Nguyên tắc chạy

1. Chạy theo pha từ rẻ đến đắt: smoke → preview → full → robustness.
2. Không full-run tất cả ngay nếu không cần; loại model yếu sớm bằng sample nhỏ.
3. Mọi kết quả phải lưu artifact để có thể audit lại:
   - `ranking.csv`
   - `ranking.json`
   - `ranking_row.json`
   - `metrics.json`
   - `config.resolved.yaml`
   - `trades.csv`
4. Dùng `--resume` cho mọi pha dài để không chạy lại phần đã xong.
5. Không chọn champion chỉ theo `composite_score`; phải xét thêm risk, consistency, coverage và trade count.
6. Không loại model chỉ vì không đứng top nếu nó có edge riêng rõ ràng.

## 4. Metric dùng để xếp hạng

### Metric chính

- `composite_score`: điểm tổng hợp chính để sort.
- `total_pnl`: tổng lợi nhuận.
- `win_rate`: tỷ lệ thắng.
- `max_drawdown`: drawdown tổng.
- `mdd_per_symbol`: drawdown trung bình/theo mã.
- `yearly_consistency`: độ ổn định theo năm.
- `trade_count`: số lượng lệnh.
- `per_symbol_coverage`: độ phủ mã.
- `avg_holding_days`: thời gian giữ lệnh trung bình.

### Guardrail bắt buộc

Một model không nên được promote nếu có một trong các vấn đề:

- `trade_count` quá thấp, kết quả dễ nhiễu.
- PnL tập trung vào quá ít symbol.
- `yearly_consistency` thấp hoặc âm nặng.
- Drawdown cao nhưng PnL không bù lại.
- Thua rule baseline rõ ràng trên hầu hết metric.
- Chỉ thắng trên preview nhỏ nhưng sụp khi full-run.

## 5. Pha 0 — Chuẩn hóa benchmark và rule baseline

**Status:** Done 2026-05-02. `champions/rule` validate/run pass, đã lưu `trades_rule.csv` và artifact chuẩn tại `results/experiments/rule/`.

**Mục tiêu:** đảm bảo rule baseline và ML model được so sánh trong cùng chuẩn artifact/ranking.

### Việc cần làm

1. Thêm experiment config cho rule baseline, ví dụ:
   - `config/experiments/champions/rule.yaml`
2. Đảm bảo `python -m stock_ml run champions/rule` chạy được.
3. Đảm bảo rule baseline có metrics giống model runs.
4. Nếu cần, tối ưu pipeline để strategy `rule` không build prediction cache thừa.
5. Chạy rule baseline trước để có mốc:

```bash
python -m stock_ml run champions/rule --device cpu --save-results
```

### Output mong muốn

- Có kết quả rule baseline trong `results/`.
- Có thể đưa rule baseline vào bảng so sánh cuối.

### Done khi

- [x] Rule baseline chạy pass: `python -m stock_ml run champions/rule --device cpu --save-results` tạo 2,585 trades.
- [x] Metrics của rule có đủ field để so với ML model: `ranking_row.json` có composite, PnL, win rate, drawdown, consistency, trade count, coverage và config hash.
- [x] Không cần thao tác thủ công ngoài CLI chuẩn.

## 6. Pha 1 — Smoke test full candidate space

**Status:** Smoke complete 2026-05-02. Đã tạo `config/experiments/matrix/full_entry_exit.yaml`; `validate` pass 40 experiments, `run-matrix --dry-run` pass, và smoke `--symbols-limit 20 --device gpu --resume` sinh đủ `ranking.csv`/`ranking.json` 40 rows tại `results/experiments/full_entry_exit/`. Trong lúc chạy có sửa mapping GRU `gpu` → `cuda` để PyTorch nhận đúng device.

**Mục tiêu:** phát hiện lỗi model/config, loại model quá yếu sớm.

### Candidate matrix đề xuất

Tạo matrix mới, ví dụ `config/experiments/matrix/full_entry_exit.yaml`:

```yaml
name_prefix: full_entry_exit

axes:
  features: [leading_v2, leading_v4]
  model_type: [lightgbm, xgboost, catboost, random_forest, gru]
  exit_model:
    - {label: no_exit_model, enabled: false, type: "null"}
    - {label: lightgbm_exit, enabled: true, type: lightgbm, forward_window: 15, loss_threshold: 0.05}
    - {label: xgboost_exit, enabled: true, type: xgboost, forward_window: 15, loss_threshold: 0.05}
    - {label: catboost_exit, enabled: true, type: catboost, forward_window: 15, loss_threshold: 0.05}
  strategy: [v22]

base:
  signals:
    target:
      type: early_wave
  split:
    first_test_year: 2023
    last_test_year: 2024
```

Tổng số tổ hợp: `2 × 5 × 4 × 1 = 40`.

### Lệnh chạy

Dry-run trước:

```bash
python -m stock_ml run-matrix matrix/full_entry_exit --dry-run
```

Smoke test:

```bash
python -m stock_ml run-matrix matrix/full_entry_exit --symbols-limit 20 --device cpu --resume
```

Xem ranking:

```bash
python -m stock_ml compare-matrix results/experiments/full_entry_exit --top 20
```

### Tiêu chí loại sau Pha 1

Loại ngay nếu:

- Run lỗi hoặc train không ổn định.
- `trade_count` quá thấp.
- `composite_score` âm nặng.
- Drawdown cao bất thường.
- Thua rule baseline rõ ràng và không có metric nào nổi bật.

Giữ lại:

- Top 12–16 theo composite.
- Bất kỳ model nào có drawdown thấp, consistency cao, hoặc coverage tốt dù composite chưa top.

### Output mong muốn

- `results/experiments/full_entry_exit_preview` hoặc `full_entry_exit` có ranking smoke.
- Danh sách candidate pass Pha 1.

### Done khi

- [x] Có `config/experiments/matrix/full_entry_exit.yaml` với 40 tổ hợp.
- [x] `python -m stock_ml validate matrix/full_entry_exit` pass.
- [x] `python -m stock_ml run-matrix matrix/full_entry_exit --dry-run` pass.
- [x] Cài/enable `catboost` trong môi trường chạy.
- [x] Chạy lại `python -m stock_ml run-matrix matrix/full_entry_exit --symbols-limit 20 --device gpu --resume` đến khi có ranking đầy đủ.
- [x] `python -m stock_ml compare-matrix results/experiments/full_entry_exit --top 20` đọc được ranking.

### Ghi chú smoke 2026-05-02

Top group theo composite trên 20 symbols:

1. `leading_v2 + random_forest + catboost/xgboost/lightgbm_exit`: score `352.9`, PnL `604.43`, WR `49.41`, MDD `9.54`, 170 trades.
2. `leading_v4 + gru + lightgbm/xgboost/catboost_exit`: score `302.3`, PnL `477.09`, WR `48.97`, MDD `9.65`, 145 trades, yearly consistency `0.8838`.
3. `leading_v2 + gru + lightgbm/xgboost/catboost_exit`: score `288.2`, PnL `420.34`, WR `48.32`, MDD `8.55`, 149 trades.

Các exit model nhiều dòng đang cho kết quả trùng nhau trong smoke, nên Pha 2 cần kiểm tra lại trên 80 symbols trước khi kết luận exit family nào tốt hơn.

## 7. Pha 2 — Preview rộng hơn cho top candidate

**Status:** Preview complete 2026-05-03. Đã chạy `python -m stock_ml run-matrix matrix/full_entry_exit --symbols-limit 80 --top-k-preview 12 --device gpu --resume`; do universe hiện có chỉ resolve được 61 symbols, preview sinh đủ `ranking.csv`/`ranking.json` 40 rows tại `results/experiments/full_entry_exit_preview/`. Top 12 full-run bị skip vì đã có artifact từ Pha 1, nên `results/experiments/full_entry_exit/` vẫn là ranking 20 symbols; dùng preview ranking để chọn candidate Pha 3.

**Mục tiêu:** kiểm tra model còn mạnh khi tăng số mã, giảm nhiễu từ sample nhỏ.

### Lệnh chạy

Dùng top-k preview tự động:

```bash
python -m stock_ml run-matrix matrix/full_entry_exit --symbols-limit 80 --top-k-preview 12 --device cpu --resume
```

Nếu muốn kiểm soát thủ công, tạo matrix nhỏ hơn chỉ gồm candidate pass Pha 1.

### Tiêu chí chọn tiếp

Giữ lại 6–8 model nếu:

- Composite vẫn nằm nhóm trên.
- `yearly_consistency` không sụp.
- `per_symbol_coverage` đủ rộng.
- Exit model cải thiện risk hoặc score so với no-exit cùng entry/feature.
- Không có dấu hiệu thắng nhờ vài mã riêng lẻ.

Loại nếu:

- Tụt hạng mạnh khi tăng symbols.
- Trade count tăng nhưng PnL/WR/drawdown xấu đi rõ.
- Bị một model cùng family hơn toàn diện.

### Output mong muốn

- Danh sách top 6–8 để full-run.
- Ghi chú model nào là candidate overall, defensive, specialist.

### Done khi

- [x] Chạy preview rộng hơn: `python -m stock_ml run-matrix matrix/full_entry_exit --symbols-limit 80 --top-k-preview 12 --device gpu --resume`.
- [x] Có ranking preview 40 rows tại `results/experiments/full_entry_exit_preview/ranking.csv` và `ranking.json`.
- [x] Coverage preview tăng từ 20 lên 61 symbols; top symbol PnL ratio của nhóm top còn khoảng `0.0713–0.0835`, tốt hơn smoke.
- [x] Chọn nhóm candidate để full-run Pha 3.

### Ghi chú preview 2026-05-03

Top group theo composite trên 61 symbols:

1. `leading_v2 + random_forest + catboost/xgboost/lightgbm_exit`: score `467.3`, PnL `2207.88`, WR `55.66`, MDD `88.92`, `mdd_per_symbol` `8.41`, 530 trades, coverage 61 symbols, top symbol PnL ratio `0.0793`. Đây là overall candidate mạnh nhất.
2. `leading_v2 + gru + catboost/lightgbm/xgboost_exit`: score `353.5`, PnL `1452.68`, WR `52.92`, MDD `109.25`, `mdd_per_symbol` `7.86`, 463 trades, top symbol PnL ratio `0.0713`. Giữ làm specialist/defensive candidate vì `mdd_per_symbol` thấp nhất trong nhóm top và pattern khác RF.
3. `leading_v4 + random_forest + xgboost/catboost/lightgbm_exit`: score `315.2`, PnL `1349.21`, WR `51.62`, MDD `99.14`, `mdd_per_symbol` `8.95`, 432 trades, top symbol PnL ratio `0.0835`. Giữ để so feature set `leading_v4` với RF.
4. `leading_v4 + gru + catboost/xgboost/lightgbm_exit`: score `267.6`, PnL `1242.20`, WR `51.02`, MDD `126.70`, `mdd_per_symbol` `8.92`, 441 trades, top symbol PnL ratio `0.0779`. Giữ nếu muốn kiểm tra GRU trên `leading_v4`, nhưng thấp hơn rõ so với `leading_v2 + gru`.

Candidate khuyến nghị cho Pha 3: giữ 8 rows gồm `leading_v2 + random_forest` với 3 exit model, `leading_v2 + gru` với 3 exit model, `leading_v4 + random_forest` với 1 exit model, và `leading_v4 + gru` với 1 exit model. Vì các exit model đang gần như trùng kết quả trong nhóm top, Pha 3 nên ưu tiên một exit đại diện nếu cần giảm chi phí, nhưng vẫn giữ đủ 3 exit cho hai nhóm mạnh nhất để kiểm chứng dominance.

## 8. Pha 3 — Full run top candidate

**Status:** Done 2026-05-03. Lệnh `run-matrix matrix/full_entry_exit --top-k-preview 8 --resume` bị skip do artifact smoke cũ ở `results/experiments/full_entry_exit/`, nên đã tạo `config/experiments/matrix/finalists_entry_exit.yaml` để chạy riêng 8 finalist. `validate` và `dry-run` pass; full-run `python -m stock_ml run-matrix matrix/finalists_entry_exit --device gpu --resume --save-results` sinh đủ `ranking.csv`/`ranking.json` 8 rows tại `results/experiments/finalists_entry_exit/`.

**Mục tiêu:** chạy dữ liệu đầy đủ để chọn nhóm champion thực sự.

### Lệnh chạy

Không dùng lại `matrix/full_entry_exit --top-k-preview 8 --resume` cho Pha 3 nếu `results/experiments/full_entry_exit/` vẫn chứa artifact smoke. Dùng matrix finalist riêng:

```bash
python -m stock_ml validate matrix/finalists_entry_exit
python -m stock_ml run-matrix matrix/finalists_entry_exit --dry-run
python -m stock_ml run-matrix matrix/finalists_entry_exit --device gpu --resume --save-results
```

Xem kết quả:

```bash
python -m stock_ml compare-matrix results/experiments/finalists_entry_exit --top 10
```

### Kết quả Pha 3 2026-05-03

Top group trên 61 symbols:

1. `leading_v2 + random_forest + catboost/xgboost/lightgbm_exit`: score `467.3`, PnL `2207.88`, WR `55.66`, MDD `88.92`, `mdd_per_symbol` `8.41`, yearly consistency `0.2892`, 530 trades, top symbol PnL ratio `0.0793`. Đây là primary champion candidate.
2. `leading_v2 + gru + catboost/lightgbm/xgboost_exit`: score `353.5`, PnL `1452.68`, WR `52.92`, MDD `109.25`, `mdd_per_symbol` `7.86`, yearly consistency `0.2736`, 463 trades, top symbol PnL ratio `0.0713`. Giữ làm defensive/specialist candidate vì `mdd_per_symbol` và concentration tốt nhất trong nhóm finalist.
3. `leading_v4 + random_forest + lightgbm/xgboost_exit`: score `315.2`, PnL `1349.21`, WR `51.62`, MDD `99.14`, `mdd_per_symbol` `8.95`, yearly consistency `0.3884`, 432 trades, top symbol PnL ratio `0.0835`. Giữ làm feature-set challenger vì consistency tốt hơn RF `leading_v2`, nhưng bị thua rõ về score/PnL/WR.

So với rule baseline: finalist ML thắng rõ về composite, win rate, max drawdown và `mdd_per_symbol`, nhưng rule vẫn có PnL tuyệt đối, trade count, coverage và yearly consistency cao hơn. Chưa promote champion cuối trước khi pass robustness split.

Exit model `catboost`, `lightgbm`, `xgboost` tiếp tục cho kết quả trùng nhau trong từng nhóm finalist; Pha 4 nên kiểm tra exit ablation bằng cả no-exit và đủ exit family để xác nhận dominance.

### Done khi

- [x] Có `config/experiments/matrix/finalists_entry_exit.yaml` với 8 finalist.
- [x] `python -m stock_ml validate matrix/finalists_entry_exit` pass.
- [x] `python -m stock_ml run-matrix matrix/finalists_entry_exit --dry-run` pass.
- [x] `python -m stock_ml run-matrix matrix/finalists_entry_exit --device gpu --resume --save-results` pass.
- [x] `python -m stock_ml compare-matrix results/experiments/finalists_entry_exit --top 10` đọc được ranking 8 rows.

### Phân nhóm sau Pha 3

#### A. Best overall champion

Candidate hiện tại: `leading_v2 + random_forest + exit_model`.

Tiêu chí:

- Composite cao nhất hoặc gần cao nhất.
- PnL tốt.
- Drawdown chấp nhận được.
- Consistency tốt.
- Trade count đủ lớn.
- Coverage không quá hẹp.

#### B. Defensive champion

Candidate hiện tại: `leading_v2 + gru + exit_model`.

Tiêu chí:

- Drawdown thấp nhất hoặc rất thấp.
- Yearly consistency tốt.
- PnL không cần cao nhất nhưng ổn định.
- Ít bị sụp ở năm xấu.

#### C. Specialist / experimental candidate

Giữ lại nếu có edge riêng:

- `leading_v2 + gru` có pattern khác RF và concentration thấp hơn.
- `leading_v4 + random_forest` có consistency cao hơn RF `leading_v2` nhưng cần robustness kiểm chứng.
- Exit model giảm drawdown rõ dù mất một phần PnL.
- Feature set mới tăng coverage hoặc consistency.

#### D. Retire

Loại nếu:

- Thua rule baseline.
- Thua model khác cùng vai trò trên hầu hết metric.
- Exit model làm kết quả xấu hơn no-exit.
- Feature set không cải thiện model nào.
- Trade quá ít hoặc quá tập trung.

## 9. Pha 4 — Ablation theo từng trục

**Status:** Done 2026-05-03. Đã dùng `results/experiments/full_entry_exit_preview/ranking.json` 40 rows trên 61 symbols để phân tích ablation theo feature, entry model và exit model; dùng `results/experiments/finalists_entry_exit/ranking.json` để xác nhận nhóm finalist.

**Mục tiêu:** biết model thắng vì entry model, exit model, feature set hay split may mắn.

### 4.1 Feature ablation

Kết luận: `leading_v2` thắng tổng thể và nên là feature set mặc định cho Pha 5.

- Trung bình `leading_v2`: score `196.0`, PnL `986.83`, WR `46.18`, MDD `129.08`, `mdd_per_symbol` `9.17`, consistency `0.6270`, 410.8 trades.
- Trung bình `leading_v4`: score `154.0`, PnL `845.99`, WR `46.30`, MDD `134.15`, `mdd_per_symbol` `9.99`, consistency `0.7340`, 405.4 trades.
- So paired theo cùng entry/exit, `leading_v2` thắng 13/20 cặp; trong nhóm top, `leading_v2` thắng rõ với `random_forest`, `gru`, `xgboost`, `lightgbm` khi có exit model.
- `leading_v4` chỉ đáng giữ như challenger/specialist vì consistency tốt hơn ở vài nhóm, nhất là `leading_v4 + gru` và `leading_v4 + random_forest`, nhưng thua rõ về score/PnL/WR ở nhóm finalist.

### 4.2 Entry model ablation

Kết luận: `random_forest` là entry winner; `gru` là specialist/defensive candidate; các entry còn lại retire khỏi champion path hiện tại.

| Entry model | Avg score | Best score | Vai trò | Ghi chú |
|---|---:|---:|---|---|
| `random_forest` | `305.0` | `467.3` | Primary | Best overall với `leading_v2 + exit`; PnL/WR cao nhất, MDD chấp nhận được. |
| `gru` | `253.4` | `353.5` | Specialist/defensive | `leading_v2 + gru + exit` có `mdd_per_symbol` `7.86` và top symbol ratio `0.0713`, tốt nhất nhóm top. |
| `xgboost` | `114.8` | `151.6` | Retire | Thua xa RF/GRU; không có edge đủ mạnh ngoài consistency. |
| `lightgbm` | `109.7` | `133.9` | Retire | Thua RF/GRU và không nổi bật về risk. |
| `catboost` | `92.1` | `105.6` | Retire | Yếu nhất nhóm entry; top symbol concentration cao hơn nhóm top. |

### 4.3 Exit model ablation

Kết luận: exit model là trục cải thiện lớn nhất so với no-exit, nhưng 3 family exit đang gần như trùng kết quả nên chưa chọn được exit family riêng.

- Trung bình no-exit: score `69.1`, PnL `630.91`, WR `41.47`, MDD `182.52`, `mdd_per_symbol` `11.13`.
- Trung bình có exit: score khoảng `210.0–210.5`, PnL khoảng `1011`, WR khoảng `47.8`, MDD khoảng `114–115`, `mdd_per_symbol` khoảng `9.05–9.07`.
- Với nhóm top, exit tăng mạnh cả score, PnL, WR và giảm drawdown:
  - `leading_v2 + random_forest`: no-exit score `82.1`, PnL `1063.37`, MDD `202.86` → exit score `467.3`, PnL `2207.88`, MDD `88.92`.
  - `leading_v2 + gru`: no-exit score `79.2`, PnL `772.18`, MDD `321.83` → exit score `353.5`, PnL `1452.68`, MDD `109.25`.
- `catboost_exit`, `lightgbm_exit`, `xgboost_exit` thường cho output giống nhau trong cùng feature/entry; Pha 5 chỉ cần giữ một exit đại diện cho robustness nếu muốn giảm chi phí, nhưng nên giữ đủ 3 exit cho winner nếu cần xác nhận artifact cuối.

### Bảng quyết định Pha 4

| Axis | Winner | Keep | Retire | Lý do |
|---|---|---|---|---|
| Feature | `leading_v2` | `leading_v2`, `leading_v4` challenger | Không retire hẳn `leading_v4` trước robustness | `leading_v2` thắng 13/20 paired và dẫn đầu finalist; `leading_v4` có consistency tốt hơn ở vài nhóm. |
| Entry | `random_forest` | `random_forest`, `gru`; giữ `leading_v4 + random_forest` làm challenger | `lightgbm`, `xgboost`, `catboost` khỏi champion path | RF thắng overall; GRU có edge concentration/`mdd_per_symbol`; các entry còn lại thua xa. |
| Exit | Có exit model | Một exit đại diện; đủ 3 exit cho winner nếu cần xác nhận cuối | `no_exit_model` khỏi champion path | Exit cải thiện mạnh score/PnL/WR/drawdown; các exit family hiện gần như trùng nhau. |

### Done khi

- [x] Phân tích đủ 40 rows preview theo feature/entry/exit.
- [x] So sánh paired `leading_v2` vs `leading_v4`.
- [x] So sánh no-exit vs exit cho từng feature/entry.
- [x] Chốt candidate cho Pha 5: `leading_v2 + random_forest + exit`, `leading_v2 + gru + exit`, và `leading_v4 + random_forest/gru + exit` làm challenger.

## 10. Pha 5 — Robustness theo split thời gian

**Status:** Done 2026-05-03. Đã chạy đủ 4 split GPU: `finalists_2019_2025`, `finalists_2021_2024`, `finalists_2022_2025`, `finalists_2023_2025`; mỗi split 8 experiments, ranking 8 rows lưu tại `results/experiments/finalists_*/`.

**Mục tiêu:** tránh chọn model overfit vào giai đoạn 2023–2024.

### Split đề xuất

Tạo các matrix robustness với cùng finalist nhưng đổi split:

```yaml
split:
  first_test_year: 2019
  last_test_year: 2025
```

```yaml
split:
  first_test_year: 2021
  last_test_year: 2024
```

```yaml
split:
  first_test_year: 2022
  last_test_year: 2025
```

```yaml
split:
  first_test_year: 2023
  last_test_year: 2025
```

### Lệnh chạy mẫu

```bash
python -m stock_ml run-matrix matrix/finalists_2019_2025 --device gpu --resume --save-results
python -m stock_ml run-matrix matrix/finalists_2021_2024 --device gpu --resume --save-results
python -m stock_ml run-matrix matrix/finalists_2022_2025 --device gpu --resume --save-results
python -m stock_ml run-matrix matrix/finalists_2023_2025 --device gpu --resume --save-results
```

### Tiêu chí pass robustness

Model pass nếu:

- Vẫn nằm nhóm top ở nhiều split.
- Không sụp hoàn toàn ở một giai đoạn.
- Yearly consistency không xấu.
- Drawdown không tăng bất thường.
- Edge chính vẫn còn thấy được.

Model fail nếu:

- Chỉ thắng ở một split.
- PnL phụ thuộc một năm hoặc một nhóm mã.
- Drawdown tăng mạnh khi đổi giai đoạn.

### Kết quả robustness 2026-05-03

Bảng tóm tắt theo từng split (rank trong split, composite score — exit family trùng nhau nên gộp):

| Nhóm | 2019–2025 | 2021–2024 | 2022–2025 | 2023–2025 | Verdict |
|---|---:|---:|---:|---:|---|
| `leading_v2 + random_forest + exit` | #1 (456.3) | #1 (475.5) | #1 (471.4) | #1 (519.1) | **PASS — primary champion** |
| `leading_v4 + random_forest + exit` | #2 (440.0) | #3 (375.4) | #2 (396.2) | #2 (485.5) | **PASS — challenger** |
| `leading_v2 + gru + exit` | #3 (436.1) | #2 (379.8) | #3 (321.4) | #3 (451.7) | **PASS — specialist** |

Chi tiết metric split dài 2019–2025:

| Nhóm | Score | PnL | WR | MDD | mdd/sym | Consistency | Trades | Coverage | Top symbol ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `leading_v2 + RF + exit` | 456.3 | 10602.88 | 56.43 | 255.74 | 17.88 | 0.6115 | 1712 | 61 | 0.0341 |
| `leading_v4 + RF + exit` | 440.0 | 8558.28 | 54.22 | 260.37 | 19.80 | 0.6744 | 1492 | 61 | 0.0380 |
| `leading_v2 + GRU + exit` | 436.1 | 8376.75 | 54.35 | 270.40 | 19.11 | 0.6960 | 1540 | 61 | 0.0424 |

Chi tiết metric trung bình 3 split ngắn 2021–2025:

| Nhóm | Avg score | Avg PnL | Avg WR | Avg MDD | Avg mdd/sym | Avg consistency | Avg trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| `leading_v2 + RF + exit` | 488.7 | 4885.63 | 56.13 | 200.13 | 13.22 | 0.4335 | 919 |
| `leading_v4 + RF + exit` | 419.0 | 3680.01 | 53.74 | 206.63 | 14.04 | 0.5889 | 784 |
| `leading_v2 + GRU + exit` | 384.3 | 3504.00 | 52.58 | 216.68 | 14.22 | 0.5374 | 832 |

Nhận xét:

- `leading_v2 + random_forest + exit` giữ top 1 cả 4 split, gồm cả split dài 2019–2025 với PnL `10602.88`, WR `56.43%`, MDD `255.74`, coverage 61 symbols → **không overfit vào 2023–2024**.
- Split dài 2019–2025 xác nhận ranking cũ: RF `leading_v2` vẫn #1; RF `leading_v4` #2; GRU `leading_v2` #3.
- `leading_v2 + gru + exit` tụt rank ở 2022–2025 và 2019–2025; edge concentration/`mdd_per_symbol` thấp hơn không xác nhận được trong robustness. Vẫn giữ làm specialist vì pattern entry/exit khác.
- `leading_v4 + random_forest + exit` có consistency tốt hơn primary ở split dài (`0.6744` vs `0.6115`) và trung bình 3 split ngắn (`0.5889`) → xác nhận vai trò challenger đáng giữ.
- Exit family (`catboost`, `lightgbm`, `xgboost`) tiếp tục cho kết quả giống nhau trong cùng nhóm entry/feature; chọn `lightgbm_exit` làm representative vì vận hành nhẹ hơn.
- MDD tăng ở các split dài do window dài hơn bao gồm giai đoạn thị trường xấu; pattern này nhất quán giữa các nhóm nên không phải flag.

### Done khi

- [x] `python -m stock_ml run-matrix matrix/finalists_2019_2025 --device gpu --resume --save-results` pass, ranking 8 rows.
- [x] `python -m stock_ml run-matrix matrix/finalists_2021_2024 --device gpu --resume --save-results` pass, ranking 8 rows.
- [x] `python -m stock_ml run-matrix matrix/finalists_2022_2025 --device gpu --resume --save-results` pass, ranking 8 rows.
- [x] `python -m stock_ml run-matrix matrix/finalists_2023_2025 --device gpu --resume --save-results` pass, ranking 8 rows.
- [x] Phân tích robustness xác nhận `leading_v2 + RF` là primary, `leading_v4 + RF` là challenger, `leading_v2 + GRU` là specialist.
- [x] Không có model nào fail toàn bộ split — tất cả finalist giữ được rank ổn định.

## 11. Pha 6 — Champion decision và cleanup

**Status:** Done 2026-05-03. Đã dùng artifact `results/experiments/finalists_entry_exit/`, `results/experiments/finalists_2019_2025/`, `results/experiments/finalists_2021_2024/`, `results/experiments/finalists_2022_2025/`, `results/experiments/finalists_2023_2025/` và `results/experiments/rule/ranking_row.json` để chốt champion cuối.

**Mục tiêu:** đưa ra quyết định cuối cùng và dọn candidate space.

### Quyết định cuối cùng

1. `primary_champion`: `leading_v2 + random_forest + exit_model`.
   - Dùng `lightgbm_exit` làm cấu hình đại diện mặc định vì metric trùng với `catboost_exit` và `xgboost_exit`, đồng thời nhẹ hơn để duy trì.
   - Full-run 2023–2024: score `467.3`, PnL `2207.88`, WR `55.66`, MDD `88.92`, `mdd_per_symbol` `8.41`, 530 trades, coverage 61 symbols.
   - Robustness: top 1 cả 4 split 2019–2025, 2021–2024, 2022–2025, 2023–2025 với score `456.3`, `475.5`, `471.4`, `519.1`.
2. `defensive_champion`: không promote riêng ở vòng này.
   - `leading_v2 + gru + exit_model` từng có `mdd_per_symbol` tốt nhất ở full-run 2023–2024 (`7.86`), nhưng robustness không xác nhận edge risk: `mdd_per_symbol` cao hơn RF ở cả 3 split và rank tụt xuống #3 ở 2022–2025/2023–2025.
3. `experimental_candidates`:
   - `leading_v4 + random_forest + lightgbm_exit`: giữ làm challenger vì đứng #2 ở 2022–2025 và 2023–2025, consistency trung bình 3 split cao nhất (`0.5889`), nhưng thua primary champion về score/PnL/WR.
   - `leading_v2 + gru + lightgbm_exit`: giữ làm specialist vì entry pattern khác RF, nhưng không dùng làm defensive default cho đến khi tuning thêm risk/consistency.
4. `retired_candidates`:
   - Entry `lightgbm`, `xgboost`, `catboost` khỏi champion path hiện tại vì thua RF/GRU rõ trong preview ablation.
   - `no_exit_model` khỏi champion path vì exit model cải thiện mạnh score/PnL/WR/drawdown.
   - Các exit family trùng metric (`catboost_exit`, `xgboost_exit`) không giữ như champion riêng; chỉ giữ artifact để audit.

### So với rule baseline

- Rule baseline: score `52.6`, PnL `7621.66`, WR `41.66`, MDD `612.42`, `mdd_per_symbol` `39.13`, consistency `1.1748`, 2585 trades.
- Primary champion thắng rất rõ về composite, win rate, max drawdown và `mdd_per_symbol`.
- Rule vẫn thắng PnL tuyệt đối, trade count và consistency do giao dịch nhiều hơn; đây là benchmark cần giữ, không phải champion mặc định vì risk-adjusted profile kém hơn.

### Quy tắc dominance đã áp dụng

Model A dominance Model B nếu A tốt hơn hoặc ngang B ở hầu hết các metric:

- composite_score >= B
- total_pnl >= B
- max_drawdown <= B
- yearly_consistency >= B
- trade_count đủ lớn hơn hoặc tương đương
- coverage >= B

Kết luận dominance:

- `leading_v2 + RF + exit` dominance `leading_v2 + GRU + exit` ở robustness vì thắng score/PnL/WR/MDD/`mdd_per_symbol` trên đa số split.
- `leading_v2 + RF + exit` chưa retire hẳn `leading_v4 + RF + exit` vì `leading_v4` có consistency cao hơn rõ, nên giữ làm challenger.
- Exit family chưa có dominance thực chất vì metrics trùng nhau; chọn `lightgbm_exit` là quyết định vận hành, không phải kết luận chất lượng model.

### Done khi

- [x] Chốt `primary_champion`.
- [x] Chốt không promote `defensive_champion` riêng ở vòng này.
- [x] Chốt `experimental_candidates`.
- [x] Chốt `retired_candidates` khỏi champion path hiện tại.
- [x] So sánh lại với rule baseline trước khi promote champion.

## 12. Thứ tự thực hiện khuyến nghị

1. Tạo `champions/rule.yaml` cho rule baseline.
2. Chạy rule baseline và xác nhận artifact/metrics.
3. Tạo `matrix/full_entry_exit.yaml`. `[done 2026-05-02]`
4. Dry-run matrix 40 tổ hợp. `[done 2026-05-02]`
5. Pha 1: chạy `--symbols-limit 20`. `[done 2026-05-02, ranking 40 rows]`
6. Review ranking, giữ top 12–16. `[done sơ bộ: RF/GRU groups dẫn đầu smoke]`
7. Pha 2: chạy `--symbols-limit 80 --top-k-preview 12`. `[done 2026-05-03, ranking 40 rows trên 61 symbols tại full_entry_exit_preview]`
8. Review ranking, giữ top 6–8. `[done: RF/GRU leading_v2 dẫn đầu; giữ thêm leading_v4 RF/GRU làm đối chứng]`
9. Pha 3: tạo `matrix/finalists_entry_exit.yaml` và full-run 8 finalist. `[done 2026-05-03, ranking 8 rows trên 61 symbols tại finalists_entry_exit]`
10. Pha 4: ablation theo feature/entry/exit. `[done 2026-05-03, RF thắng entry, leading_v2 thắng feature, exit model thắng no-exit]`
11. Pha 5: robustness theo split. `[done 2026-05-03, 4 splits × 8 experiments, RF leading_v2 top 1 cả 4 split]`
12. Pha 6: chọn champion/specialist/retire. `[done 2026-05-03, primary champion = leading_v2 + random_forest + lightgbm_exit]`

## 13. Checklist trước khi chạy dài

- `python -m stock_ml validate matrix/full_entry_exit`
- `python -m stock_ml run-matrix matrix/full_entry_exit --dry-run`
- Kiểm tra data dir resolve đúng.
- Kiểm tra output không bị track nhầm trong git.
- Chạy bằng `--resume`.
- Nếu dùng GPU, xác nhận model nào thật sự hỗ trợ GPU.
- Không đổi feature/model code giữa các pha nếu muốn so sánh công bằng.

## 14. Tiêu chí quyết định cuối

Champion cuối chỉ được chọn khi có đủ bằng chứng:

1. Thắng hoặc nằm top ở full-run.
2. Tốt hơn rule baseline rõ ràng.
3. Không bị dominance bởi model khác.
4. Pass robustness split.
5. Có artifact đầy đủ để tái tạo.
6. `config.resolved.yaml` của winner được lưu lại.

Nếu chưa đạt, không promote champion; giữ kết quả ở trạng thái candidate và chạy thêm pha kiểm chứng.
