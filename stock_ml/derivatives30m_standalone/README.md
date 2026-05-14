# Derivatives 30M Standalone

Standalone tối thiểu để build và serve model top 1 phái sinh VN30 timeframe 30M.

## Nguồn top 1

- Leaderboard: `stock_ml/results/leaderboard/by_market/vn_derivatives_30m/leaderboard.json`
- Source matrix config: `stock_ml/config/experiments/matrix/derivatives_vn30f1m_phase259_30m_exit_model_algo_micro.yaml`
- Bundle: `derivatives_vn30f1m_phase259_30m_exit_model_algo_micro`
- Run: `deriv_p259_30m_exit_model_algo_micro_signals_features-all_features-signals_entry_model_type-xgboost-signals_target-tr_dual_5_27_c3-exit_model-x27l0188_cat-fusion-exit_only-params-baseline`
- Market: `vn_derivatives_30m`
- Timeframe: `30m`
- Entry model: `xgboost`
- Exit model: `catboost`
- Top metrics: `total_pnl=981.55`, `wr=69.46`, `pf=11.552`, `trades=560`, `composite_score=268.2`

Config train top1 nằm ở `config/model_config.resolved.yaml`.
Config runtime realtime nằm ở `config/model_config.realtime_top1.yaml`.

## Cấu trúc tối thiểu

```text
stock_ml/derivatives30m_standalone/
  app/
    paths.py
    model_registry.py
    build_derivatives30m_model.py
    serve_derivatives30m_model.py
  config/
    derivatives30m_symbols.json
    model_config.resolved.yaml
  data/
    derivatives_ai_dataset/
      symbol=VN30F1M/timeframe=30m/data.csv
      symbol=VN30F2M/timeframe=30m/data.csv
  models/
  web/
    derivatives30m_model.html
```

Thư mục này không copy `results/`, `reports/`, notebook hoặc cache lớn. Runtime sẽ tự tạo `cache/`, `data/ohlcv/` và `models/derivatives30m_top1.pkl` khi cần.

## Chuẩn bị dữ liệu

Dữ liệu tối thiểu cần có:

```text
data/derivatives_ai_dataset/symbol=VN30F1M/timeframe=30m/data.csv
data/derivatives_ai_dataset/symbol=VN30F2M/timeframe=30m/data.csv
```

Schema cần các cột `timestamp`, `open`, `high`, `low`, `close`, `volume`; các cột metadata như `symbol`, `exchange`, `asset_type`, `timeframe` được giữ nếu có.

## Build model pkl

Chạy từ repo root:

```powershell
python stock_ml/derivatives30m_standalone/app/build_derivatives30m_model.py --train-scope fold_chain --context-mode no_context_v1
```

Output kỳ vọng:

```text
stock_ml/derivatives30m_standalone/models/derivatives30m_top1.pkl
```

Artifact lưu các phần cần serve:

- market/timeframe/symbols
- feature set và `feature_cols`
- target config
- entry model XGBoost
- exit model CatBoost
- fold models nếu dùng `fold_chain`
- metadata build và thống kê train

Có thể build nhanh một model toàn lịch sử bằng:

```powershell
python stock_ml/derivatives30m_standalone/app/build_derivatives30m_model.py --train-scope full_history --context-mode no_context_v1
```

`full_history` tiện để kiểm tra runtime, nhưng `fold_chain` gần hơn với leaderboard walk-forward.

## Bóc tách config top1 cho realtime

Sinh file config realtime từ run top1 (không dùng `trades.csv` để hiển thị):

```powershell
python stock_ml/derivatives30m_standalone/app/extract_top1_runtime_config.py
```

Output:

```text
stock_ml/derivatives30m_standalone/config/model_config.realtime_top1.yaml
```

Khi cần build model realtime theo profile top1, dùng file output này làm `--config`.

## Chạy standalone server

```powershell
python stock_ml/derivatives30m_standalone/app/serve_derivatives30m_model.py
```

Server chạy tại:

```text
http://127.0.0.1:5013/
```

## Giao diện

Mở trình duyệt tại:

```text
http://127.0.0.1:5013/
```

Giao diện hỗ trợ:

- chọn/search `VN30F1M`, `VN30F2M`
- chart nến 30M + volume
- marker entry/exit theo `derivatives30m_top1`
- stats tóm tắt và bảng trades
- build/clear signal cache trực tiếp trên UI

Chart dùng `lightweight-charts` qua CDN. Nếu máy không có internet, UI vẫn hiển thị stats/trades nhưng chart có thể không tải được.

## API chính

```powershell
Invoke-RestMethod http://127.0.0.1:5013/api/models
Invoke-RestMethod http://127.0.0.1:5013/api/model-info
Invoke-RestMethod http://127.0.0.1:5013/api/symbols
Invoke-RestMethod http://127.0.0.1:5013/api/data/VN30F1M
Invoke-RestMethod http://127.0.0.1:5013/api/ohlcv/VN30F1M
Invoke-RestMethod http://127.0.0.1:5013/api/signal-cache/status
Invoke-RestMethod "http://127.0.0.1:5013/api/signal/derivatives30m_top1/VN30F1M?refresh=1"
```

Nếu signal trả `202`, poll:

```powershell
Invoke-RestMethod http://127.0.0.1:5013/api/signal/derivatives30m_top1/VN30F1M/status
```

Build toàn bộ signal cache:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:5013/api/signal-cache/build
```

Clear signal cache, không xóa model pkl hoặc feature cache:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:5013/api/signal-cache/clear
```

## Ghi chú triển khai

- Standalone này dùng lại source pipeline ở `stock_ml/src` để tránh copy dư module nguồn. Vì vậy cần chạy từ repo này hoặc giữ layout repo tương ứng.
- `DataLoader` luôn được gọi với `timeframe="30m"`; không dùng hard-code `1D` của train61.
- OHLCV API giữ timestamp intraday đầy đủ dạng ISO, không cắt về ngày.
- Top1 dùng CatBoost cho exit model, môi trường cần có package `catboost`.
- Nếu build fail vì `all_features` cần context ngoài OHLCV, chỉ bổ sung context tối thiểu đúng phái sinh/30M, không copy toàn bộ context stock 1D.

## Quy ước không commit dư thừa

Các artifact runtime nên xem là generated:

```text
cache/
data/ohlcv/
models/derivatives30m_top1.pkl
```

Chỉ commit source/config/data mẫu khi thật sự cần đưa standalone đi kèm dữ liệu.
