# Train61 Standalone

Standalone app for serving signals with one pooled model trained on 61 symbols.
This folder now contains its own `src/` runtime so it can run independently of `stock_ml/src`.
Runtime is self-contained: only reads data/artifacts inside `train61_standalone/`.

## Folder layout

- `app/build_train61_model.py`: build pooled train61 artifact (single-model or multi-fold chain).
- `app/serve_train61_model.py`: Flask server + API + chart page.
- `config/model_config.resolved.yaml`: model config copy.
- `config/train61_symbols.json`: list of 61 training symbols.
- `models/`: standalone model files.
- `data/ohlcv/`: standalone OHLCV json cache for chart.
- `data/vn_stock_ai_dataset_cleaned/`: source dataset used for training/inference (`symbol=XXX/timeframe=1D/data.csv`).
- `cache/features/`: standalone feature cache.
- `cache/signals/`: standalone signal payload cache.
- `web/train61_model.html`: standalone UI page.
- `versions/`: archived model/config snapshots.

## Build model file (train 61 symbols)

Prerequisite dataset must exist in:

- `train61_standalone/data/vn_stock_ai_dataset_cleaned`

```powershell
cd stock_ml/train61_standalone
pip install -r requirements.txt
python app/build_train61_model.py
```

Build no-context (legacy):

```powershell
cd stock_ml/train61_standalone
python app/build_train61_model.py --context-mode no_context_v1 --output versions/v1_single_model_no_context/train61_single_model.no_context_v1.pkl
```

Build fold-chain artifact:

```powershell
cd stock_ml/train61_standalone
python app/build_train61_model.py --train-scope fold_chain --context-mode no_context_v1 --output versions/v4_top1_fold_chain_no_context/train61_fold_chain.top1.no_context.pkl
```

## Active Runtime Artifacts

- `train61_standalone/models/train61_single_model.pkl`
- `train61_standalone/models/train61_fold_chain.top1.no_context.pkl`

`versions/` is archive-only. Promote/copy snapshots to `models/` before using them in runtime.

## Run server

```powershell
cd stock_ml/train61_standalone
python app/serve_train61_model.py
```

Server reads only local paths under `train61_standalone/`:

- `data/vn_stock_ai_dataset_cleaned` (raw dataset)
- `data/ohlcv` (OHLCV cache for UI)
- `cache/features`, `cache/signals`
- `results/experiments/.../config.resolved.yaml` (for on-demand / pooled-global rerun modes)

Model runtime mode:

- `train61_pooled`: `pkl`
- `top1_fold_chain_no_context`: `pkl`
- `v5_top1_pooled_global_rerun`: `pooled_global_rerun`
- `top1_on_demand`: `on_demand`

Open:

- `http://127.0.0.1:5012`
