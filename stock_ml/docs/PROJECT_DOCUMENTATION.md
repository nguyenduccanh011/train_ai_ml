# VN Stock ML - Project Documentation

## 1. Muc tieu

Du an xay dung he thong thu nghiem model giao dich co phieu VN theo flow thong nhat:
- Train + backtest theo walk-forward
- So sanh cong bang nhieu version strategy tren cung dieu kien
- Quan ly model version bang config trung tam
- Export ket qua cho dashboard web

Nguon su that hien tai:
- `config/models.yaml`: model registry + pipeline + scoring
- `run_pipeline.py`: flow chay chinh
- `src/evaluation/scoring.py`: scoring va metric canonic

## 2. Kien truc hien tai

```
stock_ml/
|-- run_pipeline.py               # Entry point chinh (train/backtest/export/compare)
|-- model_manager.py              # CLI quan ly model va compare
|-- compare_rule_vs_model.py      # Rule strategy + legacy compare script
|-- config/
|   |-- models.yaml               # Single source of truth cho models/pipeline/scoring
|   `-- base.yaml                 # Training device mac dinh
|-- src/
|   |-- config_loader.py
|   |-- experiment_runner.py      # Shared runner cho scripts experiment
|   |-- backtest/
|   |   |-- engine.py             # backtest_unified()
|   |   `-- defaults.py           # default params + symbol profile configs
|   |-- data/
|   |   |-- loader.py
|   |   |-- splitter.py
|   |   `-- target.py
|   |-- evaluation/
|   |   |-- scoring.py            # composite_score() + calc_metrics()
|   |   `-- metrics.py
|   |-- export/
|   |   `-- unified_export.py
|   |-- features/
|   |   `-- engine.py
|   |-- models/
|   |   `-- registry.py
|   `-- strategies/
|       `-- legacy.py
|-- experiments/
|   |-- run_v22_final.py
|   |-- run_v23_optimal.py
|   |-- run_v24.py
|   |-- run_v25.py
|   |-- run_v26.py
|   |-- run_v27.py
|   |-- run_v26_experiments.py
|   `-- run_feature_ablation.py
|-- results/                      # trades_*.csv + trades_*.meta.json
`-- visualization/                # dashboard + du lieu export
```

## 3. Flow chuan

### 3.1 Pipeline chinh

1. Resolve symbols 1 lan (`get_pipeline_symbols`)
2. Group model theo `feature_set`
3. Train 1 lan cho moi group feature_set
4. Chay backtest cho tung version tren prediction cache
5. Luu `trades_<version>.csv` + `trades_<version>.meta.json`
6. Export JSON cho dashboard
7. Compare bang composite score

### 3.2 Fairness rule

- Cung symbol list cho tat ca version trong cung run
- Cung target config (co target fingerprint de verify cache)
- Rule baseline chay cung walk-forward windows voi ML strategy
- Cache CSV chi duoc reuse neu metadata match

### 3.3 Smart cache

`run_pipeline.py` validate metadata truoc khi reuse cache.
Neu mismatch (symbol/min_rows/feature_set/target), cache bi invalidate va version do duoc re-run de dam bao compare cong bang.

## 4. Config trung tam (`config/models.yaml`)

- `models`: registry tat ca version (active/retired)
- `pipeline`: data_dir, split params, symbol config, target, model_type
- `scoring.weights`: trong so cho composite score
- `symbol_profiles`, `rule_priority_symbols`, `score5_risky_symbols`: config cho backtest engine

## 5. Danh gia model

Metric canonic (`src/evaluation/scoring.py`):
- `calc_metrics(trades)` -> trades, wr, avg_pnl, total_pnl, pf, max_loss, avg_hold
- `composite_score(metrics)` -> score tong hop (higher is better)

Cong thuc score su dung weights tu `models.yaml`.
Tat ca compare trong `run_pipeline.py` va `model_manager.py` dung cung mot score function.

## 6. Commands chinh

### 6.1 Chay pipeline

```bash
python run_pipeline.py --version v27
python run_pipeline.py --version v27 --compare v26,v25,rule
python run_pipeline.py --all
python run_pipeline.py --all --skip-existing
python run_pipeline.py --version v27 --compare v26,v25 --force
```

### 6.2 Export

```bash
python run_pipeline.py --export-all
python run_pipeline.py --version v27 --export-only
```

### 6.3 Compare/quan ly model

```bash
python model_manager.py list
python model_manager.py compare
python model_manager.py compare --versions v27,v26,v25,rule
python model_manager.py retire v22 --reason "Superseded by V27"
python model_manager.py activate v22
```

### 6.4 Experiment scripts

```bash
python experiments/run_v26_experiments.py
python experiments/run_feature_ablation.py
python experiments/run_feature_ablation.py --group A,B --symbols ACB,FPT,HPG
```

## 7. Quy uoc clean code

- Khong import code tu `archive/` trong active flow
- Scoring phai dung `src/evaluation/scoring.py`
- Symbol resolution uu tien `get_pipeline_symbols`
- Params/version behavior dat trong `config/models.yaml`, han che hard-code
- Moi ket qua trades nen co metadata companion de support fair cache

## 8. Lich su cap nhat tai lieu

- 2026-04-22: Dong bo lai tai lieu theo kien truc hien tai (`run_pipeline.py` + model registry + fair smart cache).
