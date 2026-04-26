# Golden Baseline — 11 Champion Versions

Phase 0.2 deliverable.

## Reproduction

```bash
PYTHONHASHSEED=42 python run_pipeline.py \
  --version v22 \
  --compare v32,v34,v35b,v37a,v37a_exit,v37d,v39d,v42_a,v19_3,rule \
  --device cpu --force --no-export
```

Sau đó so checksums:

```bash
cd tests/regression/golden
sha256sum -c checksums.txt
```

## Tại sao CPU, không GPU

LightGBM GPU mode (OpenCL) **không deterministic** giữa các invocations Python — well-known issue
([microsoft/LightGBM#2479](https://github.com/microsoft/LightGBM/issues/2479)). Cùng config, cùng
seed, cùng cache → kết quả khác nhau ở 1-2 versions có decision boundary mong manh (v22 ±1 trade,
v42_a ±1 trade, PnL lệch 0.1-0.3%).

CPU mode (LightGBM histogram-based) deterministic 100%. Trade-off: training chậm hơn ~20-40% nhưng
chấp nhận được cho regression baseline.

GRU (v37d, PyTorch) GPU lại deterministic OK với `cudnn.deterministic=True` đã set ở Phase 0.1.

## Verification log

| Run | Mode | Match Run trước? |
|-----|------|------------------|
| GPU Run 1 | gpu | — (baseline tham chiếu cũ) |
| GPU Run 2 | gpu | ❌ v22 (1784→1785), v42_a (1442→1441) |
| CPU Run A | cpu | — |
| CPU Run B | cpu | ✅ 11/11 hash exact match |

## File list

| Version | Trades | WR | TotalPnL | Hash (truncated) |
|---------|-------:|----:|---------:|------------------|
| v22 | 1784 | 46.4% | +6843.6% | cb7283ef |
| v32 | 1351 | 49.6% | +7508.2% | 0eb6797a |
| v34 | 1325 | 51.0% | +7885.8% | 2dd471c1 |
| v35b | 1381 | 50.0% | +8115.5% | a58ff1da |
| v37a | 1381 | 49.7% | +8042.9% | bf5145da |
| v37a_exit | 1370 | 49.1% | +7940.3% | 3a64b4da |
| v37d (GRU) | 1407 | 48.6% | +7066.8% | e0e63239 |
| v39d | 1181 | 52.0% | +7345.7% | c220b90f |
| v42_a | 1442 | 47.9% | +7035.1% | 2a7031e4 |
| v19_3 | 1910 | 43.9% | +6775.0% | ae92bcb1 |
| rule | 2585 | 41.7% | +7621.7% | 3b983cd5 |

Chi tiết verification ở `docs/refactor/diary/2026-04-27.md`.
