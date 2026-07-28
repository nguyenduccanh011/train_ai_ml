# Stock ML — VN Stock Trading System (DB-first)

Hệ thống ML giao dịch cổ phiếu VN: backtest walk-forward per-symbol (Stage-1)
+ tầng danh mục thống nhất (Stage-2) dùng chung giữa backtest và production serving.

## Kiến trúc sống (2026-07)

```
Postgres (strategy_templates, 3k+ template)          docker compose up -d
   └─ stock_ml/scripts/run_template.py               # chạy 1 template theo id
        └─ src/pipeline/experiment.run_experiment    # dual-ML recombine, walk-forward per-year
             └─ src/backtest/engine.run_backtest     # Stage-1: BASE trades per-symbol, 1-slot
                  └─ stock_ml/portfolio/             # Stage-2: rewrite → gates → meta-priority
                     (run_portfolio, K-slot sim)     #   → K=10 sim (sizing/preempt/T+2)
Leaderboard NAV: scripts/ops/score_nav_leaderboard.py (thước NavSim2 K25, bảng leaderboard_nav)
Dashboard/API : stock_ml/api (FastAPI) + stock_ml/dashboard (docs/QUICK_START.md)
Serving prod  : repo Desktop/stock-serving — import stock_ml.portfolio qua wheel stock_ml_core
```

- Stage-2 là **một implementation duy nhất** (`stock_ml/portfolio/`) — thiết kế + lịch sử:
  [docs/refactor/PORTFOLIO_LAYER_UNIFICATION.md](../docs/refactor/PORTFOLIO_LAYER_UNIFICATION.md).
- `PortfolioConstants` mặc định `stat_mode="causal"` (deploy 3-seed T+2 ≈ 138.5% / DD −13.0%).

## Chạy nhanh

```bash
docker compose up -d                                  # Postgres stockml (port 5433)
python stock_ml/scripts/run_template.py --id 3443     # chạy champion template
python stock_ml/scripts/ops/score_nav_leaderboard.py  # chấm CAGR(NAV) leaderboard
pytest stock_ml/tests -q                              # test suite
RUN_PORTFOLIO_GOLDEN=1 pytest stock_ml/tests/test_portfolio_golden.py -q  # golden guard (~4 phút)
```

## Golden guard (bất biến champion)

`stock_ml/tests/goldens/` pin kết quả champion 3-seed byte-exact (fixtures parquet + MD5 data).
Mọi thay đổi tầng danh mục phải giữ golden pass; data snapshot lệch MD5 → re-pin có chủ đích,
không nới assert.

## Wheel serving (`stock_ml_core`)

Build từ `pyproject.stock_ml_core.toml` (swap tạm sang `pyproject.toml` rồi `python -m build --wheel`).
Chỉ ship inference surface + `stock_ml.portfolio`; không kèm duckdb/sqlalchemy/fastapi
(`test_core_facade.py` verify). Lưu ý PowerShell 5.1: đừng sửa file toml bằng
`Set-Content -Encoding utf8` (chèn BOM phá parse).

## Tài liệu

Xem [docs/README.md](../docs/README.md) (index) — đáng đọc trước:
`QUICK_BACKTEST.md` (template → leaderboard), `TEMPLATE_SUBMISSION_GUIDE.md`,
`RESEARCH_STRATEGY_MAP.md` (bản đồ nghiên cứu), `STRATEGY_ANALYSIS_TOOLKIT.md`.
