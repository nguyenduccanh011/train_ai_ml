# Leaderboard Test Files

## Structure

### golden/leaderboard.json
**Status**: Historical reference (outdated schema)
- Contains champion runs from 2020-2025 (live trading data)
- Old format: array of run objects with `run_id`, `config_hash`, etc.
- **Not used by current test suite** - kept for historical reference
- Last updated: 2026-05-03

### Current Leaderboard Format
Production leaderboards now use:
```json
{
  "generated_at": "ISO-8601 timestamp",
  "models": [
    {
      "rank": int,
      "name": str,
      "composite_score": float,
      "model_type": str,
      "audit_status": "PASS|FAIL",
      "date_run": "YYYY-MM-DD",
      "notes": str
    }
  ],
  "summary": {
    "total_models": int,
    "best_model": str,
    "best_pnl": float,
    "note": str
  }
}
```

## Leaderboard Sources of Truth

1. **results/leaderboard.json** - Primary source
2. **stock_ml/visualization/leaderboard.json** - Synced copy
3. **stock_ml/dashboard/leaderboard.json** - Synced copy

All leaderboard entries **must have** a corresponding config file in `stock_ml/config/experiments/done/` or `stock_ml/config/experiments/examples/`.

## TODO

- [ ] Migrate golden test data to new schema or mark as historical-only
- [ ] Add schema validation test for leaderboard.json files
- [ ] Document leaderboard generation process
