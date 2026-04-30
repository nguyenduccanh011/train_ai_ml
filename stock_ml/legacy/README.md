# Legacy code — pre-refactor reference

This folder contains:
- `scripts_reference/`: original `run_vXX_compare.py` scripts (read-only, model config reference)
- `strategies/`: retired backtest functions (49 versions)
- `configs/`: frozen YAML entries from old `models.yaml`
- `docs/`: legacy analysis reports

## Status: read-only

Code here is not actively maintained. To re-run a legacy version:
```bash
python -m stock_ml run legacy/v25
```

## Promotion path

If a legacy version becomes important:
```bash
python -m stock_ml migrate-legacy v25
# → creates config/experiments/legacy/v25.yaml
```

Then port any version-specific fusion logic into new architecture components
(see `docs/refactor/HOW_TO_PORT_LEGACY_VERSION.md`).
