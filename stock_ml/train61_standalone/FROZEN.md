# Train61 Standalone — FROZEN SNAPSHOT

**Status:** FROZEN at commit `646cab16` (2026-05-XX)

This package is a **self-contained snapshot** of the stock_ml pipeline, frozen for production deployment of the train61 pooled model. It contains its own `src/` copy to ensure runtime stability independent of ongoing refactors in the main `stock_ml/` codebase.

## Why frozen?

- **Production stability:** Changes to `stock_ml/src/` should not break the deployed train61 service.
- **Deployment isolation:** This package can be deployed independently without pulling the entire repo.
- **Snapshot semantics:** The code here reflects the state at the time the train61 model was trained. Re-training with a newer pipeline version would create a new snapshot.

## What this means

- **Do NOT sync** `train61_standalone/src/` with `stock_ml/src/` unless you are creating a new model snapshot.
- **Bug fixes:** If a critical bug is found (e.g., leakage, incorrect PnL calculation), fix it in BOTH places and document the divergence.
- **New features:** Add to `stock_ml/` only. If train61 needs the feature, create a new snapshot.

## Creating a new snapshot

When you want to retrain train61 with the latest pipeline:

1. Copy the current `stock_ml/src/` → `train61_standalone/src/`
2. Update `train61_standalone/requirements.txt` to match `stock_ml/requirements.txt`
3. Run `python app/build_train61_model.py` to rebuild the model
4. Test the new model via `python app/serve_train61_model.py`
5. If validated, commit with message: `train61: snapshot at <commit-hash>`
6. Archive the old snapshot to `versions/<date>/`

## Current snapshot info

- **Frozen at commit:** `646cab16` (refactor/phase-0.2-golden)
- **Date:** 2026-05-XX
- **Pipeline version:** v2 (component-based, MarketProfile, walk-forward)
- **Model:** LightGBM pooled on 61 symbols, feature_set=leading_v4, target=early_wave

## Modified files (not synced with main)

The following files in `train61_standalone/` have local modifications and are NOT synced with `stock_ml/`:

- `src/data/target.py` (M)
- `src/data/loader.py` (M)
- `src/cache/feature_cache.py` (M)
- `src/components/runners/generic_fusion.py` (M)
- `src/pipeline/build_predictions.py` (M)
- `src/pipeline/trainer.py` (M)
- `src/features/engine.py` (M)
- `app/serve_train61_model.py` (M)
- `app/model_registry.py` (M)
- `app/paths.py` (M)
- `app/build_train61_model.py` (M)

These modifications are intentional and part of the frozen snapshot. Do not attempt to merge them back to main unless creating a new snapshot.

---

For questions about this package, see `stock_ml/docs/refactor/TRAIN61_STANDALONE.md` (if exists) or contact the maintainer.
