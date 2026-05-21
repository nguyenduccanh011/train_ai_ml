# Derivatives 30m Standalone — FROZEN SNAPSHOT

**Status:** FROZEN snapshot for production deployment of the VN30 derivatives 30m top1 model.

This package is a self-contained snapshot of the stock_ml pipeline targeting VN30F1M / VN30F2M futures on the 30-minute timeframe. Code under `src/` (if present) mirrors the state of the main `stock_ml/src/` at the time the model was trained.

## Why frozen?

- **Production stability:** Changes to the main `stock_ml/src/` should not break the deployed derivatives 30m service.
- **Deployment isolation:** This package ships independently without pulling the full research repo.
- **Snapshot semantics:** Code here reflects the pipeline version used to train the current model. Re-training with a newer pipeline version creates a new snapshot.

## What this means

- **Do NOT sync** `derivatives30m_standalone/src/` (if present) with `stock_ml/src/` unless creating a new model snapshot.
- **Bug fixes:** If a critical bug is found (e.g., leakage, PnL miscalc), fix in BOTH places and document the divergence.
- **New features:** Add to `stock_ml/` only. If derivatives 30m needs it, cut a new snapshot.

## Creating a new snapshot

1. Validate the new model in `stock_ml/` first.
2. Sync `stock_ml/src/` → `derivatives30m_standalone/src/` (if applicable).
3. Update `derivatives30m_standalone/requirements.txt` to match.
4. Rebuild via `app/build_derivatives30m_model.py`.
5. Test via `app/serve_derivatives30m_model.py`.
6. Commit with message: `derivatives30m: snapshot at <commit-hash>`.

## Layout

- `app/` — build/serve scripts entry points
- `cache/` — gitignored; regenerated per environment
- `config/` — resolved YAML configs frozen at training time
- `data/` — gitignored; source OHLCV (provider-dependent)
- `models/` — gitignored; trained artifacts
- `web/` — frontend dashboard assets
