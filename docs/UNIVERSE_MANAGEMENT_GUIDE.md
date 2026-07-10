# Universe Management - User Guide

**Status**: ✅ **COMPLETE** - Full CRUD system for symbol sets  
**Version**: 1.0.0  
**Last Updated**: 2026-05-31

---

## Overview

Universe Management is a professional symbol set management system for backtesting and training. It allows researchers to:

- Create and manage named symbol sets ("universes")
- Track versions automatically
- Lock/unlock universes to prevent accidental deletion
- Clone universes for A/B testing
- Use universes in experiments via configuration
- Audit which experiments used which symbol sets

---

## Quick Start

### 1. Start API Server

```bash
cd c:\Users\DUC\ CANH\ PC\Desktop\train_ai_ml

# From project root
uvicorn stock_ml.api.main:app --reload --host 127.0.0.1 --port 8000
```

**Expected Output**:
```
INFO:     Started server process [PID]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 2. Open UI Dashboard

Go to this URL in your browser:

```
http://localhost:8000/dashboard/universe.html
```

You should see:
- **List of universes** (8 pre-seeded universes from migration)
- **Toolbar** with "New Universe" button and market filter
- **Actions** for each universe: View, Edit, Clone, Delete

---

## UI Dashboard Features

### List View

| Column | Description |
|--------|-------------|
| **Slug** | URL-safe identifier (e.g., `main_50`) |
| **Name** | Display name (e.g., "Main 50 Symbols") |
| **Market** | Market profile (e.g., `vn_stock`, `crypto_spot`) |
| **Symbols** | Count of symbols in this universe |
| **Version** | Auto-increment on any symbol change |
| **Status** | 🔒 Locked, Active/Inactive |
| **Created** | Creation date |
| **Actions** | View, Edit, Clone, Delete buttons |

### Create/Edit Modal

**Fields**:
- **Slug** (required) — alphanumeric, underscore, hyphen only; max 128 chars
- **Name** (required) — display name
- **Market** (required) — dropdown from available markets
- **Description** — optional, free text
- **Notes** — optional, metadata
- **Locked** — checkbox to prevent deletion
- **Symbols** — list selector with search
  - Search available symbols by name
  - Click to add to your set
  - Remove button on each symbol
  - Full replace on save

### Actions

- **View** — read-only detail view with all metadata
- **Edit** — modify metadata and symbol list
- **Clone** — creates copy with new slug and name
- **Delete** — soft-delete (only if NOT locked)

---

## API Endpoints

### Base URL

```
http://localhost:8000/api/v1
```

### List Universes

**GET** `/universes`

**Query Parameters**:
- `market` (optional) — filter by market (e.g., `vn_stock`)
- `active_only` (optional) — `true` (default) or `false` to include soft-deleted

**Response (200)**:
```json
{
  "universes": [
    {
      "id": 1,
      "slug": "main_50",
      "name": "Main 50 Symbols",
      "market": "vn_stock",
      "description": "Top 50 liquid VN equities",
      "symbol_count": 50,
      "version": 2,
      "is_locked": false,
      "is_active": true,
      "created_at": "2026-05-31T09:00:00",
      "updated_at": "2026-05-31T10:30:00"
    }
  ]
}
```

### Get Universe Details

**GET** `/universes/{slug}`

**Path Parameters**:
- `slug` — universe slug (e.g., `main_50`)

**Response (200)**:
```json
{
  "id": 1,
  "slug": "main_50",
  "name": "Main 50 Symbols",
  "description": "Top 50 liquid VN equities",
  "market": "vn_stock",
  "symbol_count": 50,
  "version": 2,
  "is_locked": false,
  "is_active": true,
  "notes": "Updated Q2 2026",
  "symbols": [
    {
      "symbol": "ACB",
      "group": "bank",
      "weight": null,
      "rank": 1,
      "notes": null
    },
    {
      "symbol": "BID",
      "group": "bank",
      "weight": null,
      "rank": 2,
      "notes": null
    }
  ],
  "created_at": "2026-05-31T09:00:00",
  "updated_at": "2026-05-31T10:30:00"
}
```

### Create Universe

**POST** `/universes`

**Request Body**:
```json
{
  "slug": "my_test_set",
  "name": "My Test Set",
  "market": "vn_stock",
  "description": "Test universe for experiment A",
  "notes": "Used in alpha_gate_v1 experiments",
  "symbols": [
    {"symbol": "ACB", "group": "bank"},
    {"symbol": "BID", "group": "bank"},
    {"symbol": "FPT"}
  ]
}
```

**Response (200)**:
```json
{
  "id": 9,
  "slug": "my_test_set",
  "name": "My Test Set",
  "market": "vn_stock",
  "symbol_count": 3,
  "version": 1,
  "created_at": "2026-05-31T10:45:00"
}
```

**Response (409)** — Conflict (slug already exists):
```json
{
  "detail": "Universe slug already exists: my_test_set"
}
```

### Update Universe Metadata

**PUT** `/universes/{slug}`

**Request Body** (all optional):
```json
{
  "name": "Updated Name",
  "description": "New description",
  "notes": "New notes",
  "is_locked": true
}
```

**Response (200)**:
```json
{
  "id": 1,
  "slug": "main_50",
  "name": "Updated Name",
  "market": "vn_stock",
  "version": 2,
  "is_locked": true,
  "updated_at": "2026-05-31T10:50:00"
}
```

### Add Symbols

**POST** `/universes/{slug}/symbols`

**Request Body**:
```json
{
  "symbols": [
    {"symbol": "VNM"},
    {"symbol": "VCB", "group": "bank"},
    {"symbol": "FPT", "rank": 10}
  ]
}
```

**Response (200)**:
```json
{
  "slug": "main_50",
  "symbol_count": 52,
  "version": 3,
  "added_count": 2
}
```

**Behavior**: Duplicates ignored, version auto-incremented, symbol_count updated.

### Remove Symbol

**DELETE** `/universes/{slug}/symbols/{symbol}`

**Path Parameters**:
- `slug` — universe slug
- `symbol` — symbol to remove (case-insensitive)

**Response (200)**:
```json
{
  "slug": "main_50",
  "symbol_count": 51,
  "version": 4,
  "removed": "VNM"
}
```

### Replace All Symbols

**PUT** `/universes/{slug}/symbols`

**Request Body**:
```json
{
  "symbols": [
    {"symbol": "ACB"},
    {"symbol": "BID"},
    {"symbol": "MBB"}
  ]
}
```

**Response (200)**:
```json
{
  "slug": "main_50",
  "symbol_count": 3,
  "version": 5
}
```

**Behavior**: Deletes all existing symbols, replaces with new list, increments version.

### Clone Universe

**POST** `/universes/{slug}/clone`

**Request Body**:
```json
{
  "new_slug": "main_50_backup",
  "new_name": "Main 50 Backup",
  "new_market": "vn_stock"
}
```

**Response (200)**:
```json
{
  "id": 10,
  "slug": "main_50_backup",
  "name": "Main 50 Backup",
  "market": "vn_stock",
  "symbol_count": 50,
  "version": 1
}
```

**Behavior**: Copies all symbols and metadata, creates new universe, sets notes to "Cloned from main_50".

### Delete Universe

**DELETE** `/universes/{slug}`

**Response (200)**:
```json
{
  "deleted": "main_50"
}
```

**Response (400)** — Universe is locked:
```json
{
  "detail": "Universe is locked: main_50"
}
```

**Response (404)** — Universe not found:
```json
{
  "detail": "Universe not found: main_50"
}
```

**Behavior**: Soft-delete (sets `is_active=False`), data preserved for audit.

### List Available Markets

**GET** `/markets`

**Response (200)**:
```json
{
  "markets": [
    "vn_stock",
    "vn_derivatives",
    "vn_derivatives_1d",
    "vn_derivatives_15m",
    "vn_derivatives_30m",
    "crypto_spot",
    "crypto_perp"
  ]
}
```

### List Symbols in Market

**GET** `/markets/{market}/symbols`

**Path Parameters**:
- `market` — market name (e.g., `vn_stock`)

**Response (200)**:
```json
{
  "market": "vn_stock",
  "symbol_count": 61,
  "symbols": ["ACB", "AAS", "AAV", ..., "VTP"]
}
```

---

## Using Universes in Experiments

### Method 1: DB-Backed (Recommended)

In your experiment YAML:

```yaml
name: my_experiment_v1
market: vn_stock
components: { ... }
split: { ... }
engine: { ... }

# Use universe from database
universe:
  mode: db
  slug: main_50    # references universe_sets.slug
```

**Advantages**:
- ✅ Single source of truth
- ✅ Version tracking
- ✅ Audit trail via leaderboard.runs
- ✅ Easy to share across experiments
- ✅ Easy to modify (edit UI, increment version)

### Method 2: YAML File (Still Supported)

```yaml
universe:
  mode: file
  file: config/universes/vn_top60.yaml
```

### Method 3: Explicit List (Still Supported)

```yaml
universe:
  mode: explicit
  explicit_list: [ACB, BID, MBB, TCB, VCB]
```

### Method 4: Group-Based (Now Fixed)

```yaml
universe:
  mode: group
  group: bank    # filters profile.symbols.groups where label=="bank"
```

### Method 5: CLI Override (Still Supported)

```bash
python stock_ml/scripts/run_experiments.py \
  --pending config/experiments/pending \
  --done config/experiments/done \
  --failed config/experiments/failed \
  --data-root /path/to/data \
  --symbols ACB,BID,MBB,TCB \
  --out results/
```

---

## Pre-Seeded Universes

After migration + seed, you have:

| Slug | Name | Market | Symbols | Use Case |
|------|------|--------|---------|----------|
| `vn_bank_sector` | VN Banking Sector | vn_stock | 15 | Sector strategy testing |
| `vn_top60` | Top 60 VN Equities | vn_stock | 59 | Research baseline |
| `vn_stock_default` | Default VN Stock | vn_stock | 61 | Default market profile |
| `crypto_spot_default` | Default Crypto Spot | crypto_spot | 5 | Crypto trading |
| `vn_derivatives_default` | Default Derivatives | vn_derivatives | 2 | Derivatives testing |
| `vn_derivatives_1d_default` | 1D Derivatives | vn_derivatives_1d | 2 | Daily derivatives |
| `vn_derivatives_15m_default` | 15M Derivatives | vn_derivatives_15m | 2 | Intraday testing |
| `vn_derivatives_30m_default` | 30M Derivatives | vn_derivatives_30m | 2 | Medium-term test |

---

## Workflow Examples

### Example 1: Create Test Set for Quick Validation

```
1. Open dashboard: http://localhost:8000/dashboard/universe.html
2. Click "New Universe"
3. Fill:
   - Slug: test_quick
   - Name: Quick Test (10 symbols)
   - Market: vn_stock
4. Add symbols: ACB, BID, MBB, VCB, VNM, FPT, HPG, VIC, VJC, VDS
5. Click "Save"
6. In experiment YAML: universe.mode=db, slug=test_quick
7. Run experiment
```

### Example 2: Compare Two Symbol Sets

```
1. Create universe_a: main_50
2. Create universe_b: main_50_conservative (only defensive stocks)
3. Create two experiments:
   - exp_aggressive: universe.slug=main_50
   - exp_conservative: universe.slug=main_50_conservative
4. Run both, compare leaderboard metrics
5. Version tracking: if main_50 changes, exp_aggressive still uses original
```

### Example 3: Clone for A/B Testing

```
1. Open dashboard, find "main_50"
2. Click "Clone"
   - new_slug: main_50_v2_test
   - new_name: Main 50 V2 Test
3. Edit v2_test:
   - Remove 5 volatile symbols
   - Add 5 stable symbols
   - Click Save (version goes to 2)
4. Experiment A uses "main_50" (v1)
5. Experiment B uses "main_50_v2_test" (v1 of new universe)
6. Compare results
```

---

## Maintenance

### Check Universe Count

```bash
# Via API
curl http://localhost:8000/api/v1/universes | jq '.universes | length'

# Via Python
python -c "
from sqlalchemy.orm import sessionmaker
from stock_ml.db.engine import sync_engine
from stock_ml.db.models.universe import UniverseSetModel

Session = sessionmaker(bind=sync_engine)
with Session() as s:
    count = s.query(UniverseSetModel).filter_by(is_active=True).count()
    print(f'Active universes: {count}')
"
```

### Backup Universes

```bash
# Export as JSON
curl http://localhost:8000/api/v1/universes > universes_backup_2026-05-31.json

# Export as CSV (manual export from dashboard)
```

### Reset to Defaults

```bash
# Via Python script
python stock_ml/scripts/seed_universes.py --reset

# This will:
# 1. Drop existing tables
# 2. Recreate schema
# 3. Seed from config/universes/*.yaml + market profiles
```

---

## Troubleshooting

### "Universe not found"

Check:
1. Slug spelling (case-sensitive)
2. `active_only=false` if universe was soft-deleted
3. Correct market if filtering

### "Cannot delete - locked"

1. Open universe in dashboard
2. Click "Edit"
3. Uncheck "Locked"
4. Save
5. Delete

### "Database connection failed" on startup

Make sure you're in the project root when starting API:

```bash
cd c:\Users\DUC\ CANH\ PC\Desktop\train_ai_ml
uvicorn stock_ml.api.main:app --reload
```

Not from `stock_ml/` subfolder.

### "Slug validation error"

Slug must:
- Start with lowercase letter or number
- Contain only: `a-z`, `0-9`, `_`, `-`
- Be max 128 characters
- Example: ✅ `my_test_set_2` ❌ `My Test Set!`

---

## Universe Version History & Snapshots

When you add/remove/replace symbols, the universe **version auto-increments** and a **snapshot is created**.

### Why Snapshots?

If you run a backtest with `vn_top60 v1` (60 symbols), then later add VNM to the universe (v2), the old backtest should still use the original 60 symbols for reproducibility.

### Querying Version History

```bash
# Get all versions of a universe
curl http://localhost:8000/api/v1/universes/vn_top60/versions | jq '.'

# Get specific version
curl http://localhost:8000/api/v1/universes/vn_top60/versions/1 | jq '.symbols'
```

### Python API

```python
from stock_ml.db.engine import sync_engine
from stock_ml.db.repositories.universe_repo import UniverseRepository
from sqlalchemy.orm import Session

with Session(sync_engine) as session:
    repo = UniverseRepository(session)
    
    # Get snapshot for vn_top60 at version 1
    symbols_v1 = repo.get_version_snapshot("vn_top60", version=1)
    # Returns: [{"symbol": "ACB", "group": "bank", ...}, ...]
    
    # Detect what changed between v1 and v2
    symbols_v2 = repo.get_version_snapshot("vn_top60", version=2)
    added = set(s["symbol"] for s in symbols_v2) - set(s["symbol"] for s in symbols_v1)
    removed = set(s["symbol"] for s in symbols_v1) - set(s["symbol"] for s in symbols_v2)
    print(f"Added: {added}, Removed: {removed}")
```

### Automatic Snapshot Behavior

- **On Create:** `universe.create("vn_test", "Test", "vn_stock", symbols=[...])` → v1 snapshot created
- **On Add:** `repo.add_symbols(u.id, [{"symbol": "VNM"}])` → version bumps to 2, v2 snapshot created
- **On Remove:** `repo.remove_symbol(u.id, "VNM")` → version bumps to 3, v3 snapshot created
- **On Replace:** `repo.replace_symbols(u.id, [...])` → version bumps, new snapshot created

---

## Universes in the Leaderboard

Every experiment run on your leaderboard records which universe was used.

### Leaderboard Tracking

Each backtest result includes:
- `universe_slug`: Which universe (e.g., "vn_top60")
- `universe_version`: Which version was used (e.g., 2)
- `universe_tracked`: Boolean flag — is this run's universe recorded?

### Example Query

```bash
# See which experiments used vn_top60 v2
curl 'http://localhost:8000/api/v1/leaderboard?market=vn_stock' | jq '.models[] | select(.universe_slug=="vn_top60" and .universe_version==2) | {run_name, n_symbols, composite_score}'

# Check universe coverage in leaderboard
curl 'http://localhost:8000/api/v1/leaderboard' | jq '.summary.universe_coverage'
# Output: "15/20"  (15 out of 20 runs have universe info)
```

### Backward Compatibility

- **Old runs:** `universe_slug=NULL, universe_version=NULL` (recorded before universe tracking was enabled)
- **New runs:** `universe_slug="vn_top60", universe_version=2` (automatically captured from config)
- **Fairness:** Runs with different universes are grouped separately (fair comparison)

---

## Backfill Process (Apply Symbol Changes Retroactively)

If you need to update existing experiment runs with universe information (e.g., you ran backtests before universe tracking was enabled), use the backfill script.

### When to Backfill

- ✅ After enabling universe tracking, to tag historical runs
- ✅ If you manually changed a universe and want to update past runs
- ❌ Do NOT backfill if runs are already universe-tracked (could create duplicates)

### Backfill Script

```bash
# View which runs would be updated (dry run)
python stock_ml/scripts/backfill_universe_info.py --dry-run

# Apply updates
python stock_ml/scripts/backfill_universe_info.py

# Output:
#   Found 47 runs with NULL universe_slug
#   Updated: 38
#   No config file: 5
#   No universe in config: 4
```

### What It Does

For each run with `universe_slug=NULL`:
1. Finds the experiment config file (checks `results/`, `config/experiments/`)
2. Parses `universe.slug` from YAML
3. Queries database for current version of that universe
4. Updates `leaderboard_runs` with slug + version
5. Adds audit note: `[backfilled_universe: vn_top60@v2]`

### Verification

```bash
# Check backfill results
sqlite3 results/leaderboard.db << 'SQL'
SELECT COUNT(*) as universe_tracked
FROM leaderboard_runs
WHERE universe_slug IS NOT NULL;
SQL
# Before: 0
# After:  40
```

---

## API Documentation (Interactive)

Once server is running, visit:

```
http://localhost:8000/api/docs
```

This is the Swagger UI with full endpoint documentation and try-it-out functionality.

---

## What's Next

- 📊 **Analytics**: Dashboard showing universe usage statistics
- 🔐 **Permissions**: Control who can edit/delete universes
- 📦 **Batch Operations**: Import/export multiple universes
- ✅ **Approval Workflow**: Review + approve before publishing "locked" universes
- 🔗 **Integration**: View which experiments used which universes
