# Strategy Template Submission Guide (DB-First Phase 0-3)

## Overview

The system has migrated from YAML-based experiment configuration to **database-driven strategy templates**. All experiment configurations are now managed in the database and executed through the template submission API.

## Architecture

```
┌─────────────────────────────────────────┐
│      Giao diện UI (Dashboard)           │
├─────────────────────────────────────────┤
│                                         │
│  • model-library.html (tạo components) │
│  • template-builder.html (tạo template)│
│  • template-explorer.html (xem)        │
│                                         │
└────────────────┬────────────────────────┘
                 │ POST /api/...
                 ↓
┌─────────────────────────────────────────┐
│      FastAPI Routes                     │
├─────────────────────────────────────────┤
│  • POST /api/v1/model-library/components   │
│  • POST /api/v1/templates                  │
│  • POST /api/v1/templates/{id}/submit      │
└────────────────┬────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────┐
│      Database (SQLite)                  │
├─────────────────────────────────────────┤
│  • model_components (rule, lightgbm...) │
│  • strategy_templates (experiment cfg)  │
│  • component_slots (entry/exit linking) │
│  • feature_set_catalog                  │
│  • target_catalog                       │
└────────────────┬────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────┐
│    Experiment Execution (Async Job)     │
├─────────────────────────────────────────┤
│  stock_ml/scripts/run_template.py       │
│  (loads template → runs experiment)     │
└─────────────────────────────────────────┘
```

## Step 1: Create Rule Components

### 1.1 Access Model Library
Open dashboard: `http://localhost:8000/dashboard/model-library.html`

### 1.2 Create Entry Rule Component

Click **"+ New Component"** button, fill in:

```json
{
  "name": "macd_ma20_entry",
  "role": "entry",
  "algorithm": "rule",
  "componentType": "rule",
  "isDefault": false,
  "params": {
    "conditions": [
      {"feature": "macd_hist", "op": ">", "value": 0},
      {"feature": "sma_20_ratio", "op": "<", "value": 1.0},
      {"feature": "close_to_open", "op": ">", "value": 1.0}
    ],
    "logic": "AND",
    "score_feature": "macd_hist"
  },
  "description": "Entry: MACD HIS > 0, MA20 < C, C > O"
}
```

**Note:** 
- `sma_20_ratio < 1.0` means MA20 < Close (ratio inverted)
- `close_to_open > 1.0` means Close > Open

### 1.3 Create Exit Rule Component

```json
{
  "name": "macd_ma20_exit",
  "role": "exit",
  "algorithm": "rule",
  "componentType": "rule",
  "isDefault": false,
  "params": {
    "conditions": [
      {"feature": "macd_hist", "op": "<", "value": 0},
      {"feature": "sma_20_ratio", "op": ">", "value": 1.0},
      {"feature": "close_to_open", "op": "<", "value": 1.0}
    ],
    "logic": "AND",
    "score_feature": "macd_hist"
  },
  "description": "Exit: MACD HIS < 0, C < MA20, C < O"
}
```

**Note:** Make note of the component IDs returned (e.g., ID 42 for entry, 43 for exit)

---

## Step 2: Create Strategy Template

### 2.1 Access Template Builder
Open: `http://localhost:8000/dashboard/template-builder.html`

### 2.2 Fill Template Details

**Strategy Identity:**
- Template Name: `macd_ma20_rule_v1`
- Market: `vn_stock`
- Strategy: `rule_only`
- Direction: `long`

**Components:**
- Feature Set: `leading_v2` (contains macd_hist, sma_20_ratio, close_to_open)
- Target: `trend_regime` (or your preferred target)
- Entry Model: Select rule component `macd_ma20_entry`
- Exit Model: Select rule component `macd_ma20_exit`

**Execution Config:**
- Split Type: `walk_forward_year`
- Train Years: 2
- Test Years: 1
- Gap Days: 25
- Max Hold: 20 bars
- Min Hold: 1 bar
- Hard Stop: -8%
- Commission: 0.15%
- Tax: 0.1%
- Slippage: 0.15%

### 2.3 Save Template

Click **"✓ Save Template"**

The template is now stored in the database with ID (e.g., ID 15).

---

## Step 3: Submit Experiment

### Option A: Via API (cURL)

```bash
curl -X POST http://localhost:8000/api/v1/templates/15/submit \
  -H "Content-Type: application/json" \
  -d '{
    "override_seed": 42
  }'
```

Response:
```json
{
  "job_id": "tmpl_15_macd_ma20_rule_v1",
  "template_id": 15,
  "template_name": "macd_ma20_rule_v1",
  "status": "queued",
  "log_path": "logs/tmpl_15_macd_ma20_rule_v1.log"
}
```

### Option B: Via CLI

```bash
python -m stock_ml.scripts.run_template \
  --template-id 15 \
  --seed 42 \
  --out results/
```

### Option C: Via Template Explorer UI

Open: `http://localhost:8000/dashboard/template-explorer.html`
- Find your template
- Click "Submit" button

---

## Step 4: Monitor Execution

### Check Job Status

```bash
# View logs
tail -f logs/tmpl_15_macd_ma20_rule_v1.log

# Check job registry
curl http://localhost:8000/api/v1/jobs/tmpl_15_macd_ma20_rule_v1
```

### View Results in Leaderboard

Once experiment completes:
1. Open `http://localhost:8000/dashboard/leaderboard.html`
2. Filter by template ID or name
3. View metrics: PnL, Sharpe, MaxDD, trade count, etc.

### View Template Runs

```bash
curl http://localhost:8000/api/v1/templates/15/runs
```

---

## Available Features for Rule Conditions

All features in `leading_v2`:

**Momentum:**
- `macd_hist` — MACD histogram
- `macd_line` — MACD line
- `rsi_14` — Relative Strength Index
- `roc_10` — Rate of Change

**Moving Averages:**
- `sma_20_ratio` — SMA20 / Close (>1 = above, <1 = below close)
- `sma_5_ratio` — SMA5 / Close
- `sma_50_ratio` — SMA50 / Close
- `ema_10_ratio` — EMA10 / Close

**OHLCV:**
- `close_to_open` — Close / Open (>1 = green, <1 = red)
- `ret_1d`, `ret_5d`, `ret_10d` — Returns

**Trend:**
- `adx_14` — Average Directional Index
- `plus_di_14` — Plus DI
- `minus_di_14` — Minus DI

**Volatility:**
- `atr_14_ratio` — ATR / Close
- `bb_width_20` — Bollinger Band width
- `bb_pct_20` — Bollinger Band percentage

**Volume:**
- `volume_ratio_5` — Volume / MA5
- `mfi_14` — Money Flow Index

---

## Condition Operators

Supported operators in rule conditions:

| Operator | Meaning | Example |
|----------|---------|---------|
| `>` | Greater than | `{"feature": "rsi_14", "op": ">", "value": 50}` |
| `<` | Less than | `{"feature": "rsi_14", "op": "<", "value": 30}` |
| `>=` | Greater than or equal | `{"feature": "macd_hist", "op": ">=", "value": 0}` |
| `<=` | Less than or equal | `{"feature": "close_to_open", "op": "<=", "value": 1.0}` |
| `==` | Equal | `{"feature": "atr_regime", "op": "==", "value": 1}` |
| `!=` | Not equal | `{"feature": "atr_regime", "op": "!=", "value": 0}` |

---

## Logic Operators

Combine multiple conditions:

```json
{
  "conditions": [
    {"feature": "macd_hist", "op": ">", "value": 0},
    {"feature": "sma_20_ratio", "op": "<", "value": 1.0},
    {"feature": "close_to_open", "op": ">", "value": 1.0}
  ],
  "logic": "AND"  // or "OR"
}
```

- `AND`: All conditions must be true
- `OR`: At least one condition must be true

---

## DEPRECATED: Old YAML-Based Submission

The old endpoint `POST /api/v1/experiments` is **deprecated**:

```bash
# ❌ NO LONGER SUPPORTED
curl -X POST http://localhost:8000/api/v1/experiments \
  -H "Content-Type: application/json" \
  -d '{"name":"test","config":{...}}'
# Returns: 410 Gone - Use template-based flow instead
```

Use the new template-based flow instead (Steps 1-3 above).

---

## Migration Checklist

- [x] Disable YAML experiment endpoint (`POST /api/v1/experiments`)
- [x] Implement template submission (`POST /api/v1/templates/{id}/submit`)
- [x] Create `run_template.py` for async execution
- [x] Disable YAML champion model loading in config_loader.py
- [x] All strategies now configured via UI + DB

---

## Troubleshooting

### Template submission fails with "Template not found"
- Verify template ID is correct
- Check template is marked `is_active = true` in DB

### Experiment fails with "No entry component"
- Ensure entry slot has at least one component (ML or Rule)
- Check component_slots table in DB

### Features not found error
- Verify features exist in selected feature set
- Check leading_v2.py for complete feature list
- Features are case-sensitive

### Symbols resolution fails
- Check market is valid (vn_stock, etc.)
- Verify data directory exists for market
- Check universe config if specified

---

## Future: Multi-Seed and Hyperparameter Tuning

```json
{
  "override_seed": 42,
  "n_seeds": 10,  // Run N variations (not yet implemented)
  "hyperparameter_grid": {}  // Bayesian search (Phase 0-4)
}
```
