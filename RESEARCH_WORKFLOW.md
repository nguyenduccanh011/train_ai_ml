# Research Workflow — Complete Guide

**Status**: Production-ready (Phase 0-2 complete 2026-05-31)  
**Purpose**: Iterative model optimization using variant generation, auto-rebuild, and experiment tracking

---

## Overview: 8-Step Research Iteration Cycle

```
ANALYZE (Dashboard)
    ↓
HYPOTHESIZE (Choose variant type)
    ↓
GENERATE (generate_variants.py → N YAMLs)
    ↓
RUN BATCH (run_experiments.py → auto-rebuild leaderboard)
    ↓
EVALUATE (leaderboard.html → filter by experiment_group)
    ↓
PIN WINNERS (Click "Pin" on top models)
    ↓
DEEP ANALYZE (dashboard.html → overlay pinned models)
    ↓
CLEANUP (cache_gc.py --apply-policy)
    ↓
[REPEAT from step 2]
```

---

## Step 1: ANALYZE OLD SIGNALS

**Goal**: Understand current best model performance, identify optimization opportunities.

### Open Dashboard
```bash
# Start API server (if not running)
python -m stock_ml.scripts.api_server --port 5176

# Open in browser
http://localhost:5176/visualization/dashboard.html
```

### What to Look For
- **Pinned models**: See which are currently active
- **Signal quality**: Entry/exit timing, false signals
- **Drawdown patterns**: When does the model lose?
- **Trade distribution**: Symbol concentration, time-of-day patterns
- **PnL stability**: Yearly consistency, volatility

### Take Notes
- Which parameters underperform?
- Which feature sets work best?
- What time periods are weak?
- Is the entry/exit threshold right?

---

## Step 2: HYPOTHESIZE

**Goal**: Choose ONE dimension to optimize (don't optimize everything at once).

### Options for Variant Type

| Type | Examples | When to use |
|------|----------|-----------|
| **params** | learning_rate, max_depth, n_estimators | Fine-tune existing model |
| **features** | leading_v2 vs leading_v3 vs leading_v4 | Test new indicators |
| **target** | horizon 5→10 days, threshold 0.5→1.0 | Change prediction window |
| **model** | LightGBM → XGBoost → RandomForest | Test different architectures |

### Example Hypotheses
- "Lower learning_rate (0.01→0.05) will reduce overfitting"
- "leading_v3 features capture market regime better"
- "10-day horizon trades reduce noise better than 5-day"
- "XGBoost handles non-linear patterns better"

---

## Step 3: GENERATE VARIANTS

**Goal**: Create N variant configs using `generate_variants.py` with cartesian product.

### Find a Good Base Config
```bash
# List existing champion configs
ls stock_ml/config/experiments/done/
```

### Generate Variants

#### Example 1: Hyperparameter Grid
```bash
python stock_ml/scripts/generate_variants.py \
  --base stock_ml/config/experiments/done/alpha_gate_v1.yaml \
  --group "2026-06-03_param_tuning" \
  --variant-type params \
  --grid '{
    "entry_model.params.learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
    "entry_model.params.max_depth": [5, 7, 9, 11]
  }'

# Output: 5 × 4 = 20 YAML files
# Names: 2026-06-03_param_tuning_000.yaml through _019.yaml
```

#### Example 2: Feature Set Sweep
```bash
python stock_ml/scripts/generate_variants.py \
  --base stock_ml/config/experiments/done/alpha_gate_v1.yaml \
  --group "2026-06-03_feature_sweep" \
  --variant-type features \
  --grid '{"components.features": ["leading_v2", "leading_v3", "leading_v4"]}'

# Output: 3 YAML files
```

#### Example 3: Target Horizon Tuning
```bash
python stock_ml/scripts/generate_variants.py \
  --base stock_ml/config/experiments/done/alpha_gate_v1.yaml \
  --group "2026-06-03_target_tuning" \
  --variant-type target \
  --grid '{"components.target.horizon": [5, 10, 15, 20]}'

# Output: 4 YAML files
```

### Dry-Run Preview
```bash
python stock_ml/scripts/generate_variants.py \
  --base stock_ml/config/experiments/done/alpha_gate_v1.yaml \
  --group "test_group" \
  --variant-type params \
  --grid '{"entry_model.params.learning_rate": [0.01, 0.05]}' \
  --dry-run

# Shows what would be generated WITHOUT creating files
```

### Inspect Generated YAMLs
```bash
# Check metadata was added
head -20 config/experiments/pending/2026-06-03_param_tuning_000.yaml

# Should see:
# metadata:
#   experiment_group: 2026-06-03_param_tuning
#   variant_type: params
#   parent_run_id: ""
```

---

## Step 4: RUN BATCH

**Goal**: Execute all variants in parallel, auto-rebuild leaderboard.

### Run with Auto-Rebuild
```bash
python stock_ml/scripts/run_experiments.py \
  --pending config/experiments/pending \
  --done stock_ml/config/experiments/done \
  --failed stock_ml/config/experiments/failed \
  --data-root portable_data/vn_stock_ai_dataset_cleaned \
  --symbols VNM,SOS,FPT \
  --out results \
  --parallel 4 \
  --rebuild-leaderboard

# Flags:
# --parallel 4: Run 4 backtests simultaneously
# --rebuild-leaderboard: Auto-rebuild leaderboard when done (default: True)
```

### Monitor Progress
```bash
# In another terminal, watch run directories
watch -n 2 'ls -la results/leaderboard/experiments/ | tail -20'

# Or check leaderboard size (grows as runs complete)
watch -n 5 'wc -l results/leaderboard/leaderboard.json'
```

### When Complete
```bash
# Check summary
tail -20 results/leaderboard/leaderboard.json | jq '.rows[] | {run_id, composite_score, state}'

# Should show: newly generated runs ranked by score
```

---

## Step 5: EVALUATE ON LEADERBOARD

**Goal**: Compare variants visually, apply filters, understand relative performance.

### Open Leaderboard UI
```bash
# Browser
http://localhost:5176/visualization/leaderboard.html
```

### Use Experiment Filter (NEW)
1. **Dropdown**: "All groups" → Select your experiment_group (e.g., "2026-06-03_param_tuning")
2. **Auto-Filter**: Leaderboard shows only variants from this group
3. **Quick Wins**: Top 3 rows are usually the best parameter combos

### Inspect Metadata
1. **Group Column**: Shows experiment_group name (NEW)
2. **Type Column**: Shows variant_type (params|features|target|model) (NEW)
3. **Tooltip**: Hover over Group → see metadata.notes (researcher notes)

### Sort & Compare
- **Sort by Score**: Click "Score" column header → descending
- **See Fairness**: Check if all variants have same fairness_group_key (apples-to-apples)
- **View Details**: Click run_id → see full config, metrics, warnings

### Make Notes
- Which variants are top-3?
- Is the improvement significant?
- Are any variants clearly broken?
- Do the results match your hypothesis?

---

## Step 6: PIN WINNERS

**Goal**: Move top performers to dashboard for deeper analysis.

### Pin Models
1. **Find top-3**: Leaderboard (sorted by score)
2. **Click "Pin"** button next to each
3. **Check State**: Should change to "pinned" badge
4. **Auto-Export**: Pinned models auto-export to `visualization/manifest.json`

### Pin Best + Control
Strategy:
- **Pin #1**: Best variant (candidate for production)
- **Pin #2**: Previous champion (baseline for comparison)
- **Pin #3**: Outlier (test if unusual params work)

---

## Step 7: DEEP ANALYZE ON DASHBOARD

**Goal**: Overlay pinned models, compare signal timing & trade patterns.

### Open Dashboard
```bash
http://localhost:5176/visualization/dashboard.html
```

### Compare Pinned Models
1. **Market/Timeframe**: Select same as experiment
2. **Year/Symbol**: Filter to a specific period
3. **Overlays**: Buy/sell signals from all pinned models side-by-side
4. **Metrics**: Trade PnL, win rate, holding periods

### Look For
- **Signal Timing**: Do different variants trigger at same spots?
- **Trade Quality**: Which has more winners vs false signals?
- **Stability**: Which stays consistent across symbols/periods?
- **Drawdown**: Which recovers faster after losing streaks?

### Validate Hypothesis
- Did param tuning work? (check if low LR versions better)
- Did features matter? (compare feature sets)
- Did target horizon help? (check trade quality)
- Did model change help? (compare architectures)

---

## Step 8: CLEANUP (Optional)

**Goal**: Retire old runs, free disk space using retention policy.

### Check Retention Config
```bash
# View current policy in base.yaml
grep -A 5 "retention:" stock_ml/config/base.yaml

# Output should show:
# retention:
#   keep_pinned: true
#   keep_per_group: 5
#   auto_retire_after_days: 30
#   trash_purge_after_days: 14
```

### Dry-Run Policy
```bash
python stock_ml/scripts/cache_gc.py --apply-policy

# Shows:
# - How many runs would be retired
# - How much cache would be freed
# - Trash cleanup plan
# (No changes made — dry-run only)
```

### Apply Policy
```bash
python stock_ml/scripts/cache_gc.py --apply --purge

# Actually executes:
# - Retires old trained runs (→ "retired" state)
# - Quarantines orphan cache (→ _trash/)
# - Purges trash >14 days old (→ freed disk)
```

### Verify Cleanup
```bash
# Check leaderboard state distribution
jq '.rows[] | .state' results/leaderboard/leaderboard.json | sort | uniq -c

# Should show more "retired" after cleanup
```

---

## REPEAT: Next Iteration

After Step 8, loop back to **Step 2: HYPOTHESIZE** with new insights:
- Did param tuning work? → Try even more extreme values
- Did features help? → Combine best features with new indicators
- Did target horizon improve? → Test even longer horizons
- Did model change help? → Try ensemble of best models

---

## Tips for Effective Research

### Do's ✅
- **One dimension at a time**: Tune params, OR features, OR target (not all together)
- **Document hypothesis**: Write down your prediction in metadata.notes
- **Keep champions**: Pin multiple models to see relative performance
- **Use filters**: experiment_group filter saves time finding your variants
- **Version configs**: Save winning YAML to `config/experiments/done/`

### Don'ts ❌
- **Don't tune everything**: Combinatorial explosion of variants
- **Don't ignore fairness warnings**: "Cross-group" means unequal comparison
- **Don't pin mediocre models**: Pin only top-3 to keep dashboard clean
- **Don't delete trained runs**: Keep for historical comparison
- **Don't skip cleanup**: Disk fills up quickly with cache/artifacts

---

## Example Research Session

### Session Goal: Improve Sharpe Ratio by Tuning LightGBM Parameters

```bash
# 1. HYPOTHESIZE: Lower learning_rate might reduce overfitting
# ↓

# 2. GENERATE: 12 variants (3 LRs × 4 depths)
python stock_ml/scripts/generate_variants.py \
  --base stock_ml/config/experiments/done/alpha_gate_v1.yaml \
  --group "2026-06-03_lgb_tuning" \
  --variant-type params \
  --grid '{"entry_model.params.learning_rate": [0.01, 0.03, 0.05], "entry_model.params.max_depth": [5, 7, 9, 11]}'
# ↓

# 3. RUN: Execute batch
python stock_ml/scripts/run_experiments.py \
  --pending config/experiments/pending \
  --done stock_ml/config/experiments/done \
  --failed stock_ml/config/experiments/failed \
  --data-root portable_data/vn_stock_ai_dataset_cleaned \
  --symbols VNM,SOS,FPT \
  --out results \
  --parallel 4
# ↓

# 4. EVALUATE: Open leaderboard.html
#    - Filter by "2026-06-03_lgb_tuning"
#    - Top 3: [0.01 LR + depth=7, 0.01 LR + depth=9, 0.03 LR + depth=7]
#    - Sharpe ↑ from 1.82 to 2.15 (18% improvement!)
# ↓

# 5. PIN: Select top-3
#    Click "Pin" on best 3 variants
# ↓

# 6. DEEP ANALYZE: dashboard.html
#    - Compare signals from all 3
#    - Best one has fewer false positives
#    - Hypothesis confirmed: lower LR + shallower trees = better
# ↓

# 7. CLEANUP:
python stock_ml/scripts/cache_gc.py --apply --purge
# ↓

# 8. DECIDE: Keep the 0.01 LR + depth=7 version
#    Copy to champions: cp config/experiments/pending/2026-06-03_lgb_tuning_001.yaml config/experiments/done/champion_sharpe_boost.yaml
#    Commit to git
```

**Result**: +18% Sharpe improvement found in 2 hours! 🎉

---

## Troubleshooting

### Variants Not Appearing on Leaderboard
- ✅ Check runs completed: `ls results/leaderboard/experiments/`
- ✅ Verify ranking_row.json exists in each run dir
- ✅ Rebuild manually: `python stock_ml/scripts/build_leaderboard.py rebuild`

### Filter Dropdown Empty
- ✅ Leaderboard not loaded: Refresh browser
- ✅ No grouped runs: Generate variants first
- ✅ Cache issue: `python stock_ml/scripts/cache_gc.py --apply`

### Disk Space Issues
- ✅ Check cache: `python stock_ml/scripts/cache_gc.py --apply-policy`
- ✅ Delete old runs: Mark as "retired" then cleanup
- ✅ Monitor: `du -sh results/`

---

## See Also

- **IMPLEMENTATION_ROADMAP.md** — Full architecture & completion status
- **README.md** — API endpoints & system overview
- **API.md** — REST endpoint reference
- **docs/** — Deployment & operations guides
