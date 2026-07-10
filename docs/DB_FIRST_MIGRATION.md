# DB-First Migration (Phase 0-3)

## Overview

This document describes the migration from **YAML-based experiment configuration** to **database-driven strategy templates**.

## What Changed

### Before (YAML-Based)
```bash
# Old flow
1. Write YAML file: config/experiments/pending/my_strategy.yaml
2. Run: python -m stock_ml.scripts.run_experiments --pending config/experiments/pending
3. Results appear in leaderboard with no template tracking
```

### After (DB-First)
```bash
# New flow
1. Create rule components: POST /api/v1/model-library/components
2. Create template: POST /api/v1/templates
3. Submit: POST /api/v1/templates/{id}/submit
4. Results tracked to template via template_id
```

---

## Architecture Changes

### 1. **Database Tables (NEW)**

| Table | Purpose | Status |
|-------|---------|--------|
| `feature_set_catalog` | Reusable feature set registry | ✅ Active |
| `target_catalog` | Reusable target definitions | ✅ Active |
| `model_components` | Model library (entry/exit/regime/size) | ✅ Active |
| `component_slots` | Join table (template → components) | ✅ Active |
| `strategy_templates` | Experiment configs (replaces YAML) | ✅ Active |
| `leaderboard_runs` | Results (now with template_id) | ✅ Active |

### 2. **Deprecated Endpoints**

| Endpoint | Status | Replacement |
|----------|--------|-------------|
| `POST /api/v1/experiments` | ❌ Returns 410 Gone | `POST /api/templates/{id}/submit` |
| YAML config loading | ❌ Disabled | DB template loading |
| Champion model YAML loading | ❌ Disabled | API component endpoints |

### 3. **New API Endpoints**

```
POST   /api/v1/model-library/components           Create rule/ML component
GET    /api/v1/model-library/components           List components
PUT    /api/v1/model-library/components/{id}      Update component
DELETE /api/v1/model-library/components/{id}      Delete component

GET    /api/v1/model-library/feature-sets         List feature sets
GET    /api/v1/model-library/targets              List targets
POST   /api/v1/model-library/targets              Create target

POST   /api/v1/templates                          Create template
GET    /api/v1/templates                          List templates
GET    /api/v1/templates/{id}                     Get template details
PUT    /api/v1/templates/{id}                     Update template
DELETE /api/v1/templates/{id}                     Delete template
POST   /api/v1/templates/{id}/submit              Submit template run
GET    /api/v1/templates/{id}/runs                Get template run history
```

---

## Migration Path

### Phase 1: Seed Base Data ✅
```bash
python scripts/seed_db_macd.py
# Creates:
#   - Feature set: leading_v2
#   - Target: trend_regime
```

### Phase 2: Create Components ✅
```bash
python scripts/create_macd_rule_components.py
# Creates:
#   - Entry rule: macd_ma20_entry (ID: 1)
#   - Exit rule: macd_ma20_exit (ID: 2)
```

### Phase 3: Create Template ✅
```bash
python scripts/create_macd_template.py
# Creates:
#   - Template: macd_ma20_rule_v1 (ID: 1)
#   - Links components via component_slots
```

### Phase 4: Submit & Run ✅
```bash
# Via API
curl -X POST http://localhost:8000/api/v1/templates/1/submit

# Via CLI
python -m stock_ml.scripts.run_template --template-id 1 --seed 42
```

### Phase 5: View Results ✅
```bash
# Leaderboard now includes template_id
curl http://localhost:8000/leaderboard?market=vn_stock

# Response includes:
#   "template_id": 1,
#   "template_name": "macd_ma20_rule_v1"
```

---

## File Changes

### Disabled/Deprecated
- `stock_ml/api/routes/experiments.py` — Old endpoint (returns 410)
- `stock_ml/src/utils/config_loader.py:_load_champion_models()` — Returns empty dict
- YAML champion model loading — Replaced by DB queries

### Updated
- `stock_ml/api/routes/leaderboard.py` — Added template_id to response
- `stock_ml/api/routes/templates.py` — Implemented submit endpoint
- `stock_ml/dashboard/template-explorer.html` — Added Submit button

### New
- `stock_ml/scripts/run_template.py` — Template-based runner
- `stock_ml/scripts/seed_db_macd.py` — Database seeding
- `stock_ml/scripts/create_macd_rule_components.py` — Component creation
- `stock_ml/scripts/create_macd_template.py` — Template creation
- `docs/TEMPLATE_SUBMISSION_GUIDE.md` — User guide
- `docs/DB_FIRST_MIGRATION.md` — This document

---

## Data Model

### StrategyTemplate
```python
{
  "id": 1,
  "name": "macd_ma20_rule_v1",
  "strategy": "rule_only",
  "market": "vn_stock",
  "direction": "long",
  "feature_set_id": 1,
  "target_id": 1,
  "model_mode": "rule_only",
  "component_slots": [
    {
      "slot_type": "entry",
      "rule_component_id": 1,
      "ml_component_id": null
    },
    {
      "slot_type": "exit",
      "rule_component_id": 2,
      "ml_component_id": null
    }
  ],
  "split_config": {...},
  "engine_config": {...}
}
```

### LeaderboardRun (Updated)
```python
{
  "run_id": "run_001",
  "template_id": 1,              # NEW: Links to StrategyTemplate
  "template_name": "macd_ma20",  # NEW: For quick reference
  "market": "vn_stock",
  "strategy": "rule_only",
  "total_pnl": 12500,
  "sharpe": 1.8,
  "max_drawdown": -8.5,
  ...
}
```

---

## Backward Compatibility

### Preserved
- ✅ `ExperimentConfig.from_yaml()` — Still available for legacy YAML loading
- ✅ Existing leaderboard data — `template_id` is nullable
- ✅ Non-template runs — Still appear in leaderboard (template_id = null)

### Breaking Changes
- ❌ `POST /api/v1/experiments` — No longer accepts YAML configs
- ❌ Champion model YAML loading — Removed from config_loader

### Migration Strategy for Existing Runs
1. Old runs have `template_id = null` (backward compatible)
2. New runs from templates have `template_id = <id>`
3. Filter by `template_id IS NOT NULL` to show only template-based runs
4. Can migrate old YAML configs to templates via API

---

## Benefits

| Aspect | Before | After |
|--------|--------|-------|
| **Configuration** | Text files (YAML) | Database records |
| **Discoverability** | Filesystem scan | SQL queries + UI |
| **Reusability** | Manual copy/paste | Share component IDs |
| **Versioning** | Git commits | DB versions + config_hash |
| **Tracking** | Filename matching | `template_id` foreign key |
| **UI Management** | File editor | Web UI dashboard |
| **Component Library** | Hardcoded params | Composable DB records |
| **Audit Trail** | Git history | DB timestamps + user tracking (future) |

---

## Future Enhancements (Phase 0-4+)

### Planned
- [ ] User authentication + ownership tracking
- [ ] Template versioning with rollback
- [ ] Component inheritance (base + override)
- [ ] Hyperparameter tuning via Bayesian search
- [ ] Multi-seed orchestration
- [ ] Template publishing/sharing
- [ ] Cost estimation before submission
- [ ] Run cancellation/pause

### Under Consideration
- [ ] Template cloning (copy with rename)
- [ ] A/B testing (template groups)
- [ ] Parameter optimization UI
- [ ] Result export to YAML (for reproducibility)
- [ ] CI/CD integration hooks

---

## Troubleshooting

### "Feature set not found"
```bash
# Check available feature sets
curl http://localhost:8000/api/v1/model-library/feature-sets

# Seed if missing
python scripts/seed_db_macd.py
```

### "Template not found" in submit
```bash
# Verify template exists
curl http://localhost:8000/api/v1/templates/1

# Check component slots are linked
curl http://localhost:8000/api/v1/templates/1 | jq '.componentSlots'
```

### Results not in leaderboard
```bash
# Check if run completed
tail -f logs/tmpl_1_*.log

# Check template runs specifically
curl http://localhost:8000/api/v1/templates/1/runs

# Filter leaderboard by template
curl "http://localhost:8000/leaderboard?template_id=1"
```

---

## Rollback (If Needed)

To revert to YAML-based system:
1. Restore old `experiments.py` endpoint
2. Re-enable `_load_champion_models()` in config_loader.py
3. Create `config/experiments/pending/*.yaml` files manually
4. Run: `python -m stock_ml.scripts.run_experiments --pending ...`

**Note:** This is not recommended. The new system is more robust and maintainable.

---

## References

- [Template Submission Guide](TEMPLATE_SUBMISSION_GUIDE.md)
- [Component Slots Architecture](guides/COMPONENT_SLOTS_ARCHITECTURE.md)
- [Universe Management Guide](UNIVERSE_MANAGEMENT_GUIDE.md)
