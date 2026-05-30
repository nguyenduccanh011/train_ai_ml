# leading_v3 Design — Extended Feature Engineering

**Status**: Design phase (implementation next)  
**Baseline**: leading_v2 (36 features)  
**Target**: Add cross-sectional rank, sector-relative, regime interaction  
**Timeline**: ~1-2 weeks implementation + re-run Alpha Gate

---

## Problem Statement

Alpha Gate batch results show:
- All variants negative PnL (-90% to -300%)
- Simpler models (RF) outperform complex (LGB) → **features cause overfitting**
- Longer horizons worse → noise, not signal
- **Root cause**: Feature set lacks sufficient edge

Current leading_v2 (36 features):
- ✅ Per-symbol OHLCV, momentum, volatility
- ❌ NO cross-sectional context (vs market)
- ❌ NO sector relatives (vs peers)
- ❌ Weak regime detection

---

## Solution: leading_v3 Enhancements

### Group A: Cross-Sectional Rank (5 features)
**Idea**: Stock's feature performance vs all peers TODAY

```python
def cross_sectional_features(df, universe):
    """Compute cross-sectional rank of each stock vs market.
    
    Args:
        df: price data (symbol, date, close, volume, ...)
        universe: list of all symbols
    
    Returns:
        5 features per (symbol, date):
        - momentum_rank: symbol's 20d return rank vs all symbols
        - volatility_rank: symbol's vol rank
        - volume_rank: symbol's volume rank  
        - rsi_rank: RSI rank
        - price_strength_rank: price-to-MA ratio rank
    """
```

**Why**: 
- Captures relative strength (best vs worst momentum today)
- Top 20% momentum stocks > bottom 20%
- Cross-sectional edge well-documented in quant lit

**Computation**:
```
For each date, each symbol:
  momentum_rank = rank(symbol.momentum) / len(universe)  # 0-1
  volatility_rank = rank(symbol.volatility) / len(universe)
  volume_rank = rank(symbol.volume) / len(universe)
  rsi_rank = rank(symbol.rsi) / len(universe)
  price_strength_rank = rank(symbol.price / symbol.ma20) / len(universe)
```

---

### Group B: Sector-Relative Metrics (6 features)
**Idea**: How does stock perform vs its sector?

```python
def sector_relative_features(df, sector_map):
    """Compute stock relative to sector peers.
    
    Args:
        df: price data
        sector_map: {symbol: sector}
    
    Returns:
        6 features per (symbol, date):
        - return_vs_sector: 20d return - sector_median_return
        - momentum_vs_sector: momentum - sector_median
        - volume_vs_sector: volume / sector_median_volume
        - volatility_vs_sector: volatility - sector_median
        - strength_vs_sector: relative strength rank within sector
        - beta_to_sector: correlation with sector index
    """
```

**Why**:
- Sector rotation is a real effect
- Relative to peers is more predictive than absolute
- Reduces systematic sector risk

**Computation**:
```
For each date, symbol:
  sector = sector_map[symbol]
  sector_peers = [s for s in universe if sector_map[s] == sector]
  
  return_vs_sector = symbol.return_20d - median([s.return_20d for s in sector_peers])
  momentum_vs_sector = symbol.momentum - median([s.momentum for s in sector_peers])
  volume_vs_sector = symbol.volume / median([s.volume for s in sector_peers])
  volatility_vs_sector = symbol.volatility - median([s.volatility for s in sector_peers])
  
  # Rank within sector
  strength_vs_sector = rank(symbol.price / symbol.ma20) / len(sector_peers)
  
  # Beta to sector index
  beta_to_sector = correlation(symbol.returns, sector_index.returns, window=60)
```

---

### Group C: Market Regime Interaction (4 features)
**Idea**: Adjust features based on market regime (trending vs ranging)

```python
def regime_interaction_features(df, market_index):
    """Detect market regime and interact with entry signals.
    
    Args:
        df: price data
        market_index: VNIndex or equivalent
    
    Returns:
        4 features per (symbol, date):
        - market_trend: 1 if VNIndex > MA200, 0 else
        - market_volatility_regime: 1 if realized_vol > 90th percentile, 0
        - regime_interaction_momentum: momentum * market_trend
        - regime_interaction_strength: strength * (1 - market_volatility)
    """
```

**Why**:
- Trending markets favor momentum
- High-vol markets reward different signals
- Interaction terms capture non-linearities

**Computation**:
```
For each date:
  market_close = VNIndex[date]
  market_ma200 = VNIndex.rolling(200).mean()[date]
  market_trend = 1 if market_close > market_ma200 else 0
  
  market_vol = VNIndex.returns.rolling(20).std()[date]
  vol_90th = VNIndex.returns.rolling(200).std().quantile(0.9)[date]
  market_volatility_regime = 1 if market_vol > vol_90th else 0
  
  For each symbol:
    regime_interaction_momentum = symbol.momentum * market_trend
    regime_interaction_strength = symbol.strength * (1 - market_volatility_regime)
```

---

### Group D: Liquidity-Based Filters (3 features)
**Idea**: Exclude illiquid stocks that may not trade reliably

```python
def liquidity_features(df):
    """Assess liquidity to avoid micro-cap noise.
    
    Returns:
        3 features per (symbol, date):
        - volume_rank_20d: 20d avg volume rank
        - price_level: market cap proxy (price > 50k = liquid)
        - volume_stability: std(volume) / mean(volume)
    """
```

**Why**:
- Rules-based test (-41%) trades 74 signals vs ML's 210
- May be filtering to high-quality signals implicitly
- Liquidity filter prevents slippage surprises

**Computation**:
```
For each symbol:
  volume_rank_20d = percentile_rank(volume_20d_avg) vs universe
  price_level = 1 if close > 50_000 else 0  # VN threshold
  volume_stability = std(volume_20d) / mean(volume_20d)
  
  Filter: only trade if volume_rank_20d > 0.3 AND price_level == 1
```

---

## Summary: leading_v3 Final Features (58 total)

| Group | Count | Features | Rationale |
|-------|-------|----------|-----------|
| **Existing (v2)** | 36 | OHLCV, momentum, volatility, etc. | Keep what works |
| **Cross-sectional** | 5 | Momentum/vol/volume/RSI/strength rank | Relative strength |
| **Sector-relative** | 6 | Return/momentum/volume/vol vs sector + beta | Sector edge |
| **Regime interaction** | 4 | Market trend × momentum, vol × strength | Non-linear effects |
| **Liquidity** | 3 | Volume rank, price level, vol stability | Filter noise |
| **NEW TOTAL** | **58** | | |

---

## Implementation Plan

### Phase 1: Sector Mapping (1 day)
```python
# File: stock_ml/data/sector_map.py
sector_map = {
    'VNM': 'Consumer',
    'TCB': 'Finance',
    'HPG': 'Materials',
    # ... 486 symbols
}
```
Source: TCBS, FiinPro, or manual categorization

### Phase 2: Feature Implementation (3-5 days)
```python
# File: stock_ml/src/features/leading_v3.py

class LeadingV3:
    def __init__(self, universe, sector_map, market_index):
        self.universe = universe
        self.sector_map = sector_map
        self.market_index = market_index
    
    def compute(self, df):
        """Compute all 58 features.
        
        Args:
            df: DataFrame with symbol, date, OHLCV columns
        
        Returns:
            df with 58 feature columns appended
        """
        # Group A: cross-sectional
        df = self._cross_sectional_features(df)
        
        # Group B: sector-relative
        df = self._sector_relative_features(df)
        
        # Group C: regime interaction
        df = self._regime_interaction_features(df)
        
        # Group D: liquidity
        df = self._liquidity_features(df)
        
        return df
```

### Phase 3: Registration (1 day)
```python
# File: stock_ml/src/features/registry.py
FEATURE_REGISTRY = {
    'basic_v1': BasicV1,
    'leading_v2': LeadingV2,
    'leading_v3': LeadingV3,  # NEW
}
```

### Phase 4: Alpha Gate Re-run (1 day)
```yaml
# File: config/experiments/pending/alpha_gate_v3.yaml
name: alpha_gate_v3
components:
  features: leading_v3  # NEW
  target:
    type: forward_return_regression
    horizon: 5
  entry_model:
    type: random_forest  # Use v5 best performer
    params:
      n_estimators: 300
      max_depth: 15

split:
  type: purged_kfold
  n_splits: 6
  embargo_days: 10

validation:
  n_seeds: 20
  bootstrap_iterations: 1000
  compute_dsr: true
  compute_pbo: true
```

**Expected**: RandomForest + v3 features → pass Alpha Gate (goal: DSR > 0.5, return > 12%)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Sector map outdated | Update quarterly from TCBS |
| Regime features overfit | Keep simple (MA200, 90th percentile) |
| Liquidity filter excludes all | Start loose (>20th percentile) |
| New features leak data | Compute per-symbol, embargo in backtest |

---

## Success Criteria

✅ Alpha Gate v3 passes:
- DSR > 0.5
- PBO < 30%
- Annual return > 12% net of costs
- Max DD < 25%

If fails:
- Investigate which feature group helped most (ablation test)
- Iterate v3 → v4 with stronger edge

---

## Timeline

```
Now:        Batch experiments still running (background)
Week 1:     Implement leading_v3 (sectors, cross-section, regime)
Week 2:     Alpha Gate v3 run + results
Week 3:     If pass → Phase 2 (rules). If fail → v4 iteration
```

---

## Notes

- Keep v2 features (don't throw away what works)
- Cross-sectional + sector-relative are highest ROI additions
- Regime interaction captures "when" edge exists
- RandomForest best performer → use as default model going forward
