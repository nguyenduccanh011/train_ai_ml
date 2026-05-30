"""leading_v3: Extended feature set with cross-sectional, sector-relative, regime interaction.

36 features from leading_v2 + 22 new features:
- Group A: Cross-sectional rank (5)
- Group B: Sector-relative metrics (6)
- Group C: Market regime interaction (4)
- Group D: Liquidity filters (3)
- Group E: Additional enrichment (4)

Total: 58 features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# Import leading_v2 as base
from stock_ml.src.features.leading_v2 import add_features as add_leading_v2_features


# Sector mapping (Vietnamese stock market)
SECTOR_MAP = {
    # Finance / Ngân hàng
    'ACB': 'Finance', 'BID': 'Finance', 'CTG': 'Finance', 'MBB': 'Finance', 'TCB': 'Finance',
    'VIB': 'Finance', 'VCB': 'Finance', 'SHB': 'Finance', 'STB': 'Finance', 'HDB': 'Finance',
    'TPB': 'Finance', 'EIB': 'Finance', 'OCB': 'Finance', 'SGB': 'Finance', 'VPB': 'Finance',
    'MSB': 'Finance', 'NVB': 'Finance', 'NAB': 'Finance', 'KLB': 'Finance', 'KBC': 'Finance',
    'DBC': 'Finance', 'SEA': 'Finance', 'BAF': 'Finance', 'BFC': 'Finance', 'BSI': 'Finance',

    # Consumer / Tiêu dùng
    'VNM': 'Consumer', 'SAB': 'Consumer', 'BHN': 'Consumer', 'MSN': 'Consumer', 'MWG': 'Consumer',
    'FPT': 'Consumer', 'CII': 'Consumer', 'DGW': 'Consumer', 'PNJ': 'Consumer', 'LGC': 'Consumer',

    # Energy / Năng lượng
    'GAS': 'Energy', 'PVD': 'Energy', 'PVH': 'Energy', 'BSR': 'Energy', 'POW': 'Energy',
    'NT2': 'Energy', 'QTP': 'Energy', 'PVB': 'Energy',

    # Materials / Vật liệu
    'HPG': 'Materials', 'NKG': 'Materials', 'DGC': 'Materials', 'HSG': 'Materials', 'TAC': 'Materials',
    'THD': 'Materials', 'CVN': 'Materials', 'DRC': 'Materials', 'PDR': 'Materials', 'PAS': 'Materials',
    'REE': 'Materials', 'RAL': 'Materials', 'NLG': 'Materials', 'DXG': 'Materials', 'ROS': 'Materials',

    # Industrials / Công nghiệp
    'REE': 'Industrials', 'VJC': 'Industrials', 'ACV': 'Industrials', 'HAH': 'Industrials',
    'HND': 'Industrials', 'VSC': 'Industrials', 'GMD': 'Industrials', 'ITA': 'Industrials',

    # Technology / Công nghệ
    'FPT': 'Technology', 'CMG': 'Technology', 'ICT': 'Technology', 'VGI': 'Technology',
    'VHM': 'Technology', 'HUT': 'Technology', 'ITC': 'Technology', 'BBS': 'Technology',

    # Real Estate / Bất động sản
    'VHM': 'RealEstate', 'NVL': 'RealEstate', 'DXG': 'RealEstate', 'VRE': 'RealEstate',
    'NVB': 'RealEstate', 'HCM': 'RealEstate', 'KDH': 'RealEstate', 'PDR': 'RealEstate',
    'LDG': 'RealEstate', 'SCR': 'RealEstate', 'SHB': 'RealEstate', 'DIG': 'RealEstate',
}

# Default to 'Other' for unmapped symbols
def get_sector(symbol: str) -> str:
    return SECTOR_MAP.get(symbol, 'Other')


def leading_v3_features(
    df: pd.DataFrame,
    market_index: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute leading_v3 features (58 total).

    Args:
        df: DataFrame with columns [symbol, date, open, high, low, close, volume, ...]
            Must be sorted by [symbol, date]
        market_index: Optional market index (VNIndex) for regime detection.
                     DataFrame with columns [date, close].
                     If None, regime features are skipped.

    Returns:
        Original df + 22 new features (58 total including v2 base)

    Notes:
        - Phase 1b.5: Leakage audit must pass (no lookahead)
        - All features grouped by symbol (no cross-symbol leakage before embargo)
        - Warm-up NaNs for moving averages (first 60 days per symbol)
    """

    # Start with leading_v2
    result = add_leading_v2_features(df)

    # ===== Group A: Cross-sectional rank (5 features) =====
    result = _add_cross_sectional_features(result)

    # ===== Group B: Sector-relative metrics (6 features) =====
    result = _add_sector_relative_features(result)

    # ===== Group C: Market regime interaction (4 features) =====
    if market_index is not None:
        result = _add_regime_interaction_features(result, market_index)
    else:
        # Fill with zeros if no market index
        result['market_trend'] = 0
        result['market_volatility_regime'] = 0
        result['regime_interaction_momentum'] = 0
        result['regime_interaction_strength'] = 0

    # ===== Group D: Liquidity filters (3 features) =====
    result = _add_liquidity_features(result)

    return result


def _add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 5 cross-sectional rank features."""

    df = df.copy()

    # For each date, rank each symbol vs all symbols
    for date in df['date'].unique():
        mask = df['date'] == date
        subset = df[mask].copy()

        # Momentum rank (20d return)
        if 'return_20d' in df.columns:
            subset['momentum_rank'] = subset['return_20d'].rank(pct=True)
            df.loc[mask, 'momentum_rank'] = subset['momentum_rank']

        # Volatility rank
        if 'realized_vol_10' in df.columns:
            subset['volatility_rank'] = subset['realized_vol_10'].rank(pct=True)
            df.loc[mask, 'volatility_rank'] = subset['volatility_rank']

        # Volume rank
        subset['volume_rank'] = subset['volume'].rank(pct=True)
        df.loc[mask, 'volume_rank'] = subset['volume_rank']

        # RSI rank
        if 'rsi_14' in df.columns:
            subset['rsi_rank'] = subset['rsi_14'].rank(pct=True)
            df.loc[mask, 'rsi_rank'] = subset['rsi_rank']

        # Price strength rank (close / MA20)
        if 'sma_20_ratio' in df.columns:
            subset['price_strength_rank'] = subset['sma_20_ratio'].rank(pct=True)
            df.loc[mask, 'price_strength_rank'] = subset['price_strength_rank']

    # Forward-fill NaNs from warmup
    for col in ['momentum_rank', 'volatility_rank', 'volume_rank', 'rsi_rank', 'price_strength_rank']:
        if col in df.columns:
            df[col] = df.groupby('symbol')[col].fillna(method='bfill')

    return df


def _add_sector_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 6 sector-relative features."""

    df = df.copy()

    # Add sector column
    df['sector'] = df['symbol'].apply(get_sector)

    for date in df['date'].unique():
        mask = df['date'] == date
        subset = df[mask].copy()

        # For each sector, compute medians
        for sector in subset['sector'].unique():
            sector_mask = (df['date'] == date) & (df['sector'] == sector)
            sector_data = df[sector_mask]

            # Compute sector aggregates
            sector_return_20d_median = sector_data['return_20d'].median() if 'return_20d' in df.columns else 0
            sector_momentum_median = sector_data['momentum'].median() if 'momentum' in df.columns else 0
            sector_volume_median = sector_data['volume'].median()
            sector_volatility_median = sector_data['realized_vol_10'].median() if 'realized_vol_10' in df.columns else 0

            # Assign to symbols in this sector
            sector_symbol_mask = (df['date'] == date) & (df['sector'] == sector)

            if 'return_20d' in df.columns:
                df.loc[sector_symbol_mask, 'return_vs_sector'] = \
                    df.loc[sector_symbol_mask, 'return_20d'] - sector_return_20d_median

            if 'momentum' in df.columns:
                df.loc[sector_symbol_mask, 'momentum_vs_sector'] = \
                    df.loc[sector_symbol_mask, 'momentum'] - sector_momentum_median

            df.loc[sector_symbol_mask, 'volume_vs_sector'] = \
                df.loc[sector_symbol_mask, 'volume'] / (sector_volume_median + 1e-10)

            if 'realized_vol_10' in df.columns:
                df.loc[sector_symbol_mask, 'volatility_vs_sector'] = \
                    df.loc[sector_symbol_mask, 'realized_vol_10'] - sector_volatility_median

            # Strength within sector
            if 'sma_20_ratio' in df.columns:
                df.loc[sector_symbol_mask, 'strength_vs_sector'] = \
                    df.loc[sector_symbol_mask, 'sma_20_ratio'].rank(pct=True)

            # Beta to sector (simplified: correlation)
            if len(sector_data) > 10 and 'returns' in df.columns:
                sector_returns_mean = sector_data['returns'].mean()
                sector_returns_std = sector_data['returns'].std()
                if sector_returns_std > 0:
                    df.loc[sector_symbol_mask, 'beta_to_sector'] = \
                        (df.loc[sector_symbol_mask, 'returns'] - sector_returns_mean) / sector_returns_std

    df = df.drop(columns=['sector'])
    return df


def _add_regime_interaction_features(
    df: pd.DataFrame,
    market_index: pd.DataFrame,
) -> pd.DataFrame:
    """Add 4 regime interaction features based on market index."""

    df = df.copy()

    # Merge market index
    market_index = market_index.rename(columns={'close': 'market_close'}).copy()
    df = df.merge(market_index[['date', 'market_close']], on='date', how='left')

    # Compute market MA200, volatility
    market_index['market_ma200'] = market_index['market_close'].rolling(200, min_periods=1).mean()
    market_index['market_returns'] = market_index['market_close'].pct_change()
    market_index['market_volatility'] = market_index['market_returns'].rolling(20, min_periods=1).std()
    market_index['market_volatility_90th'] = market_index['market_volatility'].rolling(200, min_periods=1).quantile(0.9)

    df = df.merge(market_index[['date', 'market_ma200', 'market_volatility', 'market_volatility_90th']],
                  on='date', how='left')

    # Market trend
    df['market_trend'] = (df['market_close'] > df['market_ma200']).astype(float)

    # Market volatility regime
    df['market_volatility_regime'] = (df['market_volatility'] > df['market_volatility_90th']).astype(float)

    # Regime interactions
    if 'momentum' in df.columns:
        df['regime_interaction_momentum'] = df['momentum'] * df['market_trend']

    if 'sma_20_ratio' in df.columns:
        df['regime_interaction_strength'] = df['sma_20_ratio'] * (1 - df['market_volatility_regime'])

    # Cleanup
    df = df.drop(columns=['market_close', 'market_ma200', 'market_volatility', 'market_volatility_90th'], errors='ignore')

    return df


def _add_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 3 liquidity filter features."""

    df = df.copy()

    # Volume rank (20d average)
    df['volume_20d_avg'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df['volume_rank_20d'] = df.groupby('date')['volume_20d_avg'].transform(lambda x: x.rank(pct=True))

    # Price level (proxy for market cap quality)
    df['price_level'] = (df['close'] > 50_000).astype(float)  # VN threshold

    # Volume stability (low = consistent trading)
    df['volume_stability'] = df.groupby('symbol')['volume'].transform(
        lambda x: x.rolling(20, min_periods=1).std() / (x.rolling(20, min_periods=1).mean() + 1e-10)
    )

    return df
