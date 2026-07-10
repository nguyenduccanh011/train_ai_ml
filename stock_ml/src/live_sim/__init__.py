"""Live simulator — day-by-day walk-forward backtesting with signal freezing.

Design: Signals are generated from T-1 data and frozen before T execution.
All filters are applied at T-1 prediction time, never re-filtered at T.
This ensures no lookahead and matches live trading behavior.
"""
