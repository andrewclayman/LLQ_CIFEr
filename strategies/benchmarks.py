"""
Benchmark strategies for comparison
- MA Crossover
- Time Series Momentum
- Buy & Hold (computed separately)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


# =============================================================================
# MA CROSSOVER (Grid-searched best params)
# =============================================================================
MA_FAST = 15
MA_SLOW = 80


def ma_crossover_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Moving Average Crossover Strategy
    Enter when fast MA > slow MA, exit when fast MA < slow MA
    """
    close = df['close'].values
    n = len(close)

    fast_ma = pd.Series(close).rolling(MA_FAST, min_periods=1).mean().values
    slow_ma = pd.Series(close).rolling(MA_SLOW, min_periods=1).mean().values

    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)

    in_position = False
    warmup = MA_SLOW

    for i in range(warmup, n):
        bullish = fast_ma[i] > slow_ma[i]

        if bullish and not in_position:
            entries[i] = True
            in_position = True
        elif not bullish and in_position:
            exits[i] = True
            in_position = False

    entries = np.roll(entries, 1)
    exits = np.roll(exits, 1)
    entries[0] = False
    exits[0] = False

    return pd.DataFrame({'entries': entries, 'exits': exits}, index=df.index)


# =============================================================================
# TIME SERIES MOMENTUM
# =============================================================================
TS_LOOKBACK = 63  # ~3 months for hourly (63 trading days * 24 hours)


def ts_momentum_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time Series Momentum Strategy
    Enter when return over lookback is positive, exit when negative
    """
    close = df['close'].values
    n = len(close)

    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)

    in_position = False
    lookback_hours = TS_LOOKBACK * 24  # Convert days to hours

    for i in range(lookback_hours, n):
        momentum = (close[i] - close[i - lookback_hours]) / close[i - lookback_hours]

        if momentum > 0 and not in_position:
            entries[i] = True
            in_position = True
        elif momentum < 0 and in_position:
            exits[i] = True
            in_position = False

    entries = np.roll(entries, 1)
    exits = np.roll(exits, 1)
    entries[0] = False
    exits[0] = False

    return pd.DataFrame({'entries': entries, 'exits': exits}, index=df.index)


# =============================================================================
# BOLLINGER BREAKOUT
# =============================================================================
BB_PERIOD = 50
BB_STD = 2.5


def bollinger_breakout_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bollinger Breakout Strategy
    Enter when price breaks above upper band, exit when drops below middle
    """
    close = df['close'].values
    n = len(close)

    ma = pd.Series(close).rolling(BB_PERIOD, min_periods=1).mean().values
    std = pd.Series(close).rolling(BB_PERIOD, min_periods=1).std().values

    upper = ma + BB_STD * std
    lower = ma - BB_STD * std

    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)

    in_position = False
    warmup = BB_PERIOD

    for i in range(warmup, n):
        if close[i] > upper[i] and not in_position:
            entries[i] = True
            in_position = True
        elif close[i] < ma[i] and in_position:  # Exit at middle band
            exits[i] = True
            in_position = False

    entries = np.roll(entries, 1)
    exits = np.roll(exits, 1)
    entries[0] = False
    exits[0] = False

    return pd.DataFrame({'entries': entries, 'exits': exits}, index=df.index)


# =============================================================================
# DONCHIAN BREAKOUT
# =============================================================================
DONCHIAN_ENTRY = 50
DONCHIAN_EXIT = 20


def donchian_breakout_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Donchian Channel Breakout
    Enter on 50-period high, exit on 20-period low
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(close)

    entry_high = pd.Series(high).rolling(DONCHIAN_ENTRY, min_periods=1).max().values
    exit_low = pd.Series(low).rolling(DONCHIAN_EXIT, min_periods=1).min().values

    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)

    in_position = False
    warmup = DONCHIAN_ENTRY

    for i in range(warmup, n):
        if close[i] >= entry_high[i-1] and not in_position:
            entries[i] = True
            in_position = True
        elif close[i] <= exit_low[i-1] and in_position:
            exits[i] = True
            in_position = False

    entries = np.roll(entries, 1)
    exits = np.roll(exits, 1)
    entries[0] = False
    exits[0] = False

    return pd.DataFrame({'entries': entries, 'exits': exits}, index=df.index)


# =============================================================================
# WRAPPER FUNCTIONS FOR CONSISTENT INTERFACE
# =============================================================================

class MACrossover:
    @staticmethod
    def compute_signals(df):
        return ma_crossover_signals(df)

    @staticmethod
    def get_strategy_info():
        return {'name': 'MA Crossover', 'params': f'fast={MA_FAST}, slow={MA_SLOW}'}


class TSMomentum:
    @staticmethod
    def compute_signals(df):
        return ts_momentum_signals(df)

    @staticmethod
    def get_strategy_info():
        return {'name': 'TS Momentum', 'params': f'lookback={TS_LOOKBACK}'}


class BollingerBreakout:
    @staticmethod
    def compute_signals(df):
        return bollinger_breakout_signals(df)

    @staticmethod
    def get_strategy_info():
        return {'name': 'Bollinger Breakout', 'params': f'period={BB_PERIOD}, std={BB_STD}'}


class DonchianBreakout:
    @staticmethod
    def compute_signals(df):
        return donchian_breakout_signals(df)

    @staticmethod
    def get_strategy_info():
        return {'name': 'Donchian Breakout', 'params': f'entry={DONCHIAN_ENTRY}, exit={DONCHIAN_EXIT}'}


BENCHMARK_STRATEGIES = {
    'ma_crossover': MACrossover,
    'ts_momentum': TSMomentum,
    'bollinger_breakout': BollingerBreakout,
    'donchian_breakout': DonchianBreakout,
}


def compute_signals(df: pd.DataFrame, strategy_name: str = 'ma_crossover') -> pd.DataFrame:
    """Generic interface for benchmark strategies."""
    if strategy_name in BENCHMARK_STRATEGIES:
        return BENCHMARK_STRATEGIES[strategy_name].compute_signals(df)
    else:
        raise ValueError(f"Unknown benchmark: {strategy_name}")
