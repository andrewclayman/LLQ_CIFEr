"""
Strategy Wrapper
=================
Provides a unified interface to run any strategy from the registry.
Imports directly from the existing PhD strategy implementations.

GA Element: This is the PHENOTYPE expression - each gene (strategy index)
maps to a concrete trading signal generator.
"""
import sys
import os
import importlib
import pandas as pd
import numpy as np

from config import STRATEGY_DIR, BENCHMARK_DIR, STRATEGY_REGISTRY

# Add strategy directories to path for importing
sys.path.insert(0, STRATEGY_DIR)
sys.path.insert(0, BENCHMARK_DIR)

# Cache loaded modules
_module_cache = {}

# Cache computed signals: (strategy_idx, data_id) -> signals DataFrame
# This avoids recomputing the same strategy on the same data across chromosomes
_signal_cache = {}


def clear_signal_cache():
    """Clear the signal cache (call between walk-forward periods)."""
    _signal_cache.clear()


def _load_strategy_module(module_name: str):
    """Load a strategy module by name, with caching."""
    if module_name in _module_cache:
        return _module_cache[module_name]

    try:
        mod = importlib.import_module(module_name)
        _module_cache[module_name] = mod
        return mod
    except Exception as e:
        print(f"  WARNING: Could not load strategy '{module_name}': {e}")
        return None


def run_strategy(strategy_idx: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run a strategy by its registry index on the given OHLCV DataFrame.

    Returns DataFrame with boolean 'entries' and 'exits' columns.
    If strategy fails, returns all-False signals (equivalent to staying flat).

    GA Element: This is the GENOTYPE -> PHENOTYPE mapping.
    Each integer gene maps to a concrete strategy execution.
    """
    if strategy_idx not in STRATEGY_REGISTRY:
        return _flat_signals(df)

    # Check signal cache
    cache_key = (strategy_idx, id(df))
    if cache_key in _signal_cache:
        return _signal_cache[cache_key]

    module_name, display_name, is_benchmark = STRATEGY_REGISTRY[strategy_idx]

    # Special case: Buy & Hold
    if module_name == 'hold':
        result = _buy_and_hold_signals(df)
        _signal_cache[cache_key] = result
        return result

    # Special case: Cash (stay flat — never trade)
    if module_name == 'flat':
        result = _flat_signals(df)
        _signal_cache[cache_key] = result
        return result

    # Benchmark strategies use a different interface
    if ':' in module_name:
        base_module, strat_name = module_name.split(':')
        mod = _load_strategy_module(base_module)
        if mod is None:
            result = _flat_signals(df)
            _signal_cache[cache_key] = result
            return result
        try:
            signals = mod.compute_signals(df, strategy_name=strat_name)
            result = _sanitize_signals(signals, df)
            _signal_cache[cache_key] = result
            return result
        except Exception as e:
            result = _flat_signals(df)
            _signal_cache[cache_key] = result
            return result

    # Standard strategy modules
    mod = _load_strategy_module(module_name)
    if mod is None:
        result = _flat_signals(df)
        _signal_cache[cache_key] = result
        return result

    try:
        signals = mod.compute_signals(df)
        result = _sanitize_signals(signals, df)
        _signal_cache[cache_key] = result
        return result
    except Exception as e:
        result = _flat_signals(df)
        _signal_cache[cache_key] = result
        return result


def _sanitize_signals(signals: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Ensure signals are properly formatted boolean arrays matching df index."""
    entries = signals['entries'].values.astype(bool) if 'entries' in signals.columns else np.zeros(len(df), dtype=bool)
    exits = signals['exits'].values.astype(bool) if 'exits' in signals.columns else np.zeros(len(df), dtype=bool)

    # Ensure same length
    if len(entries) != len(df):
        entries = np.zeros(len(df), dtype=bool)
        exits = np.zeros(len(df), dtype=bool)

    return pd.DataFrame({
        'entries': entries,
        'exits': exits
    }, index=df.index)


def _flat_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Return all-false signals (stay flat / no trading)."""
    return pd.DataFrame({
        'entries': np.zeros(len(df), dtype=bool),
        'exits': np.zeros(len(df), dtype=bool)
    }, index=df.index)


def _buy_and_hold_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Buy on first bar, never exit."""
    entries = np.zeros(len(df), dtype=bool)
    exits = np.zeros(len(df), dtype=bool)
    if len(df) > 1:
        entries[1] = True  # Enter on second bar
    return pd.DataFrame({
        'entries': entries,
        'exits': exits
    }, index=df.index)


def get_strategy_name(strategy_idx: int) -> str:
    """Get display name for a strategy index."""
    if strategy_idx in STRATEGY_REGISTRY:
        return STRATEGY_REGISTRY[strategy_idx][1]
    return f"Unknown({strategy_idx})"
