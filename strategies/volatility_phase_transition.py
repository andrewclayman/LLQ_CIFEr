"""
Volatility Phase Transition Detector
=====================================

MATHEMATICAL DOMAIN: Statistical Physics / Phase Transitions

NOVELTY CLAIM:
--------------
Models volatility as an order parameter in a phase transition. Markets
alternate between "ordered" (low vol, trending) and "disordered" (high vol,
chaotic) phases. By detecting phase transitions, we can anticipate regime
changes before they fully manifest.

The insight: Near critical points (phase transitions), systems exhibit
characteristic behavior: increased correlation length, critical slowing.
We detect these precursors to anticipate volatility spikes.

IMPLEMENTATION:
---------------
1. Compute rolling volatility as "order parameter"
2. Measure correlation length (how long vol shocks persist)
3. Detect critical slowing (autocorrelation increase before transition)
4. Enter during ordered phase, exit before/during transition

DISCOVERY TYPE: LLM-generated

WHY NOT PSEUDO-NOVEL:
---------------------
- Phase transitions from statistical mechanics (Ising model, etc.)
- Critical phenomena well-studied in physics (universality, scaling)
- Novel: applying critical slowing down as trading signal
- Different from volatility trading: predicts transitions, not just levels
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


# =============================================================================
# PARAMETERS
# =============================================================================
VOL_WINDOW = 24              # Rolling volatility window
CORRELATION_WINDOW = 48      # Window for autocorrelation
CRITICAL_THRESHOLD = 0.7     # Autocorrelation threshold for critical slowing
ORDER_VOL_PERCENTILE = 40    # Below this = ordered phase
DISORDER_VOL_PERCENTILE = 75 # Above this = disordered phase
MOMENTUM_WINDOW = 24         # Trend direction


def compute_autocorrelation(series: np.ndarray, lag: int = 1) -> float:
    """Compute autocorrelation at given lag."""
    n = len(series)
    if n < lag + 5:
        return 0.0

    mean = np.mean(series)
    var = np.var(series)
    if var < 1e-10:
        return 0.0

    cov = np.mean((series[:-lag] - mean) * (series[lag:] - mean))
    return cov / var


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trading signals based on volatility phase transitions.
    """
    close = df['close'].values
    n = len(close)

    # Compute returns
    returns = np.zeros(n)
    returns[1:] = np.diff(close) / close[:-1]

    # Rolling volatility (order parameter)
    volatility = np.zeros(n)
    for i in range(VOL_WINDOW, n):
        volatility[i] = np.std(returns[i-VOL_WINDOW:i])

    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)

    warmup = max(VOL_WINDOW, CORRELATION_WINDOW) + 100
    in_position = False
    vol_history = []
    ordered_count = 0

    for i in range(warmup, n):
        vol = volatility[i]
        vol_history.append(vol)

        if len(vol_history) > 200:
            vol_history.pop(0)

        if len(vol_history) < 50:
            continue

        # Determine volatility regime thresholds
        order_threshold = np.percentile(vol_history, ORDER_VOL_PERCENTILE)
        disorder_threshold = np.percentile(vol_history, DISORDER_VOL_PERCENTILE)

        # Compute autocorrelation of recent volatility (critical slowing indicator)
        recent_vol = np.array(vol_history[-CORRELATION_WINDOW:])
        autocorr = compute_autocorrelation(recent_vol, lag=1)

        # Detect critical slowing: high autocorrelation approaching disorder
        is_critical = (autocorr > CRITICAL_THRESHOLD and
                       vol > order_threshold and
                       vol < disorder_threshold)

        # Classify phase
        is_ordered = vol < order_threshold
        is_disordered = vol > disorder_threshold

        # Momentum
        momentum = (close[i] - close[i-MOMENTUM_WINDOW]) / close[i-MOMENTUM_WINDOW]

        if is_ordered:
            # Ordered phase - safe to trade with trend
            ordered_count += 1

            if ordered_count >= 4 and not in_position and momentum > 0.0005:
                entries[i] = True
                in_position = True
            elif in_position and momentum < -0.0005:
                exits[i] = True
                in_position = False

        elif is_critical:
            # Critical region - approaching phase transition
            ordered_count = 0

            # Exit positions before transition
            if in_position:
                exits[i] = True
                in_position = False

        elif is_disordered:
            # Disordered phase - stay out
            ordered_count = 0

            if in_position:
                exits[i] = True
                in_position = False

        else:
            # Neutral - reduce count gradually
            ordered_count = max(0, ordered_count - 1)

    # Shift by 1 to avoid lookahead
    entries = np.roll(entries, 1)
    exits = np.roll(exits, 1)
    entries[0] = False
    exits[0] = False

    return pd.DataFrame({
        'entries': entries,
        'exits': exits
    }, index=df.index if hasattr(df, 'index') else None)


def get_parameters() -> Dict[str, Any]:
    return {
        'vol_window': VOL_WINDOW,
        'correlation_window': CORRELATION_WINDOW,
        'critical_threshold': CRITICAL_THRESHOLD,
        'order_vol_percentile': ORDER_VOL_PERCENTILE,
        'disorder_vol_percentile': DISORDER_VOL_PERCENTILE,
        'momentum_window': MOMENTUM_WINDOW,
    }


def get_strategy_info() -> Dict[str, Any]:
    return {
        'name': 'Volatility Phase Transition Detector',
        'domain': 'Statistical Physics',
        'type': 'regime_adaptive',
        'direction': 'long_only',
        'novelty_claim': (
            'Models market as phase transition system. '
            'Detects critical slowing down before volatility spikes. '
            'Trades in ordered phase, exits before transitions.'
        ),
    }
