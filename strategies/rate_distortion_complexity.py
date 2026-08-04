"""
Rate-Distortion Complexity Trading
===================================

MATHEMATICAL DOMAIN: Information Theory / Rate-Distortion Theory

NOVELTY CLAIM:
--------------
Uses rate-distortion theory to measure the "compressibility" of price movements.
When prices follow simple patterns (trending), they compress well (low rate).
When prices are complex/noisy, they don't compress (high rate).

The insight: Shannon's rate-distortion function tells us the minimum bits
needed to describe data within a given distortion. Low rate = predictable
patterns = tradeable. High rate = noise = stay out.

IMPLEMENTATION:
---------------
1. Compute "compressibility" via reconstruction error at different resolutions
2. Rate = how fast error decreases with more data points
3. Low rate (high compressibility) = simple trend = follow it
4. High rate (low compressibility) = complex = don't trade

DISCOVERY TYPE: LLM-generated

WHY NOT PSEUDO-NOVEL:
---------------------
- Rate-distortion from Shannon's information theory (1959)
- Used in video compression, communications
- Novel: using compression ratio as market regime indicator
- Different from entropy: measures describability at fixed accuracy
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


# =============================================================================
# PARAMETERS
# =============================================================================
WINDOW = 48                  # Window for complexity analysis
RESOLUTIONS = [4, 8, 16, 32] # Different compression levels
SIMPLE_THRESHOLD = 0.7       # Above this = simple/compressible
COMPLEX_THRESHOLD = 0.4      # Below this = complex/noisy
MOMENTUM_WINDOW = 24         # Trend direction
CONFIRM_BARS = 3             # Bars to confirm regime


def compute_reconstruction_error(prices: np.ndarray, n_points: int) -> float:
    """
    Compute reconstruction error when representing prices with n_points.

    Lower error with fewer points = more compressible.
    """
    if len(prices) < n_points or n_points < 2:
        return 1.0

    # Downsample to n_points and interpolate back
    indices = np.linspace(0, len(prices)-1, n_points).astype(int)
    sampled = prices[indices]

    # Interpolate back to original length
    reconstructed = np.interp(
        np.arange(len(prices)),
        indices,
        sampled
    )

    # Normalized reconstruction error
    error = np.mean((prices - reconstructed)**2)
    variance = np.var(prices) + 1e-10
    return error / variance


def compute_compressibility(prices: np.ndarray) -> float:
    """
    Compute compressibility score based on rate-distortion curve.

    Score = how much the reconstruction error decreases with more points.
    High score = error drops quickly = simple pattern = compressible.
    """
    if len(prices) < max(RESOLUTIONS):
        return 0.5

    errors = []
    for res in RESOLUTIONS:
        err = compute_reconstruction_error(prices, res)
        errors.append(err)

    # Compressibility = how fast error decreases
    # Normalize by initial error
    if errors[0] < 1e-10:
        return 1.0

    # Compute area under rate-distortion curve (lower = more compressible)
    # We invert this so higher = more compressible
    area = np.trapz(errors, RESOLUTIONS)
    max_area = errors[0] * (RESOLUTIONS[-1] - RESOLUTIONS[0])

    compressibility = 1.0 - (area / (max_area + 1e-10))
    return np.clip(compressibility, 0, 1)


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trading signals based on rate-distortion complexity.
    """
    close = df['close'].values
    n = len(close)

    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)

    warmup = WINDOW + 50
    in_position = False
    simple_count = 0

    for i in range(warmup, n):
        # Compute compressibility of recent prices
        recent = close[i-WINDOW:i+1]
        compressibility = compute_compressibility(recent)

        # Momentum
        momentum = (close[i] - close[i-MOMENTUM_WINDOW]) / close[i-MOMENTUM_WINDOW]

        if compressibility > SIMPLE_THRESHOLD:
            # Simple/compressible pattern - tradeable
            simple_count += 1

            if simple_count >= CONFIRM_BARS:
                if momentum > 0.0005 and not in_position:
                    entries[i] = True
                    in_position = True
                elif momentum < -0.0005 and in_position:
                    exits[i] = True
                    in_position = False

        elif compressibility < COMPLEX_THRESHOLD:
            # Complex/noisy - stay out
            simple_count = 0

            if in_position:
                exits[i] = True
                in_position = False

        else:
            # Neutral
            simple_count = max(0, simple_count - 1)

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
        'window': WINDOW,
        'resolutions': RESOLUTIONS,
        'simple_threshold': SIMPLE_THRESHOLD,
        'complex_threshold': COMPLEX_THRESHOLD,
        'momentum_window': MOMENTUM_WINDOW,
        'confirm_bars': CONFIRM_BARS,
    }


def get_strategy_info() -> Dict[str, Any]:
    return {
        'name': 'Rate-Distortion Complexity Trading',
        'domain': 'Information Theory (Rate-Distortion)',
        'type': 'regime_adaptive',
        'direction': 'long_only',
        'novelty_claim': (
            'Uses rate-distortion to measure price compressibility. '
            'Compressible = simple pattern = tradeable. '
            'Incompressible = noise = stay out.'
        ),
    }
