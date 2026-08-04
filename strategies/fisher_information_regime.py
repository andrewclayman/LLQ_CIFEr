"""
Fisher Information Regime Detection
====================================

MATHEMATICAL DOMAIN: Information Geometry / Statistical Estimation

NOVELTY CLAIM:
--------------
Uses Fisher Information to measure the "curvature" of the return distribution.
High Fisher Information = distribution is highly sensitive to parameter changes
= unstable regime. Low Fisher Information = stable, predictable regime.

The insight: Fisher Information quantifies how much information about the
underlying parameter (trend/volatility) is contained in the data. When it
spikes, the market is in a transition state - dangerous for trend-following.

IMPLEMENTATION:
---------------
1. Estimate return distribution parameters (mean, variance) over rolling window
2. Compute Fisher Information matrix eigenvalues
3. High Fisher Info = unstable = reduce exposure or exit
4. Low Fisher Info = stable = follow momentum

DISCOVERY TYPE: LLM-generated

WHY NOT PSEUDO-NOVEL:
---------------------
- Fisher Information is from statistical estimation theory (R.A. Fisher, 1920s)
- Used in ML for natural gradient descent (Amari, 1998)
- Connection to information geometry (Rao, 1945)
- Novel application: regime detection for trading signals
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


# =============================================================================
# PARAMETERS
# =============================================================================
ESTIMATION_WINDOW = 48       # Window for parameter estimation
FISHER_WINDOW = 24           # Window for Fisher Info computation
THRESHOLD_PERCENTILE = 75    # Above this = unstable regime
STABILITY_BARS = 4           # Bars of low Fisher before entry
MOMENTUM_WINDOW = 24         # Momentum direction


def compute_fisher_information(returns: np.ndarray) -> float:
    """
    Compute Fisher Information for Gaussian model.

    For Gaussian with parameters (mu, sigma^2), the Fisher Information matrix is:
    I = [[1/sigma^2, 0], [0, 1/(2*sigma^4)]]

    We use the trace (sum of eigenvalues) as a scalar measure.
    High trace = high sensitivity to parameter changes = unstable.
    """
    n = len(returns)
    if n < 5:
        return 0.0

    mu = np.mean(returns)
    sigma2 = np.var(returns) + 1e-10

    # Fisher Information matrix diagonal elements for Gaussian
    I_mu = 1.0 / sigma2           # Sensitivity to mean
    I_sigma2 = 1.0 / (2 * sigma2**2)  # Sensitivity to variance

    # Use trace as scalar measure (sum of eigenvalues)
    fisher_trace = I_mu + I_sigma2

    # Also consider empirical deviation from Gaussian (excess info)
    # Kurtosis > 3 means fat tails = more information needed
    if n >= 10:
        kurtosis = np.mean(((returns - mu) / np.sqrt(sigma2))**4)
        excess_kurtosis = max(0, kurtosis - 3)
        fisher_trace *= (1 + 0.1 * excess_kurtosis)

    return fisher_trace


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trading signals based on Fisher Information regime detection.
    """
    close = df['close'].values
    n = len(close)

    # Compute returns
    returns = np.zeros(n)
    returns[1:] = np.diff(close) / close[:-1]

    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)

    warmup = ESTIMATION_WINDOW + 50
    in_position = False
    fisher_history = []
    stability_count = 0

    for i in range(warmup, n):
        # Get recent returns for Fisher computation
        recent_returns = returns[i - FISHER_WINDOW:i]

        # Compute Fisher Information
        fisher = compute_fisher_information(recent_returns)
        fisher_history.append(fisher)

        if len(fisher_history) > 200:
            fisher_history.pop(0)

        if len(fisher_history) < 30:
            continue

        # Adaptive threshold based on recent history
        threshold = np.percentile(fisher_history, THRESHOLD_PERCENTILE)

        # Regime classification
        is_unstable = fisher > threshold

        # Momentum for direction
        momentum = np.mean(returns[i - MOMENTUM_WINDOW:i])

        if not is_unstable:
            stability_count += 1

            # Stable regime - follow momentum
            if stability_count >= STABILITY_BARS:
                if momentum > 0.0001 and not in_position:
                    entries[i] = True
                    in_position = True
                elif momentum < -0.0001 and in_position:
                    exits[i] = True
                    in_position = False
        else:
            # Unstable regime - exit for safety
            stability_count = 0
            if in_position:
                exits[i] = True
                in_position = False

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
        'estimation_window': ESTIMATION_WINDOW,
        'fisher_window': FISHER_WINDOW,
        'threshold_percentile': THRESHOLD_PERCENTILE,
        'stability_bars': STABILITY_BARS,
        'momentum_window': MOMENTUM_WINDOW,
    }


def get_strategy_info() -> Dict[str, Any]:
    return {
        'name': 'Fisher Information Regime Detection',
        'domain': 'Information Geometry / Statistical Estimation',
        'type': 'regime_adaptive',
        'direction': 'long_only',
        'novelty_claim': (
            'Uses Fisher Information to detect regime instability. '
            'High Fisher Info = parameter sensitivity = dangerous transitions. '
            'Novel application of estimation theory to trading signals.'
        ),
    }
