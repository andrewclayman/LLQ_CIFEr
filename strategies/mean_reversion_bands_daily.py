"""
Mean Reversion Bands Daily Strategy

Mean reversion using Bollinger-style bands.
Enter when price touches lower band, exit at middle or upper band.

Mathematical basis: Price mean reversion within volatility bands.
"""

import pandas as pd
import numpy as np

def compute_signals(df, window=20, num_std=2.0):
    """
    Compute entry and exit signals based on price bands.

    Parameters:
    -----------
    df : DataFrame with OHLCV columns
    window : Window for moving average
    num_std : Number of standard deviations for bands

    Returns:
    --------
    DataFrame with 'entries' and 'exits' boolean columns
    """
    close = df['close']
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()

    lower = ma - num_std * std
    upper = ma + num_std * std

    entries = (close < lower) & (close.shift(1) >= lower.shift(1))
    exits = (close > ma) | (close > upper)

    return pd.DataFrame({
        'entries': entries.fillna(False),
        'exits': exits.fillna(False)
    })
