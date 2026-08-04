"""
RSI Divergence Daily Strategy

Detects bullish divergence between price and RSI.
Enter when price makes lower low but RSI makes higher low.

Mathematical basis: Momentum divergence as reversal signal.
"""

import pandas as pd
import numpy as np

def compute_signals(df, rsi_window=14, price_window=10):
    """
    Compute entry and exit signals based on RSI divergence.

    Parameters:
    -----------
    df : DataFrame with OHLCV columns
    rsi_window : Window for RSI calculation
    price_window : Window for price comparison

    Returns:
    --------
    DataFrame with 'entries' and 'exits' boolean columns
    """
    close = df['close']

    # Calculate RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Price lower low
    price_min = close.rolling(price_window).min()
    price_lower_low = (close == price_min) & (close < close.shift(price_window))

    # RSI higher low
    rsi_higher_low = rsi > rsi.shift(price_window)

    # Bullish divergence
    entries = price_lower_low & rsi_higher_low & (rsi < 40)
    exits = rsi > 70

    return pd.DataFrame({
        'entries': entries.fillna(False),
        'exits': exits.fillna(False)
    })
