"""
Backtester
===========
Converts entry/exit signals into P&L, computing Sharpe ratio, returns,
drawdown, and monthly breakdown.

All strategies are LONG-ONLY: enter = go long, exit = go flat.
"""
import pandas as pd
import numpy as np
from config import TRADING_DAYS_PER_YEAR


def backtest(df: pd.DataFrame, signals: pd.DataFrame) -> dict:
    """
    Run backtest on daily OHLCV data with entry/exit signals.

    Computes:
    - Total return, annualized return
    - Sharpe ratio (annualized, excess over 0% rf)
    - Max drawdown
    - Monthly returns breakdown
    - Number of trades
    - Win rate
    - Buy & hold comparison

    Returns dict with all metrics.
    """
    close = df['close'].values
    n = len(close)
    entries = signals['entries'].values
    exits = signals['exits'].values

    # Build position array: 1 = long, 0 = flat
    position = np.zeros(n)
    in_position = False
    n_trades = 0
    trade_returns = []
    trade_entry_price = 0

    for i in range(n):
        if entries[i] and not in_position:
            in_position = True
            trade_entry_price = close[i]
            n_trades += 1
        elif exits[i] and in_position:
            in_position = False
            if trade_entry_price > 0:
                trade_returns.append(close[i] / trade_entry_price - 1)

        position[i] = 1.0 if in_position else 0.0

    # Daily returns
    daily_returns = np.zeros(n)
    daily_returns[1:] = np.diff(close) / close[:-1]

    # Strategy returns: signal at end of day i takes effect for return earned
    # from day i to day i+1. So pair daily_returns[i] (close[i-1]->close[i])
    # with position[i-1] (state at end of day i-1).
    strategy_returns = np.zeros(n)
    strategy_returns[1:] = daily_returns[1:] * position[:-1]

    # Buy & hold returns
    bh_returns = daily_returns.copy()

    # Cumulative
    cum_strategy = np.cumprod(1 + strategy_returns)
    cum_bh = np.cumprod(1 + bh_returns)

    # Total return
    total_return = cum_strategy[-1] / cum_strategy[0] - 1 if cum_strategy[0] > 0 else 0
    bh_total_return = cum_bh[-1] / cum_bh[0] - 1 if cum_bh[0] > 0 else 0

    # Annualized
    n_years = n / TRADING_DAYS_PER_YEAR
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    bh_ann_return = (1 + bh_total_return) ** (1 / max(n_years, 0.01)) - 1

    # Sharpe ratio
    strat_std = np.std(strategy_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
    bh_std = np.std(bh_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = ann_return / strat_std if strat_std > 0 else 0
    bh_sharpe = bh_ann_return / bh_std if bh_std > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(cum_strategy)
    drawdown = (cum_strategy - peak) / peak
    max_drawdown = np.min(drawdown)

    # Win rate
    win_rate = np.mean([r > 0 for r in trade_returns]) if trade_returns else 0

    # Monthly returns
    monthly = _compute_monthly_returns(df.index, strategy_returns, bh_returns)

    return {
        'total_return': total_return,
        'ann_return': ann_return,
        'sharpe': sharpe,
        'bh_total_return': bh_total_return,
        'bh_ann_return': bh_ann_return,
        'bh_sharpe': bh_sharpe,
        'excess_sharpe': sharpe - bh_sharpe,
        'max_drawdown': max_drawdown,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'n_days': n,
        'monthly_returns': monthly,
        'beats_bh': sharpe > bh_sharpe,
    }


def _compute_monthly_returns(index, strategy_returns, bh_returns):
    """Compute month-by-month returns for both strategy and B&H."""
    if not hasattr(index, 'year'):
        return []

    months = []
    dates = pd.DatetimeIndex(index)
    unique_months = sorted(set(zip(dates.year, dates.month)))

    for year, month in unique_months:
        mask = (dates.year == year) & (dates.month == month)
        strat_monthly = np.prod(1 + strategy_returns[mask]) - 1
        bh_monthly = np.prod(1 + bh_returns[mask]) - 1
        months.append({
            'year': year,
            'month': month,
            'strategy_return': float(strat_monthly),
            'bh_return': float(bh_monthly),
            'excess_return': float(strat_monthly - bh_monthly),
        })

    return months
