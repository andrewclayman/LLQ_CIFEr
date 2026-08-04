"""
Walk-forward validation of all strategies on daily data.
Runs the 8 LLM-generated strategies + 5 active benchmarks (+ Buy & Hold, computed
inline) across 19 assets x 4 OOS test periods. This is the Stage 1 harness.
Outputs combined results to results/walk_forward_results.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import importlib.util
import sys
import warnings
warnings.filterwarnings('ignore')

from config import (
    STRATEGIES_DIR, TEST_DIR, RESULTS_DIR, TEST_PERIODS,
    PERIODS_PER_YEAR, TRANSACTION_COSTS, RISK_FREE_RATES,
    ASSET_CLASSES, NOVEL_STRATEGIES, DEFAULT_COST
)

# All strategies to test (module_name -> list of (strategy_label, strategy_type))
# For most modules: one strategy per module
# For benchmarks.py: four strategies from one module
STRATEGY_MANIFEST = {}

# Novel strategies
for name in NOVEL_STRATEGIES:
    STRATEGY_MANIFEST[name] = [(name, 'novel')]

# Standalone benchmark strategies
for name in ['mean_reversion_bands_daily', 'rsi_divergence_daily']:
    STRATEGY_MANIFEST[name] = [(name, 'benchmark')]

# benchmarks.py: the three active benchmarks reported in the paper
STRATEGY_MANIFEST['benchmarks'] = [
    ('ma_crossover', 'benchmark'),
    ('bollinger_breakout', 'benchmark'),
    ('donchian_breakout', 'benchmark'),
]


def load_module(module_name):
    """Load a strategy module by name."""
    path = STRATEGIES_DIR / f"{module_name}.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_rf_rate(asset):
    """Get risk-free rate for asset's base currency."""
    if asset in ['EURUSD', 'DE40', 'F40', 'STOXX50']:
        return RISK_FREE_RATES.get('EUR', 0.0)
    elif asset in ['GBPUSD', 'UK100']:
        return RISK_FREE_RATES.get('GBP', 0.01)
    elif asset in ['USDJPY', 'JP225']:
        return RISK_FREE_RATES.get('JPY', -0.001)
    else:
        return RISK_FREE_RATES.get('USD', 0.02)


def calc_sharpe(returns, rf_annual=0.02):
    """Calculate annualized Sharpe ratio for daily data."""
    rf_period = (1 + rf_annual) ** (1 / PERIODS_PER_YEAR) - 1
    excess = returns - rf_period
    if np.std(excess) < 1e-10:
        return np.nan
    return np.mean(excess) / np.std(excess) * np.sqrt(PERIODS_PER_YEAR)


def run_backtest(df, entries, exits, tc):
    """Run long-only backtest with transaction costs."""
    close = df['close'].values
    n = len(close)
    position = 0
    equity = [1.0]
    returns_list = [0.0]
    trades = []
    entry_price = 0

    for i in range(1, n):
        pnl = 0
        if position == 1:
            pnl = (close[i] - close[i-1]) / close[i-1]

        if entries[i] and position == 0:
            position = 1
            entry_price = close[i]
            new_equity = equity[-1] * (1 - tc)
            returns_list.append((new_equity / equity[-1]) - 1)
            equity.append(new_equity)
        elif exits[i] and position == 1:
            position = 0
            new_equity = equity[-1] * (1 + pnl) * (1 - tc)
            returns_list.append((new_equity / equity[-1]) - 1)
            equity.append(new_equity)
            trades.append({'return': (close[i] - entry_price) / entry_price - 2 * tc})
        else:
            new_equity = equity[-1] * (1 + pnl)
            returns_list.append(pnl if position == 1 else 0)
            equity.append(new_equity)

    return np.array(equity), np.array(returns_list), trades


def get_signals(module, strategy_label, module_name, df):
    """Get entry/exit signals from a strategy module."""
    if module_name == 'benchmarks':
        # benchmarks.py uses class-based interface
        return module.BENCHMARK_STRATEGIES[strategy_label].compute_signals(df)
    else:
        return module.compute_signals(df)


def main():
    print("=" * 70)
    print("LLM STRATEGY GENERATION — WALK-FORWARD VALIDATION")
    print("=" * 70)
    print(f"Test periods: {TEST_PERIODS}")
    print(f"Annualization: {PERIODS_PER_YEAR} periods/year")
    print()

    all_results = []

    for module_name, strategy_entries in STRATEGY_MANIFEST.items():
        module = load_module(module_name)
        if module is None:
            print(f"WARNING: Module {module_name} not found, skipping")
            continue

        for strategy_label, strategy_type in strategy_entries:
            print(f"Testing {strategy_label} ({strategy_type})...")

            for asset_dir in sorted(TEST_DIR.iterdir()):
                if not asset_dir.is_dir():
                    continue
                asset = asset_dir.name
                asset_class = ASSET_CLASSES.get(asset, 'unknown')
                tc = TRANSACTION_COSTS.get(asset, DEFAULT_COST)
                rf = get_rf_rate(asset)

                for period in TEST_PERIODS:
                    test_file = asset_dir / f"{asset.lower()}_{period}.csv"
                    if not test_file.exists():
                        continue

                    try:
                        df = pd.read_csv(test_file, index_col=0, parse_dates=True)
                        if len(df) < 50:
                            continue

                        signals = get_signals(module, strategy_label, module_name, df)
                        equity, returns, trades = run_backtest(
                            df,
                            signals['entries'].values,
                            signals['exits'].values,
                            tc
                        )

                        sharpe = calc_sharpe(returns, 0.0)
                        total_return = (equity[-1] / equity[0] - 1) * 100
                        max_dd = np.min(equity / np.maximum.accumulate(equity) - 1) * 100

                        n_periods = len(returns)
                        ann_return = ((equity[-1] / equity[0]) ** (PERIODS_PER_YEAR / max(n_periods, 1)) - 1) * 100
                        ann_vol = np.std(returns) * np.sqrt(PERIODS_PER_YEAR) * 100

                        bh_returns = df['close'].pct_change().fillna(0).values
                        bh_sharpe = calc_sharpe(bh_returns, 0.0)

                        beats = (not np.isnan(sharpe) and not np.isnan(bh_sharpe)
                                 and sharpe > bh_sharpe and len(trades) >= 5)

                        all_results.append({
                            'strategy': strategy_label,
                            'strategy_type': strategy_type,
                            'asset': asset,
                            'asset_class': asset_class,
                            'period': period,
                            'sharpe': round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
                            'bh_sharpe': round(bh_sharpe, 3) if not np.isnan(bh_sharpe) else np.nan,
                            'beats_bh': beats,
                            'total_return': round(total_return, 2),
                            'ann_return': round(ann_return, 2),
                            'ann_vol': round(ann_vol, 2),
                            'max_drawdown': round(max_dd, 2),
                            'num_trades': len(trades),
                        })

                    except Exception as e:
                        print(f"  ERROR {strategy_label}/{asset}/{period}: {str(e)[:80]}")

            count = sum(1 for r in all_results if r['strategy'] == strategy_label)
            wins = sum(1 for r in all_results if r['strategy'] == strategy_label and r['beats_bh'])
            valid = [r['sharpe'] for r in all_results
                     if r['strategy'] == strategy_label and not np.isnan(r.get('sharpe', np.nan))]
            avg = np.mean(valid) if valid else np.nan
            print(f"  -> {count} tests, avg Sharpe: {avg:.3f}, beats B&H: {wins}/{count}")

    # Save results
    df_results = pd.DataFrame(all_results)
    output_file = RESULTS_DIR / "walk_forward_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved {len(df_results)} results to {output_file}")

    # Summary tables
    print("\n" + "=" * 70)
    print("SUMMARY BY STRATEGY")
    print("=" * 70)
    valid_df = df_results.dropna(subset=['sharpe'])
    summary = valid_df.groupby(['strategy', 'strategy_type']).agg(
        mean_sharpe=('sharpe', 'mean'),
        std_sharpe=('sharpe', 'std'),
        n_tests=('sharpe', 'count'),
        wins=('beats_bh', 'sum'),
        avg_trades=('num_trades', 'mean'),
    ).round(3)
    summary['beat_rate'] = (summary['wins'] / summary['n_tests'] * 100).round(1)
    summary = summary.sort_values('mean_sharpe', ascending=False)
    print(summary.to_string())

    print("\n" + "=" * 70)
    print("SUMMARY BY ASSET CLASS")
    print("=" * 70)
    novel_df = valid_df[valid_df['strategy_type'] == 'novel']
    class_summary = novel_df.groupby('asset_class').agg(
        mean_sharpe=('sharpe', 'mean'),
        n_tests=('sharpe', 'count'),
        wins=('beats_bh', 'sum'),
    ).round(3)
    class_summary['beat_rate'] = (class_summary['wins'] / class_summary['n_tests'] * 100).round(1)
    print(class_summary.to_string())

    print("\n" + "=" * 70)
    print("SUMMARY BY PERIOD")
    print("=" * 70)
    period_summary = novel_df.groupby('period').agg(
        mean_sharpe=('sharpe', 'mean'),
        n_tests=('sharpe', 'count'),
        wins=('beats_bh', 'sum'),
    ).round(3)
    period_summary['beat_rate'] = (period_summary['wins'] / period_summary['n_tests'] * 100).round(1)
    print(period_summary.to_string())


if __name__ == "__main__":
    main()
