"""
Per-Strategy OOS Evaluation
============================
Evaluates each strategy independently on each (asset, OOS-period) combination,
producing the per-strategy ranking table for the paper (Stage 1 of the
two-stage evaluation).

Uses the same 7 walk-forward OOS periods as the GA (2018, 2020, 2021, 2022,
2023, 2024, 2025) to keep both stages aligned. Strategy parameters are
fixed (LLM-chosen for novel strategies, grid-searched for benchmarks).

Outputs:
  - results/per_strategy_results.csv   one row per (strategy, asset, period)
  - results/per_strategy_summary.csv   mean Sharpe / return / DD per strategy
  - results/per_strategy_by_class.csv  mean Sharpe per (strategy, asset_class)
  - results/per_strategy_by_period.csv mean Sharpe per (strategy, period)
"""
import os
import time
import logging
import numpy as np
import pandas as pd

from config import (
    ASSETS, ASSET_CLASSES, STRATEGY_REGISTRY,
    FULL_STRATEGY_INDICES, ACTIVE_LLM_INDICES, PAPER_BENCHMARK_INDICES,
    WALK_FORWARD_PERIODS, OUTPUT_DIR, TRADING_DAYS_PER_YEAR,
    TRANSACTION_COSTS, DEFAULT_TRANSACTION_COST,
)
from data_loader import load_asset_daily, preload_all
from strategies import run_strategy, get_strategy_name, clear_signal_cache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def backtest_strategy(df, signals, asset):
    """Run long-only backtest on (df, entry/exit signals). Returns metrics dict.

    Execution model: signals[i] decided at end of day i, position takes effect
    from day i's close, earning the day-i-to-day-i+1 return.
    """
    close = df['close'].values
    if len(close) < 2:
        return None

    daily_ret = np.diff(close) / close[:-1]   # daily_ret[k]: close[k] -> close[k+1]
    cost = TRANSACTION_COSTS.get(asset, DEFAULT_TRANSACTION_COST)

    entries = signals['entries'].values
    exits = signals['exits'].values
    positions = np.zeros(len(close))
    tc = np.zeros(len(close))
    in_pos = False
    n_trades = 0
    for i in range(len(close)):
        if entries[i] and not in_pos:
            in_pos = True
            tc[i] = cost
            n_trades += 1
        elif exits[i] and in_pos:
            in_pos = False
            tc[i] = cost
        positions[i] = 1.0 if in_pos else 0.0

    # Pair daily_ret[k] with positions[k] (state at end of day k)
    strat_daily = daily_ret * positions[:-1] - tc[1:]

    n = len(strat_daily)
    if n < 5:
        return None

    mu = float(np.mean(strat_daily))
    sigma = float(np.std(strat_daily))
    sharpe = (mu / sigma) * np.sqrt(TRADING_DAYS_PER_YEAR) if sigma > 0 else 0.0
    ann_ret = mu * TRADING_DAYS_PER_YEAR
    ann_vol = sigma * np.sqrt(TRADING_DAYS_PER_YEAR)

    cum = np.cumprod(1 + strat_daily)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0
    total_ret = float(np.prod(1 + strat_daily) - 1)

    return {
        'n_bars': n,
        'sharpe': round(sharpe, 4),
        'ann_return': round(ann_ret, 4),
        'ann_volatility': round(ann_vol, 4),
        'max_drawdown': round(max_dd, 4),
        'total_return': round(total_ret, 4),
        'n_trades': n_trades,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preload_all()

    rows = []
    strat_indices = ACTIVE_LLM_INDICES + PAPER_BENCHMARK_INDICES  # 13 strategies, no Cash

    for p_idx, (_, _, test_start, test_end) in enumerate(WALK_FORWARD_PERIODS):
        period_num = p_idx + 1
        test_year = test_start[:4]
        log.info(f"\n=== Period {period_num} ({test_year}) ===")
        period_t0 = time.time()

        for asset in ASSETS:
            df_full = load_asset_daily(asset)
            if df_full.empty:
                continue
            mask = (df_full.index >= test_start) & (df_full.index <= test_end)
            test_df = df_full[mask].copy()
            if len(test_df) < 50:
                continue

            for strat_idx in strat_indices:
                clear_signal_cache()
                try:
                    sigs = run_strategy(strat_idx, test_df)
                    metrics = backtest_strategy(test_df, sigs, asset)
                    if metrics is None:
                        continue
                    is_llm = strat_idx in ACTIVE_LLM_INDICES
                    row = {
                        'period': period_num,
                        'test_year': test_year,
                        'asset': asset,
                        'asset_class': ASSET_CLASSES.get(asset, '?'),
                        'strategy_idx': strat_idx,
                        'strategy_name': get_strategy_name(strat_idx),
                        'strategy_type': 'LLM' if is_llm else 'Benchmark',
                        **metrics,
                    }
                    rows.append(row)
                except Exception as e:
                    log.warning(f"  {asset} idx={strat_idx} FAILED: {e}")

        log.info(f"  Period {period_num} done in {time.time()-period_t0:.1f}s "
                 f"({len(rows)} rows so far)")

    df = pd.DataFrame(rows)
    raw_path = os.path.join(OUTPUT_DIR, 'per_strategy_results.csv')
    df.to_csv(raw_path, index=False)
    log.info(f"\nSaved {len(df)} rows to {raw_path}")

    # ── Summary tables ─────────────────────────────────────────
    # Per-strategy mean across all (asset, period)
    summary = (df.groupby(['strategy_idx', 'strategy_name', 'strategy_type'])
                 .agg({'sharpe': 'mean', 'ann_return': 'mean',
                       'ann_volatility': 'mean', 'max_drawdown': 'mean',
                       'n_trades': 'mean'})
                 .round(4)
                 .reset_index()
                 .sort_values('sharpe', ascending=False))
    summary_path = os.path.join(OUTPUT_DIR, 'per_strategy_summary.csv')
    summary.to_csv(summary_path, index=False)
    log.info(f"Saved per-strategy summary to {summary_path}")
    log.info("\n=== Mean Sharpe per Strategy (sorted) ===")
    log.info(summary.to_string(index=False))

    # Per-strategy x asset_class
    by_class = (df.groupby(['strategy_idx', 'strategy_name', 'asset_class'])
                  ['sharpe'].mean().round(4).unstack().reset_index())
    by_class_path = os.path.join(OUTPUT_DIR, 'per_strategy_by_class.csv')
    by_class.to_csv(by_class_path, index=False)
    log.info(f"\nSaved by-class table to {by_class_path}")

    # Per-strategy x period
    by_period = (df.groupby(['strategy_idx', 'strategy_name', 'test_year'])
                   ['sharpe'].mean().round(4).unstack().reset_index())
    by_period_path = os.path.join(OUTPUT_DIR, 'per_strategy_by_period.csv')
    by_period.to_csv(by_period_path, index=False)
    log.info(f"Saved by-period table to {by_period_path}")


if __name__ == '__main__':
    main()
