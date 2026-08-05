"""
Multi-Run Statistical Evaluator
================================

Runs the GA N times (default 50) per variant per period for statistical
robustness.  Precomputes strategy returns once per period; each GA run
then operates on cached matrix data only.

Results saved incrementally after each (period, variant) block so progress
is not lost on interruption.

Outputs:
  - multi_run_raw.csv:        Per-run results (period, variant, run, metrics)
  - multi_run_statistics.csv:  Aggregate stats per (period, variant)
"""
import os
import csv
import json
import time
import logging
import numpy as np
import pandas as pd

from config import (
    ASSETS, N_STRATEGIES,
    FULL_STRATEGY_INDICES, LLM_STRATEGY_INDICES, BENCHMARK_STRATEGY_INDICES,
    WALK_FORWARD_PERIODS, OUTPUT_DIR, FINAL_N_RUNS, TRADING_DAYS_PER_YEAR,
)
from data_loader import load_asset_daily, preload_all
from strategies import get_strategy_name, clear_signal_cache
from run_portfolio_ga import precompute_returns
from ga_engine import GAEngine, ann_sharpe
from hyperparameter_tuning import load_tuned_params

log = logging.getLogger(__name__)

VARIANTS = [
    ('Full', FULL_STRATEGY_INDICES),
    ('LLM', LLM_STRATEGY_INDICES),
    ('Benchmark', BENCHMARK_STRATEGY_INDICES),
]


def _compute_extended_metrics(daily_returns):
    """Compute extended financial metrics from daily return array."""
    n = len(daily_returns)
    if n < 30:
        return {
            'sharpe': -10.0, 'ann_return': 0.0, 'ann_volatility': 0.0,
            'sortino': 0.0, 'calmar': 0.0, 'max_drawdown': 0.0,
            'total_return': 0.0, 'win_rate': 0.0,
        }

    mu = np.mean(daily_returns)
    sigma = np.std(daily_returns)
    sharpe = (mu / sigma) * np.sqrt(TRADING_DAYS_PER_YEAR) if sigma > 0 else 0.0

    ann_return = mu * TRADING_DAYS_PER_YEAR
    ann_vol = sigma * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Sortino (downside deviation)
    downside = daily_returns[daily_returns < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 1e-9
    sortino = (mu / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR) if downside_std > 0 else 0.0

    # Max drawdown
    cum = np.cumprod(1 + daily_returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    # Calmar (ann return / abs(max drawdown))
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0

    total_return = float(np.prod(1 + daily_returns) - 1)
    win_rate = float(np.mean(daily_returns > 0)) if n > 0 else 0.0

    return {
        'sharpe': round(sharpe, 4),
        'ann_return': round(ann_return, 4),
        'ann_volatility': round(ann_vol, 4),
        'sortino': round(sortino, 4),
        'calmar': round(calmar, 4),
        'max_drawdown': round(max_dd, 4),
        'total_return': round(total_return, 4),
        'win_rate': round(win_rate, 4),
    }


def _evaluate_chromosome_oos(chromo, oos_strat, oos_bh, assets):
    """Evaluate a chromosome on OOS data. Returns (metrics_dict, daily_returns_array)."""
    strat_cols = {}
    for i, asset in enumerate(assets):
        key = (asset, int(chromo.strategy_genes[i]))
        strat_cols[asset] = oos_strat.get(key)

    bh_cols = {a: oos_bh.get(a) for a in assets}

    valid = [s for s in list(strat_cols.values()) + list(bh_cols.values())
             if s is not None and len(s) > 0]
    if not valid:
        return None, None

    common_idx = valid[0].index
    for s in valid[1:]:
        common_idx = common_idx.union(s.index)
    common_idx = common_idx.sort_values()

    strat_df = pd.DataFrame(strat_cols).reindex(common_idx).fillna(0.0)
    portfolio_ret = (strat_df.values * chromo.weight_genes).sum(axis=1)

    metrics = _compute_extended_metrics(portfolio_ret)

    # Train Sharpe is stored in chromo.fitness
    metrics['train_sharpe'] = round(chromo.fitness, 4) if chromo.fitness else 0.0

    return metrics, portfolio_ret


def _run_data_path(period, variant):
    """Path for saved oos_returns arrays."""
    return os.path.join(OUTPUT_DIR, f'run_data_p{period}_{variant}.npz')


def _save_run_data(period, variant, period_variant_data):
    """Save oos_returns arrays to disk so plots work on resume."""
    arrays = {}
    for i, rd in enumerate(period_variant_data):
        if rd['oos_returns'] is not None:
            arrays[f'oos_{i}'] = np.asarray(rd['oos_returns'])
    if arrays:
        np.savez_compressed(_run_data_path(period, variant), **arrays)


def _load_run_data(period, variant, n_runs):
    """Load saved oos_returns arrays from disk."""
    path = _run_data_path(period, variant)
    if not os.path.exists(path):
        return []
    data = np.load(path)
    result = []
    for i in range(n_runs):
        key = f'oos_{i}'
        if key in data:
            result.append({
                'best_history': [],
                'avg_history': [],
                'oos_returns': data[key],
            })
    data.close()
    return result


def run_multi_evaluation(n_runs=None, ga_params=None):
    """Run multi-run evaluation across all periods and variants.

    Args:
        n_runs: number of GA runs per variant per period (default: FINAL_N_RUNS)
        ga_params: per-period tuned params dict {period_num: params} or single
                   params dict (applied to all periods). Loaded from JSON if None.

    Returns:
        (raw_rows, stat_rows, run_data) where run_data has fitness histories
    """
    if n_runs is None:
        n_runs = FINAL_N_RUNS

    if ga_params is None:
        ga_params = load_tuned_params()
        if ga_params is None:
            log.warning("No tuned params found, using defaults")
            ga_params = {}

    # Normalise to per-period format: {period_num: params}
    # If it's a flat dict (old format / single set), apply to all periods
    if ga_params and not any(isinstance(v, dict) for v in ga_params.values()):
        ga_params = {i: ga_params for i in range(1, len(WALK_FORWARD_PERIODS) + 1)}

    log.info("=" * 70)
    log.info(f"MULTI-RUN EVALUATION ({n_runs} runs x {len(WALK_FORWARD_PERIODS)} periods x {len(VARIANTS)} variants)")
    log.info(f"Per-period tuned params: {len(ga_params)} periods")
    log.info("=" * 70)

    preload_all()

    raw_rows = []
    run_data = {}  # (period, variant) -> list of {best_history, avg_history, oos_returns}
    header_written = False

    # Check for existing progress to resume from
    raw_csv_path = os.path.join(OUTPUT_DIR, 'multi_run_raw.csv')
    completed_keys = set()
    if os.path.exists(raw_csv_path):
        try:
            existing = pd.read_csv(raw_csv_path)
            for (p, v), grp in existing.groupby(['period', 'variant']):
                if len(grp) >= n_runs:
                    completed_keys.add((int(p), v))
            if completed_keys:
                raw_rows = existing.to_dict('records')
                header_written = True
                log.info(f"Resuming: {len(completed_keys)} (period, variant) blocks already done")
                # Reload saved oos_returns for completed blocks so plots work
                for (p, v) in completed_keys:
                    run_data[(p, v)] = _load_run_data(p, v, n_runs)
        except Exception:
            pass

    for p_idx, (train_start, train_end, test_start, test_end) in enumerate(WALK_FORWARD_PERIODS):
        period_num = p_idx + 1

        # Skip if all 3 variants done for this period
        if all((period_num, v) in completed_keys for v, _ in VARIANTS):
            log.info(f"\nPeriod {period_num}: already complete, skipping")
            continue

        log.info(f"\nPeriod {period_num}: Train {train_start[:4]}-{train_end[:4]}, Test {test_start[:4]}")

        # Load data
        available_assets = []
        train_data = {}
        test_data = {}

        for asset in ASSETS:
            daily = load_asset_daily(asset)
            if daily.empty:
                continue
            train_df = daily[(daily.index >= train_start) & (daily.index <= train_end)].copy()
            test_df = daily[(daily.index >= test_start) & (daily.index <= test_end)].copy()
            if len(train_df) >= 100 and len(test_df) >= 50:
                available_assets.append(asset)
                train_data[asset] = train_df
                test_data[asset] = test_df

        log.info(f"  {len(available_assets)} assets, precomputing returns...")
        t0 = time.time()
        train_strat, train_bh, _ = precompute_returns(available_assets, train_data)
        oos_strat, oos_bh, _ = precompute_returns(available_assets, test_data)
        log.info(f"  Precompute done in {time.time() - t0:.1f}s")

        for variant_name, allowed_indices in VARIANTS:
            if (period_num, variant_name) in completed_keys:
                log.info(f"  {variant_name} variant: already complete, skipping")
                continue

            log.info(f"  Running {variant_name} variant ({n_runs} runs)...")
            t0 = time.time()

            period_variant_data = []
            variant_rows = []

            # Get per-period params (or empty dict for defaults)
            period_ga_params = ga_params.get(period_num, {})

            for run_i in range(n_runs):
                seed = period_num * 10000 + hash(variant_name) % 1000 + run_i
                ga = GAEngine(
                    available_assets, train_strat, train_bh,
                    allowed_strategies=allowed_indices,
                    seed=seed,
                    ga_params=period_ga_params,
                )
                best, best_history, avg_history = ga.run()

                metrics, oos_returns = _evaluate_chromosome_oos(
                    best, oos_strat, oos_bh, available_assets)

                if metrics is None:
                    continue

                row = {
                    'period': period_num,
                    'variant': variant_name,
                    'run': run_i + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'n_assets': len(available_assets),
                    **metrics,
                }
                raw_rows.append(row)
                variant_rows.append(row)

                period_variant_data.append({
                    'best_history': best_history,
                    'avg_history': avg_history,
                    'oos_returns': oos_returns,
                })

                if (run_i + 1) % 10 == 0:
                    log.info(f"    Run {run_i+1}/{n_runs}: OOS Sharpe={metrics['sharpe']:+.4f}")

            run_data[(period_num, variant_name)] = period_variant_data
            _save_run_data(period_num, variant_name, period_variant_data)
            elapsed = time.time() - t0
            log.info(f"  {variant_name} done in {elapsed:.1f}s")

            # ── Incremental save after each variant block ──
            _append_rows(raw_csv_path, variant_rows, write_header=(not header_written))
            header_written = True
            # Also update statistics
            stat_rows = _compute_statistics(raw_rows)
            _save_statistics(stat_rows)
            log.info(f"  Progress saved ({len(raw_rows)} total rows)")

    # Final statistics
    stat_rows = _compute_statistics(raw_rows)
    _save_statistics(stat_rows)

    return raw_rows, stat_rows, run_data


def _compute_statistics(raw_rows):
    """Compute mean/std/min/max per (period, variant)."""
    df = pd.DataFrame(raw_rows)
    metrics = ['train_sharpe', 'sharpe', 'total_return', 'max_drawdown',
               'sortino', 'calmar', 'ann_volatility', 'win_rate', 'ann_return']

    stat_rows = []
    for (period, variant), group in df.groupby(['period', 'variant']):
        row = {'period': period, 'variant': variant, 'n_runs': len(group)}
        for m in metrics:
            if m in group.columns:
                vals = group[m].values
                row[f'{m}_mean'] = round(np.mean(vals), 4)
                row[f'{m}_std'] = round(np.std(vals), 4)
                row[f'{m}_min'] = round(np.min(vals), 4)
                row[f'{m}_max'] = round(np.max(vals), 4)
        stat_rows.append(row)

    return stat_rows


def _append_rows(path, rows, write_header=False):
    """Append rows to a CSV file. Write header if file is new."""
    if not rows:
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mode = 'w' if write_header else 'a'
    with open(path, mode, newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if write_header:
            w.writeheader()
        w.writerows(rows)


def _save_statistics(stat_rows):
    """Overwrite statistics CSV with current aggregate stats."""
    if not stat_rows:
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, 'multi_run_statistics.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=stat_rows[0].keys())
        w.writeheader()
        w.writerows(stat_rows)
    log.info(f"Saved: {path}")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        handlers=[logging.StreamHandler()],
    )
    run_multi_evaluation()
