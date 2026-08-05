"""
Portfolio-Level GA Runner
=========================

For each walk-forward period:
1. Precompute all 19x19 strategy-asset daily return series (361 backtests)
2. Run GA with portfolio-level fitness (Sharpe of combined weighted returns)
3. Evaluate best chromosome out-of-sample
4. Save results
"""
import os
import sys
import time
import csv
import argparse
import logging
import numpy as np
import pandas as pd

from config import (
    ASSETS, ASSET_CLASSES, STRATEGY_REGISTRY, N_STRATEGIES,
    FULL_STRATEGY_INDICES, LLM_STRATEGY_INDICES, BENCHMARK_STRATEGY_INDICES,
    POPULATION_SIZE, N_GENERATIONS, FINAL_N_RUNS,
    WALK_FORWARD_PERIODS, OUTPUT_DIR, LOG_DIR, TRADING_DAYS_PER_YEAR,
    TRANSACTION_COSTS, DEFAULT_TRANSACTION_COST,
)
from data_loader import load_asset_daily, preload_all
from strategies import run_strategy, get_strategy_name, clear_signal_cache
from backtester import backtest
from ga_engine import GAEngine, Chromosome, ann_sharpe

# ---------------------------------------------------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'portfolio_ga.log'), mode='w'),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def compute_daily_strategy_returns(df: pd.DataFrame, signals: pd.DataFrame,
                                   asset: str = None) -> pd.Series:
    """Compute daily returns of a strategy given price data and signals.

    Execution model: signals[i] is decided at end of day i (after seeing
    close[i]). Execution lag = 1 bar: position takes effect from day i+1's
    close onward, earning daily_ret[i+1] = (close[i+2] - close[i+1]) / close[i+1].

    Applies transaction costs (half-spread) on entry and exit days.
    """
    close = df['close'].values
    daily_ret = np.diff(close) / close[:-1]   # length n-1: daily_ret[k] is close[k]->close[k+1]
    cost = TRANSACTION_COSTS.get(asset, DEFAULT_TRANSACTION_COST) if asset else 0.0

    entries = signals['entries'].values
    exits = signals['exits'].values
    positions = np.zeros(len(close))
    tc = np.zeros(len(close))  # transaction cost deductions (paid on entry/exit day)
    in_pos = False
    for i in range(len(close)):
        if entries[i] and not in_pos:
            in_pos = True
            tc[i] = cost  # pay half-spread on entry
        elif exits[i] and in_pos:
            in_pos = False
            tc[i] = cost  # pay half-spread on exit
        positions[i] = 1.0 if in_pos else 0.0

    # daily_ret[k] is earned by positions[k] (state at end of day k, holding to day k+1).
    # No lookahead: positions[k] depends only on signals[0..k] which used data up to close[k].
    strat_daily = daily_ret * positions[:-1] - tc[1:]
    return pd.Series(strat_daily, index=df.index[1:])


def precompute_returns(assets, data_dict):
    """Precompute all strategy-asset daily returns + B&H returns."""
    strat_returns = {}  # (asset, strat_idx) -> pd.Series
    bh_returns = {}     # asset -> pd.Series
    trade_info = {}     # (asset, strat_idx) -> (n_trades, win_rate)

    for asset in assets:
        df = data_dict[asset]
        if df.empty or len(df) < 60:
            continue

        close = df['close'].values
        bh_daily = np.diff(close) / close[:-1]
        bh_returns[asset] = pd.Series(bh_daily, index=df.index[1:])

        for strat_idx in range(N_STRATEGIES):
            clear_signal_cache()
            try:
                signals = run_strategy(strat_idx, df)
                strat_ret = compute_daily_strategy_returns(df, signals, asset=asset)
                strat_returns[(asset, strat_idx)] = strat_ret

                result = backtest(df, signals)
                trade_info[(asset, strat_idx)] = (
                    result.get('n_trades', 0),
                    result.get('win_rate', 0.0),
                )
            except Exception as e:
                strat_returns[(asset, strat_idx)] = pd.Series(
                    np.zeros(len(df) - 1), index=df.index[1:]
                )
                trade_info[(asset, strat_idx)] = (0, 0.0)

    return strat_returns, bh_returns, trade_info


def evaluate_oos(chromo: Chromosome, oos_strat_returns, oos_bh_returns,
                 oos_trade_info, assets):
    """Evaluate the best chromosome on OOS data with extended financial metrics."""
    # Build portfolio returns
    strat_cols = {}
    bh_cols = {}
    for i, asset in enumerate(assets):
        key = (asset, int(chromo.strategy_genes[i]))
        strat_cols[asset] = oos_strat_returns.get(key)
        bh_cols[asset] = oos_bh_returns.get(asset)

    # Align on common index
    all_series = list(strat_cols.values()) + list(bh_cols.values())
    valid_series = [s for s in all_series if s is not None and len(s) > 0]
    if not valid_series:
        return None

    common_idx = valid_series[0].index
    for s in valid_series[1:]:
        common_idx = common_idx.union(s.index)
    common_idx = common_idx.sort_values()

    strat_df = pd.DataFrame(strat_cols).reindex(common_idx).fillna(0.0)
    bh_df = pd.DataFrame(bh_cols).reindex(common_idx).fillna(0.0)

    weights = chromo.weight_genes

    portfolio_ret = (strat_df.values * weights).sum(axis=1)
    bh_portfolio_ret = (bh_df.values * weights).sum(axis=1)

    # Portfolio metrics
    port_sharpe = ann_sharpe(portfolio_ret)
    bh_sharpe = ann_sharpe(bh_portfolio_ret)

    # Total return
    port_total = float(np.prod(1 + portfolio_ret) - 1)
    bh_total = float(np.prod(1 + bh_portfolio_ret) - 1)

    # Annualised return & volatility
    mu_daily = np.mean(portfolio_ret)
    sigma_daily = np.std(portfolio_ret)
    ann_return = mu_daily * TRADING_DAYS_PER_YEAR
    ann_volatility = sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Max drawdown
    cum = np.cumprod(1 + portfolio_ret)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    # Sortino ratio (downside deviation)
    downside = portfolio_ret[portfolio_ret < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 1e-9
    sortino = (mu_daily / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR) if downside_std > 0 else 0.0

    # Calmar ratio (ann return / abs(max drawdown))
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0

    # Win rate (fraction of positive days)
    win_rate = float(np.mean(portfolio_ret > 0)) if len(portfolio_ret) > 0 else 0.0

    # Per-asset details
    asset_details = []
    for i, asset in enumerate(assets):
        strat_idx = int(chromo.strategy_genes[i])
        w = chromo.weight_genes[i]

        asset_strat = strat_df[asset].values if asset in strat_df.columns else np.zeros(len(common_idx))
        asset_bh = bh_df[asset].values if asset in bh_df.columns else np.zeros(len(common_idx))

        a_sharpe = ann_sharpe(asset_strat)
        a_bh_sharpe = ann_sharpe(asset_bh)
        a_total = float(np.prod(1 + asset_strat) - 1)
        a_bh_total = float(np.prod(1 + asset_bh) - 1)

        n_trades, a_win_rate = oos_trade_info.get((asset, strat_idx), (0, 0.0))

        asset_details.append({
            'asset': asset,
            'asset_class': ASSET_CLASSES.get(asset, '?'),
            'strategy_idx': strat_idx,
            'strategy_name': get_strategy_name(strat_idx),
            'weight': round(w, 4),
            'sharpe': round(a_sharpe, 4),
            'bh_sharpe': round(a_bh_sharpe, 4),
            'excess_sharpe': round(a_sharpe - a_bh_sharpe, 4),
            'total_return': round(a_total, 4),
            'bh_total_return': round(a_bh_total, 4),
            'n_trades': n_trades,
            'win_rate': round(a_win_rate, 2),
            'beats_bh': a_sharpe > a_bh_sharpe,
        })

    # Equal-weight B&H baseline
    eq_weights = np.ones(len(assets)) / len(assets)
    eq_bh_ret = (bh_df.values * eq_weights).sum(axis=1)
    eq_bh_sharpe = ann_sharpe(eq_bh_ret)

    return {
        'portfolio_sharpe': round(port_sharpe, 4),
        'bh_portfolio_sharpe': round(bh_sharpe, 4),
        'excess_sharpe': round(port_sharpe - bh_sharpe, 4),
        'eq_bh_sharpe': round(eq_bh_sharpe, 4),
        'excess_vs_eq_bh': round(port_sharpe - eq_bh_sharpe, 4),
        'portfolio_total_return': round(port_total, 4),
        'bh_total_return': round(bh_total, 4),
        'ann_return': round(ann_return, 4),
        'ann_volatility': round(ann_volatility, 4),
        'max_drawdown': round(max_dd, 4),
        'sortino': round(sortino, 4),
        'calmar': round(calmar, 4),
        'portfolio_win_rate': round(win_rate, 4),
        'daily_returns': portfolio_ret,
        'cumulative_returns': cum,
        'drawdown_series': dd,
        'asset_details': asset_details,
    }


def run_single_period(period_idx, train_start, train_end, test_start, test_end):
    period_num = period_idx + 1
    test_year = test_start[:4]
    log.info(f"{'='*70}")
    log.info(f"Period {period_num}: Train {train_start[:4]}-{train_end[:4]}, Test {test_year}")
    log.info(f"{'='*70}")

    # Load and slice data
    available_assets = []
    train_data = {}
    test_data = {}

    for asset in ASSETS:
        daily = load_asset_daily(asset)
        if daily.empty:
            continue

        train_mask = (daily.index >= train_start) & (daily.index <= train_end)
        test_mask = (daily.index >= test_start) & (daily.index <= test_end)

        train_df = daily[train_mask].copy()
        test_df = daily[test_mask].copy()

        if len(train_df) >= 100 and len(test_df) >= 50:
            available_assets.append(asset)
            train_data[asset] = train_df
            test_data[asset] = test_df

    log.info(f"  Available assets: {len(available_assets)}")

    # Precompute all strategy returns
    log.info(f"  Precomputing {len(available_assets)} x {N_STRATEGIES} = "
             f"{len(available_assets) * N_STRATEGIES} strategy-asset returns (training)...")
    t0 = time.time()
    train_strat, train_bh, train_trade = precompute_returns(available_assets, train_data)
    log.info(f"  Train precompute done in {time.time()-t0:.1f}s")

    log.info(f"  Precomputing OOS returns...")
    t0 = time.time()
    oos_strat, oos_bh, oos_trade = precompute_returns(available_assets, test_data)
    log.info(f"  OOS precompute done in {time.time()-t0:.1f}s")

    # ── Run FULL GA (LLM + benchmark strategies, excluding hand-coded 11-13) ──
    log.info(f"  Running FULL GA ({len(FULL_STRATEGY_INDICES)} strategies, pop={POPULATION_SIZE}, gen={N_GENERATIONS})...")
    t0 = time.time()
    full_ga = GAEngine(available_assets, train_strat, train_bh,
                       allowed_strategies=FULL_STRATEGY_INDICES,
                       seed=42 + period_idx)
    full_best, full_history, _ = full_ga.run()
    log.info(f"  Full GA done in {time.time()-t0:.1f}s. Best Sharpe: {full_best.fitness:+.4f}")

    # ── Run LLM-ONLY GA (novel strategies only) ──
    log.info(f"  Running LLM-ONLY GA ({len(LLM_STRATEGY_INDICES)} strategies, pop={POPULATION_SIZE}, gen={N_GENERATIONS})...")
    t0 = time.time()
    llm_ga = GAEngine(available_assets, train_strat, train_bh,
                      allowed_strategies=LLM_STRATEGY_INDICES,
                      seed=42 + period_idx)
    llm_best, llm_history, _ = llm_ga.run()
    log.info(f"  LLM GA done in {time.time()-t0:.1f}s. Best Sharpe: {llm_best.fitness:+.4f}")

    # ── Run BENCHMARK-ONLY GA (conventional strategies only) ──
    log.info(f"  Running BENCHMARK-ONLY GA ({len(BENCHMARK_STRATEGY_INDICES)} strategies, pop={POPULATION_SIZE}, gen={N_GENERATIONS})...")
    t0 = time.time()
    bench_ga = GAEngine(available_assets, train_strat, train_bh,
                        allowed_strategies=BENCHMARK_STRATEGY_INDICES,
                        seed=42 + period_idx)
    bench_best, bench_history, _ = bench_ga.run()
    log.info(f"  Bench GA done in {time.time()-t0:.1f}s. Best Sharpe: {bench_best.fitness:+.4f}")

    # Log evolved allocations
    log.info(f"  Full GA allocation:")
    for i, asset in enumerate(available_assets):
        strat_idx = int(full_best.strategy_genes[i])
        w = full_best.weight_genes[i]
        log.info(f"    {asset:<10} -> {get_strategy_name(strat_idx):<25} (w={w:.4f})")

    log.info(f"  LLM-Only GA allocation:")
    for i, asset in enumerate(available_assets):
        strat_idx = int(llm_best.strategy_genes[i])
        w = llm_best.weight_genes[i]
        log.info(f"    {asset:<10} -> {get_strategy_name(strat_idx):<25} (w={w:.4f})")

    log.info(f"  Benchmark GA allocation:")
    for i, asset in enumerate(available_assets):
        strat_idx = int(bench_best.strategy_genes[i])
        w = bench_best.weight_genes[i]
        log.info(f"    {asset:<10} -> {get_strategy_name(strat_idx):<25} (w={w:.4f})")

    # Evaluate all three OOS
    log.info(f"  Evaluating OOS (full)...")
    full_oos = evaluate_oos(full_best, oos_strat, oos_bh, oos_trade, available_assets)
    log.info(f"  Evaluating OOS (llm-only)...")
    llm_oos = evaluate_oos(llm_best, oos_strat, oos_bh, oos_trade, available_assets)
    log.info(f"  Evaluating OOS (benchmark)...")
    bench_oos = evaluate_oos(bench_best, oos_strat, oos_bh, oos_trade, available_assets)

    if full_oos is None or bench_oos is None or llm_oos is None:
        log.warning(f"  OOS evaluation failed!")
        return None, None, None, None, None, None

    excess = full_oos['portfolio_sharpe'] - bench_oos['portfolio_sharpe']
    llm_excess = llm_oos['portfolio_sharpe'] - bench_oos['portfolio_sharpe']

    log.info(f"  OOS Full Portfolio Sharpe:     {full_oos['portfolio_sharpe']:+.4f}")
    log.info(f"  OOS LLM-Only Portfolio Sharpe: {llm_oos['portfolio_sharpe']:+.4f}")
    log.info(f"  OOS Bench Portfolio Sharpe:    {bench_oos['portfolio_sharpe']:+.4f}")
    log.info(f"  OOS Excess Full - Bench:       {excess:+.4f}")
    log.info(f"  OOS Excess LLM - Bench:        {llm_excess:+.4f}")

    # Build summary
    full_details = full_oos['asset_details']

    summary = {
        'period': period_num,
        'train_start': train_start,
        'train_end': train_end,
        'test_start': test_start,
        'test_end': test_end,
        'n_assets': len(available_assets),
        'full_train_sharpe': round(full_best.fitness, 4),
        'full_oos_sharpe': full_oos['portfolio_sharpe'],
        'full_oos_return': full_oos['portfolio_total_return'],
        'full_max_dd': full_oos['max_drawdown'],
        'full_sortino': full_oos['sortino'],
        'full_calmar': full_oos['calmar'],
        'full_ann_vol': full_oos['ann_volatility'],
        'full_win_rate': full_oos['portfolio_win_rate'],
        'bench_train_sharpe': round(bench_best.fitness, 4),
        'bench_oos_sharpe': bench_oos['portfolio_sharpe'],
        'bench_oos_return': bench_oos['portfolio_total_return'],
        'bench_max_dd': bench_oos['max_drawdown'],
        'bench_sortino': bench_oos['sortino'],
        'bench_calmar': bench_oos['calmar'],
        'excess_sharpe': round(excess, 4),
        'llm_train_sharpe': round(llm_best.fitness, 4),
        'llm_oos_sharpe': llm_oos['portfolio_sharpe'],
        'llm_oos_return': llm_oos['portfolio_total_return'],
        'llm_max_dd': llm_oos['max_drawdown'],
        'llm_sortino': llm_oos['sortino'],
        'llm_calmar': llm_oos['calmar'],
        'llm_excess_sharpe': round(llm_excess, 4),
    }

    # Per-asset results (full GA)
    for d in full_details:
        d['period'] = period_num
        d['test_start'] = test_start
        d['test_end'] = test_end

    # Best chromosome details (full GA)
    full_chromo_details = []
    for i, asset in enumerate(available_assets):
        full_chromo_details.append({
            'period': period_num,
            'asset': asset,
            'strategy_idx': int(full_best.strategy_genes[i]),
            'strategy_name': get_strategy_name(int(full_best.strategy_genes[i])),
            'weight': round(full_best.weight_genes[i], 6),
        })

    # Best chromosome details (LLM-only GA)
    llm_chromo_details = []
    for i, asset in enumerate(available_assets):
        llm_chromo_details.append({
            'period': period_num,
            'asset': asset,
            'strategy_idx': int(llm_best.strategy_genes[i]),
            'strategy_name': get_strategy_name(int(llm_best.strategy_genes[i])),
            'weight': round(llm_best.weight_genes[i], 6),
        })

    # Best chromosome details (benchmark GA)
    bench_chromo_details = []
    for i, asset in enumerate(available_assets):
        bench_chromo_details.append({
            'period': period_num,
            'asset': asset,
            'strategy_idx': int(bench_best.strategy_genes[i]),
            'strategy_name': get_strategy_name(int(bench_best.strategy_genes[i])),
            'weight': round(bench_best.weight_genes[i], 6),
        })

    return (summary, full_details, full_chromo_details,
            llm_chromo_details, bench_oos, bench_chromo_details)


def run_legacy_single_run():
    """Legacy single-run mode (kept for backward compatibility)."""
    t0 = time.time()
    log.info("Portfolio-Level Genetic Algorithm (legacy single-run mode)")
    log.info("Fitness = absolute portfolio Sharpe")
    log.info("Comparison: Full GA (all strategies) vs Benchmark-only GA")
    log.info(f"Benchmark strategies: {BENCHMARK_STRATEGY_INDICES}")
    log.info("")

    preload_all()

    all_summaries = []
    all_asset_results = []
    all_full_chromos = []
    all_llm_chromos = []
    all_bench_chromos = []

    for i, (train_start, train_end, test_start, test_end) in enumerate(WALK_FORWARD_PERIODS):
        period_t0 = time.time()

        result = run_single_period(i, train_start, train_end, test_start, test_end)
        summary, asset_details, full_chromo, llm_chromo, bench_oos, bench_chromo = result

        if summary is None:
            continue

        elapsed = time.time() - period_t0
        summary['elapsed_seconds'] = round(elapsed, 1)

        all_summaries.append(summary)
        all_asset_results.extend(asset_details)
        all_full_chromos.extend(full_chromo)
        all_llm_chromos.extend(llm_chromo)
        all_bench_chromos.extend(bench_chromo)

        log.info(f"  Period {i+1} completed in {elapsed:.1f}s\n")

    # Write CSVs
    if all_summaries:
        path = os.path.join(OUTPUT_DIR, 'walk_forward_summary.csv')
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
            w.writeheader()
            w.writerows(all_summaries)
        log.info(f"Saved: {path}")

    if all_asset_results:
        path = os.path.join(OUTPUT_DIR, 'oos_asset_results.csv')
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_asset_results[0].keys())
            w.writeheader()
            w.writerows(all_asset_results)
        log.info(f"Saved: {path}")

    if all_full_chromos:
        path = os.path.join(OUTPUT_DIR, 'best_chromosomes.csv')
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_full_chromos[0].keys())
            w.writeheader()
            w.writerows(all_full_chromos)
        log.info(f"Saved: {path}")

    if all_llm_chromos:
        path = os.path.join(OUTPUT_DIR, 'llm_chromosomes.csv')
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_llm_chromos[0].keys())
            w.writeheader()
            w.writerows(all_llm_chromos)
        log.info(f"Saved: {path}")

    if all_bench_chromos:
        path = os.path.join(OUTPUT_DIR, 'bench_chromosomes.csv')
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_bench_chromos[0].keys())
            w.writeheader()
            w.writerows(all_bench_chromos)
        log.info(f"Saved: {path}")

    total_time = time.time() - t0
    log.info(f"\nAll periods complete. Total time: {total_time:.1f}s")

    # Summary table
    log.info("\n" + "="*110)
    log.info("WALK-FORWARD SUMMARY: Full GA vs LLM-Only GA vs Benchmark-Only GA")
    log.info("="*110)
    log.info(f"{'Period':<8} {'Test':<6} {'N':>3} {'Full OOS':>9} {'LLM OOS':>9} "
             f"{'Bench GA':>9} {'Exc Full':>9} {'Exc LLM':>9}")
    log.info("-"*110)
    for s in all_summaries:
        log.info(
            f"{s['period']:<8} {s['test_start'][:4]:<6} "
            f"{s['n_assets']:>3} {s['full_oos_sharpe']:>+9.4f} "
            f"{s['llm_oos_sharpe']:>+9.4f} "
            f"{s['bench_oos_sharpe']:>+9.4f} "
            f"{s['excess_sharpe']:>+9.4f} "
            f"{s['llm_excess_sharpe']:>+9.4f}"
        )
    log.info("="*110)

    n_periods = len(WALK_FORWARD_PERIODS)
    if all_summaries:
        avg_excess = np.mean([s['excess_sharpe'] for s in all_summaries])
        avg_excess_llm = np.mean([s['llm_excess_sharpe'] for s in all_summaries])
        n_positive = sum(1 for s in all_summaries if s['excess_sharpe'] > 0)
        n_positive_llm = sum(1 for s in all_summaries if s['llm_excess_sharpe'] > 0)
        log.info(f"Mean excess Full vs Bench GA: {avg_excess:+.4f} (positive {n_positive}/{n_periods})")
        log.info(f"Mean excess LLM vs Bench GA:  {avg_excess_llm:+.4f} (positive {n_positive_llm}/{n_periods})")

    # Weight distribution summary
    if all_full_chromos:
        log.info("\nFULL GA WEIGHT DISTRIBUTIONS:")
        for period_num in range(1, n_periods + 1):
            period_chromos = [c for c in all_full_chromos if c['period'] == period_num]
            if not period_chromos:
                continue
            sorted_by_wt = sorted(period_chromos, key=lambda x: -x['weight'])
            top3 = sorted_by_wt[:3]
            log.info(f"  Period {period_num}: "
                     + "  |  ".join(f"{c['asset']} {c['weight']:.3f} ({c['strategy_name']})"
                                    for c in top3))


def main():
    parser = argparse.ArgumentParser(description='Portfolio-Level GA Runner')
    parser.add_argument('--mode', choices=['tune', 'final', 'full', 'legacy'],
                        default='full',
                        help='tune: hyperparameter tuning only; '
                             'final: multi-run evaluation + plotting (requires tuning_results.json); '
                             'full: tune then evaluate then plot; '
                             'legacy: single-run mode (backward compatible)')
    parser.add_argument('--n-runs', type=int, default=FINAL_N_RUNS,
                        help=f'Number of GA runs per variant per period (default: {FINAL_N_RUNS})')
    args = parser.parse_args()

    if args.mode == 'legacy':
        run_legacy_single_run()
        return

    if args.mode in ('tune', 'full'):
        from hyperparameter_tuning import run_tuning
        tuned_params = run_tuning()
    else:
        tuned_params = None

    if args.mode in ('final', 'full'):
        from multi_run_evaluator import run_multi_evaluation
        from plotting import plot_all

        raw_rows, stat_rows, run_data = run_multi_evaluation(
            n_runs=args.n_runs, ga_params=tuned_params)

        log.info("\nGenerating plots...")
        plot_all(raw_rows, stat_rows, run_data)

        log.info("\nDone. Check results/ for CSVs and PNGs.")


if __name__ == '__main__':
    main()
