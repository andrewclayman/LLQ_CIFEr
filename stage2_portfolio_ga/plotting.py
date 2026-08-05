"""
Plotting Module
===============

All visualisations for the multi-run GA evaluation:
  - Learning curves (best + average fitness over generations)
  - Multi-run box plots (OOS Sharpe distribution)
  - Equity curves (cumulative return)
  - Drawdown curves
  - Return distributions (histograms)
  - Metrics comparison bar charts

All plots saved as PNGs to results/.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, WALK_FORWARD_PERIODS

log = logging.getLogger(__name__)

VARIANT_COLOURS = {'Full': '#1f77b4', 'LLM': '#ff7f0e', 'Benchmark': '#2ca02c'}


def plot_all(raw_rows, stat_rows, run_data):
    """Generate all plots from multi-run evaluation results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_learning_curves(run_data)
    plot_multi_run_box_plots(raw_rows)
    plot_equity_curves(run_data)
    plot_drawdown_curves(run_data)
    plot_return_distributions(run_data)
    plot_metrics_comparison(stat_rows)

    log.info("All plots saved to %s", OUTPUT_DIR)


def plot_learning_curves(run_data):
    """Learning curves: 5 individual runs showing best + average fitness.

    One plot per period, Full GA only. Shows single-run trajectories
    rather than averaged curves to illustrate typical GA convergence behaviour.
    """
    N_DISPLAY = 5  # number of individual runs to show

    for p_idx, (ts, te, os_, oe) in enumerate(WALK_FORWARD_PERIODS):
        period_num = p_idx + 1
        key = (period_num, 'Full')
        if key not in run_data or not run_data[key]:
            continue

        runs = run_data[key]
        valid_runs = [r for r in runs if r['best_history'] and r['avg_history']]
        if not valid_runs:
            continue

        # Select up to N_DISPLAY runs evenly spaced through the run list
        n_show = min(N_DISPLAY, len(valid_runs))
        indices = np.linspace(0, len(valid_runs) - 1, n_show, dtype=int)
        selected = [valid_runs[i] for i in indices]

        fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 5), sharey=True)
        if n_show == 1:
            axes = [axes]

        best_colours = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b']
        avg_colours = ['#aec7e8', '#ff9896', '#98df8a', '#c5b0d5', '#c49c94']

        for ax_i, (ax, run_info) in enumerate(zip(axes, selected)):
            best_h = run_info['best_history']
            avg_h = run_info['avg_history']
            gens = np.arange(1, len(best_h) + 1)

            ax.plot(gens, best_h, label='Best', color=best_colours[ax_i % len(best_colours)],
                    linewidth=1.5)
            ax.plot(gens, avg_h, label='Avg', color=avg_colours[ax_i % len(avg_colours)],
                    linewidth=1, linestyle='--', alpha=0.7)

            ax.set_xlabel('Generation')
            if ax_i == 0:
                ax.set_ylabel('Sharpe Ratio')
            ax.set_title(f'Run {indices[ax_i] + 1}', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'Period {period_num}: Learning Curves ({n_show} Individual Runs, Full GA)',
                     fontsize=13)
        fig.tight_layout()

        path = os.path.join(OUTPUT_DIR, f'learning_curve_period_{period_num}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        log.info(f"  Saved: {path}")


def plot_multi_run_box_plots(raw_rows):
    """Box plots of OOS Sharpe distribution across runs, per period and variant."""
    df = pd.DataFrame(raw_rows)
    if df.empty:
        return

    periods = sorted(df['period'].unique())
    variants = ['Full', 'LLM', 'Benchmark']

    # One combined figure with subplots per period
    n_periods = len(periods)
    fig, axes = plt.subplots(1, n_periods, figsize=(4 * n_periods, 6), sharey=True)
    if n_periods == 1:
        axes = [axes]

    for ax, period in zip(axes, periods):
        data_to_plot = []
        colours = []
        labels = []
        for v in variants:
            subset = df[(df['period'] == period) & (df['variant'] == v)]['sharpe'].values
            if len(subset) > 0:
                data_to_plot.append(subset)
                colours.append(VARIANT_COLOURS[v])
                labels.append(v)

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
            for patch, colour in zip(bp['boxes'], colours):
                patch.set_facecolor(colour)
                patch.set_alpha(0.7)

        test_year = WALK_FORWARD_PERIODS[period - 1][2][:4]
        ax.set_title(f'Period {period}\n(Test {test_year})')
        ax.axhline(y=0, color='grey', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

    axes[0].set_ylabel('OOS Sharpe Ratio')
    fig.suptitle('Multi-Run OOS Sharpe Distribution', fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'multi_run_box_plots.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_equity_curves(run_data):
    """Equity curves: median cumulative return per variant per period."""
    for p_idx, (ts, te, os_, oe) in enumerate(WALK_FORWARD_PERIODS):
        period_num = p_idx + 1

        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False

        for variant in ['Full', 'LLM', 'Benchmark']:
            key = (period_num, variant)
            if key not in run_data or not run_data[key]:
                continue

            runs = run_data[key]
            oos_arrays = [r['oos_returns'] for r in runs
                          if r['oos_returns'] is not None and len(r['oos_returns']) > 0]
            if not oos_arrays:
                continue

            # Use median run (by total return)
            total_rets = [np.prod(1 + r) - 1 for r in oos_arrays]
            median_idx = np.argsort(total_rets)[len(total_rets) // 2]
            median_returns = oos_arrays[median_idx]

            cum = np.cumprod(1 + median_returns)
            ax.plot(cum, label=variant, color=VARIANT_COLOURS[variant], linewidth=1.5)
            has_data = True

        if has_data:
            ax.axhline(y=1.0, color='grey', linestyle='--', alpha=0.5)
            ax.set_xlabel('Trading Day')
            ax.set_ylabel('Cumulative Return')
            test_year = os_[:4]
            ax.set_title(f'Period {period_num}: Equity Curves (Test {test_year}, median run)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            path = os.path.join(OUTPUT_DIR, f'equity_curve_period_{period_num}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            log.info(f"  Saved: {path}")

        plt.close(fig)


def plot_drawdown_curves(run_data):
    """Drawdown curves per variant per period (median run)."""
    for p_idx, (ts, te, os_, oe) in enumerate(WALK_FORWARD_PERIODS):
        period_num = p_idx + 1

        fig, ax = plt.subplots(figsize=(10, 5))
        has_data = False

        for variant in ['Full', 'LLM', 'Benchmark']:
            key = (period_num, variant)
            if key not in run_data or not run_data[key]:
                continue

            runs = run_data[key]
            oos_arrays = [r['oos_returns'] for r in runs
                          if r['oos_returns'] is not None and len(r['oos_returns']) > 0]
            if not oos_arrays:
                continue

            total_rets = [np.prod(1 + r) - 1 for r in oos_arrays]
            median_idx = np.argsort(total_rets)[len(total_rets) // 2]
            ret = oos_arrays[median_idx]

            cum = np.cumprod(1 + ret)
            peak = np.maximum.accumulate(cum)
            dd = (cum - peak) / np.where(peak > 0, peak, 1)

            ax.plot(dd * 100, label=variant, color=VARIANT_COLOURS[variant], linewidth=1.5)
            has_data = True

        if has_data:
            ax.set_xlabel('Trading Day')
            ax.set_ylabel('Drawdown (%)')
            test_year = os_[:4]
            ax.set_title(f'Period {period_num}: Drawdown (Test {test_year}, median run)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            path = os.path.join(OUTPUT_DIR, f'drawdown_period_{period_num}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            log.info(f"  Saved: {path}")

        plt.close(fig)


def plot_return_distributions(run_data):
    """Histogram of daily returns for each variant (median run), per period."""
    for p_idx, (ts, te, os_, oe) in enumerate(WALK_FORWARD_PERIODS):
        period_num = p_idx + 1

        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False

        for variant in ['Full', 'LLM', 'Benchmark']:
            key = (period_num, variant)
            if key not in run_data or not run_data[key]:
                continue

            runs = run_data[key]
            oos_arrays = [r['oos_returns'] for r in runs
                          if r['oos_returns'] is not None and len(r['oos_returns']) > 0]
            if not oos_arrays:
                continue

            total_rets = [np.prod(1 + r) - 1 for r in oos_arrays]
            median_idx = np.argsort(total_rets)[len(total_rets) // 2]
            ret = oos_arrays[median_idx]

            ax.hist(ret * 100, bins=50, alpha=0.5, label=variant,
                    color=VARIANT_COLOURS[variant], density=True)
            has_data = True

        if has_data:
            ax.set_xlabel('Daily Return (%)')
            ax.set_ylabel('Density')
            test_year = os_[:4]
            ax.set_title(f'Period {period_num}: Daily Return Distribution (Test {test_year})')
            ax.legend()
            ax.grid(True, alpha=0.3)

            path = os.path.join(OUTPUT_DIR, f'return_dist_period_{period_num}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            log.info(f"  Saved: {path}")

        plt.close(fig)


def plot_metrics_comparison(stat_rows):
    """Bar charts comparing key metrics across variants, per period."""
    if not stat_rows:
        return

    df = pd.DataFrame(stat_rows)
    metrics = [
        ('sharpe_mean', 'OOS Sharpe'),
        ('sortino_mean', 'Sortino'),
        ('calmar_mean', 'Calmar'),
        ('ann_volatility_mean', 'Ann. Volatility'),
        ('max_drawdown_mean', 'Max Drawdown'),
        ('total_return_mean', 'Total Return'),
    ]

    periods = sorted(df['period'].unique())
    variants = ['Full', 'LLM', 'Benchmark']

    for metric_col, metric_label in metrics:
        if metric_col not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(periods))
        width = 0.25

        for i, v in enumerate(variants):
            subset = df[df['variant'] == v].set_index('period')
            vals = [subset.loc[p, metric_col] if p in subset.index else 0 for p in periods]
            ax.bar(x + i * width, vals, width, label=v, color=VARIANT_COLOURS[v], alpha=0.8)

        ax.set_xlabel('Period')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} by Period and Variant')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'P{p}' for p in periods])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        if metric_col in ('max_drawdown_mean',):
            ax.axhline(y=0, color='grey', linestyle='--', alpha=0.5)

        safe_name = metric_col.replace('_mean', '')
        path = os.path.join(OUTPUT_DIR, f'metrics_{safe_name}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        log.info(f"  Saved: {path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
    log.info("plotting.py must be called with run_data from multi_run_evaluator")
