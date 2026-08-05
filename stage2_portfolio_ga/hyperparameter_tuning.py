"""
Hyperparameter Tuning via Optuna (TPE + MedianPruner) — Per-Period
==================================================================

Tunes GA hyperparameters independently for each walk-forward period using
Bayesian optimisation. Each period gets its own Optuna study, using ONLY
the corresponding tuning validation period to avoid data leakage.

Runs on the Full GA variant only; the same tuned params (per-period) are
used for all variants (Full / LLM / Benchmark) to ensure fair comparison.

Progress is saved incrementally:
  - Optuna study persists to SQLite (results/tuning_study_pN.db)
  - Best params JSON updated after every completed trial
  - Safe to interrupt and resume
"""
import json
import os
import time
import logging
import numpy as np
import pandas as pd
import optuna

from config import (
    ASSETS, N_STRATEGIES, FULL_STRATEGY_INDICES,
    TUNING_RANGES, TUNING_WALK_FORWARD_PERIODS,
    TUNING_N_TRIALS, TUNING_RUNS_PER_TRIAL, TUNING_TIMEOUT_SECONDS,
    OUTPUT_DIR,
)
from data_loader import load_asset_daily, preload_all
from strategies import run_strategy, clear_signal_cache
from run_portfolio_ga import compute_daily_strategy_returns, precompute_returns
from ga_engine import GAEngine, ann_sharpe

log = logging.getLogger(__name__)


def _precompute_single_tuning_period(period_idx):
    """Precompute strategy returns for a single tuning period."""
    train_start, train_end, val_start, val_end = TUNING_WALK_FORWARD_PERIODS[period_idx]

    train_data = {}
    val_data = {}
    available_assets = []

    for asset in ASSETS:
        daily = load_asset_daily(asset)
        if daily.empty:
            continue

        train_df = daily[(daily.index >= train_start) & (daily.index <= train_end)].copy()
        val_df = daily[(daily.index >= val_start) & (daily.index <= val_end)].copy()

        if len(train_df) >= 100 and len(val_df) >= 50:
            available_assets.append(asset)
            train_data[asset] = train_df
            val_data[asset] = val_df

    log.info(f"  Tuning period {train_start[:4]}-{train_end[:4]} / val {val_start[:4]}: "
             f"{len(available_assets)} assets")

    train_strat, train_bh, _ = precompute_returns(available_assets, train_data)
    val_strat, val_bh, _ = precompute_returns(available_assets, val_data)

    return {
        'assets': available_assets,
        'train_strat': train_strat,
        'train_bh': train_bh,
        'val_strat': val_strat,
        'val_bh': val_bh,
    }


def _evaluate_trial_on_period(ga_params, period, seed):
    """Run one GA with given params on one period, return validation Sharpe."""
    ga = GAEngine(
        period['assets'],
        period['train_strat'],
        period['train_bh'],
        allowed_strategies=FULL_STRATEGY_INDICES,
        seed=seed,
        ga_params=ga_params,
    )
    best, _, _ = ga.run()

    # Evaluate on validation set
    strat_cols = {}
    for i, asset in enumerate(period['assets']):
        key = (asset, int(best.strategy_genes[i]))
        strat_cols[asset] = period['val_strat'].get(key)

    bh_df = {a: period['val_bh'].get(a) for a in period['assets']}

    valid_series = [s for s in list(strat_cols.values()) + list(bh_df.values())
                    if s is not None and len(s) > 0]
    if not valid_series:
        return -10.0

    common_idx = valid_series[0].index
    for s in valid_series[1:]:
        common_idx = common_idx.union(s.index)
    common_idx = common_idx.sort_values()

    strat_df = pd.DataFrame(strat_cols).reindex(common_idx).fillna(0.0)
    weights = best.weight_genes
    portfolio_ret = (strat_df.values * weights).sum(axis=1)

    return ann_sharpe(portfolio_ret)


def _save_per_period_json(all_period_results, elapsed):
    """Save per-period tuning results to JSON."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, 'tuning_results.json')
    output = {
        'per_period': all_period_results,
        'elapsed_seconds': round(elapsed, 1),
    }
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)


def run_tuning():
    """Run per-period Optuna hyperparameter tuning. Returns dict of {period: params}."""
    log.info("=" * 70)
    log.info("PER-PERIOD HYPERPARAMETER TUNING (Optuna TPE + MedianPruner)")
    log.info("=" * 70)

    t0 = time.time()
    preload_all()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_period_results = {}

    for period_idx in range(len(TUNING_WALK_FORWARD_PERIODS)):
        period_num = period_idx + 1
        twf = TUNING_WALK_FORWARD_PERIODS[period_idx]
        log.info(f"\n{'='*60}")
        log.info(f"TUNING PERIOD {period_num}: train {twf[0][:4]}-{twf[1][:4]}, "
                 f"validate {twf[2][:4]}")
        log.info(f"{'='*60}")

        # Precompute returns for this period only
        period_data = _precompute_single_tuning_period(period_idx)

        # Check for existing study
        db_path = os.path.join(OUTPUT_DIR, f'tuning_study_p{period_num}.db')
        storage = f"sqlite:///{db_path}"

        study = optuna.create_study(
            study_name=f'portfolio_ga_tuning_p{period_num}',
            storage=storage,
            load_if_exists=True,
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=0,  # only 1 period, no intermediate steps to prune on
            ),
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        existing = len(study.trials)
        remaining = max(0, TUNING_N_TRIALS - existing)
        if existing > 0:
            log.info(f"  Resuming from {existing} existing trials "
                     f"(best so far: {study.best_value:+.4f})")

        if remaining > 0:
            def make_objective(pd):
                """Create objective closure for this period's data."""
                def objective(trial):
                    pop_size = trial.suggest_int(
                        'population_size',
                        TUNING_RANGES['population_size'][0],
                        TUNING_RANGES['population_size'][1],
                        step=50,
                    )
                    n_gen = trial.suggest_int(
                        'n_generations',
                        TUNING_RANGES['n_generations'][0],
                        TUNING_RANGES['n_generations'][1],
                        step=50,
                    )
                    cx_rate = trial.suggest_float(
                        'crossover_rate',
                        TUNING_RANGES['crossover_rate'][0],
                        TUNING_RANGES['crossover_rate'][1],
                    )
                    strat_mut = trial.suggest_float(
                        'strategy_mutation_rate',
                        TUNING_RANGES['strategy_mutation_rate'][0],
                        TUNING_RANGES['strategy_mutation_rate'][1],
                    )
                    wt_mut = trial.suggest_float(
                        'weight_mutation_rate',
                        TUNING_RANGES['weight_mutation_rate'][0],
                        TUNING_RANGES['weight_mutation_rate'][1],
                    )
                    wt_sigma = trial.suggest_float(
                        'weight_mutation_sigma',
                        TUNING_RANGES['weight_mutation_sigma'][0],
                        TUNING_RANGES['weight_mutation_sigma'][1],
                    )
                    elite = trial.suggest_int(
                        'elite_count',
                        TUNING_RANGES['elite_count'][0],
                        TUNING_RANGES['elite_count'][1],
                    )
                    cx_type = trial.suggest_categorical(
                        'crossover_type',
                        ['one_point', 'two_point', 'uniform'],
                    )

                    t_size = max(2, round(pop_size * 0.01))

                    ga_params = {
                        'population_size': pop_size,
                        'n_generations': n_gen,
                        'tournament_size': t_size,
                        'crossover_rate': cx_rate,
                        'crossover_type': cx_type,
                        'strategy_mutation_rate': strat_mut,
                        'weight_mutation_rate': wt_mut,
                        'weight_mutation_sigma': wt_sigma,
                        'elite_count': elite,
                    }

                    # Run TUNING_RUNS_PER_TRIAL GA runs, average validation Sharpe
                    run_sharpes = []
                    for run_i in range(TUNING_RUNS_PER_TRIAL):
                        seed = trial.number * 1000 + run_i
                        sharpe = _evaluate_trial_on_period(ga_params, pd, seed)
                        run_sharpes.append(sharpe)

                    return np.mean(run_sharpes)

                return objective

            p_t0 = time.time()

            def _log_callback(study, trial):
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    log.info(f"  Trial {trial.number}: val_sharpe={trial.value:+.4f}  "
                             f"best_so_far={study.best_value:+.4f}")
                elif trial.state == optuna.trial.TrialState.PRUNED:
                    log.info(f"  Trial {trial.number}: PRUNED")

            study.optimize(
                make_objective(period_data),
                n_trials=remaining,
                timeout=TUNING_TIMEOUT_SECONDS,
                callbacks=[_log_callback],
            )

            p_elapsed = time.time() - p_t0
            log.info(f"  Period {period_num} tuning done in {p_elapsed:.1f}s")

        # Extract best params for this period
        best_params = dict(study.best_params)
        best_params['tournament_size'] = max(2, round(best_params['population_size'] * 0.01))

        all_period_results[str(period_num)] = {
            'best_params': best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'n_pruned': len([t for t in study.trials
                             if t.state == optuna.trial.TrialState.PRUNED]),
        }

        log.info(f"  Best validation Sharpe: {study.best_value:+.4f}")
        log.info(f"  Best params: {best_params}")

        # Save after each period completes
        _save_per_period_json(all_period_results, time.time() - t0)

    elapsed = time.time() - t0
    log.info(f"\nAll tuning complete in {elapsed:.1f}s")

    # Final save
    _save_per_period_json(all_period_results, elapsed)
    log.info(f"Saved: {os.path.join(OUTPUT_DIR, 'tuning_results.json')}")

    return {int(k): v['best_params'] for k, v in all_period_results.items()}


def load_tuned_params():
    """Load per-period tuned params from JSON.

    Returns dict: {period_num: params_dict} or None if not found.
    """
    path = os.path.join(OUTPUT_DIR, 'tuning_results.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)

    # Support both old (single-set) and new (per-period) format
    if 'per_period' in data:
        return {int(k): v['best_params'] for k, v in data['per_period'].items()}
    elif 'best_params' in data:
        # Old format: return same params for all periods
        n_periods = len(TUNING_WALK_FORWARD_PERIODS)
        return {i: data['best_params'] for i in range(1, n_periods + 1)}
    return None


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        handlers=[logging.StreamHandler()],
    )
    run_tuning()
