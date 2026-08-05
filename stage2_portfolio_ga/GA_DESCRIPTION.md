# Portfolio-Level Genetic Algorithm — Full Description

## What This System Does

This is a **portfolio-level genetic algorithm** that simultaneously optimises which trading strategy to run on each of 19 assets AND how much capital to allocate to each. The key insight is that **genes interact**: changing one asset's strategy affects the optimal choice for all others through covariance and diversification effects. This makes the problem intractable for exhaustive search (19^19 possible strategy combinations) and necessitates a GA.

The research question: **Do LLM-generated novel trading strategies add value beyond textbook benchmarks when combined in an optimised portfolio?**

---

## Chromosome Structure

Each individual (chromosome) represents one complete portfolio configuration:

```
Chromosome = (strategy_genes[19], weight_genes[19])

strategy_genes: [3, 0, 18, 14, 7, ...]   ← which strategy to run on each asset (integer 0-18)
weight_genes:   [0.08, 0.03, 0.12, ...]   ← portfolio weight per asset (floats summing to 1.0)
```

- **19 assets**: 3 crypto, 5 forex, 8 equity indices, 3 commodities
- **19 strategies**: 11 LLM-generated novel strategies (0-10), 3 hand-coded (11-13, excluded from GA), 5 textbook benchmarks (14-18)
- Strategy and weight are paired per asset — crossover swaps both together to preserve learned pairings

---

## Strategy Registry

### LLM-Generated Novel Strategies (0-10)
| Idx | Name | Description |
|-----|------|-------------|
| 0 | Spectral Entropy | Frequency-domain disorder measure |
| 1 | Rough Path Signatures | Path signature features for regime detection |
| 2 | Information Geometry | Fisher information metric on return distributions |
| 3 | Wasserstein Distance | Optimal transport distance between return windows |
| 4 | Rate Distortion | Information-theoretic complexity measure |
| 5 | Fisher Information | Fisher information for regime identification |
| 6 | Hurst Regime | Hurst exponent for trend/mean-reversion detection |
| 7 | Transfer Entropy | Causal information flow between price series |
| 8 | Approx Entropy | Approximate entropy for predictability measurement |
| 9 | Permutation Entropy | Ordinal pattern entropy |
| 10 | Volatility Breakout | Daily volatility breakout signals |

### Hand-Coded Non-Textbook (11-13) — Excluded from GA
| Idx | Name |
|-----|------|
| 11 | Mean Reversion Bands |
| 12 | RSI Divergence |
| 13 | Regime HMM Proxy |

### Textbook Benchmarks (14-18)
| Idx | Name |
|-----|------|
| 14 | MA Crossover |
| 15 | Time-Series Momentum |
| 16 | Bollinger Breakout |
| 17 | Donchian Breakout |
| 18 | Buy & Hold |

### Strategy Subsets Used by the Three GA Variants
- **Full GA**: indices [0-10] + [14-18] = 16 strategies (LLM + benchmarks)
- **LLM-Only GA**: indices [0-10] = 11 strategies
- **Benchmark-Only GA**: indices [14-18] = 5 strategies

---

## Assets (19 Total)

| Class | Assets |
|-------|--------|
| Crypto (3) | BTCUSD, ETHUSD, BCHUSD |
| Forex (5) | EURUSD, GBPUSD, USDJPY, USDCHF, NZDUSD |
| Equity (8) | DE40, F40, STOXX50, UK100, US30, US500, USTEC, JP225 |
| Commodity (3) | XAUUSD, XAGUSD, XBRUSD |

---

## Fitness Function

**Fitness = Annualised Sharpe ratio of the combined weighted portfolio.**

```
portfolio_daily_return[t] = Σ (weight[i] × strategy_return[i][t])   for all 19 assets

fitness = (mean(portfolio_daily_return) / std(portfolio_daily_return)) × √252
```

This is NOT the average of per-asset Sharpes. It's the Sharpe of the actual blended portfolio, which means diversification benefits are captured. Two negatively correlated assets can produce a positive-Sharpe portfolio even if each is mediocre alone.

---

## GA Operators

### Initialisation
- Strategy genes: uniform random from allowed strategy set
- Weight genes: Dirichlet distribution (ensures sum to 1, avoids extreme allocations)

### Selection: Tournament
- Pick `tournament_size` random individuals, select the fittest
- Tournament size derived as `max(2, round(population_size × 0.01))` during tuning
- Default tournament size: 4 (before tuning)

### Crossover: Uniform at Asset Level
- For each of the 19 asset positions, swap with probability `crossover_rate`
- **Both strategy AND weight are swapped together** (preserves the learned pairing)
- Weights are renormalised after crossover to maintain sum-to-1 constraint

```
Parent A: asset 3 → (Wasserstein, 0.12)     Parent B: asset 3 → (MA Crossover, 0.05)
                         ↕ swap (if random < crossover_rate)
Child A:  asset 3 → (MA Crossover, 0.05)    Child B:  asset 3 → (Wasserstein, 0.12)
```

### Mutation: Two Types
1. **Strategy mutation**: Each asset's strategy gene independently reset to a random allowed strategy with probability `strategy_mutation_rate`
2. **Weight mutation**: Each asset's weight perturbed by Gaussian noise N(0, `weight_mutation_sigma`) with probability `weight_mutation_rate`, then clamped to minimum 1e-6 and renormalised

### Elitism
- Top `elite_count` individuals copied unchanged into next generation (preserves best solutions)

---

## Hyperparameters

### Defaults (Before Tuning)
| Parameter | Value |
|-----------|-------|
| Population size | 300 |
| Generations | 300 |
| Tournament size | 4 |
| Crossover rate | 0.70 |
| Strategy mutation rate | 0.15 |
| Weight mutation rate | 0.20 |
| Weight mutation sigma | 0.05 |
| Elite count | 3 |

### Tuning (Optuna TPE + MedianPruner)
Hyperparameters are tuned via Bayesian optimisation (Tree-structured Parzen Estimator) using Optuna.

**Tuning search ranges:**
| Parameter | Range | Step |
|-----------|-------|------|
| Population size | 100–300 | 50 |
| Generations | 100–300 | 50 |
| Crossover rate | 0.4–0.9 | continuous |
| Strategy mutation rate | 0.05–0.30 | continuous |
| Weight mutation rate | 0.05–0.40 | continuous |
| Weight mutation sigma | 0.01–0.15 | continuous |
| Elite count | 1–10 | 1 |

**Tuning procedure:**
- 6 tuning walk-forward periods (2yr train / 1yr validation, carved from the main 3yr train windows)
- Each trial: run GA with candidate params, 3 runs per period, average validation Sharpe
- MedianPruner: 10 startup trials, prunes unpromising configs after 2 periods evaluated
- 50 trials total, 4-hour timeout
- Tuned on **Full GA only** (same params used for all 3 variants for fair comparison)
- Tournament size derived from population size (not tuned separately)
- Progress saved to SQLite — safe to interrupt and resume

**Tuning output:** `results/tuning_results.json`

---

## Walk-Forward Evaluation Design

### 6 Rolling Walk-Forward Periods
| Period | Training | Out-of-Sample Test |
|--------|----------|--------------------|
| 1 | 2015-01-01 → 2017-12-31 | 2018 |
| 2 | 2017-01-01 → 2019-12-31 | 2020 |
| 3 | 2018-01-01 → 2020-12-31 | 2021 |
| 4 | 2019-01-01 → 2021-12-31 | 2022 |
| 5 | 2020-01-01 → 2022-12-31 | 2023 |
| 6 | 2021-01-01 → 2023-12-31 | 2024 |

- 3-year training window, 1-year out-of-sample test
- Rolling (overlapping) windows
- GA is trained on training data only; all OOS metrics are truly out-of-sample

### Tuning Walk-Forward Periods (Separate)
| Period | Training | Validation |
|--------|----------|------------|
| 1 | 2015-01-01 → 2016-12-31 | 2017 |
| 2 | 2017-01-01 → 2018-12-31 | 2019 |
| 3 | 2018-01-01 → 2019-12-31 | 2020 |
| 4 | 2019-01-01 → 2020-12-31 | 2021 |
| 5 | 2020-01-01 → 2021-12-31 | 2022 |
| 6 | 2021-01-01 → 2022-12-31 | 2023 |

These are carved from the first 2 years of each 3-year training window, with the 3rd year used as validation for hyperparameter selection.

---

## Multi-Run Statistical Evaluation (50 Runs)

For statistical robustness, each (period, variant) combination is run **50 times** with different random seeds. This produces distributions rather than single-point estimates.

**Per run, we record:**
- Training Sharpe (in-sample fitness)
- OOS Sharpe, OOS total return, OOS max drawdown
- Sortino ratio, Calmar ratio, annualised volatility, annualised return
- Portfolio-level win rate (fraction of positive days)
- Full fitness history (best + population average per generation)

**Aggregate statistics per (period, variant):**
- Mean, std, min, max for each metric

**3 variants compared:**
1. **Full GA**: LLM-generated + benchmark strategies (16 total)
2. **LLM-Only GA**: LLM-generated strategies only (11 total)
3. **Benchmark-Only GA**: Textbook benchmark strategies only (5 total)

All three use identical tuned hyperparameters for fair comparison.

---

## Transaction Costs

Half-spread (95th percentile from tick data) applied on both entry and exit:

| Asset | Cost | Asset | Cost |
|-------|------|-------|------|
| EURUSD | 0.00001 | BTCUSD | 0.00011 |
| GBPUSD | 0.00001 | ETHUSD | 0.00070 |
| USDJPY | 0.00001 | BCHUSD | 0.00498 |
| USDCHF | 0.00001 | XAUUSD | 0.00010 |
| NZDUSD | 0.00002 | XAGUSD | 0.00020 |
| US500 | 0.00003 | XBRUSD | 0.00015 |
| USTEC | 0.00004 | DE40 | 0.00005 |
| US30 | 0.00005 | UK100 | 0.00005 |
| JP225 | 0.00010 | F40 | 0.00010 |
| STOXX50 | 0.00008 | | |

---

## Performance Optimisation

### Precomputed Returns Cache
All 19×19 = 361 strategy-asset daily return series are computed once per period before the GA runs. This means each GA fitness evaluation is a pure array operation — no backtesting during evolution.

### Vectorised Fitness Evaluation
Strategy returns are stored in a 3D numpy array `returns_matrix[strategy_idx, asset_idx, day]`. Fitness evaluation is:
```python
asset_returns = returns_matrix[strategy_genes, range(n_assets), :]  # fancy indexing
portfolio_ret = weights @ asset_returns                              # dot product
sharpe = (mean(portfolio_ret) / std(portfolio_ret)) * sqrt(252)
```

This makes each fitness evaluation ~500x faster than the naive DataFrame-based approach.

### Incremental Saving
- Tuning: SQLite-backed Optuna study + JSON saved after each trial
- Multi-run: CSV appended after each (period, variant) block
- Both can be interrupted and resumed without losing progress

---

## Extended Financial Metrics

All metrics computed at portfolio level on OOS data:

| Metric | Formula |
|--------|---------|
| Sharpe Ratio | (μ / σ) × √252 |
| Annualised Return | μ × 252 |
| Annualised Volatility | σ × √252 |
| Sortino Ratio | (μ / σ_downside) × √252 |
| Calmar Ratio | Annualised Return / |Max Drawdown| |
| Max Drawdown | max peak-to-trough decline |
| Total Return | ∏(1 + r_t) − 1 |
| Win Rate | fraction of days with positive return |

---

## Visualisations (plotting.py)

| Plot | Description |
|------|-------------|
| Learning curves | Best + avg fitness over generations (per period, Full GA, with ±1 std band across 50 runs) |
| Box plots | OOS Sharpe distribution across 50 runs, per period and variant |
| Equity curves | Cumulative return over OOS period (median run per variant) |
| Drawdown curves | Drawdown over time (median run per variant) |
| Return distributions | Histogram of daily returns per variant |
| Metrics bar charts | Sharpe, Sortino, Calmar, volatility, max drawdown, return — grouped by period and variant |

All saved as PNGs to `results/`.

---

## File Structure

| File | Purpose |
|------|---------|
| `config.py` | All configuration: paths, assets, strategies, GA defaults, tuning ranges, walk-forward periods, transaction costs |
| `ga_engine.py` | GA engine: Chromosome class, population init, tournament selection, uniform crossover, mutation, vectorised fitness evaluation |
| `run_portfolio_ga.py` | Main entry point with argparse: `--mode tune\|final\|full\|legacy`, `--n-runs N` |
| `hyperparameter_tuning.py` | Optuna TPE tuning with SQLite persistence and incremental JSON saving |
| `multi_run_evaluator.py` | 50-run statistical evaluation with incremental CSV saving and resume support |
| `plotting.py` | All visualisations |
| `data_loader.py` | Loads hourly data from dev+test sets, resamples to daily OHLCV, caches |
| `strategies.py` | Unified interface to run any strategy from the registry, with signal caching |
| `backtester.py` | Converts entry/exit signals to performance metrics (P&L, Sharpe, drawdown, trades) |

---

## How to Run

```bash
# Full pipeline: tune hyperparameters, then 50-run evaluation, then generate all plots
python run_portfolio_ga.py --mode full

# Tuning only (saves results/tuning_results.json)
python run_portfolio_ga.py --mode tune

# 50-run evaluation + plots only (requires tuning_results.json)
python run_portfolio_ga.py --mode final

# Quick smoke test (3 runs instead of 50)
python run_portfolio_ga.py --mode final --n-runs 3

# Legacy single-run mode (backward compatible, no tuning or multi-run)
python run_portfolio_ga.py --mode legacy
```

All modes are safe to interrupt and resume (tuning via SQLite, multi-run via CSV checkpoint).

---

## Output Files

| File | Contents |
|------|----------|
| `results/tuning_results.json` | Best hyperparameters from Optuna tuning |
| `results/tuning_study.db` | SQLite Optuna study (all trial history) |
| `results/multi_run_raw.csv` | Per-run results: period, variant, run number, all metrics |
| `results/multi_run_statistics.csv` | Aggregate stats: mean/std/min/max per (period, variant) |
| `results/learning_curve_period_N.png` | Learning curves (6 plots) |
| `results/multi_run_box_plots.png` | OOS Sharpe box plots |
| `results/equity_curve_period_N.png` | Equity curves (6 plots) |
| `results/drawdown_period_N.png` | Drawdown curves (6 plots) |
| `results/return_dist_period_N.png` | Return distribution histograms (6 plots) |
| `results/metrics_*.png` | Metrics comparison bar charts (6 plots) |

---

## Why a GA is Necessary

The search space is combinatorially explosive:
- 16^19 ≈ 4.9 × 10^22 possible strategy assignments (Full GA)
- Each with continuous weight optimisation (19-dimensional simplex)
- Genes interact: the optimal strategy for BTCUSD depends on what's running on XAUUSD through portfolio covariance

Exhaustive search is impossible. Gradient-based methods don't apply (discrete strategy selection). The GA's population-based search with crossover (recombining good partial solutions) and mutation (exploring nearby configurations) is well-suited to this mixed discrete-continuous optimisation with strong epistasis.
