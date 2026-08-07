# Large Language Quant (LLQ)

Prompts and generated strategy code for the paper:

> **Large Language Quant: Cross-Domain Mathematical Trading Strategy Generation
> with an Agentic LLM**
> Andrew Clayman, Michael Kampouridis, Tasos Papastylianou.
> IEEE Symposium on Computational Intelligence for Financial Engineering &
> Economics (CIFEr), 2026.

This repository accompanies the paper and is provided for **reproducibility**.
It contains (i) the exact prompt used to drive the agentic strategy-generation
loop and (ii) the Python signal generators the agent produced, together with the
benchmark strategies and the out-of-sample evaluation harness.

---

## What is here

```
LLQ_CIFEr/
├── prompt/
│   └── llq_prompt.txt          # The structured LLQ generation prompt (Section III-A)
├── strategies/
│   ├── spectral_entropy.py             # (1) Spectral Entropy       — Information Theory
│   ├── rough_path_signatures.py        # (2) Rough Path Signatures  — Stochastic Analysis
│   ├── information_geometry.py         # (3) Information Geometry    — Differential Geometry
│   ├── wasserstein_distance.py         # (4) Wasserstein Distance    — Optimal Transport
│   ├── rate_distortion_complexity.py   # (5) Rate Distortion         — Information Theory
│   ├── fisher_information_regime.py    # (6) Fisher Information       — Information Geometry
│   ├── volatility_phase_transition.py  # (7) Volatility Phase Trans. — Statistical Physics
│   ├── hurst_regime_daily.py           # (8) Hurst Mean Reversion     — Fractal Analysis
│   ├── benchmarks.py                   # B3 MA Crossover, B4 Bollinger Breakout, B5 Donchian Breakout
│   ├── mean_reversion_bands_daily.py   # B1 Mean Reversion Bands
│   └── rsi_divergence_daily.py         # B2 RSI Divergence
├── config.py                   # Transaction costs, asset classes, strategy manifest (Stage 1)
├── validate.py                 # Stage 1 out-of-sample walk-forward harness
├── stage2_portfolio_ga/        # Stage 2 — portfolio-level genetic algorithm (Section III-C)
│   ├── ga_engine.py                # GA: chromosome (strategy index + weight simplex), crossover, mutation, elitism
│   ├── run_portfolio_ga.py         # Runner: one walk-forward period, precompute returns, evolve, evaluate OOS
│   ├── multi_run_evaluator.py      # 50 GA runs per (period, variant) -> distributions for inference
│   ├── hyperparameter_tuning.py    # Per-period Optuna TPE + median pruning
│   ├── config.py                   # GA params, walk-forward periods, strategy registry, variant pools
│   ├── strategies.py               # Maps a strategy index (genotype) to a signal generator (phenotype)
│   ├── backtester.py               # Long-only signals -> P&L / Sharpe / drawdown
│   ├── data_loader.py              # Loads + resamples daily OHLCV
│   ├── per_strategy_eval.py        # Stage 1 per-strategy ranking (CIFER build)
│   ├── generate_significance_tests_doc.py  # Mann-Whitney U, Holm-Bonferroni, Cohen's d, bootstrap, Kruskal-Wallis
│   ├── plotting.py                 # Result figures
│   ├── pseudocode.tex              # GA pseudocode
│   └── GA_DESCRIPTION.md           # Prose description of the GA
├── requirements.txt
└── README.md
```

The numbering `(1)–(8)` matches **Table I** of the paper (the 8 LLM-generated
strategies across 7 mathematical domains). `B1–B6` are the conventional
benchmarks; Buy & Hold (B6) is computed inline in `validate.py`.

## The generation prompt

`prompt/llq_prompt.txt` is the structured prompt given to the agent (Claude
Opus, via Claude Code) at the start of each generation session. It encodes the
constraints described in Section III-A of the paper: vectorbt as the single
backtesting engine, strict no-lookahead and dev/test isolation, a fixed
transaction-cost model, a 60-backtest budget, a minimum of 5 distinct hypotheses
in the first 25% of the budget, and the aggregated objective

```
score = median(segment_sharpe)
        − 0.5 × median(segment_max_drawdown)
        − 0.2 × median(segment_turnover_proxy)
```

Each strategy was generated in its own session, frozen, and then evaluated
out-of-sample by the author-controlled harness — never by the agent's own
session logs.

## Strategy interface

Every strategy module exposes:

```python
def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input : df with columns [open, high, low, close, volume], daily bars.
    Output: DataFrame with boolean columns [entries, exits], already
            shifted by one bar so decisions at time t use only data <= t.
    """
```

## Reproducing the out-of-sample evaluation (Stage 1)

```bash
pip install -r requirements.txt
python validate.py
```

`validate.py` loads daily OHLCV test files from `data/test/<ASSET>/`, runs each
strategy over the 4 OOS test periods across the 19 assets, applies per-asset
transaction costs (`config.py`), and writes `results/walk_forward_results.csv`
plus summary tables by strategy, asset class, and period.

> **Data note.** The daily OHLCV price data is sourced from Pepperstone and is
> **not redistributed** in this repository. Drop your own daily CSVs into
> `data/test/<ASSET>/<asset>_<period>.csv` (e.g. `data/test/BTCUSD/btcusd_2023.csv`)
> to re-run the harness. See `config.py` for the asset list and expected
> transaction costs.

## Stage 2 — portfolio genetic algorithm

`stage2_portfolio_ga/` contains the portfolio-level GA of Section III-C. The GA
jointly optimises, over the 19 assets, (i) which strategy runs on each asset and
(ii) its capital weight (a point on the probability simplex), with fitness equal
to the annualised Sharpe of the *combined* weighted portfolio (so inter-asset
covariance and diversification matter). It is run under three pools — **Full**
(LLM + benchmarks + Cash), **LLM-Only**, and **Benchmark-Only** — with per-period
hyperparameters tuned by Optuna (TPE + median pruning) on a validation split that
never sees the OOS test year.

```bash
pip install -r requirements.txt
cd stage2_portfolio_ga

# 1. Tune GA hyperparameters per period (Optuna)
python hyperparameter_tuning.py

# 2. 50 GA runs per (period, variant) -> distributions for statistical testing
python multi_run_evaluator.py

# A single period/variant GA run can also be launched directly:
python run_portfolio_ga.py
```

Paths are anchored to the repo (no absolute paths); point the GA at your local
daily data with the `LLQ_DATA_DIR` / `LLQ_TEST_SET_DIR` environment variables
(same Pepperstone data note as above — not redistributed here). The strategy
modules are loaded from the top-level `strategies/` directory.

> **Note on scope.** The default configuration reproduces the paper: **6**
> walk-forward windows P1–P6 (2018–2024, Table II) and the paper strategy pool
> (7 LLM strategies + 6 benchmarks + Cash, pinned by `ACTIVE_LLM_INDICES` /
> `PAPER_BENCHMARK_INDICES`). The `STRATEGY_REGISTRY` retains a few extra,
> unused entries (exploratory strategies not selected for the paper) purely for
> provenance — they are not part of any variant pool. A 7th walk-forward window
> (2025 OOS) is included as a commented-out optional extension.

## Citation

```bibtex
@inproceedings{clayman2026llq,
  title     = {Large Language Quant: Cross-Domain Mathematical Trading
               Strategy Generation with an Agentic LLM},
  author    = {Clayman, Andrew and Kampouridis, Michael and Papastylianou, Tasos},
  booktitle = {IEEE Symposium on Computational Intelligence for Financial
               Engineering \& Economics (CIFEr)},
  year      = {2026}
}
```
