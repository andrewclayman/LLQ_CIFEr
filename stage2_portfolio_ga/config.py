"""
Configuration for Portfolio-Level GA
=====================================

Proper GA: genes interact because fitness is the Sharpe ratio of the
combined weighted portfolio return, not the average of independent
per-asset Sharpes. Changing one asset's strategy affects the optimal
choice for all other assets through covariance/diversification effects.

Chromosome: (strategy_index, portfolio_weight) per asset
- strategy_index: integer 0-18
- portfolio_weight: float, normalised to sum to 1
"""
import os

# --- Paths ---
# Portable paths anchored to this file. The GA loads the frozen strategy
# modules from the repo's top-level ``strategies/`` directory (both the LLM
# strategies and the benchmarks live there). Set LLQ_DATA_DIR to point at your
# local daily OHLCV data (Pepperstone; not redistributed — see the main README).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

DATA_DIR = os.environ.get('LLQ_DATA_DIR', os.path.join(_REPO_ROOT, 'data'))
STRATEGY_DIR = os.path.join(_REPO_ROOT, 'strategies')
BENCHMARK_DIR = os.path.join(_REPO_ROOT, 'strategies')
OUTPUT_DIR = os.path.join(_HERE, 'results')
LOG_DIR = os.path.join(_HERE, 'logs')

# --- Assets ---
ASSETS = [
    'BTCUSD', 'ETHUSD', 'BCHUSD',
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'NZDUSD',
    'DE40', 'F40', 'STOXX50', 'UK100', 'US30', 'US500',
    'USTEC', 'JP225',
    'XAUUSD', 'XAGUSD', 'XBRUSD',
]

ASSET_CLASSES = {
    'BTCUSD': 'crypto', 'ETHUSD': 'crypto', 'BCHUSD': 'crypto',
    'EURUSD': 'forex', 'GBPUSD': 'forex', 'USDJPY': 'forex',
    'USDCHF': 'forex', 'NZDUSD': 'forex',
    'DE40': 'equity', 'F40': 'equity', 'STOXX50': 'equity',
    'UK100': 'equity', 'US30': 'equity', 'US500': 'equity',
    'USTEC': 'equity', 'JP225': 'equity',
    'XAUUSD': 'commodity', 'XAGUSD': 'commodity', 'XBRUSD': 'commodity',
}

# --- Strategy Registry ---
STRATEGY_REGISTRY = {
    0:  ('spectral_entropy',            'Spectral Entropy',          False),
    1:  ('rough_path_signatures',       'Rough Path Signatures',     False),
    2:  ('information_geometry',        'Information Geometry',       False),
    3:  ('wasserstein_distance',        'Wasserstein Distance',      False),
    4:  ('rate_distortion_complexity',  'Rate Distortion',           False),
    5:  ('fisher_information_regime',   'Fisher Information',        False),
    6:  ('hurst_regime_daily',         'Hurst Mean Reversion',       False),
    7:  ('transfer_entropy',           'Transfer Entropy',           False),
    8:  ('approximate_entropy',        'Approx Entropy',             False),
    9:  ('permutation_entropy',        'Permutation Entropy',        False),
    10: ('volatility_breakout_daily',  'Volatility Breakout',        False),
    11: ('mean_reversion_bands_daily', 'Mean Reversion Bands',       False),
    12: ('rsi_divergence_daily',       'RSI Divergence',             False),
    13: ('regime_hmm_proxy_daily',     'Regime HMM Proxy',           False),
    14: ('benchmarks:ma_crossover',    'MA Crossover',               True),
    15: ('benchmarks:ts_momentum',     'TS Momentum',                True),
    16: ('benchmarks:bollinger_breakout', 'Bollinger Breakout',      True),
    17: ('benchmarks:donchian_breakout',  'Donchian Breakout',       True),
    18: ('hold',                       'Buy & Hold',                  True),
    19: ('volatility_phase_transition', 'Volatility Phase Transition', False),
    20: ('flat',                       'Cash (Flat)',                  True),
}

N_STRATEGIES = len(STRATEGY_REGISTRY)

# Strategy pool aligned with paper:
#   7 LLM strategies (indices 0, 1, 2, 4, 5, 6, 19)
#   6 paper benchmarks (Mean Rev Bands, RSI Divergence, MA Crossover, Bollinger,
#       Donchian, Buy & Hold) — indices 11, 12, 14, 16, 17, 18
#   Cash (20) — structural no-op, available to all variants
#
# Excluded from full pool:
#   3  (Wasserstein Distance)  — 0 trades in 1-year OOS windows (240-bar baseline)
#   7  (Transfer Entropy)      — 0 trades in original evaluation
#   8  (Approx Entropy)        — 0 trades in original evaluation
#   9  (Permutation Entropy)   — 0 trades in original evaluation
#  10  (Volatility Breakout)   — superseded by Volatility Phase Transition (19)
#  13  (Regime HMM Proxy)      — non-textbook, not part of paper benchmark set
#  15  (TS Momentum)           — redundant with MA Crossover (both trend-following)
CASH_INDEX = 20
ACTIVE_LLM_INDICES = [0, 1, 2, 4, 5, 6, 19]
PAPER_BENCHMARK_INDICES = [11, 12, 14, 16, 17, 18]  # MRB, RSI Div, MA, Bollinger, Donchian, B&H
FULL_STRATEGY_INDICES = ACTIVE_LLM_INDICES + PAPER_BENCHMARK_INDICES + [CASH_INDEX]

# LLM-only subset (paper's LLM strategies + cash)
LLM_STRATEGY_INDICES = ACTIVE_LLM_INDICES + [CASH_INDEX]

# Benchmark-only subset (paper's 6 benchmarks + cash)
BENCHMARK_STRATEGY_INDICES = PAPER_BENCHMARK_INDICES + [CASH_INDEX]

# --- GA Parameters ---
POPULATION_SIZE = 300
N_GENERATIONS = 300
TOURNAMENT_SIZE = 4
CROSSOVER_RATE = 0.7
CROSSOVER_TYPE = 'two_point'  # 'one_point', 'two_point', or 'uniform'
STRATEGY_MUTATION_RATE = 0.15
WEIGHT_MUTATION_RATE = 0.20
WEIGHT_MUTATION_SIGMA = 0.05
ELITE_COUNT = 3

# --- Walk-Forward Parameters ---
# 7 rolling windows: 3yr training -> 1yr OOS
WALK_FORWARD_PERIODS = [
    ('2015-01-01', '2017-12-31', '2018-01-01', '2018-12-31'),
    ('2017-01-01', '2019-12-31', '2020-01-01', '2020-12-31'),
    ('2018-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),
    ('2019-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
    ('2020-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
    ('2021-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
    ('2022-01-01', '2024-12-31', '2025-01-01', '2025-12-31'),
]

# --- Backtesting ---
RISK_FREE_RATE = 0.0
TRADING_DAYS_PER_YEAR = 252

# --- Transaction Costs (95th percentile spread from tick data) ---
TRANSACTION_COSTS = {
    'EURUSD': 0.00001,   'GBPUSD': 0.00001,   'USDJPY': 0.00001,
    'USDCHF': 0.00001,   'NZDUSD': 0.00002,
    'BTCUSD': 0.00011,   'ETHUSD': 0.00070,   'BCHUSD': 0.00498,
    'US30':   0.00005,   'US500':  0.00003,   'USTEC':  0.00004,
    'DE40':   0.00005,   'UK100':  0.00005,   'JP225':  0.00010,
    'F40':    0.00010,   'STOXX50': 0.00008,
    'XAUUSD': 0.00010,   'XAGUSD': 0.00020,   'XBRUSD': 0.00015,
}
DEFAULT_TRANSACTION_COST = 0.00025

# --- Hyperparameter Tuning ---
TUNING_RANGES = {
    'population_size': (100, 500),
    'n_generations': (50, 500),
    'crossover_rate': (0.4, 0.9),
    'strategy_mutation_rate': (0.05, 0.30),
    'weight_mutation_rate': (0.05, 0.40),
    'weight_mutation_sigma': (0.01, 0.15),
    'elite_count': (1, 5),
}

# Tuning walk-forward: 2yr train / 1yr validation carved from the 3yr train windows
TUNING_WALK_FORWARD_PERIODS = [
    ('2015-01-01', '2016-12-31', '2017-01-01', '2017-12-31'),
    ('2017-01-01', '2018-12-31', '2019-01-01', '2019-12-31'),
    ('2018-01-01', '2019-12-31', '2020-01-01', '2020-12-31'),
    ('2019-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),
    ('2020-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
    ('2021-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
    ('2022-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
]

TUNING_N_TRIALS = 50
TUNING_RUNS_PER_TRIAL = 3
TUNING_TIMEOUT_SECONDS = None  # no timeout — run all trials to completion

# --- Multi-Run Evaluation ---
FINAL_N_RUNS = 50
