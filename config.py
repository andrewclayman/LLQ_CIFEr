"""
Configuration for LLM Strategy Generation project.
Transaction costs from 95th percentile bid-ask spreads.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent
STRATEGIES_DIR = BASE_DIR / "strategies"
DEV_DIR = BASE_DIR / "data" / "dev"
TEST_DIR = BASE_DIR / "data" / "test"
RESULTS_DIR = BASE_DIR / "results"

TEST_PERIODS = ['2013_2015', '2019_202001', '2021', '2023']
PERIODS_PER_YEAR = 252  # Daily data

# Transaction costs (95th percentile spread from tick data)
TRANSACTION_COSTS = {
    'EURUSD': 0.00001,
    'GBPUSD': 0.00001,
    'USDJPY': 0.00001,
    'USDCHF': 0.00001,
    'NZDUSD': 0.00002,
    'BTCUSD': 0.00011,
    'ETHUSD': 0.00070,
    'BCHUSD': 0.00498,
    'US30': 0.00005,
    'US500': 0.00003,
    'USTEC': 0.00004,
    'DE40': 0.00005,
    'UK100': 0.00005,
    'JP225': 0.00010,
    'F40': 0.00010,
    'STOXX50': 0.00008,
    'XAUUSD': 0.00010,
    'XAGUSD': 0.00020,
    'XBRUSD': 0.00015,
}

DEFAULT_COST = 0.00025

# Risk-free rates by currency (annual)
RISK_FREE_RATES = {
    'USD': 0.02,
    'EUR': 0.00,
    'GBP': 0.01,
    'JPY': -0.001,
    'CHF': -0.005,
    'NZD': 0.02,
    'AUD': 0.02,
}

# Asset classifications
ASSET_CLASSES = {
    'BTCUSD': 'crypto', 'ETHUSD': 'crypto', 'BCHUSD': 'crypto',
    'EURUSD': 'forex', 'GBPUSD': 'forex', 'USDJPY': 'forex',
    'USDCHF': 'forex', 'NZDUSD': 'forex',
    'US30': 'equity', 'US500': 'equity', 'USTEC': 'equity',
    'DE40': 'equity', 'UK100': 'equity', 'JP225': 'equity',
    'F40': 'equity', 'STOXX50': 'equity',
    'XAUUSD': 'commodity', 'XAGUSD': 'commodity', 'XBRUSD': 'commodity',
}

# Strategy classifications
# The 8 LLM-generated strategies reported in the paper (Table I).
NOVEL_STRATEGIES = [
    'spectral_entropy',            # Spectral Entropy      (Information Theory)
    'rough_path_signatures',       # Rough Path Signatures (Stochastic Analysis)
    'information_geometry',        # Information Geometry   (Differential Geometry)
    'wasserstein_distance',        # Wasserstein Distance   (Optimal Transport)
    'rate_distortion_complexity',  # Rate Distortion        (Information Theory)
    'fisher_information_regime',   # Fisher Information      (Information Geometry)
    'volatility_phase_transition', # Volatility Phase Trans.(Statistical Physics)
    'hurst_regime_daily',          # Hurst Mean Reversion   (Fractal Analysis)
]

# The 5 active benchmarks (B1-B5) reported in the paper; Buy & Hold (B6) is
# computed inline in validate.py.
BENCHMARK_STRATEGIES = [
    'benchmarks',  # contains: ma_crossover, bollinger_breakout, donchian_breakout
    'mean_reversion_bands_daily',
    'rsi_divergence_daily',
]
