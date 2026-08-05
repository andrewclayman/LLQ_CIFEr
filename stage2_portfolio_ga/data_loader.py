"""
Data Loader
============
Loads hourly data from BOTH dev files and test set files, merges them,
and resamples to daily OHLCV for strategy consumption.

The dev data intentionally excludes test periods (2013-2015, 2019, 2021, 2023)
to maintain OOS integrity. We merge both sources to get continuous coverage
for the GA's rolling walk-forward approach.
"""
import pandas as pd
import numpy as np
import os
from config import DATA_DIR, ASSETS

# Local OOS test-set directory (Pepperstone data; not redistributed).
# Override with the LLQ_TEST_SET_DIR environment variable.
TEST_SET_DIR = os.environ.get(
    'LLQ_TEST_SET_DIR',
    os.path.join(os.path.dirname(DATA_DIR), 'test_sets'),
)

_cache = {}


def load_asset_daily(asset: str) -> pd.DataFrame:
    """
    Load hourly data from dev + test set files, merge, and resample to daily OHLCV.
    """
    if asset in _cache:
        return _cache[asset]

    all_frames = []

    # Load dev data
    dev_path = os.path.join(DATA_DIR, f'{asset}_dev.csv')
    if os.path.exists(dev_path):
        df = pd.read_csv(dev_path, parse_dates=['datetime'])
        df = df.set_index('datetime')
        all_frames.append(df)

    # Load test set data (multiple CSV files per asset)
    test_dir = os.path.join(TEST_SET_DIR, asset)
    if os.path.isdir(test_dir):
        for csv_file in sorted(os.listdir(test_dir)):
            if not csv_file.endswith('.csv'):
                continue
            test_path = os.path.join(test_dir, csv_file)
            try:
                df = pd.read_csv(test_path, parse_dates=['datetime'])
                df = df.set_index('datetime')
                all_frames.append(df)
            except Exception:
                pass

    if not all_frames:
        print(f"  WARNING: No data files for {asset}")
        return pd.DataFrame()

    # Merge all frames and remove duplicates
    combined = pd.concat(all_frames)
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined.sort_index()

    # Resample hourly -> daily OHLCV
    daily = combined.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna(subset=['close'])

    _cache[asset] = daily
    return daily


def get_asset_data_for_period(asset: str, start: str, end: str) -> pd.DataFrame:
    """
    Get daily data for an asset within a date range.
    Returns empty DataFrame if insufficient data.
    """
    daily = load_asset_daily(asset)
    if daily.empty:
        return daily

    mask = (daily.index >= start) & (daily.index <= end)
    subset = daily[mask].copy()

    return subset


def get_available_assets_for_period(start: str, end: str, min_days: int = 200) -> list:
    """
    Return list of assets that have sufficient data for the given period.
    """
    available = []
    for asset in ASSETS:
        data = get_asset_data_for_period(asset, start, end)
        if len(data) >= min_days:
            available.append(asset)
    return available


def preload_all():
    """Preload all asset data into cache."""
    print("Loading and resampling all asset data to daily (dev + test sets)...")
    for asset in ASSETS:
        df = load_asset_daily(asset)
        if not df.empty:
            print(f"  {asset}: {len(df)} daily bars, "
                  f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        else:
            print(f"  {asset}: NO DATA")
    print()
