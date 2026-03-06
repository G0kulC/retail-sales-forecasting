"""
feature_engineering.py
-----------------------
Creates lag features and rolling statistics to help the model
understand past sales patterns (time-series style features).
"""

import pandas as pd
import numpy as np


def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by store, product, and date to ensure correct lag ordering."""
    df = df.sort_values(by=['store_id', 'product_id', 'date']).reset_index(drop=True)
    print("[INFO] Data sorted by store_id, product_id, date.")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag features:
    - sales_lag_1: Sales 1 day ago (short-term trend)
    - sales_lag_7: Sales 7 days ago (weekly seasonality)
    - sales_lag_14: Sales 14 days ago (bi-weekly pattern)
    """
    group = df.groupby(['store_id', 'product_id'])['sales']
    df['sales_lag_1'] = group.shift(1)
    df['sales_lag_7'] = group.shift(7)
    df['sales_lag_14'] = group.shift(14)
    print("[INFO] Added lag features: sales_lag_1, sales_lag_7, sales_lag_14.")
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling window statistics:
    - rolling_mean_7: 7-day rolling average (smoothed trend)
    - rolling_std_7: 7-day rolling standard deviation (volatility)
    - rolling_mean_30: 30-day rolling average (monthly trend)
    """
    group = df.groupby(['store_id', 'product_id'])['sales']
    df['rolling_mean_7'] = group.transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    df['rolling_std_7'] = group.transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).std())
    df['rolling_mean_30'] = group.transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean())
    print("[INFO] Added rolling features: rolling_mean_7, rolling_std_7, rolling_mean_30.")
    return df


def drop_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with NaN values introduced by lag features.
    These typically appear at the start of each store-product group.
    """
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    print(f"[INFO] Dropped {before - after} rows with NaN lag values. Remaining: {after} rows.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
    1. Sort data
    2. Add lag features
    3. Add rolling features
    4. Drop NaN rows
    Returns enriched DataFrame.
    """
    df = sort_data(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = drop_na_rows(df)
    print(f"[INFO] Feature engineering complete. Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_preprocessing import preprocess

    df = preprocess("dataset/sales_data.csv")
    df = engineer_features(df)
    print(df[['date', 'store_id', 'product_id', 'sales',
              'sales_lag_1', 'sales_lag_7', 'rolling_mean_7']].head(10))
