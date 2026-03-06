"""
data_preprocessing.py
---------------------
Handles loading and cleaning of retail sales data.
Extracts date features and handles missing values.
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV dataset from the given filepath."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded dataset with {len(df)} rows and {df.shape[1]} columns.")
    return df


def convert_date(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the 'date' column to datetime format."""
    df['date'] = pd.to_datetime(df['date'])
    print("[INFO] Converted 'date' column to datetime.")
    return df


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from the date column:
    - year, month, day, day_of_week
    """
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek   # 0=Monday, 6=Sunday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['day_of_year'] = df['date'].dt.dayofyear
    print("[INFO] Extracted date features: year, month, day, day_of_week, is_weekend, quarter, day_of_year.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
    - Fill numeric columns with median
    - Fill categorical columns with mode
    """
    before = df.isnull().sum().sum()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    after = df.isnull().sum().sum()
    print(f"[INFO] Handled missing values: {before} → {after} nulls remaining.")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode any remaining categorical (object) columns using label encoding.
    Skips the 'date' column.
    """
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != 'date']
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes
        print(f"[INFO] Label-encoded column: '{col}'")
    return df


def preprocess(filepath: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Load data
    2. Convert date
    3. Extract date features
    4. Handle missing values
    5. Encode categoricals
    Returns a clean DataFrame ready for feature engineering.
    """
    df = load_data(filepath)
    df = convert_date(df)
    df = extract_date_features(df)
    df = handle_missing_values(df)
    df = encode_categoricals(df)
    print(f"[INFO] Preprocessing complete. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = preprocess("dataset/sales_data.csv")
    print(df.head())
