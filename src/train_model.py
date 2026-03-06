"""
train_model.py
--------------
Trains a RandomForestRegressor on the preprocessed and feature-engineered
retail sales data. Evaluates performance and saves the model to disk.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import preprocess
from src.feature_engineering import engineer_features


# ─── Configuration ────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "sales_data.csv")
MODEL_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "model.pkl")
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# Features used to train the model
FEATURE_COLS = [
    'store_id', 'product_id',
    'year', 'month', 'day', 'day_of_week',
    'is_weekend', 'quarter', 'day_of_year',
    'promotion', 'holiday',
    'sales_lag_1', 'sales_lag_7', 'sales_lag_14',
    'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30'
]
TARGET_COL = 'sales'
# ──────────────────────────────────────────────────────────────────────────────


def load_and_prepare_data():
    """Load, preprocess, and engineer features from the raw dataset."""
    print("\n" + "="*55)
    print("  STEP 1 — Loading and Preprocessing Data")
    print("="*55)
    df = preprocess(DATASET_PATH)

    print("\n" + "="*55)
    print("  STEP 2 — Feature Engineering")
    print("="*55)
    df = engineer_features(df)
    return df


def split_data(df: pd.DataFrame):
    """Split into training and test sets (time-based split)."""
    print("\n" + "="*55)
    print("  STEP 3 — Splitting Data (Train / Test)")
    print("="*55)

    # Use last 20% of dates as test (respects temporal order)
    df = df.sort_values('date')
    split_idx = int(len(df) * (1 - TEST_SIZE))
    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[TARGET_COL]

    print(f"[INFO] Train samples: {len(X_train):,}  |  Test samples: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, test_df


def train_model(X_train, y_train):
    """Train a RandomForestRegressor on the training data."""
    print("\n" + "="*55)
    print("  STEP 4 — Training RandomForestRegressor")
    print("="*55)

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("[INFO] Model training complete.")
    return model


def evaluate_model(model, X_test, y_test, test_df):
    """Compute MAE, RMSE, and R² on the test set."""
    print("\n" + "="*55)
    print("  STEP 5 — Model Evaluation")
    print("="*55)

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  Mean Absolute Error  (MAE) : {mae:>7.2f} │")
    print(f"  │  Root Mean Sq Error  (RMSE) : {rmse:>7.2f} │")
    print(f"  │  R² Score                   : {r2:>7.4f} │")
    print(f"  └─────────────────────────────────────┘\n")

    # Save predictions vs actual plot
    _save_prediction_plot(test_df, y_pred)

    return mae, rmse, r2


def _save_prediction_plot(test_df, y_pred):
    """Save actual vs predicted sales chart."""
    sample = test_df[test_df['store_id'] == 1].head(100)
    preds  = y_pred[:len(sample)]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(sample['date'].values, sample['sales'].values, label='Actual', color='steelblue')
    ax.plot(sample['date'].values, preds,                  label='Predicted', color='tomato', linestyle='--')
    ax.set_title('Actual vs Predicted Sales (Store 1 — Test Set)', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prediction_plot.png")
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"[INFO] Prediction plot saved → {out_path}")


def save_model(model, feature_cols):
    """Persist the trained model and feature list using joblib."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    payload = {'model': model, 'feature_cols': feature_cols}
    joblib.dump(payload, MODEL_PATH)
    print(f"[INFO] Model saved → {MODEL_PATH}")


def print_feature_importance(model, feature_cols):
    """Display top-10 most important features."""
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top10 = importances.sort_values(ascending=False).head(10)
    print("\n  Top 10 Feature Importances:")
    for feat, imp in top10.items():
        bar = '█' * int(imp * 60)
        print(f"  {feat:<22} {bar}  ({imp:.4f})")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_and_prepare_data()
    X_train, X_test, y_train, y_test, test_df = split_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, test_df)
    print_feature_importance(model, FEATURE_COLS)
    save_model(model, FEATURE_COLS)

    print("\n" + "="*55)
    print("  ✅  Training pipeline completed successfully!")
    print("="*55 + "\n")
