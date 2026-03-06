"""
predict.py
----------
Loads the trained model and provides a predict_sales() function
that returns a sales forecast for given inputs.
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime


# Path to the saved model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "model.pkl")


def load_model():
    """Load the trained model payload from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            "Please run:  python src/train_model.py"
        )
    payload = joblib.load(MODEL_PATH)
    return payload['model'], payload['feature_cols']


def build_input_features(store_id: int, product_id: int, date: str,
                         promotion: int, holiday: int,
                         sales_lag_1: float = None,
                         sales_lag_7: float = None,
                         sales_lag_14: float = None,
                         rolling_mean_7: float = None,
                         rolling_std_7: float = None,
                         rolling_mean_30: float = None) -> pd.DataFrame:
    """
    Construct the feature row that the model expects.
    Lag & rolling values default to reasonable estimates if not provided.
    """
    dt = pd.to_datetime(date)

    # Default lag/rolling values using store-product average if not supplied
    default_sales = 200  # sensible fallback
    row = {
        'store_id'       : store_id,
        'product_id'     : product_id,
        'year'           : dt.year,
        'month'          : dt.month,
        'day'            : dt.day,
        'day_of_week'    : dt.dayofweek,
        'is_weekend'     : int(dt.dayofweek >= 5),
        'quarter'        : dt.quarter,
        'day_of_year'    : dt.dayofyear,
        'promotion'      : promotion,
        'holiday'        : holiday,
        'sales_lag_1'    : sales_lag_1    if sales_lag_1    is not None else default_sales,
        'sales_lag_7'    : sales_lag_7    if sales_lag_7    is not None else default_sales,
        'sales_lag_14'   : sales_lag_14   if sales_lag_14   is not None else default_sales,
        'rolling_mean_7' : rolling_mean_7 if rolling_mean_7 is not None else default_sales,
        'rolling_std_7'  : rolling_std_7  if rolling_std_7  is not None else 15.0,
        'rolling_mean_30': rolling_mean_30 if rolling_mean_30 is not None else default_sales,
    }
    return pd.DataFrame([row])


def predict_sales(store_id: int, product_id: int, date: str,
                  promotion: int, holiday: int,
                  sales_lag_1: float = None,
                  sales_lag_7: float = None,
                  sales_lag_14: float = None,
                  rolling_mean_7: float = None,
                  rolling_std_7: float = None,
                  rolling_mean_30: float = None) -> int:
    """
    Predict retail sales for a given store/product/date combination.

    Parameters
    ----------
    store_id    : Store identifier (int)
    product_id  : Product identifier (int)
    date        : Forecast date as 'YYYY-MM-DD' string
    promotion   : 1 if promotion active, else 0
    holiday     : 1 if public holiday, else 0
    sales_lag_* : Optional historical sales figures for better accuracy

    Returns
    -------
    Predicted sales as a rounded integer.
    """
    model, feature_cols = load_model()
    X = build_input_features(
        store_id, product_id, date, promotion, holiday,
        sales_lag_1, sales_lag_7, sales_lag_14,
        rolling_mean_7, rolling_std_7, rolling_mean_30
    )
    # Ensure column order matches training
    X = X[feature_cols]
    prediction = model.predict(X)[0]
    return max(0, round(int(prediction)))


if __name__ == "__main__":
    # Quick smoke test
    result = predict_sales(
        store_id=1,
        product_id=101,
        date="2025-01-15",
        promotion=1,
        holiday=0
    )
    print(f"Predicted Sales: {result} units")
