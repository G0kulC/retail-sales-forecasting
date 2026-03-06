# 🛒 Retail Sales Forecasting System

> **Final Year Machine Learning Project**  
> Predict future retail sales using historical data with Python, Scikit-learn & Streamlit.

---

## 📌 Project Overview

This project builds an **end-to-end Machine Learning pipeline** to forecast daily retail sales for multiple stores and products. It covers the full ML lifecycle — from raw data ingestion and feature engineering to model training, evaluation, and deployment via an interactive web application.

The system uses a **Random Forest Regressor** enriched with time-series features (lag values, rolling averages) to capture seasonality, promotional effects, and weekly patterns in sales data.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 EDA Notebook | Exploratory Data Analysis with rich visualizations |
| 🔧 Preprocessing | Date feature extraction, missing value handling |
| ⚙️ Feature Engineering | Lag features, rolling statistics |
| 🤖 ML Model | Random Forest Regressor with 150 estimators |
| 📈 Evaluation | MAE, RMSE, R² metrics with prediction plots |
| 🌐 Web App | Streamlit UI with 7-day forecast chart |

---

## 📁 Project Structure

```
retail_sales_forecasting/
│
├── dataset/
│   ├── sales_data.csv          # Historical sales data (13,000+ rows)
│   └── generate_data.py        # Script to regenerate dataset
│
├── models/
│   ├── model.pkl               # Trained model (created after training)
│   └── prediction_plot.png     # Actual vs Predicted chart
│
├── src/
│   ├── data_preprocessing.py   # Step 1 — Load & clean data
│   ├── feature_engineering.py  # Step 2 — Lag & rolling features
│   ├── train_model.py          # Step 3 — Train & evaluate model
│   └── predict.py              # Step 4 — Prediction function
│
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
│
├── app.py                      # Step 5 — Streamlit web app
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### 1. Clone / Download the Project
```bash
git clone https://github.com/yourusername/retail_sales_forecasting.git
cd retail_sales_forecasting
```

### 2. (Optional) Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Train the Model
```bash
python src/train_model.py
```
This will:
- Load and preprocess the dataset
- Engineer lag and rolling features
- Train a Random Forest model
- Print MAE, RMSE, R² evaluation metrics
- Save `models/model.pkl`

### Step 2 — Launch the Web App
```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**

### Step 3 — (Optional) Explore EDA Notebook
```bash
jupyter notebook notebooks/eda.ipynb
```

---

## 📋 Dataset Format

The dataset (`dataset/sales_data.csv`) contains daily sales records with the following columns:

| Column | Type | Description |
|---|---|---|
| `date` | string | Sales date (YYYY-MM-DD) |
| `store_id` | int | Store identifier (1, 2, 3) |
| `product_id` | int | Product identifier (101–104) |
| `sales` | int | Units sold |
| `promotion` | int | 1 = promotion active, 0 = none |
| `holiday` | int | 1 = public holiday, 0 = normal day |

**Example rows:**
```
date,store_id,product_id,sales,promotion,holiday
2022-01-01,1,101,240,0,0
2022-01-02,1,101,223,0,1
2022-01-03,1,101,201,0,0
2022-01-15,1,101,280,1,0
```

---

## 🧠 ML Pipeline

```
Raw CSV Data
    │
    ▼
data_preprocessing.py
    ├── Convert date → datetime
    ├── Extract: year, month, day, day_of_week, is_weekend, quarter
    ├── Handle missing values (median/mode fill)
    └── Label encode categoricals
    │
    ▼
feature_engineering.py
    ├── sales_lag_1   (yesterday's sales)
    ├── sales_lag_7   (last week's sales)
    ├── sales_lag_14  (two weeks ago)
    ├── rolling_mean_7  (7-day average)
    ├── rolling_std_7   (7-day volatility)
    └── rolling_mean_30 (30-day trend)
    │
    ▼
train_model.py
    ├── Time-based train/test split (80/20)
    ├── RandomForestRegressor (150 trees)
    └── Evaluate: MAE, RMSE, R²
    │
    ▼
model.pkl  ──→  predict.py  ──→  app.py (Streamlit)
```

---

## 📊 Example Output

### Training Console
```
=======================================================
  STEP 5 — Model Evaluation
=======================================================

  ┌─────────────────────────────────────┐
  │  Mean Absolute Error  (MAE) :  12.45 │
  │  Root Mean Sq Error  (RMSE) :  18.72 │
  │  R² Score                   :  0.9231 │
  └─────────────────────────────────────┘
```

### Web App Prediction
```
📦 Store: 1 | Product: 101 | Date: 2025-06-15 | Promotion: Yes

Predicted Sales: 285 units
```

---

## 🛠️ Technologies Used

| Library | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Core language |
| Pandas | 2.2.2 | Data manipulation |
| NumPy | 1.26.4 | Numerical computing |
| Scikit-learn | 1.5.1 | ML model |
| Matplotlib | 3.9.0 | Charts & plots |
| Seaborn | 0.13.2 | Statistical visualizations |
| Streamlit | 1.36.0 | Web application |
| Joblib | 1.4.2 | Model serialization |

---

## 📚 References

- Breiman, L. (2001). *Random Forests*. Machine Learning, 45, 5–32.
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Streamlit Documentation](https://docs.streamlit.io)

---

## 👨‍💻 Author

**[Your Name]** · Final Year B.Tech / B.Sc Computer Science  
**College:** [Your College Name]  
**Academic Year:** 2024–25

---

*This project was developed as a Final Year Project to demonstrate applied Machine Learning for real-world retail forecasting.*
