"""
app.py
------
Retail Sales Forecasting — Streamlit Web Application
Run with:  streamlit run app.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import date, timedelta

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.predict import predict_sales

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.35);
    }
    .result-number {
        font-size: 4rem;
        font-weight: 900;
        letter-spacing: -2px;
    }
    .metric-card {
        background: #f8f9ff;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e0e4ff;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #444;
        border-left: 4px solid #667eea;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stDateInput"] label,
    div[data-testid="stNumberInput"] label {
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🛒 Retail Sales Forecasting System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered sales prediction using Machine Learning · Built with Python & Streamlit</div>', unsafe_allow_html=True)
st.divider()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/combo-chart.png", width=80)
    st.markdown("### About This App")
    st.info(
        "This application uses a **Random Forest Regressor** trained on "
        "historical retail sales data to forecast future sales.\n\n"
        "**Model Features:**\n"
        "- Lag & rolling statistics\n"
        "- Date-based seasonality\n"
        "- Promotion & holiday effects"
    )
    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.markdown("🐍 Python · 🧠 Scikit-learn")
    st.markdown("📊 Pandas · 🌐 Streamlit")
    st.markdown("---")
    st.caption("© 2026 - Made with ❤️ by Retail ML")

# ─── Input Form ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Enter Prediction Parameters</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    store_id = st.selectbox(
        "🏪 Store ID",
        options=[1, 2, 3],
        help="Select the retail store location"
    )

with col2:
    product_id = st.selectbox(
        "📦 Product ID",
        options=[101, 102, 103, 104],
        help="Select the product to forecast"
    )

with col3:
    forecast_date = st.date_input(
        "📅 Forecast Date",
        value=date.today() + timedelta(days=1),
        min_value=date(2020, 1, 1),
        max_value=date(2030, 12, 31),
        help="Date for which to predict sales"
    )

col4, col5 = st.columns(2)
with col4:
    promotion = st.radio(
        "🎁 Promotion Active?",
        options=["No", "Yes"],
        horizontal=True,
        help="Is a promotional campaign running on this date?"
    )

with col5:
    holiday = st.radio(
        "🎉 Public Holiday?",
        options=["No", "Yes"],
        horizontal=True,
        help="Is this date a public holiday?"
    )

# Advanced options (collapsible)
with st.expander("⚙️ Advanced: Provide Historical Sales (optional — improves accuracy)"):
    adv_col1, adv_col2, adv_col3 = st.columns(3)
    with adv_col1:
        lag1  = st.number_input("Sales Yesterday (lag-1)",  min_value=0, value=200, step=10)
        lag7  = st.number_input("Sales 7 Days Ago (lag-7)", min_value=0, value=200, step=10)
    with adv_col2:
        lag14 = st.number_input("Sales 14 Days Ago",        min_value=0, value=200, step=10)
        rm7   = st.number_input("7-Day Rolling Mean",       min_value=0, value=200, step=10)
    with adv_col3:
        rs7   = st.number_input("7-Day Rolling Std Dev",    min_value=0, value=15,  step=5)
        rm30  = st.number_input("30-Day Rolling Mean",      min_value=0, value=200, step=10)
    use_advanced = st.checkbox("Use these values in prediction", value=False)

# ─── Prediction ───────────────────────────────────────────────────────────────
st.markdown("")
predict_btn = st.button("🔮 Predict Sales", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Generating forecast…"):
        try:
            promo_val   = 1 if promotion == "Yes" else 0
            holiday_val = 1 if holiday   == "Yes" else 0

            if use_advanced:
                predicted = predict_sales(
                    store_id=store_id,
                    product_id=product_id,
                    date=str(forecast_date),
                    promotion=promo_val,
                    holiday=holiday_val,
                    sales_lag_1=lag1,
                    sales_lag_7=lag7,
                    sales_lag_14=lag14,
                    rolling_mean_7=rm7,
                    rolling_std_7=rs7,
                    rolling_mean_30=rm30
                )
            else:
                predicted = predict_sales(
                    store_id=store_id,
                    product_id=product_id,
                    date=str(forecast_date),
                    promotion=promo_val,
                    holiday=holiday_val
                )

            # ─── Result Display ───────────────────────────────────────────
            st.markdown('<div class="section-header">📈 Forecast Result</div>', unsafe_allow_html=True)

            res_col, meta_col = st.columns([1, 2])

            with res_col:
                st.markdown(f"""
                <div class="result-card">
                    <div style="font-size:1rem; opacity:0.85; margin-bottom:0.5rem;">Predicted Sales</div>
                    <div class="result-number">{predicted:,}</div>
                    <div style="font-size:1rem; opacity:0.85; margin-top:0.5rem;">units</div>
                </div>
                """, unsafe_allow_html=True)

            with meta_col:
                st.markdown("**Prediction Summary**")
                m1, m2 = st.columns(2)
                m1.metric("🏪 Store",     f"Store {store_id}")
                m2.metric("📦 Product",   f"Product {product_id}")
                m1.metric("📅 Date",      str(forecast_date))
                m2.metric("🎁 Promotion", promotion)
                m1.metric("🎉 Holiday",   holiday)
                d = pd.to_datetime(str(forecast_date))
                m2.metric("📆 Day",       d.strftime("%A"))

            # ─── 7-Day Forecast Chart ─────────────────────────────────────
            st.markdown('<div class="section-header">📊 7-Day Sales Forecast</div>', unsafe_allow_html=True)

            dates_7, preds_7 = [], []
            for i in range(7):
                future_date = forecast_date + timedelta(days=i)
                p = predict_sales(
                    store_id=store_id,
                    product_id=product_id,
                    date=str(future_date),
                    promotion=promo_val,
                    holiday=1 if future_date.weekday() == 6 else 0
                )
                dates_7.append(future_date.strftime("%b %d\n%a"))
                preds_7.append(p)

            fig, ax = plt.subplots(figsize=(10, 3.5))
            colors = ['#667eea' if i == 0 else '#a5b4fc' for i in range(7)]
            bars = ax.bar(dates_7, preds_7, color=colors, width=0.6, edgecolor='white', linewidth=1.5)
            for bar, val in zip(bars, preds_7):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        str(val), ha='center', va='bottom', fontsize=9, fontweight='bold', color='#444')
            ax.set_ylabel("Predicted Sales (units)", fontsize=10)
            ax.set_title(f"7-Day Forecast · Store {store_id} · Product {product_id}", fontsize=12, fontweight='bold')
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_ylim(0, max(preds_7) * 1.2)
            ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            fig.patch.set_facecolor('#fafbff')
            ax.set_facecolor('#fafbff')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.success("✅ Forecast generated successfully!")

        except FileNotFoundError as e:
            st.error(f"⚠️ {e}")
            st.code("python src/train_model.py", language="bash")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            raise e

# ─── Dataset Preview ──────────────────────────────────────────────────────────
with st.expander("🗂️ View Sample Dataset"):
    try:
        df = pd.read_csv("dataset/sales_data.csv")
        st.dataframe(df.head(20), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Date Range", f"{df['date'].min()} → {df['date'].max()}")
        c3.metric("Unique Products", df['product_id'].nunique())
    except Exception:
        st.warning("Dataset not found. Please ensure dataset/sales_data.csv exists.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("🛒 Retail Sales Forecasting System · ML Project · Built with ❤️ using Python, Scikit-learn & Streamlit")
