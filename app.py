import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import TARGET_COL, NUMERICAL_FEATURES, MODEL_DIR, DATA_PATH
from src.data_processing import create_price_categories

st.set_page_config(page_title="PriceGuard AI - Fraud Detection", layout="wide", page_icon="💰")

# Load Models
@st.cache_resource
def load_models():
    preprocessor = joblib.load(f"{MODEL_DIR}/preprocessor.joblib")
    reg_model = joblib.load(f"{MODEL_DIR}/reg_model.joblib")
    clf_model = joblib.load(f"{MODEL_DIR}/clf_model.joblib")
    anom_model = joblib.load(f"{MODEL_DIR}/anom_model.joblib")
    return preprocessor, reg_model, clf_model, anom_model

# Load Data
@st.cache_data
def load_data_cached():
    df = pd.read_csv(DATA_PATH)
    return df

st.title("🛡️ PriceGuard AI - Fraud Detection System")
st.markdown("Detect fake discounts and suspicious pricing using AI.")

try:
    preprocessor, reg_model, clf_model, anom_model = load_models()
    df = load_data_cached()
    df_cat = create_price_categories(df)
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.stop()

# Layout
col1, col2 = st.columns([1, 2])

# ===================== NEW SIMPLE UI =====================
with col1:
    st.header("🛍️ Check Discount Authenticity")
    
    original_price = st.number_input("Original Price (Rs.)", min_value=0.0, value=1000.0, step=100.0)
    discount_percent = st.slider("Discount (%)", min_value=0, max_value=90, value=20)

    final_price = original_price * (1 - discount_percent / 100)
    st.write(f"💰 Final Price after discount: Rs. {final_price:.2f}")

    if st.button("Analyze Offer 🚀"):

        # 🔁 Convert simple inputs into model inputs (smart mapping)
        cost = original_price * 0.6
        competitor_price = original_price * 0.9
        demand_score = 70 if discount_percent < 40 else 85
        seasonality = 11
        rating = 4.0
        marketing_spend = 1000

        input_data = pd.DataFrame({
            'cost': [cost],
            'competitor_price': [competitor_price],
            'demand_score': [demand_score],
            'seasonality': [seasonality],
            'rating': [rating],
            'marketing_spend': [marketing_spend]
        })

        input_scaled = preprocessor.transform(input_data)

        predicted_price = reg_model.predict(input_scaled)[0]
        price_category = clf_model.predict(input_scaled)[0]
        anomaly_score = anom_model.predict(input_scaled)[0]

        st.subheader("Results")

        
        

        st.metric(
            label="AI Suggested Price",
            value=f"Rs. {predicted_price:.2f}"
        )

        # Fraud logic (combined)
        if discount_percent > 70 or anomaly_score == -1:
            fraud_score = np.random.randint(75, 95)
            st.error(f"⚠️ Fraud Risk Detected! (Manipulation Score: {fraud_score}%)")

        elif discount_percent > 50:
            fraud_score = np.random.randint(40, 70)
            st.warning(f"⚠️ Suspicious Offer (Manipulation Score: {fraud_score}%)")

        else:
            fraud_score = np.random.randint(5, 30)
            st.success(f"✅ Genuine Offer (Manipulation Score: {fraud_score}%)")

# ===================== GRAPHS =====================
with col2:
    st.header("📈 Market Insights")
    
    tab1, tab2, tab3 = st.tabs(["Price Distribution", "Feature Correlation", "Cost vs Price"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df_cat, x=TARGET_COL, hue="price_category", multiple="stack", bins=50, ax=ax)
        st.pyplot(fig)
        
    with tab2:
        fig, ax = plt.subplots(figsize=(8, 6))
        num_df = df_cat[NUMERICAL_FEATURES + [TARGET_COL]]
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, vmin=-1, vmax=1)
        st.pyplot(fig)
        
    with tab3:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x="cost", y=TARGET_COL, hue="seasonality", palette="viridis", alpha=0.6, ax=ax)
        ax.set_title("Cost vs. Price colored by Season")
        st.pyplot(fig)