"""
SDG 13 Climate Anomaly Predictor Web App
Supervised ML for UN SDG 13: Climate Action.

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

@st.cache_data
def load_data_and_model():
    """
    Load, preprocess, and train model.
    Dataset: Open-source NASA GISTEMP via DataHub (1880-2025 anomalies).
    Handles potential network issues with manual fallback note.
    """
    try:
        url = "https://datahub.io/core/global-temp/_r/-/data/annual.csv"
        df = pd.read_csv(url)
    except:
        st.error("Network issue loading data. Download 'annual.csv' from DataHub and upload manually.")
        st.stop()
    
    # Preprocess: Filter, clean, add features
    df = df[df['Source'] == 'GISTEMP']
    df = df[['Year', 'Mean']]
    df.columns = ['Year', 'Annual_Anomaly']
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Annual_Anomaly'] = pd.to_numeric(df['Annual_Anomaly'], errors='coerce')
    df = df.dropna()
    
    # Proxy features (expand with World Bank API in production)
    df['CO2_proxy'] = 280 + (df['Year'] - 1880) * 0.8
    df['Industrial_Growth'] = np.where(df['Year'] > 1950, (df['Year'] - 1950) * 0.5, 0)
    
    # Normalize & split
    X = df[['Year', 'CO2_proxy', 'Industrial_Growth']]
    y = df['Annual_Anomaly']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train supervised model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    model_r2 = r2_score(y_test, y_pred)
    
    return df, model, scaler, model_r2

# App
st.title("🌍 SDG 13: Climate Anomaly Predictor")
st.write("Supervised regression model forecasting anomalies for climate resilience.")

df, model, scaler, model_r2 = load_data_and_model()

# Visualize
st.subheader("Historical Trends")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['Year'], df['Annual_Anomaly'], label='Anomaly (°C)', color='orange')
ax.set_title('Global Temperature Anomalies (1880-2025)')
ax.legend()
st.pyplot(fig)

# Predict
st.subheader("Forecast")
year = st.slider("Year:", 2026, 2050, 2030)
if st.button("Predict"):
    future = np.array([[year, 280 + (year - 1880) * 0.8, max(0, (year - 1950) * 0.5)]])
    future_scaled = scaler.transform(future)
    pred = model.predict(future_scaled)[0]
    st.success(f"{year} Anomaly: **{pred:.2f}°C**")
    st.metric("R² Score", model_r2)

with st.sidebar:
    st.header("Ethics")
    st.write("**Bias:** Northern skew; mitigate with diverse data.")
    st.write("**Fairness:** Transparent predictions for global equity.")

st.caption("Week 2 Assignment | Open-source.")
