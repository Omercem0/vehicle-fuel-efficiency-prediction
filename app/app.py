"""
Streamlit web application for Vehicle Fuel Efficiency prediction.
Provides an interactive UI for users to input vehicle specs and get MPG estimates.
"""

import sys
import os

# Resolve project root so src package is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import numpy as np
from src.predict import predict_mpg, load_artifacts

# ---------- Page config ----------
st.set_page_config(
    page_title="Vehicle Fuel Efficiency Predictor",
    page_icon="⛽",
    layout="centered",
)

# ---------- Header ----------
st.title("⛽ Vehicle Fuel Efficiency Predictor")
st.markdown(
    "Enter the vehicle specifications below to estimate **Miles Per Gallon (MPG)**."
)
st.divider()

# ---------- Sidebar info ----------
try:
    _, _, metadata = load_artifacts()
    st.sidebar.header("Model Info")
    st.sidebar.write(f"**Model:** {metadata['best_model']}")
    st.sidebar.write(f"**Test MSE:** {metadata['mse']:.6f}")
    st.sidebar.write(f"**Test R²:** {metadata['r2']:.4f}")
except Exception:
    st.sidebar.warning("Model artifacts not found. Run `python src/train.py` first.")

# ---------- Input form ----------
col1, col2 = st.columns(2)

with col1:
    cylinders = st.selectbox("Cylinders", options=[3, 4, 5, 6, 8], index=1)
    displacement = st.number_input("Displacement (cu. inches)", min_value=50.0, max_value=500.0, value=150.0, step=1.0)
    horsepower = st.number_input("Horsepower", min_value=40.0, max_value=300.0, value=90.0, step=1.0)
    weight = st.number_input("Weight (lbs)", min_value=1500.0, max_value=6000.0, value=2800.0, step=10.0)

with col2:
    acceleration = st.number_input("Acceleration (0-60 mph, sec)", min_value=8.0, max_value=25.0, value=15.0, step=0.5)
    model_year = st.slider("Model Year", min_value=70, max_value=82, value=76)
    origin = st.selectbox("Origin", options=[1, 2, 3], format_func=lambda x: {1: "USA", 2: "Europe", 3: "Japan"}[x])

st.divider()

# ---------- Predict ----------
if st.button("🔮 Predict MPG", use_container_width=True, type="primary"):
    try:
        mpg = predict_mpg(
            cylinders=cylinders,
            displacement=displacement,
            horsepower=horsepower,
            weight=weight,
            acceleration=acceleration,
            model_year=model_year,
            origin=origin,
        )
        st.success(f"**Predicted Fuel Efficiency: {mpg:.2f} MPG**")

        # Visual gauge
        st.metric(label="Estimated MPG", value=f"{mpg:.1f}")
        if mpg >= 30:
            st.info("🌿 Great fuel efficiency!")
        elif mpg >= 20:
            st.info("👍 Average fuel efficiency.")
        else:
            st.warning("⚠️ Below average fuel efficiency.")

    except FileNotFoundError:
        st.error("Model not found. Please run `python src/train.py` first to train and save the model.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
