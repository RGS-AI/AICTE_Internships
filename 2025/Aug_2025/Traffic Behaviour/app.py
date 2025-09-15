import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load required models and encoders
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
cat_dict = joblib.load("cat_dict.pkl")

st.set_page_config(page_title="Traffic & Vehicle Behavior")
st.title("Traffic & Vehicle Behavior Prediction App")
st.markdown("Predict **Traffic Density** based on vehicle and environment factors.")

# --- Sidebar inputs ---
st.sidebar.header("Input Parameters")

def get_user_input():
    city = st.sidebar.selectbox("City", cat_dict['City'])
    vehicle = st.sidebar.selectbox("Vehicle Type", cat_dict['Vehicle Type'])
    weather = st.sidebar.selectbox("Weather", cat_dict['Weather'])
    econ = st.sidebar.selectbox("Economic Condition", cat_dict['Economic Condition'])
    day = st.sidebar.selectbox("Day Of Week", cat_dict['Day Of Week'])
    hour = st.sidebar.slider("Hour Of Day", 0, 23, 8)
    speed = st.sidebar.slider("Speed (km/h)", 0, 200, 50)
    peak = st.sidebar.selectbox("Is Peak Hour", ['True', 'False'])
    event = st.sidebar.selectbox("Random Event Occurred", ['True', 'False'])
    energy = st.sidebar.number_input("Energy Consumption (kWh)", 0.0, 500.0, 100.0)

    peak_final = 1 if peak == 'True' else 0
    event_final = 1 if event == 'True' else 0

    data = {
        'City': cat_dict['City'].tolist().index(city),
        'Vehicle Type': cat_dict['Vehicle Type'].tolist().index(vehicle),
        'Weather': cat_dict['Weather'].tolist().index(weather),
        'Economic Condition': cat_dict['Economic Condition'].tolist().index(econ),
        'Day Of Week': cat_dict['Day Of Week'].tolist().index(day),
        'Hour Of Day': hour,
        'Speed': speed,
        'Is Peak Hour': peak_final,
        'Random Event Occurred': event_final,
        'Energy Consumption': energy
    }

    return pd.DataFrame([data])

input_df = get_user_input()

# --- Prediction button ---
if st.button("Predict Traffic Density"):
    # Optional: scale the input if needed
    # scaled_input = scaler.transform(input_df)
    prediction = model.predict(input_df)
    
    st.subheader("Prediction Result")
    results=['High', 'Low', 'Medium']
    
    st.success(f"**Predicted Traffic Density:** `{results[prediction[0]]}`")

