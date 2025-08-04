
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("Farm_Irrigation_System.pkl")

st.title("🌾 Smart Irrigation System")

# Create sliders for 20 sensors
st.header("Sensor Input")
sensors = []
for i in range(20):
    value = st.slider(f"Sensor {i}", min_value=0.0, max_value=15.0, step=0.1, value=5.0)
    sensors.append(value)

# Predict button
if st.button("Predict Parcel Status"):
    input_data = np.array(sensors).reshape(1, -1)
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Results")
    for i, status in enumerate(prediction):
        if status == 1:
            st.success(f"✅ Parcel {i} needs irrigation.")
        else:
            st.info(f"💧 Parcel {i} does not need irrigation.")
