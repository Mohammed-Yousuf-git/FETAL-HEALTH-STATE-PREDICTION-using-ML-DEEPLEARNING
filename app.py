import streamlit as st
import joblib
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("/Users/usufahmed/Desktop/CARDIO/ctg_multitask_model.keras")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define class A-J descriptions
class_descriptions = {
    "A": "Normal CTG with no signs of stress.",
    "B": "Slight deviation from normal but not concerning.",
    "C": "Some mild decelerations present.",
    "D": "Increased baseline variability, needs monitoring.",
    "E": "Repeated decelerations, possible mild distress.",
    "F": "Signs of potential hypoxia or acidosis.",
    "G": "Severe decelerations and reduced variability.",
    "H": "Abnormal CTG indicating fetal distress.",
    "I": "Very poor signal, possibly pathological.",
    "J": "Critically abnormal, immediate action required."
}

# 8 Feature Autofill Sample Values
default_values = {
    "ASTV": 45.0,
    "MSTV": 1.8,
    "ALTV": 18.0,
    "SUSP": 0.12,
    "E": 10.0,
    "LD": -2.0,
    "FS": 5.0,
    "Mean": 136.0
}

st.set_page_config(page_title="Fetal State Prediction App", layout="centered")
st.title("👶 CTG-Based Fetal State Predictor")
st.markdown("Use cardiotocographic features to predict fetal state and classification (A–J).")

# Autofill toggle
autofill = st.checkbox("🔁 Autofill with sample values")

# Input form
with st.form("prediction_form"):
    val_ASTV = st.number_input("ASTV - % Abnormal Short-Term Variability", 0.0, 100.0, value=default_values["ASTV"] if autofill else 0.0)
    val_MSTV = st.number_input("MSTV - Mean Short-Term Variability (ms)", 0.0, 10.0, value=default_values["MSTV"] if autofill else 0.0)
    val_ALTV = st.number_input("ALTV - % Abnormal Long-Term Variability", 0.0, 100.0, value=default_values["ALTV"] if autofill else 0.0)
    val_SUSP = st.number_input("SUSP - Suspicious Pattern Score", 0.0, 1.0, value=default_values["SUSP"] if autofill else 0.0)
    val_E = st.number_input("E - Histogram Mode Width", 0.0, 100.0, value=default_values["E"] if autofill else 0.0)
    val_LD = st.number_input("LD - Histogram Tendency", -10.0, 10.0, value=default_values["LD"] if autofill else 0.0)
    val_FS = st.number_input("FS - Histogram Acceleration Count", 0.0, 50.0, value=default_values["FS"] if autofill else 0.0)
    val_Mean = st.number_input("Mean - Histogram Mean Heart Rate", 60.0, 200.0, value=default_values["Mean"] if autofill else 0.0)

    submit = st.form_submit_button("🔍 Predict")

# On predict
if submit:
    input_data = np.array([[val_ASTV, val_MSTV, val_ALTV, val_SUSP, val_E, val_LD, val_FS, val_Mean]])

    try:
        # Create a 40-feature array (fill other features with zeros)
        full_input = np.zeros((1, 40))
        important_indices = [16, 17, 18, 39, 34, 37, 38, 26]
        for i, idx in enumerate(important_indices):
            full_input[0][idx] = input_data[0][i]

        # Scale input and predict
        X_scaled = scaler.transform(full_input)
        predictions = model.predict(X_scaled)
        class_pred = np.argmax(predictions[0], axis=1)[0]
        nsp_pred = np.argmax(predictions[1], axis=1)[0]

        class_label = chr(ord("A") + class_pred)

        st.success(f"📊 **Predicted CTG Class**: {class_label} — {class_descriptions[class_label]}")
        st.info(f"🧠 **Fetal State Prediction (NSP)**: {['Normal', 'Suspect', 'Pathologic'][nsp_pred]}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
