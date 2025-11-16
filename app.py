import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("diabetes_model.pkl", "rb"))

st.title("🩺 Diabetes Prediction App")
st.write("Enter details below to check if the person is at risk of diabetes.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.2f")
age = st.number_input("Age", min_value=0, max_value=120, value=25)

# Prediction
if st.button("Check Result"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("⚠️ High chance of having Diabetes.")
    else:
        st.success("✅ Low chance of Diabetes.")
