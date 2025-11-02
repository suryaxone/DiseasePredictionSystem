import streamlit as st
import pandas as pd
import pickle
import os

st.title("🩺 Predict Diabetes & Heart Disease using Machine Learning")

menu = st.sidebar.selectbox("Choose Prediction Type", ["Diabetes", "Heart Disease"])

# Load model safely
def load_model(path):
    if os.path.exists(path):
        with open(path, "rb") as file:
            return pickle.load(file)
    else:
        st.error(f"Model file not found: {path}")
        return None

if menu == "Diabetes":
    model = load_model("models/diabetes_model.pkl")

    st.header("🔹 Diabetes Prediction")
    Pregnancies = st.number_input("Pregnancies", 0, 20)
    Glucose = st.number_input("Glucose", 0, 200)
    BloodPressure = st.number_input("BloodPressure", 0, 140)
    SkinThickness = st.number_input("SkinThickness", 0, 100)
    Insulin = st.number_input("Insulin", 0, 900)
    BMI = st.number_input("BMI", 0.0, 70.0)
    DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", 0.0, 2.5)
    Age = st.number_input("Age", 0, 120)

    if st.button("Predict Diabetes"):
        if model:
            result = model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])[0]
            st.success("✅ Likely Diabetic" if result == 1 else "❌ Not Diabetic")

elif menu == "Heart Disease":
    model = load_model("models/heart_model.pkl")

    st.header("❤️ Heart Disease Prediction")
    Age = st.number_input("Age", 0, 120)
    Sex = st.selectbox("Sex", [0, 1])
    Cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    Trestbps = st.number_input("Resting Blood Pressure", 80, 200)
    Chol = st.number_input("Cholesterol", 100, 400)
    Fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    Restecg = st.selectbox("Resting ECG", [0, 1, 2])
    Thalach = st.number_input("Max Heart Rate Achieved", 60, 220)
    Exang = st.selectbox("Exercise Induced Angina", [0, 1])
    Oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 10.0)
    Slope = st.selectbox("Slope of ST", [0, 1, 2])
    Ca = st.selectbox("Number of major vessels (0–4)", [0, 1, 2, 3, 4])
    Thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

    if st.button("Predict Heart Disease"):
        if model:
            result = model.predict([[Age, Sex, Cp, Trestbps, Chol, Fbs, Restecg, Thalach, Exang, Oldpeak, Slope, Ca, Thal]])[0]
            st.success("❤️ At Risk of Heart Disease" if result == 1 else "💚 Healthy Heart")
