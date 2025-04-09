import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Split data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train the model (you can later save & load it instead)
model = DecisionTreeClassifier()
model.fit(X, y)

# Streamlit UI
st.title("🩺 Diabetes Predictor")
st.markdown("Enter patient health data below:")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Predict button
if st.button("Predict"):
    input_data = [[pregnancies, glucose, blood_pressure, skin_thickness,
                   insulin, bmi, dpf, age]]
    
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ The model predicts: **Diabetic**")
    else:
        st.success("✅ The model predicts: **Not Diabetic**")
