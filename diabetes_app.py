import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
from datetime import datetime
import csv
import os

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Split data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# This function logs the input data and prediction to a CSV file with a timestamp

def log_prediction(input_data, prediction):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile("predictions_log.csv")
    
    with open("predictions_log.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow([
                "Timestamp", "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
                "Insulin", "BMI", "Diabetes Pedigree", "Age", "Prediction"
            ])
        
        writer.writerow([timestamp] + input_data[0] + [prediction])


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
    
    log_prediction(input_data, prediction)

    if prediction == 1:
        st.error("⚠️ The model predicts: **Diabetic**")
    else:
        st.success("✅ The model predicts: **Not Diabetic**")
