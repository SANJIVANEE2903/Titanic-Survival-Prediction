import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved model
model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic Survival Prediction")

# Input features
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.slider("Fare Paid", 0.0, 600.0, 50.0)

# Gender
sex = st.radio("Gender", ["Male", "Female"])
sex_male = 1 if sex == "Male" else 0

# Embarked
embarked = st.selectbox("Port of Embarkation", ["S", "Q", "C"])
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Create feature array
sample = np.array([[pclass, age, sibsp, parch, fare, sex_male, embarked_Q, embarked_S]])

if st.button("Predict Survival"):
    pred = model.predict(sample)[0]
    result = "🎉 Survived!" if pred == 1 else "💀 Did Not Survive"
    st.subheader(f"Prediction: {result}")
