import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("circle_classifier.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Circle Classifier", layout="centered")
st.title("Synthetic Circle Classifier")
st.write("Enter the features below to predict the class.")

# Input fields
x1 = st.number_input("Enter value for x1:", value=0.0, format="%.2f")
x2 = st.number_input("Enter value for x2:", value=0.0, format="%.2f")

# Prediction
if st.button("Predict"):
    features = np.array([[x1, x2]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted Class: {prediction}")

