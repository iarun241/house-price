import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("🏡 House Price Prediction App")

st.write("Enter the details of the house to predict the price:")

# Input fields
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2)
sqft = st.number_input("Square Footage", min_value=500, max_value=10000, value=1500)

if st.button("Predict Price"):
    input_data = [[bedrooms, bathrooms, sqft]]
    prediction = model.predict(input_data)
    st.success(f"💰 Estimated House Price: ${prediction[0]:,.2f}")

