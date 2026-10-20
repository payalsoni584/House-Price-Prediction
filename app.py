
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("house_price_model.pkl")

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction")
st.write("Enter your house details below.")

area = st.number_input("Area (sq. ft.)", min_value=1, value=5000)
bedrooms = st.number_input("Bedrooms", min_value=1, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, value=2)
stories = st.number_input("Stories", min_value=1, value=2)
parking = st.number_input("Parking spaces", min_value=0, value=1)

if st.button("Predict Price"):
    house = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "parking": [parking]
    })

    price = model.predict(house)[0]
    st.success(f"Predicted House Price: {price:,.2f}")
    
