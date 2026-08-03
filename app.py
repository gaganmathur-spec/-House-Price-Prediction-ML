import streamlit as st
from src.predict import predict_price

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction")
st.write("Predict house prices using a Linear Regression model.")

bedrooms = st.number_input("Bedrooms", min_value=0, value=3)
bathrooms = st.number_input("Bathrooms", min_value=0.0, value=2.0)
sqft_living = st.number_input("Living Area (sqft)", min_value=100, value=2000)
sqft_lot = st.number_input("Lot Area (sqft)", min_value=100, value=5000)
floors = st.number_input("Floors", min_value=1.0, value=1.0)
waterfront = st.selectbox("Waterfront", [0, 1])
view = st.slider("View", 0, 4, 0)
condition = st.slider("Condition", 1, 5, 3)
grade = st.slider("Grade", 1, 13, 7)
sqft_above = st.number_input("Sqft Above", min_value=100, value=1500)
sqft_basement = st.number_input("Sqft Basement", min_value=0, value=500)
yr_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2000)
yr_renovated = st.number_input("Year Renovated", min_value=0, value=0)
zipcode = st.number_input("Zipcode", value=98001)
lat = st.number_input("Latitude", value=47.5, format="%.6f")
long = st.number_input("Longitude", value=-122.2, format="%.6f")
sqft_living15 = st.number_input("Living Area 15", min_value=100, value=1800)
sqft_lot15 = st.number_input("Lot Area 15", min_value=100, value=5000)

if st.button("Predict House Price"):
    prediction = predict_price(
        bedrooms,
        bathrooms,
        sqft_living,
        sqft_lot,
        floors,
        waterfront,
        view,
        condition,
        grade,
        sqft_above,
        sqft_basement,
        yr_built,
        yr_renovated,
        zipcode,
        lat,
        long,
        sqft_living15,
        sqft_lot15
    )

    st.success(f"Predicted House Price: ${prediction:,.2f}")