import joblib
import pandas as pd
import streamlit as st

# Load the trained pipeline
model = joblib.load("model.joblib")

st.title("Food Delivery Time Predictor")
st.write("Enter the order details to predict the delivery time in minutes.")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Delivery person age", 18, 60, 30)
    rating = st.slider("Delivery person rating", 1.0, 5.0, 4.5, 0.1)
    vehicle_condition = st.selectbox("Vehicle condition (0 = worst, 3 = best)", [0, 1, 2, 3], index=2)
    multiple = st.selectbox("Number of deliveries at once", [0, 1, 2, 3], index=0)
    distance = st.number_input("Distance (km)", 0.5, 30.0, 5.0)
    hour = st.slider("Order hour", 0, 23, 19)
    prep = st.number_input("Restaurant preparation time (min)", 1.0, 30.0, 10.0)
with col2:
    traffic = st.selectbox("Traffic density", ["Low", "Medium", "High", "Jam"])
    weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Fog", "Stormy", "Sandstorms", "Windy"])
    order_type = st.selectbox("Order type", ["Snack", "Meal", "Drinks", "Buffet"])
    vehicle = st.selectbox("Vehicle type", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
    festival = st.selectbox("Festival", ["No", "Yes"])
    city = st.selectbox("City type", ["Urban", "Metropolitian", "Semi-Urban"])
    weekend = st.checkbox("Weekend")

if st.button("Predict"):
    row = pd.DataFrame([{
        "Delivery_person_Age": age,
        "Delivery_person_Ratings": rating,
        "Vehicle_condition": vehicle_condition,
        "multiple_deliveries": multiple,
        "distance_km": distance,
        "order_hour": hour,
        "is_weekend": int(weekend),
        "is_rush_hour": int(hour in [8, 9, 12, 13, 19, 20, 21, 22]),
        "prep_time_min": prep,
        "Road_traffic_density": traffic,
        "Weatherconditions": weather,
        "Type_of_order": order_type,
        "Type_of_vehicle": vehicle,
        "Festival": festival,
        "City": city,
    }])
    pred = model.predict(row)[0]
    st.success(f"Estimated delivery time: {pred:.0f} minutes")
