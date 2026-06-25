import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Dataset
X, y = fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso(alpha=0.001))
])

pipeline.fit(X_train, y_train)

# Streamlit UI
st.title("🏠 House Price Prediction")
st.write("California Housing Price Prediction App")

medinc = st.number_input("Median Income", value=3.0)
houseage = st.number_input("House Age", value=20.0)
averooms = st.number_input("Average Rooms", value=5.0)
avebedrms = st.number_input("Average Bedrooms", value=1.0)
population = st.number_input("Population", value=1000.0)
aveoccup = st.number_input("Average Occupancy", value=3.0)
latitude = st.number_input("Latitude", value=34.0)
longitude = st.number_input("Longitude", value=-118.0)

if st.button("Predict Price"):

    features = np.array([[
        medinc,
        houseage,
        averooms,
        avebedrms,
        population,
        aveoccup,
        latitude,
        longitude
    ]])

    prediction = pipeline.predict(features)

    st.success(
        f"Predicted House Price: ${prediction[0]*100000:,.0f}"
    )