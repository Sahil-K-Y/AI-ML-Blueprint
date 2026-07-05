import numpy as np
import streamlit as st

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

# =========================
# Load Dataset
# =========================

X, y = fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Model Pipeline
# =========================

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso(alpha=0.001))
])

pipeline.fit(X_train, y_train)

# =========================
# Model Performance
# =========================

y_pred = pipeline.predict(X_test)
score = r2_score(y_test, y_pred)

# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠"
)

st.title("🏠 California House Price Prediction")
st.write("Enter house details and predict the house price.")

st.sidebar.header("Model Information")
st.sidebar.write("Model : Lasso Regression")
st.sidebar.write("Dataset : California Housing")
st.sidebar.write(f"R² Score : {round(score,4)}")

# =========================
# Inputs
# =========================

medinc = st.number_input(
    "Median Income",
    min_value=0.0,
    value=3.0
)

houseage = st.number_input(
    "House Age",
    min_value=0.0,
    value=20.0
)

averooms = st.number_input(
    "Average Rooms",
    min_value=0.0,
    value=5.0
)

avebedrms = st.number_input(
    "Average Bedrooms",
    min_value=0.0,
    value=1.0
)

population = st.number_input(
    "Population",
    min_value=0.0,
    value=1000.0
)

aveoccup = st.number_input(
    "Average Occupancy",
    min_value=0.0,
    value=3.0
)

latitude = st.number_input(
    "Latitude",
    value=34.0
)

longitude = st.number_input(
    "Longitude",
    value=-118.0
)

# =========================
# Prediction
# =========================

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

    st.metric(
        "Predicted House Price",
        f"${prediction[0] * 100000:,.0f}"
    )

    st.success("Prediction Completed Successfully ✅")

# =========================
# Dataset Preview
# =========================

if st.checkbox("Show Dataset Information"):

    housing = fetch_california_housing(as_frame=True)

    st.subheader("Dataset Preview")

    st.dataframe(
        housing.frame.head()
    )