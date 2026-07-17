import streamlit as st
import joblib 
import numpy as np
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Styling
st.markdown("""
<style>
    /* Styling headers */
    .main-title {
        font-family: 'Inter', sans-serif;
        color: #1e3a8a;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        color: #4b5563;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 25px;
    }
    /* Card design */
    .input-card {
        background-color: #f3f4f6;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #2563eb;
    }
    .prediction-survived {
        background-color: #dcfce7;
        color: #14532d;
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #16a34a;
        font-size: 1.25rem;
        font-weight: bold;
        text-align: center;
    }
    .prediction-died {
        background-color: #fee2e2;
        color: #7f1d1d;
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #dc2626;
        font-size: 1.25rem;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load saved artifacts
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("SVM.pkl")
        scaler = joblib.load("scaler.pkl")
        columns = joblib.load("columns.pkl")
        return model, scaler, columns
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

model, scaler, columns = load_artifacts()

# App Header
st.markdown("<h1 class='main-title'>🚢 Titanic Survival Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Predict passenger survival chances using a trained Support Vector Machine (SVM) Classifier</p>", unsafe_allow_html=True)

if model is not None:
    # Sidebar Info
    st.sidebar.image("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=800&q=80", use_container_width=True)
    st.sidebar.markdown("### 📊 About the Model")
    st.sidebar.info(
        "This predictor uses a Support Vector Classifier (RBF Kernel) trained on the historical Titanic dataset. "
        "It achieves approximately **81% accuracy** on the test set."
    )
    st.sidebar.markdown("### 🛠️ Preprocessing Steps")
    st.sidebar.warning(
        "1. Standard Scaling of features using training moments.\n"
        "2. Categorical encoding for Sex & Embarked.\n"
        "3. Engineered Family size feature (Is_alone)."
    )

    st.markdown("### 📝 Enter Passenger Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pclass Selection
        pclass_label = st.selectbox(
            "Passenger Ticket Class (Pclass)",
            options=["1st Class", "2nd Class", "3rd Class"],
            help="1st = Upper; 2nd = Middle; 3rd = Lower"
        )
        pclass_map = {"1st Class": 1, "2nd Class": 2, "3rd Class": 3}
        pclass = pclass_map[pclass_label]

        # Sex Selection
        sex_label = st.selectbox("Gender (Sex)", options=["Female", "Male"])
        sex_map = {"Male": 0, "Female": 1}
        sex = sex_map[sex_label]

        # Age Selection
        age = st.slider("Passenger Age (Years)", min_value=1, max_value=100, value=28)

    with col2:
        # Fare Selection
        fare = st.slider("Ticket Fare Paid (£)", min_value=0.0, max_value=512.0, value=32.0, step=0.5,
                         help="Amount of money spent on the ticket.")

        # Embarked Selection
        embarked_label = st.selectbox(
            "Port of Embarkation",
            options=["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"]
        )
        embarked_map = {"Southampton (S)": 0, "Cherbourg (C)": 1, "Queenstown (Q)": 2}
        embarked = embarked_map[embarked_label]

        # Family Size Selection
        family_size = st.slider(
            "Family Members on Board",
            min_value=1,
            max_value=11,
            value=1,
            help="Total family members traveling together (including the passenger themselves)."
        )

    # Predict Button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Run Survival Prediction", type="primary", use_container_width=True):
        # Create input DataFrame to match model column expectations and order
        input_data = pd.DataFrame([[pclass, sex, age, fare, embarked, family_size]], columns=columns)
        
        # Scale the data using the loaded scaler (transform ONLY, DO NOT fit)
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 📊 Prediction Result")
        
        # Visualizing output nicely
        if prediction == 1:
            st.markdown(
                f"<div class='prediction-survived'>🎉 PASSENGER SURVIVED<br>"
                f"<span style='font-size: 0.9rem; font-weight: normal;'>"
                f"The model predicts this passenger would have survived the shipwreck.</span></div>",
                unsafe_allow_html=True
            )
            st.balloons()
        else:
            st.markdown(
                f"<div class='prediction-died'>☠️ PASSENGER DID NOT SURVIVE<br>"
                f"<span style='font-size: 0.9rem; font-weight: normal;'>"
                f"The model predicts this passenger would not have survived the shipwreck.</span></div>",
                unsafe_allow_html=True
            )
            
        # Displaying data table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Input Features Summary")
        summary_df = pd.DataFrame({
            "Feature Name": ["Pclass", "Sex (Female=1, Male=0)", "Age", "Fare", "Embarked Port Code", "Family Size (Is_alone)"],
            "Value": [pclass, sex, age, f"£{fare}", embarked, family_size]
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
else:
    st.error("Could not run application. Please ensure SVM.pkl, scaler.pkl, and columns.pkl are in the same directory.")
