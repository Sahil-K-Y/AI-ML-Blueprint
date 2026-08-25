import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ---------------------------------------------------------
# Page Configuration & Modern Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 5px;
    }
    
    /* Risk Badges */
    .risk-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid #EF4444;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .risk-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid #F59E0B;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .risk-low {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    /* Header Banner */
    .hero-banner {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        padding: 24px;
        border-radius: 14px;
        color: white;
        margin-bottom: 25px;
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
    }
    .hero-banner p {
        margin: 5px 0 0 0;
        opacity: 0.9;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model Artifact
# ---------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "churn_model.pkl")

@st.cache_resource
def load_churn_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            
            # Patch OneHotEncoder categories_ to numpy string arrays to prevent Python 3.14 / sklearn 1.6+ xp.isnan TypeError
            if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                prep = model.named_steps['preprocessor']
                if hasattr(prep, 'named_transformers_') and 'cat' in prep.named_transformers_:
                    cat_enc = prep.named_transformers_['cat']
                    if hasattr(cat_enc, 'categories_'):
                        cat_enc.categories_ = [np.array(c, dtype=str) for c in cat_enc.categories_]
            return model
        except Exception as e:
            st.error(f"Error loading model pickle: {e}")
            return None
    return None

model = load_churn_model()

# ---------------------------------------------------------
# Sidebar Navigation & Info
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4149/4149678.png", width=70)
    st.title("Navigation")
    app_mode = st.radio(
        "Select Mode",
        ["Single Customer Prediction", "Batch Prediction (CSV)", "Model Analytics & Specs"]
    )
    
    st.divider()
    st.markdown("### 📊 Dataset Overview")
    st.caption("Target: Telco Customer Churn")
    st.caption("Model: Tuned XGBoost Classifier (churn prediction.ipynb)")
    if model is not None:
        st.success("✅ Model Status: Loaded (churn_model.pkl)")
    else:
        st.warning("⚠️ Model File Missing")
        st.info("💡 Please run your Jupyter notebook `churn prediction.ipynb` to train and save `churn_model.pkl`.")

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>🔮 Customer Churn Intelligence Platform</h1>
    <p>Predict customer churn risk, analyze key drivers, and take proactive retention actions.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODE 1: Single Customer Prediction
# ---------------------------------------------------------
if app_mode == "Single Customer Prediction":
    st.subheader("📋 Enter Customer Profile & Subscription Details")
    
    with st.form("single_predict_form"):
        col1, col2, col3 = st.columns(3)
        
        # Column 1: Demographics
        with col1:
            st.markdown("#### 👤 Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Has Partner?", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12, help="Number of months customer has stayed with company")

        # Column 2: Telecom Services
        with col2:
            st.markdown("#### 📞 Services Subscribed")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        # Column 3: Billing & Contract
        with col3:
            st.markdown("#### 💳 Billing & Contract")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.5, step=1.0)
            
            # Auto-calculate suggested total charges or allow override
            suggested_total = round(float(tenure * monthly_charges), 2)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=max(18.0, suggested_total), step=10.0)

        submit_btn = st.form_submit_button("⚡ Predict Churn Risk", use_container_width=True)

    if submit_btn:
        if model is None:
            st.error("Model is not loaded. Please train or check churn_model.pkl.")
        else:
            # Build DataFrame with exact feature structure
            input_dict = {
                'gender': gender,
                'SeniorCitizen': str(senior_citizen),
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            input_df = pd.DataFrame([input_dict])
            
            # Predict
            prob = float(model.predict_proba(input_df)[0][1])
            pred_class = int(prob >= 0.5)
            
            st.divider()
            st.markdown("### 🎯 Prediction Results")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                # Risk Badge & Score Card
                prob_pct = round(prob * 100, 1)
                
                if prob >= 0.60:
                    badge_html = '<div class="risk-high">🚨 HIGH CHURN RISK</div>'
                elif prob >= 0.35:
                    badge_html = '<div class="risk-medium">⚠️ MEDIUM CHURN RISK</div>'
                else:
                    badge_html = '<div class="risk-low">✅ LOW CHURN RISK</div>'
                    
                st.markdown(badge_html, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.metric("Churn Probability", f"{prob_pct}%")
                st.progress(prob)
                
            with res_col2:
                st.markdown("#### 💡 Retention Insights & Recommended Actions")
                
                recommendations = []
                if contract == "Month-to-month":
                    recommendations.append("📌 **Contract Alert:** Customer is on a Month-to-Month contract. Offer a discounted 1-year or 2-year contract.")
                if payment_method == "Electronic check":
                    recommendations.append("💳 **Payment Method:** Electronic Check users have higher churn rates. Encourage switching to Auto-Debit (Credit Card / Bank Transfer) with a $5 bill credit.")
                if tech_support == "No" and internet_service != "No":
                    recommendations.append("🛠️ **Support Gap:** Tech Support is not enabled. Offer 3 months of free Premium Tech Support.")
                if tenure <= 12:
                    recommendations.append("🌱 **Early Journey:** Customer is in their first year of tenure. Schedule a proactive check-in call.")
                if not recommendations:
                    recommendations.append("🎉 **Low Risk:** Customer shows strong engagement signals. Recommend upselling premium streaming packages.")
                    
                for rec in recommendations:
                    st.write(rec)

# ---------------------------------------------------------
# MODE 2: Batch Prediction (CSV)
# ---------------------------------------------------------
elif app_mode == "Batch Prediction (CSV)":
    st.subheader("📂 Upload Customer Batch File (CSV)")
    st.caption("Ensure file contains the required columns: tenure, MonthlyCharges, TotalCharges, Contract, InternetService, etc.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("### Raw Data Preview", batch_df.head(5))
        
        if st.button("🚀 Process Batch Predictions", use_container_width=True):
            if model is None:
                st.error("Model is not loaded.")
            else:
                with st.spinner("Calculating churn probabilities for all records..."):
                    processed_df = batch_df.copy()
                    
                    # Ensure SeniorCitizen string formatting if present
                    if 'SeniorCitizen' in processed_df.columns:
                        processed_df['SeniorCitizen'] = processed_df['SeniorCitizen'].astype(str)
                        
                    # Handle TotalCharges numeric conversion
                    if 'TotalCharges' in processed_df.columns:
                        processed_df['TotalCharges'] = pd.to_numeric(
                            processed_df['TotalCharges'].astype(str).str.strip(), errors='coerce'
                        ).fillna(0)
                        
                    # Remove customerID if present for model prediction
                    eval_df = processed_df.drop(columns=['customerID', 'Churn'], errors='ignore')
                    
                    probs = model.predict_proba(eval_df)[:, 1]
                    batch_df['Churn_Probability'] = np.round(probs, 4)
                    batch_df['Churn_Prediction'] = (probs >= 0.5).astype(int)
                    batch_df['Risk_Level'] = pd.cut(
                        probs, bins=[-0.01, 0.35, 0.60, 1.0], labels=['Low', 'Medium', 'High']
                    )
                    
                    st.divider()
                    st.markdown("### 📊 Batch Prediction Summary")
                    
                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    mcol1.metric("Total Customers", len(batch_df))
                    mcol2.metric("High Risk Customers", int((batch_df['Risk_Level'] == 'High').sum()))
                    mcol3.metric("Medium Risk", int((batch_df['Risk_Level'] == 'Medium').sum()))
                    mcol4.metric("Average Churn Prob", f"{batch_df['Churn_Probability'].mean()*100:.1f}%")
                    
                    st.subheader("Predicted Results")
                    st.dataframe(batch_df[['customerID', 'Churn_Probability', 'Risk_Level', 'Contract', 'MonthlyCharges', 'tenure'] if 'customerID' in batch_df.columns else batch_df.columns], use_container_width=True)
                    
                    # Download CSV
                    csv_data = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predicted Churn CSV",
                        data=csv_data,
                        file_name="churn_predictions_output.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# ---------------------------------------------------------
# MODE 3: Model Analytics & Specs
# ---------------------------------------------------------
elif app_mode == "Model Analytics & Specs":
    st.subheader("⚙️ Machine Learning Pipeline Architecture")
    
    st.markdown("""
    ### 🏗️ Pipeline Components
<<<<<<< HEAD
    1. **Data Cleaner**: Standardizes numerical fields (`TotalCharges`), handles missing values.
    2. **ColumnTransformer**:
       - `StandardScaler()` on numerical columns (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`).
       - `OneHotEncoder(drop='first', handle_unknown='ignore')` on categorical columns.
    3. **SMOTE**: Oversamples minority class (Churn=1) to address class imbalance.
    4. **Classifier**: `XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)`.
=======
    1. **ColumnTransformer (Preprocessor)**:
       - `StandardScaler()` on numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`).
       - `OneHotEncoder(handle_unknown='ignore')` on categorical columns (`gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`).
    2. **Classifier**: Tuned `XGBClassifier` via `RandomizedSearchCV` (best params: `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`).
>>>>>>> 7a994c5 (chore: clean roadmap structure, sync day ranges with master curriculum, remove tracking and merge notes)
    """)
    
    if model is not None:
        st.divider()
        st.markdown("### 📋 Model Object Inspector")
        clf_step = model.named_steps.get('model') or model.named_steps.get('classifier')
        st.json({
            "pipeline_steps": [step[0] for step in model.steps],
            "classifier_params": clf_step.get_params() if clf_step else "N/A"
        })
