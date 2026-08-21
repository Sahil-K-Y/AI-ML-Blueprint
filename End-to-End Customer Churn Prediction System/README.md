# 🔮 Customer Churn Intelligence Platform

End-to-End Machine Learning system that predicts customer churn risk for a telecom company and provides actionable retention recommendations.

**Live App Features:**
- Single Customer Prediction with risk score + recommendations
- Batch Prediction via CSV upload
- Model Analytics & Pipeline Specs

---

## 📌 Project Overview

This project solves a real business problem: **identifying customers who are likely to leave** so that the company can take proactive retention actions.

| Component              | Details                                      |
|------------------------|----------------------------------------------|
| Problem Type           | Binary Classification (Churn / No Churn)     |
| Dataset                | Telco Customer Churn (IBM Sample)            |
| Model                  | Tuned XGBoost Classifier                     |
| Interface              | Streamlit Web App                            |
| Key Techniques         | SMOTE, ColumnTransformer, Pipeline           |

---

## 🚀 Features

### 1. Single Customer Prediction
- Enter customer demographics, services, and billing details
- Get **Churn Probability** + Risk Level (Low / Medium / High)
- Receive **personalized retention recommendations** based on risk drivers

### 2. Batch Prediction (CSV)
- Upload a CSV of multiple customers
- Get churn probability, prediction, and risk level for every row
- Download results as CSV

### 3. Model Analytics
- View full ML pipeline architecture
- Inspect model parameters and preprocessing steps

---

## 🛠️ Tech Stack

- **Language:** Python
- **ML Libraries:** scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **App Framework:** Streamlit
- **Data Handling:** pandas, numpy
- **Model Persistence:** pickle / joblib

---

## 📂 Project Structure

```text
End-to-End Customer Churn Prediction System/
├── app.py                          # Streamlit application
├── churn prediction.ipynb          # Full training & EDA notebook
├── churn_model.pkl                 # Trained pipeline (preprocessor + model)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset
└── README.md                       # Project documentation
```

---

## 🧠 Machine Learning Pipeline

1. **Data Cleaning**  
   - Handle missing / non-numeric `TotalCharges`  
   - Standardize data types

2. **Preprocessing (ColumnTransformer)**  
   - Numerical features → `StandardScaler`  
   - Categorical features → `OneHotEncoder(drop='first')`

3. **Class Imbalance Handling**  
   - SMOTE oversampling on the minority class (Churn = Yes)

4. **Model**  
   - `XGBClassifier` (tuned)

5. **Inference**  
   - Full pipeline is saved as `churn_model.pkl` and used directly in the Streamlit app

---

## ▶️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Sahil-K-Y/AI-ML-Blueprint.git
cd "AI-ML-Blueprint/End-to-End Customer Churn Prediction System"
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn xgboost imbalanced-learn
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📊 Dataset Information

- **Source:** Telco Customer Churn Dataset
- **Target Variable:** `Churn` (Yes / No)
- **Key Features:**
  - Demographics: Gender, SeniorCitizen, Partner, Dependents
  - Services: Phone, Internet, Online Security, Tech Support, Streaming, etc.
  - Account: Tenure, Contract, Payment Method, MonthlyCharges, TotalCharges

---

## 💡 Business Value

- Identifies high-risk customers early
- Provides clear, actionable retention suggestions
- Supports both individual and bulk decision making
- Can be extended with real-time scoring API or dashboard integration

---

## 👨‍💻 Author

**Sahil Kumar**  
BTech CSE (AI/ML) | DAV University, Jalandhar  

- GitHub: [Sahil-K-Y](https://github.com/Sahil-K-Y)
- Project Repo: [AI-ML-Blueprint](https://github.com/Sahil-K-Y/AI-ML-Blueprint)

---

## 📝 Notes

- Model was trained and evaluated in `churn prediction.ipynb`
- The saved artifact (`churn_model.pkl`) contains the complete preprocessing + model pipeline
- App includes custom CSS for a modern dark-themed UI

---

⭐ If you find this project useful, feel free to star the repository!
