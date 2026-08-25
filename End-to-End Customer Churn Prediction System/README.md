<<<<<<< HEAD
# 🔮 Customer Churn Intelligence Platform

End-to-End Machine Learning system that predicts customer churn risk for a telecom company and provides actionable retention recommendations.

**Live App Features:**
- Single Customer Prediction with risk score + recommendations
- Batch Prediction via CSV upload
- Model Analytics & Pipeline Specs
=======
# 🔮 End-to-End Customer Churn Prediction System

An end-to-end Machine Learning solution for predicting telecom customer churn, identifying high-risk customer segments, and providing automated business retention recommendations. Built with **XGBoost**, **Imbalanced-Learn (SMOTE)**, **Scikit-Learn**, and **Streamlit**.
>>>>>>> 7a994c5 (chore: clean roadmap structure, sync day ranges with master curriculum, remove tracking and merge notes)

---

## 📌 Project Overview

<<<<<<< HEAD
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
=======
Customer churn is one of the most critical metrics for subscription-based telecom businesses. Acquiring new customers can cost 5x more than retaining existing ones. 

This project provides an end-to-end pipeline:
1. **Exploratory Data Analysis (EDA) & Model Building** in Jupyter Notebook (`churn prediction.ipynb`).
2. **Automated Preprocessing & Imbalance Handling** using `ColumnTransformer` + `SMOTE`.
3. **Hyperparameter Tuning** via `RandomizedSearchCV` with XGBoost.
4. **Interactive Web Application** built with `Streamlit` (`app.py`) for single-customer risk scoring & batch predictions.

---

## 🚀 Key Features

* **📋 Single Customer Prediction**: Interactive form to input customer demographics, subscribed telecom services, contract types, and billing metrics to get real-time churn probability and actionable retention advice.
* **📂 Batch CSV Processing**: Upload customer data CSV files, process bulk churn probabilities, assign risk tiers (**Low**, **Medium**, **High**), and download predicted results.
* **⚙️ Pipeline Inspector**: View complete machine learning architecture and inspect hyperparameter details directly from the web interface.
* **💡 Business Action Recommendations**: Automated rule-based intervention advice tailored to customer risk profiles (e.g., offering discounted 1/2 year contracts for month-to-month users).

---

## 🛠️ Tech Stack & Libraries

- **Language**: Python 3.10+
- **Data Manipulation & Analysis**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost, Imbalanced-Learn (`imbalanced-learn`)
- **Web Interface**: Streamlit

---

## 📂 Directory Structure

```text
End-to-End Customer Churn Prediction System/
│
├── churn prediction.ipynb            # Jupyter Notebook: EDA, Preprocessing & Model Tuning
├── app.py                             # Streamlit Web Application Interface
├── churn_model.pkl                    # Trained Model Pipeline Artifact
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset (Telco Customer Churn)
├── requirements.txt                   # Required Python packages
└── README.md                          # Documentation
>>>>>>> 7a994c5 (chore: clean roadmap structure, sync day ranges with master curriculum, remove tracking and merge notes)
```

---

<<<<<<< HEAD
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
=======
## 🔬 Machine Learning Pipeline Details

### 1. Data Cleaning & Feature Engineering
- Handled missing/empty values in `TotalCharges`.
- Categorical features encoded with `OneHotEncoder(handle_unknown='ignore')`.
- Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) scaled with `StandardScaler()`.

### 2. Class Imbalance Handling
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** inside an `ImbPipeline` to prevent data leakage during cross-validation.

### 3. Model Architecture & Tuning
- **Classifier**: `XGBClassifier`
- **Tuning**: `RandomizedSearchCV` (CV=5)
- **Best Hyperparameters**:
  - `n_estimators`: 200
  - `learning_rate`: 0.05
  - `max_depth`: 5

---

## 💻 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/AI-ML-Blueprint.git
cd "AI-ML-Blueprint/End-to-End Customer Churn Prediction System"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Web App
>>>>>>> 7a994c5 (chore: clean roadmap structure, sync day ranges with master curriculum, remove tracking and merge notes)
```bash
streamlit run app.py
```

<<<<<<< HEAD
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
=======
Open your browser at `http://localhost:8501`.

---

## 📈 Model Performance & Results

- **Accuracy**: ~78%
- **ROC-AUC Score**: ~0.84
- **PR-AUC Score**: ~0.65

---

## 🤝 Contributing & License
Contributions and feedback are welcome! Feel free to open issues or pull requests.
License: MIT
>>>>>>> 7a994c5 (chore: clean roadmap structure, sync day ranges with master curriculum, remove tracking and merge notes)
