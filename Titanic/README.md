# 🚢 Titanic Survival Predictor

End-to-End Machine Learning project that predicts passenger survival on the Titanic using a trained Support Vector Machine (SVM) classifier, served through a polished Streamlit web application.

**Live App Features:**
- Interactive passenger detail form
- Instant survival prediction with visual feedback
- Model information & preprocessing transparency

---

## 📌 Project Overview

This project solves the classic binary classification problem: **Will a passenger survive the Titanic disaster?**

| Component              | Details                                          |
|------------------------|--------------------------------------------------|
| Problem Type           | Binary Classification (Survived / Did Not Survive) |
| Dataset                | Titanic - Machine Learning from Disaster (Kaggle) |
| Best Model             | Support Vector Classifier (RBF Kernel)           |
| Test Accuracy          | **~81.56%**                                      |
| Interface              | Streamlit Web App                                |
| Key Techniques         | Feature Engineering, Standard Scaling, Encoding  |

---

## 🚀 Features

### 1. Interactive Prediction
- Enter Passenger Class, Gender, Age, Fare, Port of Embarkation, and Family Size
- Get clear Survived / Did Not Survive prediction
- Visual feedback with balloons on survival prediction

### 2. Model Transparency
- Sidebar shows model type, approximate accuracy, and preprocessing steps
- Input feature summary table after every prediction

### 3. Clean & Modern UI
- Custom CSS styling
- Responsive two-column layout
- Clear visual distinction between survival outcomes

---

## 🛠️ Tech Stack

- **Language:** Python
- **ML Libraries:** scikit-learn (SVM, StandardScaler)
- **App Framework:** Streamlit
- **Data Handling:** pandas, numpy
- **Model Persistence:** joblib

---

## 📂 Project Structure

```text
Titanic/
├── app.py                 # Streamlit application
├── Titanic.ipynb          # Full EDA + model comparison notebook
├── SVM.pkl                # Trained SVM model
├── scaler.pkl             # Fitted StandardScaler
├── columns.pkl            # Expected feature column order
├── train.csv              # Titanic training dataset
└── README.md              # Project documentation
```

---

## 🧠 Machine Learning Pipeline

1. **Data Cleaning & Imputation**  
   - Fill missing `Age` (median) and `Embarked` (mode)

2. **Feature Engineering**  
   - Created `Is_alone` = SibSp + Parch + 1 (family size including passenger)

3. **Encoding**  
   - `Sex`: male → 0, female → 1  
   - `Embarked`: S → 0, C → 1, Q → 2

4. **Feature Selection**  
   - Final features: `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`, `Is_alone`

5. **Scaling**  
   - `StandardScaler` fitted on training data

6. **Model Comparison**  
   | Model              | Accuracy | F1 Score |
   |--------------------|----------|----------|
   | Logistic Regression| 0.7989   | 0.7188   |
   | KNN                | 0.7877   | 0.7077   |
   | Decision Tree      | 0.8212   | 0.7612   |
   | Naive Bayes        | 0.7877   | 0.7206   |
   | **SVM (RBF)**      | **0.8156** | 0.7179 |

   → Selected **SVM with RBF kernel** for the final production model.

7. **Inference**  
   - Model + scaler + column order saved and loaded in the Streamlit app

---

## ▶️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Sahil-K-Y/AI-ML-Blueprint.git
cd AI-ML-Blueprint/Titanic
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn joblib
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📊 Dataset Information

- **Source:** Titanic - Machine Learning from Disaster (Kaggle)
- **Target Variable:** `Survived` (0 = No, 1 = Yes)
- **Key Features Used:**
  - `Pclass` — Ticket class (1st, 2nd, 3rd)
  - `Sex` — Gender
  - `Age` — Passenger age
  - `Fare` — Ticket fare
  - `Embarked` — Port of embarkation (S/C/Q)
  - `Is_alone` — Engineered family size feature

---

## 💡 Key Insights from EDA

- Women and children had significantly higher survival rates
- 1st class passengers survived at much higher rates than 3rd class
- Family size showed a non-linear relationship with survival (being alone or having a large family both reduced chances)
- Fare and Pclass are strongly correlated with survival probability

---

## 👨‍💻 Author

**Sahil Kumar**  
BTech CSE (AI/ML) | DAV University, Jalandhar  

- GitHub: [Sahil-K-Y](https://github.com/Sahil-K-Y)
- Project Repo: [AI-ML-Blueprint](https://github.com/Sahil-K-Y/AI-ML-Blueprint)

---

## 📝 Notes

- Full exploratory data analysis and model comparison are available in `Titanic.ipynb`
- The Streamlit app uses cached model loading (`@st.cache_resource`) for performance
- All preprocessing (scaling + column order) is strictly matched to training for correct inference

---

⭐ If you find this project useful, feel free to star the repository!
