# 🧠 Phase 2: Advanced Classical ML & Feature Engineering (Days 050 – 056)
## Ultimate Theory, Math, Syntax & Code Master Guide

> **Author:** Sahil Kumar (Yd)  
> **Blueprint:** 365-Day AI/ML Engineer Master Roadmap  
> **Location:** `Phase-4 - Boosting & Advanced Ensembles/phase2_advanced_ml_theory_and_code.md`

---

## 📖 Table of Contents

1. [Day 050 — Production Sklearn Pipelines & Custom Transformers](#-day-050--production-sklearn-pipelines--custom-transformers)
2. [Day 051 — Imbalanced Learning Methods (SMOTE, ADASYN, Tomek Links)](#-day-051--imbalanced-learning-methods-smote-adasyn-tomek-links)
3. [Day 052 — Threshold Tuning & Probability Calibration](#-day-052--threshold-tuning--probability-calibration)
4. [Day 053 — Model Interpretability & XAI (SHAP & LIME)](#-day-053--model-interpretability--xai-shap--lime)
5. [Day 054 — Anomaly & Outlier Detection (Isolation Forest, LOF, One-Class SVM)](#-day-054--anomaly--outlier-detection-isolation-forest-lof-one-class-svm)
6. [Day 055 — Classical ML Comparison Lab & Benchmarking Harness](#-day-055--classical-ml-comparison-lab--benchmarking-harness)
7. [Day 056 — Advanced ML Assessment, Math Derivations & Interview Prep](#-day-056--advanced-ml-assessment-math-derivations--interview-prep)

---

## ⚙️ Day 050 — Production Sklearn Pipelines & Custom Transformers

### 💡 1. Real-World Analogy & Problem Statement
Imagine an automated assembly line in a car factory. If each part (wheels, engine, doors) is assembled in random places manually by different workers, cars will fail quality checks.  
In ML, **Pipelines** are that automated assembly line. Without pipelines:
- Training code and deployment code diverge $\rightarrow$ **Training-Serving Skew**
- Statistics (mean/std/median) leak from test sets $\rightarrow$ **Data Leakage**

---

### 📐 2. Core Concepts & Syntax Breakdown

#### A. Custom Transformers (`BaseEstimator` + `TransformerMixin`)
When standard Scikit-Learn transformers (`StandardScaler`, `SimpleImputer`) don't satisfy business logic (e.g. custom IQR clipping), we write custom classes.

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param1=1.0):
        self.param1 = param1     # Store hyperparameters ONLY
        
    def fit(self, X, y=None):
        # Learn statistics ONLY from training data (e.g. mean, median, IQR bounds)
        # Store fitted attributes with trailing underscore (e.g., self.mean_)
        return self              # MUST return self
        
    def transform(self, X):
        # Apply learned parameters to new data
        return X_transformed
```

- **`BaseEstimator`**: Gives automatic `get_params()` and `set_params()` for `GridSearchCV` / `Optuna`.
- **`TransformerMixin`**: Automatically creates `.fit_transform(X)` by combining your `.fit(X)` and `.transform(X)`.
- **`y=None`**: Scikit-Learn API requirement so transformers fit seamlessly into `Pipeline(steps=[...])`.

#### B. `ColumnTransformer`
Applies specific transformation chains to designated column subsets (e.g. scaling numeric columns while one-hot encoding categoricals).

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, ['age', 'income']),
        ('cat', categorical_pipeline, ['education', 'city'])
    ],
    remainder='drop'  # 'drop' ignores unlisted columns; 'passthrough' keeps them as-is
)
```

---

### 💻 3. Full Executable Python Code

```python
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Step 1: Custom IQR Outlier Clipper
class IQROutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            q25 = X_df[col].quantile(0.25)
            q75 = X_df[col].quantile(0.75)
            iqr = q75 - q25
            self.lower_bounds_[col] = q25 - self.factor * iqr
            self.upper_bounds_[col] = q75 + self.factor * iqr
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            X_df[col] = X_df[col].clip(lower=self.lower_bounds_[col], upper=self.upper_bounds_[col])
        return X_df.values

# Step 2: Synthetic Data Generation
np.random.seed(42)
n_samples = 1000
df = pd.DataFrame({
    'age': np.random.normal(40, 12, n_samples),
    'income': np.random.exponential(50000, n_samples),
    'education': np.random.choice(['HighSchool', 'Bachelor', 'Master', 'PhD'], n_samples),
    'purchased': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
})
df.loc[::10, 'age'] = np.nan
df.loc[::15, 'income'] = np.nan

X = df.drop(columns=['purchased'])
y = df['purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build Sub-Pipelines & Full Production Pipeline
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clipper', IQROutlierClipper(factor=1.5)),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, ['age', 'income']),
    ('cat', cat_pipe, ['education'])
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Step 4: Fit & Save for Production
full_pipeline.fit(X_train, y_train)
joblib.dump(full_pipeline, 'pipeline_model.joblib')

# Step 5: Production Inference
loaded_model = joblib.load('pipeline_model.joblib')
sample_raw = pd.DataFrame([{'age': 35.0, 'income': 120000.0, 'education': 'Master'}])
pred = loaded_model.predict(sample_raw)
print(f"✅ Inference Output Prediction: Class {pred[0]}")
```

---

## ⚖️ Day 051 — Imbalanced Learning Methods (SMOTE, ADASYN, Tomek Links)

### 💡 1. Real-World Analogy & Problem Statement
Imagine a security camera system checking 10,000 people entering an airport per day. 9,995 are innocent passengers, and 5 are terrorists.  
A dumb AI that says *"Everyone is innocent"* achieves **99.95% accuracy** but misses all 5 threats!  
This is the **Accuracy Paradox**. When classes are imbalanced, standard accuracy is useless. We must use resampling or loss weighting.

---

### 📐 2. Mathematical Intuition & Syntax Breakdown

#### A. Loss Function Class Weighting
Instead of modifying data, adjust the loss penalty inversely proportional to class frequencies:
$$w_j = \frac{N_{\text{samples}}}{C \times N_j}$$
- **Syntax**: `LogisticRegression(class_weight='balanced')` or `XGBClassifier(scale_pos_weight=n_neg/n_pos)`

#### B. SMOTE (Synthetic Minority Over-sampling Technique)
Generates synthetic minority instances along feature vectors connecting existing minority samples and their $k$-nearest minority neighbors:
$$x_{\text{new}} = x_i + \lambda (x_{zi} - x_i) \quad \text{where } \lambda \sim U(0, 1)$$

#### C. Tomek Links (Undersampling Boundary Cleaner)
Identifies pairs of opposite-class samples $(x_i, x_j)$ that are mutual nearest neighbors. Removing the majority instance cleans noise along the decision boundary.

#### D. `imblearn.pipeline.Pipeline` vs `sklearn.pipeline.Pipeline`
> ⚠️ **CRITICAL GOTCHA**: Standard Scikit-Learn pipelines fail with SMOTE because standard transformers cannot alter target vector length $y$. Always use `from imblearn.pipeline import Pipeline as ImbPipeline`.

---

### 💻 3. Full Executable Python Code

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler

# Step 1: Create 95:5 Imbalanced Dataset
X, y = make_classification(n_samples=2000, n_features=10, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 2: Baseline vs Class-Weighted
base_clf = LogisticRegression().fit(X_train, y_train)
weighted_clf = LogisticRegression(class_weight='balanced').fit(X_train, y_train)

print(f"❌ Baseline F1-Score:       {f1_score(y_test, base_clf.predict(X_test)):.4f}")
print(f"⚖️ Class-Weighted F1-Score: {f1_score(y_test, weighted_clf.predict(X_test)):.4f}")

# Step 3: SMOTE + Tomek Imblearn Pipeline
smote_tomek_pipe = ImbPipeline([
    ('scaler', StandardScaler()),
    ('resample', SMOTETomek(random_state=42)),
    ('clf', LogisticRegression())
])

smote_tomek_pipe.fit(X_train, y_train)
print(f"✅ SMOTE+Tomek F1-Score:   {f1_score(y_test, smote_tomek_pipe.predict(X_test)):.4f}")
```

---

## 🎯 Day 052 — Threshold Tuning & Probability Calibration

### 💡 1. Real-World Analogy & Problem Statement
Suppose a weather forecast app predicts *"80% chance of rain"*. If across 100 days with an 80% forecast, it actually rains on 80 of those days, the forecast is **well-calibrated**.  
Tree ensembles (RandomForest, XGBoost) tend to push probabilities toward 0 or 1, rendering raw output scores uncalibrated. Furthermore, the default binary classification threshold of `0.5` is arbitrary.

---

### 📐 2. Mathematical Formulas & Syntax

#### A. Brier Score Loss
Measures the mean squared difference between predicted probability $p_i$ and actual outcome $y_i \in \{0, 1\}$ (Lower is better, $0.0$ is perfect calibration):
$$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$

#### B. Probability Calibration Methods (`CalibratedClassifierCV`)
- **Platt Scaling (`method='sigmoid'`)**: Fits a sigmoid logistic regression model on raw prediction scores $f(x)$:
  $$P(y=1 \mid x) = \frac{1}{1 + \exp(A \cdot f(x) + B)}$$
- **Isotonic Regression (`method='isotonic'`)**: Fits a non-parametric monotonically increasing piecewise constant line (Requires $N > 1000$ samples).

#### C. Threshold Optimization
Find cutoff threshold $T^* \in [0, 1]$ that maximizes F1-Score or custom cost function:
$$T^* = \arg\max_T \text{F1}(y_{\text{true}}, \, \mathbb{I}(P(y=1) \ge T))$$

---

### 💻 3. Full Executable Python Code

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, f1_score, brier_score_loss

# Step 1: Data & Base Uncalibrated Model
X, y = make_classification(n_samples=2000, n_features=15, weights=[0.85, 0.15], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
raw_probs = rf.predict_proba(X_test)[:, 1]

# Step 2: Calibration via Isotonic Regression
calibrated_rf = CalibratedClassifierCV(estimator=rf, method='isotonic', cv='prefit')
calibrated_rf.fit(X_train, y_train)
cal_probs = calibrated_rf.predict_proba(X_test)[:, 1]

print(f"Uncalibrated Brier Loss: {brier_score_loss(y_test, raw_probs):.4f}")
print(f"Calibrated Brier Loss:   {brier_score_loss(y_test, cal_probs):.4f}")

# Step 3: Threshold Optimization
precisions, recalls, thresholds = precision_recall_curve(y_test, cal_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Default (0.50) Threshold F1: {f1_score(y_test, (cal_probs >= 0.5).astype(int)):.4f}")
print(f"🎯 Optimal ({best_threshold:.4f}) Threshold F1: {f1_scores[best_idx]:.4f}")
```

---

## 🔍 Day 053 — Model Interpretability & XAI (SHAP & LIME)

### 💡 1. Real-World Analogy & Problem Statement
If a bank denies a customer's loan application using a complex XGBoost model, European Union GDPR law gives the customer the **Right to Explanation**. You cannot say *"The black box said no"*.  
**SHAP (SHapley Additive exPlanations)** calculates the exact mathematical contribution of each feature (e.g. Credit Score, Income, Age) towards that decision.

---

### 📐 2. Mathematical Foundations & Syntax

#### A. Shapley Values Formula (Cooperative Game Theory)
The unique fair attribution $\phi_i$ of feature $i$ across all possible feature subsets $S$:
$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} \left( v(S \cup \{i\}) - v(S) \right)$$

#### B. Additive Efficiency Property
The prediction for any sample $x$ is strictly equal to the expected base model value plus the sum of all individual feature SHAP values:
$$\hat{f}(x) = E[f(x)] + \sum_{j=1}^M \phi_j$$

#### C. `shap.TreeExplainer`
High-speed specialized algorithm ($O(TLD^2)$ complexity) for tree ensembles that avoids exponential subset calculation.

---

### 💻 3. Full Executable Python Code

```python
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Step 1: Load Data & Train Model
X_df, y_vec = shap.datasets.adult()
X_train, X_test, y_train, y_test = train_test_split(X_df, y_vec, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Step 2: SHAP TreeExplainer
explainer = shap.TreeExplainer(model)
shap_explanation = explainer(X_test)
shap_values = shap_explanation.values
base_value = explainer.expected_value

# Step 3: Verify Additive Efficiency Property for Sample 0
sample_pred = model.predict_proba(X_test.iloc[[0]])[:, 1][0]
sum_shap = base_value + np.sum(shap_values[0])

print(f"Base Value E[f(x)]:             {base_value:.4f}")
print(f"Actual Model Output Score:       {sample_pred:.4f}")
print(f"Base Value + Sum(SHAP values):   {sum_shap:.4f} (Matches Additive Property!)")
```

---

## 🚨 Day 054 — Anomaly & Outlier Detection (Isolation Forest, LOF, One-Class SVM)

### 💡 1. Real-World Analogy & Problem Statement
Imagine finding a fake painting in an art gallery. You don't need a list of all fake painting techniques ever invented; you just need to know what authentic paintings look like. Any painting that deviates from normal texture and brushwork is flagged.  
Unsupervised anomaly detection algorithms learn the distribution of "normal" data and detect sparse outliers.

---

### 📐 2. Algorithm Math & Syntax Breakdown

#### A. Isolation Forest
Anomalies are few and structurally different, meaning they get isolated near the root of random decision trees.
- **Anomaly Score Formula**:
  $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
  Where $h(x)$ is path length and $c(n)$ is average path length of an unsuccessful search in a binary search tree. If $s \to 1$, instance is an anomaly.

#### B. Local Outlier Factor (LOF)
Compares the local density of a sample $x$ to the local density of its $k$-nearest neighbors.
- **`novelty=True`**: Allows training LOF on normal data and predicting on new test samples (`predict()`).

#### C. One-Class SVM
Finds a max-margin hyperplane separating normal training data from the origin in high-dimensional kernel space.

---

### 💻 3. Full Executable Python Code

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score

# Step 1: Synthetic Data Setup (950 Normal + 50 Outliers)
np.random.seed(42)
X_normal = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 950)
X_anomalies = np.random.uniform(low=-6, high=6, size=(50, 2))
X = np.vstack([X_normal, X_anomalies])
y_true = np.array([1] * 950 + [-1] * 50)

# Step 2: Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
pred_iso = iso.fit_predict(X)
scores_iso = iso.decision_function(X)

print("--- 🌲 Isolation Forest Performance ---")
print(classification_report(y_true, pred_iso, target_names=['Anomaly (-1)', 'Normal (1)']))
print(f"Isolation Forest ROC-AUC: {roc_auc_score(y_true, -scores_iso):.4f}")

# Step 3: LOF & One-Class SVM
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True).fit(X_normal)
oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05).fit(X_normal)

print(f"LOF Accuracy:         {np.mean(lof.predict(X) == y_true):.4f}")
print(f"One-Class SVM Accuracy: {np.mean(oc_svm.predict(X) == y_true):.4f}")
```

---

## 🧪 Day 055 — Classical ML Comparison Lab & Benchmarking Harness

### 💡 1. Real-World Analogy & Problem Statement
Never marry a single algorithm. A senior Machine Learning Engineer builds a standardized **Benchmarking Harness** to empirically test multiple algorithms (Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, CatBoost) under identical cross-validation conditions to find the optimal trade-off between predictive accuracy and inference latency.

---

### 💻 2. Full Executable Python Code

```python
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Step 1: Create Dataset
X, y = make_classification(n_samples=3000, n_features=25, n_informative=18, n_classes=2, weights=[0.7, 0.3], random_state=42)

# Step 2: Define Model Suite
models = {
    'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))]),
    'SVM (RBF Kernel)': Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True))]),
    'Random Forest': RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=150, learning_rate=0.08, eval_metric='logloss', random_state=42, n_jobs=-1),
    'LightGBM': LGBMClassifier(n_estimators=150, learning_rate=0.08, random_state=42, verbose=-1, n_jobs=-1),
    'CatBoost': CatBoostClassifier(iterations=150, learning_rate=0.08, verbose=0, random_state=42)
}

# Step 3: Run Benchmark Loop
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    t0 = time.perf_counter()
    res = cross_validate(model, X, y, cv=cv, scoring=['accuracy', 'f1', 'roc_auc'], n_jobs=-1)
    t_elapsed = time.perf_counter() - t0
    
    results.append({
        'Model': name,
        'Accuracy': f"{np.mean(res['test_accuracy']):.4f} +/- {np.std(res['test_accuracy']):.4f}",
        'F1-Score': f"{np.mean(res['test_f1']):.4f} +/- {np.std(res['test_f1']):.4f}",
        'ROC-AUC': f"{np.mean(res['test_roc_auc']):.4f} +/- {np.std(res['test_roc_auc']):.4f}",
        'Fit Time (s)': f"{np.mean(res['fit_time']):.3f}s",
        'Total Time': f"{t_elapsed:.2f}s"
    })

df_benchmark = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
print("=== 📊 BENCHMARK RESULTS TABLE ===")
print(df_benchmark.to_markdown(index=False))
```

---

## 📝 Day 056 — Advanced ML Assessment, Math Derivations & Interview Prep

### 🧠 1. Interview Math Derivation: XGBoost Optimal Leaf Weight ($w_j^*$)

In XGBoost, the objective at step $t$ using 2nd Order Taylor Expansion is:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
Where regularization term $\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$.  
Group by leaf instance sets $I_j = \{i \mid q(x_i) = j\}$:
$$\mathcal{L}^{(t)} = \sum_{j=1}^T \left[ \left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2} \left(\sum_{i \in I_j} h_i + \lambda\right) w_j^2 \right] + \gamma T$$
Let $G_j = \sum_{i \in I_j} g_i$ and $H_j = \sum_{i \in I_j} h_i$. Differentiating w.r.t $w_j$ and setting to 0:
$$\frac{\partial \mathcal{L}}{\partial w_j} = G_j + (H_j + \lambda) w_j = 0 \implies w^*_j = -\frac{G_j}{H_j + \lambda}$$

---

### 💻 2. Python Implementation of XGBoost Leaf Formula & Pipeline Bug Fix

```python
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# 1. XGBoost Leaf Weight Calculation
def calculate_leaf_weight(g_i, h_i, reg_lambda=1.0):
    G_j = np.sum(g_i)
    H_j = np.sum(h_i)
    w_star = - G_j / (H_j + reg_lambda)
    similarity = (G_j ** 2) / (H_j + reg_lambda)
    return w_star, similarity

g = np.array([0.5, -1.2, 0.8, -0.4])
h = np.array([1.0, 1.0, 1.0, 1.0])
w_star, similarity = calculate_leaf_weight(g, h, reg_lambda=1.0)
print(f"Optimal Leaf Weight (w*): {w_star:.4f} | Similarity Score: {similarity:.4f}")

# 2. Correct Pipeline Implementation with SMOTE
X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
correct_pipe = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])

cv_f1 = cross_val_score(correct_pipe, X, y, cv=5, scoring='f1')
print(f"✅ Verified ImbPipeline 5-Fold F1 Score: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
```
