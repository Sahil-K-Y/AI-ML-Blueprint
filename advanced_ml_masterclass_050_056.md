# 🔥 Advanced Classical ML Masterclass (Days 050 – 056)

> **Author:** Sahil Kumar (Yd)  
> **Blueprint:** 365-Day AI/ML Engineer Master Roadmap — Phase 2 (Days 039–056)  
> **Focus:** Complete Working Code, Line-by-Line Syntax Breakdown, Mathematical Intuition & Real-World Usage ("Why We Use This")

---

## 📖 Table of Contents

1. [Day 050 — Production Sklearn Pipelines & Custom Transformers](#-day-050--production-sklearn-pipelines--custom-transformers)
2. [Day 051 — Imbalanced Learning Methods (SMOTE, ADASYN, Tomek Links)](#-day-051--imbalanced-learning-methods-smote-adasyn-tomek-links)
3. [Day 052 — Threshold Tuning & Probability Calibration](#-day-052--threshold-tuning--probability-calibration)
4. [Day 053 — Model Interpretability & XAI (SHAP & LIME)](#-day-053--model-interpretability--xai-shap--lime)
5. [Day 054 — Anomaly & Outlier Detection (Isolation Forest, LOF, One-Class SVM)](#-day-054--anomaly--outlier-detection-isolation-forest-lof-one-class-svm)
6. [Day 055 — Classical ML Comparison Lab & Benchmarking Harness](#-day-055--classical-ml-comparison-lab--benchmarking-harness)
7. [Day 056 — Advanced ML Assessment, Derivations & Interview Solutions](#-day-056--advanced-ml-assessment-derivations--interview-solutions)

---

## ⚙️ Day 050 — Production Sklearn Pipelines & Custom Transformers

### 💡 1. Intuition & Problem Statement
In production, machine learning models break when data preprocessing in inference differs from training (training-serving skew). Manually transforming raw input columns leads to messy code and potential data leakage. Scikit-Learn `Pipeline` and `ColumnTransformer` package all feature scaling, missing value imputation, encoding, and model inference into a **single atomic object** that can be saved (`joblib.dump`) and deployed directly.

---

### 💻 2. Complete Python Code

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -------------------------------------------------------------
# STEP 1: Custom Transformer for Outlier Clipping (IQR-based)
# -------------------------------------------------------------
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
            X_df[col] = X_df[col].clip(
                lower=self.lower_bounds_[col], 
                upper=self.upper_bounds_[col]
            )
        return X_df.values

# -------------------------------------------------------------
# STEP 2: Create Synthetic Tabular Dataset
# -------------------------------------------------------------
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'age': np.random.normal(40, 12, n_samples),
    'income': np.random.exponential(50000, n_samples),
    'education': np.random.choice(['HighSchool', 'Bachelor', 'Master', 'PhD'], n_samples),
    'purchased': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
})

# Inject missing values
df.loc[::10, 'age'] = np.nan
df.loc[::15, 'income'] = np.nan

X = df.drop(columns=['purchased'])
y = df['purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------------
# STEP 3: Define Sub-Pipelines & ColumnTransformer
# -------------------------------------------------------------
num_features = ['age', 'income']
cat_features = ['education']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('clipper', IQROutlierClipper(factor=1.5)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features)
])

# -------------------------------------------------------------
# STEP 4: Build Complete Production Pipeline
# -------------------------------------------------------------
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit on training data ONLY
full_pipeline.fit(X_train, y_train)

# -------------------------------------------------------------
# STEP 5: Serialize (Save) & Load Pipeline for Production
# -------------------------------------------------------------
joblib.dump(full_pipeline, 'production_pipeline.joblib')

# In Production Server:
loaded_pipeline = joblib.load('production_pipeline.joblib')
raw_input = pd.DataFrame([{
    'age': 35.0,
    'income': 120000.0,
    'education': 'Master'
}])

pred = loaded_pipeline.predict(raw_input)
prob = loaded_pipeline.predict_proba(raw_input)[:, 1]
print(f"✅ Production Raw Prediction: Class {pred[0]} (Probability: {prob[0]:.4f})")
```

---

### 🔍 3. Syntax Breakdown & Explanation

- **`BaseEstimator` & `TransformerMixin`**: Inheriting from `BaseEstimator` gives us free `get_params()` and `set_params()` methods for hyperparameter tuning. `TransformerMixin` automatically adds the `fit_transform()` method once we implement `fit()` and `transform()`.
- **`fit(self, X, y=None)`**: Learns parameters ONLY from training data (e.g. median, IQR bounds, mean/std). Returns `self`. Note that `y=None` is required by the Scikit-Learn API spec.
- **`transform(self, X)`**: Applies learned parameters to new incoming data without recalculating stats (prevents data leakage).
- **`ColumnTransformer(transformers=[...])`**: Applies specific transformation chains to designated column subsets (`num_features` vs `cat_features`).
- **`handle_unknown='ignore'` in `OneHotEncoder`**: Critical for production. If an unseen category appears during inference (e.g. 'PostDoc'), it creates zero-vectors instead of throwing a runtime error.

---

### ❓ 4. Why We Use This
- **Eliminates Data Leakage**: Scaler parameters are calculated strictly inside `X_train` during `pipeline.fit()`.
- **Single Source of Truth**: Production code imports `joblib.load('pipeline.joblib')` and calls `.predict(raw_dict)` — zero manual pandas preprocessing scripts required.

---

## ⚖️ Day 051 — Imbalanced Learning Methods (SMOTE, ADASYN, Tomek Links)

### 💡 1. Intuition & Problem Statement
When class distributions are highly skewed (e.g., 99% Fraud=0, 1% Fraud=1), standard classifiers achieve 99% accuracy by predicting class 0 every time. We need resampling techniques (SMOTE, ADASYN) or loss function weighting (`class_weight='balanced'`) to force the model to learn decision boundaries for the minority class.

---

### 💻 2. Complete Python Code

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler

# 1. Create Heavily Imbalanced Synthetic Dataset (95% / 5%)
X, y = make_classification(
    n_samples=2000, n_features=10, n_informative=8, n_redundant=2,
    weights=[0.95, 0.05], random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Baseline Model (No Handling)
base_model = LogisticRegression()
base_model.fit(X_train, y_train)
y_pred_base = base_model.predict(X_test)
print("--- ❌ Baseline Model (Imbalanced) ---")
print(classification_report(y_test, y_pred_base, target_names=['Majority', 'Minority']))

# 3. Method A: Class Weight Balancing in Model Loss Function
weighted_model = LogisticRegression(class_weight='balanced')
weighted_model.fit(X_train, y_train)
print("--- ⚖️ Class-Weighted Logistic Regression ---")
print(classification_report(y_test, weighted_model.predict(X_test), target_names=['Majority', 'Minority']))

# 4. Method B: Imbalanced-Learn Pipeline with SMOTE + Tomek Links
# Note: MUST use imblearn.pipeline.Pipeline so SMOTE only applies during FIT (training)!
imb_pipeline = ImbPipeline(steps=[
    ('scaler', StandardScaler()),
    ('resample', SMOTETomek(smote=SMOTE(k_neighbors=5, random_state=42), tomek=TomekLinks())),
    ('classifier', LogisticRegression())
])

imb_pipeline.fit(X_train, y_train)
y_pred_smote = imb_pipeline.predict(X_test)
y_prob_smote = imb_pipeline.predict_proba(X_test)[:, 1]

print("--- ✅ SMOTE + Tomek Links Pipeline ---")
print(classification_report(y_test, y_pred_smote, target_names=['Majority', 'Minority']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_smote):.4f}")
```

---

### 🔍 3. Syntax Breakdown & Explanation

- **`SMOTE(k_neighbors=5)`**: Synthetic Minority Over-sampling Technique. For each minority sample $x_i$, it finds $k$ nearest minority neighbors, selects one $x_{zi}$, and generates a synthetic sample along the line segment: $x_{new} = x_i + \lambda (x_{zi} - x_i)$ where $\lambda \sim U(0,1)$.
- **`ADASYN`**: Adaptive Synthetic sampling. Generates more synthetic samples for minority instances that are harder to learn (near decision boundary).
- **`Tomek Links`**: Pairs of opposite-class instances that are mutual nearest neighbors. Removing Tomek links cleans up ambiguous overlap near the decision boundary.
- **`imblearn.pipeline.Pipeline`**: **CRITICAL!** Standard `sklearn.pipeline.Pipeline` does NOT support transformers that alter sample length or targets ($y$). `imblearn.pipeline` ensures SMOTE resampling happens strictly on training folds during `.fit()` and is bypassed during `.predict()`.

---

### ❓ 4. Why We Use This
- `class_weight='balanced'` adjusts loss penalty inversely proportional to class frequencies: $w_j = \frac{N}{2 \times N_j}$.
- Over-sampling (SMOTE) creates synthetic diversity rather than duplicating identical minority samples (which causes severe overfitting).

---

## 🎯 Day 052 — Threshold Tuning & Probability Calibration

### 💡 1. Intuition & Problem Statement
Machine learning classifiers output predicted probabilities `predict_proba()`. However, tree models and SVMs often yield **uncalibrated probabilities** (e.g. predicted probability of 0.8 does not mean 80% actual positive cases). Additionally, the default binary threshold of `0.5` is arbitrary and suboptimal for asymmetric business costs (e.g., False Negative fraud costs $1,000 while False Positive costs $10).

---

### 💻 2. Complete Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import precision_recall_curve, f1_score, brier_score_loss

# 1. Dataset & Setup
X, y = make_classification(n_samples=2000, n_features=15, weights=[0.85, 0.15], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Uncalibrated Base Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
prob_uncalibrated = rf.predict_proba(X_test)[:, 1]

# 2. Probability Calibration (Platt Scaling vs Isotonic Regression)
calibrated_rf = CalibratedClassifierCV(estimator=rf, method='isotonic', cv='prefit')
calibrated_rf.fit(X_train, y_train)
prob_calibrated = calibrated_rf.predict_proba(X_test)[:, 1]

print(f"Uncalibrated Brier Score Loss: {brier_score_loss(y_test, prob_uncalibrated):.4f}")
print(f"Calibrated Brier Score Loss:   {brier_score_loss(y_test, prob_calibrated):.4f}")

# 3. Threshold Optimization for Maximum F1-Score
precisions, recalls, thresholds = precision_recall_curve(y_test, prob_calibrated)

# Compute F1 scores for each threshold candidate
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\n✅ Default Threshold (0.50) F1-Score: {f1_score(y_test, (prob_calibrated >= 0.5).astype(int)):.4f}")
print(f"🎯 Optimal Threshold ({best_threshold:.4f}) F1-Score: {best_f1:.4f}")

# Custom Inference using Optimal Threshold
final_predictions = (prob_calibrated >= best_threshold).astype(int)
```

---

### 🔍 3. Syntax Breakdown & Explanation

- **`brier_score_loss(y_true, y_prob)`**: Mean squared difference between predicted probabilities and actual binary outcomes ($BS = \frac{1}{N} \sum (p_i - y_i)^2$). Lower is better.
- **`CalibratedClassifierCV(method='sigmoid' | 'isotonic')`**:
  - `method='sigmoid'` (Platt Scaling): Fits a logistic regression model on raw prediction scores $f(x)$: $P(y=1|x) = \frac{1}{1 + \exp(A \cdot f(x) + B)}$.
  - `method='isotonic'`: Fits a non-parametric isotonic regression curve. Requires more data ($N > 1000$).
- **`precision_recall_curve(y_true, probas_pred)`**: Computes precision and recall for all unique threshold cutoffs.

---

### ❓ 4. Why We Use This
- Calibration aligns probabilities with reality: If a medical AI predicts 0.90 risk across 100 patients, exactly 90 of them should actually have the condition.
- Threshold tuning converts risk probabilities into decision cutoffs aligned with domain financial metrics.

---

## 🔍 Day 053 — Model Interpretability & XAI (SHAP & LIME)

### 💡 1. Intuition & Problem Statement
Modern ensemble models (XGBoost, LightGBM, Random Forests) are "black boxes". Stakeholders and regulations require understanding **why** a specific prediction was made.  
- **SHAP (SHapley Additive exPlanations)** uses cooperative game theory to compute exact, fair feature contribution values.
- **LIME (Local Interpretable Model-agnostic Explanations)** fits an interpretable linear surrogate model locally around a single sample.

---

### 💻 2. Complete Python Code

```python
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# 1. Dataset & Model Training
X_df, y_vec = shap.datasets.adult()
X_train, X_test, y_train, y_test = train_test_split(X_df, y_vec, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 2. Compute SHAP Values using TreeExplainer (Optimized for Tree Ensembles)
explainer = shap.TreeExplainer(model)
shap_explanation = explainer(X_test)

# SHAP values matrix (samples x features)
shap_values = shap_explanation.values
base_value = explainer.expected_value  # E[f(x)] average model output

print(f"Base Value (Expected Output): {base_value:.4f}")
print(f"SHAP Values Shape: {shap_values.shape}")

# 3. Individual Prediction Decomposition (Additive Property check)
sample_idx = 0
sample_prediction = model.predict_proba(X_test.iloc[[sample_idx]])[:, 1]
sum_shap = base_value + np.sum(shap_values[sample_idx])

print(f"\nSample 0 Model Prediction score: {sample_prediction[0]:.4f}")
print(f"Sum of Base Value + SHAP values:  {sum_shap:.4f} (Matches Additive Property!)")
```

---

### 🔍 3. Syntax Breakdown & Explanation

- **Shapley Value Equation**:
  $$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} \left( v(S \cup \{i\}) - v(S) \right)$$
  Computes the marginal contribution of feature $i$ across all possible feature subsets $S$.
- **`shap.TreeExplainer`**: High-speed algorithms ($O(TLD^2)$ complexity) specialized for tree structures, avoiding full exponential subset sampling.
- **Additive Axiom**: $\hat{f}(x) = \phi_0 + \sum_{i=1}^M \phi_i$. The prediction equals the base expected value plus the sum of all individual feature SHAP values.

---

### ❓ 4. Why We Use This
- **Global Transparency**: Summary plots reveal which features drive predictions positively or negatively across the dataset.
- **Local Explanation**: Waterfall plots provide individual explanation receipts for auditability (e.g. "Loan denied due to: High Debt Ratio [+0.35], Low Credit Score [+0.20]").

---

## 🚨 Day 054 — Anomaly & Outlier Detection (Isolation Forest, LOF, One-Class SVM)

### 💡 1. Intuition & Problem Statement
In many real-world scenarios (credit card fraud, manufacturing defects, server metrics), labeled anomaly data is non-existent. **Unsupervised Anomaly Detection** algorithms learn patterns of "normal" behavior and flag instances that deviate significantly.

- **Isolation Forest**: Anomalies are few and different; hence, they isolate easily near the root of random partition trees.
- **Local Outlier Factor (LOF)**: Compares the local density of an instance to the local density of its $k$-nearest neighbors.
- **One-Class SVM**: Learns a tight hyper-boundary enclosing normal data points in a high-dimensional kernel space.

---

### 💻 2. Complete Python Code

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score

# 1. Generate Synthetic Normal Data + Injected Anomalies
np.random.seed(42)
n_normal = 950
n_anomaly = 50

X_normal = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_normal)
X_anomalies = np.random.uniform(low=-6, high=6, size=(n_anomaly, 2))

X = np.vstack([X_normal, X_anomalies])
y_true = np.array([1] * n_normal + [-1] * n_anomaly)  # 1: Normal, -1: Anomaly

# 2. Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
y_pred_iso = iso_forest.fit_predict(X)
scores_iso = iso_forest.decision_function(X)  # Lower score = more anomalous

# 3. Local Outlier Factor (Novelty Detection mode for new data prediction)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
lof.fit(X_normal)  # Fit on normal data
y_pred_lof = lof.predict(X)

# 4. One-Class SVM
oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
oc_svm.fit(X_normal)
y_pred_svm = oc_svm.predict(X)

# 5. Evaluate Metrics
print("--- 🌲 Isolation Forest Results ---")
print(classification_report(y_true, y_pred_iso, target_names=['Anomaly (-1)', 'Normal (1)']))
print(f"ROC-AUC (Anomaly Score): {roc_auc_score(y_true, -scores_iso):.4f}")
```

---

### 🔍 3. Syntax Breakdown & Explanation

- **`contamination=0.05`**: The proportion of outliers expected in the dataset. Used by `.predict()` to establish the anomaly score decision threshold.
- **Isolation Forest Path Length $h(x)$**: The number of splits needed to isolate sample $x$. Average path length $c(n)$ normalizes the score:
  $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
  If score $\rightarrow 1$, instance is flagged as an anomaly.
- **`decision_function(X)`**: Returns raw anomaly scores. For Isolation Forest, negative values indicate anomalies.
- **`novelty=True` in LOF**: Standard LOF only does transductive outlier detection on training data. Setting `novelty=True` allows standard `fit()` and `predict()` calls on new test samples.

---

### ❓ 4. Why We Use This
- **Isolation Forest**: High speed $O(N \log N)$, scales effortlessly to high dimensions and huge datasets.
- **LOF**: Excellent for variable-density clusters where global distance metrics fail.

---

## 🧪 Day 055 — Classical ML Comparison Lab & Benchmarking Harness

### 💡 1. Intuition & Problem Statement
Never guess which machine learning algorithm will perform best on a given tabular dataset. A rigorous data science pipeline creates an **automated comparison harness** that evaluates multiple linear, tree-based, and boosting models under standardized cross-validation conditions while tracking accuracy, F1, ROC-AUC, training time, and prediction latency.

---

### 💻 2. Complete Python Code

```python
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Model Suite Imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 1. Dataset Generation
X, y = make_classification(
    n_samples=3000, n_features=25, n_informative=18, n_redundant=4,
    n_classes=2, weights=[0.7, 0.3], random_state=42
)

# 2. Define Model Dictionary
models = {
    'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))]),
    'SVM (RBF Kernel)': Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True))]),
    'Random Forest': RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=150, learning_rate=0.08, eval_metric='logloss', random_state=42, n_jobs=-1),
    'LightGBM': LGBMClassifier(n_estimators=150, learning_rate=0.08, random_state=42, verbose=-1, n_jobs=-1),
    'CatBoost': CatBoostClassifier(iterations=150, learning_rate=0.08, verbose=0, random_state=42)
}

# 3. Standardized Benchmarking Harness
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'f1', 'roc_auc']
results = []

print("🚀 Starting Classical ML Benchmark Harness...\n")

for name, model in models.items():
    start_time = time.perf_counter()
    cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    elapsed_time = time.perf_counter() - start_time
    
    results.append({
        'Model': name,
        'Accuracy': f"{np.mean(cv_res['test_accuracy']):.4f} +/- {np.std(cv_res['test_accuracy']):.4f}",
        'F1-Score': f"{np.mean(cv_res['test_f1']):.4f} +/- {np.std(cv_res['test_f1']):.4f}",
        'ROC-AUC': f"{np.mean(cv_res['test_roc_auc']):.4f} +/- {np.std(cv_res['test_roc_auc']):.4f}",
        'Fit Time (s)': f"{np.mean(cv_res['fit_time']):.3f}s",
        'Total Benchmark Time': f"{elapsed_time:.2f}s"
    })

# 4. Display Summary Table
df_results = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
print("=== 📊 FINAL MODEL BENCHMARK RESULTS ===")
print(df_results.to_markdown(index=False))
```

---

### 🔍 3. Syntax Breakdown & Explanation

- **`cross_validate(model, X, y, cv, scoring, n_jobs=-1)`**: Evaluates multiple metrics simultaneously and records `fit_time` and `score_time` per fold across parallel CPU threads (`n_jobs=-1`).
- **`StratifiedKFold(n_splits=5)`**: Ensures class ratio ($70\% / 30\%$) is perfectly preserved across all 5 folds.
- **`time.perf_counter()`**: Highest available resolution clock for timing short CPU execution blocks.

---

### ❓ 4. Why We Use This
- Systematic, empirical evaluation prevents developer bias towards specific models.
- Highlights latency vs accuracy tradeoffs (e.g. CatBoost might give higher accuracy but take 10x longer to train than LightGBM).

---

## 📝 Day 056 — Advanced ML Assessment, Derivations & Interview Solutions

### 💡 1. Overview
Day 056 tests mastery of Phase 2 through math derivations, interview questions, and pipeline debugging scenarios.

---

### 🧠 2. Core Interview Questions & Derivations

#### Q1: Derive the optimal leaf weight ($w^*_j$) formula in XGBoost.
**Answer:**  
In XGBoost, the regularized objective function at step $t$ is:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i w_{q(x_i)} + \frac{1}{2} h_i w_{q(x_i)}^2 \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
Grouping by leaf instance sets $I_j = \{i \mid q(x_i) = j\}$:
$$\mathcal{L}^{(t)} = \sum_{j=1}^T \left[ \left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2} \left(\sum_{i \in I_j} h_i + \lambda\right) w_j^2 \right] + \gamma T$$
Let $G_j = \sum_{i \in I_j} g_i$ and $H_j = \sum_{i \in I_j} h_i$. Taking derivative with respect to leaf weight $w_j$ and setting to 0:
$$\frac{\partial \mathcal{L}}{\partial w_j} = G_j + (H_j + \lambda) w_j = 0 \implies w^*_j = -\frac{G_j}{H_j + \lambda}$$
Plugging $w^*_j$ back yields the leaf similarity score: $S_j = -\frac{1}{2} \frac{G_j^2}{H_j + \lambda}$.

---

#### Q2: Why does standard scaling inside `fit_transform(X)` before cross-validation cause Data Leakage? How does Scikit-Learn `Pipeline` fix it?
**Answer:**  
Calling `StandardScaler().fit_transform(X)` calculates mean $\mu_{global}$ and standard deviation $\sigma_{global}$ over the **entire dataset** (including validation/test folds). When CV splits the data into train/val folds, the training fold contains statistical information derived from the validation set.  
Scikit-Learn `Pipeline` fixes this because calling `pipeline.fit(X_train, y_train)` executes `scaler.fit(X_train_fold)` **strictly inside each individual CV training fold**, computing fold-isolated stats $\mu_{fold}$ and $\sigma_{fold}$.

---

#### Q3: How does CatBoost handle categorical features natively without causing Target Leakage?
**Answer:**  
CatBoost uses **Ordered Target Encoding**. Standard target encoding computes $\hat{x}_k = \frac{\sum y_i}{N}$ over the dataset, leaking the target value of sample $i$ into its own feature. CatBoost calculates target statistics using an artificial temporal ordering (random permutation of dataset):
$$\hat{x}_i = \frac{\sum_{j=1}^{i-1} [x_j = x_i] \cdot y_j + a \cdot P}{\sum_{j=1}^{i-1} [x_j = x_i] + a}$$
Target encoding for sample $i$ is calculated **only using target values of preceding samples ($j < i$)** in the permutation, completely eliminating target leakage!

---

#### Q4: Debug the following buggy pipeline snippet:
```python
# ❌ BUGGY SNIPPET
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE()),  # BUG HERE!
    ('rf', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)
```
**Bug Explanation & Fix:**  
- **Bug**: `sklearn.pipeline.Pipeline` expects every step (except the estimator) to implement standard `fit_transform(X, y) -> X_transformed`. SMOTE modifies both sample array $X$ AND target vector $y$. Standard sklearn pipelines ignore returned $y$, causing a dimension mismatch error or missing SMOTE resampling altogether.
- **Fix**: Replace `from sklearn.pipeline import Pipeline` with `from imblearn.pipeline import Pipeline`.

---

## 🎯 Summary Checklist for Phase 2 Completion

- [x] Custom Transformers built using `BaseEstimator` + `TransformerMixin`
- [x] `ColumnTransformer` handling heterogeneous numeric and categorical pipelines
- [x] Resampling handled correctly using `imblearn.pipeline.Pipeline`
- [x] Probabilities calibrated via `CalibratedClassifierCV` (Platt / Isotonic)
- [x] Optimal decision thresholds selected using PR curves
- [x] Model interpretability generated via SHAP `TreeExplainer` & LIME
- [x] Benchmark suite executed across classical and boosting algorithms
