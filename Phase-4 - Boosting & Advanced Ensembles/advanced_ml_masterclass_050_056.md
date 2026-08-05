# 🔥 Advanced Classical ML Masterclass (Days 050 – 056)
## Comprehensive Theory, Mathematics, Code Syntax & Production Blueprint

> **Author:** Sahil Kumar (Yd)  
> **Blueprint:** 365-Day AI/ML Engineer Master Roadmap — Phase 2 (Days 039–056)  
> **Location:** `Phase-4 - Boosting & Advanced Ensembles/advanced_ml_masterclass_050_056.md`  
> **Target:** Deep conceptual clarity, line-by-line syntax breakdown, mathematical derivations, and production implementation.

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

### 💡 1. Intuition & Real-World Problem Analogy
In machine learning engineering, transitioning from Jupyter notebooks to production REST APIs (FastAPI/Flask) is where most projects fail due to **Training-Serving Skew** and **Data Leakage**.

#### The Analogy — Industrial Water Purification Plant 🚰
- **Without Pipelines**: You clean river water in bucket A, test the pH in bucket B, add chlorine in bucket C, and finally test the water quality. When deploying to the city supply, you attempt to repeat all these manual steps. If someone skips bucket B, contaminated water flows to the city.
- **With Scikit-Learn Pipeline**: You construct a single sealed purification pipe. Dirty water goes in at one end; purified, drinkable water comes out at the other. The pipe remembers the exact filtration parameters calculated during calibration.

#### Data Leakage Prevention Mechanism:
If you run `StandardScaler().fit_transform(X)` on your entire dataset before splitting into train/test, the scalar learns the mean $\mu_{\text{total}}$ and standard deviation $\sigma_{\text{total}}$ of the **test data**. 
When using `Pipeline`, calling `pipeline.fit(X_train, y_train)` ensures that `StandardScaler.fit()` is executed **strictly on `X_train`** inside each cross-validation fold.

---

### 📐 2. Mathematical Formulation & Architecture

#### Custom Transformer Inheritance Architecture:
To integrate seamlessly into Scikit-Learn's ecosystem, a custom class must inherit from two base classes:
1. `BaseEstimator`: Provides automated parameter management methods `get_params()` and `set_params()`, enabling compatibility with `GridSearchCV`, `RandomizedSearchCV`, and `Optuna`.
2. `TransformerMixin`: Automatically implements `.fit_transform(X, y)` by chaining `.fit(X, y)` followed by `.transform(X)`.

```
                  ┌────────────────────┐      ┌─────────────────────┐
                  │   BaseEstimator    │      │  TransformerMixin   │
                  └─────────┬──────────┘      └──────────┬──────────┘
                            │                            │
                            └─────────────┬──────────────┘
                                          │
                                          ▼
                             ┌────────────────────────┐
                             │ Custom Transformer     │
                             │ - __init__()           │
                             │ - fit(X, y)            │
                             │ - transform(X)         │
                             └────────────────────────┘
```

#### IQR Outlier Clipping Formula:
For a feature vector $x$:
$$\text{Q25} = \text{Percentile}(x, 25), \quad \text{Q75} = \text{Percentile}(x, 75)$$
$$\text{IQR} = \text{Q75} - \text{Q25}$$
$$\text{Lower Bound} = \text{Q25} - c \cdot \text{IQR}, \quad \text{Upper Bound} = \text{Q75} + c \cdot \text{IQR}$$
$$\hat{x}_i = \min\left(\max\left(x_i, \text{Lower Bound}\right), \text{Upper Bound}\right)$$

---

### 💻 3. Step-by-Step Production Python Implementation

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
from sklearn.metrics import classification_report

# -----------------------------------------------------------------------------
# STEP 1: Define Custom Transformer for IQR Outlier Clipping
# -----------------------------------------------------------------------------
class IQROutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clips numeric feature values beyond the IQR thresholds.
    Learns bounds during fit() and applies them during transform().
    """
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
            # Trailing underscore denotes fitted parameters learned from training data
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

# -----------------------------------------------------------------------------
# STEP 2: Generate Synthetic Dataset with Missing Values & Mixed Types
# -----------------------------------------------------------------------------
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'age': np.random.normal(40, 12, n_samples),
    'income': np.random.exponential(50000, n_samples),
    'credit_score': np.random.normal(650, 50, n_samples),
    'education': np.random.choice(['HighSchool', 'Bachelor', 'Master', 'PhD'], n_samples),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n_samples),
    'purchased': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
})

# Inject synthetic missing values
df.loc[::10, 'age'] = np.nan
df.loc[::15, 'income'] = np.nan

X = df.drop(columns=['purchased'])
y = df['purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------------------------------------------
# STEP 3: Define Column-Specific Sub-Pipelines
# -----------------------------------------------------------------------------
numeric_features = ['age', 'income', 'credit_score']
categorical_features = ['education', 'city']

# Numeric Processing Chain: Impute -> Clip Outliers -> Scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('clipper', IQROutlierClipper(factor=1.5)),
    ('scaler', StandardScaler())
])

# Categorical Processing Chain: Mode Impute -> One-Hot Encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine via ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# -----------------------------------------------------------------------------
# STEP 4: Build Complete Production Pipeline
# -----------------------------------------------------------------------------
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Evaluate using 5-Fold Cross Validation
cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"✅ 5-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# -----------------------------------------------------------------------------
# STEP 5: Serialize & Production Deployment Test
# -------------------------------------------------------------
full_pipeline.fit(X_train, y_train)
joblib.dump(full_pipeline, 'production_pipeline.joblib')

# Load and Predict on Raw Dictionary Payload
loaded_pipeline = joblib.load('production_pipeline.joblib')
raw_input = pd.DataFrame([{
    'age': 35.0,
    'income': 120000.0,
    'credit_score': 720.0,
    'education': 'Master',
    'city': 'NYC'
}])

pred = loaded_pipeline.predict(raw_input)
prob = loaded_pipeline.predict_proba(raw_input)[:, 1]
print(f"🎯 Production Inference -> Class: {pred[0]} (Probability: {prob[0]:.4f})")
```

---

### 🔍 4. Line-by-Line Syntax & Parameter Breakdown

- `BaseEstimator`: Grants `get_params()` and `set_params()` methods so hyperparameters (like `factor`) can be searched in tuning grids.
- `TransformerMixin`: Provides `fit_transform()`.
- `self.lower_bounds_`: Trailing underscore in scikit-learn signifies attributes that are learned during `.fit()` (convention).
- `SimpleImputer(strategy='median')`: Fills missing values with column medians (robust to outliers compared to mean).
- `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`:
  - `handle_unknown='ignore'`: Critical for production. If an unseen category appears during inference, it assigns all 0s to one-hot columns instead of crashing.
  - `sparse_output=False`: Returns dense NumPy arrays for compatibility.
- `ColumnTransformer(transformers=[('name', pipeline, columns)])`: Applies specified transformation chains only to designated column names/indices.

---

### ❓ 5. Why We Use This & Production Trade-offs
- **Single Artifact Deployment**: Saves preprocessing steps + model into one `.joblib` file. Production servers call `model.predict(raw_json)` directly.
- **Zero Data Leakage**: Scaler mean and variance are computed solely inside `X_train` during `pipeline.fit()`.

---

## ⚖️ Day 051 — Imbalanced Learning Methods (SMOTE, ADASYN, Tomek Links)

### 💡 1. Intuition & Real-World Problem Analogy

#### The Accuracy Paradox ⚠️
Suppose you are building a credit card fraud detection system. Out of 100,000 daily transactions, 99,900 are legitimate (Class 0) and 100 are fraudulent (Class 1).
If a dummy model predicts *"All transactions are legitimate"*, it achieves **99.9% accuracy**! However, it catches **0% of fraudulent transactions**, leading to catastrophic financial loss.

#### Resampling Strategies Intuition:
- **`class_weight='balanced'`**: Leaves the data intact, but penalizes misclassifying a minority sample proportional to its rarity in the loss function.
- **SMOTE (Oversampling)**: Draws imaginary line segments between minority class points and creates synthetic new points along those lines.
- **ADASYN (Adaptive Oversampling)**: Focuses synthetic sample creation specifically in dense boundary regions where majority and minority classes overlap.
- **Tomek Links (Undersampling)**: Finds pairs of opposite-class points that are each other's nearest neighbors and removes the majority point to clean noise along the decision boundary.

---

### 📐 2. Mathematical Formulations

#### A. Cost-Sensitive Loss Weighting Formula
For a dataset with $N$ samples, $C$ classes, and $N_j$ samples in class $j$:
$$w_j = \frac{N}{C \times N_j}$$
For binary classification with 95% Class 0 ($N_0 = 950$) and 5% Class 1 ($N_1 = 50$):
$$w_0 = \frac{1000}{2 \times 950} \approx 0.526, \quad w_1 = \frac{1000}{2 \times 50} = 10.0$$

#### B. SMOTE Synthetic Point Generation Algorithm
1. For each minority sample $x_i \in X_{\text{minority}}$, find its $k$-nearest minority neighbors using Euclidean distance.
2. Randomly select one neighbor $x_{zi}$.
3. Generate synthetic sample $x_{\text{new}}$:
$$x_{\text{new}} = x_i + \lambda (x_{zi} - x_i) \quad \text{where } \lambda \sim U(0, 1)$$

#### C. Critical Gotcha: `imblearn.pipeline.Pipeline` vs `sklearn.pipeline.Pipeline`
Standard `sklearn.pipeline.Pipeline` expects transformers to implement `.transform(X)`. It cannot modify the length of the target vector $y$.  
Attempting to put SMOTE inside a standard Scikit-Learn pipeline will raise an error or leak data. You **must** use `imblearn.pipeline.Pipeline`.

---

### 💻 3. Step-by-Step Python Code Implementation

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler

# 1. Create Heavily Imbalanced Synthetic Dataset (95% Class 0 / 5% Class 1)
X, y = make_classification(
    n_samples=2000, n_features=10, n_informative=8, n_redundant=2,
    weights=[0.95, 0.05], random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training Class Distribution -> Class 0: {np.sum(y_train == 0)} | Class 1: {np.sum(y_train == 1)}")

# 2. Baseline Model vs Class-Weighted Model
base_model = LogisticRegression(random_state=42).fit(X_train, y_train)
weighted_model = LogisticRegression(class_weight='balanced', random_state=42).fit(X_train, y_train)

print("\n--- ❌ Baseline Model (Imbalanced) ---")
print(classification_report(y_test, base_model.predict(X_test), target_names=['Majority', 'Minority']))

print("--- ⚖️ Class-Weighted Model ---")
print(classification_report(y_test, weighted_model.predict(X_test), target_names=['Majority', 'Minority']))

# 3. SMOTE + Tomek Links Hybrid Pipeline
# MUST use imblearn.pipeline.Pipeline!
imb_pipeline = ImbPipeline(steps=[
    ('scaler', StandardScaler()),
    ('resample', SMOTETomek(smote=SMOTE(k_neighbors=5, random_state=42), tomek=TomekLinks())),
    ('classifier', LogisticRegression(random_state=42))
])

imb_pipeline.fit(X_train, y_train)
preds_smote = imb_pipeline.predict(X_test)
probs_smote = imb_pipeline.predict_proba(X_test)[:, 1]

print("--- ✅ SMOTE + Tomek Links Pipeline ---")
print(classification_report(y_test, preds_smote, target_names=['Majority', 'Minority']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, probs_smote):.4f}")
```

---

### 🔍 4. Line-by-Line Syntax & Parameter Breakdown

- `make_classification(weights=[0.95, 0.05])`: Sets sample class ratios to 95% majority and 5% minority.
- `stratify=y`: Preserves the 95:5 class ratio identically in both `y_train` and `y_test`.
- `class_weight='balanced'`: Automatically calculates inverse class frequency loss multipliers.
- `SMOTE(k_neighbors=5)`: Number of nearest neighbors used to construct synthetic line segments.
- `TomekLinks()`: Identifies and removes overlapping majority instances.
- `ImbPipeline`: Imbalanced-learn wrapper pipeline that executes `fit_resample()` during `.fit()` and skips resampling during `.predict()`.

---

## 🎯 Day 052 — Threshold Tuning & Probability Calibration

### 💡 1. Intuition & Real-World Problem Analogy

#### Probability Calibration Intuition
Imagine a weather forecaster stating *"There is an 80% chance of rain today"*. If across 100 days where an 80% rain forecast was given, it actually rained on 80 of those days, the forecaster is **well-calibrated**.  
Tree ensembles (RandomForest, XGBoost) tend to yield uncalibrated probabilities clustered away from 0 and 1 because leaf votes represent fractions of sample splits, not true underlying probabilities.

#### Threshold Tuning Intuition
Standard Scikit-Learn `.predict()` uses an arbitrary fixed decision threshold $T = 0.5$.  
In medical diagnosis, a False Negative (missing a malignant tumor) is far worse than a False Positive (doing an unnecessary extra scan). We must tune the decision threshold $T$ to maximize business value or F1-Score.

---

### 📐 2. Mathematical Formulations

#### A. Brier Score Loss Formula
Measures mean squared error of predicted probabilities:
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
Lower values mean better calibrated probabilities. Perfect calibration yields $\text{BS} = 0$.

#### B. Calibration Methods (`CalibratedClassifierCV`)
1. **Platt Scaling (`method='sigmoid'`)**: Fits a logistic regression model on uncalibrated outputs $f(x)$:
   $$\hat{P}(y=1 \mid x) = \frac{1}{1 + \exp(A \cdot f(x) + B)}$$
2. **Isotonic Regression (`method='isotonic'`)**: Non-parametric monotonically increasing stepwise function. Requires more samples ($N > 1000$).

#### C. Threshold Optimization
Given precision $P(T)$ and recall $R(T)$ as functions of decision threshold $T$:
$$F_1(T) = \frac{2 \cdot P(T) \cdot R(T)}{P(T) + R(T)}$$
Optimal threshold $T^* = \arg\max_T F_1(T)$.

---

### 💻 3. Step-by-Step Python Code Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import precision_recall_curve, f1_score, brier_score_loss

# 1. Generate Dataset & Base Uncalibrated Model
X, y = make_classification(n_samples=2000, n_features=15, weights=[0.85, 0.15], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
prob_uncal = rf.predict_proba(X_test)[:, 1]

# 2. Probability Calibration via Isotonic Regression
calibrated_rf = CalibratedClassifierCV(estimator=rf, method='isotonic', cv='prefit')
calibrated_rf.fit(X_train, y_train)
prob_cal = calibrated_rf.predict_proba(X_test)[:, 1]

print(f"Uncalibrated Brier Score Loss: {brier_score_loss(y_test, prob_uncal):.4f}")
print(f"Calibrated Brier Score Loss:   {brier_score_loss(y_test, prob_cal):.4f}")

# 3. Decision Threshold Tuning for Maximum F1-Score
precisions, recalls, thresholds = precision_recall_curve(y_test, prob_cal)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\nDefault (0.50) Threshold F1-Score: {f1_score(y_test, (prob_cal >= 0.5).astype(int)):.4f}")
print(f"🎯 Optimal ({best_threshold:.4f}) Threshold F1-Score: {best_f1:.4f}")

# Perform inference with custom optimal threshold
final_preds = (prob_cal >= best_threshold).astype(int)
```

---

### 🔍 4. Line-by-Line Syntax & Parameter Breakdown

- `CalibratedClassifierCV(estimator=rf, method='isotonic', cv='prefit')`:
  - `cv='prefit'`: Signals that the underlying estimator `rf` has already been fitted on training data.
  - `method='isotonic'`: Uses non-parametric isotonic regression curve fitting.
- `brier_score_loss(y_true, prob_pred)`: Computes mean squared error between probabilities and binary targets.
- `precision_recall_curve(y_true, prob_pred)`: Returns precision, recall, and threshold arrays for every unique cutoff.
- `(prob_cal >= best_threshold).astype(int)`: Applies optimal custom decision threshold cutoff to generate binary predictions.

---

## 🔍 Day 053 — Model Interpretability & XAI (SHAP & LIME)

### 💡 1. Intuition & Real-World Problem Analogy

#### Why Explainable AI (XAI)?
In regulated industries (banking, healthcare), black-box machine learning predictions are prohibited unless accompanied by explanations.  
For example, if a loan application is rejected, European Union GDPR regulations mandate providing specific reasons (e.g. *"Low Credit Score contributed -0.30 to your score, High Debt Ratio contributed -0.25"*).

#### SHAP vs LIME
- **SHAP (SHapley Additive exPlanations)**: Based on cooperative game theory. Computes exact, fair contribution values for each feature across all possible feature combinations. Guaranteed to sum up to the total model output (Additive Property).
- **LIME (Local Interpretable Model-agnostic Explanations)**: Fits a local, interpretable linear surrogate model in the immediate neighborhood of a single prediction instance.

---

### 📐 2. Mathematical Foundations

#### Shapley Value Formula
The Shapley value $\phi_i$ for feature $i$ given total feature set $N$:
$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} \left( v(S \cup \{i\}) - v(S) \right)$$
Where $S$ is a subset of features excluding $i$, and $v(S)$ is the model output trained on subset $S$.

#### Additive Efficiency Property
The model prediction $\hat{f}(x)$ for any sample instance $x$ is equal to the expected base value $E[f(x)]$ plus the sum of all feature SHAP values:
$$\hat{f}(x) = E[f(x)] + \sum_{j=1}^M \phi_j(x)$$

---

### 💻 3. Step-by-Step Python Code Implementation

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

# 2. Compute SHAP Values using TreeExplainer
explainer = shap.TreeExplainer(model)
shap_explanation = explainer(X_test)
shap_values = shap_explanation.values
base_value = explainer.expected_value

print(f"Base Expected Model Output E[f(x)]: {base_value:.4f}")
print(f"SHAP Values Matrix Shape:            {shap_values.shape}")

# 3. Verify Additive Property for Sample Instance 0
sample_idx = 0
sample_prediction = model.predict_proba(X_test.iloc[[sample_idx]])[:, 1][0]
sum_shap = base_value + np.sum(shap_values[sample_idx])

print(f"\nSample 0 Model Prediction Score:  {sample_prediction:.4f}")
print(f"Base Value + Sum(SHAP Values):   {sum_shap:.4f} (Matches Additive Property!)")
```

---

### 🔍 4. Line-by-Line Syntax & Parameter Breakdown

- `shap.TreeExplainer(model)`: High-speed algorithm ($O(TLD^2)$ complexity) designed for tree ensembles (XGBoost, LightGBM, Random Forest) that computes exact Shapley values without full subset sampling.
- `explainer.expected_value`: Average output score of the model over the dataset ($E[f(x)]$).
- `shap_explanation.values`: Matrix of shape `(n_samples, n_features)` containing individual Shapley attribution values.

---

## 🚨 Day 054 — Anomaly & Outlier Detection (Isolation Forest, LOF, One-Class SVM)

### 💡 1. Intuition & Real-World Problem Analogy

#### Unsupervised Anomaly Detection Concept
In network intrusion detection, server metric monitoring, or manufacturing defect inspection, labeled anomaly examples are extremely rare or unknown. Unsupervised algorithms learn the structure of normal data and flag instances that lie in low-density regions.

#### Algorithm Comparisons:
- **Isolation Forest**: Anomalies are few and different. Random partitioning trees easily isolate anomalies close to the root node (short path length).
- **Local Outlier Factor (LOF)**: Measures local density relative to $k$-nearest neighbors. Highly effective when clusters have varying densities.
- **One-Class SVM**: Learns a tight boundary sphere surrounding normal training data in high-dimensional kernel space.

---

### 📐 2. Mathematical Formulation

#### Isolation Forest Score Formula
For sample $x$ and sample size $n$:
$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
Where $h(x)$ is path length (number of splits to isolate $x$) and $c(n)$ is average path length of unsuccessful search in a binary search tree:
$$c(n) = 2 \ln(n - 1) + 0.5772156649 \text{ (Euler's Constant)} - \frac{2(n - 1)}{n}$$
- If $E(h(x)) \to 0 \implies s \to 1$ (Instance is flagged as an anomaly).
- If $E(h(x)) \to c(n) \implies s \to 0.5$ (Normal instance).

---

### 💻 3. Step-by-Step Python Code Implementation

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
pred_iso = iso_forest.fit_predict(X)
scores_iso = iso_forest.decision_function(X)

print("--- 🌲 Isolation Forest Results ---")
print(classification_report(y_true, pred_iso, target_names=['Anomaly (-1)', 'Normal (1)']))
print(f"ROC-AUC (Anomaly Score): {roc_auc_score(y_true, -scores_iso):.4f}")

# 3. Local Outlier Factor (Novelty Detection Mode)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
lof.fit(X_normal)
pred_lof = lof.predict(X)

# 4. One-Class SVM
oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
oc_svm.fit(X_normal)
pred_svm = oc_svm.predict(X)

print(f"LOF Accuracy:            {np.mean(pred_lof == y_true):.4f}")
print(f"One-Class SVM Accuracy:  {np.mean(pred_svm == y_true):.4f}")
```

---

### 🔍 4. Line-by-Line Syntax & Parameter Breakdown

- `contamination=0.05`: The expected fraction of outliers in the dataset. Used to determine score decision thresholds.
- `.fit_predict(X)`: Fits the model and outputs labels (`1` for normal, `-1` for anomaly).
- `.decision_function(X)`: Computes raw anomaly scores. Lower/negative scores indicate anomalies.
- `novelty=True` in `LocalOutlierFactor`: Enables training on clean normal data (`.fit()`) and predicting on new test samples (`.predict()`).

---

## 🧪 Day 055 — Classical ML Comparison Lab & Benchmarking Harness

### 💡 1. Intuition & Real-World Problem Analogy
Senior ML Engineers never blindly pick an algorithm based on hype. They construct an automated comparison harness to evaluate models across multiple dimensions: Accuracy, F1-Score, ROC-AUC, Fit Time, and Inference Latency.

---

### 💻 2. Step-by-Step Python Code Implementation

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

# 1. Create Dataset
X, y = make_classification(
    n_samples=3000, n_features=25, n_informative=18, n_redundant=4,
    n_classes=2, weights=[0.7, 0.3], random_state=42
)

# 2. Define Model Suite Architecture
models = {
    'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))]),
    'SVM (RBF Kernel)': Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True))]),
    'Random Forest': RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=150, learning_rate=0.08, eval_metric='logloss', random_state=42, n_jobs=-1),
    'LightGBM': LGBMClassifier(n_estimators=150, learning_rate=0.08, random_state=42, verbose=-1, n_jobs=-1),
    'CatBoost': CatBoostClassifier(iterations=150, learning_rate=0.08, verbose=0, random_state=42)
}

# 3. Standardized Benchmarking Loop
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'f1', 'roc_auc']
results = []

for name, model in models.items():
    t0 = time.perf_counter()
    res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    t_elapsed = time.perf_counter() - t0
    
    results.append({
        'Model': name,
        'Accuracy': f"{np.mean(res['test_accuracy']):.4f} +/- {np.std(res['test_accuracy']):.4f}",
        'F1-Score': f"{np.mean(res['test_f1']):.4f} +/- {np.std(res['test_f1']):.4f}",
        'ROC-AUC': f"{np.mean(res['test_roc_auc']):.4f} +/- {np.std(res['test_roc_auc']):.4f}",
        'Fit Time (s)': f"{np.mean(res['fit_time']):.3f}s",
        'Total Benchmark Time': f"{t_elapsed:.2f}s"
    })

df_results = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
print("=== 📊 FINAL MODEL BENCHMARK RESULTS ===")
print(df_results.to_markdown(index=False))
```

---

## 📝 Day 056 — Advanced ML Assessment, Derivations & Interview Solutions

### 🧠 1. Mathematical Derivation: XGBoost Optimal Leaf Weight ($w^*$)

The regularized objective at step $t$ using 2nd Order Taylor expansion:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i w_{q(x_i)} + \frac{1}{2} h_i w_{q(x_i)}^2 \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
Group by leaf instance set $I_j = \{i \mid q(x_i) = j\}$:
$$\mathcal{L}^{(t)} = \sum_{j=1}^T \left[ \left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2} \left(\sum_{i \in I_j} h_i + \lambda\right) w_j^2 \right] + \gamma T$$
Let $G_j = \sum_{i \in I_j} g_i$ and $H_j = \sum_{i \in I_j} h_i$. Differentiating with respect to $w_j$ and setting to zero:
$$\frac{\partial \mathcal{L}}{\partial w_j} = G_j + (H_j + \lambda) w_j = 0 \implies w^*_j = -\frac{G_j}{H_j + \lambda}$$
Plugging $w^*_j$ back yields the Leaf Similarity Score:
$$S_j = -\frac{1}{2} \frac{G_j^2}{H_j + \lambda}$$

---

### 💻 2. Python Code Verification

```python
import numpy as np

def compute_xgboost_leaf(gradients, hessians, reg_lambda=1.0):
    G_j = np.sum(gradients)
    H_j = np.sum(hessians)
    w_star = - G_j / (H_j + reg_lambda)
    similarity = (G_j ** 2) / (H_j + reg_lambda)
    return w_star, similarity

# Residuals g_i and Hessians h_i
g_i = np.array([0.5, -1.2, 0.8, -0.4])
h_i = np.array([1.0, 1.0, 1.0, 1.0])

w_star, sim = compute_xgboost_leaf(g_i, h_i, reg_lambda=1.0)
print(f"Calculated Optimal Leaf Weight (w*): {w_star:.4f}")
print(f"Calculated Leaf Similarity Score:    {sim:.4f}")
```
