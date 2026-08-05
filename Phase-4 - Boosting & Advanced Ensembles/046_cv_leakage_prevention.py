"""
Day 046: Cross-Validation & Leakage Prevention
------------------------------------------------
Topics Covered:
  1. Data Leakage Types (Preprocessing Leakage & Target Leakage)
  2. The WRONG Way vs RIGHT Way (Preprocessing full dataset vs within-fold/Pipeline)
  3. Cross-Validation Strategies: K-Fold, StratifiedKFold, TimeSeriesSplit
  4. Pipeline-based Prevention using Scikit-Learn Pipeline
  5. Practical Leakage Detection & Best Practices Checklist
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# ==========================================
# 1. SETUP SYNTHETIC DATASETS
# ==========================================
def create_datasets():
    """Generates synthetic datasets for classification and time-series CV demonstrations."""
    # Imbalanced Classification Dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        weights=[0.85, 0.15],  # Imbalanced: 85% Class 0, 15% Class 1
        random_state=42
    )
    
    # Time-Series Synthetic Dataset
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    df_ts = pd.DataFrame({
        "date": dates,
        "feature_1": np.sin(np.linspace(0, 20, 100)) + np.random.normal(0, 0.1, 100),
        "target": np.cos(np.linspace(0, 20, 100)) + np.random.normal(0, 0.1, 100)
    })
    
    return X, y, df_ts


# ==========================================
# 2. DEMONSTRATE PREPROCESSING LEAKAGE
# ==========================================
def demonstrate_data_leakage(X, y):
    """
    Shows the dramatic difference in evaluation scores when leakage occurs
    vs when proper isolation is enforced using Scikit-Learn Pipeline.
    """
    print("\n" + "="*60)
    print("DEMO 1: PREPROCESSING DATA LEAKAGE (WRONG vs RIGHT WAY)")
    print("="*60)

    # --- THE WRONG WAY: Scaling full dataset BEFORE Cross-Validation ---
    scaler_leaky = StandardScaler()
    X_leaky = scaler_leaky.fit_transform(X)  # Scaler calculates mean & std of TEST data too!
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression()
    
    leaky_scores = cross_val_score(model, X_leaky, y, cv=kf, scoring='accuracy')
    print(f"❌ LEAKY CV Accuracy (Scaled BEFORE split): {leaky_scores.mean():.4f} +/- {leaky_scores.std():.4f}")
    print("   -> Problem: Test fold statistics leaked into training scaling parameters!")

    # --- THE RIGHT WAY: Scaling inside CV Folds via Pipeline ---
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])
    
    clean_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='accuracy')
    print(f"\n✅ CLEAN CV Accuracy (Scaled INSIDE folds via Pipeline): {clean_scores.mean():.4f} +/- {clean_scores.std():.4f}")
    print("   -> Correct: Pipeline fits scaler ONLY on train folds for each split.")


# ==========================================
# 3. COMPARE CV STRATEGIES
# ==========================================
def compare_cv_strategies(X, y, df_ts):
    """Compares K-Fold, StratifiedKFold, and TimeSeriesSplit behavior."""
    print("\n" + "="*60)
    print("DEMO 2: CROSS-VALIDATION STRATEGIES COMPARISON")
    print("="*60)

    n_splits = 5

    # A. Standard K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"\n1️⃣ K-Fold (n_splits={n_splits}):")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        train_y_mean = y[train_idx].mean()
        val_y_mean = y[val_idx].mean()
        print(f"   Fold {fold}: Train target mean = {train_y_mean:.3f} | Val target mean = {val_y_mean:.3f}")

    # B. Stratified K-Fold (Essential for Imbalanced Data)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"\n2️⃣ Stratified K-Fold (n_splits={n_splits}):")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        train_y_mean = y[train_idx].mean()
        val_y_mean = y[val_idx].mean()
        print(f"   Fold {fold}: Train target mean = {train_y_mean:.3f} | Val target mean = {val_y_mean:.3f}")
    print("   -> Advantage: Preserves exact class ratio (15% positive) across every fold.")

    # C. Time Series Split (Expanding Window)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    print(f"\n3️⃣ Time Series Split (n_splits={n_splits}, Expanding Window):")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df_ts), 1):
        print(f"   Fold {fold}: Train indices [{train_idx[0]}..{train_idx[-1]}] (n={len(train_idx)}) | Val indices [{val_idx[0]}..{val_idx[-1]}] (n={len(val_idx)})")
    print("   -> Advantage: Prevents temporal leakage by looking only into the past!")


# ==========================================
# 4. TARGET LEAKAGE DEMONSTRATION
# ==========================================
def demonstrate_target_leakage():
    """Shows how including future or duplicate target features distorts model accuracy."""
    print("\n" + "="*60)
    print("DEMO 3: TARGET LEAKAGE DETECTION")
    print("="*60)

    np.random.seed(42)
    n = 500
    
    # Real dataset: Patient features predicting hospital readmission (0 or 1)
    age = np.random.randint(20, 80, n)
    bp = np.random.normal(120, 15, n)
    readmitted = (age > 60).astype(int)  # Ground truth
    
    # LEAKY FEATURE: "discharge_summary_code" added AFTER readmission decision was made
    leaky_feature = readmitted * 2 + np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    
    df = pd.DataFrame({'age': age, 'bp': bp, 'discharge_code': leaky_feature, 'target': readmitted})
    
    # Feature correlations with target
    correlations = df.corr()['target'].drop('target')
    print("Correlation of features with target:")
    for feature, corr in correlations.items():
        flag = "🚨 POTENTIAL LEAKAGE!" if abs(corr) > 0.8 else "✅ Normal"
        print(f"   - {feature:15s}: Correlation = {corr:+.4f} ({flag})")
    
    print("\n💡 Key Rule: Never include features generated AFTER the prediction point in time!")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    X, y, df_ts = create_datasets()
    
    demonstrate_data_leakage(X, y)
    compare_cv_strategies(X, y, df_ts)
    demonstrate_target_leakage()
    
    print("\n" + "="*60)
    print("📋 DATA LEAKAGE PREVENTION CHECKLIST:")
    print("="*60)
    print("1. Always use sklearn Pipeline for scaling, encoding, and imputation.")
    print("2. Fit preprocessing ONLY on the training split, transform test split.")
    print("3. Use StratifiedKFold for imbalanced classification tasks.")
    print("4. Use TimeSeriesSplit for temporal/time-series data (never random KFold!).")
    print("5. Audit features for target leakage (correlations > 0.9 or post-event data).")
    print("="*60)
