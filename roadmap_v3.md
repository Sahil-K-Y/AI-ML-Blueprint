# 🎯 365-Day AI/ML Engineer Master Roadmap

**Sahil Kumar (Sky)** | BTech CSE (AI/ML), DAV University, Jalandhar
**Goal:** Entry-level AI/ML Engineer role

---

## 📍 Phase Index

| Phase | Directory | Days | Focus Area |
|---|---|---|---|
| Phase 0 | Phase-0 - EDA & Feature Engineering | Foundations | Data Cleaning, EDA (Univariate/Bivariate/Multivariate) & Feature Engineering |
| Phase 1 | Phase-1 - Linear Regression & Regularization | 001–013 | Cost MSE, Gradient Descent, Linear/Ridge/Lasso/ElasticNet, SGD, House Price Project |
| Phase 2 | Phase-2 - Logistic Regression & Classification | 014–021 | Sigmoid, Softmax, ROC-AUC, PR-AUC, Regularization, Breast Cancer Project |
| Phase 3 | Phase-3 - Tree Models & SVM | 022–038 | Decision Trees, SVM Kernels, Naive Bayes, KNN, Ensembles, Random Forest |
| Phase 4 | Phase-4 - Boosting & Advanced Ensembles | 039–056 | AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost, Optuna, Feature Selection |
| Phase 5 | Phase-5 - Unsupervised Learning | 057–077 | Clustering, PCA, Time Series & E2E Tabular ML Production System |
| Phase 6 | Phase-6 - Deep Learning & PyTorch | 078–122 | Neural Networks, PyTorch, Backpropagation, WandB, Docker, FastAPI, pytest, PyTorch Pipeline Project |
| Phase 7 | Phase-7 - Computer Vision & Multimodal AI | 123–164 | CNNs, ResNet, YOLOv8, U-Net, Grad-CAM, Computer Vision System Project |
| Phase 8 | Phase-8 - Sequence Models & Recommender Systems | 165–189 | RNNs, LSTMs, GRUs, Attention Mechanism, Recommender Systems (CF, Matrix Factorization, Hybrid, NDCG) |
| Phase 9 | Phase-9 - NLP & Transformer Ecosystem | 190–238 | BERT, RoBERTa, GPT, Hugging Face Fine-Tuning, Transformer NLP App Project |
| Phase 10 | Phase-10 - MLOps & Data Engineering | 239–267 | Docker, MLflow, DVC, FastAPI, CI/CD, Drift Monitoring, A/B Testing, MLOps Project |
| Phase 11 | Phase-11 - Generative AI & RAG | 268–316 | LLMs, Vector DBs, RAG, Hybrid Search, LoRA/QLoRA, RAG System Project |
| Phase 12 | Phase-12 - Agentic AI Systems | 317–344 | LangGraph, Multi-Agent Supervisors, HITL, MCP, Agentic AI System Project |
| Phase 13 | Phase-13 - Reinforcement Learning | 345–358 | MDP, Bellman Equations, Q-Learning, DQN, PPO |
| Phase 14 | Phase-14 - LLMOps & Capstone | 359–374 | Observability, Guardrails, Responsible AI, Enterprise AI System Capstone Project |

---

## 🏆 Mandatory Portfolio Projects

| # | Day | Project | Core Stack |
|---|---|---|---|
| 1 |  069–076  | End-to-End Tabular ML System | Sklearn Pipeline → Optuna → FastAPI → Streamlit → Docker |
| 2 |  114–120  | Deep Learning PyTorch Pipeline | PyTorch → WandB → FastAPI → pytest → Docker |
| 3 |  153–162  | Computer Vision System | YOLOv8/U-Net → Grad-CAM → FastAPI → ONNX → Docker |
| 4 |  226–233  | Transformer NLP Application | Fine-tuned BERT/RoBERTa → FastAPI → Docker |
| 5 |  262–266  | Production MLOps Pipeline | MLflow → DVC → Docker → CI/CD → Drift Monitoring |
| 6 |  307–315  | Advanced Production RAG System | Hybrid Search, Reranker, RAGAS, Langfuse, Docker |
| 7 |  337–343  | Production Agentic AI System | LangGraph, Tools, HITL, MCP, Docker |
| 8 |  368–371  | Flagship Capstone — Enterprise AI System | Multi-Agent + RAG + MLOps + Guardrails + Cloud |

---

## 📅 Day-by-Day Execution Schedule

---

# 🟢 PHASE 0: EDA & Feature Engineering
> **Folder:** `Phase-0 - EDA & Feature Engineering`
> **Focus:** Data Cleaning, Univariate/Bivariate/Multivariate Analysis, Outlier Detection, Feature Encoding, Scaling, Skewness Handling, and Preprocessing Pipelines (16 Core Foundations Notebooks).

---

# 🟢 PHASE 1: Linear Regression & Regularization (Days 001–013)
> **Folder:** `Phase-1 - Linear Regression & Regularization`

---

### Day 001 — Cost Function MSE
- **Tasks (3-4h):** Residual calculations, cost curves, error definitions.
- **Output:** ``001 - Cost Function MSE.ipynb`
- **Topics:**
  - Residual (Error) — difference between actual and predicted values
  - Mean Squared Error (MSE) — mathematical formula and properties
  - Cost function curve — plotting MSE vs parameter values
  - Why squaring errors? — penalizing large errors, mathematical convenience
  - Error definitions — absolute error, squared error, mean error variants

### Day 002 — Gradient Descent
- **Tasks (3-4h):** Learning rate dynamics, convergence batch math, updates.
- **Output:** `002 - Gradient Descent.ipynb`
- **Topics:**
  - Gradient Descent intuition — walking downhill on cost surface
  - Learning rate (α) — step size control, too large vs too small
  - Convergence criteria — when to stop iterating
  - Batch Gradient Descent — full dataset per update step
  - Parameter update rule — θ = θ - α * ∂J/∂θ

### Day 003 — Linear Regression Implementation
- **Tasks (3-4h):** Fitting & predicting coefficients, regression lines.
- **Output:** `003 - Linear Regression Implementation.ipynb`
- **Topics:**
  - Simple Linear Regression — y = mx + b fitting
  - Normal Equation — closed-form solution (θ = (X^T X)^(-1) X^T y)
  - Coefficient interpretation — slope and intercept meaning
  - Regression line plotting — best fit visualization
  - Prediction using learned coefficients

### Day 004 — Model Evaluation
- **Tasks (3-4h):** R2 score, Adjusted R2, MAE, MSE, RMSE math.
- **Output:** `004 - Model Evaluation.ipynb`
- **Topics:**
  - R² Score — coefficient of determination (explained variance ratio)
  - Adjusted R² — penalizing extra features that don't improve model
  - Mean Absolute Error (MAE) — average absolute residual
  - Mean Squared Error (MSE) — average squared residual
  - Root Mean Squared Error (RMSE) — interpretable error in original units

### Day 005 — Multiple Linear Regression
- **Tasks (3-4h):** Independent variables selection, OLS assumptions checks.
- **Output:** `005 - Multiple Linear Regression.ipynb`
- **Topics:**
  - Multiple independent variables — extending to higher dimensions
  - Feature selection for regression — choosing relevant predictors
  - Ordinary Least Squares (OLS) assumptions — linearity, independence, homoscedasticity, normality
  - Multicollinearity detection — VIF (Variance Inflation Factor)
  - Matrix notation for multiple regression — Y = Xβ + ε

### Day 006 — Polynomial Regression
- **Tasks (3-4h):** PolynomialFeatures, non-linear fits, overfitting checks.
- **Output:** `006 - Polynomial Regression.ipynb`
- **Topics:**
  - PolynomialFeatures — generating x², x³, interaction terms
  - Non-linear curve fitting — capturing curved relationships
  - Degree selection — choosing polynomial order
  - Overfitting detection — high-degree polynomials fitting noise
  - Train vs test performance comparison for polynomial models

### Day 007 — Ridge & Lasso Regression
- **Tasks (3-4h):** L1/L2 regularization penalties, alpha shrinkage tuning.
- **Output:** `007 - Ridge & Lasso Regression.ipynb`
- **Topics:**
  - L2 Regularization (Ridge) — penalty term λΣβ² shrinking coefficients
  - L1 Regularization (Lasso) — penalty term λΣ|β| producing sparse models
  - Alpha (λ) hyperparameter — controlling regularization strength
  - Coefficient shrinkage paths — plotting coefficients vs alpha
  - Ridge vs Lasso comparison — when to use which

### Day 008 — Elastic Net Regression
- **Tasks (3-4h):** Combined L1/L2 penalty parameters, l1_ratio tuning.
- **Output:** `008 - Elastic Net Regression.ipynb`
- **Topics:**
  - Elastic Net — combining L1 and L2 penalties (α * L1 + (1-α) * L2)
  - l1_ratio parameter — balance between Lasso and Ridge
  - When to use Elastic Net — correlated features, grouped selection
  - Hyperparameter tuning — optimizing alpha and l1_ratio together
  - Comparison with pure Ridge and Lasso performance

### Day 009 — SGD Regressor
- **Tasks (3-4h):** Stochastic Gradient Descent solver, large scale datasets.
- **Output:** `009 - SGD Regressor.ipynb`
- **Topics:**
  - Stochastic Gradient Descent (SGD) — single sample updates
  - SGDRegressor in Scikit-Learn — online learning solver
  - Large-scale dataset handling — memory-efficient training
  - Partial fit for streaming data — incremental learning
  - Convergence behavior — noisier but faster than batch GD

### Day 010 — App Planning
- **Tasks (3-4h):** UI requirements specifications, serialization formats.
- **Output:** `010 - App Planning.ipynb`
- **Topics:**
  - Application UI planning — wireframing prediction interface
  - Requirements specification — inputs, outputs, user flow
  - Model serialization formats — pickle, joblib for saving models
  - API endpoint design — prediction request/response structure
  - Deployment architecture planning — local vs cloud serving

### Day 011 — House Price Prediction Project
- **Tasks (3-4h):** Ingestion pipeline, EDA, categorical encoding, models tuning.
- **Output:** `011 - House Price Prediction Project.ipynb`
- **Topics:**
  - Data ingestion pipeline — loading and parsing raw housing data
  - Exploratory Data Analysis (EDA) — distributions, correlations, missing values
  - Categorical encoding — Label Encoding, One-Hot Encoding
  - Model training — fitting multiple regression variants
  - Hyperparameter tuning — optimizing model parameters

### Day 012 — Simple UI Integration
- **Tasks (3-4h):** Interactive sliders, input fields, prediction callbacks.
- **Output:** `012 - Simple UI Integration.ipynb`
- **Topics:**
  - Interactive UI components — sliders, input fields for features
  - Prediction callbacks — triggering model inference on user input
  - Real-time output display — showing predicted house price
  - Input validation — handling edge cases in user inputs
  - UI-model integration — connecting frontend to trained model

### Day 013 — Regression Mini Project Polish
- **Tasks (3-4h):** Final pipeline integration, styling forms, deployment tests.
- **Output:** `013 - Regression Mini Project Polish.ipynb`
- **Topics:**
  - Final pipeline integration — end-to-end data flow
  - UI styling and polishing — clean, professional interface
  - Form validation and error handling
  - Deployment testing — local server verification
  - Project documentation and README

---

# 🟢 PHASE 2: Logistic Regression & Classification (Days 014–021)
> **Folder:** Phase-2 - Logistic Regression & Classification

---

### Day 014 — Logistic Regression Intuition & Sigmoid
- **Tasks (3-4h):** Sigmoid function mapping, odds ratio, Log Loss formulation.
- **Output:** `014 - Logistic Regression Intuition & Sigmoid.ipynb`
- **Topics:**
  - Sigmoid function — σ(z) = 1/(1 + e^(-z)) mapping to [0,1]
  - Odds ratio — P(event) / P(not event)
  - Log-odds (Logit) — log(P/(1-P)) = β₀ + β₁x
  - Log Loss (Binary Cross-Entropy) — -[y·log(p) + (1-y)·log(1-p)]
  - Decision boundary — threshold for binary classification

### Day 015 — Logistic Regression Implementation & Metrics
- **Tasks (3-4h):** Model training, Accuracy, Precision, Recall, F1-score.
- **Output:** `015 - Logistic Regression Implementation & Metrics.ipynb`
- **Topics:**
  - Logistic Regression model training — fitting on binary target
  - Accuracy — (TP + TN) / Total predictions
  - Precision — TP / (TP + FP), positive predictive value
  - Recall (Sensitivity) — TP / (TP + FN), true positive rate
  - F1-Score — harmonic mean of Precision and Recall

### Day 016 — Multiclass Classification (OvR vs OvO)
- **Tasks (3-4h):** One-vs-Rest (OvR), One-vs-One (OvO) decision boundary maps.
- **Output:** `016 - Multiclass Classification (OvR vs OvO).ipynb`
- **Topics:**
  - One-vs-Rest (OvR) — training K binary classifiers for K classes
  - One-vs-One (OvO) — training K(K-1)/2 pairwise classifiers
  - Decision boundary visualization — plotting multi-class regions
  - Voting mechanisms — aggregating binary classifier predictions
  - Scikit-Learn multi_class parameter configuration

### Day 017 — Softmax Regression
- **Tasks (3-4h):** Softmax formula, multiclass probabilities, cross-entropy.
- **Output:** `017 - Softmax Regression.ipynb`
- **Topics:**
  - Softmax function — exp(zᵢ) / Σexp(zⱼ) for multi-class probabilities
  - Multi-class probability distribution — outputs summing to 1
  - Cross-Entropy Loss — -Σyᵢ·log(pᵢ) for K classes
  - Softmax regression vs OvR comparison
  - Temperature scaling in softmax — sharpening/smoothing probabilities

### Day 018 — ROC-AUC & PR-AUC Curves
- **Tasks (3-4h):** True Positive vs False Positive rates curves, thresholds sweeps.
- **Output:** `018 - ROC-AUC & PR-AUC Curves.ipynb`
- **Topics:**
  - ROC Curve — True Positive Rate vs False Positive Rate at all thresholds
  - AUC (Area Under ROC Curve) — aggregate classifier performance measure
  - Precision-Recall (PR) Curve — for imbalanced datasets
  - Threshold sweep analysis — impact of classification threshold
  - ROC-AUC vs PR-AUC — when to prefer which metric

### Day 019 — Multiclass Pipeline & Evaluation
- **Tasks (3-4h):** Preprocessing pipelines, scaling inputs, classification reports.
- **Output:** `019 - Multiclass Pipeline & Evaluation.ipynb`
- **Topics:**
  - Sklearn Pipeline — chaining preprocessing + model steps
  - StandardScaler / MinMaxScaler — input feature scaling
  - Classification report — per-class precision, recall, F1
  - Cross-validation within pipeline — preventing data leakage
  - Multi-class evaluation — macro, micro, weighted averaging

### Day 020 — Regularization in Logistic Regression
- **Tasks (3-4h):** C parameters sweeps, L1/L2 penalties shrinkage.
- **Output:** `020 - Regularization in Logistic Regression.ipynb`
- **Topics:**
  - C parameter in Logistic Regression — inverse regularization strength (C = 1/λ)
  - L1 penalty (penalty='l1') — sparse feature selection
  - L2 penalty (penalty='l2') — coefficient shrinkage
  - Regularization strength sweep — plotting accuracy vs C values
  - Feature importance via L1 coefficients — zero-weight features

### Day 021 — Breast Cancer Classification Project
- **Tasks (3-4h):** Model training pipeline, StratifiedKFold validation, metrics logs.
- **Output:** `021 - Breast Cancer Classification Project.ipynb`
- **Topics:**
  - Breast cancer dataset — loading and exploring sklearn.datasets
  - Model training pipeline — preprocessing + classification
  - StratifiedKFold validation — preserving class distribution across folds
  - Metrics logging — tracking precision, recall, F1 per fold
  - Best model selection — comparing cross-validation scores

---

# 🟢 PHASE 3: Tree Models & SVM (Days 022–038)
> **Folder:** Phase-3 - Tree Models & SVM

---

### Day 022 — Decision Tree Intuition
- **Tasks (3-4h):** Splitting metrics: Gini Impurity, Shannon Entropy formulas.
- **Output:** `022 - Decision Tree Intuition.ipynb`
- **Topics:**
  - Decision Tree splitting — recursive binary partitioning
  - Gini Impurity — 1 - Σpᵢ², measuring node impurity
  - Shannon Entropy — -Σpᵢ·log₂(pᵢ), information content
  - Information Gain — parent impurity minus weighted child impurity
  - Greedy split selection — choosing feature and threshold maximizing gain

### Day 023 — Decision Tree Implementation
- **Tasks (3-4h):** DecisionTreeClassifier tuning, tree depth limits.
- **Output:** `023 - Decision Tree Implementation.ipynb`
- **Topics:**
  - DecisionTreeClassifier — Scikit-Learn implementation
  - max_depth parameter — controlling tree depth
  - min_samples_split / min_samples_leaf — leaf size constraints
  - Tree visualization — plotting decision tree structure
  - Feature importance from tree — Gini-based importance scores

### Day 024 — Decision Tree Overfitting & Pruning
- **Tasks (3-4h):** Cost Complexity Pruning (ccp_alpha), validation curves.
- **Output:** `024 - Decision Tree Pruning.ipynb`
- **Topics:**
  - Overfitting in decision trees — memorizing training noise
  - Pre-pruning — max_depth, min_samples_split, max_leaf_nodes
  - Post-pruning (Cost Complexity) — ccp_alpha parameter
  - Validation curves — plotting accuracy vs ccp_alpha
  - Optimal tree selection — balancing complexity and accuracy

### Day 025 — Decision Tree Regressor
- **Tasks (3-4h):** Continuous target splits, variance reduction metrics.
- **Output:** `025 - Decision Tree Regressor.ipynb`
- **Topics:**
  - Decision Tree for regression — predicting continuous targets
  - Variance reduction — splitting criterion for regression trees
  - Mean prediction at leaf nodes — averaging target values
  - MSE-based splitting — minimizing squared error at splits
  - Comparison with linear regression — piecewise constant vs linear

### Day 026 — SVM Intuition
- **Tasks (3-4h):** Margin optimization, support vectors coordinates, slack parameters.
- **Output:** `026 - SVM Intuition.ipynb`
- **Topics:**
  - Maximum margin classifier — finding widest separating hyperplane
  - Support Vectors — data points closest to decision boundary
  - Margin optimization — maximizing 2/||w|| subject to constraints
  - Slack variables (ξ) — allowing soft margin misclassifications
  - C parameter — tradeoff between margin width and violations

### Day 027 — SVM Kernels
- **Tasks (3-4h):** Linear, Polynomial, Radial Basis Function (RBF) tricks.
- **Output:** `027 - SVM Kernels.ipynb`
- **Topics:**
  - Linear Kernel — K(x,y) = x^T y, linearly separable data
  - Polynomial Kernel — K(x,y) = (γx^T y + r)^d, curved boundaries
  - RBF (Radial Basis Function) Kernel — K(x,y) = exp(-γ||x-y||²), non-linear mapping
  - Kernel trick — implicit high-dimensional feature mapping
  - Kernel selection guidelines — when to use which kernel

### Day 028 — SVM Tuning
- **Tasks (3-4h):** Grid search on C and Gamma regularization parameters.
- **Output:** `028 - SVM Tuning.ipynb`
- **Topics:**
  - Grid Search over C and gamma — exhaustive hyperparameter search
  - C parameter tuning — regularization strength control
  - Gamma parameter tuning — RBF kernel width control
  - Cross-validated grid search — GridSearchCV in sklearn
  - Best parameter combination — balancing bias-variance

### Day 029 — Naive Bayes
- **Tasks (3-4h):** Bayes theorem, conditional independence, Laplace smoothing.
- **Output:** `029 - Naive Bayes.ipynb`
- **Topics:**
  - Bayes' Theorem — P(A|B) = P(B|A)·P(A) / P(B)
  - Conditional independence assumption — features independent given class
  - Gaussian Naive Bayes — continuous features with normal distribution
  - Multinomial Naive Bayes — text/count data classification
  - Laplace smoothing (α) — handling zero-frequency problem

### Day 030 — KNN Basics
- **Tasks (3-4h):** Euclidean, Manhattan distance metrics, neighbor size select.
- **Output:** `030 - KNN Basics.ipynb`
- **Topics:**
  - K-Nearest Neighbors concept — classify by majority vote of neighbors
  - Euclidean distance — ||x - y||₂ = √Σ(xᵢ - yᵢ)²
  - Manhattan distance — ||x - y||₁ = Σ|xᵢ - yᵢ|
  - K value selection — odd K for binary, impact on decision boundary
  - Distance-weighted voting — closer neighbors have more influence

### Day 031 — KNN Implementation
- **Tasks (3-4h):** Feature scaling verification, predictions boundary plots.
- **Output:** `031 - KNN Implementation.ipynb`
- **Topics:**
  - KNN implementation with Scikit-Learn — KNeighborsClassifier
  - Feature scaling requirement — StandardScaler before KNN
  - Decision boundary plotting — visualizing K impact on boundaries
  - K-value optimization — cross-validation for best K
  - KNN limitations — curse of dimensionality, slow prediction

### Day 032 — Model Leaderboard
- **Tasks (3-4h):** Accuracy, precision, recall comparison benchmark logs.
- **Output:** `032 - Model Leaderboard.ipynb`
- **Topics:**
  - Model comparison benchmark — testing all classifiers on same data
  - Accuracy comparison table — ranking models by accuracy
  - Precision and Recall comparison — per-model metrics
  - Training time vs accuracy tradeoff
  - Best model selection — choosing for deployment

### Day 033 — Classification Project
- **Tasks (3-4h):** Baseline candidate models evaluations pipeline.
- **Output:** `033 - Classification Project.ipynb`
- **Topics:**
  - End-to-end classification pipeline — data to evaluation
  - Baseline model establishment — simple model as reference
  - Multiple candidate model evaluation — comparing algorithms
  - Cross-validation scoring — robust performance estimation
  - Final model selection and justification

### Day 034 — Ensemble Methods
- **Tasks (3-4h):** VotingClassifiers, BaggingClassifiers, Out-of-Bag (OOB) scores.
- **Output:** `034 - Voting Bagging Ensembles.ipynb`
- **Topics:**
  - VotingClassifier — hard voting (majority) vs soft voting (probability average)
  - BaggingClassifier — bootstrap aggregation of base estimators
  - Out-of-Bag (OOB) score — validation without held-out set
  - Ensemble diversity — why combining models reduces variance
  - Scikit-Learn ensemble API — parameter configuration

### Day 035 — Random Forest Basics
- **Tasks (3-4h):** Bootstrap aggregation, random subspace features allocations.
- **Output:** `035 - Random Forest Basics.ipynb`
- **Topics:**
  - Random Forest — bagging + random feature subspace selection
  - Bootstrap sampling — sampling with replacement for each tree
  - Random subspace method — selecting √p features per split
  - Variance reduction through averaging — ensemble smoothing
  - Random Forest vs single Decision Tree comparison

### Day 036 — Random Forest Tuning
- **Tasks (3-4h):** Optuna search over trees count, depth, split thresholds.
- **Output:** `036 - Random Forest Tuning.ipynb`
- **Topics:**
  - Optuna hyperparameter optimization — Bayesian search framework
  - n_estimators tuning — number of trees in forest
  - max_depth and max_features tuning — tree complexity control
  - min_samples_split / min_samples_leaf — split threshold optimization
  - Cross-validated objective function — Optuna trial evaluation

### Day 037 — Feature Importance Analysis
- **Tasks (3-4h):** Mean Decrease in Impurity (MDI) vs Permutation importance.
- **Output:** `037 - Feature Importance.ipynb`
- **Topics:**
  - Mean Decrease in Impurity (MDI) — Gini importance from tree splits
  - Permutation Importance — accuracy drop when feature is shuffled
  - MDI vs Permutation comparison — bias in MDI for high-cardinality features
  - Feature importance visualization — bar plots of importance scores
  - Feature selection using importance — removing low-importance features

### Day 038 — Trees SVM Assessment
- **Tasks (3-4h):** Conceptual test review questions answers logs.
- **Output:** `038 - Trees SVM Assessment.ipynb`
- **Topics:**
  - Conceptual review — Decision Trees, SVM, ensemble theory
  - Technical question-answer format — interview preparation
  - Mathematical concepts review — Gini, Entropy, margins, kernels
  - Algorithm comparison — strengths and weaknesses of each model
  - Edge case understanding — when models fail and why

---

# 🟢 PHASE 4: Boosting & Advanced Ensembles (Days 039–056)
> **Folder:** Phase-4 - Boosting & Advanced Ensembles

---

### Day 039 — AdaBoost - Adaptive Boosting
- **Tasks (3-4h):** Exponential loss optimization, weak learner coefficients, sample weight updates.
- **Output:** `039 - AdaBoost.ipynb`
- **Topics:**
  - AdaBoost algorithm — Adaptive Boosting sequential learning
  - Exponential loss optimization — AdaBoost's loss function
  - Weak learner coefficients (αₜ) — classifier weight calculation
  - Sample weight updates — increasing weight on misclassified samples
  - Decision stumps — depth-1 trees as weak learners

### Day 040 — Gradient Boosting Intuition & Impl
- **Tasks (3-4h):** Residual fitting math, loss functions (MSE, LogLoss), step-by-step custom implementation in Python
- **Output:** `040_gradient_boosting.ipynb`
- **Topics:**
  - Gradient Boosting intuition — sequentially fitting residuals
  - Residual fitting math — each tree predicts errors of previous ensemble
  - MSE loss function for gradient boosting — gradient = -(y - ŷ)
  - LogLoss for classification boosting — gradient = p - y
  - Step-by-step custom Gradient Boosting implementation in Python

### Day 041 — AdaBoost Deep Dive
- **Tasks (3-4h):** Decision stumps, sample weighting updates, learning rate shrinkage math & Scikit-Learn lab
- **Output:** `041_adaboost_deepdive.ipynb`
- **Topics:**
  - AdaBoost deep dive — mathematical derivation of weight updates
  - Decision stumps — single split weak learners
  - Sample weighting updates — exponential reweighting formula
  - Learning rate shrinkage — controlling ensemble contribution
  - Scikit-Learn AdaBoostClassifier lab — hands-on parameter tuning

### Day 042 — XGBoost Architecture & Practice
- **Tasks (3-4h):** Exact vs approximate greedy algorithm, regularized objective, tree pruning, xgboost API
- **Output:** `042_xgboost_practice.ipynb`
- **Topics:**
  - XGBoost vs Standard Gradient Boosting (GBM) — key differences (2nd order optimization, regularization, split finding, missing value handling, parallelization)
  - Example Dataset Setup — 5-sample housing price data for manual walkthrough
  - Regularized Objective Function & 2nd Order Taylor Expansion — full math (gradient g_i, hessian h_i, MSE & LogLoss derivatives)
  - Optimal Leaf Weight (w*) & Similarity Score — deriving w* = -G/(H+λ), Similarity = G²/(H+λ), Split Gain formula
  - Custom Split Finding — evaluating candidate thresholds, computing Gain with L2 regularization (λ=1.0)
  - Exact vs Approximate Greedy Algorithm — quantile sketch, weighted quantile (hessian-based), histogram method (tree_method='hist')
  - Tree Pruning with γ (gamma) — minimum gain threshold, branch pruning logic
  - XGBoost Scikit-Learn API — XGBRegressor with hyperparameters (learning_rate, max_depth, gamma, reg_alpha, reg_lambda)
  - XGBoost Native DMatrix API — binary classification, watchlists, eval metrics, early stopping
  - Feature Importance & Tree Visualization — plot_importance (weight-based)

### Day 043 — LightGBM Implementation
- **Tasks (3-4h):** Leaf-wise vs level-wise tree growth, histogram-based splits, handling categorical features natively
- **Output:** `043_lightgbm_practice.ipynb`
- **Topics:**
  - LightGBM vs XGBoost — key architectural differences
  - Leaf-wise tree growth — best-first split strategy (vs level-wise)
  - Histogram-based split finding — binning continuous features for speed
  - Gradient-based One-Side Sampling (GOSS) — prioritizing high-gradient samples
  - Exclusive Feature Bundling (EFB) — reducing feature dimensionality
  - Native categorical feature handling — optimal split without one-hot encoding
  - LightGBM Python API — LGBMClassifier / LGBMRegressor, Dataset, train()
  - num_leaves parameter — controlling tree complexity in leaf-wise growth

### Day 044 — CatBoost & Categorical Benchmark
- **Tasks (3-4h):** Symmetric trees, ordered boosting, target encoding without target leakage, benchmarking ensemble models
- **Output:** `044_catboost_benchmark.ipynb`
- **Topics:**
  - CatBoost architecture — symmetric (oblivious) decision trees
  - Ordered Boosting — permutation-based training to reduce prediction shift
  - Target encoding without leakage — ordered target statistics
  - Symmetric tree structure — all leaves at same depth, same split features
  - CatBoost native categorical handling — no manual encoding needed
  - Benchmarking XGBoost vs LightGBM vs CatBoost — accuracy, speed, memory
  - CatBoost Python API — CatBoostClassifier / CatBoostRegressor, Pool()

### Day 045 — Hyperparameter Tuning Frameworks
- **Tasks (3-4h):** GridSearch, RandomizedSearch, and writing an Optuna automated search study for boosting models
- **Output:** `045_optuna_tuning.ipynb`
- **Topics:**
  - GridSearchCV — exhaustive search over parameter grid
  - RandomizedSearchCV — random sampling from parameter distributions
  - Optuna framework — Bayesian optimization with Tree-structured Parzen Estimator (TPE)
  - Optuna Study creation — define objective function, create_study(), optimize()
  - Optuna trial suggest API — suggest_float, suggest_int, suggest_categorical
  - Pruning unpromising trials — MedianPruner for early stopping bad configs
  - Visualization — Optuna plot_optimization_history, plot_param_importances

### Day 046 — Cross-Validation & Leakage Prevention
- **Tasks (3-4h):** K-Fold, StratifiedKFold, TimeSeriesSplit, detecting & preventing data leakage during preprocessing
- **Output:** `046_cv_leakage_prevention.ipynb`
- **Topics:**
  - K-Fold Cross-Validation — splitting data into K equal folds
  - StratifiedKFold — preserving class distribution in each fold
  - TimeSeriesSplit — expanding window for temporal data
  - Data leakage types — target leakage, train-test contamination
  - Preprocessing leakage — fitting scaler on full data before split
  - Pipeline-based prevention — sklearn Pipeline ensuring correct fit/transform order
  - Leakage detection checklist — identifying common leakage patterns

### Day 047 — Bias-Variance Tradeoff & Learning Curves
- **Tasks (3-4h):** Plotting training vs validation error curves, diagnosing underfitting vs overfitting, mitigation strategies
- **Output:** `047_learning_curves.ipynb`
- **Topics:**
  - Bias-Variance Tradeoff — decomposing total error = bias² + variance + noise
  - Learning curves — plotting train vs validation error vs training size
  - Diagnosing underfitting — high bias, both errors high
  - Diagnosing overfitting — high variance, gap between train and validation
  - Mitigation strategies — more data, regularization, simpler model, ensembles
  - Validation curves — plotting error vs hyperparameter values

### Day 048 — Advanced Feature Engineering
- **Tasks (3-4h):** Missing value imputation, outlier detection (IQR, Z-score), One-Hot/Target encoding, Power Transformers
- **Output:** `048_feature_engineering.ipynb`
- **Topics:**
  - Missing value imputation — SimpleImputer (mean/median/mode), KNNImputer
  - Outlier detection — IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
  - Outlier detection — Z-score method (|z| > 3)
  - One-Hot Encoding — creating binary dummy variables
  - Target Encoding — replacing categories with target mean
  - Power Transformers — Box-Cox and Yeo-Johnson for normalizing skewed distributions
  - Feature scaling — StandardScaler, MinMaxScaler, RobustScaler

### Day 049 — Feature Selection Techniques
- **Tasks (3-4h):** Filter methods (Correlation), Wrapper methods (RFE/RFECV), Embedded methods (L1/Tree importance)
- **Output:** `049_feature_selection.ipynb`
- **Topics:**
  - Filter methods — Pearson Correlation, mutual information, chi-squared test
  - Wrapper methods — Recursive Feature Elimination (RFE/RFECV)
  - Embedded methods — L1 (Lasso) coefficient-based selection
  - Tree-based importance — feature importance from Random Forest/XGBoost
  - SelectKBest and SelectFromModel — sklearn feature selection utilities
  - Feature selection pipeline — integrating selection into sklearn Pipeline

### Day 050 — Production Sklearn Pipelines
- **Tasks (3-4h):** Building custom Transformers, ColumnTransformer, Pipeline, saving & loading via joblib/pickle
- **Output:** `050_sklearn_pipelines.py`
- **Topics:**
  - Custom Transformer — extending sklearn BaseEstimator and TransformerMixin
  - ColumnTransformer — applying different preprocessing to different columns
  - Pipeline — chaining preprocessing steps and model
  - FunctionTransformer — wrapping custom functions as transformers
  - Model serialization — saving pipeline with joblib and pickle
  - Loading and inference — loading saved pipeline for production predictions

### Day 051 — Imbalanced Learning Methods
- **Tasks (3-4h):** Class weighting (class_weight='balanced'), SMOTE, ADASYN, Tomek Links using imbalanced-learn
- **Output:** `051_imbalanced_learning.ipynb`
- **Topics:**
  - Class imbalance problem — skewed class distributions affecting model
  - class_weight='balanced' — adjusting loss function weights inversely proportional to class frequency
  - SMOTE (Synthetic Minority Over-sampling) — generating synthetic minority samples
  - ADASYN — Adaptive Synthetic Sampling focusing on harder examples
  - Tomek Links — cleaning overlapping majority samples near boundary
  - imbalanced-learn library — Pipeline integration with over/under-sampling

### Day 052 — Threshold Tuning & Calibration
- **Tasks (3-4h):** Probability calibration (CalibratedClassifierCV), Reliability curves, custom decision threshold optimization
- **Output:** `052_calibration_thresholds.ipynb`
- **Topics:**
  - Probability calibration — aligning predicted probabilities with actual frequencies
  - CalibratedClassifierCV — Platt scaling (sigmoid) and isotonic regression
  - Reliability curves (calibration plots) — comparing predicted vs actual probabilities
  - Custom decision threshold optimization — maximizing F1 or business metric
  - Threshold tuning — precision-recall tradeoff at different cutoffs

### Day 053 — Model Interpretability (XAI)
- **Tasks (3-4h):** Game theory math of SHAP (TreeExplainer, summary/force plots) & LIME for local surrogate explanations
- **Output:** `053_shap_lime_xai.ipynb`
- **Topics:**
  - SHAP (SHapley Additive exPlanations) — game theory-based feature attribution
  - TreeExplainer — fast SHAP for tree-based models (XGBoost, LightGBM)
  - SHAP summary plot — global feature importance with direction
  - SHAP force plot — individual prediction explanation
  - LIME (Local Interpretable Model-agnostic Explanations) — local surrogate models
  - SHAP waterfall plot — step-by-step contribution breakdown

### Day 054 — Anomaly & Outlier Detection
- **Tasks (3-4h):** Isolation Forests, One-Class SVM, LOF (Local Outlier Factor) for unsupervised anomaly scoring
- **Output:** `054_anomaly_detection.ipynb`
- **Topics:**
  - Isolation Forest — isolating anomalies via random recursive partitioning
  - One-Class SVM — learning boundary around normal data in kernel space
  - LOF (Local Outlier Factor) — density-based anomaly scoring
  - Contamination parameter — expected proportion of outliers
  - Anomaly scoring — decision_function scores and thresholding
  - Comparison benchmark — Isolation Forest vs One-Class SVM vs LOF

### Day 055 — Classical ML Comparison Lab
- **Tasks (3-4h):** Comparative benchmark of LR, SVM, Random Forest, XGBoost, LightGBM, and CatBoost on complex tabular data
- **Output:** `055_ml_benchmark_lab.ipynb`
- **Topics:**
  - Multi-model benchmark setup — consistent train/test split across models
  - Logistic Regression vs SVM vs Random Forest comparison
  - XGBoost vs LightGBM vs CatBoost comparison — boosting battle
  - Evaluation metrics — accuracy, F1, ROC-AUC, training time
  - Complex tabular dataset — feature engineering + model selection
  - Results visualization — comparison bar plots and tables

### Day 056 — Advanced ML Knowledge Assessment
- **Tasks (3-4h):** Solving 20 complex technical interview questions, math derivations, and debugging buggy pipelines
- **Output:** `056_ml_assessment_notes.md`
- **Topics:**
  - Technical interview question solving — 20 complex ML problems
  - Mathematical derivations — gradient descent, regularization, tree splitting
  - Debugging buggy ML pipelines — identifying common errors
  - Algorithm selection reasoning — justifying model choices
  - Bias-variance, overfitting, and leakage conceptual questions

---

# 🔵 PHASE 5: Unsupervised Learning, Time Series & E2E ML Project (Days 057–077)
> **Folder:** Phase-5 - Unsupervised Learning

---

### Day 057 — K-Means Clustering
- **Tasks (3-4h):** Lloyd's algorithm math, K-Means++ initialization, Inertia vs Elbow method, Silhouette score analysis
- **Output:** `057_kmeans_clustering.ipynb`
- **Topics:**
  - K-Means clustering — partitioning data into K clusters
  - Lloyd's algorithm — iterative assign-update convergence
  - K-Means++ initialization — smart centroid seeding to avoid poor starts
  - Inertia (WCSS) — within-cluster sum of squares
  - Elbow method — plotting inertia vs K to find optimal clusters
  - Silhouette score — measuring cluster cohesion vs separation [-1, 1]

### Day 058 — Hierarchical Clustering
- **Tasks (3-4h):** Agglomerative vs Divisive, Ward's linkage, distance metrics, dendrogram plotting & tree cuts
- **Output:** `058_hierarchical_dendrograms.ipynb`
- **Topics:**
  - Agglomerative clustering — bottom-up hierarchical merging
  - Divisive clustering — top-down hierarchical splitting
  - Ward's linkage — minimizing total within-cluster variance on merge
  - Distance metrics — single, complete, average, Ward linkage comparison
  - Dendrogram plotting — visualizing merge hierarchy
  - Tree cuts — choosing number of clusters from dendrogram height

### Day 059 — Density-Based Clustering (DBSCAN)
- **Tasks (3-4h):** Core, border, noise points intuition, eps & min_samples tuning, HDBSCAN introduction
- **Output:** `059_dbscan_hdbscan.ipynb`
- **Topics:**
  - DBSCAN algorithm — Density-Based Spatial Clustering
  - Core points, border points, noise points — point classification
  - eps parameter — neighborhood radius
  - min_samples parameter — minimum points to form dense region
  - HDBSCAN — Hierarchical DBSCAN for varying density clusters
  - Advantages over K-Means — arbitrary cluster shapes, noise handling

### Day 060 — Gaussian Mixture Models (GMM)
- **Tasks (3-4h):** Expectation-Maximization (EM) algorithm math, soft clustering, AIC/BIC for optimal component selection
- **Output:** `060_gmm_em_algorithm.ipynb`
- **Topics:**
  - Gaussian Mixture Models — probabilistic soft clustering
  - Expectation-Maximization (EM) algorithm — E-step (assign) + M-step (update)
  - Soft clustering — probability of belonging to each cluster
  - AIC (Akaike Information Criterion) — model selection penalizing complexity
  - BIC (Bayesian Information Criterion) — stricter complexity penalty than AIC
  - Optimal component selection — choosing number of Gaussians

### Day 061 — Principal Component Analysis (PCA)
- **Tasks (3-4h):** Covariance matrix, Eigenvalues/Eigenvectors derivation, explained variance ratio, dimensionality reduction
- **Output:** `061_pca_from_scratch.ipynb`
- **Topics:**
  - Principal Component Analysis (PCA) — linear dimensionality reduction
  - Covariance matrix computation — measuring feature co-variation
  - Eigenvalues and Eigenvectors — directions of maximum variance
  - Explained variance ratio — proportion of variance per component
  - Dimensionality reduction — projecting to lower-dimensional space
  - PCA from scratch — implementing using NumPy eigendecomposition

### Day 062 — Non-Linear Dim Reduction (t-SNE & UMAP)
- **Tasks (3-4h):** Manifold learning intuition, t-SNE perplexity tuning, UMAP global vs local structure preservation
- **Output:** `062_tsne_umap_viz.ipynb`
- **Topics:**
  - Manifold learning — non-linear dimensionality reduction concept
  - t-SNE (t-distributed Stochastic Neighbor Embedding) — preserving local structure
  - Perplexity parameter in t-SNE — effective number of neighbors
  - UMAP (Uniform Manifold Approximation and Projection) — faster alternative to t-SNE
  - Global vs local structure preservation — UMAP advantage over t-SNE
  - Visualization of high-dimensional data — 2D/3D scatter plots

### Day 063 — Association Rule Mining
- **Tasks (3-4h):** Apriori & FP-Growth algorithms, Support, Confidence, Lift metrics for market basket analysis
- **Output:** `063_association_rules.ipynb`
- **Topics:**
  - Apriori algorithm — frequent itemset mining with support threshold
  - FP-Growth algorithm — compressed frequent pattern tree (no candidate generation)
  - Support metric — frequency of itemset in transactions
  - Confidence metric — P(B|A) for rule A → B
  - Lift metric — Confidence / P(B), measuring association strength
  - Market basket analysis — discovering product co-purchase patterns

### Day 064 — Unsupervised Clustering Evaluation Lab
- **Tasks (3-4h):** Evaluating clusters with Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), profiling clusters
- **Output:** `064_clustering_eval_lab.ipynb`
- **Topics:**
  - Adjusted Rand Index (ARI) — chance-corrected clustering agreement
  - Normalized Mutual Information (NMI) — information-theoretic similarity
  - Silhouette analysis — per-cluster cohesion evaluation
  - Cluster profiling — characterizing clusters by feature statistics
  - Internal vs external evaluation — with and without ground truth labels

### Day 065 — Time Series Decomposition & Stationarity
- **Tasks (3-4h):** Additive/Multiplicative decomposition, Trend, Seasonality, Noise, Stationarity & ADF Test math
- **Output:** `065_ts_stationarity_adf.ipynb`
- **Topics:**
  - Time series components — Trend, Seasonality, Noise/Residual
  - Additive decomposition — y = Trend + Seasonal + Residual
  - Multiplicative decomposition — y = Trend × Seasonal × Residual
  - Stationarity concept — constant mean, variance, autocorrelation over time
  - ADF (Augmented Dickey-Fuller) Test — statistical test for stationarity
  - Differencing — making non-stationary series stationary

### Day 066 — Time Series Feature Engineering
- **Tasks (3-4h):** Lag features, rolling window statistics, expanding windows, time-based dummy variables, Walk-Forward CV
- **Output:** `066_ts_feature_engineering.ipynb`
- **Topics:**
  - Lag features — using past values as predictors (y_{t-1}, y_{t-2}, ...)
  - Rolling window statistics — moving average, moving std
  - Expanding window statistics — cumulative calculations
  - Time-based dummy variables — day of week, month, holiday flags
  - Walk-Forward Cross-Validation — respecting temporal order in splits

### Day 067 — Classical Forecasting (ARIMA/SARIMA)
- **Tasks (3-4h):** Autocorrelation (ACF) & Partial Autocorrelation (PACF) plots, AR(p), I(d), MA(q), SARIMA seasonal terms
- **Output:** `067_arima_sarima_modeling.ipynb`
- **Topics:**
  - Autocorrelation Function (ACF) — correlation of series with lagged self
  - Partial Autocorrelation Function (PACF) — direct correlation removing intermediate lags
  - AR(p) — Autoregressive model of order p
  - I(d) — Integration (differencing) of order d
  - MA(q) — Moving Average model of order q
  - SARIMA — Seasonal ARIMA with seasonal AR, I, MA terms

### Day 068 — Time Series Evaluation & ML Approaches
- **Tasks (3-4h):** MAPE, RMSE, MASE metrics, forecasting using XGBoost with lag features vs Prophet
- **Output:** `068_ts_forecasting_eval.ipynb`
- **Topics:**
  - MAPE — Mean Absolute Percentage Error
  - RMSE — Root Mean Squared Error for forecasting
  - MASE — Mean Absolute Scaled Error (scale-independent metric)
  - ML-based forecasting — XGBoost with lag features for time series
  - Prophet — Facebook's forecasting tool for trend + seasonality
  - ARIMA vs ML comparison — classical vs machine learning approaches

### Day 069 — [PROJECT 1] E2E ML: Problem & EDA
- **Tasks (3-4h):** Business scoping, dataset ingestion, thorough Exploratory Data Analysis (EDA), target distribution checks
- **Output:** `Project_1/01_eda.ipynb`
- **Topics:**
  - PROJECT 1 — Business problem scoping and objective definition
  - Dataset ingestion — loading and initial data inspection
  - Exploratory Data Analysis (EDA) — statistical summaries, visualizations
  - Target distribution analysis — checking for imbalance or skew
  - Feature-target relationship exploration — correlation, scatter plots

### Day 070 — [PROJECT 1] Feature Pipeline
- **Tasks (3-4h):** Building production-grade Scikit-Learn Pipeline & ColumnTransformer for tabular data preprocessing
- **Output:** `Project_1/src/pipeline.py`
- **Topics:**
  - PROJECT 1 — Production-grade Scikit-Learn Pipeline design
  - ColumnTransformer setup — numeric vs categorical preprocessing
  - Imputation strategy — handling missing values in pipeline
  - Encoding strategy — One-Hot / Ordinal encoding in pipeline
  - Scaling strategy — StandardScaler / RobustScaler integration

### Day 071 — [PROJECT 1] Baseline Modeling
- **Tasks (3-4h):** Training multiple baseline models, setting up Stratified K-Fold CV, tracking baseline metrics
- **Output:** `Project_1/02_baselines.py`
- **Topics:**
  - PROJECT 1 — Training multiple baseline models
  - Stratified K-Fold CV — robust evaluation with class balance
  - Baseline metrics tracking — accuracy, F1, ROC-AUC per model
  - Model comparison — selecting top candidates for tuning
  - Overfitting check — comparing train vs validation scores

### Day 072 — [PROJECT 1] Advanced Tuning
- **Tasks (3-4h):** Optuna hyperparameter optimization for XGBoost/LightGBM, ensemble stacking & blending
- **Output:** `Project_1/03_tuning.py`
- **Topics:**
  - PROJECT 1 — Optuna hyperparameter optimization for XGBoost/LightGBM
  - Search space definition — learning rate, depth, regularization ranges
  - Ensemble stacking — combining predictions from multiple models
  - Blending — weighted average of model predictions
  - Final model selection — best single model vs ensemble performance

### Day 073 — [PROJECT 1] Error Analysis & XAI
- **Tasks (3-4h):** Diagnostic error analysis, computing SHAP values, feature importance plots, residual analysis
- **Output:** `Project_1/04_xai_error_analysis.ipynb`
- **Topics:**
  - PROJECT 1 — Diagnostic error analysis on misclassified samples
  - SHAP values computation — TreeExplainer for feature attribution
  - Feature importance plots — global and local explanations
  - Residual analysis — patterns in prediction errors
  - Edge case identification — samples where model consistently fails

### Day 074 — [PROJECT 1] FastAPI Backend & Streamlit
- **Tasks (3-4h):** Developing REST API using FastAPI (/predict endpoint) & interactive Streamlit UI dashboard
- **Output:** `Project_1/app/main.py`
- **Topics:**
  - PROJECT 1 — FastAPI REST API development (/predict endpoint)
  - Request/response schema — Pydantic model for input validation
  - Model loading at startup — efficient serving architecture
  - Streamlit UI dashboard — interactive prediction interface
  - API-UI integration — connecting frontend to backend

### Day 075 — [PROJECT 1] Unit Testing & Documentation
- **Tasks (3-4h):** Writing pytest unit tests for data preprocessing and inference functions, complete README.md
- **Output:** `Project_1/tests/test_pipeline.py`
- **Topics:**
  - PROJECT 1 — pytest unit tests for data preprocessing functions
  - Testing inference pipeline — input → output validation
  - Edge case testing — handling null values, out-of-range inputs
  - README.md documentation — project overview, setup, usage instructions
  - Code organization — clean project structure

### Day 076 — [PROJECT 1] Deployment & Demo Video
- **Tasks (3-4h):** Dockerizing the application, deploying to cloud service (Render/HuggingFace Spaces), recording demo
- **Output:** `Project_1/Dockerfile + Live App URL`
- **Topics:**
  - PROJECT 1 — Dockerfile writing for containerization
  - Docker image building and testing locally
  - Cloud deployment — Render / HuggingFace Spaces
  - Demo video recording — showcasing live application
  - Final project polish and submission

> 🏆 **Project 1 Complete — End-to-End Tabular ML System**

### Day 077 — ML Phase Revision & System Design
- **Tasks (3-4h):** Classical ML interview drill, system design for tabular prediction systems, Mock Q&A review
- **Output:** `077_ml_system_design_notes.md`
- **Topics:**
  - Classical ML comprehensive revision — all algorithms reviewed
  - System design for tabular prediction — architecture patterns
  - ML interview drill — technical question practice
  - Mock Q&A review — simulating interview scenarios
  - End-to-end ML pipeline design — data to deployment

---

# 🟣 PHASE 6: Deep Learning Foundations & PyTorch (Days 078–122)

---

### Day 078 — Biological vs Artificial Neurons
- **Tasks (3-4h):** Biological vs Artificial neuron, Perceptron architecture, decision boundaries, XOR problem limitation
- **Output:** `078_perceptron_xor.ipynb`
- **Topics:**
  - Biological neuron — dendrites, axon, synapses
  - Artificial neuron (Perceptron) — weighted sum + activation
  - Perceptron architecture — single layer, binary classification
  - Linear decision boundaries — what perceptron can learn
  - XOR problem — limitation of single-layer perceptron (non-linear)

### Day 079 — Multi-Layer Perceptrons & Activations
- **Tasks (3-4h):** Hidden layers, matrix representation, Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, Softmax mathematical properties
- **Output:** `079_mlp_activations.ipynb`
- **Topics:**
  - Hidden layers — adding depth for non-linear representations
  - Matrix representation — weight matrices, bias vectors
  - Sigmoid activation — σ(z) = 1/(1+e^(-z)), vanishing gradient issue
  - Tanh activation — tanh(z), zero-centered output
  - ReLU activation — max(0, z), solving vanishing gradient
  - Leaky ReLU — allowing small negative values
  - GELU activation — Gaussian Error Linear Unit (used in Transformers)
  - Softmax — multi-class output probability distribution

### Day 080 — Forward Propagation Math
- **Tasks (3-4h):** Linear transformations (Z = WX + b), non-linear mappings, tensor shape transformations across layers
- **Output:** `080_forward_prop_math.ipynb`
- **Topics:**
  - Linear transformation — Z = WX + b at each layer
  - Non-linear mapping — applying activation function to Z
  - Tensor shape tracking — input → hidden → output dimensions
  - Layer-by-layer computation — sequential forward propagation
  - Batch processing — matrix multiplication for multiple samples simultaneously

### Day 081 — Deep Learning Loss Functions
- **Tasks (3-4h):** Mathematical formulation of MSE, Binary Cross-Entropy, Categorical Cross-Entropy, Sparse Cross-Entropy
- **Output:** `081_loss_functions.ipynb`
- **Topics:**
  - MSE Loss — ½Σ(yᵢ - ŷᵢ)² for regression
  - Binary Cross-Entropy — -[y·log(p) + (1-y)·log(1-p)]
  - Categorical Cross-Entropy — -Σyᵢ·log(pᵢ) with one-hot targets
  - Sparse Categorical Cross-Entropy — integer labels (no one-hot needed)
  - Loss function selection — matching loss to task type

### Day 082 — Backpropagation Intuition
- **Tasks (3-4h):** Computational graphs, error propagation backward through network, intuitive chain rule application
- **Output:** `082_backprop_intuition.md`
- **Topics:**
  - Computational graph — representing forward pass as directed graph
  - Error propagation — flowing gradients backward through network
  - Chain rule application — multiplying local gradients along path
  - Gradient flow intuition — how loss connects to each weight
  - Why backpropagation is efficient — computing all gradients in one pass

### Day 083 — Backpropagation Mathematics
- **Tasks (3-4h):** Detailed partial derivative derivations (∂L/∂W, ∂L/∂b) for multi-layer networks
- **Output:** `083_backprop_derivation.pdf`
- **Topics:**
  - Partial derivative derivation — ∂L/∂W for weight matrices
  - Partial derivative derivation — ∂L/∂b for bias vectors
  - Chain rule through activation functions — ∂σ/∂z computation
  - Multi-layer gradient computation — layer-by-layer backward pass
  - Jacobian matrices — derivatives for vector-valued functions

### Day 084 — Neural Network from Scratch in NumPy
- **Tasks (3-4h):** Coding complete MLP, forward pass, backpropagation, and weight update loop using ONLY pure NumPy
- **Output:** `084_numpy_mlp_scratch.ipynb`
- **Topics:**
  - MLP from scratch in NumPy — no framework, pure math
  - Forward pass implementation — matrix multiplications and activations
  - Backpropagation implementation — gradient computation and accumulation
  - Weight update loop — gradient descent parameter updates
  - Training on toy dataset — verifying convergence

### Day 085 — Vanishing & Exploding Gradients
- **Tasks (3-4h):** Mathematical cause of vanishing/exploding gradients in deep networks, gradient clipping implementation
- **Output:** `085_gradient_problems.ipynb`
- **Topics:**
  - Vanishing gradient problem — gradients shrinking in deep networks
  - Exploding gradient problem — gradients growing uncontrollably
  - Mathematical cause — repeated multiplication of small/large Jacobians
  - Gradient clipping implementation — torch.nn.utils.clip_grad_norm_
  - Activation function choice impact — ReLU solving vanishing gradients

### Day 086 — Stochastic & Mini-Batch GD
- **Tasks (3-4h):** Batch GD vs SGD vs Mini-Batch GD, epoch vs iteration, batch size impact on gradient variance
- **Output:** `086_minibatch_gd.ipynb`
- **Topics:**
  - Batch Gradient Descent — full dataset per update (stable but slow)
  - Stochastic Gradient Descent (SGD) — single sample per update (noisy but fast)
  - Mini-Batch GD — compromise between batch and stochastic
  - Epoch vs iteration — full pass vs single batch update
  - Batch size impact — effect on gradient variance and convergence speed

### Day 087 — Momentum & Nesterov Accelerations
- **Tasks (3-4h):** Exponentially weighted averages, Momentum update math, Nesterov Accelerated Gradient (NAG) intuition
- **Output:** `087_momentum_nag.ipynb`
- **Topics:**
  - Exponentially weighted moving averages — smoothing gradient history
  - Momentum update — v = βv + (1-β)∇L, accelerating convergence
  - Momentum intuition — ball rolling downhill with inertia
  - Nesterov Accelerated Gradient (NAG) — look-ahead gradient computation
  - NAG vs standard Momentum — anticipatory correction

### Day 088 — Adaptive Optimizers (Adam & AdamW)
- **Tasks (3-4h):** Adagrad, RMSprop, Adam (first & second moment estimation), AdamW weight decay correction math
- **Output:** `088_adam_adamw_math.ipynb`
- **Topics:**
  - Adagrad — adaptive per-parameter learning rate (accumulating squared gradients)
  - RMSprop — fixing Adagrad's decaying learning rate with exponential average
  - Adam optimizer — combining Momentum (1st moment) + RMSprop (2nd moment)
  - Bias correction in Adam — correcting initialization bias in early steps
  - AdamW — decoupled weight decay (proper L2 regularization with Adam)

### Day 089 — Optimizer Comparison Benchmark
- **Tasks (3-4h):** Coding comparison notebook evaluating SGD, Momentum, RMSprop, and AdamW on noisy loss surfaces
- **Output:** `089_optimizer_benchmark.ipynb`
- **Topics:**
  - Optimizer comparison experiment — same model, different optimizers
  - SGD vs Momentum vs RMSprop vs AdamW — loss curve comparison
  - Noisy loss surface evaluation — optimizer robustness testing
  - Convergence speed comparison — iterations to reach target loss
  - Best practices — when to use which optimizer

### Day 090 — Weight Initialization Strategies
- **Tasks (3-4h):** Symmetry breaking, Xavier/Glorot initialization, He/Kaiming initialization math for ReLU networks
- **Output:** `090_weight_initialization.ipynb`
- **Topics:**
  - Symmetry breaking — why random initialization is necessary
  - Xavier/Glorot initialization — Var(W) = 2/(n_in + n_out), for sigmoid/tanh
  - He/Kaiming initialization — Var(W) = 2/n_in, designed for ReLU networks
  - Zero initialization problem — all neurons learn same features
  - Initialization impact on training — convergence speed and stability

### Day 091 — Learning Rate Scheduling
- **Tasks (3-4h):** Step decay, Exponential decay, Cosine Annealing, OneCycleLR policy, Learning Rate finder algorithm
- **Output:** `091_lr_schedules.ipynb`
- **Topics:**
  - Step decay schedule — reducing LR by factor every N epochs
  - Exponential decay — LR = LR₀ × e^(-kt)
  - Cosine Annealing — LR follows cosine curve from max to min
  - OneCycleLR policy — warmup → peak → anneal in one cycle
  - Learning Rate Finder — sweep LR range to find optimal starting point

### Day 092 — Neural Network Regularization
- **Tasks (3-4h):** L1/L2 weight decay math, Dropout mechanism (Inverted Dropout math), Data Augmentation intuition
- **Output:** `092_dropout_regularization.ipynb`
- **Topics:**
  - L1/L2 weight decay — adding penalty to loss function
  - Dropout mechanism — randomly zeroing neurons during training
  - Inverted Dropout — scaling by 1/(1-p) to maintain expected values
  - Dropout rate selection — typically 0.2-0.5 for hidden layers
  - Data Augmentation — increasing effective training set size

### Day 093 — Normalization Layers (BatchNorm/LayerNorm)
- **Tasks (3-4h):** Batch Normalization math (mean, variance, scale γ, shift β), Layer Normalization comparison
- **Output:** `093_batchnorm_layernorm.ipynb`
- **Topics:**
  - Batch Normalization — normalizing activations across mini-batch
  - BN math — μ_B, σ²_B, normalize, scale (γ), shift (β)
  - BN benefits — faster training, higher learning rates, regularization effect
  - Layer Normalization — normalizing across features (not batch)
  - BatchNorm vs LayerNorm — when to use each (CNNs vs Transformers)

### Day 094 — Early Stopping & Model Checkpointing
- **Tasks (3-4h):** Monitoring validation loss, patience parameter, saving best weights, preventing overtraining
- **Output:** `094_early_stopping.py`
- **Topics:**
  - Early Stopping — halting training when validation loss stops improving
  - Patience parameter — epochs to wait before stopping
  - Model Checkpointing — saving best weights during training
  - Restoring best weights — loading checkpoint after training
  - Preventing overtraining — validation loss as stopping criterion

### Day 095 — PyTorch Fundamentals & Tensors
- **Tasks (3-4h):** PyTorch Tensor creation, GPU acceleration (.to("cuda")), tensor operations, broadcasting rules
- **Output:** `095_pytorch_tensors.ipynb`
- **Topics:**
  - PyTorch Tensor creation — torch.tensor, torch.zeros, torch.randn
  - GPU acceleration — .to('cuda'), .to('mps') for Apple Silicon
  - Tensor operations — element-wise, matrix multiplication, reshaping
  - Broadcasting rules — automatic shape expansion for operations
  - Tensor dtypes — float32, float64, int64, type conversion

### Day 096 — PyTorch Autograd Engine
- **Tasks (3-4h):** Computational graph creation, requires_grad=True, .backward() gradient computation, torch.no_grad()
- **Output:** `096_pytorch_autograd.ipynb`
- **Topics:**
  - Computational graph in PyTorch — dynamic graph construction
  - requires_grad=True — enabling gradient tracking
  - .backward() — computing gradients via automatic differentiation
  - torch.no_grad() context — disabling gradient computation for inference
  - .grad attribute — accessing computed gradients on tensors

### Day 097 — PyTorch Data Ingestion Architecture
- **Tasks (3-4h):** Custom Dataset class subclassing, __len__, __getitem__, DataLoader (batching, shuffling, workers)
- **Output:** `097_pytorch_dataloaders.ipynb`
- **Topics:**
  - Custom Dataset class — subclassing torch.utils.data.Dataset
  - __len__ method — returning dataset size
  - __getitem__ method — returning single sample by index
  - DataLoader — batching, shuffling, parallel loading (num_workers)
  - Data transforms — preprocessing within Dataset pipeline

### Day 098 — PyTorch nn.Module Architecture
- **Tasks (3-4h):** Building custom models inheriting nn.Module, nn.Linear, nn.Sequential, parameter inspection
- **Output:** `098_pytorch_nn_module.py`
- **Topics:**
  - nn.Module — base class for all PyTorch models
  - Custom model — defining __init__ (layers) and forward (computation)
  - nn.Linear — fully connected layer (weight matrix + bias)
  - nn.Sequential — stacking layers without custom forward
  - Parameter inspection — model.parameters(), named_parameters()

### Day 099 — Standard PyTorch Training Loop
- **Tasks (3-4h):** Writing explicit training loop: zero_grad(), forward pass, loss calculation, backward(), optimizer.step()
- **Output:** `099_pytorch_train_loop.py`
- **Topics:**
  - Training loop anatomy — the 5-step PyTorch ritual
  - optimizer.zero_grad() — clearing accumulated gradients
  - Forward pass — model(input) to get predictions
  - Loss computation — criterion(predictions, targets)
  - loss.backward() — computing gradients via backpropagation
  - optimizer.step() — updating weights using gradients

### Day 100 — Validation Loop & Model Persistence
- **Tasks (3-4h):** Evaluating model in eval() mode with torch.no_grad(), saving/loading state dicts (torch.save)
- **Output:** `100_validation_checkpointing.py`
- **Topics:**
  - model.eval() mode — disabling dropout, using running BN stats
  - torch.no_grad() context — saving memory during evaluation
  - Validation loop — computing metrics without gradient updates
  - torch.save(model.state_dict()) — saving model weights
  - model.load_state_dict() — loading saved weights for inference

### Day 101 — Device Agnostic PyTorch Execution
- **Tasks (3-4h):** Multi-device execution (cpu, cuda, mps), mixed-precision training using torch.cuda.amp.autocast
- **Output:** `101_device_agnostic_amp.py`
- **Topics:**
  - Device-agnostic code — torch.device('cuda' if available else 'cpu')
  - Multi-device support — CPU, CUDA (NVIDIA), MPS (Apple)
  - Moving models and data to device — .to(device)
  - Mixed-precision training — torch.cuda.amp.autocast for FP16
  - GradScaler — preventing underflow in mixed-precision training

### Day 102 — Hyperparameter Tuning PyTorch & Optuna
- **Tasks (3-4h):** Integrating Optuna with PyTorch for automated learning rate, batch size, and architecture search
- **Output:** `102_pytorch_optuna.py`
- **Topics:**
  - Optuna + PyTorch integration — optimizing neural network hyperparameters
  - Learning rate search — suggest_float for log-uniform sampling
  - Batch size optimization — suggest_categorical for size selection
  - Architecture search — suggest_int for hidden layer sizes
  - Trial pruning — early stopping unpromising hyperparameter configs

### Day 103 — Deep Learning Experiment Tracking
- **Tasks (3-4h):** Logging metrics, loss curves, model artifacts, and hyperparameters using Weights & Biases (WandB)
- **Output:** `103_wandb_tracking.py`
- **Topics:**
  - Weights & Biases (WandB) setup — wandb.init(), wandb.log()
  - Logging metrics — loss curves, accuracy per epoch
  - Logging hyperparameters — wandb.config for reproducibility
  - Model artifact tracking — saving model checkpoints to WandB
  - Dashboard visualization — real-time training monitoring

### Day 104 — Pure PyTorch MLP Implementation
- **Tasks (3-4h):** End-to-end PyTorch script building multi-layer perceptron for multi-class classification from scratch
- **Output:** `104_pytorch_mlp_complete.py`
- **Topics:**
  - End-to-end PyTorch MLP — complete classification script
  - Multi-class classification — output layer with softmax
  - Data loading — custom Dataset + DataLoader pipeline
  - Training + validation loops — tracking both losses
  - Model evaluation — accuracy, confusion matrix

### Day 105 — Deep Learning Image/Tabular Classifier
- **Tasks (3-4h):** Building production PyTorch classification pipeline with custom loss functions and evaluation metrics
- **Output:** `105_pytorch_classifier.py`
- **Topics:**
  - Production classification pipeline — structured PyTorch project
  - Custom loss functions — implementing task-specific losses
  - Evaluation metrics computation — precision, recall, F1 in PyTorch
  - Model selection — comparing architectures and hyperparameters
  - Inference pipeline — loading model and predicting on new data

### Day 106 — Debugging Neural Network Training
- **Tasks (3-4h):** Systematic debugging: checking gradient magnitudes, detecting NaN loss, overfitting single batch test
- **Output:** `106_dl_debugging_guide.md`
- **Topics:**
  - Gradient magnitude checking — detecting vanishing/exploding gradients
  - NaN loss detection — identifying numerical instability
  - Overfitting single batch test — verifying model can memorize small data
  - Learning rate debugging — loss not decreasing diagnosis
  - Shape mismatch debugging — tracking tensor dimensions through network

### Day 107 — Ablation Studies & Loss Landscapes
- **Tasks (3-4h):** Performing ablation experiments on layer depth, activation functions, regularization; plotting loss curves
- **Output:** `107_ablation_study.ipynb`
- **Topics:**
  - Ablation study design — systematically removing components
  - Layer depth ablation — impact of adding/removing hidden layers
  - Activation function comparison — ReLU vs GELU vs Tanh
  - Regularization ablation — dropout rate impact on generalization
  - Loss landscape plotting — visualizing optimization surface

### Day 108 — Self-Supervised Learning Concepts
- **Tasks (3-4h):** Pretext tasks vs downstream tasks, contrastive learning intuition, self-supervised representation learning
- **Output:** `108_ssl_concepts.md`
- **Topics:**
  - Self-supervised learning — learning representations without labels
  - Pretext tasks — artificially created tasks for pretraining
  - Downstream tasks — target tasks using pretrained representations
  - Contrastive learning — pulling similar pairs closer, pushing dissimilar apart
  - Benefits — label-efficient learning, transferable representations

### Day 109 — Autoencoders Fundamentals
- **Tasks (3-4h):** Encoder-Decoder architecture, latent space representation, reconstruction loss (MSE), dimensionality reduction
- **Output:** `109_autoencoders_basics.ipynb`
- **Topics:**
  - Autoencoder architecture — Encoder compresses, Decoder reconstructs
  - Latent space — compressed intermediate representation
  - Reconstruction loss (MSE) — minimizing input-output difference
  - Bottleneck layer — forcing compressed representation
  - Dimensionality reduction via Autoencoders — alternative to PCA

### Day 110 — Representation Learning Lab
- **Tasks (3-4h):** Training Autoencoder on MNIST/CIFAR, extracting latent embeddings, visualizing latent space via t-SNE
- **Output:** `110_representation_lab.ipynb`
- **Topics:**
  - Training Autoencoder on MNIST/CIFAR — image reconstruction task
  - Extracting latent embeddings — using encoder output as features
  - Latent space visualization — t-SNE on learned embeddings
  - Representation quality — cluster separation in latent space
  - Comparison with PCA — non-linear vs linear representations

### Day 111 — Docker Fundamentals
- **Tasks (3-4h):** Containers vs VMs, Docker architecture, essential commands (build, run, ps, exec, stop)
- **Output:** `111_docker_basics.sh`
- **Topics:**
  - Containers vs VMs — isolation, resource sharing, speed differences
  - Docker architecture — daemon, client, images, containers, registry
  - Essential CLI commands — build, run, ps, exec, stop, rm, rmi
  - Dockerfile basics — FROM, WORKDIR, COPY, RUN, CMD, EXPOSE
  - Container lifecycle — created, running, paused, stopped, exited

### Day 112 — FastAPI Fundamentals
- **Tasks (3-4h):** Path/query parameters, request body schemas, Swagger docs
- **Output:** `112_fastapi_basics.py`
- **Topics:**
  - FastAPI framework overview — automatic docs, high performance, type hints
  - Path parameters vs Query parameters — definitions and routing
  - Request body schemas — Pydantic models for data validation
  - Interactive Swagger / OpenAPI docs — automatic generation at /docs
  - Running FastAPI server — uvicorn server startup and reload

### Day 113 — pytest Fundamentals
- **Tasks (3-4h):** Unit tests, assertions, fixtures (@pytest.fixture), mocking
- **Output:** `113_pytest_basics.py`
- **Topics:**
  - Automated testing principles — unit vs integration vs regression testing
  - Writing pytest functions — assertion statements and test naming conventions
  - Pytest fixtures — `@pytest.fixture` for setup and teardown reuse
  - Mocking dependencies — using `unittest.mock` / `mocker` to isolate units
  - Running test suites — flags (`-v`, `-k`, `--cov`), test output interpretation

### Day 114 — [PROJECT 2] PyTorch DL System Setup
- **Tasks (3-4h):** Project planning, repository structuring, problem selection (e.g., custom image/tabular classification)
- **Output:** `Project_2/01_setup.md`
- **Topics:**
  - PROJECT 2 — Deep Learning system project planning
  - Repository structure — organizing src/, data/, models/, app/
  - Problem selection — custom image or tabular classification task
  - Technology stack — PyTorch, WandB, FastAPI
  - Milestones and deliverables planning

### Day 115 — [PROJECT 2] Dataset & Model Build
- **Tasks (3-4h):** Custom PyTorch Dataset, DataLoader with augmentations, custom PyTorch model architecture script
- **Output:** `Project_2/src/dataset.py`
- **Topics:**
  - PROJECT 2 — Custom PyTorch Dataset implementation
  - DataLoader with augmentations — transforms pipeline
  - Custom model architecture — designing layer structure
  - Data preprocessing — normalization, augmentation strategy
  - Train/val/test split — data partitioning

### Day 116 — [PROJECT 2] Training & Tracking
- **Tasks (3-4h):** Complete training pipeline with WandB experiment tracking, Learning Rate scheduler, validation loop
- **Output:** `Project_2/src/train.py`
- **Topics:**
  - PROJECT 2 — Complete training pipeline with WandB tracking
  - Learning Rate scheduler integration — OneCycleLR / CosineAnnealing
  - Validation loop — per-epoch evaluation
  - Experiment tracking — logging loss, accuracy, learning rate
  - Checkpoint saving — best model persistence

### Day 117 — [PROJECT 2] Evaluation & Diagnostics
- **Tasks (3-4h):** Confusion matrix computation, per-class F1-score evaluation, misclassification analysis
- **Output:** `Project_2/src/evaluate.py`
- **Topics:**
  - PROJECT 2 — Confusion matrix computation
  - Per-class F1-score evaluation — identifying weak classes
  - Misclassification analysis — examining failure cases
  - Error categorization — systematic failure pattern identification
  - Model improvement recommendations — based on error analysis

### Day 118 — [PROJECT 2] Model Serving REST API
- **Tasks (3-4h):** Wrapping PyTorch model in FastAPI endpoint, handling tensor conversions, returning JSON response
- **Output:** `Project_2/app/serve.py`
- **Topics:**
  - PROJECT 2 — FastAPI endpoint for PyTorch model serving
  - Tensor conversion — handling input data to tensor format
  - JSON response formatting — returning predictions
  - Async endpoint — non-blocking inference
  - API documentation — Swagger/OpenAPI auto-generated docs

### Day 119 — [PROJECT 2] Testing & Containerization
- **Tasks (3-4h):** Unit testing PyTorch model shapes & outputs via pytest, writing production Dockerfile
- **Output:** `Project_2/Dockerfile`
- **Topics:**
  - PROJECT 2 — pytest unit tests for model shapes and outputs
  - Testing forward pass — verifying output dimensions
  - Testing preprocessing — input validation
  - Dockerfile — containerizing PyTorch application
  - Docker build and test — local container verification

### Day 120 — [PROJECT 2] Deployment & Report
- **Tasks (3-4h):** Deploying containerized PyTorch service, writing comprehensive project README.md with WandB plots
- **Output:** `Project_2/README.md + Live URL`
- **Topics:**
  - PROJECT 2 — Deploying containerized PyTorch service
  - Cloud deployment — selecting hosting platform
  - README.md — comprehensive project documentation
  - WandB plots integration — embedding training visualizations
  - Live demo URL — accessible deployed application

> 🏆 **Project 2 Complete — Deep Learning PyTorch Pipeline**

### Day 121 — Deep Learning Foundations Revision
- **Tasks (3-4h):** Deep Learning math review, backpropagation derivation check, PyTorch syntax refresh
- **Output:** `121_dl_revision.md`
- **Topics:**
  - Deep Learning math review — backpropagation derivations check
  - PyTorch syntax refresh — API patterns and best practices
  - Optimizer and scheduler review — Adam, AdamW, cosine annealing
  - Regularization review — dropout, weight decay, batch norm
  - Architecture design principles — depth, width, skip connections

### Day 122 — Deep Learning Assessment Drill
- **Tasks (3-4h):** Solving 20 advanced PyTorch & neural network architecture interview questions
- **Output:** `122_dl_assessment.md`
- **Topics:**
  - Advanced PyTorch interview questions — 20 problems
  - Neural network architecture design questions
  - Backpropagation and gradient flow problems
  - Optimization and training strategy questions
  - Debugging and troubleshooting scenarios

---

# 🟠 PHASE 7: Computer Vision & Multimodal AI (Days 123–164)

---

### Day 123 — Digital Image Fundamentals & OpenCV
- **Tasks (3-4h):** Image representation (RGB, BGR, HSV, Grayscale), pixel arrays, OpenCV basic operations, color spaces
- **Output:** `123_opencv_basics.ipynb`
- **Topics:**
  - Image representation — RGB, BGR, HSV, Grayscale pixel arrays
  - OpenCV library basics — cv2.imread, cv2.imshow, cv2.cvtColor
  - Color space conversions — RGB ↔ HSV ↔ Grayscale
  - Pixel manipulation — accessing and modifying pixel values
  - Basic operations — resizing, cropping, rotating images

### Day 124 — Convolution Operation Mechanics
- **Tasks (3-4h):** 2D Spatial Convolution math, Kernels/Filters (Sobel, Gaussian, Laplacian), feature map computation
- **Output:** `124_convolution_math.ipynb`
- **Topics:**
  - 2D Spatial Convolution — sliding kernel over image
  - Kernel/Filter types — Sobel (edge), Gaussian (blur), Laplacian (edge)
  - Feature map computation — element-wise multiply and sum
  - Convolution vs correlation — kernel flipping difference
  - Edge detection — applying Sobel filters to detect gradients

### Day 125 — Padding, Stride & Output Dimensions
- **Tasks (3-4h):** Valid vs Same Padding, Stride parameter impact, mathematical formula for output dimension calculations
- **Output:** `125_conv_dimensions_math.md`
- **Topics:**
  - Valid Padding — no padding, output smaller than input
  - Same Padding — zero padding to keep output size equal to input
  - Stride parameter — step size of kernel sliding
  - Output dimension formula — O = (I - K + 2P) / S + 1
  - Multi-channel convolution — 3D kernels for RGB images

### Day 126 — Pooling Operations
- **Tasks (3-4h):** Max Pooling, Average Pooling, Global Average Pooling math, translation invariance properties
- **Output:** `126_pooling_layers.ipynb`
- **Topics:**
  - Max Pooling — taking maximum value in each pooling window
  - Average Pooling — taking mean value in each pooling window
  - Global Average Pooling (GAP) — averaging entire feature map to single value
  - Translation invariance — pooling providing spatial robustness
  - Dimensionality reduction — reducing spatial dimensions while keeping channels

### Day 127 — Convolutional Neural Network (CNN)
- **Tasks (3-4h):** Stacking Conv-ReLU-Pool layers, Flattening, Dense layers, receptive field expansion concept
- **Output:** `127_cnn_architecture.ipynb`
- **Topics:**
  - CNN architecture — stacking Conv → ReLU → Pool layers
  - Flattening — converting 3D feature maps to 1D vector
  - Dense (Fully Connected) layers — classification head
  - Receptive field — how deep layers see larger input regions
  - Feature hierarchy — low-level edges → mid-level textures → high-level objects

### Day 128 — Classic Architectures (LeNet & AlexNet)
- **Tasks (3-4h):** Historical evolution: LeNet-5 vs AlexNet, ReLU activation shift, GPU training introduction
- **Output:** `128_lenet_alexnet.ipynb`
- **Topics:**
  - LeNet-5 (1998) — pioneering CNN for digit recognition
  - AlexNet (2012) — deeper CNN, ReLU activation, GPU training
  - Historical evolution — from handcrafted features to learned features
  - ReLU activation shift — replacing sigmoid/tanh in CNNs
  - Dropout introduction in AlexNet — regularization for deep networks

### Day 129 — VGG Architecture & 3×3 Convolutions
- **Tasks (3-4h):** VGG-16 / VGG-19 design principles, power of stacked small 3×3 filters over large filters
- **Output:** `129_vgg16_architecture.ipynb`
- **Topics:**
  - VGG-16 / VGG-19 architecture — very deep, uniform design
  - 3×3 convolution stacking — two 3×3 = one 5×5 receptive field
  - Depth advantage — more non-linearities with fewer parameters
  - VGG design principles — simplicity and uniformity
  - Parameter count analysis — why VGG is memory-heavy

### Day 130 — ResNet & Skip Connections
- **Tasks (3-4h):** Vanishing gradient in deep networks, Residual Block math (F(x) + x), Identity mappings, ResNet-50
- **Output:** `130_resnet_skip_connections.ipynb`
- **Topics:**
  - Vanishing gradient in very deep CNNs — degradation problem
  - Residual Block — F(x) + x skip connection
  - Identity mapping — gradient highway through skip connections
  - ResNet-50 architecture — bottleneck blocks with 1×1 convolutions
  - Why ResNets work — easier to learn residual than full mapping

### Day 131 — Custom CNN in PyTorch from Scratch
- **Tasks (3-4h):** Building multi-layer PyTorch CNN from scratch with explicit shape calculation comments
- **Output:** `131_pytorch_cnn_scratch.py`
- **Topics:**
  - Custom CNN in PyTorch — nn.Conv2d, nn.MaxPool2d, nn.Linear
  - Shape calculation — tracking tensor dimensions through layers
  - Forward method — defining computation graph
  - Training on image dataset — MNIST/CIFAR classification
  - Model evaluation — accuracy and loss curves

### Day 132 — Computer Vision Data Augmentation
- **Tasks (3-4h):** Spatial transformations: random cropping, flipping, rotation, color jittering using torchvision.transforms
- **Output:** `132_vision_augmentations.ipynb`
- **Topics:**
  - Random cropping — RandomResizedCrop for scale invariance
  - Random flipping — RandomHorizontalFlip for mirror invariance
  - Random rotation — RandomRotation for orientation invariance
  - Color jittering — brightness, contrast, saturation, hue variation
  - torchvision.transforms — composing augmentation pipelines

### Day 133 — Advanced Augmentation (Albumentations)
- **Tasks (3-4h):** Advanced techniques: CutMix, MixUp, Mosaic augmentation using Albumentations library
- **Output:** `133_advanced_albumentations.ipynb`
- **Topics:**
  - CutMix — replacing image patch with another image's patch
  - MixUp — blending two images with interpolated labels
  - Mosaic augmentation — combining 4 images into one training sample
  - Albumentations library — fast, flexible augmentation pipeline
  - Advanced spatial transforms — elastic, grid distortion, perspective

### Day 134 — Transfer Learning Concepts
- **Tasks (3-4h):** Pretrained models on ImageNet, Feature Extraction (freezing backbone) vs Fine-Tuning top layers
- **Output:** `134_transfer_learning_intro.ipynb`
- **Topics:**
  - Pretrained models — ImageNet weights as starting point
  - Feature Extraction — freezing backbone, training only classifier head
  - Fine-Tuning — unfreezing some/all layers for domain adaptation
  - Transfer learning benefits — less data needed, faster convergence
  - When to freeze vs fine-tune — dataset size and domain similarity

### Day 135 — Transfer Learning Fine-Tuning Lab
- **Tasks (3-4h):** Unfreezing upper convolutional blocks, differential learning rates, fine-tuning ResNet in PyTorch
- **Output:** `135_transfer_learning_lab.py`
- **Topics:**
  - Selective unfreezing — unfreezing upper convolutional blocks first
  - Differential learning rates — lower LR for early layers, higher for later
  - Fine-tuning ResNet in PyTorch — practical implementation
  - Training strategy — freeze → train head → unfreeze → fine-tune
  - Performance comparison — feature extraction vs full fine-tuning

### Day 136 — EfficientNet & MobileNet
- **Tasks (3-4h):** Depthwise Separable Convolutions (MobileNet), Compound Scaling law (EfficientNet) for edge devices
- **Output:** `136_efficientnet_mobilenet.ipynb`
- **Topics:**
  - Depthwise Separable Convolutions — depthwise + pointwise factorization
  - MobileNet architecture — efficient model for mobile/edge devices
  - Compound Scaling (EfficientNet) — scaling depth, width, resolution together
  - FLOPs and parameter efficiency — MobileNet vs VGG comparison
  - Edge deployment considerations — latency and model size constraints

### Day 137 — CV Performance Metrics
- **Tasks (3-4h):** Precision, Recall, F1-Score, Top-1 vs Top-5 Accuracy, Confusion Matrix visualization
- **Output:** `137_cv_metrics.ipynb`
- **Topics:**
  - Precision, Recall, F1-Score — per-class and macro/micro averaging
  - Top-1 Accuracy — correct class has highest probability
  - Top-5 Accuracy — correct class in top 5 predictions
  - Confusion Matrix visualization — heatmap of predictions vs actuals
  - Multi-class evaluation — micro vs macro vs weighted averaging

### Day 138 — Object Detection Fundamentals
- **Tasks (3-4h):** Bounding box representations, anchor boxes concept
- **Output:** `138_object_detection_intro.md`
- **Topics:**
  - Bounding box representations — (x, y, w, h) center format
  - Bounding box representations — (x_min, y_min, x_max, y_max) corner format
  - Anchor boxes — predefined boxes at different scales and aspect ratios
  - Object detection pipeline — region proposal + classification
  - Single-stage vs Two-stage detectors — speed vs accuracy tradeoff

### Day 139 — Detection Evaluation (IoU, NMS, mAP)
- **Tasks (3-4h):** Intersection over Union (IoU) math, Non-Maximum Suppression (NMS) algorithm, mAP@50, mAP@50-95
- **Output:** `139_iou_nms_map_math.ipynb`
- **Topics:**
  - Intersection over Union (IoU) — overlap area / union area
  - Non-Maximum Suppression (NMS) — removing duplicate detections
  - mAP@50 — mean Average Precision at IoU threshold 0.5
  - mAP@50-95 — averaged over IoU thresholds from 0.5 to 0.95
  - Precision-Recall curve — computing AP as area under PR curve

### Day 140 — YOLO Single-Stage Architecture
- **Tasks (3-4h):** Single-shot object detection intuition, grid cell prediction, YOLO loss function components
- **Output:** `140_yolo_architecture.md`
- **Topics:**
  - YOLO concept — You Only Look Once, single-pass detection
  - Grid cell prediction — dividing image into S×S grid
  - YOLO loss function — localization + confidence + classification terms
  - Anchor-based prediction — predicting offsets from anchor boxes
  - Speed advantage — real-time detection capability

### Day 141 — Custom Object Detection (YOLOv8)
- **Tasks (3-4h):** Ultralytics YOLOv8 library, dataset formatting (Roboflow), training YOLOv8 on custom object dataset
- **Output:** `141_yolov8_custom_training.ipynb`
- **Topics:**
  - Ultralytics YOLOv8 library — modern YOLO implementation
  - Dataset formatting — Roboflow annotation export
  - Custom training — model.train() on custom dataset
  - Training monitoring — mAP, loss curves during training
  - Inference on custom images — model.predict() usage

### Day 142 — Image Segmentation Concepts
- **Tasks (3-4h):** Semantic Segmentation vs Instance Segmentation vs Panoptic Segmentation concepts
- **Output:** `142_segmentation_concepts.md`
- **Topics:**
  - Semantic Segmentation — classifying every pixel (no instance distinction)
  - Instance Segmentation — separate mask per object instance
  - Panoptic Segmentation — combining semantic + instance segmentation
  - Pixel-level classification — dense prediction task
  - Applications — autonomous driving, medical imaging, satellite analysis

### Day 143 — U-Net Architecture
- **Tasks (3-4h):** Encoder-Decoder contracting/expanding path, Skip Connections between encoder and decoder blocks
- **Output:** `143_unet_architecture.ipynb`
- **Topics:**
  - U-Net architecture — symmetric encoder-decoder structure
  - Contracting path (Encoder) — downsampling with Conv + Pool
  - Expanding path (Decoder) — upsampling with transposed convolutions
  - Skip Connections — concatenating encoder features to decoder
  - Why skip connections help — preserving spatial detail for precise segmentation

### Day 144 — Segmentation Metrics & Loss Functions
- **Tasks (3-4h):** Dice Coefficient math, Jaccard Index (IoU), Binary Cross-Entropy + Dice Loss combination
- **Output:** `144_dice_loss_metrics.ipynb`
- **Topics:**
  - Dice Coefficient — 2|A∩B| / (|A| + |B|), overlap similarity
  - Jaccard Index (IoU) — |A∩B| / |A∪B|, intersection over union
  - Binary Cross-Entropy loss — per-pixel classification loss
  - Dice Loss — 1 - Dice Coefficient, differentiable segmentation loss
  - Combined BCE + Dice Loss — leveraging both loss functions

### Day 145 — Vision Transformers (ViT) Intuition
- **Tasks (3-4h):** Shifting from Convolution to Self-Attention, patch embeddings, positional encodings in images
- **Output:** `145_vit_intuition.md`
- **Topics:**
  - From Convolution to Self-Attention — paradigm shift in vision
  - Patch embeddings — dividing image into non-overlapping patches
  - Positional encodings for images — adding spatial information
  - Why Transformers for vision — global receptive field from layer 1
  - ViT vs CNN comparison — data efficiency and scalability tradeoffs

### Day 146 — Vision Transformer (ViT) Architecture
- **Tasks (3-4h):** Image patch extraction (16×16), linear projection, Transformer encoder blocks, [CLS] token for CV
- **Output:** `146_vit_architecture.ipynb`
- **Topics:**
  - Image patch extraction — splitting image into 16×16 patches
  - Linear projection — flattening and projecting patches to embeddings
  - Transformer encoder blocks — Multi-Head Attention + FFN
  - [CLS] token — classification token prepended to patch sequence
  - ViT classification head — MLP on [CLS] token output

### Day 147 — Multimodal Vision-Language Models
- **Tasks (3-4h):** CLIP (Contrastive Language-Image Pre-training) architecture, zero-shot image classification
- **Output:** `147_clip_multimodal.ipynb`
- **Topics:**
  - CLIP architecture — dual encoder (Image + Text) with contrastive learning
  - Contrastive pre-training — matching image-text pairs
  - Zero-shot image classification — classifying without task-specific training
  - Text-guided image understanding — natural language as classifier
  - CLIP applications — image search, visual QA, content moderation

### Day 148 — Practical OpenCV Edge & Contour Lab
- **Tasks (3-4h):** Canny Edge Detection, Sobel filters, finding & drawing contours, morphological operations
- **Output:** `148_opencv_contours_lab.ipynb`
- **Topics:**
  - Canny Edge Detection — multi-stage edge detection algorithm
  - Sobel filters — gradient-based edge detection (horizontal/vertical)
  - Contour finding — cv2.findContours for object boundary detection
  - Contour drawing — cv2.drawContours for visualization
  - Morphological operations — erosion, dilation, opening, closing

### Day 149 — PyTorch Image Classification Pipeline
- **Tasks (3-4h):** Complete production script for training PyTorch vision model on custom image dataset
- **Output:** `149_pytorch_image_classifier.py`
- **Topics:**
  - Complete PyTorch image classification pipeline — end-to-end script
  - Custom image dataset loading — ImageFolder / custom Dataset
  - Training loop with augmentations — transforms during training
  - Evaluation on test set — accuracy, per-class metrics
  - Model saving and inference — deployment-ready code

### Day 150 — Fine-Tuning Pretrained Vision Models
- **Tasks (3-4h):** Fine-tuning timm (PyTorch Image Models) library models (e.g., ConvNeXt, Swin Transformer)
- **Output:** `150_timm_fine_tuning.py`
- **Topics:**
  - timm library — PyTorch Image Models collection
  - ConvNeXt — modernized CNN with Transformer-inspired design
  - Swin Transformer — shifted window attention for efficient vision
  - Fine-tuning pretrained timm models — transfer learning pipeline
  - Model selection from timm — choosing architecture for task

### Day 151 — Object Detection Inference Lab
- **Tasks (3-4h):** Running real-time YOLOv8 bounding box inference on video streams and static images
- **Output:** `151_yolo_inference_lab.py`
- **Topics:**
  - YOLOv8 inference on images — model.predict() with confidence threshold
  - YOLOv8 inference on video — frame-by-frame detection pipeline
  - Bounding box drawing — overlaying predictions on frames
  - Real-time detection — processing video stream at high FPS
  - Post-processing — NMS, confidence filtering, class filtering

### Day 152 — U-Net Segmentation Lab
- **Tasks (3-4h):** Implementing and training U-Net model for medical or satellite image binary segmentation
- **Output:** `152_unet_segmentation_lab.py`
- **Topics:**
  - U-Net implementation in PyTorch — encoder-decoder model code
  - Training U-Net for binary segmentation — medical or satellite images
  - Data loading for segmentation — paired image-mask datasets
  - Training loop — Dice Loss optimization
  - Segmentation prediction visualization — overlay masks on images

### Day 153 — [PROJECT 3] CV System Planning
- **Tasks (3-4h):** Scoping CV project (e.g., Custom Defect Detection using YOLOv8 or Medical Image U-Net)
- **Output:** `Project_3/01_scoping.md`

### Day 154 — [PROJECT 3] Data Pipeline & Annotations
- **Tasks (3-4h):** Dataset preparation, labeling via Roboflow/Label Studio, applying Albumentations pipeline
- **Output:** `Project_3/src/dataset.py`

### Day 155 — [PROJECT 3] Model Baseline Training
- **Tasks (3-4h):** Training baseline model, verifying loss convergence, logging experiment parameters
- **Output:** `Project_3/src/train_baseline.py`

### Day 156 — [PROJECT 3] Model Fine-Tuning
- **Tasks (3-4h):** Fine-tuning model backbone, hyperparameter search, optimizing anchor/input resolution
- **Output:** `Project_3/src/train_advanced.py`

### Day 157 — [PROJECT 3] Quantitative Evaluation
- **Tasks (3-4h):** Computing mAP scores, Confusion Matrix, IoU distribution plots across validation set
- **Output:** `Project_3/02_evaluation.ipynb`

### Day 158 — [PROJECT 3] Grad-CAM & Error Diagnostics
- **Tasks (3-4h):** Visualizing Grad-CAM attention heatmaps to debug misclassifications and false positives
- **Output:** `Project_3/03_gradcam_diagnostics.ipynb`

### Day 159 — [PROJECT 3] Real-Time Inference Module
- **Tasks (3-4h):** Building low-latency OpenCV inference pipeline with bounding box/mask overlay drawing
- **Output:** `Project_3/src/infer.py`

### Day 160 — [PROJECT 3] FastAPI Service & UI
- **Tasks (3-4h):** Wrapping model inside FastAPI app (/detect endpoint), creating Streamlit upload UI
- **Output:** `Project_3/app/main.py`

### Day 161 — [PROJECT 3] Dockerization & ONNX Export
- **Tasks (3-4h):** Exporting PyTorch model to ONNX Runtime for 2-3× speedup, writing multi-stage Dockerfile
- **Output:** `Project_3/Dockerfile`

### Day 162 — [PROJECT 3] Cloud Deployment
- **Tasks (3-4h):** Deploying containerized CV microservice to AWS/GCP or HuggingFace Spaces
- **Output:** `Project_3/README.md + Live URL`

> 🏆 **Project 3 Complete — Computer Vision System**

### Day 163 — Computer Vision Revision
- **Tasks (3-4h):** Reviewing CNN math, YOLO loss function, U-Net architecture, ViT tokenization
- **Output:** `163_cv_revision.md`

### Day 164 — Computer Vision Technical Assessment
- **Tasks (3-4h):** Answering 20 senior-level Computer Vision & Multimodal AI interview questions
- **Output:** `164_cv_assessment.md`

---

# 🔴 PHASE 8: Sequence Models & Recommender Systems (Days 165–189)

---

### Day 165 — Sequential Data & Recurrent Concept
- **Tasks (3-4h):** Tabular vs Image vs Sequential data, time unfolding intuition, hidden state (h_t) concept
- **Output:** `165_sequence_data_intro.md`

### Day 166 — Vanilla Recurrent Neural Network (RNN)
- **Tasks (3-4h):** RNN Forward Pass math (h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b_h)), weight sharing across time steps
- **Output:** `166_rnn_forward_math.ipynb`

### Day 167 — Backpropagation Through Time (BPTT)
- **Tasks (3-4h):** Mathematical formulation of BPTT, unrolling time steps, gradient summation across sequences
- **Output:** `167_bptt_derivation.pdf`

### Day 168 — Vanishing/Exploding Gradients in RNNs
- **Tasks (3-4h):** Matrix multiplication over long sequence lengths (W^T), vanishing gradient in temporal domain
- **Output:** `168_rnn_vanishing_gradients.ipynb`

### Day 169 — Gradient Clipping Techniques
- **Tasks (3-4h):** Norm-based gradient clipping math (torch.nn.utils.clip_grad_norm_) to prevent exploding gradients
- **Output:** `169_gradient_clipping.py`

### Day 170 — Long Short-Term Memory (LSTM)
- **Tasks (3-4h):** Cell State (C_t) highway concept, resolving long-term dependency vanishing gradients
- **Output:** `170_lstm_intuition.md`

### Day 171 — LSTM Gate Mechanics & Math
- **Tasks (3-4h):** Mathematical equations of Forget Gate (f_t), Input Gate (i_t), Candidate (C̃_t), Output Gate (o_t)
- **Output:** `171_lstm_gates_math.ipynb`

### Day 172 — Gated Recurrent Unit (GRU)
- **Tasks (3-4h):** GRU architecture: Reset Gate (r_t), Update Gate (z_t), parameter efficiency compared to LSTM
- **Output:** `172_gru_architecture.ipynb`

### Day 173 — Bidirectional RNNs & Deep Stacked RNNs
- **Tasks (3-4h):** Forward vs Backward hidden states concatenation, stacking multiple recurrent layers
- **Output:** `173_bidirectional_rnn.ipynb`

### Day 174 — PyTorch Sequence Processing
- **Tasks (3-4h):** Variable sequence lengths, torch.nn.utils.rnn.pack_padded_sequence & pad_packed_sequence
- **Output:** `174_packed_sequences.py`

### Day 175 — Sequence-to-Sequence (Seq2Seq)
- **Tasks (3-4h):** Encoder-Decoder architecture for sequence generation (Machine Translation, Text Summarization)
- **Output:** `175_seq2seq_architecture.ipynb`

### Day 176 — Teacher Forcing Technique
- **Tasks (3-4h):** Teacher forcing during training vs autoregressive decoding during inference, exposure bias problem
- **Output:** `176_teacher_forcing.ipynb`

### Day 177 — Attention Mechanism Intuition
- **Tasks (3-4h):** Bottleneck problem in fixed-length encoder vectors, Bahdanau (Additive) & Luong (Multiplicative) attention
- **Output:** `177_attention_intuition.md`

### Day 178 — Attention Implementation from Scratch
- **Tasks (3-4h):** Implementing Luong Attention score calculation in PyTorch
- **Output:** `178_attention_scratch.py`

### Day 179 — Sequence Architecture Comparison
- **Tasks (3-4h):** Benchmark comparison: Vanilla RNN vs LSTM vs GRU vs Bidirectional LSTM on time series/text
- **Output:** `179_sequence_benchmark.ipynb`

### Day 180 — Sequence Model Project Planning
- **Tasks (3-4h):** Scoping sequence project (e.g., Time-Series Forecasting or Sentiment Analysis)
- **Output:** `180_project_scoping.md`

### Day 181 — Sequence Dataset Pipeline & Windowing
- **Tasks (3-4h):** Creating sliding window datasets for sequential data, sequence padding, PyTorch DataLoader setup
- **Output:** `181_sequence_pipeline.py`

### Day 182 — Sequence Model Training & Loss Check
- **Tasks (3-4h):** Training Bidirectional LSTM / GRU model, tracking validation loss, preventing sequence overfitting
- **Output:** `182_train_sequence_model.py`

### Day 183 — Sequence Model Evaluation
- **Tasks (3-4h):** Evaluating sequence predictions (RMSE, MAE for TS or F1-Score for text), visualization
- **Output:** `183_sequence_evaluation.ipynb`

### Day 184 — Sequence Model API & Streamlit Serving
- **Tasks (3-4h):** Packaging sequence inference logic inside FastAPI endpoint and Streamlit interactive web app
- **Output:** `184_app_serve.py`

### Day 185 — Sequence Models Revision
- **Tasks (3-4h):** Reviewing LSTM gate math, GRU equations, BPTT, packed sequences, and attention alignment
- **Output:** `185_sequence_revision.md`

---

# 🟤 PHASE 9: NLP & Transformer Ecosystem (Days 190–238)

---

### Day 186 — Recommender Systems: Collaborative Filtering
- **Tasks (3-4h):** User-based & Item-based Collaborative Filtering, similarity metrics (Cosine, Pearson correlation), user-item interaction matrix
- **Output:** `186_collaborative_filtering.ipynb`
- **Topics:**
  - Collaborative Filtering intuition — leveraging collective user preferences
  - User-based Collaborative Filtering — finding similar users to recommend items
  - Item-based Collaborative Filtering — finding items similar to user's liked items
  - Similarity metrics — Cosine Similarity, Pearson Correlation Coefficient, Jaccard Index
  - User-Item interaction matrix — explicit (ratings) vs implicit (clicks, views) data
  - Sparsity & cold-start challenges in collaborative filtering

### Day 187 — Matrix Factorization Techniques (SVD & ALS)
- **Tasks (3-4h):** Singular Value Decomposition (SVD), Alternating Least Squares (ALS) for implicit feedback, matrix factorization math for recommendations
- **Output:** `187_matrix_factorization_svd_als.ipynb`
- **Topics:**
  - Matrix Factorization concept — decomposing user-item matrix into latent factor matrices
  - Singular Value Decomposition (SVD) — mathematical formulation and dimensionality reduction
  - Truncated SVD / Funk SVD — handling unobserved entries in sparse rating matrices
  - Alternating Least Squares (ALS) — optimizing user and item matrices alternately
  - Implicit feedback modeling — confidence weighting and ALS for implicit datasets
  - Learning latent features — interpreting user preference and item characteristic vectors

### Day 188 — Content-Based & Hybrid Recommender Systems
- **Tasks (3-4h):** Content-based filtering using feature profiles & TF-IDF/embeddings, hybrid recommender architecture (switching, feature combination)
- **Output:** `188_content_hybrid_recommenders.ipynb`
- **Topics:**
  - Content-based filtering — recommending items with similar attributes/features
  - Item profile building — TF-IDF vectors, categorical attributes, text embeddings
  - User profile construction — aggregating feature profiles of user-interacted items
  - Hybrid recommender systems — combining collaborative filtering and content-based approaches
  - Hybrid architecture designs — weighted, switching, mixed, feature combination hybrids
  - Mitigating cold-start problem using content signals for new items/users

### Day 189 — Recommender Systems Evaluation Metrics
- **Tasks (3-4h):** Offline evaluation metrics: Precision@K, Recall@K, Mean Average Precision (MAP@K), Normalized Discounted Cumulative Gain (NDCG)
- **Output:** `189_recommender_eval_metrics.ipynb`
- **Topics:**
  - Recommender evaluation paradigm — top-K recommendation evaluation
  - Precision@K & Recall@K — measuring proportion of relevant items in top-K recommendations
  - Mean Average Precision (MAP@K) — evaluating rank-aware recommendation accuracy
  - Cumulative Gain (CG) & Discounted Cumulative Gain (DCG) — position-weighted relevance
  - Normalized Discounted Cumulative Gain (NDCG) — DCG normalized by Ideal DCG (IDCG)
  - Beyond accuracy metrics — coverage, diversity, novelty, and serendipity

### Day 190 — Natural Language Processing Pipeline
- **Tasks (3-4h):** Text cleaning: lowercasing, regex filtering, stopword removal, stemming (Porter) vs lemmatization (WordNet)
- **Output:** `190_nlp_pipeline_cleaning.ipynb`

### Day 191 — Text Tokenization Fundamentals
- **Tasks (3-4h):** Character-level vs Word-level vs Subword-level tokenization concepts, vocabulary building
- **Output:** `191_tokenization_basics.ipynb`

### Day 192 — Bag-of-Words (BoW) Representation
- **Tasks (3-4h):** Vector Space Model, Document-Term Matrix (DTM), CountVectorizer implementation, sparsity issues
- **Output:** `192_bag_of_words.ipynb`

### Day 193 — N-Grams Language Modeling
- **Tasks (3-4h):** Unigrams, Bigrams, Trigrams, capturing local word order context, vocabulary explosion challenges
- **Output:** `193_ngrams_representation.ipynb`

### Day 194 — Term Frequency-Inverse Document Frequency
- **Tasks (3-4h):** TF-IDF mathematical formula, inverse document frequency weighting
- **Output:** `194_tfidf_from_scratch.ipynb`

### Day 195 — Classical Text Classification Lab
- **Tasks (3-4h):** Building baseline text classification using TF-IDF + Multinomial Naive Bayes & Logistic Regression
- **Output:** `195_classical_nlp_lab.ipynb`

### Day 196 — Word Embeddings Intuition
- **Tasks (3-4h):** Discrete sparse vectors vs continuous dense vector spaces, distributed representations
- **Output:** `196_word_embeddings_intro.md`

### Day 197 — Word2Vec Architecture (CBOW)
- **Tasks (3-4h):** Continuous Bag-of-Words architecture: predicting target word from context words
- **Output:** `197_word2vec_cbow.ipynb`

### Day 198 — Word2Vec Architecture (Skip-gram)
- **Tasks (3-4h):** Skip-gram: predicting context words from target word, Negative Sampling optimization math
- **Output:** `198_word2vec_skipgram.ipynb`

### Day 199 — Global Vectors for Word Representation
- **Tasks (3-4h):** GloVe mathematical objective, matrix factorization on global co-occurrence statistics
- **Output:** `199_glove_embeddings.ipynb`

### Day 200 — Embedding Visualization & Analogy Lab
- **Tasks (3-4h):** Loading pretrained Word2Vec/GloVe, vector arithmetic ("King - Man + Woman = Queen")
- **Output:** `200_embedding_visualizations.ipynb`

### Day 201 — NLP Performance Evaluation Metrics
- **Tasks (3-4h):** Perplexity, BLEU score, ROUGE-1/2/L, Exact Match (EM)
- **Output:** `201_nlp_metrics_eval.ipynb`

### Day 202 — Self-Attention Mechanics (Q, K, V)
- **Tasks (3-4h):** Queries, Keys, Values abstraction intuition, Linear transformations of input vectors
- **Output:** `202_self_attention_qkv.md`

### Day 203 — Scaled Dot-Product Attention Math
- **Tasks (3-4h):** Attention(Q,K,V) = softmax(QK^T/√d_k)V, scaling factor importance
- **Output:** `203_scaled_dot_product_math.ipynb`

### Day 204 — Multi-Head Attention (MHA) Architecture
- **Tasks (3-4h):** Parallel attention heads, projecting to multiple subspaces, concatenating & linear output
- **Output:** `204_multihead_attention.ipynb`

### Day 205 — Positional Encoding Mechanics
- **Tasks (3-4h):** Sinusoidal Positional Encoding formulas, injecting spatial position into non-recurrent models
- **Output:** `205_positional_encoding.ipynb`

### Day 206 — Transformer Encoder Architecture
- **Tasks (3-4h):** LayerNorm (Pre-LN vs Post-LN), Residual Connections, FFN, complete Encoder block
- **Output:** `206_transformer_encoder.ipynb`

### Day 207 — Transformer Decoder Architecture
- **Tasks (3-4h):** Masked Multi-Head Attention (causal masking), Cross-Attention block
- **Output:** `207_transformer_decoder.ipynb`

### Day 208 — Full Encoder-Decoder Transformer
- **Tasks (3-4h):** Complete "Attention Is All You Need" architecture
- **Output:** `208_full_transformer_arch.ipynb`

### Day 209 — Bidirectional Encoder Representations (BERT)
- **Tasks (3-4h):** BERT architecture (Encoder-only), MLM & NSP pretraining objectives
- **Output:** `209_bert_architecture.md`

### Day 210 — Masked Language Modeling (MLM) Deep Dive
- **Tasks (3-4h):** 15% token masking strategy, fine-tuning BERT for classification
- **Output:** `210_mlm_deepdive.ipynb`

### Day 211 — Generative Pre-trained Transformer (GPT)
- **Tasks (3-4h):** GPT architecture (Decoder-only), autoregressive causal language modeling, zero/few-shot
- **Output:** `211_gpt_autoregressive.md`

### Day 212 — Modern Subword Tokenizers
- **Tasks (3-4h):** BPE, WordPiece (BERT), SentencePiece (Unigram) algorithms step-by-step
- **Output:** `212_subword_tokenizers.ipynb`

### Day 213 — Hugging Face Ecosystem Overview
- **Tasks (3-4h):** transformers, datasets, tokenizers, accelerate, and Hugging Face Hub workflow
- **Output:** `213_hf_ecosystem_overview.py`

### Day 214 — Hugging Face Datasets & Tokenizers
- **Tasks (3-4h):** Loading datasets, fast tokenization via AutoTokenizer, dynamic padding
- **Output:** `214_hf_datasets_tokenizers.py`

### Day 215 — Transformer Fine-Tuning Classification
- **Tasks (3-4h):** Fine-tuning AutoModelForSequenceClassification using HF Trainer API & PyTorch
- **Output:** `215_transformer_classification.py`

### Day 216 — Named Entity Recognition (NER)
- **Tasks (3-4h):** Token classification, BIO tagging scheme, fine-tuning BERT for NER
- **Output:** `216_ner_token_classification.py`

### Day 217 — Question Answering with BERT
- **Tasks (3-4h):** Extractive QA, predicting start and end token logits over context paragraphs
- **Output:** `217_extractive_qa_bert.py`

### Day 218 — Text Summarization (BART / T5)
- **Tasks (3-4h):** Sequence-to-sequence fine-tuning for abstractive summarization
- **Output:** `218_abstractive_summarization.py`

### Day 219 — Text Generation Decoding Parameters
- **Tasks (3-4h):** Greedy search, Beam search, Temperature scaling, Top-k sampling, Top-p (Nucleus) sampling
- **Output:** `219_generation_decoding_params.py`

### Day 220 — NLP Evaluation Metrics Lab
- **Tasks (3-4h):** Hands-on evaluation lab computing ROUGE scores using evaluate library
- **Output:** `220_nlp_evaluation_lab.py`

### Day 221 — Building Transformer Encoder from Scratch
- **Tasks (3-4h):** Coding complete Transformer Encoder layer in pure PyTorch without nn.Transformer
- **Output:** `221_transformer_from_scratch.py`

### Day 222 — Fine-Tuning BERT for Sentiment Analysis
- **Tasks (3-4h):** Complete script fine-tuning bert-base-uncased on IMDB / Customer Reviews dataset
- **Output:** `222_bert_sentiment_fine_tune.py`

### Day 223 — Local GPT-2 Inference & Text Generation
- **Tasks (3-4h):** Loading pretrained GPT-2, experimenting with temperature, top-p, repetition penalties
- **Output:** `223_gpt2_inference_lab.py`

### Day 224 — Self-Supervised Pretraining Principles
- **Tasks (3-4h):** Masked Autoencoders (MAE), denoising objectives, pretraining cost vs fine-tuning efficiency
- **Output:** `224_pretraining_concepts.md`

### Day 225 — Contrastive Learning in NLP
- **Tasks (3-4h):** SimCSE, learning high-quality sentence vectors
- **Output:** `225_contrastive_nlp.ipynb`

### Day 226 — [PROJECT 4] Transformer NLP Setup
- **Tasks (3-4h):** Scoping NLP project (Multi-label Intent Classification or Domain-Specific NER)
- **Output:** `Project_4/01_scoping.md`

### Day 227 — [PROJECT 4] Data Pipeline & Tokenization
- **Tasks (3-4h):** Dataset cleaning, tokenization with AutoTokenizer, creating splits
- **Output:** `Project_4/src/dataset.py`

### Day 228 — [PROJECT 4] Baseline Classical NLP
- **Tasks (3-4h):** Training TF-IDF + Logistic Regression baseline
- **Output:** `Project_4/src/baseline.py`

### Day 229 — [PROJECT 4] Transformer Fine-Tuning
- **Tasks (3-4h):** Fine-tuning RoBERTa/DeBERTa using Trainer API with WandB tracking
- **Output:** `Project_4/src/train.py`

### Day 230 — [PROJECT 4] Quantitative Evaluation
- **Tasks (3-4h):** Per-class Precision, Recall, F1-Score, confusion matrix analysis
- **Output:** `Project_4/02_evaluation.ipynb`

### Day 231 — [PROJECT 4] Error Diagnostics & Analysis
- **Tasks (3-4h):** Diagnostic analysis on misclassified text samples, edge cases
- **Output:** `Project_4/03_error_analysis.ipynb`

### Day 232 — [PROJECT 4] FastAPI & Streamlit UI
- **Tasks (3-4h):** Building production FastAPI REST endpoint with batch inference & Streamlit UI
- **Output:** `Project_4/app/main.py`

### Day 233 — [PROJECT 4] Containerization & Cloud Deploy
- **Tasks (3-4h):** Writing multi-stage Dockerfile, deploying NLP microservice
- **Output:** `Project_4/Dockerfile + Live URL`

> 🏆 **Project 4 Complete — Transformer NLP Application**

### Day 234 — NLP Foundations Revision
- **Output:** `234_nlp_foundations_revision.md`

### Day 235 — Transformer Architecture Revision
- **Output:** `235_transformer_revision.md`

### Day 236 — NLP & Transformers Interview Drills
- **Output:** `236_nlp_interview_drills.md`

### Day 237 — NLP Comprehensive Knowledge Assessment
- **Output:** `237_nlp_assessment_results.md`

### Day 238 — Buffer & Skills Reinforcement Day
- **Output:** `238_buffer_reinforcement.md`

---

# ⚙️ PHASE 10: MLOps & Data Engineering Essentials (Days 239–267)

---

### Day 239 — Advanced Git & GitHub Collaboration
- **Tasks (3-4h):** Git branching strategies (GitFlow), rebase vs merge, interactive rebase, merge conflicts
- **Output:** `239_git_workflow_guide.md`

### Day 240 — SQL Essentials for Data Engineering
- **Tasks (3-4h):** SELECT, WHERE, GROUP BY, HAVING, ORDER BY, aggregate functions
- **Output:** `240_sql_essentials.sql`

### Day 241 — Advanced SQL (Joins, CTEs, Windows)
- **Tasks (3-4h):** INNER/LEFT/RIGHT/FULL JOINs, CTEs, Window Functions (ROW_NUMBER, RANK, LEAD/LAG)
- **Output:** `241_advanced_sql_queries.sql`

### Day 242 — Data Validation & Schema Integrity
- **Tasks (3-4h):** Data quality checks, schema enforcement using Great Expectations or Pydantic
- **Output:** `242_data_validation.py`

### Day 243 — RESTful API Architecture Principles
- **Tasks (3-4h):** REST constraints, HTTP methods, status codes, request/response headers
- **Output:** `243_rest_api_principles.md`

### Day 244 — Production Web Framework (FastAPI)
- **Tasks (3-4h):** FastAPI fundamentals, path/query parameters, request body schemas, Swagger docs
- **Output:** `244_fastapi_basics.py`

### Day 245 — Data Validation with Pydantic
- **Tasks (3-4h):** Pydantic BaseModel schemas, field validation, custom validators, type hints
- **Output:** `245_pydantic_validation.py`

### Day 246 — High-Performance Model Serving
- **Tasks (3-4h):** Loading models at startup (lifespan), async endpoints, batching
- **Output:** `246_fastapi_model_serving.py`

### Day 247 — Automated Unit Testing with pytest
- **Tasks (3-4h):** Unit tests, assertions, fixtures (@pytest.fixture), mocking (unittest.mock)
- **Output:** `247_pytest_suite.py`

### Day 248 — Production Logging & Configuration
- **Tasks (3-4h):** Structured logging (loguru), environment variables (python-dotenv), config management
- **Output:** `248_logging_config.py`

### Day 249 — Docker Fundamentals & Containerization
- **Tasks (3-4h):** Containers vs VMs, Docker architecture, essential commands (build, run, ps, exec, stop)
- **Output:** `249_docker_basics.sh`

### Day 250 — Dockerizing Machine Learning Applications
- **Tasks (3-4h):** Writing efficient Dockerfile: base image, WORKDIR, requirements.txt, CMD
- **Output:** `250_Dockerfile_ml`

### Day 251 — Multi-Container Apps with Docker Compose
- **Tasks (3-4h):** docker-compose.yml syntax, services (API + Web UI + DB), networks, volumes
- **Output:** `251_docker_compose.yml`

### Day 252 — ML Experiment Tracking (MLflow)
- **Tasks (3-4h):** mlflow.log_param, mlflow.log_metric, mlflow.log_artifact, MLflow UI server
- **Output:** `252_mlflow_tracking.py`

### Day 253 — MLflow Model Registry & Lifecycle
- **Tasks (3-4h):** Registering models, stage transitions (Staging → Production → Archived)
- **Output:** `253_mlflow_registry.py`

### Day 254 — Data Version Control (DVC)
- **Tasks (3-4h):** dvc init, dvc add, remote storage (S3/GCP/local), Git + DVC synergy
- **Output:** `254_dvc_setup.sh`

### Day 255 — Reproducible Pipelines with DVC
- **Tasks (3-4h):** dvc.yaml pipeline stages, dependency graphs, dvc repro
- **Output:** `255_dvc.yaml`

### Day 256 — Continuous Integration / CD Concepts
- **Tasks (3-4h):** CI/CD principles, GitHub Actions architecture: workflows, jobs, steps, runners
- **Output:** `256_cicd_concepts.md`

### Day 257 — Automated CI Workflow (GitHub Actions)
- **Tasks (3-4h):** Writing .github/workflows/ci.yml: linting (flake8/black), pytest on PR
- **Output:** `257_ci_workflow.yml`

### Day 258 — Production Model Monitoring
- **Tasks (3-4h):** Operational vs ML performance metrics, monitoring latency, throughput, error rates
- **Output:** `258_model_monitoring_intro.md`

### Day 259 — A/B Testing & Experimentation Design
- **Tasks (3-4h):** Statistical significance testing, sample size calculation, online vs offline evaluation, hypothesis testing for ML model comparisons
- **Output:** `259_ab_testing_experimentation.ipynb`
- **Topics:**
  - A/B Testing fundamentals — split testing control vs treatment models in production
  - Hypothesis testing for ML models — null hypothesis (H₀), alternative hypothesis (H₁)
  - Statistical significance tests — two-sample t-test, Z-test, Chi-Square test
  - Sample size determination — power analysis, significance level (α), statistical power (1-β), minimum detectable effect (MDE)
  - Online vs Offline evaluation — metrics alignment (offline loss/F1 vs online CTR/conversion)
  - Guardrail metrics, variance reduction techniques (CUPED), and preventing p-hacking

### Day 260 — Data Drift & Concept Drift Detection
- **Tasks (3-4h):** KS test, PSI, drift detection reports using Evidently AI
- **Output:** `260_drift_detection.py`

### Day 261 — Cloud Deployment Architecture
- **Tasks (3-4h):** AWS EC2, S3, ECS / GCP Cloud Run, serverless container deployment
- **Output:** `261_cloud_architecture.md`

### Day 262 — [PROJECT 5] MLOps Pipeline Setup
- **Tasks (3-4h):** Scoping MLOps project: Git repo, DVC data tracking & MLflow server
- **Output:** `Project_5/01_setup.sh`

### Day 263 — [PROJECT 5] DVC Reproducible Pipeline
- **Output:** `Project_5/dvc.yaml`

### Day 264 — [PROJECT 5] Experiment Tracking & Registry
- **Output:** `Project_5/src/train.py`

### Day 265 — [PROJECT 5] Containerization & Testing
- **Output:** `Project_5/Dockerfile`

### Day 266 — [PROJECT 5] CI/CD & Drift Monitoring
- **Output:** `Project_5/.github/workflows/cd.yml`

> 🏆 **Project 5 Complete — Production MLOps Pipeline**

### Day 267 — MLOps & Engineering Revision
- **Output:** `267_mlops_revision.md`

---

# ✨ PHASE 11: Generative AI, RAG & LLM Fine-Tuning (Days 268–316)

---

### Day 268 — Large Language Model (LLM) Foundations
- **Tasks (3-4h):** Evolution from BERT/GPT to modern LLMs, Decoder-only dominance, emergent abilities
- **Output:** `268_llm_foundations.md`

### Day 269 — Pretraining & Compute Scaling Laws
- **Tasks (3-4h):** Pretraining datasets, Chinchilla scaling laws (20× tokens per parameter)
- **Output:** `269_scaling_laws.md`

### Day 270 — Modern LLM Tokenization Deep Dive
- **Tasks (3-4h):** BPE, tiktoken library, handling special tokens
- **Output:** `270_llm_tokenization.md`

### Day 271 — Context Windows & Positional Embeddings
- **Tasks (3-4h):** Absolute vs Relative positional encodings, RoPE, ALiBi, extending context windows
- **Output:** `271_rope_context_windows.md`

### Day 272 — Key-Value (KV) Cache & Inference
- **Tasks (3-4h):** KV Cache mechanics, PagedAttention (vLLM)
- **Output:** `272_kv_cache_mechanics.ipynb`

### Day 273 — Prompt Engineering Fundamentals
- **Tasks (3-4h):** System/User/Assistant messages, Zero-Shot, Few-Shot, Chain-of-Thought (CoT)
- **Output:** `273_prompt_engineering.py`

### Day 274 — Advanced Prompting Techniques
- **Tasks (3-4h):** Tree-of-Thoughts (ToT), Self-Consistency, prompt chaining
- **Output:** `274_advanced_prompting.py`

### Day 275 — Structured Output Generation
- **Tasks (3-4h):** Enforcing JSON output, Pydantic function parameters, instructor library
- **Output:** `275_structured_outputs.py`

### Day 276 — Systematic Prompt Evaluation Framework
- **Tasks (3-4h):** Prompt test suites, evaluating consistency, latency, correctness
- **Output:** `276_prompt_evaluation.py`

### Day 277 — Vector Embeddings for Generative AI
- **Tasks (3-4h):** Text embeddings, OpenAI vs sentence-transformers (BGE)
- **Output:** `277_vector_embeddings.ipynb`

### Day 278 — Vector Similarity Search Mathematics
- **Tasks (3-4h):** Euclidean Distance vs Cosine Similarity vs Dot Product math
- **Output:** `278_similarity_metrics_math.ipynb`

### Day 279 — Vector Indexing Algorithms (HNSW)
- **Tasks (3-4h):** Exact Search vs ANN: HNSW graph
- **Output:** `279_hnsw_indexing_concepts.md`

### Day 280 — Local Vector Search (FAISS)
- **Tasks (3-4h):** IndexFlatL2, IndexIVFFlat, indexing & querying
- **Output:** `280_faiss_vector_db.py`

### Day 281 — Production Vector Databases (Chroma/Qdrant)
- **Tasks (3-4h):** Creating collections, adding document chunks + metadata, similarity queries
- **Output:** `281_chroma_qdrant_lab.py`

### Day 282 — Retrieval-Augmented Generation (RAG)
- **Tasks (3-4h):** RAG architecture overview: overcoming hallucinations, knowledge cutoffs
- **Output:** `282_rag_architecture_overview.md`

### Day 283 — Document Ingestion & ETL Pipelines
- **Tasks (3-4h):** Parsing PDFs, Markdown, HTML using Unstructured, PyPDF, LlamaIndex
- **Output:** `283_document_ingestion.py`

### Day 284 — Document Chunking Strategies
- **Tasks (3-4h):** Fixed-size vs Recursive Character vs Semantic chunking
- **Output:** `284_chunking_strategies.py`

### Day 285 — Vector Retrieval & Context Injection
- **Tasks (3-4h):** Retrieving top-k chunks, constructing context-augmented prompt
- **Output:** `285_vector_retrieval_injection.py`

### Day 286 — End-to-End Basic RAG Pipeline
- **Tasks (3-4h):** Complete naive RAG: Ingest → Chunk → Embed → Store → Retrieve → Generate
- **Output:** `286_basic_rag_pipeline.py`

### Day 287 — Hybrid Search Architecture
- **Tasks (3-4h):** BM25 + Dense Vector Search using Reciprocal Rank Fusion (RRF)
- **Output:** `287_hybrid_search_bm25.py`

### Day 288 — Query Transformation & Expansion
- **Tasks (3-4h):** Query Rewriting, Multi-Query Generation
- **Output:** `288_query_transformations.py`

### Day 289 — Hypothetical Document Embeddings (HyDE)
- **Tasks (3-4h):** Generating hypothetical answer document via LLM for vector retrieval
- **Output:** `289_hyde_retrieval.py`

### Day 290 — Two-Stage Retrieval with Cross-Encoders
- **Tasks (3-4h):** Bi-Encoder (fast retrieval) + Cross-Encoder Reranking (Cohere/BGE-Reranker)
- **Output:** `290_cross_encoder_reranking.py`

### Day 291 — Contextual Compression & Summarization
- **Tasks (3-4h):** Filtering irrelevant sentences from retrieved chunks before prompt injection
- **Output:** `291_contextual_compression.py`

### Day 292 — Advanced Production RAG Pipeline
- **Tasks (3-4h):** Combining Hybrid Search + Query Rewriting + Cross-Encoder Reranking
- **Output:** `292_advanced_rag_pipeline.py`

### Day 293 — RAG Evaluation Frameworks
- **Tasks (3-4h):** Context Precision, Context Recall, Faithfulness, Answer Relevance
- **Output:** `293_rag_evaluation_intro.md`

### Day 294 — Automated RAG Evaluation with RAGAS
- **Tasks (3-4h):** Running RAGAS evaluation library, computing quantitative scores
- **Output:** `294_ragas_evaluation_lab.py`

### Day 295 — Prompt Injection & Guardrails
- **Tasks (3-4h):** Defending against Prompt Injection, NeMo Guardrails / Llama Guard
- **Output:** `295_prompt_security_guardrails.py`

### Day 296 — LLM Safety, Moderation & PII Masking
- **Tasks (3-4h):** OpenAI Moderation API, PII detection & masking using Presidio
- **Output:** `296_pii_masking_moderation.py`

### Day 297 — Decision Matrix: RAG vs Fine-Tuning
- **Tasks (3-4h):** When to use Prompt Engineering vs RAG vs Fine-Tuning vs Training from Scratch
- **Output:** `297_rag_vs_finetuning_matrix.md`

### Day 298 — Parameter-Efficient Fine-Tuning (PEFT)
- **Tasks (3-4h):** Full Fine-Tuning bottlenecks vs PEFT (Adapters, Prefix Tuning, LoRA)
- **Output:** `298_peft_concepts.md`

### Day 299 — Low-Rank Adaptation (LoRA) Mathematics
- **Tasks (3-4h):** W = W₀ + (α/r)·B·A, rank decomposition, alpha scaling
- **Output:** `299_lora_math_derivation.pdf`

### Day 300 — Quantized LoRA (QLoRA) & Quantization
- **Tasks (3-4h):** 4-bit NF4, Double Quantization, Paged Optimizers, QLoRA architecture
- **Output:** `300_qlora_quantization.md`

### Day 301 — Model Quantization Techniques (GGUF/AWQ)
- **Tasks (3-4h):** PTQ: FP16 → INT8 → INT4, GGUF format, AWQ, GPTQ concepts
- **Output:** `301_quantization_gguf_awq.md`

### Day 302 — QLoRA Fine-Tuning Hands-On Lab
- **Tasks (3-4h):** Fine-tuning Llama-3 / Mistral 7B using peft, bitsandbytes, TRL SFTTrainer
- **Output:** `302_qlora_finetuning_lab.py`

### Day 303 — LLM API Integration & Function Calling
- **Tasks (3-4h):** OpenAI / Anthropic API, function JSON schemas, execution loop
- **Output:** `303_function_calling_loop.py`

### Day 304 — Structured Function Calling Workflows
- **Tasks (3-4h):** Parsing complex tool outputs, multi-tool selection, parameter validation
- **Output:** `304_structured_function_calling.py`

### Day 305 — Streaming Tokens in Web Applications
- **Tasks (3-4h):** Server-Sent Events (SSE), streaming LLM responses in FastAPI & Streamlit
- **Output:** `305_streaming_fastapi.py`

### Day 306 — Semantic Caching for LLM Queries
- **Tasks (3-4h):** GPTCache / Redis semantic cache for cost & latency reduction
- **Output:** `306_semantic_caching.py`

### Day 307 — [PROJECT 6] Advanced RAG System Setup
- **Output:** `Project_6/01_scoping.md`

### Day 308 — [PROJECT 6] Ingestion & Vector Pipeline
- **Output:** `Project_6/src/ingest.py`

### Day 309 — [PROJECT 6] Advanced Retrieval & Rerank
- **Output:** `Project_6/src/retriever.py`

### Day 310 — [PROJECT 6] Generation & Guardrails
- **Output:** `Project_6/src/generator.py`

### Day 311 — [PROJECT 6] RAGAS Systematic Evaluation
- **Output:** `Project_6/02_ragas_eval.ipynb`

### Day 312 — [PROJECT 6] Observability Integration
- **Output:** `Project_6/src/observability.py`

### Day 313 — [PROJECT 6] FastAPI Backend & Streaming UI
- **Output:** `Project_6/app/main.py`

### Day 314 — [PROJECT 6] Dockerization & Deployment
- **Output:** `Project_6/Dockerfile + Live URL`

### Day 315 — [PROJECT 6] Production Observability Audit
- **Output:** `Project_6/README.md`

> 🏆 **Project 6 Complete — Advanced Production RAG System**

### Day 316 — Generative AI & RAG Phase Revision
- **Output:** `316_genai_revision.md`

---

# 🤖 PHASE 12: Agentic AI Systems & Multi-Agent Workflows (Days 317–344)

---

### Day 317 — Agentic AI Fundamentals
- **Tasks (3-4h):** Chains vs Autonomous Agents, Perception-Reasoning-Action loop, agent autonomy spectrum
- **Output:** `317_agentic_ai_intro.md`

### Day 318 — Tool Calling & Execution Mechanics
- **Tasks (3-4h):** Agent tool bindings, translating NL into tool arguments, execution sandboxing
- **Output:** `318_tool_calling_mechanics.py`

### Day 319 — Tool Schema Design & Error Handling
- **Tasks (3-4h):** Pydantic tool schemas, error feedback loops to LLM for self-correction
- **Output:** `319_tool_schema_design.py`

### Day 320 — ReAct Reasoning Framework
- **Tasks (3-4h):** Reason-Act: Thought → Action → Observation execution loops
- **Output:** `320_react_agent_from_scratch.py`

### Day 321 — Planning & Goal Decomposition
- **Tasks (3-4h):** Task Decomposition, Sub-goal Generation, Refinement & Reflection
- **Output:** `321_plan_and_solve_agent.py`

### Day 322 — Agent Memory Systems
- **Tasks (3-4h):** Short-Term (Context Window) vs Long-Term Memory (Vector DB / Key-Value Store)
- **Output:** `322_agent_memory_systems.py`

### Day 323 — Stateful Agent Graphs (LangGraph)
- **Tasks (3-4h):** Why DAGs fail for complex agents: Cyclic graphs, LangGraph state machines
- **Output:** `323_langgraph_concepts.md`

### Day 324 — LangGraph Architecture
- **Tasks (3-4h):** StateGraph, TypedDict State, Nodes, Edges, START and END
- **Output:** `324_langgraph_basics.py`

### Day 325 — Conditional Routing in Graphs
- **Tasks (3-4h):** add_conditional_edges, Router functions, dynamic state-based branching
- **Output:** `325_langgraph_routing.py`

### Day 326 — Graph State Persistence & Checkpoints
- **Tasks (3-4h):** MemorySaver checkpointer, thread-based tracking, time-travel debugging
- **Output:** `326_langgraph_persistence.py`

### Day 327 — Human-in-the-Loop (HITL) Workflows
- **Tasks (3-4h):** Graph interrupt nodes (interrupt_before), human approval for sensitive tools
- **Output:** `327_langgraph_hitl.py`

### Day 328 — Parallel Node Execution & Subgraphs
- **Tasks (3-4h):** Parallel branches (asyncio), nesting subgraphs within parent graphs
- **Output:** `328_parallel_subgraphs.py`

### Day 329 — Agent Error Recovery & Fallbacks
- **Tasks (3-4h):** Circuit breakers, max iteration limits, infinite loop prevention, fallback routing
- **Output:** `329_agent_error_handling.py`

### Day 330 — Multi-Agent Systems Architecture
- **Tasks (3-4h):** Single Agent bottlenecks vs Multi-Agent Specialization, message routing
- **Output:** `330_multi_agent_architectures.md`

### Day 331 — Multi-Agent Supervisor Pattern
- **Tasks (3-4h):** Supervisor / Router pattern: orchestration agent delegating to worker agents
- **Output:** `331_supervisor_multi_agent.py`

### Day 332 — Model Context Protocol (MCP)
- **Tasks (3-4h):** MCP standards, Host-Client-Server relationship
- **Output:** `332_mcp_fundamentals.md`

### Day 333 — Model Context Protocol (MCP) Integration
- **Tasks (3-4h):** Building custom MCP Server, connecting to LangGraph Client
- **Output:** `333_mcp_server_client.py`

### Day 334 — Agent Systematic Evaluation
- **Tasks (3-4h):** Goal Completion Rate, Tool Selection Accuracy, Trajectory Efficiency
- **Output:** `334_agent_evaluation_metrics.md`

### Day 335 — LLM-as-a-Judge Evaluation
- **Tasks (3-4h):** Automated evaluation using strong LLM as judge to rate agent trajectories
- **Output:** `335_llm_as_judge_eval.py`

### Day 336 — Agent Security & Action Guardrails
- **Tasks (3-4h):** Sandboxing code execution, preventing prompt injection tool hijack, rate limiting
- **Output:** `336_agent_security_sandboxing.py`

### Day 337 — [PROJECT 7] Agentic AI System Setup
- **Output:** `Project_7/01_scoping.md`

### Day 338 — [PROJECT 7] Tool Suite & Sandbox Build
- **Output:** `Project_7/src/tools.py`

### Day 339 — [PROJECT 7] LangGraph Workflow Construction
- **Output:** `Project_7/src/agent.py`

### Day 340 — [PROJECT 7] HITL & Multi-Agent Orchestration
- **Output:** `Project_7/src/graph_advanced.py`

### Day 341 — [PROJECT 7] MCP Integration
- **Output:** `Project_7/src/mcp_integration.py`

### Day 342 — [PROJECT 7] Agent Trajectory Evaluation
- **Output:** `Project_7/02_trajectory_eval.ipynb`

### Day 343 — [PROJECT 7] Production FastAPI & UI Deploy
- **Output:** `Project_7/app/main.py`

> 🏆 **Project 7 Complete — Production Agentic AI System**

### Day 344 — Agentic AI Systems Revision
- **Output:** `344_agentic_ai_revision.md`

---

# 🧠 PHASE 13: Reinforcement Learning Fundamentals (Days 345–358)

---

### Day 345 — Reinforcement Learning Fundamentals
- **Tasks (3-4h):** Agent, Environment, State (S), Action (A), Reward (R), Policy (π), Return (G_t), Discount (γ)
- **Output:** `345_rl_concepts_intro.md`

### Day 346 — Markov Decision Processes (MDP)
- **Tasks (3-4h):** Markov Property, State Transition Probability Matrix, MDP formulation
- **Output:** `346_mdp_formulation.md`

### Day 347 — Bellman Equations Intuition
- **Tasks (3-4h):** Value Function V(s), Action-Value Q(s,a), Bellman Expectation & Optimality Equations
- **Output:** `347_bellman_equations_math.pdf`

### Day 348 — Dynamic Programming Concepts
- **Tasks (3-4h):** Policy Evaluation, Policy Iteration, Value Iteration (Gridworld)
- **Output:** `348_dynamic_programming_rl.ipynb`

### Day 349 — Monte Carlo Methods for RL
- **Tasks (3-4h):** Model-free RL, First-visit vs Every-visit MC, Monte Carlo Control
- **Output:** `349_monte_carlo_rl.ipynb`

### Day 350 — Temporal-Difference (TD) Learning
- **Tasks (3-4h):** TD Error δ_t = R_{t+1} + γ·V(S_{t+1}) - V(S_t), TD(0) vs Monte Carlo
- **Output:** `350_td_learning_math.ipynb`

### Day 351 — Q-Learning Algorithm (Off-Policy)
- **Tasks (3-4h):** Q(S,A) ← Q(S,A) + α[R + γ·max_a Q(S',a) - Q(S,A)], Q-Table
- **Output:** `351_q_learning_scratch.py`

### Day 352 — SARSA Algorithm (On-Policy)
- **Tasks (3-4h):** SARSA update rule, On-policy vs Off-policy comparison
- **Output:** `352_sarsa_scratch.py`

### Day 353 — Exploration vs Exploitation
- **Tasks (3-4h):** ε-Greedy, Decaying ε, Upper Confidence Bound (UCB)
- **Output:** `353_exploration_vs_exploitation.py`

### Day 354 — Gymnasium Environment Workflow
- **Tasks (3-4h):** env.reset(), env.step(action), observation/action spaces
- **Output:** `354_gymnasium_workflow.py`

### Day 355 — Deep Q-Networks (DQN) Intuition
- **Tasks (3-4h):** Scaling Q-Learning with Neural Networks, Replay Buffer, Target Net
- **Output:** `355_dqn_architecture_concepts.md`

### Day 356 — DQN Implementation Guided Lab
- **Tasks (3-4h):** DQN in PyTorch to solve CartPole-v1
- **Output:** `356_dqn_cartpole_pytorch.py`

### Day 357 — Policy Gradients & PPO Overview
- **Tasks (3-4h):** Value-based vs Policy-based RL, REINFORCE, PPO intuition
- **Output:** `357_policy_gradients_ppo.md`

### Day 358 — RL Mini-Project & Phase Revision
- **Tasks (3-4h):** Training RL agent on Gymnasium, logging reward curves, RL interview Q&A
- **Output:** `358_rl_miniproject_revision.ipynb`

---

# 🚀 PHASE 14: LLMOps, Production Capstone & Career Readiness (Days 359–374)

---

### Day 359 — LLM Observability Platforms
- **Tasks (3-4h):** Production tracing with Langfuse / Arize Phoenix — latency, token costs, traces
- **Output:** `359_langfuse_observability.py`

### Day 360 — Prompt Versioning & Experimentation
- **Tasks (3-4h):** Prompt templates as code, version control, performance tracking
- **Output:** `360_prompt_version_tracking.py`

### Day 361 — Production Vector DB Scaling
- **Tasks (3-4h):** Sharding, HNSW tuning, memory management for millions of vectors
- **Output:** `361_vector_db_scaling_guide.md`

### Day 362 — Latency & Cost Optimization
- **Tasks (3-4h):** Token optimization, model size selection (8B vs 70B), batch vs streaming
- **Output:** `362_cost_latency_optimization.md`

### Day 363 — Production Caching & Rate Limiting
- **Tasks (3-4h):** Redis-backed rate limiting, multi-level caching (Exact + Semantic)
- **Output:** `363_rate_limiting_caching.py`

### Day 364 — Fallback & Retry Strategies
- **Tasks (3-4h):** Exponential backoff, fallback cascades (Primary → Backup → Local Model)
- **Output:** `364_resilient_llm_fallbacks.py`

### Day 365 — Production Evaluation Pipelines
- **Tasks (3-4h):** CI/CD evaluation: LLM-as-a-Judge and regression tests on PRs
- **Output:** `365_prod_eval_pipeline.py`

### Day 366 — Security, Secrets & Access Control
- **Tasks (3-4h):** API key rotation, RBAC, secret vaults, data privacy compliance
- **Output:** `366_production_security.md`

### Day 367 — Responsible AI — Bias & Fairness
- **Tasks (3-4h):** Fairness metrics (demographic parity, equalized odds), bias detection in training data, model cards documentation
- **Output:** `367_responsible_ai_fairness.ipynb`
- **Topics:**
  - Responsible AI principles — fairness, accountability, transparency, ethics
  - Fairness metrics — Demographic Parity, Equalized Odds, Equal Opportunity, Disparate Impact Ratio
  - Bias detection & mitigation — pre-processing (reweighing), in-processing (fair constraints), post-processing (threshold adjustment)
  - Data bias auditing — representation bias, historical bias, measurement bias in datasets
  - Model Cards — standardized documentation for model capabilities, limitations, and evaluation metrics
  - Compliance & governance — AI ethics guidelines, safety auditing frameworks

### Day 368 — [FINAL CAPSTONE] Architecture Design
- **Tasks (3-4h):** System Design for Flagship Capstone (Multi-Agent + RAG + MLOps + Observability)
- **Output:** `Capstone/01_system_architecture.png`

### Day 369 — [FINAL CAPSTONE] Core System Build
- **Tasks (3-4h):** Implementing core AI engines, data pipelines, vector DBs, LangGraph graphs, tools
- **Output:** `Capstone/src/core_engine.py`

### Day 370 — [FINAL CAPSTONE] Evaluation & Safety
- **Tasks (3-4h):** RAGAS evaluation, LLM-as-judge, NeMo Guardrails, PII masking
- **Output:** `Capstone/src/safety_eval.py`

### Day 371 — [FINAL CAPSTONE] E2E Cloud Deployment
- **Tasks (3-4h):** Docker Compose multi-service stack, cloud deployment, Langfuse observability
- **Output:** `Capstone/docker-compose.yml + Live URL`

> 🏆 **Project 8 Complete — Enterprise Production AI System (Flagship Capstone)**

### Day 372 — Portfolio Optimization & GitHub Polish
- **Tasks (3-4h):** Auditing GitHub profile, polishing READMEs with architecture diagrams & Loom demos
- **Output:** `372_portfolio_audit.md`

### Day 373 — Master Technical Interview Review
- **Tasks (3-4h):** Comprehensive review: ML math, Deep Learning, PyTorch, Transformers, System Design, LLM Engineering
- **Output:** `373_master_interview_cheat_sheet.md`

### Day 374 — Final Mock Interview & Skills Audit
- **Tasks (3-4h):** Comprehensive 3-hour technical mock interview, auditing 1-year progress against industry standards
- **Output:** `374_final_skills_audit_completed.md`

---

## 📊 Technical Glossary

| # | Formula | Description |
|---|---|---|
| 1 | `L = (1/m) Σ(yᵢ - ŷᵢ)²` | Mean Squared Error |
| 2 | `θ = θ - α · ∇J(θ)` | Gradient Descent Update |
| 3 | `σ(z) = 1/(1 + e^(-z))` | Sigmoid Function |
| 4 | `-[y·log(p) + (1-y)·log(1-p)]` | Log Loss (Binary Cross-Entropy) |
| 5 | `1 - Σpᵢ²` | Gini Impurity |
| 6 | `-Σpᵢ·log₂(pᵢ)` | Shannon Entropy |
| 7 | P(A\ |B) = P(B\|A)·P(A) / P(B)` | Bayes' Theorem |
| 8 | `α = ½·ln((1-ε)/ε)` | AdaBoost Alpha |
| 9 | `(Σgᵢ)² / (Σhᵢ + λ)` | XGBoost Similarity Score |
| 10 | `max(0, z)` | ReLU Activation |
| 11 | `x·Φ(x)` | GELU Activation |
| 12 | `O = (I - K + 2P) / S + 1` | CNN Output Dimension |
| 13 | `σ(W_f·[h_{t-1}, x_t] + b_f)` | LSTM Forget Gate |
| 14 | `softmax(QKᵀ/√d_k)·V` | Scaled Dot-Product Attention |
| 15 | `z = (x - μ) / σ` | Standard Scaler |
| 16 | `R + γ·max V(s')` | Bellman Equation |
| 17 | 2\ |A∩B\| / (\|A\| + \|B\|)` | Dice Coefficient |
| 18 | `exp(zᵢ) / Σexp(zⱼ)` | Softmax Function |
| 19 | `W = W₀ + (α/r)·B·A` | LoRA Weight Update |
| 20 | `exp(zᵢ) / Σexp(zⱼ)` | Softmax Function |

---

> **🎯 End of 365-Day AI/ML Engineer Master Roadmap**
> **Sahil Kumar (Sky) — BTech CSE (AI/ML), DAV University, Jalandhar**
