# Advanced AI/ML Engineer Master Roadmap (2026 Edition)
**Timeline**: June 1, 2026 → December 31, 2026 (214 Days)  
**Daily Study Time**: 5 Hours  
**Target Profile**: Advanced AI/ML Engineer (Solid in Math, Advanced Architectures, Generative AI, and MLOps Systems)  
**Author**: Sahil-K-Y (Enhanced & Programmed by Arena AI)

---

## 📈 Executive Summary: What makes this an "Advanced" Roadmap?
Most standard roadmaps focus on using libraries (`scikit-learn.fit()`, `model.predict()`). This master curriculum is designed to teach you **how models work mathematically, how to optimize them for hardware (GPUs/TPUs), and how to architect them into high-performance real-world systems**.

### 🌟 Advanced Technical Highlights:
*   **Mathematical Rigor**: Every algorithm is explored down to its matrix calculus derivatives, loss formulations, and convergence bounds.
*   **PyTorch Mastery**: Focuses 100% on PyTorch, learning to write custom autograd functions, custom memory-efficient training loops, and distributed training setups (DDP).
*   **Explainable AI (XAI)**: Explicit training on SHAP, LIME, and feature attribution math.
*   **State-of-the-Art CV & NLP**: Deep dives into Vision Transformers (ViT), YOLOv8 custom heads, self-attention, and FlashAttention optimizations.
*   **MLOps Systems**: Advanced containerization, ONNX optimizations, Triton Inference Server dynamic batching, Kubernetes orchestration, Feast feature stores, and CI/CD automation pipelines.
*   **Generative & Agentic AI**: 4-bit double quantization (QLoRA) fine-tuning, stateful cyclic graphs with conditional routing (**LangGraph**), and multi-agent teams (**CrewAI**).

---

## 📅 Daily Study Split (5 Hours)
*   **Theory & Concept Study**: 90–120 min (Focused on papers, mathematical derivations, and architecture designs)
*   **Coding & Notebook Practice**: 120 min (Writing algorithms from scratch or constructing production code pipelines)
*   **Project Implementation**: 60 min (Building the 14 flagship production-grade capstone projects)
*   **Revision & Documentation**: 30 min (Writing technical summaries and pushing daily progress to GitHub)
*   **SQL/DSA/Interview Practice**: 30 min (Preparing for FAANG/MNC technical rounds)

---


### 📅 Day 001 | 01 Jun 2026, Mon
*   **Phase**: Phase 1: Linear Regression & Regularization
*   **Advanced Topic**: **Cost Function MSE Math**
*   **Deep-Dive Subtopics**: Vectorized Mean Squared Error, matrix notation of residuals, convexity proofs, Hessian of MSE loss, geometric interpretation in Hilbert space.
*   **Hands-on Deliverable**: `Day 001 - Cost Function MSE.ipynb (Pure NumPy vectorized MSE calculator and proof of convexity).`
*   **Interview Edge (Advanced Attributions)**: *"Explain how to prove that the MSE cost function is convex and why that guarantees a global minimum during gradient descent."*

---

### 📅 Day 002 | 02 Jun 2026, Tue
*   **Phase**: Phase 1: Linear Regression & Regularization
*   **Advanced Topic**: **Gradient Descent convergence**
*   **Deep-Dive Subtopics**: Lipschitz continuity of gradients, mathematical bounds for learning rates, convergence rates of Batch, Stochastic, and Mini-batch GD.
*   **Hands-on Deliverable**: `Day 002 - Gradient Descent.ipynb (Step-by-step Batch vs. SGD convergence comparison from scratch).`
*   **Interview Edge (Advanced Attributions)**: *"What is the mathematical relation between the maximum learning rate and the Lipschitz constant of the gradient of the loss function?"*

---

### 📅 Day 003 | 03 Jun 2026, Wed
*   **Phase**: Phase 1: Linear Regression & Regularization
*   **Advanced Topic**: **Linear Regression Closed-Form Solver**
*   **Deep-Dive Subtopics**: Normal Equation derivation, Cholesky decomposition vs. SVD (Singular Value Decomposition) solvers, computational complexity comparisons O(n³).
*   **Hands-on Deliverable**: `Day 003 - Linear Regression SVD Solver.ipynb (Custom linear regression class using SVD/Pseudoinverse).`
*   **Interview Edge (Advanced Attributions)**: *"Why does scikit-learn's LinearRegression use SVD (via PINV) instead of directly calculating the Normal Equation $(X^T X)^{-1} X^T y$?"*

---

### 📅 Day 004 | 04 Jun 2026, Thu
*   **Phase**: Phase 1: Linear Regression & Regularization
*   **Advanced Topic**: **Model Evaluation Rigor**
*   **Deep-Dive Subtopics**: Adjusted R² mathematical derivation, proof of why Adjusted R² can decrease when adding random noise features, MAE vs. RMSE sensitivity to outliers.
*   **Hands-on Deliverable**: `Day 004 - Model Evaluation.ipynb (Evaluation script computing R², Adjusted R², MAE, RMSE, and MAPE from scratch).`
*   **Interview Edge (Advanced Attributions)**: *"Under what exact conditions would MAE be a mathematically superior loss function to MSE? Prove it using maximum likelihood estimation."*

---

### 📅 Day 005 | 05 Jun 2026, Fri
*   **Phase**: Phase 1: Linear Regression & Regularization
*   **Advanced Topic**: **Multiple Regression & Multicollinearity**
*   **Deep-Dive Subtopics**: Multicollinearity, Variance Inflation Factor (VIF) math, checking model assumptions (homoscedasticity, Breusch-Pagan test, QQ plot math).
*   **Hands-on Deliverable**: `Day 005 - Multiple Linear Regression.ipynb (VIF calculator and regression assumption diagnostics pipeline).`
*   **Interview Edge (Advanced Attributions)**: *"What happens to the variance of the beta coefficients in a linear regression model when multicollinearity is present? Explain mathematically."*

---

### 📅 Day 006 | 06 Jun 2026, Sat
*   **Phase**: Phase 1: Linear Regression & Regularization
*   **Advanced Topic**: **Polynomial Regression & Bias-Variance Tradeoff**
*   **Deep-Dive Subtopics**: PolynomialFeatures, Runge's phenomenon, bias-variance tradeoff math derivation, overfitting with high degree feature mapping.
*   **Hands-on Deliverable**: `Day 006 - Polynomial Regression.ipynb (Fitting high-degree polynomials, plotting training vs. validation loss curve).`
*   **Interview Edge (Advanced Attributions)**: *"What is Runge's phenomenon and how does regularization mitigate it in polynomial regression?"*

---

### 📅 Day 007 | 07 Jun 2026, Sun
*   **Phase**: Phase 1: Linear Regression & Regularization
*   **Advanced Topic**: **Ridge & Lasso (L1 & L2 Regularization)**
*   **Deep-Dive Subtopics**: L1 penalty (Lasso) vs. L2 penalty (Ridge), geometric proof of why L1 induces sparsity (corners of the constraint L1-ball), coordinate descent optimization.
*   **Hands-on Deliverable**: `Day 007 - Ridge & Lasso Regression.ipynb (Sparsity visualization plotting coefficient paths as alpha varies).`
*   **Interview Edge (Advanced Attributions)**: *"Provide a rigorous mathematical explanation of why Lasso (L1) results in sparse coefficients while Ridge (L2) only shrinks them toward zero."*

---

### 📅 Day 008 | 08 Jun 2026, Mon
*   **Phase**: Phase 1: Linear Regression & Regularization
*   **Advanced Topic**: **Elastic Net Regression**
*   **Deep-Dive Subtopics**: Elastic Net objective function, l1_ratio hyperparameter, combining L1/L2 penalties, when to use Elastic Net over Lasso (grouped variable selection).
*   **Hands-on Deliverable**: `Day 008 - Elastic Net Regression.ipynb (Tuning l1_ratio and alpha using cross-validation).`
*   **Interview Edge (Advanced Attributions)**: *"Explain the grouping effect of Elastic Net and why it is mathematically superior to Lasso when dealing with highly correlated features."*

---

### 📅 Day 009 | 09 Jun 2026, Tue
*   **Phase**: Phase 1: Linear Regression & Regularization
*   **Advanced Topic**: **SGD Regressor & Online Learning**
*   **Deep-Dive Subtopics**: Stochastic Gradient Descent for regression, large-scale training, learning rate schedules (optimal, inverse scaling, adaptive), partial_fit API.
*   **Hands-on Deliverable**: `Day 009 - SGD Regressor.ipynb (Online learning simulation updating model coefficients on streaming data chunks).`
*   **Interview Edge (Advanced Attributions)**: *"How does online learning with SGD Regressor handle out-of-core datasets that do not fit into RAM?"*

---

### 📅 Day 010 | 10 Jun 2026, Wed
*   **Phase**: Phase 1: Linear Regression & Regularization
*   **Advanced Topic**: **Mini Regression Project App Planning**
*   **Deep-Dive Subtopics**: Designing robust architectures: mapping UI inputs to preprocessing pipelines, model serialization options (pickle vs. joblib vs. ONNX), serialization security.
*   **Hands-on Deliverable**: `Day 010 - Mini Regression App Planning.ipynb (Designing a standard model API contract and schema mappings).`
*   **Interview Edge (Advanced Attributions)**: *"Why is pickling a Python model considered a security vulnerability in enterprise environments, and what are the safer alternatives?"*

---

### 📅 Day 011 | 11 Jun 2026, Thu
*   **Phase**: Phase 1: Linear Regression & Regularization
*   **Advanced Topic**: **House Price Prediction Project**
*   **Deep-Dive Subtopics**: End-to-End Pipeline: Ames Housing dataset, advanced EDA, log transformations of skewed targets, multi-stage pipelines (scaling + regularized model).
*   **Hands-on Deliverable**: `Day 11 - House Price Prediction Project.ipynb (Complete end-to-end model pipeline yielding top 10% Kaggle-equivalent score).`
*   **Interview Edge (Advanced Attributions)**: *"How do you mathematically treat features that are highly skewed before feeding them to a linear model? Describe the Box-Cox transformation."*

---

### 📅 Day 012 | 12 Jun 2026, Fri
*   **Phase**: Phase 1: Linear Regression & Regularization
*   **Advanced Topic**: **Simple UI Prediction Integration**
*   **Deep-Dive Subtopics**: Linking serialized model files to interactive widgets, converting raw dictionary payloads into model inputs, running real-time predictions.
*   **Hands-on Deliverable**: `Day 12 - Simple UI Integration.ipynb (Jupyter-widgets based interactive real-time house prediction dashboard).`
*   **Interview Edge (Advanced Attributions)**: *"Why is it critical to apply the *exact same* scaling parameters fitted on the training set to the real-time inference payloads?"*

---

### 📅 Day 013 | 13 Jun 2026, Sat
*   **Phase**: Phase 1: Linear Regression & Regularization
*   **Advanced Topic**: **Regression Project Polish & Wrap-up**
*   **Deep-Dive Subtopics**: Deploying static regression APIs, code modularization, profiling inference times, creating diagnostic unit tests for the pipeline.
*   **Hands-on Deliverable**: `Day 13 - Regression Mini Project Polish.ipynb (Productionized, modular python script directory for house pricing model).`
*   **Interview Edge (Advanced Attributions)**: *"How do you write a unit test to verify that your data processing pipeline handles missing categorical features without throwing a runtime error?"*

---

### 📅 Day 014 | 14 Jun 2026, Sun
*   **Phase**: Phase 2: Logistic Regression & Classification
*   **Advanced Topic**: **Logistic Regression Intuition & Sigmoid**
*   **Deep-Dive Subtopics**: Sigmoid function mathematical derivation, odds ratio and log-odds (logit), Bernoulli maximum likelihood estimation, Log Loss (Binary Cross-Entropy) cost function derivation.
*   **Hands-on Deliverable**: `Day 14 - Logistic Regression Intuition & Sigmoid.ipynb (Plotting sigmoid, deriving gradient of log loss mathematically).`
*   **Interview Edge (Advanced Attributions)**: *"Derive the gradient of the Binary Cross-Entropy loss function with respect to the weight vector in Logistic Regression."*

---

### 📅 Day 015 | 15 Jun 2026, Mon
*   **Phase**: Phase 2: Logistic Regression & Classification
*   **Advanced Topic**: **Logistic Regression Implementation & Metrics**
*   **Deep-Dive Subtopics**: Training LogisticRegression, predicting classification probabilities vs. class indices, Accuracy, Precision, Recall, and F1-score mathematical formulations.
*   **Hands-on Deliverable**: `Day 15 - Logistic Regression Implementation & Metrics.ipynb (Custom metric calculator from a confusion matrix without using scikit-learn).`
*   **Interview Edge (Advanced Attributions)**: *"If your model has high recall but low precision, what does that imply about your model's classification thresholds and predictions?"*

---

### 📅 Day 016 | 16 Jun 2026, Tue
*   **Phase**: Phase 2: Logistic Regression & Classification
*   **Advanced Topic**: **Multiclass Classification (OvR vs OvO)**
*   **Deep-Dive Subtopics**: One-vs-Rest (OvR) vs. One-vs-One (OvO) decision boundaries, computational complexity differences during training and inference.
*   **Hands-on Deliverable**: `Day 16 - Multiclass Classification.ipynb (Visualizing OvR vs OvO decision boundaries on 2D synthetic dataset).`
*   **Interview Edge (Advanced Attributions)**: *"If you have $k$ classes, how many binary models must be trained for OvO versus OvR? Compare the training times."*

---

### 📅 Day 017 | 17 Jun 2026, Wed
*   **Phase**: Phase 2: Logistic Regression & Classification
*   **Advanced Topic**: **Softmax Regression (Multinomial Logistic)**
*   **Deep-Dive Subtopics**: Softmax activation function, mapping logits to probability distributions, Multinomial Cross-Entropy loss formulation, gradient of Softmax loss.
*   **Hands-on Deliverable**: `Day 17 - Softmax Regression.ipynb (Implementing Softmax regression forward pass and cross-entropy evaluation in NumPy).`
*   **Interview Edge (Advanced Attributions)**: *"Show mathematically how Softmax regression collapses to binary Logistic Regression when the number of classes $K = 2$."*

---

### 📅 Day 018 | 18 Jun 2026, Thu
*   **Phase**: Phase 2: Logistic Regression & Classification
*   **Advanced Topic**: **ROC-AUC & PR-AUC Curves**
*   **Deep-Dive Subtopics**: True Positive Rate (TPR) vs. False Positive Rate (FPR) curve derivation, Area Under Curve (AUC), Precision-Recall (PR) curves, handling highly imbalanced distributions.
*   **Hands-on Deliverable**: `Day 18 - ROC-AUC & PR-AUC Curves.ipynb (Plotting ROC and PR curves from scratch as threshold varies from 0 to 1).`
*   **Interview Edge (Advanced Attributions)**: *"Why is PR-AUC a much more informative metric than ROC-AUC when evaluating models trained on highly imbalanced datasets (e.g., fraud detection)?"*

---

### 📅 Day 019 | 19 Jun 2026, Fri
*   **Phase**: Phase 2: Logistic Regression & Classification
*   **Advanced Topic**: **Multiclass Pipeline & Evaluation**
*   **Deep-Dive Subtopics**: Standard scaling, multiclass classification on the Iris dataset, micro-averaged, macro-averaged, and weighted F1-scores.
*   **Hands-on Deliverable**: `Day 19 - Multiclass Pipeline & Evaluation.ipynb (Full classification pipeline with detailed classification reports).`
*   **Interview Edge (Advanced Attributions)**: *"What is the mathematical difference between Macro-F1 and Micro-F1, and when should you prioritize each?"*

---

### 📅 Day 020 | 20 Jun 2026, Sat
*   **Phase**: Phase 2: Logistic Regression & Classification
*   **Advanced Topic**: **Regularization in Logistic Regression**
*   **Deep-Dive Subtopics**: Tuning `C` parameter (inverse regularization strength), L1 vs. L2 penalties in log loss, visualizing weight decay as `C` decreases.
*   **Hands-on Deliverable**: `Day 20 - Regularization in Logistic Regression.ipynb (Plotting feature coefficients vs. $C$ parameter value).`
*   **Interview Edge (Advanced Attributions)**: *"How does the $C$ parameter in scikit-learn's LogisticRegression relate to the alpha regularization parameter in linear regression?"*

---

### 📅 Day 021 | 21 Jun 2026, Sun
*   **Phase**: Phase 2: Logistic Regression & Classification
*   **Advanced Topic**: **Breast Cancer Classification Project**
*   **Deep-Dive Subtopics**: Applying stratified splits, standard scaling, training L1/L2 regularized Logistic Regression on Breast Cancer dataset, detailed classification evaluations.
*   **Hands-on Deliverable**: `Day 21 - Breast Cancer Classification Project.ipynb (Polished breast cancer diagnosis classifier with ROC curve analysis).`
*   **Interview Edge (Advanced Attributions)**: *"What is the clinical risk of prioritizing Precision over Recall in medical diagnostic models?"*

---

### 📅 Day 022 | 22 Jun 2026, Mon
*   **Phase**: Phase 3: Tree Models & SVM
*   **Advanced Topic**: **Decision Tree — Intuition & Mathematics**
*   **Deep-Dive Subtopics**: Splitting criteria: Shannon Entropy, Gini Impurity, Information Gain mathematical formulations, CART splitting algorithm math.
*   **Hands-on Deliverable**: `Day 22 - Decision Tree Intuition.ipynb (Custom function computing Gini Impurity and Entropy for potential splits).`
*   **Interview Edge (Advanced Attributions)**: *"Why is Gini Impurity computationally faster than Entropy during decision tree construction?"*

---

### 📅 Day 023 | 23 Jun 2026, Tue
*   **Phase**: Phase 3: Tree Models & SVM
*   **Advanced Topic**: **Decision Tree — Implementation**
*   **Deep-Dive Subtopics**: Training `DecisionTreeClassifier`, hyperparameter analysis: `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`.
*   **Hands-on Deliverable**: `Day 23 - Decision Tree Implementation.ipynb (Analyzing decision tree training shapes as hyperparameters are restricted).`
*   **Interview Edge (Advanced Attributions)**: *"How do restriction hyperparameters like `min_samples_leaf` mathematically affect the bias and variance of a decision tree?"*

---

### 📅 Day 024 | 24 Jun 2026, Wed
*   **Phase**: Phase 3: Tree Models & SVM
*   **Advanced Topic**: **Decision Tree — Overfitting & Pruning**
*   **Deep-Dive Subtopics**: Cost-Complexity Pruning (CCP), the mathematical objective of CCP ($R_a(T) = R(T) + a|T|$), finding optimal `ccp_alpha` using grid search.
*   **Hands-on Deliverable**: `Day 24 - Decision Tree Pruning.ipynb (Generating ccp_alpha paths and plotting model accuracy vs. alpha).`
*   **Interview Edge (Advanced Attributions)**: *"Explain cost-complexity pruning and how the parameter alpha balances structural tree complexity with training error."*

---

### 📅 Day 025 | 25 Jun 2026, Thu
*   **Phase**: Phase 3: Tree Models & SVM
*   **Advanced Topic**: **Decision Tree — Regressor & Comparison**
*   **Deep-Dive Subtopics**: Splitting criteria for regression trees (reducing Variance/MSE), training `DecisionTreeRegressor`, comparing decision boundaries with linear models.
*   **Hands-on Deliverable**: `Day 25 - Decision Tree Regressor.ipynb (Fitting continuous sine wave targets, plotting step-function decision tree boundaries).`
*   **Interview Edge (Advanced Attributions)**: *"Why do decision trees produce highly non-linear, discontinuous step-function decision boundaries, and how does this differ from linear models?"*

---

### 📅 Day 026 | 26 Jun 2026, Fri
*   **Phase**: Phase 3: Tree Models & SVM
*   **Advanced Topic**: **SVM — Support Vector Machine Intuition**
*   **Deep-Dive Subtopics**: Maximum margin formulation, Support vectors identification, hard vs. soft margins, hinge loss optimization, soft margin slack variables ($e_i$).
*   **Hands-on Deliverable**: `Day 26 - SVM Intuition.ipynb (Visualizing support vectors and classification margin boundaries as C varies).`
*   **Interview Edge (Advanced Attributions)**: *"How does the slack variable formulation in Soft-Margin SVM balance maximum margin separation with training classification errors?"*

---

### 📅 Day 027 | 27 Jun 2026, Sat
*   **Phase**: Phase 3: Tree Models & SVM
*   **Advanced Topic**: **SVM — Kernel Trick & Implementation**
*   **Deep-Dive Subtopics**: Mercer's Theorem, the Kernel Trick, mathematical formulation of Polynomial, RBF (Radial Basis Function), and Sigmoid kernels.
*   **Hands-on Deliverable**: `Day 27 - SVM Kernels.ipynb (SVM boundary mappings using Linear, Polynomial, and RBF kernels on non-linear datasets).`
*   **Interview Edge (Advanced Attributions)**: *"Explain the Kernel Trick mathematically and explain why it allows us to optimize in infinite-dimensional spaces without explicit mapping."*

---

### 📅 Day 028 | 28 Jun 2026, Sun
*   **Phase**: Phase 3: Tree Models & SVM
*   **Advanced Topic**: **SVM — Hyperparameter Tuning**
*   **Deep-Dive Subtopics**: Optimizing `C` and `gamma` parameters on an RBF kernel, mapping the speed/accuracy/memory trade-offs of SVMs on large datasets.
*   **Hands-on Deliverable**: `Day 28 - SVM Tuning.ipynb (Grid search optimization of SVC hyperparameters, plotting accuracy heatmaps).`
*   **Interview Edge (Advanced Attributions)**: *"What is the physical meaning of the `gamma` parameter in an RBF kernel, and how does its value affect model bias and variance?"*

---

### 📅 Day 029 | 29 Jun 2026, Mon
*   **Phase**: Phase 3: Tree Models & SVM
*   **Advanced Topic**: **Naive Bayes — Theory & Variants**
*   **Deep-Dive Subtopics**: Bayes theorem, conditional independence assumption, GaussianNB, MultinomialNB, and BernoulliNB formulations, Laplace smoothing math.
*   **Hands-on Deliverable**: `Day 29 - Naive Bayes.ipynb (Building a Multinomial Naive Bayes classifier with Laplace smoothing from scratch).`
*   **Interview Edge (Advanced Attributions)**: *"Why do we need Laplace smoothing in Naive Bayes, and what is the mathematical consequence of setting the smoothing parameter alpha to zero?"*

---

### 📅 Day 030 | 30 Jun 2026, Tue
*   **Phase**: Phase 3: Tree Models & SVM
*   **Advanced Topic**: **KNN — k-Nearest Neighbors**
*   **Deep-Dive Subtopics**: Distance metrics (Minkowski, Manhattan, Euclidean, Chebyshev), choosing optimal K, computational complexity of brute force search $O(d \cdot N)$.
*   **Hands-on Deliverable**: `Day 30 - KNN Basics.ipynb (KNN classifier implementation using custom distance calculations).`
*   **Interview Edge (Advanced Attributions)**: *"What is the Curse of Dimensionality, and why does it render distance-based algorithms like KNN highly ineffective in high-dimensional spaces?"*

---

### 📅 Day 031 | 01 Jul 2026, Wed
*   **Phase**: Phase 3: Tree Models & SVM
*   **Advanced Topic**: **KNN — Implementation & Scaling Impact**
*   **Deep-Dive Subtopics**: Training KNeighborsClassifier, demonstrating the absolute requirement of standard scaling, distance-weighted KNN formulations.
*   **Hands-on Deliverable**: `Day 31 - KNN Implementation.ipynb (Demonstrating classification shifts with/without standard scaling).`
*   **Interview Edge (Advanced Attributions)**: *"How does distance weighting inside KNN mathematically alter the final class probability calculation?"*

---

### 📅 Day 032 | 02 Jul 2026, Thu
*   **Phase**: Phase 3: Tree Models & SVM
*   **Advanced Topic**: **Model Selection Guidelines & Leaderboard**
*   **Deep-Dive Subtopics**: Setting up standard dataset evaluations across Scikit-Learn classifiers, tracking metrics, profiling memory usage, and execution speed.
*   **Hands-on Deliverable**: `Day 32 - Model Leaderboard.ipynb (Establishing a master comparison notebook showing test accuracy, F1, and training latency).`
*   **Interview Edge (Advanced Attributions)**: *"If your model must perform real-time inference in under 5 milliseconds on a CPU, why might you select Logistic Regression over SVM or KNN?"*

---

### 📅 Day 033 | 03 Jul 2026, Fri
*   **Phase**: Phase 3: Tree Models & SVM
*   **Advanced Topic**: **Classification Project — Algorithm Selection**
*   **Deep-Dive Subtopics**: Ingesting a clinical classification dataset, building preprocessors, evaluating baseline algorithms, and selecting candidates based on precision-recall requirements.
*   **Hands-on Deliverable**: `Day 33 - Classification Project.ipynb (Robust pipeline identifying candidate models for target dataset).`
*   **Interview Edge (Advanced Attributions)**: *"How do you determine whether a classification dataset's features should undergo standard scaling vs. normalization for optimal model performance?"*

---

### 📅 Day 034 | 04 Jul 2026, Sat
*   **Phase**: Phase 3: Tree Models & SVM
*   **Advanced Topic**: **Ensemble Methods — Voting & Bagging**
*   **Deep-Dive Subtopics**: VotingClassifier (hard vs. soft voting calculations), BaggingClassifier (bootstrap aggregation), out-of-bag (OOB) score evaluation math.
*   **Hands-on Deliverable**: `Day 34 - Voting Bagging Ensembles.ipynb (Implementing hard and soft voting on base classifiers).`
*   **Interview Edge (Advanced Attributions)**: *"Explain mathematically why soft voting is typically superior to hard voting in ensemble classification."*

---

### 📅 Day 035 | 05 Jul 2026, Sun
*   **Phase**: Phase 3: Tree Models & SVM
*   **Advanced Topic**: **Random Forest — Theory & Implementation**
*   **Deep-Dive Subtopics**: Bagging + random feature subspace splits, mathematically reducing correlation between tree models to decrease final ensemble variance.
*   **Hands-on Deliverable**: `Day 035 - Random Forest Basics.ipynb (Custom script training forest of trees and tracking out-of-bag training validation).`
*   **Interview Edge (Advanced Attributions)**: *"How does Random Forest mathematically guarantee a reduction in variance compared to a single decision tree? Derive the ensemble variance formula."*

---

### 📅 Day 036 | 06 Jul 2026, Mon
*   **Phase**: Phase 3: Tree Models & SVM
*   **Advanced Topic**: **Random Forest — Hyperparameter Tuning**
*   **Deep-Dive Subtopics**: Optimizing forest hyperparameter combinations (depth, estimator count, sample restrictions) using modern Bayesian optimization (Optuna).
*   **Hands-on Deliverable**: `Day 036 - Random Forest Tuning.ipynb (An Optuna study tuning forest models and plotting hyperparameter parallel coordinate graphs).`
*   **Interview Edge (Advanced Attributions)**: *"How does Optuna use Tree-structured Parzen Estimators (TPE) to find hyperparameter combinations faster than random search?"*

---

### 📅 Day 037 | 07 Jul 2026, Tue
*   **Phase**: Phase 3: Tree Models & SVM
*   **Advanced Topic**: **Feature Importance Analysis**
*   **Deep-Dive Subtopics**: Mean Decrease in Impurity (MDI) bias towards continuous/high-cardinality features, Permutation Importance math, selecting robust feature subsets.
*   **Hands-on Deliverable**: `Day 037 - Feature Importance.ipynb (Comparing Scikit-Learn MDI importance vs. permutation feature importance).`
*   **Interview Edge (Advanced Attributions)**: *"Why is impurity-based feature importance biased towards high-cardinality categorical features, and how does Permutation Importance fix this?"*

---

### 📅 Day 038 | 08 Jul 2026, Wed
*   **Phase**: Phase 3: Tree Models & SVM
*   **Advanced Topic**: **Phase Review & Trees/SVM Assessment**
*   **Deep-Dive Subtopics**: Answering advanced interview questions, analyzing decision boundaries, and reviewing the mathematical trade-offs between SVM, Tree Models, and KNN.
*   **Hands-on Deliverable**: `Day 038 - Trees SVM Assessment.ipynb (A comprehensive notebook containing 20 written question answers and boundary visualizations).`
*   **Interview Edge (Advanced Attributions)**: *"Explain the structural and optimization differences between Support Vector Machines and Decision Trees. When is each preferred?"*

---

### 📅 Day 039 | 09 Jul 2026, Thu
*   **Phase**: Phase 4: Boosting & Advanced Ensembles
*   **Advanced Topic**: **AdaBoost — Adaptive Boosting**
*   **Deep-Dive Subtopics**: Exponential loss optimization, computing weak learner coefficient $\alpha_t$, step-by-step sample weight updates, updating weights of misclassified samples.
*   **Hands-on Deliverable**: `Day 039 - AdaBoost.ipynb (Implementing custom AdaBoost with simple decision stumps from scratch).`
*   **Interview Edge (Advanced Attributions)**: *"Derive the step-by-step formula for the estimator weight alpha in AdaBoost starting from exponential loss minimization."*

---

### 📅 Day 040 | 10 Jul 2026, Fri
*   **Phase**: Phase 4: Boosting & Advanced Ensembles
*   **Advanced Topic**: **Gradient Boosting — Theory & Implementation**
*   **Deep-Dive Subtopics**: Sequential residual fitting, Gradient Boosting math formulation, shrinking estimators via learning rate, custom loss functions.
*   **Hands-on Deliverable**: `Day 040 - Gradient Boosting.ipynb (Implementing custom Gradient Boosting regression tree algorithm).`
*   **Interview Edge (Advanced Attributions)**: *"Why can Gradient Boosting be interpreted as performing gradient descent in function space?"*

---

### 📅 Day 041 | 11 Jul 2026, Sat
*   **Phase**: Phase 4: Boosting & Advanced Ensembles
*   **Advanced Topic**: **XGBoost — Setup & First Model**
*   **Deep-Dive Subtopics**: XGBoost regularized objective function, Taylor expansion approximation of loss, structural score calculation of nodes, training first XGBClassifier.
*   **Hands-on Deliverable**: `Day 041 - XGBoost Intro.ipynb (Installing XGBoost and training classification models on baseline tabular data).`
*   **Interview Edge (Advanced Attributions)**: *"How does XGBoost mathematically incorporate regularization directly into its tree-splitting criteria?"*

---

### 📅 Day 042 | 12 Jul 2026, Sun
*   **Phase**: Phase 4: Boosting & Advanced Ensembles
*   **Advanced Topic**: **XGBoost — Hyperparameter Deep Dive**
*   **Deep-Dive Subtopics**: Deep-dive into parameters: `colsample_bytree`, `subsample`, `scale_pos_weight`, `reg_alpha` (L1), and `reg_lambda` (L2).
*   **Hands-on Deliverable**: `Day 042 - XGBoost Tuning.ipynb (Systematic hyperparameter search using grid search and Optuna).`
*   **Interview Edge (Advanced Attributions)**: *"How does `scale_pos_weight` mathematically adjust the loss computation to handle imbalanced datasets in XGBoost?"*

---

### 📅 Day 043 | 13 Jul 2026, Mon
*   **Phase**: Phase 4: Boosting & Advanced Ensembles
*   **Advanced Topic**: **XGBoost — Early Stopping & Callbacks**
*   **Deep-Dive Subtopics**: Custom callbacks, tracking cross-validation evaluation sets, early_stopping_rounds to prevent overfitting in boosting loops.
*   **Hands-on Deliverable**: `Day 043 - XGBoost Early Stopping.ipynb (Plotting train vs. validation log-loss curves and finding optimal iteration limits).`
*   **Interview Edge (Advanced Attributions)**: *"What is the mathematical risk of setting the early stopping rounds too low during high-learning-rate boosting training?"*

---

### 📅 Day 044 | 14 Jul 2026, Tue
*   **Phase**: Phase 4: Boosting & Advanced Ensembles
*   **Advanced Topic**: **LightGBM — Leaf-wise Growth & Speed**
*   **Deep-Dive Subtopics**: Leaf-wise (best-first) growth strategy, Gradient-based One-Side Sampling (GOSS), Exclusive Feature Bundling (EFB), histogram-based splitting.
*   **Hands-on Deliverable**: `Day 044 - LightGBM.ipynb (Implementing high-performance LightGBM models on high-cardinality datasets).`
*   **Interview Edge (Advanced Attributions)**: *"Explain mathematically how Gradient-based One-Side Sampling (GOSS) speeds up training without losing model accuracy."*

---

### 📅 Day 045 | 15 Jul 2026, Wed
*   **Phase**: Phase 4: Boosting & Advanced Ensembles
*   **Advanced Topic**: **CatBoost — Native Categorical Handling**
*   **Deep-Dive Subtopics**: Ordered boosting math, symmetric trees construction, on-the-fly calculation of target statistics with leakage protection.
*   **Hands-on Deliverable**: `Day 045 - CatBoost.ipynb (CatBoost training with complex raw object features without pre-encoding).`
*   **Interview Edge (Advanced Attributions)**: *"How does CatBoost's ordered boosting mathematically prevent target leakage during categorical target encoding?"*

---

### 📅 Day 046 | 16 Jul 2026, Thu
*   **Phase**: Phase 4: Boosting & Advanced Ensembles
*   **Advanced Topic**: **Ensemble Comparison — RF vs. XGB vs. LGBM vs. CatBoost**
*   **Deep-Dive Subtopics**: Comprehensive benchmark: tracking training speed, memory footprints, and testing metrics across classical and boosting architectures.
*   **Hands-on Deliverable**: `Day 046 - Boosting Benchmark.ipynb (Benchmarking forest and boosting engines on single large dataset).`
*   **Interview Edge (Advanced Attributions)**: *"Under what structural data conditions is CatBoost superior to LightGBM or XGBoost?"*

---

### 📅 Day 047 | 17 Jul 2026, Fri
*   **Phase**: Phase 4: Boosting & Advanced Ensembles
*   **Advanced Topic**: **Stacking & Blending Ensembles**
*   **Deep-Dive Subtopics**: StackingClassifier implementation, Level-0 base estimators, Level-1 meta-estimator, preventing target leakage using out-of-fold predictions.
*   **Hands-on Deliverable**: `Day 047 - Stacking Ensembles.ipynb (Multi-level classification stack showing improved validation score).`
*   **Interview Edge (Advanced Attributions)**: *"Why must we use out-of-fold (OOF) predictions when training the Level-1 meta-estimator in a stacking architecture?"*

---

### 📅 Day 048 | 18 Jul 2026, Sat
*   **Phase**: Phase 4: Boosting & Advanced Ensembles
*   **Advanced Topic**: **Kaggle Competition — Setup, EDA & Baseline**
*   **Deep-Dive Subtopics**: Establishing competitive environments: Kaggle API data download, descriptive exploratory analysis, fitting a robust RandomForest baseline.
*   **Hands-on Deliverable**: `Day 048 - Kaggle Baseline.ipynb (EDA log-distribution checks and basic submission generation).`
*   **Interview Edge (Advanced Attributions)**: *"How does performing log-transformations of highly skewed features mathematically benefit boosting algorithms?"*

---

### 📅 Day 049 | 19 Jul 2026, Sun
*   **Phase**: Phase 4: Boosting & Advanced Ensembles
*   **Advanced Topic**: **Kaggle — Validation Strategy**
*   **Deep-Dive Subtopics**: StratifiedKFold validation strategies, evaluating validation stability, mapping local CV scores to public leaderboards.
*   **Hands-on Deliverable**: `Day 049 - Kaggle Validation.ipynb (Writing cross-validation helper functions that return clean OOF arrays).`
*   **Interview Edge (Advanced Attributions)**: *"How do you mathematically diagnose model overfitting by comparing your local CV scores to a Kaggle public leaderboard score?"*

---

### 📅 Day 050 | 20 Jul 2026, Mon
*   **Phase**: Phase 4: Boosting & Advanced Ensembles
*   **Advanced Topic**: **Kaggle — Feature Engineering**
*   **Deep-Dive Subtopics**: Target-guided mean encodings, numerical feature interactions, categorical frequency mapping, handling unseen categories.
*   **Hands-on Deliverable**: `Day 050 - Kaggle Feature Engineering.ipynb (Generating custom features on a housing/tabular competition dataset).`
*   **Interview Edge (Advanced Attributions)**: *"Why is target-guided mean encoding susceptible to target leakage, and how do you implement smoothing to prevent it?"*

---

### 📅 Day 051 | 21 Jul 2026, Tue
*   **Phase**: Phase 4: Boosting & Advanced Ensembles
*   **Advanced Topic**: **Kaggle — Hyperparameter Tuning (Optuna)**
*   **Deep-Dive Subtopics**: Writing customized Optuna objective functions, defining continuous/discrete hyperparameter search spaces, running accelerated studies.
*   **Hands-on Deliverable**: `Day 051 - Kaggle Optuna Tuning.ipynb (A parallelized Optuna tuning loop optimizing LightGBM parameters).`
*   **Interview Edge (Advanced Attributions)**: *"Explain how the covariance matrix adaptation evolution strategy (CMA-ES) can be used inside Optuna for advanced hyperparameter searches."*

---

### 📅 Day 052 | 22 Jul 2026, Wed
*   **Phase**: Phase 4: Boosting & Advanced Ensembles
*   **Advanced Topic**: **Kaggle — Ensemble Blending & Final Submission**
*   **Deep-Dive Subtopics**: Averaging predictions, geometric mean blending, ranking correlations of model predictions, constructing the final submission file.
*   **Hands-on Deliverable**: `Day 052 - Kaggle Final Blending.ipynb (Generating optimal ensemble blend probabilities based on model correlation matrices).`
*   **Interview Edge (Advanced Attributions)**: *"Why is it optimal to blend models that have highly uncorrelated prediction probabilities?"*

---

### 📅 Day 053 | 23 Jul 2026, Thu
*   **Phase**: Phase 4: Boosting & Advanced Ensembles
*   **Advanced Topic**: **Phase Review & Boosting Assessment**
*   **Deep-Dive Subtopics**: Review of adaptive/gradient boosting math, structured comparisons, answering complex boosting architectural interview questions.
*   **Hands-on Deliverable**: `Day 053 - Boosting Assessment.ipynb (Assessment script checking conceptual model properties).`
*   **Interview Edge (Advanced Attributions)**: *"How does leaf-wise growth in LightGBM differ from level-wise growth in XGBoost, and what are the respective risks of overfitting?"*

---

### 📅 Day 054 | 24 Jul 2026, Fri
*   **Phase**: Phase 5: Advanced Preprocessing, Imbalanced Learning & XAI
*   **Advanced Topic**: **Advanced Imputation Methods**
*   **Deep-Dive Subtopics**: KNNImputer (n_neighbors, weights), MICE (Multivariate Imputation by Chained Equations) via IterativeImputer in Scikit-Learn.
*   **Hands-on Deliverable**: `Day 054 - Advanced Imputation.ipynb (Comparing univariate Mean imputation vs. KNN and MICE imputation on validation sets).`
*   **Interview Edge (Advanced Attributions)**: *"Explain the MICE algorithm mathematically and why it is superior to simple univariate imputation for missing data."*

---

### 📅 Day 055 | 25 Jul 2026, Sat
*   **Phase**: Phase 5: Advanced Preprocessing, Imbalanced Learning & XAI
*   **Advanced Topic**: **Outlier Treatment & Isolation Forests**
*   **Deep-Dive Subtopics**: IsolationForest anomaly scores, contamination parameters, evaluating robust scaling methods, automated outlier removal pipelines.
*   **Hands-on Deliverable**: `Day 055 - Isolation Forest Outliers.ipynb (IsolationForest outlier identification and feature scaling validation).`
*   **Interview Edge (Advanced Attributions)**: *"Explain the mathematics of path length calculation inside an Isolation Forest and why anomalies have shorter path lengths."*

---

### 📅 Day 056 | 26 Jul 2026, Sun
*   **Phase**: Phase 5: Advanced Preprocessing, Imbalanced Learning & XAI
*   **Advanced Topic**: **Advanced Categorical Encoding**
*   **Deep-Dive Subtopics**: Target encoding with smoothing, m-estimate encoding, handling high-cardinality categorical columns, weight of evidence (WoE) encoding.
*   **Hands-on Deliverable**: `Day 056 - Target Encoding.ipynb (Custom target encoder implementation with regularized smoothing).`
*   **Interview Edge (Advanced Attributions)**: *"Derive the formula for smoothed target encoding and explain how the prior probability balances the empirical target mean of a category."*

---

### 📅 Day 057 | 27 Jul 2026, Mon
*   **Phase**: Phase 5: Advanced Preprocessing, Imbalanced Learning & XAI
*   **Advanced Topic**: **Power Transformers & Quantile Transformers**
*   **Deep-Dive Subtopics**: PowerTransformer (Box-Cox, Yeo-Johnson equations), QuantileTransformer (uniform/normal target mappings) for stabilizing variance.
*   **Hands-on Deliverable**: `Day 057 - Power Transforms.ipynb (Visualizing skewed distributions before and after Yeo-Johnson and Quantile transformations).`
*   **Interview Edge (Advanced Attributions)**: *"Under what conditions would a QuantileTransformer be preferred over a Box-Cox PowerTransformer?"*

---

### 📅 Day 058 | 28 Jul 2026, Tue
*   **Phase**: Phase 5: Advanced Preprocessing, Imbalanced Learning & XAI
*   **Advanced Topic**: **Feature Selection Methods**
*   **Deep-Dive Subtopics**: Recursive Feature Elimination with CV (RFECV), L1-based feature selection using SelectFromModel, variance thresholds.
*   **Hands-on Deliverable**: `Day 058 - Feature Selection.ipynb (Feature selection comparison checking impact on model generalization).`
*   **Interview Edge (Advanced Attributions)**: *"How does L1 regularization mathematically function as a feature selector, and why does L2 regularization fail to yield sparsity?"*

---

### 📅 Day 059 | 29 Jul 2026, Wed
*   **Phase**: Phase 5: Advanced Preprocessing, Imbalanced Learning & XAI
*   **Advanced Topic**: **Spline Features & Binning**
*   **Deep-Dive Subtopics**: SplineTransformer (fitting piecewise polynomials), KBinsDiscretizer, encoding non-linear spatial dependencies into linear estimators.
*   **Hands-on Deliverable**: `Day 059 - Splines Binning.ipynb (Fitting continuous cyclical data with B-splines to train a robust linear regression model).`
*   **Interview Edge (Advanced Attributions)**: *"How do spline features mathematically enable a linear regression model to fit highly complex, non-linear functions?"*

---

### 📅 Day 060 | 30 Jul 2026, Thu
*   **Phase**: Phase 5: Advanced Preprocessing, Imbalanced Learning & XAI
*   **Advanced Topic**: **ColumnTransformers & Pipelines in Sklearn**
*   **Deep-Dive Subtopics**: Building Scikit-Learn pipelines, utilizing ColumnTransformer, combining Numeric and Categorical paths, Custom Transformer subclasses.
*   **Hands-on Deliverable**: `Day 060 - Sklearn Pipelines.ipynb (Constructing modular, reproducible, high-throughput preprocessing pipelines).`
*   **Interview Edge (Advanced Attributions)**: *"What is the physical meaning of `fit_transform` versus `transform` in Scikit-Learn pipelines, and why must you never call `fit` on test data?"*

---

### 📅 Day 061 | 31 Jul 2026, Fri
*   **Phase**: Phase 5: Advanced Preprocessing, Imbalanced Learning & XAI
*   **Advanced Topic**: **Data Leakage Prevention in Production**
*   **Deep-Dive Subtopics**: Target leakage, data contamination audits, building leak-proof pipeline architectures for real-time inference environments.
*   **Hands-on Deliverable**: `Day 061 - Data Leakage.ipynb (Automated script scanning raw code pipelines and checking for temporal/distribution leakages).`
*   **Interview Edge (Advanced Attributions)**: *"What is data leakage? Provide three distinct examples of how data leakage can slip into standard machine learning preprocessing loops."*

---

### 📅 Day 062 | 01 Aug 2026, Sat
*   **Phase**: Phase 5: Advanced Preprocessing, Imbalanced Learning & XAI
*   **Advanced Topic**: **Model Calibration & Probability Tuning**
*   **Deep-Dive Subtopics**: Platt Scaling, Isotonic Regression, CalibratedClassifierCV, reliability diagrams, Brier score mathematical evaluation.
*   **Hands-on Deliverable**: `Day 062 - Model Calibration.ipynb (Plotting reliability curves and calibrating models to obtain true probability predictions).`
*   **Interview Edge (Advanced Attributions)**: *"Derive the Brier score formulation. When is Isotonic Regression preferred over Platt Scaling for probability calibration?"*

---

### 📅 Day 063 | 02 Aug 2026, Sun
*   **Phase**: Phase 5: Advanced Preprocessing, Imbalanced Learning & XAI
*   **Advanced Topic**: **Imbalanced Classification — Metrics & Target Moving**
*   **Deep-Dive Subtopics**: Precision-Recall AUC (PR-AUC), F1 optimization, adjusting prediction decision thresholds using cost-benefit optimization matrices.
*   **Hands-on Deliverable**: `Day 063 - Imbalanced Metrics.ipynb (Running classification threshold sweep and finding optimal F1/cost-optimized boundaries).`
*   **Interview Edge (Advanced Attributions)**: *"Explain mathematically why Accuracy is a misleading metric for highly imbalanced datasets, and define the balanced accuracy formula."*

---

### 📅 Day 064 | 03 Aug 2026, Mon
*   **Phase**: Phase 5: Advanced Preprocessing, Imbalanced Learning & XAI
*   **Advanced Topic**: **Imbalanced Sampling (SMOTE & Variants)**
*   **Deep-Dive Subtopics**: Minority class generation: SMOTE math, ADASYN (Adaptive Synthetic sampling), undersampling with Tomek Links and Edited Nearest Neighbors.
*   **Hands-on Deliverable**: `Day 064 - Imbalanced Sampling.ipynb (Applying SMOTE + Tomek oversampling and comparing performance on boosted classifiers).`
*   **Interview Edge (Advanced Attributions)**: *"Explain the mathematical steps of SMOTE minority sample generation. What is the difference between SMOTE and Borderline-SMOTE?"*

---

### 📅 Day 065 | 04 Aug 2026, Tue
*   **Phase**: Phase 5: Advanced Preprocessing, Imbalanced Learning & XAI
*   **Advanced Topic**: **Cost-Sensitive Learning & Class Weights**
*   **Deep-Dive Subtopics**: Adjusting training loss via `class_weight` multipliers, setting custom sample weights, modifying boosting objective functions.
*   **Hands-on Deliverable**: `Day 065 - Cost Sensitive Learning.ipynb (Implementing class weights in Random Forest and XGBoost classifiers).`
*   **Interview Edge (Advanced Attributions)**: *"How does adding class weights mathematically alter the loss gradient computation during SGD optimization loops?"*

---

### 📅 Day 066 | 05 Aug 2026, Wed
*   **Phase**: Phase 5: Advanced Preprocessing, Imbalanced Learning & XAI
*   **Advanced Topic**: **SHAP — Shapley Additive Explanations**
*   **Deep-Dive Subtopics**: Game theory coalitions, TreeExplainer, generating summary plots, local water plots, global dependency and interaction plots.
*   **Hands-on Deliverable**: `Day 066 - SHAP Interpretability.ipynb (Explaining gradient-boosted model predictions using SHAP values).`
*   **Interview Edge (Advanced Attributions)**: *"Define the Shapley value equation and explain the efficiency, symmetry, dummy, and additivity axioms of SHAP."*

---

### 📅 Day 067 | 06 Aug 2026, Thu
*   **Phase**: Phase 5: Advanced Preprocessing, Imbalanced Learning & XAI
*   **Advanced Topic**: **LIME & Local Interpretability**
*   **Deep-Dive Subtopics**: Local surrogate model optimization, `LimeTabularExplainer` implementation, checking local coefficients, comparing SHAP vs. LIME.
*   **Hands-on Deliverable**: `Day 067 - LIME Interpretability.ipynb (Visualizing local feature attributions on single model predictions using LIME).`
*   **Interview Edge (Advanced Attributions)**: *"How does LIME construct a local surrogate model? Contrast LIME's local perturbation approach with SHAP's global game-theory approach."*

---

### 📅 Day 068 | 07 Aug 2026, Fri
*   **Phase**: Phase 5: Advanced Preprocessing, Imbalanced Learning & XAI
*   **Advanced Topic**: **Optuna — Hyperparameter Optimization Project**
*   **Deep-Dive Subtopics**: Building a unified codebase: integrating column transformers, SMOTE sampling, XGBoost, Optuna tuning, and SHAP explanations.
*   **Hands-on Deliverable**: `Day 068 - XAI Tuning Project.ipynb (Polished repository of an end-to-end explained, optimized tabular classifier).`
*   **Interview Edge (Advanced Attributions)**: *"How do you design a custom Optuna metric evaluation that optimizes for recall while keeping precision above a strict threshold?"*

---

### 📅 Day 069 | 08 Aug 2026, Sat
*   **Phase**: Phase 5: Advanced Preprocessing, Imbalanced Learning & XAI
*   **Advanced Topic**: **Phase Review & Preprocessing/XAI Assessment**
*   **Deep-Dive Subtopics**: Review of preprocessing mathematics, imbalanced classification, and explainable AI theory questions.
*   **Hands-on Deliverable**: `Day 069 - Preprocessing XAI Assessment.ipynb (Written solutions to 15 advanced tabular preprocessing and XAI interview questions).`
*   **Interview Edge (Advanced Attributions)**: *"Explain how data preprocessing choices (e.g., standard scaling vs. robust scaling) affect explainability metrics like SHAP and LIME."*

---

### 📅 Day 070 | 09 Aug 2026, Sun
*   **Phase**: Phase 6: Unsupervised Learning
*   **Advanced Topic**: **Clustering Concepts**
*   **Deep-Dive Subtopics**: Supervised vs. Unsupervised, distance metrics (Euclidean, Mahalanobis), intra-cluster density vs. inter-cluster distance.
*   **Hands-on Deliverable**: `Day 070 - Unsupervised Intro.ipynb (Implementing distance matrix computation using vectorized NumPy operations).`
*   **Interview Edge (Advanced Attributions)**: *"What is the Mahalanobis distance, and why is it mathematically superior to Euclidean distance when features are highly correlated?"*

---

### 📅 Day 071 | 10 Aug 2026, Mon
*   **Phase**: Phase 6: Unsupervised Learning
*   **Advanced Topic**: **K-Means Clustering — Intuition & Lloyd's Algorithm**
*   **Deep-Dive Subtopics**: Inertia definition, centroid updates, Lloyd's algorithm convergence proof, K-means++ initialization algorithm and probability math.
*   **Hands-on Deliverable**: `Day 071 - Kmeans Intuition.ipynb (Implementing Lloyd's KMeans clustering algorithm with custom initialization from scratch).`
*   **Interview Edge (Advanced Attributions)**: *"Prove mathematically why the K-Means algorithm is guaranteed to converge to a local minimum of the inertia function."*

---

### 📅 Day 072 | 11 Aug 2026, Tue
*   **Phase**: Phase 6: Unsupervised Learning
*   **Advanced Topic**: **K-Means — Elbow Method & Silhouette Analysis**
*   **Deep-Dive Subtopics**: Inertia curves vs. cluster count, Silhouette coefficient equation, plotting sample silhouette widths per cluster, profiling clusters.
*   **Hands-on Deliverable**: `Day 072 - Kmeans Analysis.ipynb (Silhouette plot generator and clustering profiling utility).`
*   **Interview Edge (Advanced Attributions)**: *"Define the Silhouette coefficient for a data point. What does a negative silhouette value indicate about a point's cluster assignment?"*

---

### 📅 Day 073 | 12 Aug 2026, Wed
*   **Phase**: Phase 6: Unsupervised Learning
*   **Advanced Topic**: **DBSCAN Clustering — Density-based**
*   **Deep-Dive Subtopics**: Core points, border points, noise points, mathematical definitions of density-reachability and density-connectivity, eps, min_samples.
*   **Hands-on Deliverable**: `Day 073 - DBSCAN Clustering.ipynb (Comparing DBSCAN and KMeans on complex, non-linear shapes like concentric rings).`
*   **Interview Edge (Advanced Attributions)**: *"Explain why DBSCAN is superior to KMeans when clusters have non-spherical shapes, and what happens when cluster densities vary widely."*

---

### 📅 Day 074 | 13 Aug 2026, Thu
*   **Phase**: Phase 6: Unsupervised Learning
*   **Advanced Topic**: **Hierarchical Clustering & Dendrograms**
*   **Deep-Dive Subtopics**: Agglomerative clustering, Single, Complete, Average, and Ward linkage criteria formulations, computing cophenetic distance, plotting dendrograms.
*   **Hands-on Deliverable**: `Day 074 - Hierarchical Clustering.ipynb (Building agglomerative pipelines, cutting dendrograms at dynamic thresholds).`
*   **Interview Edge (Advanced Attributions)**: *"Derive the Ward linkage criterion optimization objective. How does it mathematically balance variance minimization when merging clusters?"*

---

### 📅 Day 075 | 14 Aug 2026, Fri
*   **Phase**: Phase 6: Unsupervised Learning
*   **Advanced Topic**: **PCA — Principal Component Analysis Math & Variance**
*   **Deep-Dive Subtopics**: Orthogonal projections, maximizing projection variance, Covariance matrix derivation, eigenvectors and eigenvalues decomposition, SVD.
*   **Hands-on Deliverable**: `Day 075 - PCA Mathematics.ipynb (Implementing PCA from scratch using Singular Value Decomposition on raw matrix data).`
*   **Interview Edge (Advanced Attributions)**: *"Prove that the first principal component is the eigenvector of the sample covariance matrix corresponding to its largest eigenvalue."*

---

### 📅 Day 076 | 15 Aug 2026, Sat
*   **Phase**: Phase 6: Unsupervised Learning
*   **Advanced Topic**: **PCA — Feature Extraction & Reconstruction**
*   **Deep-Dive Subtopics**: Dimensionality reduction, projection onto principal components, reconstruction of original features, calculating reconstruction MSE.
*   **Hands-on Deliverable**: `Day 076 - PCA Projection.ipynb (Compressing high-dimensional datasets and evaluating reconstruction loss boundaries).`
*   **Interview Edge (Advanced Attributions)**: *"How do you mathematically determine the optimal number of principal components to retain using the Kaiser rule or elbow scree plots?"*

---

### 📅 Day 077 | 16 Aug 2026, Sun
*   **Phase**: Phase 6: Unsupervised Learning
*   **Advanced Topic**: **t-SNE for High-Dimensional Visualization**
*   **Deep-Dive Subtopics**: Manifold learning, probability mapping, Kullback-Leibler (KL) divergence objective function, tuning perplexity, overcrowding problem.
*   **Hands-on Deliverable**: `Day 077 - tSNE Visualization.ipynb (Fitting t-SNE on MNIST digits, demonstrating the impact of perplexity changes).`
*   **Interview Edge (Advanced Attributions)**: *"Define the probability distributions of pairwise similarities in t-SNE. Why does t-SNE use a Student-t distribution in the low-dimensional space?"*

---

### 📅 Day 078 | 17 Aug 2026, Mon
*   **Phase**: Phase 6: Unsupervised Learning
*   **Advanced Topic**: **UMAP — Faster Non-linear Visualization**
*   **Deep-Dive Subtopics**: Fuzzy simplicial sets, Riemannian geometry concepts, preserving global vs. local structures, UMAP hyperparameters (n_neighbors, min_dist).
*   **Hands-on Deliverable**: `Day 078 - UMAP Visualization.ipynb (Comparing UMAP vs. t-SNE visualization speed and global clustering separations).`
*   **Interview Edge (Advanced Attributions)**: *"How does UMAP mathematically preserve global data topology better than t-SNE?"*

---

### 📅 Day 079 | 18 Aug 2026, Tue
*   **Phase**: Phase 6: Unsupervised Learning
*   **Advanced Topic**: **Anomaly Detection — Isolation Forest & LOF**
*   **Deep-Dive Subtopics**: Unsupervised anomaly detection: Isolation Forest path length calculations vs. Local Outlier Factor (LOF) relative local density scores.
*   **Hands-on Deliverable**: `Day 079 - Anomaly Detection.ipynb (Benchmarking unsupervised anomaly models on synthetically contaminated datasets).`
*   **Interview Edge (Advanced Attributions)**: *"Derive the anomaly score equation in Isolation Forest and explain what the normalized path length $s(x, n)$ indicates."*

---

### 📅 Day 080 | 19 Aug 2026, Wed
*   **Phase**: Phase 6: Unsupervised Learning
*   **Advanced Topic**: **Gaussian Mixture Models (GMM)**
*   **Deep-Dive Subtopics**: Expectation-Maximization (EM) algorithm, soft assignment probability, covariance matrices (spherical, diagonal, tied, full), AIC/BIC metrics.
*   **Hands-on Deliverable**: `Day 080 - GMM Clustering.ipynb (Fitting GMM to overlapping data, using AIC/BIC to select optimal component count).`
*   **Interview Edge (Advanced Attributions)**: *"Derive the E-step and M-step equations for a Gaussian Mixture Model trained on multi-dimensional continuous data."*

---

### 📅 Day 081 | 20 Aug 2026, Thu
*   **Phase**: Phase 6: Unsupervised Learning
*   **Advanced Topic**: **Customer Segmentation Project**
*   **Deep-Dive Subtopics**: Ingesting tabular transactions data, RFM analysis, log transforms, PCA dimensionality reduction, KMeans profiling, cluster dashboards.
*   **Hands-on Deliverable**: `Day 081 - Customer Segmentation.ipynb (Customer profiling pipeline generating target group profiles).`
*   **Interview Edge (Advanced Attributions)**: *"How do you mathematically interpret cluster centroid values when clustering has been performed on PCA-reduced representations?"*

---

### 📅 Day 082 | 21 Aug 2026, Fri
*   **Phase**: Phase 6: Unsupervised Learning
*   **Advanced Topic**: **Dimensionality Reduction Project**
*   **Deep-Dive Subtopics**: Evaluating downstream classification performance on raw, PCA-reduced, t-SNE, and UMAP-transformed datasets, analyzing processing speed.
*   **Hands-on Deliverable**: `Day 082 - Dim Reduction Project.ipynb (Script evaluating model accuracy vs. input feature dimensionality).`
*   **Interview Edge (Advanced Attributions)**: *"Does projecting data using PCA always improve classification accuracy? Discuss the impact of linear assumptions."*

---

### 📅 Day 083 | 22 Aug 2026, Sat
*   **Phase**: Phase 6: Unsupervised Learning
*   **Advanced Topic**: **Clustering Comparison & Metrics**
*   **Deep-Dive Subtopics**: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Homogeneity, Completeness, V-Measure mathematical formulas.
*   **Hands-on Deliverable**: `Day 083 - Clustering Metrics.ipynb (Computing clustering indices from ground-truth validations).`
*   **Interview Edge (Advanced Attributions)**: *"Define the Adjusted Rand Index. Why is the adjustment for chance critical when evaluating clustering quality?"*

---

### 📅 Day 084 | 23 Aug 2026, Sun
*   **Phase**: Phase 6: Unsupervised Learning
*   **Advanced Topic**: **Unsupervised Learning Capstone Setup**
*   **Deep-Dive Subtopics**: Selecting dataset, defining clustering objective, setting up unsupervised pipelines with scaling, PCA, and KMeans.
*   **Hands-on Deliverable**: `Day 084 - Unsupervised Capstone.ipynb (Baseline pipeline architecture template for the capstone clustering task).`
*   **Interview Edge (Advanced Attributions)**: *"How do you construct a pipeline that automates both PCA component selection and KMeans clustering with optimal silhouette scores?"*

---

### 📅 Day 085 | 24 Aug 2026, Mon
*   **Phase**: Phase 6: Unsupervised Learning
*   **Advanced Topic**: **Phase Review & Unsupervised Assessment**
*   **Deep-Dive Subtopics**: Review of clustering math, principal components math, manifold learning, and unsupervised anomaly detection.
*   **Hands-on Deliverable**: `Day 085 - Unsupervised Assessment.ipynb (Answers to 15 advanced unsupervised learning system design questions).`
*   **Interview Edge (Advanced Attributions)**: *"How would you design a real-time unsupervised anomaly detection system for credit card transactions with sub-10ms latency requirements?"*

---

### 📅 Day 086 | 25 Aug 2026, Tue
*   **Phase**: Phase 7: Time Series Analysis
*   **Advanced Topic**: **Time Series Components & Decomposition**
*   **Deep-Dive Subtopics**: Additive vs. multiplicative models, moving averages, trend extraction, seasonality extraction, statsmodels seasonal_decompose.
*   **Hands-on Deliverable**: `Day 086 - Time Series Intro.ipynb (Implementing seasonal decomposition from scratch using moving averages).`
*   **Interview Edge (Advanced Attributions)**: *"Under what conditions would you choose a multiplicative decomposition model over an additive decomposition model?"*

---

### 📅 Day 087 | 26 Aug 2026, Wed
*   **Phase**: Phase 7: Time Series Analysis
*   **Advanced Topic**: **Stationarity & Dickey-Fuller Test**
*   **Deep-Dive Subtopics**: Weak stationarity (constant mean, variance, autocovariance), differencing, Augmented Dickey-Fuller (ADF) test equations, Phillips-Perron test.
*   **Hands-on Deliverable**: `Day 087 - Stationarity.ipynb (Calculating rolling statistics and evaluating stationarity using statsmodels ADF test).`
*   **Interview Edge (Advanced Attributions)**: *"Prove mathematically why a random walk process is non-stationary, and show how first-order differencing resolves it."*

---

### 📅 Day 088 | 27 Aug 2026, Thu
*   **Phase**: Phase 7: Time Series Analysis
*   **Advanced Topic**: **ACF & PACF Plots**
*   **Deep-Dive Subtopics**: Autocorrelation Function (ACF) math, Partial Autocorrelation Function (PACF) math, Yule-Walker equations, identifying AR(p) and MA(q) lags.
*   **Hands-on Deliverable**: `Day 088 - ACF PACF.ipynb (Plotting autocorrelation and partial autocorrelation and identifying lag limits).`
*   **Interview Edge (Advanced Attributions)**: *"Explain how the PACF plot isolating lag correlations differs mathematically from the standard ACF plot."*

---

### 📅 Day 089 | 28 Aug 2026, Fri
*   **Phase**: Phase 7: Time Series Analysis
*   **Advanced Topic**: **ARIMA Models**
*   **Deep-Dive Subtopics**: Autoregressive Integrated Moving Average, model parameters (p, d, q), backtesting predictions, residual diagnostics (Ljung-Box test).
*   **Hands-on Deliverable**: `Day 089 - ARIMA.ipynb (Fitting ARIMA models on stock indexes and checking residuals for white noise behavior).`
*   **Interview Edge (Advanced Attributions)**: *"State the mathematical equation of an ARIMA(1, 1, 1) process. What are the stationarity and invertibility conditions for it?"*

---

### 📅 Day 090 | 29 Aug 2026, Sat
*   **Phase**: Phase 7: Time Series Analysis
*   **Advanced Topic**: **SARIMA Models**
*   **Deep-Dive Subtopics**: Seasonal ARIMA models, identifying seasonal orders (P, D, Q, s), managing multi-seasonal parameters, residual diagnostics.
*   **Hands-on Deliverable**: `Day 090 - SARIMA.ipynb (Fitting seasonal SARIMA models to forecast monthly electricity/airline demand).`
*   **Interview Edge (Advanced Attributions)**: *"Explain how seasonal differencing differs from ordinary differencing, and write the SARIMA(1,0,0)x(0,1,0)[12] model equation."*

---

### 📅 Day 091 | 30 Aug 2026, Sun
*   **Phase**: Phase 7: Time Series Analysis
*   **Advanced Topic**: **Facebook Prophet Basics**
*   **Deep-Dive Subtopics**: Prophet additive model formulation (trend piecewise linear/logistic, Fourier seasonality, holidays), formatting datasets with ds/y.
*   **Hands-on Deliverable**: `Day 091 - Prophet Intro.ipynb (Fitting Facebook Prophet on simple daily website traffic logs).`
*   **Interview Edge (Advanced Attributions)**: *"How does Facebook Prophet mathematically formulate seasonality using Fourier series? Write the equation."*

---

### 📅 Day 092 | 31 Aug 2026, Mon
*   **Phase**: Phase 7: Time Series Analysis
*   **Advanced Topic**: **Prophet Advanced Features**
*   **Deep-Dive Subtopics**: Adding user holidays, custom regressors, detecting trend changepoints, plotting uncertainty intervals.
*   **Hands-on Deliverable**: `Day 092 - Prophet Advanced.ipynb (Configuring multi-variable Prophet forecast with custom calendar inputs).`
*   **Interview Edge (Advanced Attributions)**: *"How does Prophet model changepoint detection mathematically, and how does the prior parameter control trend elasticity?"*

---

### 📅 Day 093 | 01 Sep 2026, Tue
*   **Phase**: Phase 7: Time Series Analysis
*   **Advanced Topic**: **Time Series Forecasting with ML**
*   **Deep-Dive Subtopics**: Converting series to supervised datasets, lag features creation, rolling aggregate windows, date-time features, XGBoost for forecasting.
*   **Hands-on Deliverable**: `Day 093 - ML Time Series.ipynb (Feature engineering pipeline converting continuous time-series into tabular datasets for XGBoost).`
*   **Interview Edge (Advanced Attributions)**: *"Explain why using future date-time features (like year, day of week) inside boosting models can lead to catastrophic trend extrapolation failures."*

---

### 📅 Day 094 | 02 Sep 2026, Wed
*   **Phase**: Phase 7: Time Series Analysis
*   **Advanced Topic**: **Demand Forecasting Project**
*   **Deep-Dive Subtopics**: End-to-End Project: ingesting retail store sales, differencing, comparing ARIMA, Prophet, and XGBoost forecasts using MAPE and RMSE.
*   **Hands-on Deliverable**: `Day 094 - Demand Forecasting Project.ipynb (A master time series evaluation comparison notebook on real retail data).`
*   **Interview Edge (Advanced Attributions)**: *"How do you handle zero-demand days (zero sales) mathematically when evaluating forecast accuracy using MAPE?"*

---

### 📅 Day 095 | 03 Sep 2026, Thu
*   **Phase**: Phase 7: Time Series Analysis
*   **Advanced Topic**: **Advanced Time Series**
*   **Deep-Dive Subtopics**: Vector Autoregressive (VAR) models for multivariate series, GARCH volatility modeling intuition, Granger Causality tests.
*   **Hands-on Deliverable**: `Day 095 - Advanced Time Series.ipynb (Running Granger causality and fitting VAR models to macroeconomic datasets).`
*   **Interview Edge (Advanced Attributions)**: *"Explain the Granger Causality test mathematically and write the mathematical representation of a Vector Autoregressive model of order 1."*

---

### 📅 Day 096 | 04 Sep 2026, Fri
*   **Phase**: Phase 7: Time Series Analysis
*   **Advanced Topic**: **Phase Review & TS Assessment**
*   **Deep-Dive Subtopics**: Analyzing time series validation strategies (TimeSeriesSplit, Walk-Forward Validation), assessing time series theory.
*   **Hands-on Deliverable**: `Day 096 - TS Assessment.ipynb (Written solutions to 12 core time series forecasting interview questions).`
*   **Interview Edge (Advanced Attributions)**: *"Why is k-fold cross-validation mathematically invalid for time-series forecasting datasets, and how does TimeSeriesSplit solve it?"*

---

### 📅 Day 097 | 05 Sep 2026, Sat
*   **Phase**: Phase 8: Deep Learning — ANN & Optimizers
*   **Advanced Topic**: **Deep Learning Intro & Perceptron**
*   **Deep-Dive Subtopics**: Biological vs. Artificial neural networks, mathematical McCulloch-Pitts neuron, Perceptron learning rule, proving the XOR limitation.
*   **Hands-on Deliverable**: `Day 097 - Perceptron.ipynb (Implementing a single Perceptron with step activation function from scratch).`
*   **Interview Edge (Advanced Attributions)**: *"Prove mathematically why a single-layer Perceptron cannot solve the non-linear XOR logic gate problem."*

---

### 📅 Day 098 | 06 Sep 2026, Sun
*   **Phase**: Phase 8: Deep Learning — ANN & Optimizers
*   **Advanced Topic**: **Multi-layer Perceptrons & Forward Propagation**
*   **Deep-Dive Subtopics**: Multi-Layer Perceptron (MLP) architectures, input, hidden, and output representation layers, vectorized forward propagation equations.
*   **Hands-on Deliverable**: `Day 098 - MLP Forward.ipynb (Implementing feedforward network forward propagation loop in pure vectorized NumPy).`
*   **Interview Edge (Advanced Attributions)**: *"Explain how hidden layer representations in neural networks mathematically construct higher-order non-linear feature maps."*

---

### 📅 Day 099 | 07 Sep 2026, Mon
*   **Phase**: Phase 8: Deep Learning — ANN & Optimizers
*   **Advanced Topic**: **Activation Functions**
*   **Deep-Dive Subtopics**: Sigmoid, Tanh, ReLU, Leaky ReLU, GELU (Gaussian Error Linear Unit), Swish, and Softmax formulas, derivative calculations, vanishing gradients.
*   **Hands-on Deliverable**: `Day 099 - Activations.ipynb (Plotting activations and their derivatives, checking gradient profiles).`
*   **Interview Edge (Advanced Attributions)**: *"What is the vanishing gradient problem, and how do modern activation functions like ReLU and GELU mathematically mitigate it?"*

---

### 📅 Day 100 | 08 Sep 2026, Tue
*   **Phase**: Phase 8: Deep Learning — ANN & Optimizers
*   **Advanced Topic**: **Loss Functions**
*   **Deep-Dive Subtopics**: Mean Squared Error (MSE), L1 loss, Huber Loss, Binary Cross-Entropy (BCE), Categorical Cross-Entropy, Sparse Categorical Cross-Entropy.
*   **Hands-on Deliverable**: `Day 100 - Loss Functions.ipynb (Writing standard loss functions and their derivatives in Python).`
*   **Interview Edge (Advanced Attributions)**: *"Derive the mathematical formulation of Categorical Cross-Entropy loss from the Kullback-Leibler divergence between target and prediction."*

---

### 📅 Day 101 | 09 Sep 2026, Wed
*   **Phase**: Phase 8: Deep Learning — ANN & Optimizers
*   **Advanced Topic**: **Backpropagation — Chain Rule & Math**
*   **Deep-Dive Subtopics**: Computational graphs, local gradients, multi-variable chain rule derivation, computing derivatives of loss w.r.t. weights and biases.
*   **Hands-on Deliverable**: `Day 101 - Backpropagation Math.ipynb (Mathematical worksheets step-by-step backpropagating loss through a 3-layer neural network).`
*   **Interview Edge (Advanced Attributions)**: *"State the general backpropagation equations for layer $l$ using delta notations. Derive the error delta relation between layer $l$ and $l+1$."*

---

### 📅 Day 102 | 10 Sep 2026, Thu
*   **Phase**: Phase 8: Deep Learning — ANN & Optimizers
*   **Advanced Topic**: **Manual Backpropagation Implementation**
*   **Deep-Dive Subtopics**: Assembling forward pass, backward pass, and parameter updates inside raw NumPy vectors, training a 2-layer MLP on non-linear data.
*   **Hands-on Deliverable**: `Day 102 - Numpy Neural Network.ipynb (Numpy-only multi-layer classification neural network trained from scratch).`
*   **Interview Edge (Advanced Attributions)**: *"Show how to write a vectorized weight update equation for the output layer weights using outer products of activations and gradients."*

---

### 📅 Day 103 | 11 Sep 2026, Fri
*   **Phase**: Phase 8: Deep Learning — ANN & Optimizers
*   **Advanced Topic**: **Optimizers**
*   **Deep-Dive Subtopics**: Stochastic Gradient Descent (SGD), Momentum, Adagrad, RMSprop, Adam (Adaptive Moment Estimation), AdamW weight decay correction.
*   **Hands-on Deliverable**: `Day 103 - Optimizers.ipynb (Custom optimizer implementations comparison checking training path behaviors).`
*   **Interview Edge (Advanced Attributions)**: *"Why does AdamW split the L2 weight decay penalty from the gradient calculation? Contrast this with standard Adam weight decay."*

---

### 📅 Day 104 | 12 Sep 2026, Sat
*   **Phase**: Phase 8: Deep Learning — ANN & Optimizers
*   **Advanced Topic**: **Learning Rate Scheduling**
*   **Deep-Dive Subtopics**: Step Decay, Exponential Decay, Cosine Annealing scheduler equations, Learning Rate finders, cyclic learning rates.
*   **Hands-on Deliverable**: `Day 104 - LR Scheduling.ipynb (Writing and plotting dynamic learning rate schedules over epochs).`
*   **Interview Edge (Advanced Attributions)**: *"What is the mathematical advantage of Cosine Annealing with Warm Restarts over standard step-decay schedulers?"*

---

### 📅 Day 105 | 13 Sep 2026, Sun
*   **Phase**: Phase 8: Deep Learning — ANN & Optimizers
*   **Advanced Topic**: **TensorFlow & Keras Basics**
*   **Deep-Dive Subtopics**: Dynamic vs. Static computation graphs, tf.Tensor, tf.Variable, tf.GradientTape, Sequential API, constructing baseline Dense networks.
*   **Hands-on Deliverable**: `Day 105 - Keras Basics.ipynb (TensorFlow tensor manipulations and tf.GradientTape automatic differentiation validations).`
*   **Interview Edge (Advanced Attributions)**: *"How does tf.GradientTape mathematically construct a tape for reverse-mode automatic differentiation in TensorFlow?"*

---

### 📅 Day 106 | 14 Sep 2026, Mon
*   **Phase**: Phase 8: Deep Learning — ANN & Optimizers
*   **Advanced Topic**: **ANN Implementation in Keras**
*   **Deep-Dive Subtopics**: Model compilation parameters, training configuration, early stopping callbacks, TensorBoard tracking integration.
*   **Hands-on Deliverable**: `Day 106 - Keras ANN.ipynb (Keras classification neural network trained on synthetic non-linear classifications).`
*   **Interview Edge (Advanced Attributions)**: *"What is the difference between sequential models, functional APIs, and subclassed models in Keras?"*

---

### 📅 Day 107 | 15 Sep 2026, Tue
*   **Phase**: Phase 8: Deep Learning — ANN & Optimizers
*   **Advanced Topic**: **PyTorch Basics — Tensors & Autograd**
*   **Deep-Dive Subtopics**: PyTorch tensors, device properties (CPU/CUDA/MPS), computational graphs generation, dynamic graph execution, `backward()` gradients.
*   **Hands-on Deliverable**: `Day 107 - PyTorch Basics.ipynb (PyTorch tensor math, verifying autograd gradients against analytical derivatives).`
*   **Interview Edge (Advanced Attributions)**: *"How does PyTorch's dynamic computation graph (autograd) construct and destroy the backward execution tree during a forward-backward pass?"*

---

### 📅 Day 108 | 16 Sep 2026, Wed
*   **Phase**: Phase 8: Deep Learning — ANN & Optimizers
*   **Advanced Topic**: **PyTorch Custom Models & Datasets**
*   **Deep-Dive Subtopics**: Subclassing `nn.Module` and implementing `forward`, subclassing `Dataset` and overriding `__len__` and `__getitem__`, `DataLoader` loops.
*   **Hands-on Deliverable**: `Day 108 - PyTorch Datasets.ipynb (Writing fully custom PyTorch multi-class classifiers with customized datasets).`
*   **Interview Edge (Advanced Attributions)**: *"Why is overriding `__getitem__` inside PyTorch Datasets optimal for streaming massive, multi-gigabyte image folders without crashing RAM?"*

---

### 📅 Day 109 | 17 Sep 2026, Thu
*   **Phase**: Phase 8: Deep Learning — ANN & Optimizers
*   **Advanced Topic**: **PyTorch Training Loop**
*   **Deep-Dive Subtopics**: Writing standard PyTorch training loop routines: epoch iterations, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`, evaluation models.
*   **Hands-on Deliverable**: `Day 109 - PyTorch Training Loop.ipynb (Complete training loop with batch tracking and validation scoring).`
*   **Interview Edge (Advanced Attributions)**: *"Why must we call `model.train()` during training and `model.eval()` and `with torch.no_grad()` during model evaluation loops?"*

---

### 📅 Day 110 | 18 Sep 2026, Fri
*   **Phase**: Phase 8: Deep Learning — ANN & Optimizers
*   **Advanced Topic**: **Dropout & Batch Normalization**
*   **Deep-Dive Subtopics**: Dropout mathematical scaling, Batch Normalization scaling equations, moving means and variances tracking, layer positioning (before/after activation).
*   **Hands-on Deliverable**: `Day 110 - Dropout BatchNorm.ipynb (Adding Dropout and BatchNorm to PyTorch models and checking validation stability).`
*   **Interview Edge (Advanced Attributions)**: *"Provide the exact mathematical equations for Batch Normalization during training and explain how it differs during test inference."*

---

### 📅 Day 111 | 19 Sep 2026, Sat
*   **Phase**: Phase 8: Deep Learning — ANN & Optimizers
*   **Advanced Topic**: **ANN Hyperparameter Tuning & Optuna**
*   **Deep-Dive Subtopics**: Optimizing neural layers count, neuron expansions, optimizers, learning rate combinations using parallelized Optuna searches in PyTorch.
*   **Hands-on Deliverable**: `Day 111 - ANN Tuning Project.ipynb (Robust script utilizing Optuna to find top performing PyTorch configuration).`
*   **Interview Edge (Advanced Attributions)**: *"How do you configure Optuna's MedianPruner callback to aggressively prune poorly performing neural network training trials early?"*

---

### 📅 Day 112 | 20 Sep 2026, Sun
*   **Phase**: Phase 8: Deep Learning — ANN & Optimizers
*   **Advanced Topic**: **Phase Review & DL Foundations Review**
*   **Deep-Dive Subtopics**: Review of forward-backward computation graphs, automatic differentiation, optimizer math, and regularizations.
*   **Hands-on Deliverable**: `Day 112 - DL Foundations Review.ipynb (Written answers to 15 advanced deep learning foundational questions).`
*   **Interview Edge (Advanced Attributions)**: *"Compare reverse-mode automatic differentiation with forward-mode automatic differentiation. Under what conditions is reverse-mode preferred?"*

---

### 📅 Day 113 | 21 Sep 2026, Mon
*   **Phase**: Phase 9: Deep Learning — CNN
*   **Advanced Topic**: **CNN Intuition**
*   **Deep-Dive Subtopics**: Spatial dependencies, convolutions, filter kernels (Sobel, Gaussian), channels, stride, padding (valid, same), output dimension math.
*   **Hands-on Deliverable**: `Day 113 - CNN Intuition.ipynb (Custom function executing 2D matrix convolution using custom kernels in NumPy).`
*   **Interview Edge (Advanced Attributions)**: *"State the formula to compute the output spatial dimensions of a convolutional layer given input size $W$, kernel size $K$, padding $P$, and stride $S$."*

---

### 📅 Day 114 | 22 Sep 2026, Tue
*   **Phase**: Phase 9: Deep Learning — CNN
*   **Advanced Topic**: **Pooling Layers & CNN Architecture Design**
*   **Deep-Dive Subtopics**: MaxPooling spatial invariance, AveragePooling, flattening layer transitions, fully-connected output mappings, pooling parameter reductions.
*   **Hands-on Deliverable**: `Day 114 - Pooling Layers.ipynb (Analyzing activation shape size transformations through sequential convolutional and pooling layers).`
*   **Interview Edge (Advanced Attributions)**: *"Why does MaxPooling introduce translation invariance? Explain how pooling helps control parameter counts and overfitting."*

---

### 📅 Day 115 | 23 Sep 2026, Wed
*   **Phase**: Phase 9: Deep Learning — CNN
*   **Advanced Topic**: **CNN Implementation in Keras & PyTorch**
*   **Deep-Dive Subtopics**: Constructing 2D CNN layers: `Conv2D` vs. `nn.Conv2d`, building classification CNNs, tracing tensor dimensions throughout the network.
*   **Hands-on Deliverable**: `Day 115 - CNN Code.ipynb (CNN training on the MNIST/CIFAR-10 datasets using both PyTorch and Keras layers).`
*   **Interview Edge (Advanced Attributions)**: *"How does PyTorch's channel-first input notation `(Batch, Channel, Height, Width)` differ from TensorFlow's channel-last notation?"*

---

### 📅 Day 116 | 24 Sep 2026, Thu
*   **Phase**: Phase 9: Deep Learning — CNN
*   **Advanced Topic**: **Image Data Augmentation**
*   **Deep-Dive Subtopics**: Flipping, cropping, affine transforms, random brightness, contrast changes, writing automated data augmentation pipelines.
*   **Hands-on Deliverable**: `Day 116 - Data Augmentation.ipynb (Setting up PyTorch torchvision transforms and visual data augmentations).`
*   **Interview Edge (Advanced Attributions)**: *"How does image augmentation mathematically act as a regularizer, and how does it affect target generalization variance?"*

---

### 📅 Day 117 | 25 Sep 2026, Fri
*   **Phase**: Phase 9: Deep Learning — CNN
*   **Advanced Topic**: **CNN Architectures — LeNet, AlexNet, VGG**
*   **Deep-Dive Subtopics**: Structural parameters and evolutions: LeNet-5 spatial resolutions, AlexNet kernel setups, VGG-16 deep stacking using small 3x3 kernels.
*   **Hands-on Deliverable**: `Day 117 - Classic CNNs.ipynb (Rebuilding the classic VGG-16 model architectural blocks in PyTorch).`
*   **Interview Edge (Advanced Attributions)**: *"Why did VGG-16 transition to stacking multiple 3x3 convolutions instead of using larger 5x5 or 7x7 filters? Prove the parameter savings."*

---

### 📅 Day 118 | 26 Sep 2026, Sat
*   **Phase**: Phase 9: Deep Learning — CNN
*   **Advanced Topic**: **ResNet & Skip Connections**
*   **Deep-Dive Subtopics**: Residual blocks, identity mappings, solving the vanishing gradient problem in very deep networks, ResNet structural scaling.
*   **Hands-on Deliverable**: `Day 118 - ResNet.ipynb (Implementing ResNet residual blocks and building a custom ResNet-18 model).`
*   **Interview Edge (Advanced Attributions)**: *"Show mathematically how ResNet skip connections prevent gradients from vanishing during backpropagation down hundreds of layers."*

---

### 📅 Day 119 | 27 Sep 2026, Sun
*   **Phase**: Phase 9: Deep Learning — CNN
*   **Advanced Topic**: **Inception & MobileNet**
*   **Deep-Dive Subtopics**: Inception multi-scale convolutions (1x1, 3x3, 5x5 blocks), depthwise separable convolutions, pointwise convolutions, edge devices design.
*   **Hands-on Deliverable**: `Day 119 - Inception MobileNet.ipynb (Implementing depthwise separable convolutions in PyTorch, parameter tracking).`
*   **Interview Edge (Advanced Attributions)**: *"Prove the mathematical parameter reduction of a 3x3 Depthwise Separable convolution compared to a standard 3x3 convolution."*

---

### 📅 Day 120 | 28 Sep 2026, Mon
*   **Phase**: Phase 9: Deep Learning — CNN
*   **Advanced Topic**: **Transfer Learning — Feature Extraction**
*   **Deep-Dive Subtopics**: Loading pre-trained ImageNet classifiers, freezing backbone features layers, replacing classifier classification head, training baseline transfer layers.
*   **Hands-on Deliverable**: `Day 120 - Transfer Learning Features.ipynb (Using pre-trained ResNet-50 for feature extraction on custom image dataset).`
*   **Interview Edge (Advanced Attributions)**: *"Why do convolutional filters trained on ImageNet generalize remarkably well as feature extractors for highly diverse visual datasets?"*

---

### 📅 Day 121 | 29 Sep 2026, Tue
*   **Phase**: Phase 9: Deep Learning — CNN
*   **Advanced Topic**: **Transfer Learning — Fine-Tuning**
*   **Deep-Dive Subtopics**: Unfreezing upper convolutional layers, training with small learning rates, applying differential optimizer learning rates per network block.
*   **Hands-on Deliverable**: `Day 121 - Transfer Learning Finetuning.ipynb (Unfreezing ResNet top blocks, optimizing classification with small learning rates).`
*   **Interview Edge (Advanced Attributions)**: *"Why must we freeze base layers initially and train the classification head before attempting full network fine-tuning?"*

---

### 📅 Day 122 | 30 Sep 2026, Wed
*   **Phase**: Phase 9: Deep Learning — CNN
*   **Advanced Topic**: **Image Classification Project**
*   **Deep-Dive Subtopics**: End-to-End Project: data cleaning, custom augmentations, training fine-tuned pre-trained models, plotting Grad-CAM activation outputs.
*   **Hands-on Deliverable**: `Day 122 - Image Classifier Project.ipynb (Complete image classifier with full evaluation metrics and visual activation analyses).`
*   **Interview Edge (Advanced Attributions)**: *"Explain how Grad-CAM mathematically calculates activation map weights using the gradients of score target values."*

---

### 📅 Day 123 | 01 Oct 2026, Thu
*   **Phase**: Phase 9: Deep Learning — CNN
*   **Advanced Topic**: **Phase Review & CNN Assessment**
*   **Deep-Dive Subtopics**: Review of CNN spatial mathematics, classic and modern CNN structures, and parameter calculations.
*   **Hands-on Deliverable**: `Day 123 - CNN Assessment.ipynb (Written solutions to 15 complex convolutional network interview questions).`
*   **Interview Edge (Advanced Attributions)**: *"Calculate the total number of trainable parameters in a convolutional layer with 64 filters of size 3x3, stride 1, padding same, on RGB input images."*

---

### 📅 Day 124 | 02 Oct 2026, Fri
*   **Phase**: Phase 10: Deep Learning — RNN & Sequence Models
*   **Advanced Topic**: **Sequential Data & RNN Intuition**
*   **Deep-Dive Subtopics**: Sequence inputs (text, audio, signals), hidden state recurrence, standard Recurrent Neural Network equations, Backpropagation Through Time (BPTT).
*   **Hands-on Deliverable**: `Day 124 - RNN Intuition.ipynb (Implementing custom simple RNN cell update loops from scratch in raw NumPy).`
*   **Interview Edge (Advanced Attributions)**: *"Explain mathematically why standard RNNs suffer from vanishing and exploding gradient problems during long-sequence training backprop."*

---

### 📅 Day 125 | 03 Oct 2026, Sat
*   **Phase**: Phase 10: Deep Learning — RNN & Sequence Models
*   **Advanced Topic**: **LSTM — Long Short-Term Memory**
*   **Deep-Dive Subtopics**: Forget, Input, and Output gate mathematical activation equations, cell state vector, tracking long-term and short-term dependencies.
*   **Hands-on Deliverable**: `Day 125 - LSTM.ipynb (Custom manual construction of LSTM forward update equations in PyTorch).`
*   **Interview Edge (Advanced Attributions)**: *"Explain how the Cell State linear pathway inside an LSTM prevents gradient vanishing over hundreds of sequence tokens."*

---

### 📅 Day 126 | 04 Oct 2026, Sun
*   **Phase**: Phase 10: Deep Learning — RNN & Sequence Models
*   **Advanced Topic**: **GRU — Gated Recurrent Unit**
*   **Deep-Dive Subtopics**: Reset and Update gate mathematical configurations, unified hidden state representation, parameter efficiency vs. LSTM.
*   **Hands-on Deliverable**: `Day 126 - GRU.ipynb (Building custom GRU cell networks and evaluating CIFAR sequential classifications).`
*   **Interview Edge (Advanced Attributions)**: *"State the gate update equations for a GRU. What are the key mathematical differences between LSTM and GRU?"*

---

### 📅 Day 127 | 05 Oct 2026, Mon
*   **Phase**: Phase 10: Deep Learning — RNN & Sequence Models
*   **Advanced Topic**: **Bidirectional RNNs & Stacked LSTMs**
*   **Deep-Dive Subtopics**: Forward and backward sequence representations, Bidirectional hidden state concatenation, stacking recurrent layers.
*   **Hands-on Deliverable**: `Day 127 - Bidirectional RNN.ipynb (Constructing bidirectional stacked LSTM networks on sentiment classification classifications).`
*   **Interview Edge (Advanced Attributions)**: *"Under what NLP sequence task conditions is a Bidirectional RNN preferred over a standard unidirectional causal RNN?"*

---

### 📅 Day 128 | 06 Oct 2026, Tue
*   **Phase**: Phase 10: Deep Learning — RNN & Sequence Models
*   **Advanced Topic**: **RNN for Time Series Forecasting**
*   **Deep-Dive Subtopics**: Windowing continuous 1D/Multi-dimensional sequence tensors, training PyTorch LSTM sequence predictors, recursive forecasting validation.
*   **Hands-on Deliverable**: `Day 128 - LSTM Time Series.ipynb (Sequence window preprocessing pipeline and LSTM continuous multivariate forecasting).`
*   **Interview Edge (Advanced Attributions)**: *"How do you map temporal series data shapes into `(Batch, Sequence, Features)` format required for PyTorch recurrent cells?"*

---

### 📅 Day 129 | 07 Oct 2026, Wed
*   **Phase**: Phase 10: Deep Learning — RNN & Sequence Models
*   **Advanced Topic**: **Autoencoders — Architecture**
*   **Deep-Dive Subtopics**: Encoder representations, bottleneck dimensional constraints, Decoder networks, reconstruction loss mathematical optimization.
*   **Hands-on Deliverable**: `Day 129 - Autoencoder Basics.ipynb (Custom PyTorch undercomplete Autoencoder trained to compress and reconstruct complex signal vectors).`
*   **Interview Edge (Advanced Attributions)**: *"Prove mathematically why a single-layer linear autoencoder with MSE reconstruction loss is equivalent to Principal Component Analysis."*

---

### 📅 Day 130 | 08 Oct 2026, Thu
*   **Phase**: Phase 10: Deep Learning — RNN & Sequence Models
*   **Advanced Topic**: **Denoising & Variational Autoencoders**
*   **Deep-Dive Subtopics**: Reparameterization trick math, latent space standard normal mapping ($z \sim N(\mu, \sigma^2)$), Kullback-Leibler (KL) Divergence loss math.
*   **Hands-on Deliverable**: `Day 130 - VAE Basics.ipynb (Implementing a PyTorch Variational Autoencoder, plotting reconstructed latent spatial embeddings).`
*   **Interview Edge (Advanced Attributions)**: *"Explain the Reparameterization Trick in VAEs and write down the complete ELBO (Evidence Lower Bound) optimization loss function."*

---

### 📅 Day 131 | 09 Oct 2026, Fri
*   **Phase**: Phase 10: Deep Learning — RNN & Sequence Models
*   **Advanced Topic**: **GANs — Generative Adversarial Networks Basics**
*   **Deep-Dive Subtopics**: Adversarial game theory minimax objectives, Generator structural configurations, Discriminator classifiers, custom adversarial training loops.
*   **Hands-on Deliverable**: `Day 131 - GAN Intro.ipynb (Building baseline fully-connected GAN generating synthetic data scatter distributions from scratch).`
*   **Interview Edge (Advanced Attributions)**: *"State the minimax objective function of a Generative Adversarial Network. Explain the equilibrium point mathematically."*

---

### 📅 Day 132 | 10 Oct 2026, Sat
*   **Phase**: Phase 10: Deep Learning — RNN & Sequence Models
*   **Advanced Topic**: **DCGAN & Training Challenges**
*   **Deep-Dive Subtopics**: Transpose convolutions (deconvolutions), Deep Convolutional GAN architectures, batch normalization layers, mode collapse debugging.
*   **Hands-on Deliverable**: `Day 132 - DCGAN.ipynb (Implementing a PyTorch DCGAN generating 28x28 grayscale handwritten digits).`
*   **Interview Edge (Advanced Attributions)**: *"What is Mode Collapse in GANs? Detail three distinct techniques used to stabilize training and prevent mode collapse."*

---

### 📅 Day 133 | 11 Oct 2026, Sun
*   **Phase**: Phase 10: Deep Learning — RNN & Sequence Models
*   **Advanced Topic**: **Sequence Project & DL Revision**
*   **Deep-Dive Subtopics**: Project: LSTM-based text generation; comprehensive revision of deep learning framework differences, model training steps.
*   **Hands-on Deliverable**: `Day 133 - LSTM Text Gen Project.ipynb (Text generation using character-level LSTMs, and comparative framework checklists).`
*   **Interview Edge (Advanced Attributions)**: *"Why does text generation with LSTM require soft probability distributions and temperature adjustments during sampling predictions?"*

---

### 📅 Day 134 | 12 Oct 2026, Mon
*   **Phase**: Phase 11: Natural Language Processing
*   **Advanced Topic**: **NLP Pipeline & Text Preprocessing**
*   **Deep-Dive Subtopics**: Sentence and Word tokenizations, regular expression cleans, sentence segmentation, building standard preprocessors pipelines.
*   **Hands-on Deliverable**: `Day 134 - NLP Pipeline.ipynb (Custom class applying tokenization, stripping HTML tags, removing special chars and casing variations).`
*   **Interview Edge (Advanced Attributions)**: *"Explain the byte-pair encoding (BPE) algorithm conceptually and why it is preferred over whitespace tokenization in modern LLMs."*

---

### 📅 Day 135 | 13 Oct 2026, Tue
*   **Phase**: Phase 11: Natural Language Processing
*   **Advanced Topic**: **Stemming, Lemmatization, Stop Words**
*   **Deep-Dive Subtopics**: Porter Stemming algorithm limitations, WordNet Lemmatization using Part-of-Speech tagging context, cleaning boilerplate stop words.
*   **Hands-on Deliverable**: `Day 135 - Text Normalization.ipynb (Comparing stemming vs. POS-aware lemmatization outputs on structured articles).`
*   **Interview Edge (Advanced Attributions)**: *"Contrast stemming and lemmatization. Why can stemming yield non-words, and how does lemmatization preserve semantics?"*

---

### 📅 Day 136 | 14 Oct 2026, Wed
*   **Phase**: Phase 11: Natural Language Processing
*   **Advanced Topic**: **Bag of Words & TF-IDF Vectorization**
*   **Deep-Dive Subtopics**: Term Frequency (TF) and Inverse Document Frequency (IDF) calculations, sparse matrices storage, n-gram bounds, min_df, max_df tuning.
*   **Hands-on Deliverable**: `Day 136 - TFIDF.ipynb (Writing a vectorized TF-IDF vectorizer matrix generator using only raw NumPy math).`
*   **Interview Edge (Advanced Attributions)**: *"State the mathematical formula of TF-IDF. Why is the logarithm applied to the inverse document frequency term?"*

---

### 📅 Day 137 | 15 Oct 2026, Thu
*   **Phase**: Phase 11: Natural Language Processing
*   **Advanced Topic**: **Word Embeddings — Word2Vec & GloVe**
*   **Deep-Dive Subtopics**: Continuous Bag of Words (CBOW) vs. Skip-Gram architectures, Negative Sampling optimization math, utilizing gensim embedding libraries.
*   **Hands-on Deliverable**: `Day 137 - Word2Vec.ipynb (Loading pre-trained Word2Vec embeddings and performing vector arithmetic like King - Man + Woman).`
*   **Interview Edge (Advanced Attributions)**: *"Explain Skip-Gram with Negative Sampling mathematically. Why is negative sampling computationally superior to softmax updates?"*

---

### 📅 Day 138 | 16 Oct 2026, Fri
*   **Phase**: Phase 11: Natural Language Processing
*   **Advanced Topic**: **Text Classification with ML**
*   **Deep-Dive Subtopics**: Vectorizing texts, training Logistic Regression, SVM, and Naive Bayes, evaluating metrics, building pipeline architectures.
*   **Hands-on Deliverable**: `Day 138 - ML Text Classification.ipynb (Pipeline classification on news/spam data using TF-IDF and MultinomialNB).`
*   **Interview Edge (Advanced Attributions)**: *"Why is Multinomial Naive Bayes exceptionally effective for high-dimensional text classification tasks when features represent TF-IDF scores?"*

---

### 📅 Day 139 | 17 Oct 2026, Sat
*   **Phase**: Phase 11: Natural Language Processing
*   **Advanced Topic**: **Text Classification with DL**
*   **Deep-Dive Subtopics**: Word indices, padding, nn.Embedding layers, spatial 1D Convolutions, sequence recurrent layers (LSTMs) for text classification.
*   **Hands-on Deliverable**: `Day 139 - DL Text Classification.ipynb (PyTorch model compiling embedding sequences and training classifier on sentiment data).`
*   **Interview Edge (Advanced Attributions)**: *"Explain how the dynamic embedding layer in PyTorch translates token indices into multi-dimensional dense vectors during training."*

---

### 📅 Day 140 | 18 Oct 2026, Sun
*   **Phase**: Phase 11: Natural Language Processing
*   **Advanced Topic**: **Attention Mechanism Intuition**
*   **Deep-Dive Subtopics**: Bottleneck of seq2seq models, Query-Key-Value similarity projections, attention weights computations, dot-product attention.
*   **Hands-on Deliverable**: `Day 140 - Attention Mechanism.ipynb (Implementing a standalone Scaled Dot-Product Attention module in PyTorch from scratch).`
*   **Interview Edge (Advanced Attributions)**: *"Write the mathematical equation of Scaled Dot-Product Attention. Why is the attention score divided by the square root of feature dimension?"*

---

### 📅 Day 141 | 19 Oct 2026, Mon
*   **Phase**: Phase 11: Natural Language Processing
*   **Advanced Topic**: **Transformer Architecture — Encoder & Decoder**
*   **Deep-Dive Subtopics**: Self-attention, Positional encodings (sinusoidal formulations), Multi-Head Attention blocks, Feed-Forward sublayers, Layer Normalizations.
*   **Hands-on Deliverable**: `Day 141 - Transformers.ipynb (Rebuilding a full single multi-head transformer encoder block from scratch in PyTorch).`
*   **Interview Edge (Advanced Attributions)**: *"Explain why Layer Normalization is applied inside transformer blocks instead of Batch Normalization. Detail pre-norm vs. post-norm."*

---

### 📅 Day 142 | 20 Oct 2026, Tue
*   **Phase**: Phase 11: Natural Language Processing
*   **Advanced Topic**: **BERT — Masked Language Modeling & NSP**
*   **Deep-Dive Subtopics**: Bidirectional Transformer encoder, pre-training goals: Masked LM (MLM) and Next Sentence Prediction (NSP), token embeddings.
*   **Hands-on Deliverable**: `Day 142 - BERT Intro.ipynb (Loading pretrained BERT, analyzing output embeddings shapes of single tokens).`
*   **Interview Edge (Advanced Attributions)**: *"How does BERT's bidirectional attention mechanism mathematically differ from causal (decoder-only) auto-regressive attention masks?"*

---

### 📅 Day 143 | 21 Oct 2026, Wed
*   **Phase**: Phase 11: Natural Language Processing
*   **Advanced Topic**: **Fine-tuning BERT for Text Classification**
*   **Deep-Dive Subtopics**: Hugging Face Hub ecosystem, using `Trainer` API, configuring `AutoModelForSequenceClassification`, handling inputs tokenization boundaries.
*   **Hands-on Deliverable**: `Day 143 - BERT Finetuning.ipynb (Fine-tuning DistilBERT on IMDb movie review dataset, tracking metrics via tensorboard).`
*   **Interview Edge (Advanced Attributions)**: *"What is the function of the `[CLS]` token in BERT, and why is its final hidden state vector utilized as the input for classification tasks?"*

---

### 📅 Day 144 | 22 Oct 2026, Thu
*   **Phase**: Phase 11: Natural Language Processing
*   **Advanced Topic**: **Named Entity Recognition (NER) & spaCy**
*   **Deep-Dive Subtopics**: NER sequence tag labels, standard BIO tag formats, utilizing spaCy pipeline architectures, training custom named entities models.
*   **Hands-on Deliverable**: `Day 144 - spaCy NER.ipynb (Training spaCy pipeline model to extract specialized custom entities like product codes).`
*   **Interview Edge (Advanced Attributions)**: *"What does BIO formatting mean in sequence labeling? Show how to label the sentence 'Sahil lives in Ludhiana' using BIO-NER tags."*

---

### 📅 Day 145 | 23 Oct 2026, Fri
*   **Phase**: Phase 11: Natural Language Processing
*   **Advanced Topic**: **Topic Modeling — LDA & NMF**
*   **Deep-Dive Subtopics**: Latent Dirichlet Allocation generative math, Dirichlet priors, Non-Negative Matrix Factorization, checking semantic coherence.
*   **Hands-on Deliverable**: `Day 145 - Topic Modeling.ipynb (Running LDA on text collections and analyzing topic coherence graphs).`
*   **Interview Edge (Advanced Attributions)**: *"Explain the generative process of Latent Dirichlet Allocation (LDA) and the physical interpretation of its alpha and beta hyperparameters."*

---

### 📅 Day 146 | 24 Oct 2026, Sat
*   **Phase**: Phase 11: Natural Language Processing
*   **Advanced Topic**: **Sequence-to-Sequence & Machine Translation**
*   **Deep-Dive Subtopics**: Seq2Seq architectures, encoder-decoder models with attention, implementing translation networks, BLEU evaluation metrics.
*   **Hands-on Deliverable**: `Day 146 - Seq2Seq Translation.ipynb (Training PyTorch translation model, evaluating performance using nltk BLEU calculations).`
*   **Interview Edge (Advanced Attributions)**: *"How does teacher forcing mathematically accelerate Seq2Seq model training, and what is the training-to-test divergence risk associated with it?"*

---

### 📅 Day 147 | 25 Oct 2026, Sun
*   **Phase**: Phase 11: Natural Language Processing
*   **Advanced Topic**: **Text Summarization**
*   **Deep-Dive Subtopics**: Extractive vs. Abstractive summaries, pre-trained BART and T5 model structures, evaluating outputs with ROUGE metrics.
*   **Hands-on Deliverable**: `Day 147 - Summarization.ipynb (Generating abstractive document summaries using pre-trained Hugging Face T5 models).`
*   **Interview Edge (Advanced Attributions)**: *"What are the mathematical formulations of ROUGE-1, ROUGE-2, and ROUGE-L metrics? Contrast precision and recall in ROUGE."*

---

### 📅 Day 148 | 26 Oct 2026, Mon
*   **Phase**: Phase 11: Natural Language Processing
*   **Advanced Topic**: **Sentiment Analysis Project**
*   **Deep-Dive Subtopics**: Fine-tuning RoBERTa classifier, performing error auditing, confusion matrix calculations, exporting production-ready weights.
*   **Hands-on Deliverable**: `Day 148 - Sentiment Analysis Project.ipynb (Robust movie/product review sentiment classifier project with evaluation scripts).`
*   **Interview Edge (Advanced Attributions)**: *"Why does RoBERTa perform significantly better than standard BERT on sentiment classification tasks? List the pretraining modifications."*

---

### 📅 Day 149 | 27 Oct 2026, Tue
*   **Phase**: Phase 11: Natural Language Processing
*   **Advanced Topic**: **Phase Review & NLP Assessment**
*   **Deep-Dive Subtopics**: Transformer self-attention layers math, sequence tokenizer constraints, classification metrics, NLP interview checklists.
*   **Hands-on Deliverable**: `Day 149 - NLP Assessment.ipynb (Written solutions to 15 complex transformer and language processing interview questions).`
*   **Interview Edge (Advanced Attributions)**: *"Derive the computational complexity of the Self-Attention mechanism of sequence length $L$. Why does this make long contexts challenging?"*

---

### 📅 Day 150 | 28 Oct 2026, Wed
*   **Phase**: Phase 12: Computer Vision
*   **Advanced Topic**: **OpenCV Image Basics & Operations**
*   **Deep-Dive Subtopics**: BGR color space representations, matrix slicing, pixel coordinates transformations, OpenCV arithmetic overlays, color masks.
*   **Hands-on Deliverable**: `Day 150 - OpenCV Basics.ipynb (Reading, crop/scaling, masking colored regions using numpy and cv2 libraries).`
*   **Interview Edge (Advanced Attributions)**: *"How are multi-channel image frames structured in computer memory? Compare row-major vs. column-last arrangements."*

---

### 📅 Day 151 | 29 Oct 2026, Thu
*   **Phase**: Phase 12: Computer Vision
*   **Advanced Topic**: **Image Processing Filters**
*   **Deep-Dive Subtopics**: Spatial kernels transformations, box filters, Gaussian blur equations, morphology: dilation, erosion, opening, closing transformations.
*   **Hands-on Deliverable**: `Day 151 - Image Filters.ipynb (Applying custom blur, sharpening, and morphological filters to image files).`
*   **Interview Edge (Advanced Attributions)**: *"Write the mathematical representation of a 2D Gaussian filter kernel. How does kernel size affect high-frequency noise suppression?"*

---

### 📅 Day 152 | 30 Oct 2026, Fri
*   **Phase**: Phase 12: Computer Vision
*   **Advanced Topic**: **Edge Detection & Contours**
*   **Deep-Dive Subtopics**: Image gradient calculations, Sobel filters, Canny edge detection algorithm steps, contour mapping, drawing external boundaries.
*   **Hands-on Deliverable**: `Day 152 - Edge Contours.ipynb (Applying Canny edge detectors, isolating and measuring external shapes coordinates).`
*   **Interview Edge (Advanced Attributions)**: *"Detail the mathematical steps of Canny Edge Detection, specifically non-maximum suppression and hysteresis thresholding."*

---

### 📅 Day 153 | 31 Oct 2026, Sat
*   **Phase**: Phase 12: Computer Vision
*   **Advanced Topic**: **Feature Detection — SIFT & ORB**
*   **Deep-Dive Subtopics**: SIFT scale-space extremum, keypoints detection, ORB (Oriented FAST and Rotated BRIEF), brute-force and FLANN descriptor matching.
*   **Hands-on Deliverable**: `Day 153 - Feature Detection.ipynb (Extracting matching features between images under rotational and scale shifts).`
*   **Interview Edge (Advanced Attributions)**: *"Explain why ORB is rotationally invariant. Contrast local keypoint features (SIFT/ORB) with global deep representations."*

---

### 📅 Day 154 | 01 Nov 2026, Sun
*   **Phase**: Phase 12: Computer Vision
*   **Advanced Topic**: **Object Detection Concepts**
*   **Deep-Dive Subtopics**: Bounding boxes, multi-class predictions, Intersection over Union (IoU) calculation, anchor boxes, mean Average Precision (mAP) metrics.
*   **Hands-on Deliverable**: `Day 154 - Object Detection Concepts.ipynb (Writing IoU computation and non-maximum suppression (NMS) functions in pure NumPy).`
*   **Interview Edge (Advanced Attributions)**: *"Derive the formula for Mean Average Precision (mAP) at threshold IoU@0.5. How is the recall-precision curve calculated?"*

---

### 📅 Day 155 | 02 Nov 2026, Mon
*   **Phase**: Phase 12: Computer Vision
*   **Advanced Topic**: **YOLO Intuition & Architecture**
*   **Deep-Dive Subtopics**: One-stage spatial grid predictions, bounding box regressions, anchor-free YOLO setups, YOLO objective loss functions.
*   **Hands-on Deliverable**: `Day 155 - YOLO Intuition.ipynb (Custom script dividing image into grids and calculating boundary regressions math).`
*   **Interview Edge (Advanced Attributions)**: *"Explain the categorical grid division strategy in YOLO. How does YOLOv8 perform anchor-free detection compared to older YOLO models?"*

---

### 📅 Day 156 | 03 Nov 2026, Tue
*   **Phase**: Phase 12: Computer Vision
*   **Advanced Topic**: **YOLOv8 Implementation & Inference**
*   **Deep-Dive Subtopics**: Ultralytics library API, running pre-trained classification, segmentation, and detection models, processing inference frames.
*   **Hands-on Deliverable**: `Day 156 - YOLOv8 Inference.ipynb (Detecting classes on stock street video frames, plotting prediction outputs).`
*   **Interview Edge (Advanced Attributions)**: *"What is the structure of the output tensor returned by a standard YOLOv8 object detection model?"*

---

### 📅 Day 157 | 04 Nov 2026, Wed
*   **Phase**: Phase 12: Computer Vision
*   **Advanced Topic**: **YOLO Custom Training & Roboflow**
*   **Deep-Dive Subtopics**: Formatting annotations datasets, setting up YAML files, custom training weights optimization, tracking metrics (mAP50-95).
*   **Hands-on Deliverable**: `Day 157 - YOLO Training.ipynb (Training custom YOLOv8 detector on custom annotated dataset, plotting precision graphs).`
*   **Interview Edge (Advanced Attributions)**: *"How do you prepare dataset annotations in YOLO format? Explain the class bounding box coordinate calculations (center x, y, width, height)."*

---

### 📅 Day 158 | 05 Nov 2026, Thu
*   **Phase**: Phase 12: Computer Vision
*   **Advanced Topic**: **Image Segmentation — U-Net & Mask R-CNN**
*   **Deep-Dive Subtopics**: Semantic vs. Instance segmentation, U-Net encoder contracting path, skip connections, up-convolution decoder paths.
*   **Hands-on Deliverable**: `Day 158 - Segmentation.ipynb (Implementing U-Net model architecture in PyTorch for pixel-level semantic mask segmentation).`
*   **Interview Edge (Advanced Attributions)**: *"Why are skip connections vital inside U-Net architectures? Explain how they preserve high-resolution spatial details."*

---

### 📅 Day 159 | 06 Nov 2026, Fri
*   **Phase**: Phase 12: Computer Vision
*   **Advanced Topic**: **Face Detection & Recognition**
*   **Deep-Dive Subtopics**: Cascades models vs. MTCNN face detections, generating facial embeddings using FaceNet, cosine similarity validation tests.
*   **Hands-on Deliverable**: `Day 159 - Face Recognition.ipynb (Real-time face verification checking facial embedding similarity using cosine distances).`
*   **Interview Edge (Advanced Attributions)**: *"How does FaceNet utilize Triplet Loss during training to minimize distance between anchor-positive and maximize anchor-negative faces?"*

---

### 📅 Day 160 | 07 Nov 2026, Sat
*   **Phase**: Phase 12: Computer Vision
*   **Advanced Topic**: **Stable Diffusion & Image Generation**
*   **Deep-Dive Subtopics**: Forward and reverse diffusion math, text embeddings via CLIP encoder, latent space denoising steps, UNet scheduler loop.
*   **Hands-on Deliverable**: `Day 160 - Diffusion Models.ipynb (Running text-to-image and image-to-image generation scripts via Hugging Face diffusers).`
*   **Interview Edge (Advanced Attributions)**: *"How does Stable Diffusion perform its denoising process in latent space rather than pixel space? Explain the computational efficiency."*

---

### 📅 Day 161 | 08 Nov 2026, Sun
*   **Phase**: Phase 12: Computer Vision
*   **Advanced Topic**: **Video Processing & Optical Flow**
*   **Deep-Dive Subtopics**: OpenCV VideoCapture streams, tracking movement via Lucas-Kanade optical flow, frame-by-frame background subtraction.
*   **Hands-on Deliverable**: `Day 161 - Video Processing.ipynb (Video movement tracker plotting trajectory lines on live webcam feeds).`
*   **Interview Edge (Advanced Attributions)**: *"Explain the mathematical assumptions of Lucas-Kanade optical flow, specifically brightness constancy and spatial coherence."*

---

### 📅 Day 162 | 09 Nov 2026, Mon
*   **Phase**: Phase 12: Computer Vision
*   **Advanced Topic**: **Image Captioning & VQA**
*   **Deep-Dive Subtopics**: Visual-Language encoders-decoders, CNN feature map extractions feed into LSTM text decoders, Visual Question Answering.
*   **Hands-on Deliverable**: `Day 162 - Image Captioning.ipynb (Generating descriptive textual captions from uploaded image frames).`
*   **Interview Edge (Advanced Attributions)**: *"Explain how attention weights help sequential text models focus on specific spatial grid regions of CNN feature maps during decoding."*

---

### 📅 Day 163 | 10 Nov 2026, Tue
*   **Phase**: Phase 12: Computer Vision
*   **Advanced Topic**: **Multi-Modal Models — CLIP & LLaVA**
*   **Deep-Dive Subtopics**: Contrastive Image-Text pretraining, mapping image-text representations to unified latent embeddings, zero-shot visual tagging.
*   **Hands-on Deliverable**: `Day 163 - Multimodal.ipynb (Running zero-shot image tagging and visual question tasks using pre-trained CLIP and LLaVA).`
*   **Interview Edge (Advanced Attributions)**: *"Derive the Contrastive Loss function utilized in training CLIP. How does it align multi-modal representations?"*

---

### 📅 Day 164 | 11 Nov 2026, Wed
*   **Phase**: Phase 12: Computer Vision
*   **Advanced Topic**: **Object Detection Application Project**
*   **Deep-Dive Subtopics**: Custom webcam stream dashboard: integrating custom YOLOv8 model inside real-time Streamlit dashboard UI.
*   **Hands-on Deliverable**: `Day 164 - CV Capstone Project.ipynb (Webcam real-time detection stream app with adjustable parameter slider UI).`
*   **Interview Edge (Advanced Attributions)**: *"How do you optimize frame-by-frame inference pipeline to prevent memory leaks and maintain steady FPS inside a Python app?"*

---

### 📅 Day 165 | 12 Nov 2026, Thu
*   **Phase**: Phase 13: MLOps — Infrastructure & APIs
*   **Advanced Topic**: **MLOps Intro & Workflow Lifecycle**
*   **Deep-Dive Subtopics**: Standardized workflows (Experimentation, reproducible environments, serialization, testing deployments, feedback loops).
*   **Hands-on Deliverable**: `Day 165 - MLOps Intro.ipynb (Audit checklist and architecture layout templates for production-ready AI services).`
*   **Interview Edge (Advanced Attributions)**: *"What are the key differences between standard DevOps and Machine Learning Operations (MLOps)?"*

---

### 📅 Day 166 | 13 Nov 2026, Fri
*   **Phase**: Phase 13: MLOps — Infrastructure & APIs
*   **Advanced Topic**: **Docker Basics — Images & Containers**
*   **Deep-Dive Subtopics**: Virtualization vs. containerization, Docker Engine, writing Dockerfiles (`FROM`, `RUN`, `COPY`, `EXPOSE`, `CMD`), container commands.
*   **Hands-on Deliverable**: `Day 166 - Docker Basics.ipynb (Constructing and launching lightweight python docker environments).`
*   **Interview Edge (Advanced Attributions)**: *"Explain how Docker layering works and how to optimize Dockerfile instructions to leverage build cache and minimize final image sizes."*

---

### 📅 Day 167 | 14 Nov 2026, Sat
*   **Phase**: Phase 13: MLOps — Infrastructure & APIs
*   **Advanced Topic**: **Docker Compose for Multi-Container Apps**
*   **Deep-Dive Subtopics**: Orchestrating services using `docker-compose.yml`, managing ports maps, networks interfaces, persistent volume mounts.
*   **Hands-on Deliverable**: `Day 167 - Docker Compose.ipynb (Setting up Docker Compose to orchestrate API backend, Redis database, and client services).`
*   **Interview Edge (Advanced Attributions)**: *"Why is it bad practice to run database systems inside unmounted docker containers in production? Explain volume bindings."*

---

### 📅 Day 168 | 15 Nov 2026, Sun
*   **Phase**: Phase 13: MLOps — Infrastructure & APIs
*   **Advanced Topic**: **FastAPI Basics**
*   **Deep-Dive Subtopics**: Asynchronous ASGI servers (Uvicorn), Pydantic Request schemas validation, path parameters, query variables.
*   **Hands-on Deliverable**: `Day 168 - FastAPI Basics.ipynb (Constructing a simple asynchronous API with request validations).`
*   **Interview Edge (Advanced Attributions)**: *"How does FastAPI use Pydantic to enforce type checking and input sanitization at the ASGI application boundary?"*

---

### 📅 Day 169 | 16 Nov 2026, Mon
*   **Phase**: Phase 13: MLOps — Infrastructure & APIs
*   **Advanced Topic**: **Serving ML Models with FastAPI**
*   **Deep-Dive Subtopics**: Optimized setups: loading serialized model weights once inside startup event hook, defining asynchronous post inference endpoints.
*   **Hands-on Deliverable**: `Day 169 - FastAPI Model Serving.ipynb (Asynchronous API serving predictions for an active classification model).`
*   **Interview Edge (Advanced Attributions)**: *"Why is it critical to load model weights during FastAPI startup rather than inside the POST endpoint function call?"*

---

### 📅 Day 170 | 17 Nov 2026, Tue
*   **Phase**: Phase 13: MLOps — Infrastructure & APIs
*   **Advanced Topic**: **Containerizing FastAPI App with Docker**
*   **Deep-Dive Subtopics**: Building lightweight model serving containers, utilizing multi-stage Docker builds, handling model dependencies.
*   **Hands-on Deliverable**: `Day 170 - Dockerized FastAPI.ipynb (Dockerfile compiling python dependencies and serving model API via Uvicorn in container).`
*   **Interview Edge (Advanced Attributions)**: *"How does multi-stage building in Docker help reduce final image sizes and secure production model weights?"*

---

### 📅 Day 171 | 18 Nov 2026, Wed
*   **Phase**: Phase 13: MLOps — Infrastructure & APIs
*   **Advanced Topic**: **Streamlit for Quick UI Prototyping**
*   **Deep-Dive Subtopics**: Constructing fast web inputs interfaces (buttons, file uploaders, sliders, markdown formats), managing interactive states.
*   **Hands-on Deliverable**: `Day 171 - Streamlit UI.ipynb (Streamlit client mockup for tabular classifiers prediction interactions).`
*   **Interview Edge (Advanced Attributions)**: *"How does Streamlit's rendering loop work? Explain how session state is utilized to prevent variable resets during re-renders."*

---

### 📅 Day 172 | 19 Nov 2026, Thu
*   **Phase**: Phase 13: MLOps — Infrastructure & APIs
*   **Advanced Topic**: **Streamlit Interactive Dashboard Project**
*   **Deep-Dive Subtopics**: Building modular dashboard displaying target datasets statistics, distribution plots, and calling local model predictions.
*   **Hands-on Deliverable**: `Day 172 - Streamlit Project.ipynb (Polished UI dashboard allowing user to upload custom tabular files and visualize predictions).`
*   **Interview Edge (Advanced Attributions)**: *"Why should you split the Streamlit dashboard UI code from the model prediction math via an API boundary?"*

---

### 📅 Day 173 | 20 Nov 2026, Fri
*   **Phase**: Phase 13: MLOps — Infrastructure & APIs
*   **Advanced Topic**: **Model Serving Optimization (ONNX Runtime)**
*   **Deep-Dive Subtopics**: Serializing PyTorch models to ONNX (Open Neural Network Exchange), running optimized predictions with ONNX Runtime.
*   **Hands-on Deliverable**: `Day 173 - ONNX Conversion.ipynb (Converting classification model to ONNX, benchmarking CPU prediction speedups).`
*   **Interview Edge (Advanced Attributions)**: *"How does ONNX optimize computation graphs (layer fusions, constant folding) to accelerate inference execution times?"*

---

### 📅 Day 174 | 21 Nov 2026, Sat
*   **Phase**: Phase 13: MLOps — Infrastructure & APIs
*   **Advanced Topic**: **Phase Review & App Deployment**
*   **Deep-Dive Subtopics**: Verifying API payloads, docker logs inspection, running stress tests on local API endpoints, code audits.
*   **Hands-on Deliverable**: `Day 174 - API Deployment Review.ipynb (Written solutions to 12 API architecture and containerization questions).`
*   **Interview Edge (Advanced Attributions)**: *"Explain how ASGI servers manage concurrent incoming prediction requests using asynchronous event loops under heavy traffic loads."*

---

### 📅 Day 175 | 22 Nov 2026, Sun
*   **Phase**: Phase 14: MLOps — Experiment Tracking, Pipelines & CI/CD
*   **Advanced Topic**: **MLflow Experiment Tracking**
*   **Deep-Dive Subtopics**: Experiment logging, tracking continuous hyperparameters and evaluations metrics using `mlflow.log_param`, `mlflow.log_metric`, `mlflow.autolog()`.
*   **Hands-on Deliverable**: `Day 175 - MLflow Tracking.ipynb (MLflow tracking server integrations logging multiple booster run results).`
*   **Interview Edge (Advanced Attributions)**: *"Why is centralized experiment tracking critical in collaborative team environments? Describe how MLflow stores model runs metadata."*

---

### 📅 Day 176 | 23 Nov 2026, Mon
*   **Phase**: Phase 14: MLOps — Experiment Tracking, Pipelines & CI/CD
*   **Advanced Topic**: **MLflow Model Registry & Stage Management**
*   **Deep-Dive Subtopics**: Registering model versions, managing production, staging, and archived lifecycle gates via python APIs.
*   **Hands-on Deliverable**: `Day 176 - MLflow Registry.ipynb (Script querying MLflow and programmatically promoting top-scoring model to Production stage).`
*   **Interview Edge (Advanced Attributions)**: *"How do you implement model stage gates in MLflow programmatically? Describe the model versioning validation flow."*

---

### 📅 Day 177 | 24 Nov 2026, Tue
*   **Phase**: Phase 14: MLOps — Experiment Tracking, Pipelines & CI/CD
*   **Advanced Topic**: **DVC (Data Version Control) Basics**
*   **Deep-Dive Subtopics**: DVC architecture, data files serialization mapping, storing pointer hashes in Git, configuring remote storage on AWS S3.
*   **Hands-on Deliverable**: `Day 177 - DVC Basics.ipynb (Setting up DVC repository tracking multi-gigabyte raw dataset files).`
*   **Interview Edge (Advanced Attributions)**: *"Why should raw dataset files never be checked directly into Git repositories? Explain how DVC pointer files solve this."*

---

### 📅 Day 178 | 25 Nov 2026, Wed
*   **Phase**: Phase 14: MLOps — Experiment Tracking, Pipelines & CI/CD
*   **Advanced Topic**: **DVC Pipeline Integration**
*   **Deep-Dive Subtopics**: Defining stages inside `dvc.yaml` (dependencies, outputs), running pipeline reproducible loops using `dvc repro`.
*   **Hands-on Deliverable**: `Day 178 - DVC Pipelines.ipynb (Constructing modular training pipeline with separate clean, feature-engineer, train steps).`
*   **Interview Edge (Advanced Attributions)**: *"How does DVC determine if a pipeline stage needs to be re-run or if it can be retrieved from cache? Describe state tracking."*

---

### 📅 Day 179 | 26 Nov 2026, Thu
*   **Phase**: Phase 14: MLOps — Experiment Tracking, Pipelines & CI/CD
*   **Advanced Topic**: **Feast Feature Store Concepts**
*   **Deep-Dive Subtopics**: Establishing unified features layers, offline features retrievals (training), online features stores (Redis low-latency serving).
*   **Hands-on Deliverable**: `Day 179 - Feature Store.ipynb (Setting up Feast registry and retrieving feature views for real-time model inputs).`
*   **Interview Edge (Advanced Attributions)**: *"What is a Feature Store? How does Feast prevent online-offline feature skew during model training and serving?"*

---

### 📅 Day 180 | 27 Nov 2026, Fri
*   **Phase**: Phase 14: MLOps — Experiment Tracking, Pipelines & CI/CD
*   **Advanced Topic**: **GitHub Actions CI/CD Basics**
*   **Deep-Dive Subtopics**: Continuous Integration concepts, YAML configurations, steps, runners, managing workspace environments, storing encrypted secrets.
*   **Hands-on Deliverable**: `Day 180 - Github Actions.ipynb (Baseline workflow file executing lint checks and formatting on push to main).`
*   **Interview Edge (Advanced Attributions)**: *"How does CI/CD automation help teams maintain model code quality? Explain the function of runners and triggers."*

---

### 📅 Day 181 | 28 Nov 2026, Sat
*   **Phase**: Phase 14: MLOps — Experiment Tracking, Pipelines & CI/CD
*   **Advanced Topic**: **CI/CD Pipeline for Automated Testing**
*   **Deep-Dive Subtopics**: Writing Pytest assertions testing classification inputs shapes, verifying data scalers properties, mock data inputs testing.
*   **Hands-on Deliverable**: `Day 181 - Model Unit Testing.ipynb (Writing robust pytest script testing model utility directories and data types validation).`
*   **Interview Edge (Advanced Attributions)**: *"Why is writing unit tests for machine learning code more challenging than traditional software? List three ML-specific test cases."*

---

### 📅 Day 182 | 29 Nov 2026, Sun
*   **Phase**: Phase 14: MLOps — Experiment Tracking, Pipelines & CI/CD
*   **Advanced Topic**: **CI/CD for Model Training & Deployment**
*   **Deep-Dive Subtopics**: Writing GitHub Actions compiling docker images, running pytest suites on PR, pushing validated images to Docker registry.
*   **Hands-on Deliverable**: `Day 182 - Github CD Pipeline.ipynb (Continuous delivery workflow building and uploading backend containers on successful build).`
*   **Interview Edge (Advanced Attributions)**: *"How do you implement a automated rollback strategy in your CD pipeline if a newly deployed model container fails health checks?"*

---

### 📅 Day 183 | 30 Nov 2026, Mon
*   **Phase**: Phase 14: MLOps — Experiment Tracking, Pipelines & CI/CD
*   **Advanced Topic**: **Model Monitoring — Data Drift**
*   **Deep-Dive Subtopics**: Data distribution drift checks, generating interactive drift reports with Evidently AI, Kolmogorov-Smirnov test validations.
*   **Hands-on Deliverable**: `Day 183 - Data Drift.ipynb (Simulating production drift on tabular columns and producing diagnostic reports).`
*   **Interview Edge (Advanced Attributions)**: *"What is data drift? Describe the statistical math behind the Kolmogorov-Smirnov test used to identify drift in continuous features."*

---

### 📅 Day 184 | 01 Dec 2026, Tue
*   **Phase**: Phase 14: MLOps — Experiment Tracking, Pipelines & CI/CD
*   **Advanced Topic**: **Model Monitoring — Concept Drift**
*   **Deep-Dive Subtopics**: Concept drift vs. data drift, tracking true classification performance degradation, triggering automated retraining pipelines.
*   **Hands-on Deliverable**: `Day 184 - Concept Drift.ipynb (Pipeline monitoring error metrics, triggering model updates if threshold exceeds).`
*   **Interview Edge (Advanced Attributions)**: *"What is concept drift? Provide an example of how changing real-world consumer behavior results in concept drift, and how to detect it."*

---

### 📅 Day 185 | 02 Dec 2026, Wed
*   **Phase**: Phase 14: MLOps — Experiment Tracking, Pipelines & CI/CD
*   **Advanced Topic**: **Kubernetes Basics**
*   **Deep-Dive Subtopics**: Container orchestration architecture: Master vs. worker nodes, Pods configs, Deployments structures, Services load balancers.
*   **Hands-on Deliverable**: `Day 185 - Kubernetes Basics.ipynb (Setting up local Kubernetes clusters, running kubectl deployments commands).`
*   **Interview Edge (Advanced Attributions)**: *"Explain the role of Pods, Deployments, and Services in Kubernetes. How does Kubernetes manage high service availability?"*

---

### 📅 Day 186 | 03 Dec 2026, Thu
*   **Phase**: Phase 14: MLOps — Experiment Tracking, Pipelines & CI/CD
*   **Advanced Topic**: **Kubernetes Service Deployment**
*   **Deep-Dive Subtopics**: Writing scaling deployment YAML configurations, setting up service endpoints, load balancing local multi-pod services.
*   **Hands-on Deliverable**: `Day 186 - Kubernetes Deploy.ipynb (Kubernetes configuration YAML directory serving scalable model endpoints).`
*   **Interview Edge (Advanced Attributions)**: *"Explain how Kubernetes handles rolling-update deployment strategies for model serving containers to ensure zero downtime."*

---

### 📅 Day 187 | 04 Dec 2026, Fri
*   **Phase**: Phase 14: MLOps — Experiment Tracking, Pipelines & CI/CD
*   **Advanced Topic**: **Cloud Deployment Basics — AWS SageMaker**
*   **Deep-Dive Subtopics**: Registering images, setting up training jobs in SageMaker, deploying real-time prediction endpoints, configuring storage buckets.
*   **Hands-on Deliverable**: `Day 187 - AWS SageMaker.ipynb (Setting up AWS client connections, deploying containerized endpoints programmatically).`
*   **Interview Edge (Advanced Attributions)**: *"How does AWS SageMaker manage model training scaling via managed EC2 instances? What are the benefits of multi-model endpoints?"*

---

### 📅 Day 188 | 05 Dec 2026, Sat
*   **Phase**: Phase 14: MLOps — Experiment Tracking, Pipelines & CI/CD
*   **Advanced Topic**: **Cloud Deployment Basics — GCP Vertex AI**
*   **Deep-Dive Subtopics**: Vertex Workbench notebooks, executing containerized pipelines, deploying prediction endpoints using Vertex container registries.
*   **Hands-on Deliverable**: `Day 188 - GCP Vertex AI.ipynb (Configuring Vertex model uploads and launching prediction endpoints on Vertex AI).`
*   **Interview Edge (Advanced Attributions)**: *"Describe the architecture of Vertex AI Pipelines. How does it leverage Kubeflow under the hood for model pipeline tracking?"*

---

### 📅 Day 189 | 06 Dec 2026, Sun
*   **Phase**: Phase 14: MLOps — Experiment Tracking, Pipelines & CI/CD
*   **Advanced Topic**: **Flagship MLOps Pipeline Capstone Project**
*   **Deep-Dive Subtopics**: Constructing complete unified environment: integrating DVC dataset versions, training with MLflow tracking, automated pytest checks in Git Actions.
*   **Hands-on Deliverable**: `Day 189 - MLOps Capstone Project.ipynb (Robust MLOps repository of an automated, validated, tracked production pipeline).`
*   **Interview Edge (Advanced Attributions)**: *"How do you design a complete MLOps pipeline that securely triggers retraining when drift is detected, without human intervention?"*

---

### 📅 Day 190 | 07 Dec 2026, Mon
*   **Phase**: Phase 15: LLMs & Generative AI
*   **Advanced Topic**: **LLM Architectures & Tokenization**
*   **Deep-Dive Subtopics**: Decoder-only transformer models (GPT family), Byte-Pair Encoding (BPE), SentencePiece algorithms, attention vocabulary limits.
*   **Hands-on Deliverable**: `Day 190 - LLM Architecture.ipynb (Visualizing tokenization outputs using huggingface tokenizers, comparing vocab splits).`
*   **Interview Edge (Advanced Attributions)**: *"Explain how subword tokenizers handle Out-of-Vocabulary (OOV) tokens mathematically, and why character-level tokenization is sub-optimal."*

---

### 📅 Day 191 | 08 Dec 2026, Tue
*   **Phase**: Phase 15: LLMs & Generative AI
*   **Advanced Topic**: **Prompt Engineering**
*   **Deep-Dive Subtopics**: System vs. User instruction alignments, zero-shot and few-shot formatting patterns, Chain-of-Thought prompting, ReAct templates.
*   **Hands-on Deliverable**: `Day 191 - Prompt Engineering.ipynb (Systematic prompt evaluations measuring output variations as system prompt changes).`
*   **Interview Edge (Advanced Attributions)**: *"What is the mathematical rationale behind Chain-of-Thought prompting? How does letting models write intermediate steps improve outcomes?"*

---

### 📅 Day 192 | 09 Dec 2026, Wed
*   **Phase**: Phase 15: LLMs & Generative AI
*   **Advanced Topic**: **OpenAI API & Chat Completions**
*   **Deep-Dive Subtopics**: OpenAI Python library calls, parameters configuration (temperature, top_p, frequency_penalty), executing structured API function calls.
*   **Hands-on Deliverable**: `Day 192 - OpenAI API.ipynb (Connecting OpenAI client, generating structured JSON outputs using native function calling arguments).`
*   **Interview Edge (Advanced Attributions)**: *"Contrast the effects of `temperature` vs. `top_p` sampling parameters on the probability distribution of generated tokens."*

---

### 📅 Day 193 | 10 Dec 2026, Thu
*   **Phase**: Phase 15: LLMs & Generative AI
*   **Advanced Topic**: **LangChain Basics — Chains & Prompts**
*   **Deep-Dive Subtopics**: PromptTemplates mapping, integrating chains, handling memory state transitions using ConversationBufferMemory.
*   **Hands-on Deliverable**: `Day 193 - LangChain Basics.ipynb (Constructing modular chain prompting pipelines processing sequential query actions).`
*   **Interview Edge (Advanced Attributions)**: *"How does LangChain manage conversation context windows under token constraints? Describe how semantic summarization pruning works."*

---

### 📅 Day 194 | 11 Dec 2026, Fri
*   **Phase**: Phase 15: LLMs & Generative AI
*   **Advanced Topic**: **LangChain Agents & Tools**
*   **Deep-Dive Subtopics**: Constructing autonomous agents, configuring ReAct loops, binding custom python tools functions to LLM decision layers.
*   **Hands-on Deliverable**: `Day 194 - LangChain Agents.ipynb (Agent equipped with custom tools searching local directories and computing complex formulas).`
*   **Interview Edge (Advanced Attributions)**: *"Explain how the ReAct framework organizes prompt sequences into 'Thought', 'Action', 'Action Input', and 'Observation' loops."*

---

### 📅 Day 195 | 12 Dec 2026, Sat
*   **Phase**: Phase 15: LLMs & Generative AI
*   **Advanced Topic**: **RAG Concepts**
*   **Deep-Dive Subtopics**: Document chunking strategies (recursive, semantic splits), generating vector embeddings via dense models, similarity search mathematics.
*   **Hands-on Deliverable**: `Day 195 - RAG Concepts.ipynb (Comparing text splitting outputs, computing cosine similarity matrices over embedding arrays from scratch).`
*   **Interview Edge (Advanced Attributions)**: *"Under what spatial distribution conditions is Cosine Similarity mathematically preferred over Dot Product for vector similarity searches?"*

---

### 📅 Day 196 | 13 Dec 2026, Sun
*   **Phase**: Phase 15: LLMs & Generative AI
*   **Advanced Topic**: **RAG Implementation — LangChain + Chroma**
*   **Deep-Dive Subtopics**: Ingesting multi-format files, chunking, loading vector embeddings to Chroma DB, building standard retrieval loops.
*   **Hands-on Deliverable**: `Day 196 - RAG Implementation.ipynb (Complete conversational retrieval chain answering user queries over local directories).`
*   **Interview Edge (Advanced Attributions)**: *"Explain the entire data flow of a standard RAG pipeline from user query ingestion to final generation output."*

---

### 📅 Day 197 | 14 Dec 2026, Mon
*   **Phase**: Phase 15: LLMs & Generative AI
*   **Advanced Topic**: **Advanced RAG**
*   **Deep-Dive Subtopics**: Implementing Query Expansion, Hypothetical Document Embeddings (HyDE), context compression filters, re-ranking via Cohere Rerank.
*   **Hands-on Deliverable**: `Day 197 - Advanced RAG.ipynb (Advanced retrieval pipeline yielding superior accuracy compared to standard baseline RAG configurations).`
*   **Interview Edge (Advanced Attributions)**: *"How does a cross-encoder re-ranker mathematically improve context relevance compared to standard bi-encoder similarity retrieval?"*

---

### 📅 Day 198 | 15 Dec 2026, Tue
*   **Phase**: Phase 15: LLMs & Generative AI
*   **Advanced Topic**: **Fine-Tuning LLMs — LoRA & QLoRA**
*   **Deep-Dive Subtopics**: Parameter-efficient fine-tuning (PEFT), low-rank adaptation matrices math ($W_0 + B \cdot A$), 4-bit double quantization, bitsandbytes.
*   **Hands-on Deliverable**: `Day 198 - LoRA Finetuning.ipynb (Fine-tuning open-source LLM on custom dataset using QLoRA configurations on consumer GPUs).`
*   **Interview Edge (Advanced Attributions)**: *"Derive the parameter savings math of LoRA. Why does QLoRA use NormalFloat4 (NF4) quantization instead of standard FP4?"*

---

### 📅 Day 199 | 16 Dec 2026, Wed
*   **Phase**: Phase 15: LLMs & Generative AI
*   **Advanced Topic**: **LLM Safety & Guardrails**
*   **Deep-Dive Subtopics**: Mitigating prompt injections, applying PII data sanitizations, designing validation layers using NeMo Guardrails safety policies.
*   **Hands-on Deliverable**: `Day 199 - LLM Safety.ipynb (Script validating and sanitizing inputs and outputs of active LLM pipelines).`
*   **Interview Edge (Advanced Attributions)**: *"How do you mathematically detect semantic similarity of prompt inputs to identify adversarial prompt injection vectors?"*

---

### 📅 Day 200 | 17 Dec 2026, Thu
*   **Phase**: Phase 16: Agentic AI & Systems
*   **Advanced Topic**: **AI Agents & Autonomous Planning**
*   **Deep-Dive Subtopics**: Goal decomposition loops, state management, stateful agent loops, short-term and long-term memory configurations.
*   **Hands-on Deliverable**: `Day 200 - Agent Planning.ipynb (State-loop agent breaking down complex goals into sequential action queues from scratch).`
*   **Interview Edge (Advanced Attributions)**: *"Explain how autonomous agents use recursive self-correction to refine planning when a tool call returns an error."*

---

### 📅 Day 201 | 18 Dec 2026, Fri
*   **Phase**: Phase 16: Agentic AI & Systems
*   **Advanced Topic**: **Multi-Agent Systems — CrewAI**
*   **Deep-Dive Subtopics**: Defining specialist agents, configuring individual roles and backstories, sequential and hierarchical team collaborations.
*   **Hands-on Deliverable**: `Day 201 - CrewAI Systems.ipynb (Crew of researcher, writer, and editor agents producing polished technical writeups).`
*   **Interview Edge (Advanced Attributions)**: *"How do multi-agent systems coordinate workflows to minimize task dependencies and prevent circular planning loops?"*

---

### 📅 Day 202 | 19 Dec 2026, Sat
*   **Phase**: Phase 16: Agentic AI & Systems
*   **Advanced Topic**: **LangGraph — Stateful Agent Workflows**
*   **Deep-Dive Subtopics**: StateGraphs, Nodes (agents/computations), Edges (state transitions), managing state dictionary, cyclic graph loops.
*   **Hands-on Deliverable**: `Day 202 - LangGraph Basics.ipynb (Building stateful, cyclic conversational graphs using LangGraph API).`
*   **Interview Edge (Advanced Attributions)**: *"How does LangGraph maintain state consistency across multiple asynchronous agent executions? Describe graph state updates."*

---

### 📅 Day 203 | 20 Dec 2026, Sun
*   **Phase**: Phase 16: Agentic AI & Systems
*   **Advanced Topic**: **LangGraph — Conditional Edges & Human-in-the-Loop**
*   **Deep-Dive Subtopics**: Implementing conditional nodes, state validation routing, designing human validation interruption gates.
*   **Hands-on Deliverable**: `Day 203 - LangGraph Advanced.ipynb (Graph executing automated steps, breaking for human approval on specific actions).`
*   **Interview Edge (Advanced Attributions)**: *"How do you implement a human-in-the-loop state machine interruption inside a cyclic LangGraph environment?"*

---

### 📅 Day 204 | 21 Dec 2026, Mon
*   **Phase**: Phase 16: Agentic AI & Systems
*   **Advanced Topic**: **LLM Evaluation**
*   **Deep-Dive Subtopics**: BLEU, ROUGE, BERTScore, LLM-as-a-Judge evaluations frameworks (G-Eval criteria, metric alignment, testing reliability).
*   **Hands-on Deliverable**: `Day 204 - LLM Evaluation.ipynb (Setting up automated evaluation script scoring LLM answers against ground truth criteria).`
*   **Interview Edge (Advanced Attributions)**: *"What is BERTScore? How does LLM-as-a-Judge evaluate subjective generation criteria like tone, style, and helpfulness?"*

---

### 📅 Day 205 | 22 Dec 2026, Tue
*   **Phase**: Phase 16: Agentic AI & Systems
*   **Advanced Topic**: **RAG Chatbot Project**
*   **Deep-Dive Subtopics**: Integrating PDF doc ingestion, indexing vector embeddings inside Pinecone, LangChain backends, Streamlit chat interface.
*   **Hands-on Deliverable**: `Day 205 - RAG Chatbot Project.ipynb (Polished interactive chatbot running on local vector database index).`
*   **Interview Edge (Advanced Attributions)**: *"How do you configure an online vector database index like Pinecone to automatically scale and support low-latency RAG serving?"*

---

### 📅 Day 206 | 23 Dec 2026, Wed
*   **Phase**: Phase 16: Agentic AI & Systems
*   **Advanced Topic**: **Multi-Agent Research System Project**
*   **Deep-Dive Subtopics**: Autonomous multi-agent system executing live Google queries, compiling text resources, writing reports, editing layout.
*   **Hands-on Deliverable**: `Day 206 - Agent Project.ipynb (Streamlit app launching CrewAI multi-agent workspace and rendering PDF research reports).`
*   **Interview Edge (Advanced Attributions)**: *"What is the system design architecture of a resilient, fault-tolerant multi-agent system running on distributed APIs?"*

---

### 📅 Day 207 | 24 Dec 2026, Thu
*   **Phase**: Phase 16: Agentic AI & Systems
*   **Advanced Topic**: **Phase Review & GenAI Assessment**
*   **Deep-Dive Subtopics**: Review of generative and agentic systems, comparing RAG vs. fine-tuning, assessing agent safety models.
*   **Hands-on Deliverable**: `Day 207 - GenAI Review.ipynb (Written answers to 15 advanced generative and agentic AI system design questions).`
*   **Interview Edge (Advanced Attributions)**: *"Explain the semantic and economic trade-offs of using RAG vs. Fine-tuning for domain-specific enterprise knowledge extraction."*

---

### 📅 Day 208 | 25 Dec 2026, Fri
*   **Phase**: Phase 17: Career Preparation
*   **Advanced Topic**: **ML System Design Framework**
*   **Deep-Dive Subtopics**: System design requirements, data ingestion layers, model scaling, Triton Inference servers dynamic batching, logging, drift metrics.
*   **Hands-on Deliverable**: `Day 208 - System Design.ipynb (Architecture diagram worksheets designing high-throughput recommender/search pipelines).`
*   **Interview Edge (Advanced Attributions)**: *"Explain how you would design a high-throughput video recommendation system serving 100M users with sub-50ms latency."*

---

### 📅 Day 209 | 26 Dec 2026, Sat
*   **Phase**: Phase 17: Career Preparation
*   **Advanced Topic**: **ML Coding Interview Algorithms from Scratch**
*   **Deep-Dive Subtopics**: NumPy-only vectorizations: closed-form Linear Regression, KMeans cluster centroids, KNN classification, evaluation metrics calculations.
*   **Hands-on Deliverable**: `Day 209 - Coding Interview.ipynb (Script compiling complete from-scratch builds of classical models using NumPy).`
*   **Interview Edge (Advanced Attributions)**: *"Write a NumPy-only class for Ridge Regression that correctly computes regularized analytical weights."*

---

### 📅 Day 210 | 27 Dec 2026, Sun
*   **Phase**: Phase 17: Career Preparation
*   **Advanced Topic**: **Behavioral Interviews & STAR Method**
*   **Deep-Dive Subtopics**: Constructing behavioral templates: Situation, Task, Action, Result mapping of complex machine learning failures and successes.
*   **Hands-on Deliverable**: `Day 210 - Behavioral Prep.ipynb (Formulated sheets structuring stories for communication, leadership, and technical complexity).`
*   **Interview Edge (Advanced Attributions)**: *"How do you answer the question: 'Tell me about a time your production machine learning model failed, and how you fixed it?'"*

---

### 📅 Day 211 | 28 Dec 2026, Mon
*   **Phase**: Phase 17: Career Preparation
*   **Advanced Topic**: **Resume Polish & Portfolio Website**
*   **Deep-Dive Subtopics**: Resume polishing templates: quantifying metrics results, showcasing scalable systems, building online portfolios layout.
*   **Hands-on Deliverable**: `Day 211 - Resume Portfolio.ipynb (Complete polished markdown resume with structural metrics quantification examples).`
*   **Interview Edge (Advanced Attributions)**: *"How do you rewrite generic bullet points on an ML resume to emphasize engineering scale and quantified business impact?"*

---

### 📅 Day 212 | 29 Dec 2026, Tue
*   **Phase**: Phase 17: Career Preparation
*   **Advanced Topic**: **Portfolio Checkpoint & GitHub Optimization**
*   **Deep-Dive Subtopics**: Polishing GitHub profiles: adding professional README directories, system flowcharts diagrams, licensing, Docker configurations.
*   **Hands-on Deliverable**: `Day 212 - GitHub Audit.ipynb (Audit checklists verifying that all 14 projects contain setup guides and run commands).`
*   **Interview Edge (Advanced Attributions)**: *"Why is having reproducible Docker setup guides inside your project repositories crucial for passing technical screens?"*

---

### 📅 Day 213 | 30 Dec 2026, Wed
*   **Phase**: Phase 17: Career Preparation
*   **Advanced Topic**: **Mock Interviews & Practical Drills**
*   **Deep-Dive Subtopics**: Executing simulated mock drills: 1-hour coding tests, system design whiteboard case studies, mathematical examinations.
*   **Hands-on Deliverable**: `Day 213 - Mock Drills.ipynb (Diagnostic scoring template mapping performance across core technical interview criteria).`
*   **Interview Edge (Advanced Attributions)**: *"How do you systematically tackle an ambiguous ML System Design interview question under 45 minutes? Describe the steps."*

---

### 📅 Day 214 | 31 Dec 2026, Thu
*   **Phase**: Phase 17: Career Preparation
*   **Advanced Topic**: **Final Day Celebration & Continuous Learning Plan**
*   **Deep-Dive Subtopics**: Graduation checklists, setting up automated research newsletter feeds (arXiv, paperswithcode), active community contribution plans.
*   **Hands-on Deliverable**: `Day 214 - Final Journey.ipynb (Notebook indexing top AI research resources and continuous education tracking guidelines).`
*   **Interview Edge (Advanced Attributions)**: *"How do you maintain technical edge in a rapidly changing AI/ML industry? Detail your plan for continuing research paper reviews."*

---