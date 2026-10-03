# AI-ML-Blueprint
### From First-Principles Mathematics to Production ML & Deep Learning Systems

A comprehensive 365-day engineering roadmap implementing machine learning and deep learning algorithms from scratch, validating mathematical formulations, and engineering scalable production systems.

---

## Engineering Philosophy

> **"Understand the math from scratch. Optimize the algorithms for speed. Package the systems for scale."**

Most learning workflows rely on black-box estimators (`model.fit()`, `model.predict()`). This blueprint follows a rigorous first-principles paradigm:
1. **Derive the Mathematics:** Loss functions, closed-form equations, analytical gradient derivations, and optimization manifolds.
2. **Implement from Scratch:** Vectorized NumPy implementations without relying on high-level wrappers.
3. **Benchmark against Industry Baselines:** Compare convergence, loss curves, and computational runtimes against standard libraries (`scikit-learn`, `PyTorch`).
4. **End-to-End System Packaging:** Deploy models with data validation pipelines, RESTful APIs, containerization, and monitoring.

---

## Curriculum & Repository Architecture

The blueprint spans **12 core phases** mapped across a 365-day curriculum:

```text
AI-ML-Blueprint/
|-- Phase-0 - EDA & Feature Engineering/         # Preprocessing, Transformations & Scaling
|-- Phase-1 - Linear Regression & Regularization/  # OLS, GD, Ridge, Lasso, ElasticNet
|-- Phase-2 - Logistic Regression & Classif./     # Sigmoid, Cross-Entropy, Softmax, ROC-AUC
|-- Phase-3 - Tree Models & SVM/                 # CART, Random Forests, Margin Classifiers
|-- Phase-4 - Boosting & Advanced Ensembles/     # AdaBoost, Gradient Boost, XGBoost, LightGBM
|-- Phase-5 - Unsupervised Learning/             # Lloyd's K-Means, K-Means++, PCA, Time Series
|-- Phase-6 - Deep Learning & PyTorch/           # Backprop, Optimizers (AdamW), Regularization
|-- End-to-End Customer Churn Prediction System/  # Flagship Production Tabular ML Project
|-- Titanic/                                     # Classical Feature Engineering Benchmark
`-- boosting-visualizer/                         # Decision Boundary Interactive Tools
```

---

## Phase-by-Phase Progress & Index

| Phase | Core Domain | Focus Topics & Implementations | Status |
| :--- | :--- | :--- | :---: |
| **Phase 0** | **EDA & Feature Engineering** | Distribution analysis, Outlier detection (IQR, Z-Score), Power transforms (Box-Cox, Yeo-Johnson), Target/One-Hot encoding, Missing value imputation pipelines | Completed |
| **Phase 1** | **Linear Regression & Regularization** | Normal Equation closed-form solver, Gradient Descent variants (Batch/SGD), Ridge ($L_2$ shrinkage), Lasso ($L_1$ sparsity), ElasticNet grid | Completed |
| **Phase 2** | **Logistic Regression & Classification** | Sigmoid & Log-Loss derivation, Multiclass Softmax, OvR vs OvO decision boundaries, Precision-Recall & ROC-AUC curves, Cost-sensitive matrices | Completed |
| **Phase 3** | **Tree Models & Support Vector Machines** | CART splitting criterion (Gini vs Entropy), Pruning strategies, Random Forest bagging & out-of-bag error, Hard/Soft margin SVM, Kernel trick (RBF/Poly) | Completed |
| **Phase 4** | **Boosting & Advanced Ensembles** | Exponential loss & AdaBoost weight updates, Gradient Boosting pseudo-residuals, XGBoost second-order Taylor expansion & histogram pruning, LightGBM, CatBoost | Completed |
| **Phase 5** | **Unsupervised Learning & Time Series** | Lloyd's Algorithm, K-Means++ $D(x)^2$ seeding, Elbow method vs Silhouette Analysis, DBSCAN density reachability, PCA from scratch (Covariance to SVD), Time series decomposition (ARIMA/SARIMA) | Completed |
| **Phase 6** | **Deep Learning Foundations & PyTorch** | Computational Graphs & Autograd, Forward/Backward propagation scratch math, Vanishing/Exploding gradients, Mini-Batch GD, Momentum/NAG, Adam & AdamW decoupling, He/Xavier initialization, Inverted Dropout | Active (Day 078-092+) |
| **Phase 7-12** | **CV, NLP, LLMs & MLOps** | CNNs, Vision Transformers, HuggingFace fine-tuning, RAG pipelines, MLflow, DVC, FastAPI, Docker | Scheduled |

---

## Deep Dive: Selected Mathematical Implementations

### 1. K-Means++ Initialization Probability Distribution
To avoid poor local minima from random initialization, initial centroids are sampled using distance-weighted probability:
$$P(x) = \frac{D(x)^2}{\sum_{x' \in X} D(x')^2}$$
where $D(x)$ is the shortest Euclidean distance from data point $x$ to the closest centroid already chosen.

### 2. AdamW Weight Decay Decoupling
Standard Adam couples $L_2$ regularization directly into the running gradient moments $\hat{m}_t$ and $\hat{v}_t$. **AdamW** restores true weight decay by decoupling the regularization step directly onto the parameter update:
$$\theta_t = \theta_{t-1} - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

### 3. Inverted Dropout Scaling
To preserve expected activation magnitude during inference without requiring post-hoc test scaling:
$$\tilde{a}^{(l)} = \frac{m^{(l)} \odot a^{(l)}}{1 - p}, \quad m^{(l)} \sim \text{Bernoulli}(1 - p)$$

---

## Featured Production Projects

### End-to-End Customer Churn Prediction System
* **Objective:** Predict telecom customer churn with high recall on imbalanced distributions.
* **Architecture:** Scikit-Learn `ColumnTransformer` pipelines, categorical encoders, hyperparameter tuning, model serialization, and clean modular evaluation.
* **Tech Stack:** Python, Pandas, Scikit-Learn, Joblib.

### Titanic Survival Predictor
* **Objective:** Comprehensive feature engineering and classification benchmark.
* **Key Highlights:** Title extraction, family size compounding, iterative imputation, and cross-validated ensemble models.

---

## Tech Stack & Tooling

- **Languages:** Python 3.10+, C++ (Algorithmic Foundations)
- **Scientific Computing:** NumPy, SciPy, Pandas
- **Machine Learning:** Scikit-Learn, XGBoost, LightGBM, CatBoost
- **Deep Learning:** PyTorch, Torchvision, Autograd
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Development & Versioning:** Git, GitHub, Jupyter Notebooks

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github-placeholder/Sahil-K-Y/AI-ML-Blueprint
cd AI-ML-Blueprint
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn torch torchvision jupyterlab
```

### 4. Launch JupyterLab
```bash
jupyter lab
```

---

## Author

**Sahil Kumar**  
*B.Tech CSE (Artificial Intelligence), DAV University, Jalandhar*  
- **GitHub:** Sahil-K-Y  
- **Specialization:** Machine Learning Systems, Deep Learning, and Algorithmic Problem Solving in C++

---

## License

This project is licensed under the MIT License - free for educational and personal use.
