# 🎯 The 100-Project Blueprint (July 1 — December 31, 2026)

This document contains a structured, chronologically-ordered catalog of **100 Machine Learning and AI engineering projects** from basic to advanced difficulty. They are designed to align with the daily learning topics of the master roadmap, ensuring a project-focused, "build-in-public" engineering profile by the end of December.

---

## 📈 Phase 3: Classification Foundations & Logistic Regression (July 1 – July 14)
*Focus: Project Architecture, Sigmoids, Odds/Log-Odds, Evaluation Metrics, and Titanic Classification.*

| # | Date | Project Title | Difficulty | Core Stack | Expected Deliverable |
|---|------|---------------|------------|------------|----------------------|
| 1 | 01 Jul | Basic Pipeline Verification Utility | Basic | `pytest`, `sklearn` | Preprocessing unit tests for data types and shape stability. |
| 2 | 03 Jul | Sigmoid Function Visualizer | Basic | `matplotlib`, `numpy` | Plot of odds ratios and log-odds mapped to sigmoid probabilities. |
| 3 | 05 Jul | Binary Classification on Iris | Basic | `pandas`, `sklearn` | Binary mapping classifier with custom decision thresholds. |
| 4 | 07 Jul | Log Loss Calculator from Scratch | Basic | `numpy` | Custom log loss formula compared against `sklearn.metrics`. |
| 5 | 09 Jul | Titanic Survival Predictor v1 | Basic | `sklearn`, `pandas` | Baseline logistic regression classifier using default thresholds. |
| 6 | 11 Jul | Multiclass Softmax Classifier | Basic | `sklearn` | Multi-category classifier trained on the Wine dataset. |
| 7 | 13 Jul | Metrics Calculator from Scratch | Intermediate | `numpy`, `pandas` | Custom precision, recall, and F1 calculations. |
| 8 | 14 Jul | ROC & Precision-Recall Profiler | Intermediate | `matplotlib` | Curves evaluating metric behavior on skewed target classes. |

---

## 🔍 Phase 4: Core Classifiers & Preprocessing (July 15 – July 30)
*Focus: Missingness Taxonomies, Advanced Imputers, Outliers, Naive Bayes, KNN, and Model Calibration.*

| # | Date | Project Title | Difficulty | Core Stack | Expected Deliverable |
|---|------|---------------|------------|------------|----------------------|
| 9 | 16 Jul | Missingness Taxonomy Profiler | Basic | `scipy.stats` | Statistical test suite identifying MCAR/MAR/MNAR null patterns. |
| 10 | 18 Jul | Advanced Imputation Benchmarker | Intermediate | `sklearn.impute` | KNNImputer vs IterativeImputer comparison on Boston/California datasets. |
| 11 | 20 Jul | Outlier Detector Suite | Intermediate | `sklearn.ensemble` | Isolation Forest & LOF comparison for anomaly cleaning. |
| 12 | 22 Jul | Custom Winsorization Transformer | Intermediate | `sklearn.base` | Custom sklearn-compatible transformer class to clip outliers. |
| 13 | 24 Jul | Cyclic Encodings for Temporal Data | Intermediate | `numpy`, `pandas` | Sine/cosine transformation pipeline for dates & weekdays. |
| 14 | 26 Jul | KNN Distance Metrics Analyzer | Basic | `sklearn.neighbors` | KNN classifier evaluating Euclidean vs Manhattan distance metrics. |
| 15 | 27 Jul | SMS Text Spam Classifier | Basic | `sklearn.naive_bayes` | Naive Bayes model using TF-IDF for SMS spam/ham classification. |
| 16 | 28 Jul | Heart Disease Classifier Preprocessor | Intermediate | `sklearn.compose` | Clean `ColumnTransformer` pipelines separating scaling and encoding. |
| 17 | 29 Jul | Heart Disease Model Optimizer | Intermediate | `sklearn.model_selection` | Hyperparameter optimization using Grid and Randomized search. |
| 18 | 30 Jul | Classifier Calibration Profiler | Advanced | `sklearn.calibration` | Reliability curves with Platt scaling and Isotonic regression calibration. |

---

## 🌲 Phase 5: Trees, Ensembles, SVM, Clustering & Time Series (July 31 – August 19)
*Focus: Decision Trees, Boosting, Stacking, Support Vector Machines, K-Means, DBSCAN, PCA, Prophet, and Recommenders.*

| # | Date | Project Title | Difficulty | Core Stack | Expected Deliverable |
|---|------|---------------|------------|------------|----------------------|
| 19 | 31 Jul | Entropy & Gini Split Calculator | Basic | `numpy` | Interactive manual split calculator mapping information gain. |
| 20 | 02 Aug | Decision Tree Complexity Visualizer | Basic | `dtreeviz`, `sklearn` | Decision boundary visualization mapping depth vs pruning. |
| 21 | 04 Aug | Random Forest Classifier | Basic | `sklearn.ensemble` | Breast cancer diagnosis classifier utilizing ensemble trees. |
| 22 | 06 Aug | OOB Error vs Validation Plotter | Intermediate | `sklearn.ensemble` | Dynamic tracker mapping Out-of-Bag error trends during training. |
| 23 | 08 Aug | AdaBoost Classifier from Scratch | Intermediate | `numpy` | Custom training loop updating sample weights and decision stumps. |
| 24 | 10 Aug | Gradient Boosting Classifier | Intermediate | `sklearn.ensemble` | Custom loss gradient boosting classifier. |
| 25 | 11 Aug | XGBoost vs LightGBM Benchmarker | Advanced | `xgboost`, `lightgbm` | Execution speed and metric profiling on a large dataset. |
| 26 | 12 Aug | CatBoost Optuna Optimizer | Advanced | `optuna`, `catboost` | Automated hyperparameter tuning dashboard using Optuna. |
| 27 | 13 Aug | Heterogeneous Stacking Ensemble | Advanced | `sklearn.ensemble` | Stacking classifier using RF, LogReg, and SVM meta-learners. |
| 28 | 14 Aug | SVM Kernel Decision Boundary Plotter | Intermediate | `sklearn.svc` | Visual plots comparing Linear, Polynomial, and RBF kernels. |
| 29 | 15 Aug | K-Means Silhouette Optimizer | Intermediate | `sklearn.cluster` | Elbow and Silhouette plots for optimal cluster evaluation. |
| 30 | 16 Aug | DBSCAN Customer Segmenter | Intermediate | `sklearn.cluster` | Density-based clustering profile identifying outlier segments. |
| 31 | 17 Aug | Dimensionality Reducer (PCA) | Intermediate | `sklearn.decomposition` | Principal components variance analyzer and coordinate projection. |
| 32 | 18 Aug | Prophet vs ARIMA Sales Forecaster | Advanced | `prophet`, `statsmodels` | Forecast overlay mapping monthly metrics and seasonal trends. |
| 33 | 19 Aug | Content-Based Movie Recommender | Intermediate | `sklearn`, `pandas` | Recommendation engine matching titles via Cosine Similarity. |

---

## ⚙️ Phase 6: Production Scaffolding & APIs (August 20 – August 30)
*Focus: Model Serialization, FastAPI, Pydantic validation, Pytest integration, Docker, Docker Compose, and async predictions.*

| # | Date | Project Title | Difficulty | Core Stack | Expected Deliverable |
|---|------|---------------|------------|------------|----------------------|
| 34 | 21 Aug | FastAPI Model Predictor API | Basic | `fastapi`, `joblib` | REST endpoint loading serialized model and returning predictions. |
| 35 | 22 Aug | Pydantic Request Validator | Basic | `pydantic` | Input schemas validating type constraints and null values. |
| 36 | 24 Aug | ML API Exception Handler | Intermediate | `fastapi` | JSON error returns for empty fields or scaling out-of-bounds. |
| 37 | 26 Aug | FastAPI Pytest Tester | Intermediate | `pytest`, `fastapi` | Route tests using Client API. |
| 38 | 27 Aug | Multi-Model API Router | Intermediate | `fastapi` | Structured routers separating regression and classification routes. |
| 39 | 28 Aug | Dockerized Inference Container | Intermediate | `docker` | Single-stage Dockerfile packaging API, dependencies, and model. |
| 40 | 29 Aug | Multi-Container Compose Setup | Advanced | `docker-compose` | Docker Compose file orchestration between API and Redis caching. |
| 41 | 30 Aug | Async Predict Queue Server | Advanced | `celery`, `redis` | Async queue system processing background inference batches. |

---

## 🧠 Phase 7: Deep Learning Foundations (August 31 – September 14)
*Focus: Perceptrons, Autograd, Keras Sequential & Functional APIs, and PyTorch Training Loops.*

| # | Date | Project Title | Difficulty | Core Stack | Expected Deliverable |
|---|------|---------------|------------|------------|----------------------|
| 42 | 31 Aug | Perceptron from Scratch | Basic | `numpy` | Forward/backward pass mapping logical gates (AND/OR). |
| 43 | 02 Sep | Activation Functions Derivative Sandbox | Basic | `numpy`, `matplotlib` | Visual representation of vanishing/exploding gradients. |
| 44 | 04 Sep | Grad Descent Loss Surface Plotter | Intermediate | `numpy`, `matplotlib` | Cost path map comparing SGD vs Batch Gradient Descent. |
| 45 | 06 Sep | Keras Sequential MLP on MNIST | Intermediate | `keras`, `tensorflow` | Tabular neural network reaching >95% test accuracy. |
| 46 | 08 Sep | Multi-Input Keras Functional Network | Intermediate | `keras` | Model combining continuous and categorical input paths. |
| 47 | 09 Sep | PyTorch Autograd Profiler | Basic | `torch` | Calculation of gradients on custom equation graphs. |
| 48 | 11 Sep | PyTorch Custom Tabular Dataset | Intermediate | `torch.utils.data` | Custom DataLoader handling batch transformations. |
| 49 | 12 Sep | PyTorch MLP Training Loop | Intermediate | `torch` | Clean training loop with zero_grad, backward, and step calls. |
| 50 | 13 Sep | PyTorch Regularization Suite | Intermediate | `torch` | Comparison of L2 weight decay vs Dropout rates. |
| 51 | 14 Sep | PyTorch Custom Early Stopper | Advanced | `torch` | Early stopping callback based on validation loss thresholds. |

---

## 👁️ Phase 8: Computer Vision & Generative AI (September 15 – September 30)
*Focus: CNN Architectures, Augmentations, Transfer Learning, VAEs, and Stable Diffusion img2img pipelines.*

| # | Date | Project Title | Difficulty | Core Stack | Expected Deliverable |
|---|------|---------------|------------|------------|----------------------|
| 52 | 16 Sep | PyTorch Custom CNN Classifier | Basic | `torch`, `torchvision` | Convolutional model with max pooling on FashionMNIST. |
| 53 | 18 Sep | Augmentation Impact Benchmarker | Basic | `torchvision` | Metrics comparison comparing augmented vs raw images. |
| 54 | 20 Sep | ResNet50 Transfer Learning Engine | Intermediate | `torchvision.models` | Fine-tuned ResNet model on a small custom dataset. |
| 55 | 22 Sep | Grad-CAM Heatmap overlay | Advanced | `torch`, `opencv` | Visual heatmap highlighting CNN classification regions. |
| 56 | 24 Sep | Image Reconstruction Autoencoder | Intermediate | `torch` | Autoencoder compressing images into latent dimensions. |
| 57 | 26 Sep | VAE Digit Generator | Advanced | `torch` | Variational Autoencoder plotting latent space digit grids. |
| 58 | 27 Sep | Stable Diffusion Inference Client | Intermediate | `diffusers`, `torch` | Text-to-image generator using pre-trained weights. |
| 59 | 28 Sep | Image-to-Image (img2img) Client | Intermediate | `diffusers` | Image generator modifying seed images based on prompts. |
| 60 | 29 Sep | CNN Architecture search | Advanced | `optuna`, `torch` | Automated search optimizing filters, kernel sizes, and learning rates. |
| 61 | 30 Sep | Object Detection API wrapper | Advanced | `yolov8`, `fastapi` | API returning detected bounding boxes on image inputs. |

---

## 🔤 Phase 9: Natural Language Processing (October 1 – October 12)
*Focus: Regex text cleaning, Tokenization, Bag of Words, TF-IDF, Word2Vec, spaCy pipelines, NER, and Sentiment analysis.*

| # | Date | Project Title | Difficulty | Core Stack | Expected Deliverable |
|---|------|---------------|------------|------------|----------------------|
| 62 | 02 Oct | Regex Text Cleaning utility | Basic | `re`, `nltk` | Clean pipeline handling HTML tags, URLs, and lowercasing. |
| 63 | 04 Oct | Bag of Words vs TF-IDF Benchmarker | Basic | `sklearn.feature_extraction` | Vector comparison evaluating sentiment performance. |
| 64 | 06 Oct | Word2Vec Semantic Math Client | Intermediate | `gensim` | Semantic calculations mapping relationships (e.g. King - Man + Woman = Queen). |
| 65 | 08 Oct | spaCy Custom Component Component | Intermediate | `spacy` | Custom component processing specific text entity triggers. |
| 66 | 09 Oct | spaCy Named Entity Recognition (NER) | Intermediate | `spacy` | NER model trained on clinical or resume datasets. |
| 67 | 10 Oct | Multi-Engine Sentiment Profiler | Intermediate | `nltk.vader`, `textblob` | Sentiment score analyzer for customer reviews. |
| 68 | 11 Oct | Email Spam Naive Bayes Classifier | Intermediate | `sklearn` | Clean email spam classifier using vectorizers and Naive Bayes. |
| 69 | 12 Oct | Cosine Similarity Text Search | Intermediate | `sklearn`, `pandas` | Sentence search engine ranking relevance scores. |

---

## 🤖 Phase 10: Transformers, LLMs, RAG & Agents (October 13 – October 27)
*Focus: Attention Mechanisms, Hugging Face, BERT Fine-tuning, RAG Vector Stores, RAGAS evaluations, AI Agents, and Reinforcement Learning.*

| # | Date | Project Title | Difficulty | Core Stack | Expected Deliverable |
|---|------|---------------|------------|------------|----------------------|
| 70 | 14 Oct | Self-Attention Heatmap Simulator | Basic | `numpy`, `seaborn` | Visual heatmap mapping token relationships in text queries. |
| 71 | 16 Oct | Zero-Shot Classifier Client | Basic | `transformers` | Hugging Face pipeline categorizing text into dynamic label sets. |
| 72 | 18 Oct | BERT Fine-Tuner for Sentiment Analysis | Intermediate | `transformers`, `torch` | Custom BERT classification model trained using Trainer API. |
| 73 | 20 Oct | BERT Model Server | Intermediate | `fastapi`, `transformers` | Endpoint serving fine-tuned BERT models. |
| 74 | 22 Oct | Document Vector Indexer (RAG 1) | Intermediate | `chromadb`, `langchain` | File parser chunking text and storing vector embeddings. |
| 75 | 24 Oct | RAG Chat System (RAG 2) | Advanced | `ollama`, `langchain` | QA system querying contextual document references. |
| 76 | 25 Oct | RAG Evaluation Evaluator | Advanced | `deepeval`, `ragas` | Automated test suite validating context recall and faithfulness. |
| 77 | 26 Oct | Custom ReAct Framework AI Agent | Advanced | `python` | Tool-using AI agent coordinating math and web search. |
| 78 | 27 Oct | Gymnasium Environment Runner | Basic | `gymnasium` | Env runner testing baseline cartpole performance. |
| 79 | 27 Oct | Q-Learning FrozenLake Solver | Advanced | `gymnasium`, `numpy` | Agent training with tabular updates solving FrozenLake. |

---

## 💻 Phases 11 & 12: React Frontends & Full-Stack Integration (October 28 – November 21)
*Focus: React Forms, Axios integration, Protected Routes, JSON Web Tokens (JWT), and MongoDB integration.*

| # | Date | Project Title | Difficulty | Core Stack | Expected Deliverable |
|---|------|---------------|------------|------------|----------------------|
| 80 | 29 Oct | Vanilla DOM Form Handler | Basic | `html`, `javascript` | Form sending input parameters to ML APIs. |
| 81 | 01 Nov | React Prediction Form Component | Basic | `react` | Form inputs with active range validation. |
| 82 | 04 Nov | Axios API Client Module | Intermediate | `react`, `axios` | Frontend handler for loader displays and error states. |
| 83 | 07 Nov | React Analytics Dashboard | Intermediate | `react`, `chart.js` | Visual dashboard representing prediction metrics. |
| 84 | 10 Nov | Secure JWT Token Generator | Intermediate | `fastapi`, `jose` | Secure credential validator issuing token access. |
| 85 | 13 Nov | React Protected Route | Intermediate | `react-router` | Route guard redirecting unauthenticated users. |
| 86 | 15 Nov | MongoDB Prediction History Store | Intermediate | `pymongo`, `mongodb` | Database logging model features and outputs. |
| 87 | 17 Nov | Full-Stack Sentiment Dashboard | Advanced | `react`, `fastapi`, `mongodb` | Integrated portal tracking real-time text analysis. |
| 88 | 19 Nov | Database-backed Chat Interface | Advanced | `react`, `fastapi`, `mongodb` | RAG interface saving message history. |
| 89 | 21 Nov | Tailwind CSS Model Hub | Advanced | `react`, `tailwind` | Central visual dashboard for all deployed models. |

---

## 🚀 Phases 13 & 14: MLOps Pipelines & Flagship Capstone (November 22 – December 17)
*Focus: DVC, Feast, MLflow, BentoML serving, Kubernetes, Github Actions, SHAP, Evidently, and Capstone packaging.*

| # | Date | Project Title | Difficulty | Core Stack | Expected Deliverable |
|---|------|---------------|------------|------------|----------------------|
| 90 | 24 Nov | DVC Stage Tracker Pipeline | Basic | `dvc` | Version-controlled pipeline tracking stages and parameters. |
| 91 | 26 Nov | Feast Feature Store | Intermediate | `feast` | Offline/online store managing training features. |
| 92 | 28 Nov | MLflow Experiment Tracker | Intermediate | `mlflow` | Run log database tracking hyperparameters, metrics, and models. |
| 93 | 30 Nov | BentoML Service Service | Intermediate | `bentoml` | Self-contained prediction package running REST/gRPC. |
| 94 | 02 Dec | Minikube deployment manifests | Advanced | `kubernetes`, `docker` | Deployment and service configurations running serving pods. |
| 95 | 04 Dec | CI/CD GitHub Actions Test Suite | Advanced | `github-actions`, `pytest` | Workflow validating code and models on repository pushes. |
| 96 | 06 Dec | SHAP Feature Explainability | Intermediate | `shap` | Visual charts explaining local and global model predictions. |
| 97 | 08 Dec | Evidently Model Drift Monitor | Advanced | `evidently` | Automated checks logging dataset and classification drift. |
| 98 | 11 Dec | Capstone Phase 1: Pipeline Scaffold | Advanced | `sklearn`, `pandas` | Core preprocessing pipeline with dataset audits. |
| 99 | 14 Dec | Capstone Phase 2: Model Stack | Advanced | `optuna`, `xgboost`, `shap` | Tuned model ensemble with explainability charts. |
| 100 | 17 Dec | Capstone Phase 3: Full Stack Compose | Advanced | `docker-compose`, `react` | Multi-container setup running backend, frontend, and DB. |

---

## 🎓 Phases 15 & 16: Job Readiness & Interview Drills (December 18 – December 31)
*Focus: Portfolio optimization, conceptual drills, and final review.*

The remaining days from **December 18 to December 31** are reserved for compiling these 100 projects into a centralized portfolio website, polishing GitHub repository documentation, writing final case studies, and running conceptual ML/DL/MLOps mock interview drills.
