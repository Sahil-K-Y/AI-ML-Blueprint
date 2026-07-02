import os
import shutil
import json

def create_notebook_template(title, description, subtopics, filename):
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    "\n",
                    f"**Description:** {description}\n",
                    "\n",
                    "**Subtopics to learn:**\n",
                    "".join([f"  - {sub}\n" for sub in subtopics]),
                    "\n",
                    "## 📖 Concept Notes\n",
                    "> Write your 3-5 key takeaways/notes here.\n",
                    "\n",
                    "## 💻 Code & Practice\n",
                    "Implement your exercises and tests in this section."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Set style\n",
                    "sns.set_theme(style=\"darkgrid\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🛠️ Build / Submit Task\n",
                    "Write your main implementation cells below."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Your code here\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ✅ Verification\n",
                    "Verify your results and metrics here."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Your verification code here\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

def main():
    notebooks_dir = r"d:\repos\AI-ML-Blueprint\Phase-3 - Logistic Regression Deep Dive\notebooks"
    archive_dir = os.path.join(notebooks_dir, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    
    # 1. Archive existing files
    for filename in os.listdir(notebooks_dir):
        file_path = os.path.join(notebooks_dir, filename)
        if os.path.isfile(file_path) and filename.endswith(".ipynb"):
            dest_path = os.path.join(archive_dir, filename)
            shutil.move(file_path, dest_path)
            print(f"Archived {filename}")
            
    # 2. Restore and rename the completed Day 32 (Sigmoid) notebook to Day 31
    completed_sigmoid_src = os.path.join(archive_dir, "Day_032_Sigmoid_Odds__LogOdds.ipynb")
    completed_sigmoid_dst = os.path.join(notebooks_dir, "Day_031_Sigmoid_Odds_LogLoss.ipynb")
    if os.path.exists(completed_sigmoid_src):
        shutil.copy(completed_sigmoid_src, completed_sigmoid_dst)
        print("Restored and renamed completed Sigmoid notebook to Day_031_Sigmoid_Odds_LogLoss.ipynb")
        
    # 3. Create new templates for Days 32 to 44
    new_days = {
        32: {
            "title": "📅 Day 032: Logistic Regression Implementation",
            "desc": "Classifier fit/predict, probability scores, coefficient interpretation.",
            "subs": ["Logistic Regression workflow", "StandardScaler before LR", "predict vs predict_proba"],
            "filename": "Day_032_Logistic_Regression_Implementation.ipynb"
        },
        33: {
            "title": "📅 Day 033: Classification Metrics I",
            "desc": "Core metrics: Accuracy, Precision, Recall, Confusion Matrix.",
            "subs": ["TP, TN, FP, FN", "Accuracy, precision, recall", "When accuracy is misleading"],
            "filename": "Day_033_Classification_Metrics_I.ipynb"
        },
        34: {
            "title": "📅 Day 034: Classification Metrics II",
            "desc": "F1 Score, specificity, and metric tradeoffs.",
            "subs": ["F1 score calculation", "Precision-Recall tradeoff", "Business impacts of FP vs FN"],
            "filename": "Day_034_Classification_Metrics_II.ipynb"
        },
        35: {
            "title": "📅 Day 035: ROC-AUC & PR-AUC Curves",
            "desc": "Threshold-independent evaluation curves.",
            "subs": ["ROC curve (TPR vs FPR)", "AUC score meaning", "PR curve (Precision vs Recall) for imbalanced data", "Threshold selection based on costs"],
            "filename": "Day_035_ROCAUC_PRAUC_Interpretation.ipynb"
        },
        36: {
            "title": "📅 Day 036: Multiclass Classification",
            "desc": "One-vs-Rest, One-vs-One, and Softmax.",
            "subs": ["OvR vs OvO strategies", "Softmax activation function", "Macro/Micro/Weighted averages"],
            "filename": "Day_036_Multiclass_Classification.ipynb"
        },
        37: {
            "title": "📅 Day 037: Regularization in Logistic Regression",
            "desc": "L1/L2 regularization and scaling parameters.",
            "subs": ["L1 vs L2 penalties", "C parameter controls", "Solver compatibility (liblinear, lbfgs)"],
            "filename": "Day_037_Regularization_in_Logistic_Regression.ipynb"
        },
        38: {
            "title": "📅 Day 038: Breast Cancer Classifier Mini Project",
            "desc": "Complete baseline implementation and pipeline finalization.",
            "subs": ["Pipeline consolidation", "Hyperparameter tuning", "GitHub/LinkedIn updates"],
            "filename": "Day_038_Breast_Cancer_Classifier_Project.ipynb"
        },
        39: {
            "title": "📅 Day 039: Decision Tree Intuition",
            "desc": "Tree splits, entropy, and Gini impurity.",
            "subs": ["Entropy and Gini formulas", "Information Gain", "Recursive partitioning"],
            "filename": "Day_039_Decision_Tree_Intuition.ipynb"
        },
        40: {
            "title": "📅 Day 040: Decision Tree Classification Code",
            "desc": "Training and evaluating DecisionTreeClassifier.",
            "subs": ["DecisionTreeClassifier API", "Feature importance metrics", "Train/evaluate splits"],
            "filename": "Day_040_Decision_Tree_Classification_Code.ipynb"
        },
        41: {
            "title": "📅 Day 041: Tree Overfitting & Pruning",
            "desc": "Regularizing decision trees to prevent overfitting.",
            "subs": ["max_depth, min_samples_split, min_samples_leaf", "Cost-complexity pruning", "Validation curves"],
            "filename": "Day_041_Decision_Tree_Pruning.ipynb"
        },
        42: {
            "title": "📅 Day 042: Decision Tree Visualization",
            "desc": "Exporting and plotting trained trees.",
            "subs": ["plot_tree & export_text", "Interpreting branch rules", "Visual storytelling for business stakeholders"],
            "filename": "Day_042_Decision_Tree_Visualization.ipynb"
        },
        43: {
            "title": "📅 Day 043: Decision Tree Regressor",
            "desc": "Regression trees and split criteria.",
            "subs": ["Regression splitting (MSE reduction)", "DecisionTreeRegressor API", "Comparison with Linear Regression"],
            "filename": "Day_043_Decision_Tree_Regressor.ipynb"
        },
        44: {
            "title": "📅 Day 044: Decision Tree Mini Project",
            "desc": "Building and evaluating a decision tree classifier.",
            "subs": ["Model setup and tuning", "Evaluation metrics", "Repository commit"],
            "filename": "Day_044_Decision_Tree_Project.ipynb"
        }
    }
    
    for day, info in new_days.items():
        dst_file = os.path.join(notebooks_dir, info["filename"])
        create_notebook_template(info["title"], info["desc"], info["subs"], dst_file)
        print(f"Created template for Day {day}: {info['filename']}")

if __name__ == "__main__":
    main()
