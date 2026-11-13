# Student Performance Analysis

> A data analysis and machine-learning project to explore factors associated with student academic performance and build an interpretable baseline model for final-grade estimation.

**Status:** Project specification / planned implementation  
**Repository:** [AI-ML-Blueprint](https://github.com/Sahil-K-Y/AI-ML-Blueprint)

---

## 1. Project overview

Student Performance Analysis studies how demographic, family, social, school, and study-related attributes relate to students' academic results. The project will use exploratory data analysis (EDA), statistical summaries, data visualization, and supervised machine learning to investigate patterns in the dataset.

The goal is to create a reproducible educational data-science project—not to label students, make admissions decisions, or treat model predictions as a measure of a student's ability or potential.

## 2. Problem statement

Academic performance can be associated with multiple interacting factors. A simple comparison of final grades does not explain these relationships, and a predictive model can be misleading if it uses information that would not have been available at prediction time.

This project will:

1. Explore grade distributions and relationships among available variables.
2. Identify patterns and associations in study habits, absences, family context, and school-related attributes where available.
3. Build and compare baseline regression models for the final grade.
4. Explain model behaviour and document limitations, bias risks, and data leakage controls.

## 3. Dataset

**Planned dataset:** UCI Machine Learning Repository — [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance).

The dataset contains student records from two Portuguese secondary-school courses:

- **student-mat.csv** — Mathematics course.
- **student-por.csv** — Portuguese language course.

It includes student, family, social, school, and academic attributes. The final course grade is represented by **G3**; **G1** and **G2** represent earlier-period grades.

Before analysis, the implementation must inspect the dataset documentation, column definitions, missing values, duplicates, categorical values, and grade distributions. Do not assume the two course files represent independent students; check the source documentation before combining them.

## 4. Analysis questions

The EDA will investigate questions such as:

- How are final grades distributed?
- How strongly are G1 and G2 associated with G3?
- How do study time, absences, and past failures relate to grades?
- What patterns appear across school, family, and support-related variables?
- Do relationships differ between the Mathematics and Portuguese datasets?
- Which variables appear useful for prediction, and which may introduce leakage or unfair proxies?

These are questions to investigate, not conclusions assumed in advance. Correlation and model feature importance do not establish causation.

## 5. Planned workflow

1. **Data understanding:** read the dataset documentation and inspect schema, types, ranges, missingness, and duplicates.
2. **Data cleaning:** validate values, handle missing data if present, and document every transformation.
3. **Exploratory analysis:** univariate, bivariate, and multivariate analysis with clear visualizations.
4. **Preprocessing:** encode categorical variables and scale numerical features where required, using pipelines fitted only on training data.
5. **Baseline modelling:** compare a simple mean/median baseline with Linear Regression and suitable tree-based regressors.
6. **Evaluation:** use a held-out test set and cross-validation on the training data.
7. **Interpretation:** examine coefficients or feature importance, and use a suitable explanation method where appropriate.
8. **Reporting:** summarize findings, model limitations, and reproducibility instructions.

## 6. Prediction target and leakage policy

The primary target is **G3 (final grade)**, treated as a regression target.

The prediction question must be stated precisely:

- **Early prediction:** exclude G1 and G2, because those grades would not yet be available.
- **Later-term prediction:** G1 and/or G2 may be included only if the project explicitly defines the prediction point as after those grades are known.

Do not compare these two settings as if they were the same task. Report which features were used in every experiment. All preprocessing and feature selection must be fitted within the training folds to prevent data leakage.

## 7. Models to compare

Start with simple, interpretable baselines:

- Dummy Regressor
- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor

Additional models should be introduced only when they add a meaningful comparison. Keep the test set separate until final evaluation.

## 8. Evaluation metrics

For regression, report:

- **MAE:** average absolute prediction error, in grade points.
- **RMSE:** gives greater weight to larger errors, in grade points.
- **R²:** proportion of target variance explained relative to a mean baseline.

Include cross-validation results, test-set results, and a comparison with the Dummy Regressor. Do not report accuracy for a regression task. If grade bands are later introduced as a separate classification experiment, define the bands before modelling and report classification metrics separately.

## 9. Planned repository structure

- **Student-Performance-Analysis/README.md** — project overview and methodology.
- **data/raw/** — source data; keep out of Git unless licensing permits.
- **data/processed/** — generated data; normally not committed.
- **notebooks/01_data_understanding.ipynb** — schema and data-quality checks.
- **notebooks/02_exploratory_data_analysis.ipynb** — EDA and visualizations.
- **notebooks/03_model_comparison.ipynb** — baseline modelling and evaluation.
- **src/preprocessing.py** — reusable preprocessing.
- **src/train.py** — model training.
- **src/evaluate.py** — evaluation and reporting.
- **reports/figures/** — generated figures.
- **requirements.txt** — dependencies.
- **.gitignore** — excludes local data, caches, and generated artifacts.

This is the proposed structure; files and results should be added as the implementation is completed. Do not commit private student records or generated model files unnecessarily.

## 10. Tools and libraries

- Python
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook / Google Colab

PyTorch is not required for the initial version. It can be added later as an optional neural-network comparison after establishing reliable classical ML baselines.

## 11. Responsible use and limitations

- The dataset is historical and limited to particular schools and courses; results may not generalize to other education systems.
- Observational associations do not prove that a factor causes a change in grades.
- Demographic and family variables may act as sensitive or unfair proxies. Their use must be justified and limitations discussed.
- Predictions should be treated as uncertain estimates, not labels of student ability.
- Do not use this project to rank, punish, admit, or exclude individual students.
- Follow the dataset's terms and cite the UCI dataset in any report or presentation.

## 12. Expected deliverables

- Reproducible data-cleaning and EDA notebooks.
- Clear visualizations with written interpretations.
- A leakage-aware baseline modelling comparison.
- Evaluation results using MAE, RMSE, and R².
- A concise report describing findings, limitations, and responsible-use considerations.

## 13. Reference

Dua, D. and Graff, C. (2019). **UCI Machine Learning Repository**. University of California, Irvine, School of Information and Computer Sciences. Student Performance dataset.

Dataset: https://archive.ics.uci.edu/dataset/320/student+performance
