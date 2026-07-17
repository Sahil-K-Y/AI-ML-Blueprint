import json
import os

def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": cells
    }

def md(source):
    return {"cell_type": "markdown", "id": f"md_{hash(source)}", "metadata": {}, "source": [line + "\\n" for line in source.split("\\n")]}

def code(source):
    return {"cell_type": "code", "execution_count": None, "id": f"code_{hash(source)}",
            "metadata": {}, "outputs": [], "source": [line + "\\n" for line in source.split("\\n")]}

def write_nb(filename, cells):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb(cells), f, indent=1, ensure_ascii=False)
    print(f"Generated: {filename}")

# DAY 32
d32 = [
    md("# Day 32: Model Leaderboard\\n\\n**Phase 3: Tree Models & SVM**\\n\\nAaj hum ek Model Leaderboard banayenge. Iska matlab hai hum ek hi dataset par multiple ML algorithms train karenge aur compare karenge ki kaunsa best perform karta hai.\\n\\n### Topics Covered:\\n- Model Comparison Pipeline\\n- Cross-validation for Unbiased Accuracy\\n- Leaderboard Visualizations"),
    md("## Importing Libraries\\nSabse pehle saari required libraries import karte hain."),
    code("import pandas as pd\\nimport numpy as np\\nimport matplotlib.pyplot as plt\\nimport seaborn as sns"),
    md("## Loading Dataset\\nHum Red Wine Quality dataset use karenge."),
    code("url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'\\ndf = pd.read_csv(url, sep=';')"),
    md("## Viewing Data\\nDataset ka structure dekhte hain."),
    code("df.head()"),
    md("## Data Preprocessing\\nHum problem ko binary classification mein convert karenge:\\nQuality >= 7 is 'High Quality' (1), warna (0)."),
    code("df['high_quality'] = (df['quality'] >= 7).astype(int)"),
    md("Features aur Target alag karte hain."),
    code("X = df.drop(['quality', 'high_quality'], axis=1)\\ny = df['high_quality']"),
    md("## Train-Test Split"),
    code("from sklearn.model_selection import train_test_split\\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)"),
    md("## Feature Scaling\\nDistance-based models (jaise KNN, SVM) ke liye scaling zaroori hoti hai."),
    code("from sklearn.preprocessing import StandardScaler\\nscaler = StandardScaler()\\nX_train_sc = scaler.fit_transform(X_train)\\nX_test_sc = scaler.transform(X_test)"),
    md("## Initializing Models\\nHum saare models initialize karte hain."),
    code("from sklearn.linear_model import LogisticRegression\\nfrom sklearn.tree import DecisionTreeClassifier\\nfrom sklearn.svm import SVC\\nfrom sklearn.naive_bayes import GaussianNB\\nfrom sklearn.neighbors import KNeighborsClassifier\\nfrom sklearn.ensemble import RandomForestClassifier"),
    code("models = {\\n    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),\\n    'Decision Tree': DecisionTreeClassifier(random_state=42),\\n    'SVM': SVC(kernel='rbf', random_state=42),\\n    'Naive Bayes': GaussianNB(),\\n    'KNN': KNeighborsClassifier(n_neighbors=5),\\n    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)\\n}"),
    md("## Cross-Validation and Building Leaderboard\\nHar model ka 5-fold cross-validation accuracy calculate karte hain."),
    code("from sklearn.model_selection import cross_val_score"),
    code("results = []\\nfor name, model in models.items():\\n    scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='accuracy')\\n    results.append({'Model': name, 'CV Mean': scores.mean(), 'CV Std': scores.std()})"),
    md("Leaderboard Dataframe banate hain."),
    code("leaderboard = pd.DataFrame(results).sort_values('CV Mean', ascending=False).reset_index(drop=True)\\nleaderboard"),
    md("## Plotting the Leaderboard"),
    code("plt.figure(figsize=(10, 6))\\nsns.barplot(x='CV Mean', y='Model', data=leaderboard, xerr=leaderboard['CV Std'], palette='Set2')\\nplt.title('Model Leaderboard - Wine Quality')\\nplt.xlabel('CV Accuracy (5-fold)')\\nplt.show()"),
    md("## Final Evaluation on Test Set\\nBest model ko final test set pe check karte hain."),
    code("best_name = leaderboard.iloc[0]['Model']\\nbest_model = models[best_name]\\nbest_model.fit(X_train_sc, y_train)"),
    code("from sklearn.metrics import accuracy_score\\ny_pred = best_model.predict(X_test_sc)\\nprint(f'Final Test Accuracy for {best_name}: {accuracy_score(y_test, y_pred):.4f}')")
]
write_nb("Day 32 - Model Leaderboard.ipynb", d32)

# DAY 33
d33 = [
    md("# Day 33: Classification Project\\n\\n**Phase 3: Tree Models & SVM**\\n\\nAaj hum ek End-to-End Classification Project karenge taaki Pipeline aur Real-world workflow samajh aaye.\\n\\n### Topics Covered:\\n- End-to-End Classification Workflow\\n- Exploratory Data Analysis (EDA)\\n- Scikit-Learn Pipelines\\n- GridSearchCV for SVM"),
    md("## Importing Libraries"),
    code("import pandas as pd\\nimport numpy as np\\nimport matplotlib.pyplot as plt\\nimport seaborn as sns"),
    md("## Loading Dataset\\nBreast Cancer dataset use karenge classification ke liye."),
    code("from sklearn.datasets import load_breast_cancer\\ncancer = load_breast_cancer()\\ndf = pd.DataFrame(cancer.data, columns=cancer.feature_names)\\ndf['target'] = cancer.target"),
    md("## Viewing Data"),
    code("df.head()"),
    md("## Exploratory Data Analysis (EDA)\\nDekhte hain kis feature ka target se highest correlation hai."),
    code("corr = df.corr()\\ntop_features = corr.index[abs(corr['target']) > 0.6]"),
    md("Heatmap banate hain in top features ka."),
    code("plt.figure(figsize=(10, 8))\\nsns.heatmap(df[top_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')\\nplt.title('Correlation Heatmap')\\nplt.show()"),
    md("## Train-Test Split"),
    code("from sklearn.model_selection import train_test_split\\nX = df.drop('target', axis=1)\\ny = df['target']\\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)"),
    md("## Building a Pipeline\\nData scaling aur model training ko ek single pipeline mein wrap karte hain."),
    code("from sklearn.pipeline import Pipeline\\nfrom sklearn.preprocessing import StandardScaler\\nfrom sklearn.svm import SVC"),
    code("pipeline = Pipeline([\\n    ('scaler', StandardScaler()),\\n    ('svm', SVC(probability=True, random_state=42))\\n])"),
    md("## Hyperparameter Tuning with GridSearchCV\\nPipeline ke parameters tune karenge."),
    code("from sklearn.model_selection import GridSearchCV"),
    md("Parameters define karte hain (note `svm__` prefix kyunki ye pipeline ka step name hai)."),
    code("param_grid = {\\n    'svm__C': [0.1, 1, 10, 100],\\n    'svm__gamma': ['scale', 0.01, 0.1, 1],\\n    'svm__kernel': ['rbf', 'linear']\\n}"),
    md("GridSearch setup and training."),
    code("grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1, verbose=1)\\ngrid.fit(X_train, y_train)"),
    md("Best parameters."),
    code("print('Best Parameters:', grid.best_params_)"),
    md("## Final Evaluation\\nTest set par predictions check karte hain."),
    code("from sklearn.metrics import classification_report, confusion_matrix\\nbest_model = grid.best_estimator_\\ny_pred = best_model.predict(X_test)"),
    md("Classification Report:"),
    code("print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))"),
    md("Confusion Matrix:"),
    code("cm = confusion_matrix(y_test, y_pred)\\nsns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Malignant','Benign'], yticklabels=['Malignant','Benign'])\\nplt.title('Confusion Matrix')\\nplt.xlabel('Predicted')\\nplt.ylabel('Actual')\\nplt.show()")
]
write_nb("Day 33 - Classification Project.ipynb", d33)

# DAY 34
d34 = [
    md("# Day 34: Voting & Bagging Ensembles\\n\\n**Phase 3: Tree Models & SVM**\\n\\nAaj hum Ensemble Learning algorithms implement karenge, specifically **Voting Classifier** aur **Bagging Classifier**.\\n\\n### Topics Covered:\\n- Hard Voting vs Soft Voting\\n- Bagging with Decision Trees\\n- Out-of-Bag (OOB) Evaluation"),
    md("## Importing Libraries"),
    code("import pandas as pd\\nimport numpy as np\\nimport matplotlib.pyplot as plt\\nimport seaborn as sns"),
    md("## Loading Toy Dataset\\nHum `make_moons` dataset use karenge kyunki ye non-linear ensembles ko test karne ke liye perfect hai."),
    code("from sklearn.datasets import make_moons\\nX, y = make_moons(n_samples=500, noise=0.30, random_state=42)"),
    md("## Train-Test Split"),
    code("from sklearn.model_selection import train_test_split\\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"),
    md("## 1. Voting Classifier\\nHum 3 different models train karenge aur Voting Classifier se combine karenge."),
    code("from sklearn.ensemble import VotingClassifier\\nfrom sklearn.linear_model import LogisticRegression\\nfrom sklearn.ensemble import RandomForestClassifier\\nfrom sklearn.svm import SVC\\nfrom sklearn.metrics import accuracy_score"),
    md("Models initialize karte hain."),
    code("log_clf = LogisticRegression(random_state=42)\\nrnd_clf = RandomForestClassifier(random_state=42)\\nsvm_clf = SVC(random_state=42, probability=True)"),
    md("### Hard Voting\\nHard voting mein model majority class ko choose karta hai."),
    code("voting_hard = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')"),
    code("voting_hard.fit(X_train, y_train)\\ny_pred_hard = voting_hard.predict(X_test)\\nprint('Hard Voting Accuracy:', accuracy_score(y_test, y_pred_hard))"),
    md("### Soft Voting\\nSoft voting har class ki average probability ke basis par predict karta hai, jo generally better results deta hai."),
    code("voting_soft = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='soft')"),
    code("voting_soft.fit(X_train, y_train)\\ny_pred_soft = voting_soft.predict(X_test)\\nprint('Soft Voting Accuracy:', accuracy_score(y_test, y_pred_soft))"),
    md("## 2. Bagging Classifier\\nBagging mein hum same model (jaise Decision Tree) ko multiple baar train karte hain on different subsets of training data (bootstrap sampling)."),
    code("from sklearn.ensemble import BaggingClassifier\\nfrom sklearn.tree import DecisionTreeClassifier"),
    md("500 Decision Trees ka ek ensemble banate hain."),
    code("bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=500, max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)"),
    code("bag_clf.fit(X_train, y_train)\\ny_pred_bag = bag_clf.predict(X_test)\\nprint('Bagging Accuracy:', accuracy_score(y_test, y_pred_bag))"),
    md("## Out-of-Bag (OOB) Evaluation\\nBagging mein jo ~37% data sampling se bach jata hai, usko validation ke liye use kiya ja sakta hai (OOB Score)."),
    code("bag_oob = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=500, bootstrap=True, oob_score=True, random_state=42)"),
    code("bag_oob.fit(X_train, y_train)\\nprint('OOB Score:', bag_oob.oob_score_)\\nprint('Actual Test Accuracy:', accuracy_score(y_test, bag_oob.predict(X_test)))")
]
write_nb("Day 34 - Voting Bagging Ensembles.ipynb", d34)

# DAY 35
d35 = [
    md("# Day 35: Random Forest Basics\\n\\n**Phase 3: Tree Models & SVM**\\n\\nRandom Forest ek ensemble method hai jo multiple Decision Trees ka use karke variance ko reduce karta hai. Ye essentially Bagging + Feature Randomness hai.\\n\\n### Topics Covered:\\n- Training a Random Forest\\n- Comparing Random Forest with BaggingClassifier\\n- Extra-Trees Classifier"),
    md("## Importing Libraries"),
    code("import pandas as pd\\nimport numpy as np\\nimport matplotlib.pyplot as plt\\nimport seaborn as sns"),
    md("## Loading Dataset"),
    code("from sklearn.datasets import make_moons\\nX, y = make_moons(n_samples=500, noise=0.30, random_state=42)"),
    md("## Train-Test Split"),
    code("from sklearn.model_selection import train_test_split\\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"),
    md("## Training Random Forest Classifier\\nScikit-Learn ka `RandomForestClassifier` bagging aur feature selection ko automatically optimize karta hai."),
    code("from sklearn.ensemble import RandomForestClassifier\\nfrom sklearn.metrics import accuracy_score"),
    code("rf_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)\\nrf_clf.fit(X_train, y_train)"),
    md("Predicting on test set."),
    code("y_pred_rf = rf_clf.predict(X_test)\\nprint('Random Forest Accuracy:', accuracy_score(y_test, y_pred_rf))"),
    md("## Random Forest vs BaggingClassifier\\nTheoretically, ek Random Forest wahi hai jo ek `BaggingClassifier` hai, jisme base estimator Decision Tree ho jisme max_features ko limit kiya gaya ho.\\nChaliye dono ko compare karte hain."),
    code("from sklearn.ensemble import BaggingClassifier\\nfrom sklearn.tree import DecisionTreeClassifier"),
    code("bag_clf = BaggingClassifier(DecisionTreeClassifier(splitter='best', max_features='sqrt', max_leaf_nodes=16, random_state=42), n_estimators=500, random_state=42, n_jobs=-1)\\nbag_clf.fit(X_train, y_train)"),
    md("Dono ke predictions compare karte hain."),
    code("y_pred_bag = bag_clf.predict(X_test)\\nsame_predictions = np.sum(y_pred_bag == y_pred_rf)\\ntotal_predictions = len(y_pred_rf)\\nprint(f'Predictions identical: {same_predictions}/{total_predictions}')"),
    md("## Extra-Trees Classifier\\nEk step aage badhte hue, Extra-Trees (Extremely Randomized Trees) features ke liye random thresholds generate karte hain bajaye iske ki best threshold dhundha jaye. Isse model ki training aur fast ho jati hai aur variance further kam hota hai."),
    code("from sklearn.ensemble import ExtraTreesClassifier"),
    code("extra_clf = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)\\nextra_clf.fit(X_train, y_train)"),
    md("Evaluating Extra-Trees."),
    code("y_pred_extra = extra_clf.predict(X_test)\\nprint('Extra-Trees Accuracy:', accuracy_score(y_test, y_pred_extra))")
]
write_nb("Day 35 - Random Forest Basics.ipynb", d35)

# DAY 36
d36 = [
    md("# Day 36: Random Forest Hyperparameter Tuning\\n\\n**Phase 3: Tree Models & SVM**\\n\\nAaj hum Random Forest model ki performance ko improve karne ke liye uske hyperparameters ko tune karna seekhenge.\\n\\n### Topics Covered:\\n- Understanding Key Hyperparameters (`n_estimators`, `max_depth`, `max_features`)\\n- GridSearchCV for exhaustive tuning\\n- RandomizedSearchCV for faster tuning\\n- Comparing Baseline vs Tuned Models"),
    md("## Importing Libraries\\nSabse pehle hum zaroori libraries import karenge. Pandas data manipulation ke liye aur Seaborn datasets load karne ke liye."),
    code("import pandas as pd\\nimport numpy as np\\nimport matplotlib.pyplot as plt\\nimport seaborn as sns"),
    md("## Loading Dataset\\nHum Scikit-Learn ka breast cancer dataset use karenge classification ke liye."),
    code("from sklearn.datasets import load_breast_cancer\\ndata = load_breast_cancer()\\ndf = pd.DataFrame(data.data, columns=data.feature_names)\\ndf['target'] = data.target"),
    md("## Viewing Data\\nEk baar dataset ka structure check kar lete hain."),
    code("df.head()"),
    md("## Train-Test Split\\nData ko training aur testing sets mein divide karte hain (80% train, 20% test)."),
    code("from sklearn.model_selection import train_test_split\\nX = df.drop(columns=['target'])\\ny = df['target']\\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)"),
    md("## Baseline Model Training\\nEk default Random Forest model banate hain bina kisi tuning ke taaki hum baad mein compare kar sakein ki tuning se kitna fayda hua."),
    code("from sklearn.ensemble import RandomForestClassifier\\nfrom sklearn.metrics import accuracy_score\\nrf_baseline = RandomForestClassifier(random_state=42)\\nrf_baseline.fit(X_train, y_train)"),
    md("## Baseline Evaluation\\nBaseline model ki accuracy check karte hain test set par."),
    code("y_pred_baseline = rf_baseline.predict(X_test)\\nprint('Baseline Accuracy:', accuracy_score(y_test, y_pred_baseline))"),
    md("## Hyperparameter Tuning with GridSearchCV\\nGridSearchCV saare diye gaye hyperparameter combinations ko exhaustively test karta hai aur best combination nikalta hai."),
    code("from sklearn.model_selection import GridSearchCV"),
    md("Grid define karte hain:\\n- `n_estimators`: Number of trees\\n- `max_depth`: Depth of each tree\\n- `min_samples_split`: Minimum samples required to split a node"),
    code("param_grid = {\\n    'n_estimators': [50, 100, 200],\\n    'max_depth': [None, 10, 20],\\n    'min_samples_split': [2, 5]\\n}"),
    md("GridSearchCV object initialize karte hain 5-fold cross-validation ke sath."),
    code("grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)"),
    md("GridSearch ko train karte hain. Ye process thoda time le sakta hai kyunki ye bahut saare models train karta hai."),
    code("grid_search.fit(X_train, y_train)"),
    md("## Best Parameters\\nGridSearch ne kaunse parameters best find kiye?"),
    code("print('Best Parameters:', grid_search.best_params_)"),
    md("## Evaluating the Tuned Model\\nGridSearch ke `best_estimator_` ko use karke predictions nikalte hain aur accuracy dekhte hain."),
    code("best_rf = grid_search.best_estimator_\\ny_pred_tuned = best_rf.predict(X_test)\\nprint('Tuned Model Accuracy:', accuracy_score(y_test, y_pred_tuned))"),
    md("## RandomizedSearchCV\\nGridSearchCV bahut slow ho sakta hai agar parameters zyada hon. RandomizedSearchCV random combinations pick karta hai, isliye faster hai."),
    code("from sklearn.model_selection import RandomizedSearchCV\\nfrom scipy.stats import randint"),
    md("Distributions define karte hain jahan se random values pick hongi."),
    code("param_dist = {\\n    'n_estimators': randint(50, 300),\\n    'max_depth': [None, 10, 20, 30],\\n    'max_features': ['sqrt', 'log2']\\n}"),
    md("RandomizedSearchCV initialize karke 10 random combinations test karenge."),
    code("random_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42), param_distributions=param_dist, n_iter=10, cv=5, random_state=42, n_jobs=-1)"),
    md("Training the RandomizedSearch."),
    code("random_search.fit(X_train, y_train)"),
    md("Best parameters from RandomizedSearch."),
    code("print('Best Parameters (Randomized):', random_search.best_params_)"),
    md("Evaluating RandomizedSearch model."),
    code("y_pred_rand = random_search.best_estimator_.predict(X_test)\\nprint('Randomized Tuned Accuracy:', accuracy_score(y_test, y_pred_rand))")
]
write_nb("Day 36 - Random Forest Tuning.ipynb", d36)

# DAY 37
d37 = [
    md("# Day 37: Feature Importance\\n\\n**Phase 3: Tree Models & SVM**\\n\\nRandom Forest ka ek sabse bada advantage uski explainability hai. Aaj hum seekhenge ki model predictions mein kaunse features sabse zyada important role play karte hain.\\n\\n### Topics Covered:\\n- Extracting Feature Importances\\n- Visualizing Feature Importances with Bar Plots\\n- Understanding the business value of Feature Importance"),
    md("## Importing Libraries"),
    code("import pandas as pd\\nimport numpy as np\\nimport matplotlib.pyplot as plt\\nimport seaborn as sns"),
    md("## Loading Dataset\\nHum Iris dataset load karenge."),
    code("from sklearn.datasets import load_iris\\niris = load_iris()\\ndf = pd.DataFrame(iris.data, columns=iris.feature_names)\\ndf['species'] = iris.target"),
    md("## Viewing Data"),
    code("df.head()"),
    md("## Train-Test Split"),
    code("from sklearn.model_selection import train_test_split\\nX = df.drop(columns=['species'])\\ny = df['species']\\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)"),
    md("## Training the Random Forest\\nFeature importance nikalne ke liye pehle model train karna padta hai."),
    code("from sklearn.ensemble import RandomForestClassifier\\nrf_model = RandomForestClassifier(n_estimators=100, random_state=42)\\nrf_model.fit(X_train, y_train)"),
    md("## Extracting Feature Importances\\nRandom Forest object ke paas ek attribute hota hai `.feature_importances_` jo har feature ka importance score batata hai. In sabhi scores ka sum 1 hota hai."),
    code("importances = rf_model.feature_importances_\\nprint(importances)"),
    md("In scores ko unke feature names ke sath map karte hain taaki samajhne mein aasaani ho."),
    code("feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})\\nfeature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)\\nfeature_imp_df"),
    md("## Visualizing Feature Importance\\nNumbers se behtar graphs hote hain. Chaliye ek horizontal bar plot banate hain importances ko visualize karne ke liye."),
    code("plt.figure(figsize=(8, 5))\\nsns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')\\nplt.title('Random Forest Feature Importance')\\nplt.xlabel('Importance Score')\\nplt.ylabel('Features')\\nplt.show()"),
    md("## Conclusion\\nGraph clearly dikhata hai ki `petal length` aur `petal width` sabse zyada important features hain Iris flower ki species predict karne ke liye, jabki `sepal length` aur `sepal width` ki importance bahut kam hai.")
]
write_nb("Day 37 - Feature Importance.ipynb", d37)

# DAY 38
d38 = [
    md("# Day 38: Trees & SVM Assessment\\n\\n**Phase 3: Tree Models & SVM**\\n\\nAaj humara assessment day hai! Hum ab tak padhe hue 3 main algorithms: **Decision Trees**, **Random Forests**, aur **SVM** ko ek hi classification problem par compare karenge.\\n\\n### Topics Covered:\\n- Model Comparison Pipeline\\n- Cross-validation Scores\\n- Confusion Matrix Visualization for multiple models"),
    md("## Importing Libraries"),
    code("import pandas as pd\\nimport numpy as np\\nimport matplotlib.pyplot as plt\\nimport seaborn as sns"),
    md("## Creating a Synthetic Dataset\\nClassification problem ke liye hum `make_classification` use karke ek 1000 samples ka dataset generate karenge."),
    code("from sklearn.datasets import make_classification\\nX, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)\\nprint('Data Shape:', X.shape)"),
    md("## Train-Test Split"),
    code("from sklearn.model_selection import train_test_split\\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"),
    md("## Initializing Models\\nTeeno models ko initialize karte hain default hyperparameters ke sath."),
    code("from sklearn.tree import DecisionTreeClassifier\\nfrom sklearn.ensemble import RandomForestClassifier\\nfrom sklearn.svm import SVC\\ndt_model = DecisionTreeClassifier(random_state=42)\\nrf_model = RandomForestClassifier(n_estimators=100, random_state=42)\\nsvm_model = SVC(random_state=42)"),
    md("Ek dictionary banate hain taaki iteration easy ho jaye."),
    code("models = {'Decision Tree': dt_model, 'Random Forest': rf_model, 'SVM': svm_model}"),
    md("## Training and Evaluating Models\\nHum loop lagakar teeno models ko train karenge, test set par accuracy calculate karenge, aur predictions save karenge."),
    code("from sklearn.metrics import accuracy_score\\npredictions = {}\\nfor name, model in models.items():\\n    model.fit(X_train, y_train)\\n    y_pred = model.predict(X_test)\\n    predictions[name] = y_pred\\n    acc = accuracy_score(y_test, y_pred)\\n    print(f'{name} Accuracy: {acc:.4f}')"),
    md("## Visualizing Confusion Matrices\\nAccuracy dekhne ke baad, dekhte hain kaunsa model kahan mistakes kar raha hai (False Positives vs False Negatives) using Confusion Matrix."),
    code("from sklearn.metrics import confusion_matrix\\nfig, axes = plt.subplots(1, 3, figsize=(18, 5))\\nfor i, (name, y_pred) in enumerate(predictions.items()):\\n    cm = confusion_matrix(y_test, y_pred)\\n    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)\\n    axes[i].set_title(f'{name} Confusion Matrix')\\n    axes[i].set_xlabel('Predicted')\\n    axes[i].set_ylabel('Actual')\\nplt.tight_layout()\\nplt.show()"),
    md("## Conclusion\\n- Random Forest usually Decision Tree se better perform karta hai due to ensemble nature.\\n- SVM ki performance scale aur hyperparameter (C, gamma) par strongly depend karti hai. Synthetic data par default SVM kabhi kabhi achha perform nahi karta bina proper scaling ke.")
]
write_nb("Day 038 - Trees SVM Assessment.ipynb", d38)
