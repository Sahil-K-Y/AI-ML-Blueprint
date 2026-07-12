import json
import os

notebooks_dir = r'D:\Desktop\repos\AI-ML-Blueprint\Phase-3 - Tree Models & SVM'

def make_cell(cell_type, source, execution_count=None, id_str=None):
    """Create a notebook cell dict."""
    import uuid
    cell = {
        'cell_type': cell_type,
        'id': id_str or str(uuid.uuid4()),
        'metadata': {},
        'source': source if isinstance(source, list) else [source]
    }
    if cell_type == 'code':
        cell['execution_count'] = execution_count
        cell['outputs'] = []
    return cell

def make_notebook(cells):
    """Create a complete notebook dict."""
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3 (ipykernel)',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'codemirror_mode': {'name': 'ipython', 'version': 3},
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'name': 'python',
                'nbconvert_exporter': 'python',
                'pygments_lexer': 'ipython3',
                'version': '3.14.2'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 5
    }

# ============================================================
# DAY 29 - Naive Bayes
# Dataset: Wine (sklearn) - 178 samples, 13 features, 3 classes
# ============================================================
cells_d29 = []

cells_d29.append(make_cell('markdown', [
    '# Day 29: Naive Bayes — Theory & Variants\n',
    '\n',
    '**Phase 3: Tree Models & SVM**\n',
    '\n',
    'Naive Bayes ek probability-based classifier hai jo Bayes Theorem pe based hai. "Naive" isliye kehte hain kyunki ye maan leta hai ki saare features independent hain (jo reality mein rarely hota hai).\n',
    '\n',
    '### Topics Covered:\n',
    '- Bayes Theorem ka intuition\n',
    '- GaussianNB (continuous features ke liye)\n',
    '- MultinomialNB (discrete/count features ke liye)\n',
    '- Laplace Smoothing\n',
    '- Model comparison\n',
    '\n',
    '### Dataset: Wine\n',
    '- 178 samples, 13 chemical features (alcohol, malic_acid, ash, etc.)\n',
    '- 3 target classes: different wine cultivars\n',
    '- Ye dataset basic iris se **zyada complex** hai — real chemical analysis data\n',
]))

cells_d29.append(make_cell('markdown', [
    '## 1. Bayes Theorem — Samajh\n',
    '\n',
    'Bayes theorem ka formula:\n',
    '```\n',
    'P(Class | Features) = P(Features | Class) * P(Class) / P(Features)\n',
    '```\n',
    '\n',
    'Matlab: Pahle se pata hota hai kis class ki probability kitni hai (Prior), fir data dekh ke update karte hain (Posterior).\n',
    '\n',
    'Naive assumption: Saare features **independent** hain. Isliye:\n',
    '```\n',
    'P(Features | Class) = P(F1|Class) * P(F2|Class) * ... * P(Fn|Class)\n',
    '```\n',
    '\n',
    '3 variants:\n',
    '- **GaussianNB** — continuous features, normal distribution assume karta hai\n',
    '- **MultinomialNB** — discrete counts (bag of words, frequency)\n',
    '- **BernoulliNB** — binary features (0/1)\n',
]))

cells_d29.append(make_cell('markdown', [
    '## 2. Import Libraries\n',
]))

cells_d29.append(make_cell('code', [
    '# Import Libraries\n',
    'import numpy as np\n',
    'import pandas as pd\n',
    'import matplotlib.pyplot as plt\n',
    'import seaborn as sns\n',
    '\n',
    'from sklearn.datasets import load_wine\n',
    'from sklearn.model_selection import train_test_split, cross_val_score\n',
    'from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB\n',
    'from sklearn.preprocessing import StandardScaler\n',
    'from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n',
], execution_count=None))

cells_d29.append(make_cell('markdown', [
    '## 3. Load Dataset — Wine\n',
    '\n',
    'Ye dataset Italy mein 3 different wine cultivars ke chemical analysis ka hai. 13 features hain jaise alcohol, malic acid, ash, magnesium, etc.\n',
]))

cells_d29.append(make_cell('code', [
    '# Load Wine Dataset\n',
    'wine = load_wine()\n',
    'df = pd.DataFrame(wine.data, columns=wine.feature_names)\n',
    'df[\'target\'] = wine.target\n',
    'df[\'target_name\'] = df[\'target\'].map({i: name for i, name in enumerate(wine.target_names)})\n',
    '\n',
    'print(f"Shape: {df.shape}")\n',
    'print(f"Features: {len(wine.feature_names)}")\n',
    'print(f"Target classes: {wine.target_names}")\n',
    'print(f"Samples per class:\\n{df[\'target_name\'].value_counts()}")\n',
    'df.head()\n',
], execution_count=None))

cells_d29.append(make_cell('markdown', [
    '## 4. Train-Test Split\n',
]))

cells_d29.append(make_cell('code', [
    'X = wine.data\n',
    'y = wine.target\n',
    '\n',
    'X_train, X_test, y_train, y_test = train_test_split(\n',
    '    X, y, test_size=0.2, random_state=42, stratify=y\n',
    ')\n',
    '\n',
    'print(f"Train: {X_train.shape}, Test: {X_test.shape}")\n',
], execution_count=None))

cells_d29.append(make_cell('markdown', [
    '## 5. GaussianNB Model\n',
    '\n',
    'GaussianNB har feature ke liye Normal (Gaussian) distribution assume karta hai. Har class ke liye har feature ka mean aur variance nikalta hai, phir probability calculate karta hai.\n',
    '\n',
    'Formula: P(x_i | y) = 1/(sqrt(2πσ²)) * exp(-(x_i - μ)² / (2σ²))\n',
]))

cells_d29.append(make_cell('code', [
    '# GaussianNB — continuous features ke liye best\n',
    'gnb = GaussianNB()\n',
    'gnb.fit(X_train, y_train)\n',
    '\n',
    'y_pred = gnb.predict(X_test)\n',
    'acc = accuracy_score(y_test, y_pred)\n',
    '\n',
    'print(f"GaussianNB Test Accuracy: {acc:.4f}")\n',
    'print(f"\\nClassification Report:")\n',
    'print(classification_report(y_test, y_pred, target_names=wine.target_names))\n',
], execution_count=None))

cells_d29.append(make_cell('markdown', [
    '## 6. Cross-Validation Score\n',
]))

cells_d29.append(make_cell('code', [
    '# Cross-validation se zyada reliable score\n',
    'cv_scores = cross_val_score(GaussianNB(), X, y, cv=5)\n',
    'print(f"CV Scores: {cv_scores}")\n',
    'print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")\n',
], execution_count=None))

cells_d29.append(make_cell('markdown', [
    '## 7. MultinomialNB vs GaussianNB\n',
    '\n',
    'MultinomialNB discrete data ke liye hota hai (counts, frequencies). Continuous data pe isko apply karne se pahle scaling karni padti hai (positive values laani hoti hain).\n',
]))

cells_d29.append(make_cell('code', [
    '# MultinomialNB ko positive values chahiye — MinMaxScaler use karo\n',
    'from sklearn.preprocessing import MinMaxScaler\n',
    '\n',
    'scaler = MinMaxScaler()\n',
    'X_train_scaled = scaler.fit_transform(X_train)\n',
    'X_test_scaled = scaler.transform(X_test)\n',
    '\n',
    'mnb = MultinomialNB()\n',
    'mnb.fit(X_train_scaled, y_train)\n',
    'y_pred_mnb = mnb.predict(X_test_scaled)\n',
    'print(f"MultinomialNB Accuracy: {accuracy_score(y_test, y_pred_mnb):.4f}")\n',
], execution_count=None))

cells_d29.append(make_cell('markdown', [
    '## 8. Confusion Matrix\n',
]))

cells_d29.append(make_cell('code', [
    '# Confusion Matrix visualize karo\n',
    'cm = confusion_matrix(y_test, y_pred)\n',
    'plt.figure(figsize=(6, 4))\n',
    'sns.heatmap(cm, annot=True, fmt=\'d\', cmap=\'Blues\',\n',
    '            xticklabels=wine.target_names,\n',
    '            yticklabels=wine.target_names)\n',
    'plt.title(\'Confusion Matrix — GaussianNB (Wine Dataset)\')\n',
    'plt.ylabel(\'Actual\')\n',
    'plt.xlabel(\'Predicted\')\n',
    'plt.show()\n',
], execution_count=None))

cells_d29.append(make_cell('markdown', [
    '## Summary\n',
    '\n',
    '| Variant | Use Case |\n',
    '|---------|----------|\n',
    '| GaussianNB | Continuous features (Wine, Iris)\n',
    '| MultinomialNB | Counts, text data (Bag of Words)\n',
    '| BernoulliNB | Binary features (0/1)\n',
    '\n',
    '**Pros:** Fast, simple, small data pe accha kaam karta hai\n',
    '**Cons:** Independence assumption rarely true; correlated features pe performance girta hai\n',
]))

notebook_d29 = make_notebook(cells_d29)

# ============================================================
# DAY 30 - KNN Basics
# Dataset: Digits (sklearn) - 1797 samples, 64 features, 10 classes
# ============================================================
cells_d30 = []

cells_d30.append(make_cell('markdown', [
    '# Day 30: KNN — k-Nearest Neighbors\n',
    '\n',
    '**Phase 3: Tree Models & SVM**\n',
    '\n',
    'KNN ek **instance-based** (lazy) learner hai — model training time pe kuch nahi seekhta, prediction time pe data point ke nearest neighbors ko dekh ke majority vote karta hai.\n',
    '\n',
    '### Topics Covered:\n',
    '- KNN algorithm intuition\n',
    '- Distance metrics (Euclidean, Manhattan, Minkowski)\n',
    '- Choosing optimal k value\n',
    '- Impact of feature scaling\n',
    '\n',
    '### Dataset: Digits (Handwritten)\n',
    '- 1797 samples, 64 features (8x8 pixel images)\n',
    '- 10 classes (digits 0-9)\n',
    '- Ye basic datasets se **hat ke** hai — image data hai\n',
]))

cells_d30.append(make_cell('markdown', [
    '## 1. KNN Algorithm — Kaise kaam karta hai?\n',
    '\n',
    '1. Ek naya data point aata hai\n',
    '2. Us point ke **k nearest neighbors** dhundo (distance measure karke)\n',
    '3. Un neighbors ka majority vote lo\n',
    '4. Jo class zyada baar aayi, wahi predicted class\n',
    '\n',
    '**k = 1:** Sirf ek nearest neighbor — noise sensitive\n',
    '**k = large:** Zyada neighbors — smooth boundary but underfitting\n',
    '\n',
    'Distance formulas:\n',
    '- Euclidean: d = sqrt(Σ(xᵢ - yᵢ)²)\n',
    '- Manhattan: d = Σ|xᵢ - yᵢ|\n',
    '- Minkowski: d = (Σ|xᵢ - yᵢ|ᵖ)^(1/p) — general form\n',
]))

cells_d30.append(make_cell('markdown', [
    '## 2. Import Libraries\n',
]))

cells_d30.append(make_cell('code', [
    'import numpy as np\n',
    'import pandas as pd\n',
    'import matplotlib.pyplot as plt\n',
    'import seaborn as sns\n',
    '\n',
    'from sklearn.datasets import load_digits\n',
    'from sklearn.model_selection import train_test_split, cross_val_score\n',
    'from sklearn.neighbors import KNeighborsClassifier\n',
    'from sklearn.preprocessing import StandardScaler\n',
    'from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n',
], execution_count=None))

cells_d30.append(make_cell('markdown', [
    '## 3. Load Digits Dataset\n',
    '\n',
    'Ye dataset 8x8 pixel ke handwritten digits (0-9) ka hai. Har image ko flatten karke 64 features banaye gaye hain.\n',
]))

cells_d30.append(make_cell('code', [
    '# Load Digits Dataset\n',
    'digits = load_digits()\n',
    'X = digits.data\n',
    'y = digits.target\n',
    '\n',
    'print(f"Dataset shape: {X.shape}")\n',
    'print(f"Number of classes: {len(np.unique(y))}")\n',
    'print(f"Feature dimension: {digits.images.shape[1]}x{digits.images.shape[2]} pixels")\n',
    '\n',
    '# Pehle 5 digits visualize karo\n',
    'fig, axes = plt.subplots(1, 5, figsize=(10, 3))\n',
    'for i, ax in enumerate(axes):\n',
    '    ax.imshow(digits.images[i], cmap=\'gray\')\n',
    '    ax.set_title(f\'Label: {digits.target[i]}\')\n',
    '    ax.axis(\'off\')\n',
    'plt.show()\n',
], execution_count=None))

cells_d30.append(make_cell('markdown', [
    '## 4. Train-Test Split + Scaling\n',
    '\n',
    'KNN distance-based hai, isliye **scaling zaroori** hai. Warna bada scale wala feature dominate karega.\n',
]))

cells_d30.append(make_cell('code', [
    '# Split & Scale — KNN ke liye scaling mandatory!\n',
    'X_train, X_test, y_train, y_test = train_test_split(\n',
    '    X, y, test_size=0.2, random_state=42, stratify=y\n',
    ')\n',
    '\n',
    'scaler = StandardScaler()\n',
    'X_train_scaled = scaler.fit_transform(X_train)\n',
    'X_test_scaled = scaler.transform(X_test)\n',
    '\n',
    'print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")\n',
], execution_count=None))

cells_d30.append(make_cell('markdown', [
    '## 5. KNN Model (k=5 default)\n',
]))

cells_d30.append(make_cell('code', [
    '# KNN with default k=5\n',
    'knn = KNeighborsClassifier(n_neighbors=5)\n',
    'knn.fit(X_train_scaled, y_train)\n',
    '\n',
    'y_pred = knn.predict(X_test_scaled)\n',
    'print(f"KNN (k=5) Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")\n',
    'print(f"\\nClassification Report:")\n',
    'print(classification_report(y_test, y_pred))\n',
], execution_count=None))

cells_d30.append(make_cell('markdown', [
    '## 6. Different Distance Metrics Compare\n',
]))

cells_d30.append(make_cell('code', [
    '# Euclidean (p=2), Manhattan (p=1), Minkowski with different p\n',
    'metrics = [\'euclidean\', \'manhattan\', \'minkowski\']\n',
    'for metric in metrics:\n',
    '    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)\n',
    '    scores = cross_val_score(knn, X_train_scaled, y_train, cv=3)\n',
    '    print(f"{metric:12s}: {scores.mean():.4f} (+/- {scores.std():.4f})")\n',
], execution_count=None))

cells_d30.append(make_cell('markdown', [
    '## 7. Effect of k — Elbow Method\n',
]))

cells_d30.append(make_cell('code', [
    '# Different k values try karo, accuracy plot karo\n',
    'k_range = range(1, 31)\n',
    'train_accs = []\n',
    'test_accs = []\n',
    '\n',
    'for k in k_range:\n',
    '    knn = KNeighborsClassifier(n_neighbors=k)\n',
    '    knn.fit(X_train_scaled, y_train)\n',
    '    train_accs.append(accuracy_score(y_train, knn.predict(X_train_scaled)))\n',
    '    test_accs.append(accuracy_score(y_test, knn.predict(X_test_scaled)))\n',
    '\n',
    'plt.figure(figsize=(10, 5))\n',
    'plt.plot(k_range, train_accs, label=\'Train Accuracy\', marker=\'o\')\n',
    'plt.plot(k_range, test_accs, label=\'Test Accuracy\', marker=\'s\')\n',
    'plt.xlabel(\'k (Number of Neighbors)\')\n',
    'plt.ylabel(\'Accuracy\')\n',
    'plt.title(\'KNN: Effect of k on Accuracy\')\n',
    'plt.legend()\n',
    'plt.grid(True)\n',
    'plt.show()\n',
], execution_count=None))

cells_d30.append(make_cell('markdown', [
    '## Summary\n',
    '\n',
    '- **Small k** → Low bias, high variance (overfitting)\n',
    '- **Large k** → High bias, low variance (underfitting)\n',
    '- Optimal k dhundhne ke liye cross-validation use karo\n',
    '- **Scaling must** — KNN distance-based hai\n',
]))

notebook_d30 = make_notebook(cells_d30)

# ============================================================
# DAY 31 - KNN Implementation & Scaling Impact
# Dataset: Penguins (seaborn) - 344 samples, 7 features, 3 classes
# ============================================================
cells_d31 = []

cells_d31.append(make_cell('markdown', [
    '# Day 31: KNN — Implementation & Scaling Impact\n',
    '\n',
    '**Phase 3: Tree Models & SVM**\n',
    '\n',
    'KNN distance-based algorithm hai — isliye feature scaling ka impact bahut bada hota hai. Aaj hum scaling ke saath aur bina scaling ke comparison dekhenge.\n',
    '\n',
    '### Topics Covered:\n',
    '- Scaling impact demonstration\n',
    '- Finding optimal k with GridSearchCV\n',
    '- Distance-weighted KNN\n',
    '- Comparison: unscaled vs scaled\n',
    '- KNN Regressor\n',
    '\n',
    '### Dataset: Penguins (Seaborn)\n',
    '- 344 samples, 7 features (bill_length, bill_depth, flipper_length, etc.)\n',
    '- 3 classes (Adelie, Chinstrap, Gentoo)\n',
    '- Feature scales different hain — scaling impact clearly dikhega\n',
]))

cells_d31.append(make_cell('markdown', [
    '## 1. Import Libraries\n',
]))

cells_d31.append(make_cell('code', [
    'import numpy as np\n',
    'import pandas as pd\n',
    'import matplotlib.pyplot as plt\n',
    'import seaborn as sns\n',
    '\n',
    'from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score\n',
    'from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor\n',
    'from sklearn.preprocessing import StandardScaler\n',
    'from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n',
], execution_count=None))

cells_d31.append(make_cell('markdown', [
    '## 2. Load Penguins Dataset\n',
    '\n',
    'Antarctic penguins ka data. Bill length, flipper length jaise features hain — inke scales different hain.\n',
]))

cells_d31.append(make_cell('code', [
    '# Load Penguins Dataset\n',
    'df = sns.load_dataset(\'penguins\')\n',
    'print(f"Raw shape: {df.shape}")\n',
    'print(f"Missing values:\\n{df.isnull().sum()}")\n',
    '\n',
    '# Drop missing values\n',
    'df = df.dropna()\n',
    'print(f"After dropna: {df.shape}")\n',
    'print(f"\\nFeature value ranges:")\n',
    'print(df.describe())\n',
    'df.head()\n',
], execution_count=None))

cells_d31.append(make_cell('markdown', [
    '## 3. Prepare Features & Target\n',
]))

cells_d31.append(make_cell('code', [
    '# Encode target & select numeric features\n',
    'from sklearn.preprocessing import LabelEncoder\n',
    '\n',
    'cat_cols = df.select_dtypes(include=[\'object\', \'category\']).columns\n',
    'print(f"Categorical columns: {list(cat_cols)}")\n',
    '\n',
    '# Species column ko encode karo\n',
    'le = LabelEncoder()\n',
    'y = le.fit_transform(df[\'species\'])\n',
    'print(f"Classes: {list(le.classes_)}")\n',
    '\n',
    '# Numeric features select karo\n',
    'X = df.select_dtypes(include=[np.number])\n',
    'print(f"Feature matrix: {X.shape}")\n',
    'X.head()\n',
], execution_count=None))

cells_d31.append(make_cell('markdown', [
    '## 4. Scale Impact — Unscaled vs Scaled\n',
]))

cells_d31.append(make_cell('code', [
    '# Split\n',
    'X_train, X_test, y_train, y_test = train_test_split(\n',
    '    X, y, test_size=0.2, random_state=42, stratify=y\n',
    ')\n',
    '\n',
    '# Without scaling\n',
    'knn_unscaled = KNeighborsClassifier(n_neighbors=5)\n',
    'knn_unscaled.fit(X_train, y_train)\n',
    'unscaled_acc = accuracy_score(y_test, knn_unscaled.predict(X_test))\n',
    '\n',
    '# With scaling\n',
    'scaler = StandardScaler()\n',
    'X_train_scaled = scaler.fit_transform(X_train)\n',
    'X_test_scaled = scaler.transform(X_test)\n',
    '\n',
    'knn_scaled = KNeighborsClassifier(n_neighbors=5)\n',
    'knn_scaled.fit(X_train_scaled, y_train)\n',
    'scaled_acc = accuracy_score(y_test, knn_scaled.predict(X_test_scaled))\n',
    '\n',
    'print(f"Without Scaling: {unscaled_acc:.4f}")\n',
    'print(f"With Scaling:    {scaled_acc:.4f}")\n',
    'print(f"Difference:      {scaled_acc - unscaled_acc:+.4f}")\n',
], execution_count=None))

cells_d31.append(make_cell('markdown', [
    '## 5. Optimal k with GridSearchCV\n',
]))

cells_d31.append(make_cell('code', [
    '# GridSearchCV se best k dhundo\n',
    'param_grid = {\'n_neighbors\': range(1, 31), \'weights\': [\'uniform\', \'distance\']}\n',
    '\n',
    'grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring=\'accuracy\', n_jobs=-1)\n',
    'grid.fit(X_train_scaled, y_train)\n',
    '\n',
    'print(f"Best params: {grid.best_params_}")\n',
    'print(f"Best CV accuracy: {grid.best_score_:.4f}")\n',
    'print(f"Test accuracy: {accuracy_score(y_test, grid.predict(X_test_scaled)):.4f}")\n',
], execution_count=None))

cells_d31.append(make_cell('markdown', [
    '## 6. Distance-Weighted vs Uniform\n',
    '\n',
    'Distance-weighted KNN — paas wale neighbors ka zyada vote weight hota hai. Uniform mein sabka equal.\n',
]))

cells_d31.append(make_cell('code', [
    '# Distance-weighted KNN\n',
    'knn_dist = KNeighborsClassifier(n_neighbors=5, weights=\'distance\')\n',
    'knn_dist.fit(X_train_scaled, y_train)\n',
    'dist_acc = accuracy_score(y_test, knn_dist.predict(X_test_scaled))\n',
    'print(f"Distance-weighted accuracy: {dist_acc:.4f}")\n',
], execution_count=None))

cells_d31.append(make_cell('markdown', [
    '## 7. KNN Regressor\n',
    '\n',
    'KNN sirf classification ke liye nahi, regression ke liye bhi use hota hai. Neighbors ke target values ka average le lete hain.\n',
]))

cells_d31.append(make_cell('code', [
    '# KNN Regression — flipper_length predict karo\n',
    'from sklearn.datasets import fetch_california_housing\n',
    'from sklearn.metrics import r2_score\n',
    '\n',
    'housing = fetch_california_housing()\n',
    'X_h, y_h = housing.data[:2000], housing.target[:2000]  # sample for speed\n',
    '\n',
    'X_train, X_test, y_train, y_test = train_test_split(X_h, y_h, test_size=0.2, random_state=42)\n',
    '\n',
    'scaler = StandardScaler()\n',
    'X_train_scaled = scaler.fit_transform(X_train)\n',
    'X_test_scaled = scaler.transform(X_test)\n',
    '\n',
    'knn_reg = KNeighborsRegressor(n_neighbors=5)\n',
    'knn_reg.fit(X_train_scaled, y_train)\n',
    'y_pred = knn_reg.predict(X_test_scaled)\n',
    'print(f"KNN Regression R² Score: {r2_score(y_test, y_pred):.4f}")\n',
], execution_count=None))

cells_d31.append(make_cell('markdown', [
    '## Summary\n',
    '\n',
    '- **Scaling ka impact bahut bada hota hai** — hamesha scale karo KNN ke liye\n',
    '- Distance-weighted KNN usually uniform se better hota hai\n',
    '- KNN regression ke liye bhi kaam karta hai (neighbors ka average)\n',
    '- Best k dhundhne ke liye GridSearchCV use karo\n',
]))

notebook_d31 = make_notebook(cells_d31)

# ============================================================
# DAY 32 - Model Selection Guidelines & Leaderboard
# Dataset: Wine Quality (UCI via URL) - 4898 samples, 11 features
# ============================================================
cells_d32 = []

cells_d32.append(make_cell('markdown', [
    '# Day 32: Model Selection Guidelines & Leaderboard\n',
    '\n',
    '**Phase 3: Tree Models & SVM**\n',
    '\n',
    'Aaj hum ek hi dataset pe multiple models (Logistic Regression, SVM, KNN, Decision Tree, Naive Bayes) apply karenge aur compare karenge — jaisi competition mein hota hai.\n',
    '\n',
    '### Topics Covered:\n',
    '- Cross-validation comparison of LR, SVM, KNN, DT, NB\n',
    '- Standardized model leaderboard\n',
    '- Time vs Accuracy trade-off\n',
    '- Model selection guidelines\n',
    '\n',
    '### Dataset: Wine Quality (White Wine)\n',
    '- 4898 samples, 11 features (acidity, sugar, pH, alcohol, etc.)\n',
    '- Target: wine quality (0-10 scale) → binary classification (good/bad)\n',
    '- Ye basic datasets se **zyada realistic** hai\n',
]))

cells_d32.append(make_cell('markdown', [
    '## 1. Import Libraries\n',
]))

cells_d32.append(make_cell('code', [
    'import numpy as np\n',
    'import pandas as pd\n',
    'import matplotlib.pyplot as plt\n',
    'import seaborn as sns\n',
    'import time\n',
    '\n',
    'from sklearn.model_selection import train_test_split, cross_val_score\n',
    'from sklearn.preprocessing import StandardScaler, LabelEncoder\n',
    'from sklearn.metrics import accuracy_score, classification_report\n',
    '\n',
    '# Classifiers\n',
    'from sklearn.linear_model import LogisticRegression\n',
    'from sklearn.svm import SVC\n',
    'from sklearn.neighbors import KNeighborsClassifier\n',
    'from sklearn.tree import DecisionTreeClassifier\n',
    'from sklearn.naive_bayes import GaussianNB\n',
], execution_count=None))

cells_d32.append(make_cell('markdown', [
    '## 2. Load Wine Quality Dataset\n',
]))

cells_d32.append(make_cell('code', [
    '# Wine Quality dataset — UCI se CSV download\n',
    'url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"\n',
    'df = pd.read_csv(url, sep=\';\')\n',
    '\n',
    'print(f"Shape: {df.shape}")\n',
    'print(f"Target distribution:\\n{df[\'quality\'].value_counts().sort_index()}")\n',
    'df.head()\n',
], execution_count=None))

cells_d32.append(make_cell('markdown', [
    '## 3. Binary Classification Setup\n',
    '\n',
    'Quality 0-5 → Bad (0), Quality 6-10 → Good (1)\n',
]))

cells_d32.append(make_cell('code', [
    '# Convert to binary classification\n',
    'df[\'quality_binary\'] = (df[\'quality\'] >= 6).astype(int)\n',
    'print(f"Binary class distribution:\\n{df[\'quality_binary\'].value_counts()}")\n',
    '\n',
    'X = df.drop(columns=[\'quality\', \'quality_binary\'])\n',
    'y = df[\'quality_binary\']\n',
    '\n',
    'X_train, X_test, y_train, y_test = train_test_split(\n',
    '    X, y, test_size=0.2, random_state=42, stratify=y\n',
    ')\n',
    '\n',
    'scaler = StandardScaler()\n',
    'X_train_scaled = scaler.fit_transform(X_train)\n',
    'X_test_scaled = scaler.transform(X_test)\n',
    '\n',
    'print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")\n',
], execution_count=None))

cells_d32.append(make_cell('markdown', [
    '## 4. Model Leaderboard\n',
    '\n',
    'Har model ko train karo, time measure karo, CV accuracy aur test accuracy compare karo.\n',
]))

cells_d32.append(make_cell('code', [
    '# All models with default params\n',
    'models = {\n',
    '    \'Logistic Regression\': LogisticRegression(max_iter=1000, random_state=42),\n',
    '    \'SVM (RBF)\': SVC(random_state=42),\n',
    '    \'KNN (k=5)\': KNeighborsClassifier(n_neighbors=5),\n',
    '    \'Decision Tree\': DecisionTreeClassifier(random_state=42),\n',
    '    \'Gaussian Naive Bayes\': GaussianNB()\n',
    '}\n',
    '\n',
    'results = []\n',
    '\n',
    'for name, model in models.items():\n',
    '    start = time.time()\n',
    '    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3)\n',
    '    train_time = time.time() - start\n',
    '\n',
    '    model.fit(X_train_scaled, y_train)\n',
    '    y_pred = model.predict(X_test_scaled)\n',
    '    test_acc = accuracy_score(y_test, y_pred)\n',
    '\n',
    '    results.append({\n',
    '        \'Model\': name,\n',
    '        \'CV Accuracy\': f"{cv_scores.mean():.4f}",\n',
    '        \'Test Accuracy\': f"{test_acc:.4f}",\n',
    '        \'Time (s)\': f"{train_time:.4f}"\n',
    '    })\n',
    '\n',
    'leaderboard = pd.DataFrame(results)\n',
    'print("=== Model Leaderboard ===\\n")\n',
    'print(leaderboard.to_string(index=False))\n',
], execution_count=None))

cells_d32.append(make_cell('markdown', [
    '## 5. Visualization\n',
]))

cells_d32.append(make_cell('code', [
    '# Bar plot compare\n',
    'plt.figure(figsize=(10, 5))\n',
    'x = np.arange(len(results))\n',
    'cv_accs = [float(r[\'CV Accuracy\']) for r in results]\n',
    'test_accs = [float(r[\'Test Accuracy\']) for r in results]\n',
    '\n',
    'plt.bar(x - 0.2, cv_accs, 0.4, label=\'CV Accuracy\')\n',
    'plt.bar(x + 0.2, test_accs, 0.4, label=\'Test Accuracy\')\n',
    'plt.xticks(x, [r[\'Model\'] for r in results], rotation=45, ha=\'right\')\n',
    'plt.ylabel(\'Accuracy\')\n',
    'plt.title(\'Model Comparison — Wine Quality Dataset\')\n',
    'plt.legend()\n',
    'plt.tight_layout()\n',
    'plt.show()\n',
], execution_count=None))

cells_d32.append(make_cell('markdown', [
    '## Summary — Model Selection Guidelines\n',
    '\n',
    '| Scenario | Best Model |\n',
    '|----------|-----------|\n',
    '| Small dataset (<1000 samples) | SVM / Naive Bayes |\n',
    '| Large dataset (>10k samples) | Logistic Regression / Linear SVM |\n',
    '| Text classification | Naive Bayes (Multinomial) |\n',
    '| Image data | KNN / Deep Learning |\n',
    '| When interpretability needed | Decision Tree / Logistic Regression |\n',
    '| When speed matters | Naive Bayes / Logistic Regression |\n',
    '| Complex relationships | SVM (RBF) / Ensemble methods |\n',
]))

notebook_d32 = make_notebook(cells_d32)

# ============================================================
# DAY 33 - Classification Project
# Dataset: Mushroom (UCI) - 8124 samples, 22 features, 2 classes
# ============================================================
cells_d33 = []

cells_d33.append(make_cell('markdown', [
    '# Day 33: Classification Project — Algorithm Selection\n',
    '\n',
    '**Phase 3: Tree Models & SVM**\n',
    '\n',
    'Real-world project workflow: Load raw data, explore, preprocess, train multiple models, evaluate, select best.\n',
    '\n',
    '### Topics Covered:\n',
    '- Data cleaning (missing values, categorical encoding)\n',
    '- Multiple classifier baseline\n',
    '- Evaluation with multiple metrics\n',
    '- Selecting best candidate\n',
    '\n',
    '### Dataset: Mushroom Classification\n',
    '- 8124 samples, 22 categorical features (cap_shape, gill_color, odor, etc.)\n',
    '- Target: edible (e) vs poisonous (p)\n',
    '- Completely categorical — preprocessing practice ke liye excellent\n',
]))

cells_d33.append(make_cell('markdown', [
    '## 1. Import Libraries\n',
]))

cells_d33.append(make_cell('code', [
    'import numpy as np\n',
    'import pandas as pd\n',
    'import matplotlib.pyplot as plt\n',
    'import seaborn as sns\n',
    'import time\n',
    'import warnings\n',
    'warnings.filterwarnings(\'ignore\')\n',
    '\n',
    'from sklearn.model_selection import train_test_split, cross_val_score\n',
    'from sklearn.preprocessing import LabelEncoder\n',
    'from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score\n',
    '\n',
    'from sklearn.linear_model import LogisticRegression\n',
    'from sklearn.tree import DecisionTreeClassifier\n',
    'from sklearn.ensemble import RandomForestClassifier\n',
    'from sklearn.naive_bayes import GaussianNB\n',
    'from sklearn.svm import SVC\n',
], execution_count=None))

cells_d33.append(make_cell('markdown', [
    '## 2. Load Mushroom Dataset\n',
]))

cells_d33.append(make_cell('code', [
    '# Load Mushroom Dataset\n',
    'url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"\n',
    'columns = [\'class\', \'cap_shape\', \'cap_surface\', \'cap_color\', \'bruises\', \'odor\',\n',
    '           \'gill_attachment\', \'gill_spacing\', \'gill_size\', \'gill_color\',\n',
    '           \'stalk_shape\', \'stalk_root\', \'stalk_surface_above\', \'stalk_surface_below\',\n',
    '           \'stalk_color_above\', \'stalk_color_below\', \'veil_type\', \'veil_color\',\n',
    '           \'ring_number\', \'ring_type\', \'spore_print_color\', \'population\', \'habitat\']\n',
    '\n',
    'df = pd.read_csv(url, header=None, names=columns)\n',
    'print(f"Shape: {df.shape}")\n',
    'print(f"Target: {df[\'class\'].value_counts().to_dict()}")\n',
    'df.head()\n',
], execution_count=None))

cells_d33.append(make_cell('markdown', [
    '## 3. Encode Categorical Features\n',
]))

cells_d33.append(make_cell('code', [
    '# Saare features categorical hain — encode karo\n',
    'le_dict = {}\n',
    'df_encoded = df.copy()\n',
    'for col in df_encoded.columns:\n',
    '    le_dict[col] = LabelEncoder()\n',
    '    df_encoded[col] = le_dict[col].fit_transform(df_encoded[col])\n',
    '\n',
    'X = df_encoded.drop(columns=[\'class\'])\n',
    'y = df_encoded[\'class\']\n',
    '\n',
    'print(f"Feature matrix: {X.shape}")\n',
    'print(f"Classes: edible={le_dict[\'class\'].classes_[0]}, poisonous={le_dict[\'class\'].classes_[1]}")\n',
], execution_count=None))

cells_d33.append(make_cell('markdown', [
    '## 4. Train-Test Split\n',
]))

cells_d33.append(make_cell('code', [
    'X_train, X_test, y_train, y_test = train_test_split(\n',
    '    X, y, test_size=0.2, random_state=42, stratify=y\n',
    ')\n',
    'print(f"Train: {X_train.shape}, Test: {X_test.shape}")\n',
], execution_count=None))

cells_d33.append(make_cell('markdown', [
    '## 5. Train Multiple Models\n',
]))

cells_d33.append(make_cell('code', [
    'models = {\n',
    '    \'Logistic Regression\': LogisticRegression(max_iter=1000, random_state=42),\n',
    '    \'Decision Tree\': DecisionTreeClassifier(random_state=42),\n',
    '    \'Random Forest\': RandomForestClassifier(n_estimators=100, random_state=42),\n',
    '    \'Gaussian NB\': GaussianNB(),\n',
    '    \'SVM\': SVC(probability=True, random_state=42)\n',
    '}\n',
    '\n',
    'results = []\n',
    'for name, model in models.items():\n',
    '    start = time.time()\n',
    '    model.fit(X_train, y_train)\n',
    '    train_time = time.time() - start\n',
    '\n',
    '    y_pred = model.predict(X_test)\n',
    '    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, \'predict_proba\') else None\n',
    '\n',
    '    results.append({\n',
    '        \'Model\': name,\n',
    '        \'Accuracy\': f"{accuracy_score(y_test, y_pred):.4f}",\n',
    '        \'Precision\': f"{precision_score(y_test, y_pred):.4f}",\n',
    '        \'Recall\': f"{recall_score(y_test, y_pred):.4f}",\n',
    '        \'F1-Score\': f"{f1_score(y_test, y_pred):.4f}",\n',
    '        \'Time\': f"{train_time:.4f}s"\n',
    '    })\n',
    '\n',
    'print("=== Model Comparison ===\\n")\n',
    'print(pd.DataFrame(results).to_string(index=False))\n',
], execution_count=None))

cells_d33.append(make_cell('markdown', [
    '## Summary\n',
    '\n',
    '- Different models different strengths\n',
    '- Accuracy alone enough nahi — Precision, Recall, F1 bhi dekho\n',
    '- Mushroom dataset pe Decision Tree / Random Forest generally best aata hai kyunki decision boundaries categorical hai\n',
]))

notebook_d33 = make_notebook(cells_d33)

# ============================================================
# DAY 34 - Ensemble Methods — Voting & Bagging
# Dataset: same Digits dataset
# ============================================================
cells_d34 = []

cells_d34.append(make_cell('markdown', [
    '# Day 34: Ensemble Methods — Voting & Bagging\n',
    '\n',
    '**Phase 3: Tree Models & SVM**\n',
    '\n',
    'Ensemble = Multiple models ko combine karke better prediction lena. "Unity is Strength" wala concept.\n',
    '\n',
    '### Topics Covered:\n',
    '- Hard Voting vs Soft Voting\n',
    '- VotingClassifier\n',
    '- BaggingClassifier (Bootstrap Aggregating)\n',
    '- Out-of-Bag (OOB) Score\n',
    '- Ensemble vs Individual comparison\n',
    '\n',
    '### Dataset: Digits (Handwritten)\n',
    '- 1797 samples, 64 features\n',
    '- 10 classes — ensemble ka asar clearly dikhega\n',
]))

cells_d34.append(make_cell('markdown', [
    '## 1. Voting — Samajh\n',
    '\n',
    '**Hard Voting:** Har model apna vote deta hai (class prediction), jo class ko zyada votes milte hain wo winner\n',
    '**Soft Voting:** Har model probability deta hai, sabki probability average karte hain, jo class ki sabse zyada average probability wo winner\n',
    '\n',
    '**Bagging:** Ek hi algorithm ko different bootstrap samples (with replacement) pe train karte hain, phir average/majority vote lete hain. Variance reduce hota hai.\n',
]))

cells_d34.append(make_cell('markdown', [
    '## 2. Import Libraries\n',
]))

cells_d34.append(make_cell('code', [
    'import numpy as np\n',
    'import pandas as pd\n',
    'import matplotlib.pyplot as plt\n',
    'import seaborn as sns\n',
    '\n',
    'from sklearn.datasets import load_digits\n',
    'from sklearn.model_selection import train_test_split, cross_val_score\n',
    'from sklearn.preprocessing import StandardScaler\n',
    'from sklearn.metrics import accuracy_score\n',
    '\n',
    '# Ensemble\n',
    'from sklearn.ensemble import VotingClassifier, BaggingClassifier\n',
    'from sklearn.tree import DecisionTreeClassifier\n',
    'from sklearn.neighbors import KNeighborsClassifier\n',
    'from sklearn.linear_model import LogisticRegression\n',
    'from sklearn.svm import SVC\n',
], execution_count=None))

cells_d34.append(make_cell('markdown', [
    '## 3. Load + Split + Scale\n',
]))

cells_d34.append(make_cell('code', [
    'digits = load_digits()\n',
    'X, y = digits.data, digits.target\n',
    '\n',
    'X_train, X_test, y_train, y_test = train_test_split(\n',
    '    X, y, test_size=0.2, random_state=42, stratify=y\n',
    ')\n',
    '\n',
    'scaler = StandardScaler()\n',
    'X_train_scaled = scaler.fit_transform(X_train)\n',
    'X_test_scaled = scaler.transform(X_test)\n',
    '\n',
    'print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")\n',
], execution_count=None))

cells_d34.append(make_cell('markdown', [
    '## 4. VotingClassifier — Hard vs Soft\n',
]))

cells_d34.append(make_cell('code', [
    '# Individual models\n',
    'lr = LogisticRegression(max_iter=1000, random_state=42)\n',
    'knn = KNeighborsClassifier(n_neighbors=5)\n',
    'svm = SVC(probability=True, random_state=42)\n',
    'dt = DecisionTreeClassifier(random_state=42)\n',
    '\n',
    '# Hard Voting\n',
    'voting_hard = VotingClassifier(\n',
    '    estimators=[(\'lr\', lr), (\'knn\', knn), (\'svm\', svm), (\'dt\', dt)],\n',
    '    voting=\'hard\'\n',
    ')\n',
    '\n',
    '# Soft Voting\n',
    'voting_soft = VotingClassifier(\n',
    '    estimators=[(\'lr\', lr), (\'knn\', knn), (\'svm\', svm), (\'dt\', dt)],\n',
    '    voting=\'soft\'\n',
    ')\n',
    '\n',
    'voting_hard.fit(X_train_scaled, y_train)\n',
    'voting_soft.fit(X_train_scaled, y_train)\n',
    '\n',
    'print(f"Hard Voting Accuracy: {accuracy_score(y_test, voting_hard.predict(X_test_scaled)):.4f}")\n',
    'print(f"Soft Voting Accuracy: {accuracy_score(y_test, voting_soft.predict(X_test_scaled)):.4f}")\n',
], execution_count=None))

cells_d34.append(make_cell('markdown', [
    '## 5. Compare Individual vs Ensemble\n',
]))

cells_d34.append(make_cell('code', [
    'models = {\n',
    '    \'Logistic Regression\': lr,\n',
    '    \'KNN (k=5)\': knn,\n',
    '    \'SVM\': svm,\n',
    '    \'Decision Tree\': dt,\n',
    '    \'Hard Voting\': voting_hard,\n',
    '    \'Soft Voting\': voting_soft\n',
    '}\n',
    '\n',
    'accs = {}\n',
    'for name, model in models.items():\n',
    '    model.fit(X_train_scaled, y_train)\n',
    '    accs[name] = accuracy_score(y_test, model.predict(X_test_scaled))\n',
    '\n',
    'pd.DataFrame(list(accs.items()), columns=[\'Model\', \'Accuracy\']).set_index(\'Model\')\n',
], execution_count=None))

cells_d34.append(make_cell('markdown', [
    '## 6. BaggingClassifier\n',
]))

cells_d34.append(make_cell('code', [
    '# Bagging with Decision Trees\n',
    'bagging = BaggingClassifier(\n',
    '    estimator=DecisionTreeClassifier(),\n',
    '    n_estimators=50,\n',
    '    max_samples=0.8,\n',
    '    oob_score=True,\n',
    '    random_state=42\n',
    ')\n',
    'bagging.fit(X_train_scaled, y_train)\n',
    '\n',
    'print(f"Bagging Test Accuracy: {accuracy_score(y_test, bagging.predict(X_test_scaled)):.4f}")\n',
    'print(f"OOB Score (internal CV): {bagging.oob_score_:.4f}")\n',
], execution_count=None))

cells_d34.append(make_cell('markdown', [
    '## 7. Compare Bagging vs Single Tree\n',
]))

cells_d34.append(make_cell('code', [
    'single_tree = DecisionTreeClassifier(random_state=42)\n',
    'single_tree.fit(X_train_scaled, y_train)\n',
    'single_acc = accuracy_score(y_test, single_tree.predict(X_test_scaled))\n',
    '\n',
    'print(f"Single Decision Tree: {single_acc:.4f}")\n',
    'print(f"Bagging (50 trees):   {accuracy_score(y_test, bagging.predict(X_test_scaled)):.4f}")\n',
    'print(f"Improvement:          {accuracy_score(y_test, bagging.predict(X_test_scaled)) - single_acc:+.4f}")\n',
], execution_count=None))

cells_d34.append(make_cell('markdown', [
    '## Summary\n',
    '\n',
    '- **Voting:** Different models ka combination = more stable predictions\n',
    '- **Bagging:** Single algorithm + bootstrap samples = variance reduction\n',
    '- Soft voting usually hard se better hota hai (probabilities zyada informative hain)\n',
    '- Bagging overfitting reduce karta hai\n',
]))

notebook_d34 = make_notebook(cells_d34)

# ============================================================
# DAY 35 - Random Forest Basics
# Dataset: Obesity Risk dataset (UCI/OpenML)
# ============================================================
cells_d35 = []

cells_d35.append(make_cell('markdown', [
    '# Day 35: Random Forest — Theory & Implementation\n',
    '\n',
    '**Phase 3: Tree Models & SVM**\n',
    '\n',
    'Random Forest = Bagging + Random feature selection. Multiple decision trees banata hai, har tree different subset of features pe train hota hai, phir average/majority vote leta hai.\n',
    '\n',
    '### Topics Covered:\n',
    '- Random Forest intuition\n',
    '- n_estimators (number of trees)\n',
    '- max_features (random feature subset)\n',
    '- Feature importance\n',
    '- OOB score\n',
    '\n',
    '### Dataset: Penguins (from seaborn) — different perspective\n',
]))

cells_d35.append(make_cell('markdown', [
    '## 1. Random Forest — Kaise kaam karta hai?\n',
    '\n',
    '1. Dataset se bootstrap samples lo (with replacement)\n',
    '2. Har sample pe decision tree train karo, lekin har split pe random features select karo\n',
    '3. Saare trees ka prediction lo\n',
    '4. Classification → majority vote, Regression → average\n',
    '\n',
    'Bagging vs Random Forest — difference:\n',
    '- Bagging: har split pe saare features consider\n',
    '- Random Forest: har split pe random subset of features\n',
]))

cells_d35.append(make_cell('markdown', [
    '## 2. Import Libraries\n',
]))

cells_d35.append(make_cell('code', [
    'import numpy as np\n',
    'import pandas as pd\n',
    'import matplotlib.pyplot as plt\n',
    'import seaborn as sns\n',
    '\n',
    'from sklearn.model_selection import train_test_split, cross_val_score\n',
    'from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n',
    'from sklearn.tree import DecisionTreeClassifier\n',
    'from sklearn.preprocessing import LabelEncoder\n',
    'from sklearn.metrics import accuracy_score, classification_report\n',
], execution_count=None))

cells_d35.append(make_cell('markdown', [
    '## 3. Load Penguins Dataset\n',
]))

cells_d35.append(make_cell('code', [
    'df = sns.load_dataset(\'penguins\').dropna()\n',
    '\n',
    '# Encode target\n',
    'le = LabelEncoder()\n',
    'y = le.fit_transform(df[\'species\'])\n',
    'X = df.select_dtypes(include=[np.number])\n',
    '\n',
    'X_train, X_test, y_train, y_test = train_test_split(\n',
    '    X, y, test_size=0.2, random_state=42, stratify=y\n',
    ')\n',
    'print(f"Train: {X_train.shape}, Test: {X_test.shape}")\n',
], execution_count=None))

cells_d35.append(make_cell('markdown', [
    '## 4. Random Forest Classifier — Default\n',
]))

cells_d35.append(make_cell('code', [
    '# Default Random Forest (100 trees)\n',
    'rf = RandomForestClassifier(n_estimators=100, random_state=42)\n',
    'rf.fit(X_train, y_train)\n',
    '\n',
    'y_pred = rf.predict(X_test)\n',
    'print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.4f}")\n',
    'print(f"\\nClassification Report:\\n{classification_report(y_test, y_pred)}")\n',
    'print(f"\\nOOB Score: {rf.oob_score_:.4f}")\n',
], execution_count=None))

cells_d35.append(make_cell('markdown', [
    '## 5. Effect of n_estimators (Number of Trees)\n',
]))

cells_d35.append(make_cell('code', [
    '# Kitne trees enough hain?\n',
    'n_estimators_range = [1, 5, 10, 20, 50, 100, 200, 500]\n',
    'scores = []\n',
    '\n',
    'for n in n_estimators_range:\n',
    '    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)\n',
    '    cv = cross_val_score(rf, X_train, y_train, cv=3).mean()\n',
    '    scores.append(cv)\n',
    '\n',
    'plt.figure(figsize=(8, 4))\n',
    'plt.plot(n_estimators_range, scores, marker=\'o\')\n',
    'plt.xscale(\'log\')\n',
    'plt.xlabel(\'Number of Trees (n_estimators)\')\n',
    'plt.ylabel(\'CV Accuracy\')\n',
    'plt.title(\'Random Forest: Effect of n_estimators\')\n',
    'plt.grid(True)\n',
    'plt.show()\n',
], execution_count=None))

cells_d35.append(make_cell('markdown', [
    '## 6. Feature Importance\n',
]))

cells_d35.append(make_cell('code', [
    '# RF automatically feature importance deta hai\n',
    'importances = pd.DataFrame({\n',
    '    \'Feature\': X.columns,\n',
    '    \'Importance\': rf.feature_importances_\n',
    '}).sort_values(\'Importance\', ascending=False)\n',
    '\n',
    'plt.figure(figsize=(8, 4))\n',
    'sns.barplot(data=importances, x=\'Importance\', y=\'Feature\')\n',
    'plt.title(\'Random Forest Feature Importance\')\n',
    'plt.tight_layout()\n',
    'plt.show()\n',
], execution_count=None))

cells_d35.append(make_cell('markdown', [
    '## 7. Random Forest Regressor\n',
]))

cells_d35.append(make_cell('code', [
    '# Regression — flipper length predict karo\n',
    'y_reg = df[\'flipper_length_mm\']\n',
    'X_reg = df.drop(columns=[\'species\', \'flipper_length_mm\']).select_dtypes(include=[np.number])\n',
    '\n',
    'from sklearn.metrics import r2_score\n',
    'X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)\n',
    '\n',
    'rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)\n',
    'rf_reg.fit(X_train_r, y_train_r)\n',
    'y_pred_r = rf_reg.predict(X_test_r)\n',
    'print(f"RF Regressor R²: {r2_score(y_test_r, y_pred_r):.4f}")\n',
], execution_count=None))

cells_d35.append(make_cell('markdown', [
    '## Summary\n',
    '\n',
    '- Random Forest = Multiple random trees ka ensemble\n',
    '- n_estimators increase → performance improve but diminishing returns\n',
    '- Feature importance built-in milta hai\n',
    '- Overfitting kam hota hai single tree ke comparison mein\n',
]))

notebook_d35 = make_notebook(cells_d35)

# ============================================================
# DAY 36 - Random Forest Hyperparameter Tuning
# Dataset: Digits — larger dataset for tuning
# ============================================================
cells_d36 = []

cells_d36.append(make_cell('markdown', [
    '# Day 36: Random Forest — Hyperparameter Tuning\n',
    '\n',
    '**Phase 3: Tree Models & SVM**\n',
    '\n',
    'Random Forest ke multiple hyperparameters hain jinhe tune karke performance improve kar sakte hain. Aaj RandomizedSearchCV se tuning karenge.\n',
    '\n',
    '### Topics Covered:\n',
    '- n_estimators, max_depth, min_samples_split, min_samples_leaf\n',
    '- max_features tuning\n',
    '- RandomizedSearchCV vs GridSearchCV\n',
    '- Criterion (Gini vs Entropy)\n',
    '- Tuning impact analysis\n',
    '\n',
    '### Dataset: Digits (Handwritten)\n',
    '- Larger dataset — tuning ka asar clearly dikhega\n',
]))

cells_d36.append(make_cell('markdown', [
    '## 1. Import Libraries\n',
]))

cells_d36.append(make_cell('code', [
    'import numpy as np\n',
    'import pandas as pd\n',
    'import matplotlib.pyplot as plt\n',
    'import seaborn as sns\n',
    'import time\n',
    '\n',
    'from sklearn.datasets import load_digits\n',
    'from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score\n',
    'from sklearn.ensemble import RandomForestClassifier\n',
    'from sklearn.metrics import accuracy_score, classification_report\n',
], execution_count=None))

cells_d36.append(make_cell('markdown', [
    '## 2. Load Dataset\n',
]))

cells_d36.append(make_cell('code', [
    'digits = load_digits()\n',
    'X, y = digits.data, digits.target\n',
    'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n',
    'print(f"Train: {X_train.shape}, Test: {X_test.shape}")\n',
], execution_count=None))

cells_d36.append(make_cell('markdown', [
    '## 3. Default RF Baseline\n',
]))

cells_d36.append(make_cell('code', [
    'rf_default = RandomForestClassifier(random_state=42)\n',
    'rf_default.fit(X_train, y_train)\n',
    'default_acc = accuracy_score(y_test, rf_default.predict(X_test))\n',
    'print(f"Default RF Accuracy: {default_acc:.4f}")\n',
], execution_count=None))

cells_d36.append(make_cell('markdown', [
    '## 4. RandomizedSearchCV\n',
]))

cells_d36.append(make_cell('code', [
    '# RandomizedSearchCV — GridSearchCV se fast (random combinations try karta hai)\n',
    'param_dist = {\n',
    '    \'n_estimators\': [50, 100, 200, 300],\n',
    '    \'max_depth\': [None, 10, 20, 30],\n',
    '    \'min_samples_split\': [2, 5, 10],\n',
    '    \'min_samples_leaf\': [1, 2, 4],\n',
    '    \'max_features\': [\'sqrt\', \'log2\', None],\n',
    '    \'criterion\': [\'gini\', \'entropy\']\n',
    '}\n',
    '\n',
    'random_search = RandomizedSearchCV(\n',
    '    RandomForestClassifier(random_state=42),\n',
    '    param_distributions=param_dist,\n',
    '    n_iter=30,  # 30 random combinations\n',
    '    cv=3,\n',
    '    scoring=\'accuracy\',\n',
    '    n_jobs=-1,\n',
    '    random_state=42,\n',
    '    verbose=1\n',
    ')\n',
    '\n',
    'start = time.time()\n',
    'random_search.fit(X_train, y_train)\n',
    'print(f"\\nTuning time: {time.time() - start:.2f}s")\n',
    'print(f"Best params: {random_search.best_params_}")\n',
    'print(f"Best CV score: {random_search.best_score_:.4f}")\n',
], execution_count=None))

cells_d36.append(make_cell('markdown', [
    '## 5. Tuned Model Evaluation\n',
]))

cells_d36.append(make_cell('code', [
    'rf_tuned = random_search.best_estimator_\n',
    'tuned_acc = accuracy_score(y_test, rf_tuned.predict(X_test))\n',
    '\n',
    'print(f"Default RF:      {default_acc:.4f}")\n',
    'print(f"Tuned RF:        {tuned_acc:.4f}")\n',
    'print(f"Improvement:     {tuned_acc - default_acc:+.4f}")\n',
    'print(f"\\nClassification Report:")\n',
    'print(classification_report(y_test, rf_tuned.predict(X_test)))\n',
], execution_count=None))

cells_d36.append(make_cell('markdown', [
    '## 6. Feature Importance from Tuned Model\n',
]))

cells_d36.append(make_cell('code', [
    '# Tuned model ki feature importance\n',
    'importances = pd.DataFrame({\n',
    '    \'pixel\': [f\'p{i}\' for i in range(64)],\n',
    '    \'importance\': rf_tuned.feature_importances_\n',
    '}).sort_values(\'importance\', ascending=False).head(10)\n',
    '\n',
    'plt.figure(figsize=(8, 4))\n',
    'sns.barplot(data=importances, x=\'importance\', y=\'pixel\')\n',
    'plt.title(\'Top 10 Important Pixels (Tuned RF)\')\n',
    'plt.tight_layout()\n',
    'plt.show()\n',
], execution_count=None))

cells_d36.append(make_cell('markdown', [
    '## Summary\n',
    '\n',
    '- **n_estimators**: zyada trees = better (but diminishing returns)\n',
    '- **max_depth**: control overfitting\n',
    '- **min_samples_split/leaf**: prevent overfitting\n',
    '- **max_features**: diversity in trees\n',
    '- RandomizedSearchCV > GridSearchCV for large parameter spaces\n',
]))

notebook_d36 = make_notebook(cells_d36)

# ============================================================
# DAY 37 - Feature Importance Analysis
# Dataset: California Housing (sklearn)
# ============================================================
cells_d37 = []

cells_d37.append(make_cell('markdown', [
    '# Day 37: Feature Importance Analysis\n',
    '\n',
    '**Phase 3: Tree Models & SVM**\n',
    '\n',
    'Feature importance batata hai ki model ke liye kaunsa feature sabse important hai. 3 methods: Tree-based (impurity), Permutation importance, and Analysis.\n',
    '\n',
    '### Topics Covered:\n',
    '- Impurity-based importance (Random Forest)\n',
    '- Permutation importance\n',
    '- Feature selection based on importance\n',
    '- Feature importance visualization\n',
    '\n',
    '### Dataset: California Housing\n',
    '- 20640 samples, 8 features (MedInc, HouseAge, AveRooms, etc.)\n',
    '- Ye basic datasets se **hat ke** hai — regression problem\n',
]))

cells_d37.append(make_cell('markdown', [
    '## 1. Import Libraries\n',
]))

cells_d37.append(make_cell('code', [
    'import numpy as np\n',
    'import pandas as pd\n',
    'import matplotlib.pyplot as plt\n',
    'import seaborn as sns\n',
    '\n',
    'from sklearn.datasets import fetch_california_housing\n',
    'from sklearn.model_selection import train_test_split\n',
    'from sklearn.ensemble import RandomForestRegressor\n',
    'from sklearn.metrics import r2_score\n',
    'from sklearn.inspection import permutation_importance\n',
    'from sklearn.feature_selection import SelectFromModel\n',
], execution_count=None))

cells_d37.append(make_cell('markdown', [
    '## 2. Load California Housing Dataset\n',
]))

cells_d37.append(make_cell('code', [
    '# Load California Housing\n',
    'housing = fetch_california_housing()\n',
    'X = pd.DataFrame(housing.data, columns=housing.feature_names)\n',
    'y = housing.target\n',
    '\n',
    'print(f"Dataset shape: {X.shape}")\n',
    'print(f"Feature names: {list(housing.feature_names)}")\n',
    'X.head()\n',
], execution_count=None))

cells_d37.append(make_cell('markdown', [
    '## 3. Train Random Forest\n',
]))

cells_d37.append(make_cell('code', [
    'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n',
    '\n',
    'rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)\n',
    'rf.fit(X_train, y_train)\n',
    '\n',
    'print(f"R² Score: {r2_score(y_test, rf.predict(X_test)):.4f}")\n',
], execution_count=None))

cells_d37.append(make_cell('markdown', [
    '## 4. Method 1: Tree-based (Impurity) Importance\n',
]))

cells_d37.append(make_cell('code', [
    '# Built-in impurity-based importance\n',
    'impurity_imp = pd.DataFrame({\n',
    '    \'Feature\': housing.feature_names,\n',
    '    \'Importance\': rf.feature_importances_\n',
    '}).sort_values(\'Importance\', ascending=False)\n',
    '\n',
    'print("=== Impurity-based Importance ===\\n")\n',
    'print(impurity_imp.to_string(index=False))\n',
    '\n',
    'plt.figure(figsize=(8, 4))\n',
    'sns.barplot(data=impurity_imp, x=\'Importance\', y=\'Feature\')\n',
    'plt.title(\'Feature Importance (Random Forest)\')\n',
    'plt.tight_layout()\n',
    'plt.show()\n',
], execution_count=None))

cells_d37.append(make_cell('markdown', [
    '## 5. Method 2: Permutation Importance\n',
]))

cells_d37.append(make_cell('code', [
    '# Permutation importance — feature ko randomly shuffle karo, dekh do kitna score girti hai\n',
    '# Agar feature important hai → score zyada girega\n',
    'perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)\n',
    '\n',
    'perm_df = pd.DataFrame({\n',
    '    \'Feature\': housing.feature_names,\n',
    '    \'Importance\': perm_imp.importances_mean,\n',
    '    \'Std\': perm_imp.importances_std\n',
    '}).sort_values(\'Importance\', ascending=False)\n',
    '\n',
    'print("=== Permutation Importance ===\\n")\n',
    'print(perm_df.to_string(index=False))\n',
    '\n',
    'plt.figure(figsize=(8, 4))\n',
    'sns.barplot(data=perm_df, x=\'Importance\', y=\'Feature\', xerr=perm_df[\'Std\'])\n',
    'plt.title(\'Permutation Feature Importance\')\n',
    'plt.tight_layout()\n',
    'plt.show()\n',
], execution_count=None))

cells_d37.append(make_cell('markdown', [
    '## 6. Method 3: Feature Selection\n',
]))

cells_d37.append(make_cell('code', [
    '# SelectFromModel — sirf important features select karo\n',
    'selector = SelectFromModel(rf, threshold=\'median\')\n',
    'X_selected = selector.fit_transform(X_train, y_train)\n',
    'selected_features = X.columns[selector.get_support()]\n',
    '\n',
    'print(f"Original features: {X.shape[1]}")\n',
    'print(f"Selected features: {X_selected.shape[1]}")\n',
    'print(f"Selected: {list(selected_features)}")\n',
], execution_count=None))

cells_d37.append(make_cell('markdown', [
    '## Summary\n',
    '\n',
    '| Method | Pros | Cons |\n',
    '|--------|------|------|\n',
    '| Impurity-based | Fast, built-in | Biased towards high-cardinality features |\n',
    '| Permutation | Model-agnostic, more accurate | Slow, multiple repeats needed |\n',
    '\n',
    '- MedInc (Median Income) California housing mein sabse important feature hai\n',
    '- Feature selection model ko simpler aur faster banata hai\n',
]))

notebook_d37 = make_notebook(cells_d37)

# ============================================================
# DAY 38 - Phase Review & Assessment
# ============================================================
cells_d38 = []

cells_d38.append(make_cell('markdown', [
    '# Day 38: Phase Review & Assessment\n',
    '\n',
    '**Phase 3: Tree Models & SVM**\n',
    '\n',
    'Phase 3 ka summary — Decision Trees, SVM, KNN, Naive Bayes, Ensemble methods sabka revision.\n',
    '\n',
    '### Topics Covered:\n',
    '- Phase 3 concepts summary\n',
    '- 20 practice interview questions\n',
    '- Quick code revision snippets\n',
]))

cells_d38.append(make_cell('markdown', [
    '## Phase 3 Summary Table\n',
    '\n',
    '| Day | Topic | Key Concept |\n',
    '|-----|-------|-------------|\n',
    '| 22 | Decision Tree Intuition | Gini, Entropy, Information Gain |\n',
    '| 23 | Decision Tree Implementation | max_depth, min_samples_split |\n',
    '| 24 | Pruning | ccp_alpha, overfitting control |\n',
    '| 25 | Decision Tree Regressor | MSE-based splits |\n',
    '| 26 | SVM Intuition | Margin, Support Vectors, C |\n',
    '| 27 | SVM Kernel Trick | RBF, Linear, Polynomial kernels |\n',
    '| 28 | SVM Tuning | GridSearchCV, C, gamma |\n',
    '| 29 | Naive Bayes | Bayes Theorem, GaussianNB |\n',
    '| 30 | KNN Basics | Distance metrics, k selection |\n',
    '| 31 | KNN Implementation | Scaling impact, weighted KNN |\n',
    '| 32 | Model Leaderboard | Cross-validation comparison |\n',
    '| 33 | Classification Project | End-to-end workflow |\n',
    '| 34 | Voting & Bagging | Ensembles, OOB score |\n',
    '| 35 | Random Forest Basics | n_estimators, feature importance |\n',
    '| 36 | Random Forest Tuning | RandomizedSearchCV |\n',
    '| 37 | Feature Importance | Permutation importance |\n',
]))

cells_d38.append(make_cell('markdown', [
    '## Practice Interview Questions\n',
    '\n',
    '### Decision Trees:\n',
    '1. Gini impurity vs Entropy mein kya difference hai?\n',
    '2. Decision tree overfit kyun karta hai? Kaise rokoge?\n',
    '3. ccp_alpha kya hai?\n',
    '4. Decision tree regression ke liye kaise kaam karta hai?\n',
    '\n',
    '### SVM:\n',
    '5. SVM ka objective function kya hai?\n',
    '6. Hard margin vs Soft margin?\n',
    '7. C parameter kya control karta hai?\n',
    '8. Kernel trick kyun use karte hain?\n',
    '9. RBF kernel ka gamma parameter kya control karta hai?\n',
    '10. SVM ke liye scaling kyun zaroori hai?\n',
    '\n',
    '### Naive Bayes:\n',
    '11. "Naive" kyun kehte hain?\n',
    '12. GaussianNB vs MultinomialNB kab use karte hain?\n',
    '13. Laplace smoothing kya hai?\n',
    '\n',
    '### KNN:\n',
    '14. KNN lazy learner kyun kehlata hai?\n',
    '15. Optimal k kaise select karte hain?\n',
    '16. Scaling ka KNN pe kya asar padta hai?\n',
    '\n',
    '### Ensemble:\n',
    '17. Hard Voting vs Soft Voting?\n',
    '18. Bagging overfit kaise reduce karta hai?\n',
    '19. Random Forest, Bagging se kaise different hai?\n',
    '20. OOB score kya hai?\n',
]))

cells_d38.append(make_cell('markdown', [
    '## Quick Code Revision\n',
    '\n',
    'Saare models ka ek-ek line code reference.\n',
]))

cells_d38.append(make_cell('code', [
    '# === Quick Revision: All Models in Phase 3 ===\n',
    '\n',
    '# Decision Tree\n',
    'from sklearn.tree import DecisionTreeClassifier\n',
    'dt = DecisionTreeClassifier(max_depth=5, random_state=42)\n',
    '\n',
    '# SVM\n',
    'from sklearn.svm import SVC\n',
    'svm = SVC(C=1, gamma=\'scale\', kernel=\'rbf\', random_state=42)\n',
    '\n',
    '# Naive Bayes\n',
    'from sklearn.naive_bayes import GaussianNB\n',
    'nb = GaussianNB()\n',
    '\n',
    '# KNN\n',
    'from sklearn.neighbors import KNeighborsClassifier\n',
    'knn = KNeighborsClassifier(n_neighbors=5, weights=\'distance\')\n',
    '\n',
    '# Voting Ensemble\n',
    'from sklearn.ensemble import VotingClassifier\n',
    'ensemble = VotingClassifier([(\'dt\', dt), (\'svm\', svm), (\'knn\', knn)], voting=\'soft\')\n',
    '\n',
    '# Bagging\n',
    'from sklearn.ensemble import BaggingClassifier\n',
    'bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, oob_score=True)\n',
    '\n',
    '# Random Forest\n',
    'from sklearn.ensemble import RandomForestClassifier\n',
    'rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)\n',
    '\n',
    'print(\"All models ready!\")\n',
], execution_count=None))

cells_d38.append(make_cell('markdown', [
    '## Next Steps\n',
    '\n',
    'Phase 3 complete! 🎉\n',
    '\n',
    'Phase 4 mein hum **Boosting & Advanced Ensembles** padhenge:\n',
    '- AdaBoost\n',
    '- Gradient Boosting\n',
    '- XGBoost\n',
    '- LightGBM\n',
    '- CatBoost\n',
    '- Stacking Ensembles\n',
    '- Kaggle Competition\n',
]))

notebook_d38 = make_notebook(cells_d38)

# ============================================================
# SAVE ALL NOTEBOOKS
# ============================================================

notebooks = [
    ('Day 29 - Naive Bayes.ipynb', notebook_d29),
    ('Day 30 - KNN Basics.ipynb', notebook_d30),
    ('Day 31 - KNN Implementation.ipynb', notebook_d31),
    ('Day 32 - Model Leaderboard.ipynb', notebook_d32),
    ('Day 33 - Classification Project.ipynb', notebook_d33),
    ('Day 34 - Voting Bagging Ensembles.ipynb', notebook_d34),
    ('Day 35 - Random Forest Basics.ipynb', notebook_d35),
    ('Day 36 - Random Forest Tuning.ipynb', notebook_d36),
    ('Day 37 - Feature Importance.ipynb', notebook_d37),
    ('Day 038 - Trees SVM Assessment.ipynb', notebook_d38),
]

for filename, notebook in notebooks:
    filepath = os.path.join(notebooks_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"Created: {filename}")
