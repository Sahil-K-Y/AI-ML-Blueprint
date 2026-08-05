# 🌀 Phase 5 — Unsupervised Learning & Dimensionality Reduction (Days 057 – 061)
## Comprehensive Theory, Mathematics, Code Syntax & Production Blueprint

> **Author:** Sahil Kumar (Yd)  
> **Blueprint:** 365-Day AI/ML Engineer Master Roadmap — Phase 5 (Days 057–061)  
> **Location:** `Phase-5 - Unsupervised Learning/unsupervised_learning_masterclass_057_061.md`  
> **Target:** Master Clustering Algorithms, Density Estimation, Dimensionality Reduction, SVD/PCA Mathematics, and High-Dimensional Visualizations.

---

## 📖 Table of Contents

1. [Day 057 — K-Means Clustering, K-Means++ & Cluster Validation (Elbow & Silhouette)](#-day-057--k-means-clustering-k-means--cluster-validation)
2. [Day 058 — Hierarchical Agglomerative Clustering & Density-Based DBSCAN](#-day-058--hierarchical-agglomerative-clustering--density-based-dbscan)
3. [Day 059 — Principal Component Analysis (PCA), Covariance & SVD Derivation](#-day-059--principal-component-analysis-pca-covariance--svd-derivation)
4. [Day 060 — Non-Linear Dimensionality Reduction (t-SNE & UMAP)](#-day-060--non-linear-dimensionality-reduction-t-sne--umap)
5. [Day 061 — Gaussian Mixture Models (GMM) & Expectation-Maximization (EM) Algorithm](#-day-061--gaussian-mixture-models-gmm--expectation-maximization-em-algorithm)

---

## 🎯 Day 057 — K-Means Clustering, K-Means++ & Cluster Validation

### 💡 1. Intuition & Real-World Problem Analogy

#### Supervised vs Unsupervised Learning
In Supervised Learning, we are given feature vectors $x_i$ and target labels $y_i$. In Unsupervised Learning, there are **no target labels $y_i$**. The goal is to discover underlying hidden structures, patterns, or groupings within data.

#### The Analogy — E-Commerce Customer Segmentation 🛒
Imagine an online shop with millions of customers. You have customer data (Total Spend, Visit Frequency, Items Purchased), but no predefined categories.  
- **Goal**: Automatically group customers into distinct personas (e.g., *"VIP Big Spenders"*, *"Bargain Hunters"*, *"Occasional Buyers"*) to deliver targeted marketing campaigns.

#### Lloyd's K-Means Algorithm Iterative Steps:
1. **Initialize**: Pick $K$ initial centroid seeds $\mu_1, \mu_2, \dots, \mu_K$.
2. **Assign Step**: Assign each data point $x_i$ to its closest centroid based on Euclidean distance:
   $$C^{(t)}(i) = \arg\min_k \|x_i - \mu_k^{(t)}\|^2$$
3. **Update Step**: Recompute each centroid $\mu_k$ as the mean of all points assigned to cluster $k$:
   $$\mu_k^{(t+1)} = \frac{1}{|S_k^{(t)}|} \sum_{x_i \in S_k^{(t)}} x_i$$
4. **Repeat**: Iterate steps 2 & 3 until centroids stop moving ($\|\mu_k^{(t+1)} - \mu_k^{(t)}\| < \epsilon$).

#### K-Means++ Smart Initialization:
Random initialization can lead to poor local minima if centroids start close together.  
**K-Means++** selects the first centroid uniformly at random, then chooses subsequent centroids with probability proportional to the squared distance to the nearest already chosen centroid:
$$P(x_i) = \frac{D(x_i)^2}{\sum_j D(x_j)^2}$$

---

### 📐 2. Mathematical Formulations & Validation Metrics

#### A. WCSS (Within-Cluster Sum of Squares / Inertia):
$$\text{WCSS} = \sum_{k=1}^K \sum_{x_i \in S_k} \|x_i - \mu_k\|^2$$

#### B. Silhouette Coefficient Formula:
For a sample instance $x_i$:
- $a(i) = \text{Mean intra-cluster distance}$ (average distance to all other points in the same cluster).
- $b(i) = \text{Mean nearest-cluster distance}$ (average distance to all points in the closest neighboring cluster).

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
- $s(i) \to +1$: Sample is well-matched to its own cluster and far from neighbors.
- $s(i) \approx 0$: Sample lies on the decision boundary between two clusters.
- $s(i) \to -1$: Sample is misclassified into the wrong cluster.

---

### 💻 3. Python Code Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# 1. Generate Synthetic 2D Clustering Data
X, y_true = make_blobs(n_samples=1000, centers=4, cluster_std=0.85, random_state=42)

# 2. Fit K-Means++ with K=4
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)
cluster_labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# 3. Cluster Validation Metrics
inertia = kmeans.inertia_
sil_score = silhouette_score(X, cluster_labels)

print(f"[Day 057] WCSS (Inertia):           {inertia:.4f}")
print(f"[Day 057] Mean Silhouette Score: {sil_score:.4f}")

# 4. Optimal K Search via Elbow Method & Silhouette Analysis
inertia_list = []
sil_list = []
k_range = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42).fit(X)
    inertia_list.append(km.inertia_)
    sil_list.append(silhouette_score(X, km.labels_))
```

---

## 🌳 Day 058 — Hierarchical Agglomerative Clustering & DBSCAN

### 💡 1. Intuition & Concepts

#### Hierarchical Agglomerative Clustering (Bottom-Up)
Starts with every sample in its own individual cluster. At each step, it merges the pair of clusters with the smallest distance according to a chosen **Linkage Criterion**:
1. **Single Linkage**: Distance between the two *closest* points of two clusters.
2. **Complete Linkage**: Distance between the two *furthest* points of two clusters.
3. **Average Linkage**: Average distance between all pairs of points.
4. **Ward's Linkage**: Minimizes variance growth (sum of squared differences) when merging clusters.

Visualized using a **Dendrogram** tree structure.

#### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
Unlike K-Means (which assumes spherical clusters of equal size), **DBSCAN** discovers clusters of arbitrary shapes and isolates noise points.

#### Core Definitions:
- **Core Point**: A point with at least `min_samples` within distance $\epsilon$ (eps).
- **Border Point**: A point within $\epsilon$ distance of a Core Point, but has fewer than `min_samples` neighbors.
- **Noise Point**: Neither a Core Point nor a Border Point (labeled `-1`).

---

### 💻 2. Python Code Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Non-Convex Moon Dataset (Challenging for K-Means)
X_moons, _ = make_moons(n_samples=600, noise=0.08, random_state=42)

# 2. Agglomerative Hierarchical Clustering (Ward Linkage)
agg_cls = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels_agg = agg_cls.fit_predict(X_moons)

# 3. DBSCAN Density-Based Clustering
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_moons)

n_clusters_db = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise_db = list(labels_dbscan).count(-1)

print(f"[Day 058 DBSCAN] Discovered Clusters: {n_clusters_db} | Noise Points: {n_noise_db}")
```

---

## 📉 Day 059 — Principal Component Analysis (PCA) & SVD

### 💡 1. Intuition & Variance Maximization

#### Dimensionality Reduction Concept
In datasets with tens or hundreds of features, many features are highly correlated or redundant (Curse of Dimensionality).  
**PCA** finds a new orthogonal coordinate system where:
- **PC1 (Principal Component 1)**: Direction of maximum variance in the data.
- **PC2**: Orthogonal to PC1, capturing the maximum remaining variance.

---

### 📐 2. Mathematical Derivation: Covariance Matrix & Eigen-Decomposition

1. **Center Data**: Subtract column means $\mu$:
   $$\bar{X} = X - \mu$$
2. **Covariance Matrix**:
   $$\Sigma = \frac{1}{N-1} \bar{X}^T \bar{X}$$
3. **Eigen-Decomposition**:
   $$\Sigma v_i = \lambda_i v_i$$
   Where $v_i$ is the $i$-th eigenvector (principal component loading vector) and $\lambda_i$ is the eigenvalue representing variance along $v_i$.
4. **Explained Variance Ratio**:
   $$\text{EVR}_i = \frac{\lambda_i}{\sum_{k=1}^D \lambda_k}$$

#### Singular Value Decomposition (SVD) Relation:
Any matrix $\bar{X} \in \mathbb{R}^{N \times D}$ can be decomposed as:
$$\bar{X} = U \Sigma_{\text{svd}} V^T$$
Where $V$ columns are the eigenvectors of $\bar{X}^T \bar{X}$, and singular values $\sigma_i = \sqrt{\lambda_i (N-1)}$.

---

### 💻 3. Python Code Implementation

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. High-Dimensional Digits Dataset (8x8 images = 64 features)
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

# 2. Scale Features
X_scaled = StandardScaler().fit_transform(X_digits)

# 3. Fit PCA to preserve 95% of total variance
pca_95 = PCA(n_components=0.95, random_state=42)
X_pca_95 = pca_95.fit_transform(X_scaled)

print(f"[Day 059 PCA] Original Dimensions: {X_scaled.shape[1]} -> Reduced Dimensions: {X_pca_95.shape[1]}")
print(f"[Day 059 PCA] Cumulative Variance Retained: {np.sum(pca_95.explained_variance_ratio_)*100:.2f}%")
```

---

## 🌌 Day 060 — Non-Linear Dimensionality Reduction (t-SNE & UMAP)

### 💡 1. Intuition & Local Structure Preservation

While PCA is a linear projection (struggles with non-linear manifolds like Swiss Rolls), **t-SNE** and **UMAP** map high-dimensional non-linear structures into 2D/3D visual embeddings by preserving local pairwise similarities.

#### t-SNE (t-Distributed Stochastic Neighbor Embedding):
1. Computes conditional probabilities $p_{j|i}$ that point $x_i$ picks $x_j$ as neighbor under a Gaussian distribution:
   $$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$
2. In low-dimensional space, uses Student-t distribution ($1$-degree of freedom) to solve the **Crowding Problem**:
   $$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_k \sum_{l \neq k} (1 + \|y_k - y_l\|^2)^{-1}}$$
3. Minimizes Kullback-Leibler (KL) divergence using Gradient Descent:
   $$\text{KL}(P \parallel Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

#### Critical Parameters:
- **Perplexity**: Smooth measure of the effective number of neighbors ($5 - 50$).
- **Learning Rate**: Controls optimization step size.

---

### 💻 2. Python Code Implementation

```python
from sklearn.manifold import TSNE

# Fit t-SNE on 64-dimensional Digits dataset
tsne = TSNE(n_components=2, perplexity=30.0, learning_rate='auto', init='pca', random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

print(f"[Day 060 t-SNE] Transformed 64D Digits dataset to 2D Embedding Shape: {X_tsne.shape}")
print(f"[Day 060 t-SNE] Final KL-Divergence Loss: {tsne.kl_divergence_:.4f}")
```

---

## 🔮 Day 061 — Gaussian Mixture Models (GMM) & EM Algorithm

### 💡 1. Intuition & Soft Clustering

K-Means performs **Hard Clustering** (assigns each point strictly to 1 cluster) and assumes spherical shapes.  
**Gaussian Mixture Models (GMM)** perform **Soft Probabilistic Clustering** by modeling data as a mixture of $K$ multivariate Gaussian distributions with arbitrary covariance orientation ellipses.

---

### 📐 2. Mathematical Formulation: Expectation-Maximization (EM) Algorithm

#### Probability Density Function:
$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$$
Where $\pi_k$ is the mixture weight of cluster $k$ ($\sum \pi_k = 1$).

#### 1. E-Step (Expectation):
Compute posterior cluster probability (responsibility $\gamma_{ik}$) for each sample $i$ belonging to cluster $k$:
$$\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$

#### 2. M-Step (Maximization):
Re-estimate parameters using computed responsibilities:
$$N_k = \sum_{i=1}^N \gamma_{ik}$$
$$\mu_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} x_i$$
$$\Sigma_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} (x_i - \mu_k^{\text{new}})(x_i - \mu_k^{\text{new}})^T$$
$$\pi_k^{\text{new}} = \frac{N_k}{N}$$

#### Model Selection Criteria:
- **BIC (Bayesian Information Criterion)**: $\text{BIC} = -2 \ln \hat{L} + p \ln N$ (Lower is better).
- **AIC (Akaike Information Criterion)**: $\text{AIC} = -2 \ln \hat{L} + 2p$.

---

### 💻 3. Python Code Implementation

```python
from sklearn.mixture import GaussianMixture

# Fit Gaussian Mixture Model with Full Covariance
gmm = GaussianMixture(n_components=4, covariance_type='full', max_iter=100, random_state=42)
gmm.fit(X)

labels_gmm = gmm.predict(X)
probs_gmm = gmm.predict_proba(X)  # Soft probability matrix (N, K)

print(f"[Day 061 GMM] Converged Iterations: {gmm.n_iter_}")
print(f"[Day 061 GMM] Lower Bound Log-Likelihood: {gmm.lower_bound_:.4f}")
print(f"[Day 061 GMM] Model BIC Score:           {gmm.bic(X):.4f}")
```
