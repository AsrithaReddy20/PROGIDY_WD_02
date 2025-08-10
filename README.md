# PROGIDY_WD_02
# kmeans_customer_segmentation.py
"""
Customer Segmentation using K-Means
Save this file and run: python kmeans_customer_segmentation.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# -------------------------
# 1) Load dataset
# -------------------------
FNAME = "Mall_Customers.csv"  # change if your file name is different
if not os.path.isfile(FNAME):
    raise FileNotFoundError(f"Place the dataset file named '{FNAME}' in this folder or update FNAME.")

df = pd.read_csv(FNAME)
print("Loaded dataset: shape =", df.shape)
print("Columns:", list(df.columns))
print("\nSample rows:")
print(df.head())

# -------------------------
# 2) Quick cleaning & rename
# -------------------------
# Rename common Kaggle column names to simple names if they exist
col_map = {}
if 'Annual Income (k$)' in df.columns:
    col_map['Annual Income (k$)'] = 'AnnualIncome'
if 'Spending Score (1-100)' in df.columns:
    col_map['Spending Score (1-100)'] = 'SpendingScore'
if 'CustomerID' in df.columns:
    col_map['CustomerID'] = 'CustomerID'
if 'Age' in df.columns:
    col_map['Age'] = 'Age'
if 'Gender' in df.columns:
    col_map['Gender'] = 'Gender'

df = df.rename(columns=col_map)
print("\nRenamed columns (if matched):", list(df.columns))

# Drop any completely empty columns and duplicates
df = df.dropna(how='all').drop_duplicates()
print("After dropna/drop_duplicates: shape =", df.shape)

# -------------------------
# 3) Feature selection
# Option A (visual, 2D): AnnualIncome & SpendingScore
# Option B (multi-feature): Age, AnnualIncome, SpendingScore, Gender(encoded)
# -------------------------
# Check availability:
use_2d = ('AnnualIncome' in df.columns) and ('SpendingScore' in df.columns)

if not use_2d:
    raise ValueError("This script expects 'Annual Income (k$)' and 'Spending Score (1-100)' (or renamed) in the CSV.")

# Create a multi-feature dataset (if possible)
features = []
if 'Age' in df.columns:
    features.append('Age')
features += ['AnnualIncome', 'SpendingScore']

df_model = df.copy()

# Encode Gender if present
if 'Gender' in df_model.columns:
    le = LabelEncoder()
    df_model['GenderEncoded'] = le.fit_transform(df_model['Gender'])
    features.append('GenderEncoded')

print("\nFeatures used for clustering:", features)

X = df_model[features].values

# -------------------------
# 4) Scale features
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# 5) Elbow method (inertia) to pick k
# -------------------------
inertia = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, inertia, 'bx-')
plt.xlabel('k (number of clusters)')
plt.ylabel('Inertia (sum of squared distances)')
plt.title('Elbow Method for optimal k')
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# 6) Silhouette scores for k = 2..10
# -------------------------
print("\nSilhouette scores:")
sil_scores = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores[k] = sil
    print(f" k = {k}: silhouette = {sil:.4f}")

# You should pick k by combining elbow (inertia) + silhouette score.
# For Mall dataset typical results often show 4-6 clusters as sensible.

# -------------------------
# 7) Fit final KMeans (set chosen_k based on above)
# -------------------------
chosen_k = max(sil_scores, key=sil_scores.get)  # simple auto-pick: best silhouette
print(f"\nAuto-selected k (best silhouette): {chosen_k}")

kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df_model['Cluster'] = labels

# -------------------------
# 8) Inspect cluster counts and centers (in original scale)
# -------------------------
print("\nCluster counts:")
print(df_model['Cluster'].value_counts().sort_index())

centers_scaled = kmeans.cluster_centers_
centers_orig = scaler.inverse_transform(centers_scaled)
centers_df = pd.DataFrame(centers_orig, columns=features)
centers_df.index.name = 'Cluster'
print("\nCluster centers (original scale):")
print(centers_df)

# Show aggregated stats per cluster
print("\nCluster profiles (mean of features):")
print(df_model.groupby('Cluster')[features].mean().round(2))

# -------------------------
# 9) Visualization
# 9a) If you used AnnualIncome & SpendingScore, plot 2D clusters for interpretability
# -------------------------
if 'AnnualIncome' in df_model.columns and 'SpendingScore' in df_model.columns:
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df_model, x='AnnualIncome', y='SpendingScore', hue='Cluster', palette='tab10', s=60)
    # plot centers (project centers to these two columns)
    if set(['AnnualIncome','SpendingScore']).issubset(features):
        # find index of those two features in the features list
        idx_income = features.index('AnnualIncome')
        idx_spend = features.index('SpendingScore')
        centers_2d = centers_orig[:, [idx_income, idx_spend]]
        plt.scatter(centers_2d[:,0], centers_2d[:,1], c='black', s=200, marker='X', label='Centers')
        plt.legend()
    plt.title(f'Clusters (k={chosen_k}) on AnnualIncome vs SpendingScore')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------
# 9b) PCA projection (for multi-dim features)
# -------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette='tab10', s=60)
plt.title('Clusters visualized on PCA (2 components)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# 10) Save results
# -------------------------
OUT = "customers_with_clusters.csv"
df_result = df.copy()
df_result['Cluster'] = df_model['Cluster']
df_result.to_csv(OUT, index=False)
print(f"\nSaved results to {OUT}")

# End of script
