# PROGIDY_WD_02
Customer Segmentation using K-Means Clustering
This project implements a K-Means clustering algorithm to group customers based on their purchasing behavior using the Mall Customers Dataset from Kaggle.

Key Steps:

Data Loading & Exploration – Loaded dataset, checked missing values, and explored feature distribution.

Feature Selection & Encoding – Selected relevant features such as Age, Annual Income, and Spending Score; encoded categorical features (Gender).

Feature Scaling – Standardized features for better clustering performance.

Choosing Optimal Clusters – Applied Elbow Method and Silhouette Score to determine the best k.

Model Training – Trained a K-Means clustering model with the chosen number of clusters.

Visualization – Used 2D scatter plots and PCA for visualizing customer clusters.

Export Results – Saved customer segmentation results to a CSV file for business use.

Technologies Used:

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Dataset Source: Mall Customers Dataset (Kaggle)
