# Task 8: Clustering with K-Means

This project demonstrates how to use K-Means Clustering, an unsupervised learning algorithm, to segment mall customers based on their demographic and spending behavior. The goal is to identify distinct customer groups for potential targeted marketing strategies.

---

## Objective

- Apply K-Means Clustering on a customer dataset.
- Use the Elbow Method to find the optimal number of clusters.
- Evaluate clustering quality using the Silhouette Score.
- Visualize clusters with PCA and other plots for meaningful interpretation.

---

## Dataset

**Dataset Name:** Mall Customer Segmentation Data  
**Source:** Public dataset from Kaggle  
**Features Used:**
- Gender (encoded as numeric)
- Age
- Annual Income (k$)
- Spending Score (1-100)

---

## Tools and Libraries Used

- Python 3.8
- Pandas – Data manipulation
- Matplotlib & Seaborn – Data visualization
- Scikit-learn – KMeans, PCA, StandardScaler, Silhouette Score

---

## How It Works

### 1. Load and Preprocess Data
- Encoded Gender as 0 (Male) and 1 (Female).
- Selected only relevant numerical features.
- Standardized the data using StandardScaler.

### 2. Elbow Method
- Plotted inertia for k=1 to 10 clusters.
- Chose the optimal value of k where the "elbow" is observed, typically k = 5.

### 3. Apply K-Means
- Fitted KMeans model with the optimal k.
- Assigned cluster labels to each customer.

### 4. PCA for 2D Visualization
- Reduced dimensions from 3D to 2D using PCA.
- Visualized clusters in 2D space using scatter plots.

### 5. Evaluation
- Used Silhouette Score to assess clustering performance.
- Score near 0.55 indicates moderate separation between clusters.

---

## Visualizations

| Plot | Description |
|------|-------------|
| Elbow Plot | Helps determine the optimal number of clusters |
| PCA Scatter Plot | Visual representation of customer segments |
| Count Plot | Number of customers in each cluster |
| Bar Plot | Average spending score by cluster |
| Age vs Income | Cluster-wise scatter plot of Age vs Annual Income |


