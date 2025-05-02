# K-Nearest Neighbors (KNN) Classification – AI & ML Internship Task 6

This project implements the K-Nearest Neighbors (KNN) classification algorithm on the Iris dataset using Python. The goal is to understand how instance-based learning works, how distance metrics affect predictions, and how model performance varies with the number of neighbors (K). Bonus enhancements such as PCA visualization, distance metric comparison, and model persistence are also included.

---

## Problem Statement

Implement the KNN algorithm for multi-class classification, visualize how the algorithm behaves for different values of K, tune hyperparameters, and evaluate model performance using accuracy and confusion matrix. The Iris dataset is used for this classification task.

---

## Tools & Libraries Used

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib (for model saving)

---

## Dataset

**Dataset**: Iris Dataset  
**Source**: UCI Machine Learning Repository  
**Features**:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width  

**Target Classes**:
- Setosa
- Versicolor
- Virginica

---

## Implementation Steps

### 1. Data Preprocessing
- Loaded Iris dataset using Scikit-learn
- Normalized the features using `StandardScaler`
- Split the data into training and testing sets (80/20)

### 2. Model Training & Evaluation
- Trained `KNeighborsClassifier` from Scikit-learn
- Evaluated using:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
- Compared different values of K from 1 to 20
- Selected the optimal K with the highest test accuracy

### 3. Hyperparameter Tuning (Grid Search)
- Used `GridSearchCV` to tune:
  - `n_neighbors`
  - `metric` (Euclidean, Manhattan, Minkowski)

### 4. Visualization
- Plotted accuracy vs. K graph
- Plotted confusion matrix using Seaborn
- Used PCA to reduce features to 2D for visualization
- Plotted decision boundary using top 2 features

### 5. Advanced Add-Ons
- Evaluated using Cross-Validation
- Compared distance metrics (Euclidean vs. Manhattan)
- Saved the model using Joblib
- Reloaded and used the model for prediction

---

## Results

| Experiment                      | Output                   |
|--------------------------------|---------------------------|
| Optimal K                      | 3                         |
| Test Set Accuracy              | 100%                      |
| Best Distance Metric           | Euclidean                 |
| Cross-Validation Accuracy      | ~96.6% average            |
| Logistic Regression Accuracy   | ~96.6% (for comparison)   |

---

## Key Learnings

- KNN is an instance-based algorithm, meaning no training phase—just storage of the dataset.
- Distance metrics (e.g., Euclidean, Manhattan) directly influence prediction performance.
- Normalizing data is crucial in distance-based models.
- Choosing the right K value is a trade-off between underfitting and overfitting.
- KNN is sensitive to noisy data and irrelevant features.

