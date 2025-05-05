# AI & ML Internship - Task 1 Data Cleaning & Preprocessing using Titanic Dataset

### Objective
Learn how to clean and prepare raw data for machine learning by handling missing values, encoding categorical variables, standardizing features, and removing outliers.

---

### Dataset
- **Name:** Titanic Dataset
- **Source:** [Click here to download](https://www.kaggle.com/datasets/brendan45774/titanic)
- **File used:** `titanic.csv`

---

###  Tools & Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (LabelEncoder, StandardScaler)

---

###  Steps Performed

#### 1. Importing and Exploring Dataset
- Loaded Titanic dataset and checked data types, null values, and general info using `df.info()` and `df.head()`.

#### 2. Handling Missing Values
- Replaced missing values in **Age** column with the **mean**.
- Dropped **Cabin** column due to excessive missing data.
- Filled missing values in **Embarked** with **mode**.

#### 3. Encoding Categorical Features
- Used **Label Encoding** for `Sex`.
- Applied **One-Hot Encoding** for `Embarked` with `drop_first=True` to avoid dummy variable trap.

#### 4. Feature Scaling
- Applied **StandardScaler** on numerical features: `Age` and `Fare` for normalization.

#### 5. Outlier Detection and Removal
- Used **Boxplots** to visualize outliers in `Age` and `Fare`.
- Removed outliers from `Fare` and `Age` using **IQR (Interquartile Range)** method.

---

###  Visualizations
- Boxplots before and after removing outliers to understand data distribution and extreme values.

---
# AI & ML Internship - Task 2: Exploratory Data Analysis (EDA)

## Objective
Perform Exploratory Data Analysis (EDA) to understand the dataset using statistical summaries and visualizations.

## Dataset
- **Name**: Iris Flower Dataset
- **Source**: Preloaded via Seaborn library
- **Features**: 
  - `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `species`

## Tools Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Plotly (optional)

## What I Did

1. **Loaded the dataset** using `seaborn.load_dataset("iris")`
2. **Generated Summary Statistics** using `.describe()` and `.info()`
3. **Visualized Distributions**:
   - Histograms
   - Boxplots
   - Violin plots
   - Swarm plots
   - KDE (Density plots)
4. **Explored Relationships**:
   - Pairplot for feature interactions
   - Correlation matrix for numeric features
   - Line fit plots using `lmplot`
5. **Species-wise Correlation Heatmaps**:
   - Created separate heatmaps for each species to analyze internal feature correlation
6. **Feature Engineering**:
   - Introduced a new feature: `petal_area = petal_length * petal_width`
7. **Outlier Detection** using the IQR method

## Inferences

- **Petal-based features** are most important in separating species.
- **Setosa** is clearly distinguishable from others.
- **Versicolor** and **Virginica** show partial overlap in feature space.
- **Petal length and width** have strong correlation, especially in Virginica.
- **Setosa** has weak correlation among features, unlike the other two classes.
- No missing values or major anomalies found in the dataset.


## Learning Outcomes

- Understood how to interpret data using visual and statistical methods.
- Gained insight into feature relationships and data patterns.
- Learned how to break down data by class for deeper EDA.
- Practiced multiple visualization tools like heatmaps, KDE, and violin plots.

# Task 3 - Linear Regression | AI & ML Internship

## Objective:
- Understand and implement Simple and Multiple Linear Regression.

## Tools Used:
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Dataset:
- House Price Prediction Dataset
- Features: Area, Bedrooms, Bathrooms, Stories, etc.
- Target: House Price

## Steps Completed:

### 1. Data Preprocessing
- Loaded the dataset
- Converted categorical variables using One-Hot Encoding

### 2. Simple Linear Regression
- Used only 'area' as the independent variable
- Trained a linear model
- Evaluated using MAE, MSE, and R² Score
- Plotted Regression Line

### 3. Multiple Linear Regression
- Used all independent variables
- Trained a multiple linear regression model
- Evaluated using MAE, MSE, and R² Score
- Interpreted model coefficients
- Plotted Actual vs Predicted Prices
- Plotted Residuals Distribution

## Evaluation Metrics:
| Model                     | MAE     | MSE       | R² Score |
|----------------------------|---------|-----------|----------|
| Simple Linear Regression   | ~430000 | ~3.2e+11  | ~0.63    |
| Multiple Linear Regression | ~380000 | ~2.5e+11  | ~0.72    |

# Task 4: Classification with Logistic Regression

## Objective
The objective of this task is to build a binary classifier using Logistic Regression to predict whether a tumor is malignant or benign using the Breast Cancer Wisconsin dataset.

## Tools and Libraries Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Dataset
The Breast Cancer Wisconsin dataset, available via Scikit-learn, is used in this task. It consists of various features derived from digitized images of breast mass samples.

- Target classes:
  - 0 = Malignant
  - 1 = Benign
- Features include mean, standard error, and worst measurements of radius, texture, perimeter, area, etc.

## Workflow Summary

### 1. Data Loading
The dataset was loaded using `sklearn.datasets.load_breast_cancer()` and converted into Pandas DataFrames.

### 2. Data Splitting
Data was split into training (80%) and testing (20%) sets using `train_test_split()`.

### 3. Feature Scaling
Features were standardized using `StandardScaler` to improve the performance and convergence of the logistic regression model.

### 4. Model Training
A logistic regression model was trained using `LogisticRegression()` from Scikit-learn.

### 5. Model Evaluation
Predictions were made on the test set and evaluated using:
- Confusion matrix
- Classification report (precision, recall, F1-score)
- ROC curve and AUC score

### 6. Threshold Tuning and Sigmoid Explanation
The effect of different classification thresholds was explored. The sigmoid function, which transforms linear combinations into probabilities, was also explained and visualized.

## Evaluation Metrics

### Confusion Matrix
Summarizes the number of true positives, true negatives, false positives, and false negatives.

### Classification Report
Provides precision, recall, F1-score, and support for each class.

### ROC-AUC Curve
The ROC curve plots the true positive rate against the false positive rate. The AUC (Area Under the Curve) indicates the overall performance of the model. An AUC close to 1.0 represents a highly accurate model.

## Key Concepts and Questions

| Concept | Explanation |
|--------|-------------|
| Logistic vs Linear Regression | Logistic regression is used for classification and outputs probabilities using the sigmoid function, while linear regression predicts continuous values. |
| Sigmoid Function | A mathematical function that maps any real-valued number into the (0,1) interval, used to predict probabilities. |
| Precision vs Recall | Precision measures how many of the predicted positives are actual positives, while recall measures how many actual positives were correctly predicted. |
| ROC-AUC | Receiver Operating Characteristic curve and its area under the curve help visualize and quantify model performance. |
| Confusion Matrix | A table used to evaluate the performance of a classification algorithm. |
| Class Imbalance | An unequal distribution of classes can bias predictions. Strategies include resampling or using performance metrics like F1-score. |
| Threshold Selection | Choosing a probability threshold affects classification decisions and should be based on the application’s tolerance for false positives/negatives. |
| Multiclass Logistic Regression | Logistic regression can be extended to multiclass problems using strategies like One-vs-Rest or multinomial logistic regression. |

# Task 5: Decision Trees and Random Forests - AI & ML Internship

This project is a part of the AI & ML Internship Program and focuses on building classification models using Decision Trees and Random Forests. The dataset used is related to heart disease prediction, where the goal is to classify whether a patient has heart disease based on several clinical features.

## Objective

- Build and evaluate a Decision Tree classifier.
- Visualize the decision-making process of the tree.
- Train a Random Forest classifier and compare its performance.
- Analyze feature importance.
- Evaluate model stability using cross-validation.

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Graphviz

## Dataset

We use the Heart Disease dataset, which includes 14 attributes such as age, sex, chest pain type, cholesterol, resting blood pressure, and more. The target column indicates the presence (1) or absence (0) of heart disease.

You can find a similar dataset here:  
[Kaggle - Heart Disease UCI](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

## Methodology

1. **Data Preprocessing**: Load the dataset, check for missing values, and split it into features and target.
2. **Train-Test Split**: Use 80-20 ratio for splitting data into training and test sets.
3. **Decision Tree Classifier**:
   - Train with controlled depth to prevent overfitting.
   - Visualize the tree using Scikit-learn's plotting and Graphviz.
   - Evaluate using confusion matrix, accuracy, and classification report.
4. **Random Forest Classifier**:
   - Train using an ensemble of 100 trees.
   - Evaluate similarly and compare with the Decision Tree.
   - Use feature importance to interpret the model.
5. **Cross-Validation**: Perform 5-fold cross-validation to assess model robustness.

## Results

| Model           | Accuracy (Test Set) | Cross-Validation Accuracy (5-Fold) |
|----------------|---------------------|------------------------------------|
| Decision Tree  | 80%  | 83.41%                 |
| Random Forest  | 98.53%  | 99.70%                 |

The Random Forest model typically shows higher accuracy and more consistent performance across folds, indicating better generalization.

---

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

---
# Task 7: Support Vector Machines (SVM)

## Objective
The goal of this task was to apply Support Vector Machines (SVMs) for binary classification using both **Linear** and **RBF (Radial Basis Function)** kernels. The task involved training, visualizing, tuning, and evaluating SVM models.

## Tools Used
- Python
- Scikit-learn
- NumPy
- Matplotlib

## Dataset
We used the **Breast Cancer Wisconsin dataset**, which is commonly used for binary classification tasks. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

## Steps Performed

1. **Data Loading and Preprocessing**
   - Loaded dataset using `sklearn.datasets.load_breast_cancer`.
   - Normalized features using `StandardScaler`.
   - Split the data into training and testing sets.

2. **Model Training**
   - Trained two SVM models:
     - SVM with Linear Kernel
     - SVM with RBF Kernel

3. **Visualization**
   - Plotted decision boundaries for both kernels using the first two principal components (2D).

4. **Evaluation**
   - Evaluated model performance using accuracy, precision, recall, and f1-score.
   - Compared metrics of both kernels.

5. **Hyperparameter Tuning**
   - Used `GridSearchCV` to tune hyperparameters `C` and `gamma` for the RBF kernel.

6. **Cross-Validation**
   - Applied 5-fold cross-validation to ensure generalization of the models.

## Results

### Linear Kernel
- **Accuracy**: 90%
- **Precision**: 0.91 (class 0), 0.90 (class 1)
- **Recall**: 0.81 (class 0), 0.95 (class 1)
- **F1-score**: 0.86 (class 0), 0.92 (class 1)

### RBF Kernel
- **Accuracy**: 91%
- **Precision**: 0.91 (class 0), 0.90 (class 1)
- **Recall**: 0.83 (class 0), 0.95 (class 1)
- **F1-score**: 0.87 (class 0), 0.93 (class 1)

## Conclusion

In this task, we successfully implemented Support Vector Machines (SVM) for binary classification using both Linear and RBF kernels. The RBF kernel slightly outperformed the linear one in terms of accuracy and recall.

### Key Takeaways
- Linear SVM performed well for linearly separable data.
- RBF Kernel captured more complex patterns and provided slightly better performance.
- Visualizing decision boundaries helped in understanding the model’s classification strategy.
- GridSearchCV and cross-validation were effective for optimizing and validating the model.
- We gained a practical understanding of margin maximization, kernel trick, and hyperparameter tuning in SVM.

## References
- [Scikit-learn SVM Documentation](https://scikit-learn.org/stable/modules/svm.html)
- [Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)


## Author
Yug Thummar  
AI & ML Internship Participant  

