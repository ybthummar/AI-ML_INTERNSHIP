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

