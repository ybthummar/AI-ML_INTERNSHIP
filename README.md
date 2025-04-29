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


## Author
Yug Thummar  
AI & ML Internship Participant  

