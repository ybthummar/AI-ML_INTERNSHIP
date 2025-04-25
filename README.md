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

###  Final Results
- Cleaned Dataset Shape: `(741, 10)` *(after dropping outliers and cleaning)*
- No null values remain.
- Categorical features are encoded.
- Numerical features are standardized and outliers are handled.

---

###  What I Learned
- Difference between **mean, mode imputation** and when to apply each.
- How to handle **categorical variables** with **Label Encoding** and **One-Hot Encoding**.
- Difference between **Normalization vs Standardization**.
- How to use **IQR** for outlier detection.
- Importance of **data preprocessing** in improving ML model performance.

---

# AI & ML Internship – Task 2: Exploratory Data Analysis (EDA)

## Objective
Perform Exploratory Data Analysis (EDA) using statistical summaries and visualizations to understand the dataset and identify patterns or anomalies.

## Dataset Overview
- **Dataset**: Iris Flower Dataset  
- **Source**: Loaded using Seaborn library  
- **Features**:  
  - sepal_length  
  - sepal_width  
  - petal_length  
  - petal_width  
  - species (target variable)

## Tools and Libraries Used
- Python 3.8  
- Pandas  
- Matplotlib  
- Seaborn  
- Plotly (optional)

## EDA Process and Analysis

### 1. Dataset Loading
- Loaded the Iris dataset using `sns.load_dataset('iris')`.

### 2. Summary Statistics
- Used `.describe()` and `.info()` to view data types, missing values, and statistical summaries.
- Confirmed that there are no missing or null values in the dataset.

### 3. Visualizations
- Created the following plots:
  - Histograms to view distributions of each numerical feature
  - Boxplots to identify outliers and understand feature spread
  - Pairplot to explore relationships between features by species
  - Correlation heatmap to identify highly correlated features

### 4. Key Insights
- Petal length and petal width are highly positively correlated.
- Setosa species is clearly separable from the others in the dataset.
- Sepal width shows more variability compared to other features.
- Classes are balanced with 50 instances each.

## Learning Outcomes
- Learned how to generate summary statistics and interpret them.
- Understood how to use visualizations to analyze and compare features.
- Practiced identifying patterns, trends, and feature relationships that could help in model building.
  
## Author
Yug Thummar  
AI & ML Internship Participant  

