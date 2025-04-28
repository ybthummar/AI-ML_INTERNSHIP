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


## Author
Yug Thummar  
AI & ML Internship Participant  

