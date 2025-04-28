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

