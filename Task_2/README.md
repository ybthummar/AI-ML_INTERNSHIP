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
