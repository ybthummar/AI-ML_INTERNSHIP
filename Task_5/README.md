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
