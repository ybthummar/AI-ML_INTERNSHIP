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
