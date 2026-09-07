# Ridge and Lasso Regression

## 📌 Overview
Ridge and Lasso Regression are regularized versions of Linear Regression used to reduce overfitting and control model complexity.

## 🎯 Objective
Compare Ridge and Lasso Regression for predicting student marks based on hours studied.

## 🛠️ Technologies
- Python
- NumPy
- Matplotlib
- Scikit-learn

## 🔄 Workflow
1. Load dataset
2. Split data into training and testing sets
3. Train Ridge and Lasso models
4. Make predictions
5. Evaluate models
6. Compare regression lines

## 📊 Results

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Ridge | 0.50 | 0.678 | 0.9991 |
| Lasso | 0.552 | 0.745 | 0.9990 |

## 📈 Visualization
The plot compares the Ridge and Lasso regression lines with the actual data points.

## 📚 Key Learning
- Ridge uses L2 regularization.
- Lasso uses L1 regularization.
- `alpha` controls regularization strength.
- Ridge and Lasso help reduce overfitting.