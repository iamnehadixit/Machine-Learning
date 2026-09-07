import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------
# 1. Dataset
# ---------------------------------------

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([35, 42, 48, 55, 61, 68, 74, 81, 88, 94])


# ---------------------------------------
# 2. Train-Test Split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------
# 3. Create Models
# ---------------------------------------

ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=1.0)


# ---------------------------------------
# 4. Train Models
# ---------------------------------------

ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)


# ---------------------------------------
# 5. Predictions
# ---------------------------------------

ridge_pred = ridge_model.predict(X_test)
lasso_pred = lasso_model.predict(X_test)


# ---------------------------------------
# 6. Model Evaluation
# ---------------------------------------

ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_rmse = np.sqrt(ridge_mse)
ridge_r2 = r2_score(y_test, ridge_pred)

lasso_mae = mean_absolute_error(y_test, lasso_pred)
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_rmse = np.sqrt(lasso_mse)
lasso_r2 = r2_score(y_test, lasso_pred)


# ---------------------------------------
# 7. Results
# ---------------------------------------

print("----- Ridge Regression -----")
print("Coefficient:", ridge_model.coef_[0])
print("Intercept:", ridge_model.intercept_)
print("MAE:", ridge_mae)
print("MSE:", ridge_mse)
print("RMSE:", ridge_rmse)
print("R² Score:", ridge_r2)

print("\n----- Lasso Regression -----")
print("Coefficient:", lasso_model.coef_[0])
print("Intercept:", lasso_model.intercept_)
print("MAE:", lasso_mae)
print("MSE:", lasso_mse)
print("RMSE:", lasso_rmse)
print("R² Score:", lasso_r2)


# ---------------------------------------
# 8. Prediction for New Data
# ---------------------------------------

new_hours = np.array([[7]])

ridge_prediction = ridge_model.predict(new_hours)
lasso_prediction = lasso_model.predict(new_hours)

print("\n----- Prediction for 7 Hours -----")
print("Ridge Prediction:", ridge_prediction[0])
print("Lasso Prediction:", lasso_prediction[0])


# ---------------------------------------
# 9. Visualization
# ---------------------------------------

plt.scatter(X, y, label="Actual Data")

plt.plot(
    X,
    ridge_model.predict(X),
    label="Ridge Regression"
)

plt.plot(
    X,
    lasso_model.predict(X),
    label="Lasso Regression"
)

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Ridge vs Lasso Regression")
plt.legend()

plt.savefig(
    "ridge_lasso_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()