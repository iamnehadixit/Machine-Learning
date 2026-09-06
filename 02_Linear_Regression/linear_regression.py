import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------------
# 1. Dataset
# ---------------------------------------

# Hours studied and corresponding marks
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
# 3. Create Linear Regression Model
# ---------------------------------------

model = LinearRegression()

# ---------------------------------------
# 4. Train Model
# ---------------------------------------

model.fit(X_train, y_train)

# ---------------------------------------
# 5. Prediction
# ---------------------------------------

y_pred = model.predict(X_test)

# ---------------------------------------
# 6. Model Parameters
# ---------------------------------------

print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# ---------------------------------------
# 7. Model Evaluation
# ---------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# ---------------------------------------
# 8. Prediction for New Data
# ---------------------------------------

hours = np.array([[7]])

prediction = model.predict(hours)

print("\nPrediction:")
print("Hours Studied:", hours[0][0])
print("Predicted Marks:", prediction[0])

# ---------------------------------------
# 9. Visualization
# ---------------------------------------

plt.scatter(X, y, label="Actual Data")
plt.plot(X, model.predict(X), label="Regression Line")

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression - Hours vs Marks")
plt.legend()

# Save figure
plt.savefig("linear_regression_plot.png", dpi=300, bbox_inches="tight")

plt.show()