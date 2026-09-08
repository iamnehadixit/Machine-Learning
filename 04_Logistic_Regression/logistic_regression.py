import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------------------------------
# 1. Dataset
# ---------------------------------------

X = np.array([
    1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,
    5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10
]).reshape(-1, 1)

y = np.array([
    0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1
])


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
# 3. Create Model
# ---------------------------------------

model = LogisticRegression()


# ---------------------------------------
# 4. Train Model
# ---------------------------------------

model.fit(X_train, y_train)


# ---------------------------------------
# 5. Prediction
# ---------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------
# 6. Evaluation
# ---------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("----- Logistic Regression -----")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# ---------------------------------------
# 7. Confusion Matrix
# ---------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ---------------------------------------
# 8. Prediction for New Data
# ---------------------------------------

new_hours = np.array([[6]])

prediction = model.predict(new_hours)
probability = model.predict_proba(new_hours)

print("\n----- Prediction -----")
print("Hours Studied:", new_hours[0][0])
print("Predicted Class:", prediction[0])
print("Probability [Fail, Pass]:", probability[0])


# ---------------------------------------
# 9. Visualization
# ---------------------------------------

X_plot = np.linspace(1, 10, 100).reshape(-1, 1)

probabilities = model.predict_proba(X_plot)[:, 1]

plt.scatter(X, y, label="Actual Data")
plt.plot(
    X_plot,
    probabilities,
    label="Logistic Regression"
)

plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression - Pass/Fail")
plt.legend()

plt.savefig(
    "logistic_regression_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()