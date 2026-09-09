# ============================================================
# K-Nearest Neighbors (KNN) Classification
# Dataset: Iris Dataset
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

print("----- Dataset Information -----")
print("Number of Samples:", X.shape[0])
print("Number of Features:", X.shape[1])
print("Classes:", iris.target_names)


# ------------------------------------------------------------
# 2. Train-Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 3. Feature Scaling
# ------------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ------------------------------------------------------------
# 4. Find Best K using Cross-Validation
# ------------------------------------------------------------

k_values = range(1, 21)
cv_scores = []

for k in k_values:

    model = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    cv_scores.append(scores.mean())


best_k = k_values[np.argmax(cv_scores)]
best_cv_score = max(cv_scores)

print("\n----- Best K Selection -----")
print("Best K:", best_k)
print("Best Cross-Validation Accuracy:", round(best_cv_score, 4))


# ------------------------------------------------------------
# 5. Train Final KNN Model
# ------------------------------------------------------------

model = KNeighborsClassifier(
    n_neighbors=best_k
)

model.fit(X_train, y_train)


# ------------------------------------------------------------
# 6. Prediction
# ------------------------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------------------------
# 7. Model Evaluation
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n----- KNN Model Evaluation -----")
print("Accuracy:", round(accuracy, 4))

print("\n----- Confusion Matrix -----")
print(confusion_matrix(y_test, y_pred))

print("\n----- Classification Report -----")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# ------------------------------------------------------------
# 8. Predict a New Flower
# ------------------------------------------------------------

new_flower = np.array([
    [5.1, 3.5, 1.4, 0.2]
])

new_flower_scaled = scaler.transform(new_flower)

prediction = model.predict(new_flower_scaled)

print("\n----- New Flower Prediction -----")
print("Predicted Class:", iris.target_names[prediction[0]])


# ------------------------------------------------------------
# 9. Plot Accuracy vs K
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    cv_scores,
    marker="o"
)

plt.xlabel("Value of K")
plt.ylabel("Cross-Validation Accuracy")
plt.title("KNN: Accuracy vs K")
plt.xticks(list(k_values))
plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 10. Display Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    range(len(iris.target_names)),
    iris.target_names
)

plt.yticks(
    range(len(iris.target_names)),
    iris.target_names
)

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.show()