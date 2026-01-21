# ================================
# Advertising Dataset – ML Models
# ================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# Classification models
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import mean_squared_error, accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("Advertising.csv")

# Features
X = data[['TV', 'Radio', 'Newspaper']]

# Target for regression
y_reg = data['Sales']

# Target for classification
y_clf = (data['Sales'] >= data['Sales'].mean()).astype(int)

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42)

_, _, y_clf_train, y_clf_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42)

# -----------------------------
# Feature Scaling (for KNN)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================
# REGRESSION MODELS
# =============================

print("\n--- REGRESSION RESULTS ---")

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_reg_train)
lr_pred = lr.predict(X_test)
print("Linear Regression MSE:", mean_squared_error(y_reg_test, lr_pred))

# 2. Multiple Linear Regression (same model, multiple features)
mlr = LinearRegression()
mlr.fit(X_train, y_reg_train)
mlr_pred = mlr.predict(X_test)
print("Multiple Linear Regression MSE:", mean_squared_error(y_reg_test, mlr_pred))

# 3. KNN Regression
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_scaled, y_reg_train)
knn_reg_pred = knn_reg.predict(X_test_scaled)
print("KNN Regression MSE:", mean_squared_error(y_reg_test, knn_reg_pred))

# 4. Decision Tree Regression
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X_train, y_reg_train)
dt_reg_pred = dt_reg.predict(X_test)
print("Decision Tree Regression MSE:", mean_squared_error(y_reg_test, dt_reg_pred))

# =============================
# CLASSIFICATION MODELS
# =============================

print("\n--- CLASSIFICATION RESULTS ---")

# 5. Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_clf_train)
nb_pred = nb.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_clf_test, nb_pred))

# 6. KNN Classification
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_scaled, y_clf_train)
knn_clf_pred = knn_clf.predict(X_test_scaled)
print("KNN Classification Accuracy:", accuracy_score(y_clf_test, knn_clf_pred))

# 7. Decision Tree Classification
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_clf_train)
dt_clf_pred = dt_clf.predict(X_test)
print("Decision Tree Classification Accuracy:", accuracy_score(y_clf_test, dt_clf_pred))
