"""
Machine Learning Practice: Regression & Classification
Session 8: November 3, 2025
Anna Smirnova

Combined exercise covering:
- Linear Regression (Week 7)
- Classification (Week 8)

You'll work with real datasets and implement both regression and classification models!
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# ============================================================================
# PART 1: LINEAR REGRESSION - Predicting House Prices
# ============================================================================

print("="*60)
print("PART 1: LINEAR REGRESSION - House Price Prediction")
print("="*60)

# Sample housing data (size in sq ft, bedrooms, age in years, price in $1000s)
housing_data = {
    'size': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    'bedrooms': [3, 3, 2, 4, 2, 3, 4, 4, 3, 3],
    'age': [10, 8, 15, 5, 20, 7, 3, 2, 12, 9],
    'price': [245, 312, 279, 308, 199, 219, 405, 324, 319, 255]
}

df_housing = pd.DataFrame(housing_data)

# TODO 1.1: Create features (X) and target (y)
# HINT: X should have 'size', 'bedrooms', 'age' columns
# HINT: y should be the 'price' column
X_housing = None  # TODO: Extract features
y_housing = None  # TODO: Extract target

# TODO 1.2: Split data into training and testing sets
# HINT: Use train_test_split with test_size=0.2, random_state=42
X_train, X_test, y_train, y_test = None, None, None, None  # TODO

# TODO 1.3: Create and train a Linear Regression model
# HINT: model = LinearRegression()
# HINT: model.fit(X_train, y_train)
model_regression = None  # TODO: Create model
# TODO: Train the model

# TODO 1.4: Make predictions on the test set
y_pred = None  # TODO: Predict on X_test

# TODO 1.5: Evaluate the model
# Calculate Mean Squared Error (MSE) and R² score
mse = None  # TODO: Use mean_squared_error(y_test, y_pred)
r2 = None  # TODO: Use r2_score(y_test, y_pred)

print("\nRegression Results:")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# TODO 1.6: Test your model with a new house
# A house with: 1800 sq ft, 3 bedrooms, 6 years old
new_house = [[1800, 3, 6]]
predicted_price = None  # TODO: Predict price for new_house
print(f"\nPredicted price for new house: ${predicted_price}")


# ============================================================================
# PART 2: CLASSIFICATION - Iris Flower Species
# ============================================================================

print("\n" + "="*60)
print("PART 2: CLASSIFICATION - Iris Species Recognition")
print("="*60)

# Sample Iris data (sepal_length, sepal_width, petal_length, petal_width, species)
# Species: 0=Setosa, 1=Versicolor, 2=Virginica
iris_data = {
    'sepal_length': [5.1, 4.9, 4.7, 7.0, 6.4, 6.9, 6.3, 5.8, 6.7, 5.7],
    'sepal_width': [3.5, 3.0, 3.2, 3.2, 3.2, 3.1, 2.3, 2.7, 3.3, 2.8],
    'petal_length': [1.4, 1.4, 1.3, 4.7, 4.5, 4.9, 4.4, 3.9, 5.7, 4.1],
    'petal_width': [0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 1.3, 1.2, 2.5, 1.3],
    'species': [0, 0, 0, 1, 1, 1, 1, 1, 2, 1]
}

df_iris = pd.DataFrame(iris_data)

# TODO 2.1: Create features (X) and target (y)
# HINT: X should have all columns except 'species'
# HINT: y should be the 'species' column
X_iris = None  # TODO: Extract features
y_iris = None  # TODO: Extract target

# TODO 2.2: Split data into training and testing sets
X_train_iris, X_test_iris, y_train_iris, y_test_iris = None, None, None, None  # TODO

# TODO 2.3: Create and train a K-Nearest Neighbors classifier
# HINT: Use n_neighbors=3
knn_model = None  # TODO: Create KNeighborsClassifier
# TODO: Train the model

# TODO 2.4: Make predictions
y_pred_knn = None  # TODO: Predict on X_test_iris

# TODO 2.5: Evaluate the KNN model
accuracy_knn = None  # TODO: Use accuracy_score(y_test_iris, y_pred_knn)
print(f"\nK-NN Accuracy: {accuracy_knn:.2%}")

# TODO 2.6: Create and train a Decision Tree classifier
tree_model = None  # TODO: Create DecisionTreeClassifier
# TODO: Train the model

# TODO 2.7: Make predictions with Decision Tree
y_pred_tree = None  # TODO: Predict on X_test_iris

# TODO 2.8: Evaluate the Decision Tree model
accuracy_tree = None  # TODO: Calculate accuracy
print(f"Decision Tree Accuracy: {accuracy_tree:.2%}")

# TODO 2.9: Compare models - which one performed better?
print("\nModel Comparison:")
print(f"K-NN: {accuracy_knn:.2%}")
print(f"Decision Tree: {accuracy_tree:.2%}")
# TODO: Print which model is better

# TODO 2.10: Create a confusion matrix for the better model
# HINT: Use confusion_matrix(y_test_iris, y_pred_better)
cm = None  # TODO: Create confusion matrix
print("\nConfusion Matrix:")
print(cm)


# ============================================================================
# PART 3: MINI PROJECT - Student Grade Prediction
# ============================================================================

print("\n" + "="*60)
print("PART 3: MINI PROJECT - Predict Student Pass/Fail")
print("="*60)

# Student data (study_hours, previous_score, attendance%, passed)
# passed: 0=Failed, 1=Passed
student_data = {
    'study_hours': [2, 4, 1, 5, 3, 6, 2, 7, 4, 1],
    'previous_score': [55, 67, 45, 88, 62, 91, 58, 95, 70, 40],
    'attendance': [60, 75, 55, 90, 70, 95, 65, 98, 80, 50],
    'passed': [0, 1, 0, 1, 1, 1, 0, 1, 1, 0]
}

df_students = pd.DataFrame(student_data)

# TODO 3.1: Split features and target
X_students = None  # TODO
y_students = None  # TODO

# TODO 3.2: Split into train/test sets
X_train_s, X_test_s, y_train_s, y_test_s = None, None, None, None  # TODO

# TODO 3.3: Try BOTH K-NN and Decision Tree
# Which one works better for this dataset?

# K-NN Model
knn_student = None  # TODO: Create and train
# TODO: Predict and evaluate

# Decision Tree Model
tree_student = None  # TODO: Create and train
# TODO: Predict and evaluate

# TODO 3.4: Predict for a new student
# Study hours: 5, Previous score: 75, Attendance: 85%
new_student = [[5, 75, 85]]
# TODO: Predict if they will pass or fail
# TODO: Print the prediction


# ============================================================================
# BONUS CHALLENGES - Level 1 (Recommended)
# ============================================================================

print("\n" + "="*60)
print("BONUS CHALLENGES - LEVEL 1")
print("="*60)

# BONUS 1: Feature Importance - What Actually Matters?
# TODO: For the decision tree model, check which features are most important
# HINT: tree_model.feature_importances_ gives importance scores
# HINT: Plot a bar chart showing feature names vs importance
# Question: Is petal_length more important than sepal_width for iris classification?

# BONUS 2: Residual Analysis - Is Your Model Systematically Wrong?
# TODO: For housing regression, plot residuals (actual - predicted) vs predicted values
# HINT: residuals = y_test - y_pred
# HINT: plt.scatter(y_pred, residuals); plt.axhline(y=0, color='r', linestyle='--')
# Good model: Residuals randomly scattered around 0
# Bad model: Clear pattern in residuals (e.g., all positive for high prices)

# BONUS 3: The KNN Scaling Mystery
# TODO: Train KNN on housing data WITHOUT scaling features
# TODO: Then train again WITH StandardScaler
# HINT: from sklearn.preprocessing import StandardScaler
# Question: Why does accuracy change dramatically?
# Reason: KNN uses distance - a 1000 sq ft difference dominates a 1 bedroom difference!

# BONUS 4: Learning Curves - How Much Data Do I Need?
# TODO: Train model on 20%, 40%, 60%, 80%, 100% of data
# TODO: Plot training size vs accuracy
# Question: Does more data always help? Where does it plateau?


# ============================================================================
# BONUS CHALLENGES - Level 2 (Advanced)
# ============================================================================

print("\n" + "="*60)
print("BONUS CHALLENGES - LEVEL 2 (ADVANCED)")
print("="*60)

# ADVANCED 1: SHAP Values - WHY Did the Model Decide?
# pip install shap
# TODO: Use SHAP to explain individual predictions
# HINT: import shap; explainer = shap.TreeExplainer(tree_model)
# HINT: shap_values = explainer.shap_values(X_test_iris)
# HINT: shap.summary_plot(shap_values, X_test_iris)
# This shows: "For THIS flower, the model chose species 2 because petal_length=5.7"

# ADVANCED 2: Decision Boundaries - See How Models Think
# TODO: For 2D iris data (use only 2 features), plot decision boundaries
# HINT: Create a mesh grid of points, predict class for each point
# HINT: Use plt.contourf() to visualize the decision regions
# Compare: How do KNN vs Decision Tree boundaries differ?
# KNN: Smooth, local boundaries
# Decision Tree: Sharp, axis-aligned boundaries

# ADVANCED 3: Yellowbrick - Professional ML Visualization
# pip install yellowbrick
# TODO: Use Yellowbrick for instant visual diagnostics
# HINT: from yellowbrick.classifier import ConfusionMatrix
# HINT: from yellowbrick.model_selection import LearningCurve
# Examples:
#   - Confusion matrix heatmap (better than print)
#   - Learning curves (auto-generated)
#   - Residual plots for regression
#   - Classification reports with color coding

# ADVANCED 4: Real Data Challenge - Stock Price Prediction
# TODO: Download real stock data using yfinance
# pip install yfinance
# HINT: import yfinance as yf
# HINT: data = yf.download('AAPL', start='2023-01-01', end='2024-01-01')
# Challenge: Predict tomorrow's price from today's features
# Features: Open, High, Low, Volume, Moving Average
# WARNING: If this actually worked well, you'd be rich! (It doesn't)

# ADVANCED 5: The Overfitting Detective
# TODO: Create a decision tree with max_depth=1 (underfitting)
# TODO: Create a decision tree with max_depth=20 (overfitting)
# TODO: Create a decision tree with max_depth=3 (just right?)
# Compare training accuracy vs test accuracy for each
# The Goldilocks Problem: Not too simple, not too complex!

# ADVANCED 6: Ensemble Power - Random Forest
# TODO: Compare single DecisionTreeClassifier vs RandomForestClassifier
# HINT: from sklearn.ensemble import RandomForestClassifier
# HINT: Use same parameters, just change the model
# Question: Why do 100 mediocre trees beat 1 perfect tree?

# ADVANCED 7: Probability Calibration - "How Sure Are You?"
# TODO: Instead of just predictions, get probability scores
# HINT: knn_model.predict_proba(X_test_iris)
# This returns: [0.05, 0.10, 0.85] meaning "85% confident it's species 2"
# Use case: "Only flag for manual review if confidence < 80%"


print("\n" + "="*60)
print("Exercise Complete!")
print("="*60)
print("\nKey Takeaways:")
print("1. Regression predicts continuous values (prices, scores)")
print("2. Classification predicts categories (species, pass/fail)")
print("3. Always split data into train/test sets")
print("4. Different models work better for different problems")
print("5. Evaluation metrics: MSE/R² for regression, accuracy for classification")
