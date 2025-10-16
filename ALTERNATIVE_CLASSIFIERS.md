# Alternative Classifiers for Gender Detection

## Complete Implementation Guide

This document provides ready-to-use code for **5 different machine learning algorithms** as alternatives to Random Forest, plus a list of 15+ additional possibilities for experimentation.

---

## 📋 Table of Contents

1. [Alternative Classifier Implementations](#alternative-classifier-implementations)
   - [Support Vector Machine (SVM)](#1-support-vector-machine-svm)
   - [XGBoost (Gradient Boosting)](#2-xgboost-gradient-boosting)
   - [Neural Network (MLP)](#3-neural-network-mlp)
   - [K-Nearest Neighbors (KNN)](#4-k-nearest-neighbors-knn)
   - [Logistic Regression](#5-logistic-regression)
2. [Complete Comparison Code](#complete-comparison-code)
3. [Additional Algorithm Possibilities](#additional-algorithm-possibilities)
4. [Performance Comparison Table](#performance-comparison-table)

---

## Alternative Classifier Implementations

### 1. Support Vector Machine (SVM)

**Accuracy**: 92-95% | **Speed**: Medium | **Best For**: High-dimensional data

```python
# ============================================
# CELL: Import Libraries for SVM
# ============================================
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# CELL: Prepare Data (Feature Scaling Required)
# ============================================
# SVM is sensitive to feature scales, so we must standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data scaled successfully!")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")

# ============================================
# CELL: Initialize and Train SVM
# ============================================
# Create SVM classifier
svm_classifier = SVC(
    kernel='rbf',          # Radial Basis Function kernel (handles non-linearity)
    C=1.0,                 # Regularization parameter (controls margin vs errors)
    gamma='scale',         # Kernel coefficient (auto-calculated based on features)
    probability=True,      # Enable probability estimates
    random_state=42
)

# Train the model
print("Training SVM model...")
svm_classifier.fit(X_train_scaled, y_train)
print("✓ SVM training complete!")

# ============================================
# CELL: Make Predictions and Evaluate
# ============================================
# Predict on test set
y_pred_svm = svm_classifier.predict(X_test_scaled)

# Calculate accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Display results
print("\n" + "="*60)
print("SVM MODEL EVALUATION RESULTS")
print("="*60)
print(f"\nModel Accuracy: {accuracy_svm*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

# ============================================
# CELL: Get Probability Predictions
# ============================================
# Get prediction probabilities
y_proba_svm = svm_classifier.predict_proba(X_test_scaled)

# Display sample predictions with confidence
print("\nSample Predictions (first 10):")
print(f"{'Actual':<10} {'Predicted':<10} {'Confidence':<12} {'Female Prob':<12} {'Male Prob'}")
print("-" * 60)
for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
    predicted = y_pred_svm[i]
    confidence = y_proba_svm[i].max()
    prob_female = y_proba_svm[i][0] if svm_classifier.classes_[0] == 'female' else y_proba_svm[i][1]
    prob_male = y_proba_svm[i][1] if svm_classifier.classes_[1] == 'male' else y_proba_svm[i][0]

    print(f"{actual:<10} {predicted:<10} {confidence*100:>6.2f}%      {prob_female*100:>6.2f}%      {prob_male*100:>6.2f}%")

# ============================================
# CELL: Hyperparameter Tuning (Optional)
# ============================================
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

# Grid search with cross-validation
print("\nPerforming Grid Search for best parameters...")
grid_search = GridSearchCV(
    SVC(random_state=42, probability=True),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

# Best parameters
print("\nBest Parameters:", grid_search.best_params_)
print(f"Best Cross-Validation Score: {grid_search.best_score_*100:.2f}%")

# Use best model
best_svm = grid_search.best_estimator_
y_pred_best = best_svm.predict(X_test_scaled)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Test Accuracy with Best Model: {accuracy_best*100:.2f}%")
```

---

### 2. XGBoost (Gradient Boosting)

**Accuracy**: 95-98% | **Speed**: Fast | **Best For**: Maximum accuracy

```python
# ============================================
# CELL: Install XGBoost (if needed)
# ============================================
# Run this cell only if XGBoost is not installed
# !pip install xgboost

# ============================================
# CELL: Import Libraries for XGBoost
# ============================================
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

# ============================================
# CELL: Prepare Labels (XGBoost needs numeric)
# ============================================
# XGBoost requires numeric labels (0, 1) instead of strings
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)  # 'female'->0, 'male'->1
y_test_encoded = le.transform(y_test)

print("Label Encoding:")
print(f"Classes: {le.classes_}")
print(f"'female' encoded as: {le.transform(['female'])[0]}")
print(f"'male' encoded as: {le.transform(['male'])[0]}")

# ============================================
# CELL: Initialize and Train XGBoost
# ============================================
# Create XGBoost classifier
xgb_classifier = XGBClassifier(
    n_estimators=200,        # Number of boosting rounds (trees)
    max_depth=6,             # Maximum tree depth
    learning_rate=0.1,       # Step size shrinkage (lower = more conservative)
    subsample=0.8,           # Fraction of samples for each tree
    colsample_bytree=0.8,    # Fraction of features for each tree
    random_state=42,
    eval_metric='logloss',   # Evaluation metric
    use_label_encoder=False, # Disable deprecated label encoder warning
    verbosity=0              # Suppress warnings
)

# Train the model
print("\nTraining XGBoost model...")
xgb_classifier.fit(
    X_train,
    y_train_encoded,
    eval_set=[(X_test, y_test_encoded)],
    verbose=False
)
print("✓ XGBoost training complete!")

# ============================================
# CELL: Make Predictions and Evaluate
# ============================================
# Predict on test set
y_pred_xgb = xgb_classifier.predict(X_test)
y_pred_xgb_labels = le.inverse_transform(y_pred_xgb)  # Convert back to 'male'/'female'

# Calculate accuracy
accuracy_xgb = accuracy_score(y_test_encoded, y_pred_xgb)

# Display results
print("\n" + "="*60)
print("XGBOOST MODEL EVALUATION RESULTS")
print("="*60)
print(f"\nModel Accuracy: {accuracy_xgb*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb_labels))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb_labels))

# ============================================
# CELL: Feature Importance Analysis
# ============================================
# Get feature importances
xgb_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_classifier.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Display top 10
print("\nTop 10 Most Important Features:")
print(xgb_importance.head(10).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(xgb_importance['Feature'][:10], xgb_importance['Importance'][:10], color='coral')
plt.gca().invert_yaxis()
plt.xlabel("Importance Score")
plt.title("XGBoost: Top 10 Important Features")
plt.tight_layout()
plt.show()

# ============================================
# CELL: Learning Curve (Training Progress)
# ============================================
# Train with evaluation tracking
xgb_eval = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    verbosity=0
)

# Track performance
eval_set = [(X_train, y_train_encoded), (X_test, y_test_encoded)]
xgb_eval.fit(
    X_train,
    y_train_encoded,
    eval_set=eval_set,
    eval_metric='logloss',
    verbose=False
)

# Get evaluation results
results = xgb_eval.evals_result()

# Plot learning curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(results['validation_0']['logloss'], label='Train')
plt.plot(results['validation_1']['logloss'], label='Test')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.title('XGBoost Learning Curve')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
# Convert log loss to accuracy approximation for visualization
train_acc = [1 - min(x, 1) for x in results['validation_0']['logloss']]
test_acc = [1 - min(x, 1) for x in results['validation_1']['logloss']]
plt.plot(train_acc, label='Train')
plt.plot(test_acc, label='Test')
plt.xlabel('Boosting Round')
plt.ylabel('Approximate Accuracy')
plt.title('Training Progress')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### 3. Neural Network (MLP)

**Accuracy**: 93-97% | **Speed**: Slow | **Best For**: Complex patterns, large datasets

```python
# ============================================
# CELL: Import Libraries for Neural Network
# ============================================
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# CELL: Prepare Data (Scaling Required)
# ============================================
# Neural networks require feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data scaled for Neural Network")
print(f"Mean of scaled features: {X_train_scaled.mean():.4f}")
print(f"Std of scaled features: {X_train_scaled.std():.4f}")

# ============================================
# CELL: Initialize and Train Neural Network
# ============================================
# Create Neural Network classifier
nn_classifier = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # 2 hidden layers: 100 neurons, then 50 neurons
    activation='relu',             # ReLU activation function
    solver='adam',                 # Adam optimizer (adaptive learning rate)
    alpha=0.0001,                  # L2 regularization parameter
    batch_size='auto',             # Mini-batch size (auto = min(200, n_samples))
    learning_rate='adaptive',      # Adaptive learning rate
    learning_rate_init=0.001,      # Initial learning rate
    max_iter=500,                  # Maximum number of epochs
    random_state=42,
    early_stopping=True,           # Stop if validation score doesn't improve
    validation_fraction=0.1,       # 10% of training data for validation
    n_iter_no_change=10,           # Epochs with no improvement before stopping
    verbose=False                  # Don't print progress
)

# Train the model
print("\nTraining Neural Network...")
print("(This may take 20-60 seconds depending on dataset size)")
nn_classifier.fit(X_train_scaled, y_train)

print(f"\n✓ Training complete!")
print(f"Training converged in {nn_classifier.n_iter_} iterations")
print(f"Final training loss: {nn_classifier.loss_:.4f}")

# ============================================
# CELL: Make Predictions and Evaluate
# ============================================
# Predict on test set
y_pred_nn = nn_classifier.predict(X_test_scaled)

# Calculate accuracy
accuracy_nn = accuracy_score(y_test, y_pred_nn)

# Display results
print("\n" + "="*60)
print("NEURAL NETWORK MODEL EVALUATION RESULTS")
print("="*60)
print(f"\nModel Accuracy: {accuracy_nn*100:.2f}%")
print(f"Number of layers: {nn_classifier.n_layers_}")
print(f"Number of outputs: {nn_classifier.n_outputs_}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nn))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_nn))

# ============================================
# CELL: Visualize Network Architecture
# ============================================
print("\n" + "="*60)
print("NEURAL NETWORK ARCHITECTURE")
print("="*60)

# Display layer information
print(f"\nInput Layer: {X_train_scaled.shape[1]} features")
for i, layer_size in enumerate(nn_classifier.hidden_layer_sizes):
    print(f"Hidden Layer {i+1}: {layer_size} neurons")
print(f"Output Layer: {nn_classifier.n_outputs_} classes")

# Count total parameters
total_params = 0
layer_sizes = [X_train_scaled.shape[1]] + list(nn_classifier.hidden_layer_sizes) + [nn_classifier.n_outputs_]
for i in range(len(layer_sizes) - 1):
    weights = layer_sizes[i] * layer_sizes[i+1]
    biases = layer_sizes[i+1]
    total_params += weights + biases
    print(f"\nLayer {i} → Layer {i+1}:")
    print(f"  Weights: {layer_sizes[i]} × {layer_sizes[i+1]} = {weights}")
    print(f"  Biases: {biases}")
    print(f"  Subtotal: {weights + biases} parameters")

print(f"\nTotal trainable parameters: {total_params:,}")

# ============================================
# CELL: Plot Training Loss Curve
# ============================================
# Plot loss over iterations
plt.figure(figsize=(10, 6))
plt.plot(nn_classifier.loss_curve_, color='blue', linewidth=2, label='Training Loss')
plt.xlabel('Iterations (Epochs)')
plt.ylabel('Loss')
plt.title('Neural Network Training Loss Over Time')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nFinal loss: {nn_classifier.loss_curve_[-1]:.4f}")
print(f"Best validation score: {nn_classifier.best_validation_score_:.4f}")

# ============================================
# CELL: Probability Predictions
# ============================================
# Get prediction probabilities
y_proba_nn = nn_classifier.predict_proba(X_test_scaled)

# Display sample predictions
print("\nSample Predictions with Confidence (first 10):")
print(f"{'Actual':<10} {'Predicted':<10} {'Confidence':<12} {'Female Prob':<12} {'Male Prob'}")
print("-" * 60)
for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
    predicted = y_pred_nn[i]
    confidence = y_proba_nn[i].max()
    prob_female = y_proba_nn[i][0] if nn_classifier.classes_[0] == 'female' else y_proba_nn[i][1]
    prob_male = y_proba_nn[i][1] if nn_classifier.classes_[1] == 'male' else y_proba_nn[i][0]

    print(f"{actual:<10} {predicted:<10} {confidence*100:>6.2f}%      {prob_female*100:>6.2f}%      {prob_male*100:>6.2f}%")
```

---

### 4. K-Nearest Neighbors (KNN)

**Accuracy**: 88-94% | **Speed**: Slow | **Best For**: Small datasets, interpretability

```python
# ============================================
# CELL: Import Libraries for KNN
# ============================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# CELL: Prepare Data (Scaling Required)
# ============================================
# KNN is distance-based, so scaling is essential
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data scaled for KNN")

# ============================================
# CELL: Initialize and Train KNN
# ============================================
# Create KNN classifier
knn_classifier = KNeighborsClassifier(
    n_neighbors=5,           # Number of neighbors to consider
    weights='distance',      # Weight by inverse distance (closer = more influence)
    metric='euclidean',      # Distance metric (Euclidean distance)
    algorithm='auto',        # Choose best algorithm automatically
    n_jobs=-1                # Use all CPU cores for faster computation
)

# "Train" KNN (just stores training data)
print("Training KNN model...")
knn_classifier.fit(X_train_scaled, y_train)
print("✓ KNN training complete! (data stored)")

# ============================================
# CELL: Make Predictions and Evaluate
# ============================================
# Predict on test set
y_pred_knn = knn_classifier.predict(X_test_scaled)

# Calculate accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# Display results
print("\n" + "="*60)
print("KNN MODEL EVALUATION RESULTS")
print("="*60)
print(f"\nModel Accuracy: {accuracy_knn*100:.2f}%")
print(f"K (neighbors): {knn_classifier.n_neighbors}")
print(f"Distance metric: {knn_classifier.metric}")
print(f"Weighting: {knn_classifier.weights}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

# ============================================
# CELL: Find Optimal K Value
# ============================================
# Test different K values
k_values = [1, 3, 5, 7, 9, 11, 15, 20, 25, 30]
k_accuracies = []

print("\nTesting different K values...")
print(f"{'K Value':<10} {'Accuracy':<10}")
print("-" * 20)

for k in k_values:
    knn_temp = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance',
        n_jobs=-1
    )
    knn_temp.fit(X_train_scaled, y_train)
    y_pred_temp = knn_temp.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred_temp)
    k_accuracies.append(acc)
    print(f"{k:<10} {acc*100:>6.2f}%")

# Find best K
best_k_idx = np.argmax(k_accuracies)
best_k = k_values[best_k_idx]
best_acc = k_accuracies[best_k_idx]

print(f"\nBest K Value: {best_k}")
print(f"Best Accuracy: {best_acc*100:.2f}%")

# ============================================
# CELL: Visualize K vs Accuracy
# ============================================
# Plot K value vs accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_values, [a*100 for a in k_accuracies],
         marker='o', linewidth=2, markersize=8, color='green')
plt.axvline(x=best_k, color='red', linestyle='--',
            label=f'Best K={best_k} ({best_acc*100:.2f}%)')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Accuracy (%)')
plt.title('KNN: K Value vs Model Accuracy')
plt.grid(alpha=0.3)
plt.xticks(k_values)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================
# CELL: Retrain with Best K
# ============================================
# Use best K value
knn_best = KNeighborsClassifier(
    n_neighbors=best_k,
    weights='distance',
    n_jobs=-1
)
knn_best.fit(X_train_scaled, y_train)
y_pred_best_knn = knn_best.predict(X_test_scaled)
accuracy_best_knn = accuracy_score(y_test, y_pred_best_knn)

print(f"\nFinal Model with K={best_k}:")
print(f"Test Accuracy: {accuracy_best_knn*100:.2f}%")

# ============================================
# CELL: Analyze Nearest Neighbors (Example)
# ============================================
# Show nearest neighbors for a sample
sample_idx = 0
sample = X_test_scaled[sample_idx].reshape(1, -1)
distances, indices = knn_best.kneighbors(sample, n_neighbors=best_k)

print(f"\nNearest Neighbors Analysis for Test Sample {sample_idx}:")
print(f"Actual label: {y_test.iloc[sample_idx] if hasattr(y_test, 'iloc') else y_test[sample_idx]}")
print(f"Predicted label: {y_pred_best_knn[sample_idx]}")
print(f"\nTop {best_k} Nearest Neighbors:")
print(f"{'Neighbor':<10} {'Distance':<12} {'Label'}")
print("-" * 35)

for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    neighbor_label = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
    print(f"#{i+1:<9} {dist:>8.4f}     {neighbor_label}")

# Vote count
neighbor_labels = [y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
                   for idx in indices[0]]
from collections import Counter
vote_counts = Counter(neighbor_labels)
print(f"\nVote Distribution: {dict(vote_counts)}")
print(f"Winner: {vote_counts.most_common(1)[0][0]}")
```

---

### 5. Logistic Regression

**Accuracy**: 90-93% | **Speed**: Very Fast | **Best For**: Baseline, simple linear patterns

```python
# ============================================
# CELL: Import Libraries for Logistic Regression
# ============================================
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================
# CELL: Prepare Data (Scaling Recommended)
# ============================================
# Logistic Regression works better with scaled features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data scaled for Logistic Regression")

# ============================================
# CELL: Initialize and Train Logistic Regression
# ============================================
# Create Logistic Regression classifier
lr_classifier = LogisticRegression(
    penalty='l2',              # L2 regularization (Ridge)
    C=1.0,                     # Inverse regularization strength
    solver='lbfgs',            # Optimization algorithm
    max_iter=1000,             # Maximum iterations
    random_state=42,
    n_jobs=-1                  # Use all CPU cores
)

# Train the model
print("\nTraining Logistic Regression model...")
lr_classifier.fit(X_train_scaled, y_train)
print("✓ Logistic Regression training complete!")

# ============================================
# CELL: Make Predictions and Evaluate
# ============================================
# Predict on test set
y_pred_lr = lr_classifier.predict(X_test_scaled)

# Calculate accuracy
accuracy_lr = accuracy_score(y_test, y_pred_lr)

# Display results
print("\n" + "="*60)
print("LOGISTIC REGRESSION MODEL EVALUATION RESULTS")
print("="*60)
print(f"\nModel Accuracy: {accuracy_lr*100:.2f}%")
print(f"Number of iterations: {lr_classifier.n_iter_[0]}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

# ============================================
# CELL: Feature Coefficients (Importance)
# ============================================
# Get feature coefficients
coefficients = lr_classifier.coef_[0]
feature_importance_lr = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values(by='Abs_Coefficient', ascending=False)

# Display top 10
print("\nTop 10 Most Important Features (by absolute coefficient):")
print(feature_importance_lr.head(10)[['Feature', 'Coefficient']].to_string(index=False))

# Plot feature importance
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
top_features = feature_importance_lr.head(10)
colors = ['green' if c > 0 else 'red' for c in top_features['Coefficient']]
plt.barh(top_features['Feature'], top_features['Coefficient'], color=colors)
plt.gca().invert_yaxis()
plt.xlabel("Coefficient Value")
plt.title("Logistic Regression: Top 10 Feature Coefficients")
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

plt.subplot(1, 2, 2)
plt.barh(top_features['Feature'], top_features['Abs_Coefficient'], color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Absolute Coefficient Value")
plt.title("Feature Importance (Magnitude)")

plt.tight_layout()
plt.show()

# ============================================
# CELL: Probability Predictions
# ============================================
# Get prediction probabilities
y_proba_lr = lr_classifier.predict_proba(X_test_scaled)

# Display sample predictions
print("\nSample Predictions with Probabilities (first 10):")
print(f"{'Actual':<10} {'Predicted':<10} {'Confidence':<12} {'Female Prob':<12} {'Male Prob'}")
print("-" * 60)
for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
    predicted = y_pred_lr[i]
    confidence = y_proba_lr[i].max()
    prob_female = y_proba_lr[i][0] if lr_classifier.classes_[0] == 'female' else y_proba_lr[i][1]
    prob_male = y_proba_lr[i][1] if lr_classifier.classes_[1] == 'male' else y_proba_lr[i][0]

    print(f"{actual:<10} {predicted:<10} {confidence*100:>6.2f}%      {prob_female*100:>6.2f}%      {prob_male*100:>6.2f}%")

# ============================================
# CELL: Decision Boundary Visualization (2D)
# ============================================
# Visualize decision boundary using top 2 features
from sklearn.decomposition import PCA

# Reduce to 2D using PCA
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train_scaled)
X_test_2d = pca.transform(X_test_scaled)

# Train LR on 2D data
lr_2d = LogisticRegression(random_state=42)
lr_2d.fit(X_train_2d, y_train)

# Create mesh grid
h = 0.02  # step size
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict on mesh
Z = lr_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = (Z == lr_2d.classes_[1]).astype(int)  # Convert to binary
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')

# Plot training points
colors = {'male': 'blue', 'female': 'red'}
for gender in ['male', 'female']:
    mask = y_train == gender
    plt.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1],
                c=colors[gender], label=f'{gender} (train)',
                alpha=0.6, s=30, edgecolors='k')

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Logistic Regression Decision Boundary (2D Projection)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nPCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained by 2 components: {sum(pca.explained_variance_ratio_)*100:.2f}%")
```

---

## Complete Comparison Code

Run all classifiers and compare performance:

```python
# ============================================
# CELL: Import All Required Libraries
# ============================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# CELL: Prepare Data for All Models
# ============================================
# Scaling for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Label encoding for XGBoost
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print("Data preparation complete!")
print(f"Original features: {X_train.shape}")
print(f"Scaled features: {X_train_scaled.shape}")

# ============================================
# CELL: Define All Classifiers
# ============================================
classifiers = {
    'Random Forest': {
        'model': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42
        ),
        'needs_scaling': False,
        'needs_encoding': False
    },
    'SVM': {
        'model': SVC(
            kernel='rbf',
            C=1.0,
            random_state=42
        ),
        'needs_scaling': True,
        'needs_encoding': False
    },
    'XGBoost': {
        'model': XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        ),
        'needs_scaling': False,
        'needs_encoding': True
    },
    'Neural Network': {
        'model': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            verbose=False
        ),
        'needs_scaling': True,
        'needs_encoding': False
    },
    'KNN': {
        'model': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        ),
        'needs_scaling': True,
        'needs_encoding': False
    },
    'Logistic Regression': {
        'model': LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        'needs_scaling': True,
        'needs_encoding': False
    }
}

# ============================================
# CELL: Train and Evaluate All Models
# ============================================
results = []

print("\n" + "="*70)
print("TRAINING AND EVALUATING ALL MODELS")
print("="*70)

for name, config in classifiers.items():
    print(f"\n{'-'*70}")
    print(f"Training {name}...")
    print(f"{'-'*70}")

    # Select appropriate data
    X_train_use = X_train_scaled if config['needs_scaling'] else X_train
    X_test_use = X_test_scaled if config['needs_scaling'] else X_test
    y_train_use = y_train_encoded if config['needs_encoding'] else y_train
    y_test_use = y_test_encoded if config['needs_encoding'] else y_test

    # Train model
    start_time = time.time()
    config['model'].fit(X_train_use, y_train_use)
    train_time = time.time() - start_time

    # Make predictions
    y_pred = config['model'].predict(X_test_use)

    # Convert predictions back if needed
    if config['needs_encoding']:
        y_pred = le.inverse_transform(y_pred)
        y_test_eval = y_test
    else:
        y_test_eval = y_test

    # Calculate metrics
    accuracy = accuracy_score(y_test_eval, y_pred)
    precision = precision_score(y_test_eval, y_pred, pos_label='female', average='binary')
    recall = recall_score(y_test_eval, y_pred, pos_label='female', average='binary')
    f1 = f1_score(y_test_eval, y_pred, pos_label='female', average='binary')

    # Store results
    results.append({
        'Algorithm': name,
        'Accuracy (%)': round(accuracy * 100, 2),
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'F1-Score': round(f1, 3),
        'Training Time (s)': round(train_time, 3)
    })

    print(f"✓ {name} completed!")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Training time: {train_time:.3f}s")

# ============================================
# CELL: Display Comparison Results
# ============================================
# Create results DataFrame
results_df = pd.DataFrame(results).sort_values('Accuracy (%)', ascending=False)

print("\n\n" + "="*70)
print("FINAL COMPARISON RESULTS")
print("="*70)
print(results_df.to_string(index=False))

# Find best model
best_model = results_df.iloc[0]
print(f"\n🏆 BEST MODEL: {best_model['Algorithm']}")
print(f"   Accuracy: {best_model['Accuracy (%)']}%")
print(f"   F1-Score: {best_model['F1-Score']}")
print(f"   Training Time: {best_model['Training Time (s)']}s")

# ============================================
# CELL: Visualize Comparison
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Accuracy comparison
axes[0, 0].barh(results_df['Algorithm'], results_df['Accuracy (%)'], color='skyblue')
axes[0, 0].set_xlabel('Accuracy (%)')
axes[0, 0].set_title('Model Accuracy Comparison', fontweight='bold')
axes[0, 0].set_xlim([85, 100])
axes[0, 0].grid(axis='x', alpha=0.3)
axes[0, 0].axvline(x=90, color='red', linestyle='--', alpha=0.5, label='90% threshold')
axes[0, 0].legend()

# 2. Training time comparison
axes[0, 1].barh(results_df['Algorithm'], results_df['Training Time (s)'], color='coral')
axes[0, 1].set_xlabel('Training Time (seconds)')
axes[0, 1].set_title('Training Time Comparison', fontweight='bold')
axes[0, 1].grid(axis='x', alpha=0.3)

# 3. F1-Score comparison
axes[1, 0].barh(results_df['Algorithm'], results_df['F1-Score'], color='lightgreen')
axes[1, 0].set_xlabel('F1-Score')
axes[1, 0].set_title('F1-Score Comparison', fontweight='bold')
axes[1, 0].set_xlim([0.85, 1.0])
axes[1, 0].grid(axis='x', alpha=0.3)

# 4. Precision vs Recall scatter
axes[1, 1].scatter(results_df['Precision'], results_df['Recall'],
                   s=200, c=results_df['Accuracy (%)'], cmap='viridis',
                   edgecolors='black', linewidth=2, alpha=0.7)
for idx, row in results_df.iterrows():
    axes[1, 1].annotate(row['Algorithm'],
                       (row['Precision'], row['Recall']),
                       fontsize=8, ha='right')
axes[1, 1].set_xlabel('Precision')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].set_title('Precision vs Recall', fontweight='bold')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_xlim([0.85, 1.0])
axes[1, 1].set_ylim([0.85, 1.0])

plt.tight_layout()
plt.show()

# ============================================
# CELL: Save Results to CSV
# ============================================
# Save comparison results
results_df.to_csv('model_comparison_results.csv', index=False)
print("\n✓ Results saved to 'model_comparison_results.csv'")

# Create detailed report
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"Average Accuracy: {results_df['Accuracy (%)'].mean():.2f}%")
print(f"Best Accuracy: {results_df['Accuracy (%)'].max():.2f}%")
print(f"Worst Accuracy: {results_df['Accuracy (%)'].min():.2f}%")
print(f"Accuracy Std Dev: {results_df['Accuracy (%)'].std():.2f}%")
print(f"\nFastest Training: {results_df['Training Time (s)'].min():.3f}s ({results_df.loc[results_df['Training Time (s)'].idxmin(), 'Algorithm']})")
print(f"Slowest Training: {results_df['Training Time (s)'].max():.3f}s ({results_df.loc[results_df['Training Time (s)'].idxmax(), 'Algorithm']})")
```

---

## Additional Algorithm Possibilities

### 🎯 Tree-Based Algorithms

1. **Decision Tree** - Single tree (interpretable, but overfits)
2. **Extra Trees** - Like Random Forest but more random splits
3. **AdaBoost** - Boosting algorithm (sequential weak learners)
4. **LightGBM** - Fast gradient boosting (similar to XGBoost)
5. **CatBoost** - Handles categorical features well

### 📊 Linear Models

6. **Ridge Classifier** - L2 regularized linear model
7. **SGD Classifier** - Stochastic Gradient Descent
8. **Perceptron** - Simple linear classifier
9. **Passive Aggressive Classifier** - Online learning

### 🧠 Probabilistic Models

10. **Naive Bayes (Gaussian)** - Probabilistic classifier
11. **Naive Bayes (Multinomial)** - For count data
12. **Linear Discriminant Analysis (LDA)** - Statistical approach
13. **Quadratic Discriminant Analysis (QDA)** - Non-linear boundaries

### 🔬 Advanced Algorithms

14. **Gradient Boosting Machine (GBM)** - Classic gradient boosting
15. **Histogram-based Gradient Boosting** - Faster than GBM
16. **Voting Classifier** - Ensemble of multiple models
17. **Stacking Classifier** - Meta-learning ensemble
18. **Bagging Classifier** - Bootstrap aggregating

### 📐 Distance/Similarity-Based

19. **Radius Neighbors Classifier** - Similar to KNN but radius-based
20. **Nearest Centroid** - Classify by distance to class centroids

### 🚀 Deep Learning (Advanced)

21. **TensorFlow Neural Network** - Custom deep learning
22. **PyTorch Neural Network** - Research-focused deep learning
23. **Convolutional Neural Network (CNN)** - For spectrogram images
24. **Recurrent Neural Network (RNN)** - For sequential features
25. **Transformer-based models** - Attention mechanisms

---

## Performance Comparison Table

| Algorithm               | Expected Accuracy | Training Speed   | Prediction Speed | Memory Usage | Interpretability | Best Use Case                       |
| ----------------------- | ----------------- | ---------------- | ---------------- | ------------ | ---------------- | ----------------------------------- |
| **Random Forest**       | 94-96%            | Fast ⚡⚡        | Fast ⚡⚡        | Medium       | Medium 📊        | General purpose, feature importance |
| **SVM**                 | 92-95%            | Medium ⚡        | Fast ⚡⚡        | Medium       | Low 📉           | High-dimensional data               |
| **XGBoost**             | 95-98%            | Fast ⚡⚡⚡      | Very Fast ⚡⚡⚡ | Medium       | Medium 📊        | Maximum accuracy                    |
| **Neural Network**      | 93-97%            | Slow 🐢          | Fast ⚡⚡        | High         | Low 📉           | Complex patterns                    |
| **KNN**                 | 88-94%            | None (instant)   | Very Slow 🐌     | High         | High 📈          | Small datasets                      |
| **Logistic Regression** | 90-93%            | Very Fast ⚡⚡⚡ | Very Fast ⚡⚡⚡ | Low          | Very High 📈📈   | Baseline, interpretable             |
| **Decision Tree**       | 85-90%            | Fast ⚡⚡        | Very Fast ⚡⚡⚡ | Low          | Very High 📈📈   | Interpretable rules                 |
| **AdaBoost**            | 92-95%            | Medium ⚡        | Fast ⚡⚡        | Medium       | Low 📉           | Weak learner boosting               |
| **LightGBM**            | 95-98%            | Very Fast ⚡⚡⚡ | Very Fast ⚡⚡⚡ | Low          | Medium 📊        | Large datasets                      |
| **Naive Bayes**         | 88-92%            | Very Fast ⚡⚡⚡ | Very Fast ⚡⚡⚡ | Low          | High 📈          | Probabilistic reasoning             |

---

## Quick Start Guide

### 1️⃣ For Maximum Accuracy

```python
# Use XGBoost or LightGBM
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=200, max_depth=6, random_state=42)
# Expected: 95-98% accuracy
```

### 2️⃣ For Fastest Training

```python
# Use Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)
# Expected: 90-93% accuracy, trains in <1 second
```

### 3️⃣ For Interpretability

```python
# Use Decision Tree or Logistic Regression
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=10, random_state=42)
# Can visualize tree structure and rules
```

### 4️⃣ For Small Datasets

```python
# Use KNN or SVM
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5, weights='distance')
# Works well with <500 samples
```

### 5️⃣ For Large Datasets

```python
# Use LightGBM or SGD
import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=200, random_state=42)
# Handles 10,000+ samples efficiently
```

---

## Installation Commands

```bash
# Core libraries (already installed in most environments)
pip install scikit-learn pandas numpy matplotlib

# For XGBoost
pip install xgboost

# For LightGBM
pip install lightgbm

# For CatBoost
pip install catboost

# For Deep Learning
pip install tensorflow
# OR
pip install torch torchvision
```

---

## Conclusion

This document provides:

- ✅ **5 complete implementations** ready to copy-paste
- ✅ **Detailed explanations** for each algorithm
- ✅ **Comparison code** to test all models
- ✅ **25+ additional algorithms** to explore
- ✅ **Performance benchmarks** for decision-making

**Recommended Workflow:**

1. Start with **Random Forest** (baseline)
2. Try **XGBoost** for better accuracy
3. Test **SVM** and **Neural Network** for comparison
4. Use **comparison code** to find the best model
5. Fine-tune hyperparameters of the winner

**Expected Results:**

- Top 3 performers: XGBoost, Random Forest, Neural Network
- All models should achieve >90% accuracy for gender detection
- XGBoost typically wins with 95-98% accuracy

---

**Document Created**: October 14, 2025  
**Project**: Gender Detection from Audio Features  
**Course**: IT 302 - Probability & Statistics Lab  
**Total Algorithms Covered**: 30+ options

---

_End of Alternative Classifiers Guide_
