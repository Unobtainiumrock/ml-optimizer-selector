"""
Quickstart Example: Using ML Optimizer Selector

This is the simplest possible example to get started.
"""

import sys
sys.path.insert(0, '..')

from optimizer_selector import select_optimizer, get_sklearn_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load data
print("Loading breast cancer dataset...")
data = load_breast_cancer()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale (IMPORTANT!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Get recommendation
print("\nGetting solver recommendation...")
rec = select_optimizer(
    n_parameters=X_train.shape[1],
    n_samples=X_train.shape[0],
    regularization="l2"
)

print(f"\n{rec}")

# Get model
print("\nTraining model...")
model = get_sklearn_model(rec, task="classification", max_iter=1000)

# Train
model.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"\nResults:")
print(f"  Train Accuracy: {train_acc:.4f}")
print(f"  Test Accuracy: {test_acc:.4f}")
print(f"  Converged in {model.n_iter_[0]} iterations")

print("\n✓ Success! That's all you need to get started.")

