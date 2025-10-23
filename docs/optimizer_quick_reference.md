# Optimizer Selector - Quick Reference

## Installation

Place `optimizer_selector.py` in your project directory or Python path.

## Basic Usage

```python
from optimizer_selector import select_optimizer, get_sklearn_model

# Get recommendation
rec = select_optimizer(
    n_parameters=1000,      # Number of features
    n_samples=10000,        # Number of training samples
    regularization="l2",    # "none", "l1", "l2", or "elasticnet"
    dataset_in_memory=True  # Does dataset fit in RAM?
)

# Get configured sklearn model
model = get_sklearn_model(rec, task="classification", max_iter=1000)

# Train as usual
model.fit(X_train, y_train)
```

## Decision Tree Logic

```
Is n_parameters > 100,000?
├─ YES → Use SGD/Mini-batch SGD
│        (only scalable option for very high dimensions)
│
└─ NO → Do you need L1 regularization?
    ├─ YES → Use SAGA
    │        (SAGA handles non-smooth L1 penalty)
    │
    └─ NO → Is dataset in memory?
        ├─ NO → Use Mini-batch SGD
        │       (for out-of-core learning)
        │
        └─ YES → Is n_samples > 1,000,000?
            ├─ YES → Use SAG
            │        (optimized for large in-memory datasets)
            │
            └─ NO → Use L-BFGS ✓
                    (optimal for most problems)
```

## API Reference

### `select_optimizer()`

**Parameters:**
- `n_parameters: int` - Number of model parameters/features
- `n_samples: int` - Number of training samples
- `regularization: str` - One of: `"none"`, `"l1"`, `"l2"`, `"elasticnet"`
- `dataset_in_memory: bool` - Whether entire dataset fits in memory
- `is_multiclass: bool` - Whether problem is multiclass (optional)
- `max_memory_gb: float` - Maximum memory budget in GB (optional)

**Returns:**
- `OptimizerRecommendation` object with:
  - `.solver` - Recommended solver name
  - `.reason` - Explanation for recommendation
  - `.sklearn_params` - Dict of sklearn parameters

### `get_sklearn_model()`

**Parameters:**
- `recommendation: OptimizerRecommendation` - From `select_optimizer()`
- `task: str` - Either `"classification"` or `"regression"`
- `**extra_kwargs` - Additional sklearn parameters (e.g., `max_iter`, `random_state`)

**Returns:**
- Configured sklearn model ready for training

## Common Scenarios

### Small Dataset with L2 Regularization
```python
rec = select_optimizer(
    n_parameters=100,
    n_samples=1000,
    regularization="l2"
)
# Recommendation: lbfgs
```

### Feature Selection with L1 (Lasso)
```python
rec = select_optimizer(
    n_parameters=5000,
    n_samples=10000,
    regularization="l1"
)
# Recommendation: saga
```

### Large Dataset
```python
rec = select_optimizer(
    n_parameters=10000,
    n_samples=2000000,
    regularization="l2",
    dataset_in_memory=True
)
# Recommendation: sag
```

### Very Large Parameter Space
```python
rec = select_optimizer(
    n_parameters=150000,
    n_samples=50000,
    regularization="l2"
)
# Recommendation: sgd
```

### Memory-Constrained Environment
```python
rec = select_optimizer(
    n_parameters=50000,
    n_samples=100000,
    regularization="l2",
    max_memory_gb=1.0
)
# Will avoid L-BFGS if memory requirements exceed limit
```

### Out-of-Memory Dataset
```python
rec = select_optimizer(
    n_parameters=1000,
    n_samples=10000000,
    regularization="l2",
    dataset_in_memory=False
)
# Recommendation: sgd (use SGDClassifier/SGDRegressor)
```

## Complete Example

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from optimizer_selector import select_optimizer, get_sklearn_model

# Generate data
X, y = make_classification(n_samples=5000, n_features=500, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Get optimal solver
rec = select_optimizer(
    n_parameters=X_train.shape[1],
    n_samples=X_train.shape[0],
    regularization="l2",
    dataset_in_memory=True
)

print(rec)  # See recommendation and reasoning

# Get model with recommended solver
model = get_sklearn_model(
    rec,
    task="classification",
    max_iter=1000,
    random_state=42
)

# Train and evaluate
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {accuracy:.4f}")
```

## Solver Characteristics Comparison

| Solver | Time/Iter | Memory | Convergence | Best For | Supports L1? |
|--------|-----------|--------|-------------|----------|--------------|
| **lbfgs** | $O(mn)$ | $O(mn)$ | Superlinear | Small-medium datasets | ❌ |
| **saga** | $O(n)$ | $O(n)$ | Linear | L1 regularization, large data | ✅ |
| **sag** | $O(n)$ | $O(n)$ | Linear | Very large in-memory datasets | ❌ |
| **sgd** | $O(1)$ | $O(n)$ | Sublinear | Massive/out-of-memory data | ✅* |
| **liblinear** | $O(n)$ | $O(n)$ | Linear | Small binary classification | ✅ |

*SGD supports L1 but requires different class (SGDClassifier vs LogisticRegression)

## When to Override Recommendations

The selector uses general heuristics. Consider manual selection if:

1. **Time constraints**: Need fastest training (use liblinear for small data)
2. **Reproducibility**: Some solvers have different convergence behavior
3. **Special requirements**: Specific solver features needed
4. **Empirical testing**: You've benchmarked and found different solver works better

## Tips

- **Always specify `max_iter`**: Default 100 may be too low for some problems
- **Scale features**: All solvers benefit from standardized features
- **Check convergence**: Look at `model.n_iter_` to see if solver converged
- **Benchmark when uncertain**: Test multiple solvers on a subset of data
- **Memory is tricky**: If you get memory errors, try `max_memory_gb` parameter

## Troubleshooting

**Problem: ConvergenceWarning**
```python
# Solution: Increase max_iter or scale features
model = get_sklearn_model(rec, task="classification", max_iter=5000)
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
```

**Problem: Training too slow**
```python
# Solution: Try reducing dataset or using SGD
rec = select_optimizer(n_parameters=n, n_samples=n_samples//10, ...)
# Or force SGD
from sklearn.linear_model import SGDClassifier
model = SGDClassifier(max_iter=1000)
```

**Problem: Out of memory error**
```python
# Solution: Use memory constraint parameter
rec = select_optimizer(..., max_memory_gb=4.0)
# Or force out-of-core learning
rec = select_optimizer(..., dataset_in_memory=False)
```

## Further Reading

- [sklearn LogisticRegression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Choosing the right estimator](https://scikit-learn.org/stable/tutorial/machine_learning_map/)
- [Original paper: Nocedal & Wright, "Numerical Optimization" (L-BFGS)](https://books.google.com/books/about/Numerical_Optimization.html?id=VbHYoSyelFcC)

