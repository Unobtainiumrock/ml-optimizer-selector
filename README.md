# ML Optimizer Selector 🎯

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%3E%3D1.0-orange.svg)](https://scikit-learn.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A practical utility for automatically selecting the optimal optimization solver for machine learning models based on problem characteristics.

## 🌟 Features

- **Automatic solver selection** based on dataset size, regularization type, and memory constraints
- **Decision tree logic** implementing best practices from numerical optimization
- **sklearn-compatible** - drops right into your existing workflow
- **Type-safe** with full type hints and validation
- **Well-documented** with extensive examples and explanations
- **Educational** - includes theoretical background and trade-off analysis

## 📘 Comprehensive Optimization Notes

**[📄 optimization_notes.pdf](optimization_notes.pdf)** - In-depth theoretical guide

This comprehensive PDF covers:
- **L-BFGS and Second-Order Optimization Methods** - Complete mathematical foundations
- **Quasi-Newton Methods** - How L-BFGS approximates the Hessian efficiently using only $m$ vector pairs
- **Convergence Analysis** - Why L-BFGS achieves superlinear convergence (nearly independent of condition number)
- **Two-Loop Recursion Algorithm** - The core computational engine with detailed complexity analysis
- **Memory Trade-offs** - Detailed space complexity ($O(mn)$ vs $O(n^2)$) and storage comparisons
- **Why L-BFGS Cannot Handle L1 Regularization** - The smoothness requirement and proximal methods
- **Comparison with First-Order Methods** - When and why to use each solver

> 💡 **Start here** if you want to understand the mathematical foundations behind the optimizer selection logic!

## 📁 Project Structure

```
ml-optimizer-selector/
├── optimizer_selector.py          # Core implementation
├── optimization_notes.pdf          # 📘 Comprehensive theoretical guide (START HERE!)
├── README.md                       # This file
├── LICENSE                         # GPL v3 License
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation
├── CONTRIBUTING.md                 # Contribution guidelines
├── examples/                       # Example scripts
│   ├── optimizer_demo.py          # Basic demonstrations
│   └── use_with_labs.py           # Lab-specific examples
└── docs/                          # Documentation
    ├── optimizer_quick_reference.md   # Quick reference guide
    └── time-space-trade-offs.md       # Theoretical background
```

### Core Module

**`optimizer_selector.py`** - Main implementation with decision tree logic
- `select_optimizer()` - Get solver recommendation based on problem size and constraints
- `get_sklearn_model()` - Instantiate sklearn model with recommended solver
- Type-safe, well-documented, production-ready code

### Documentation

**`docs/optimizer_quick_reference.md`** - Quick reference guide
- API documentation
- Common scenarios
- Troubleshooting tips

**`docs/time-space-trade-offs.md`** - Theoretical background
- Comprehensive comparison table
- Time/space complexity analysis
- Why L-BFGS is often optimal

### Examples

**`examples/optimizer_demo.py`** - Basic demonstrations
- Classification examples
- Parameter scaling demo
- Memory constraints demo

**`examples/use_with_labs.py`** - Lab-specific examples
- Regularization comparison
- Hyperparameter tuning
- Practical tips

## 📦 Installation

### Option 1: Direct Download (Recommended)

```bash
# Clone or download this repository
git clone https://github.com/yourusername/ml-optimizer-selector.git
cd ml-optimizer-selector

# Install dependencies
pip install -r requirements.txt

# Use it in your project
from optimizer_selector import select_optimizer, get_sklearn_model
```

### Option 2: Install as Package

```bash
# From the repository directory
pip install .

# Or in development mode
pip install -e .
```

### Dependencies

- Python 3.8+
- scikit-learn >= 1.0.0
- numpy >= 1.20.0

## 🚀 Quick Start

### Basic Usage

```python
from optimizer_selector import select_optimizer, get_sklearn_model

# Get recommendation
rec = select_optimizer(
    n_parameters=100,
    n_samples=1000,
    regularization="l2"
)

# Get configured model
model = get_sklearn_model(rec, task="classification", max_iter=1000)

# Train
model.fit(X_train, y_train)
```

### Run Examples

```bash
# See basic examples
python examples/optimizer_demo.py

# See lab-specific examples
python examples/use_with_labs.py

# Run the selector standalone to see example recommendations
python optimizer_selector.py
```

## 🎯 Decision Logic

The selector implements this decision tree:

```
Is n_parameters > 100,000?
├─ YES → SGD/Mini-batch SGD
└─ NO → Do you need L1 regularization?
    ├─ YES → SAGA
    └─ NO → Is dataset in memory?
        ├─ NO → Mini-batch SGD
        └─ YES → Is n_samples > 1,000,000?
            ├─ YES → SAG
            └─ NO → L-BFGS ✓ (optimal for most cases)
```

## 📊 Solver Comparison

| Solver | When to Use | Supports L1? | Typical Iterations |
|--------|-------------|--------------|-------------------|
| **lbfgs** | Small-medium datasets, L2/no reg | ❌ | 10-100 |
| **saga** | L1 regularization, large datasets | ✅ | 100-1000 |
| **sag** | Very large in-memory datasets | ❌ | 100-500 |
| **sgd** | Massive/out-of-memory datasets | ✅ | 1000+ |

## 💡 Key Insights

### Why L-BFGS is Often Best

1. **Uses curvature information** (second-order) - smarter than gradient descent
2. **Memory efficient** - stores only $m$ vectors instead of full Hessian
3. **Fast convergence** - superlinear instead of linear
4. **Optimal for medium problems** - the "Goldilocks zone"

### The Trade-off

- **Per-iteration cost**: Full Newton > L-BFGS > GD > SGD
- **Iterations needed**: Full Newton < L-BFGS < GD < SGD  
- **Total time (medium data)**: L-BFGS wins!
- **Total time (huge data)**: SGD wins!

### Why L-BFGS Can't Handle L1

L1 regularization ($|\theta|$) is **non-differentiable at zero**, which breaks:
- The smoothness assumption of BFGS
- The secant condition: $B_k s_k = y_k$
- Ability to find exact zeros (sparsity)

Solution: Use **SAGA** which employs proximal methods (soft thresholding).

## 📚 Use Cases

### For Your Labs

**Lab 3: Regularization in Logistic Regression**
```python
# Compare different regularization types
for reg in ["none", "l1", "l2"]:
    rec = select_optimizer(
        n_parameters=n_features,
        n_samples=n_samples,
        regularization=reg
    )
    model = get_sklearn_model(rec, task="classification", max_iter=10000)
    model.fit(X_train_scaled, y_train)
    # Evaluate...
```

### For Hyperparameter Tuning
```python
rec = select_optimizer(n_parameters=n, n_samples=m, regularization="l2")
model = get_sklearn_model(rec, task="classification")

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### For Large Datasets
```python
# Automatically selects SGD for massive datasets
rec = select_optimizer(
    n_parameters=100000,
    n_samples=5000000,
    regularization="l2",
    dataset_in_memory=False
)
# Returns: SGDClassifier recommendation
```

### With Memory Constraints
```python
# Avoid memory issues
rec = select_optimizer(
    n_parameters=50000,
    n_samples=100000,
    regularization="l2",
    max_memory_gb=2.0  # Only 2GB available
)
```

## ⚙️ Best Practices

1. **Always scale features** - Use `StandardScaler` or `MinMaxScaler`
2. **Use cross-validation** - Don't trust single train/test split
3. **Set max_iter high** - Default 100 is often too low (try 10000)
4. **Check convergence** - Look at `model.n_iter_` attribute
5. **Start with recommendation** - But validate empirically

## 🔧 Integration with Your Workflow

### Step-by-Step Template

```python
from optimizer_selector import select_optimizer, get_sklearn_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# 1. Load and split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Scale features (IMPORTANT!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Get solver recommendation
rec = select_optimizer(
    n_parameters=X_train.shape[1],
    n_samples=X_train.shape[0],
    regularization="l2",  # or "l1", "none", "elasticnet"
    dataset_in_memory=True
)

print(f"Using {rec.solver}: {rec.reason}")

# 4. Get configured model
model = get_sklearn_model(
    rec, 
    task="classification",  # or "regression"
    max_iter=10000,
    random_state=42
)

# 5. Train
model.fit(X_train_scaled, y_train)

# 6. Evaluate with cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 7. Test
test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
print(f"Test Accuracy: {test_accuracy:.4f}")
```

## 📖 Further Reading

- **📘 Comprehensive Theory**: See [`optimization_notes.pdf`](optimization_notes.pdf) - **START HERE for deep understanding!**
- **Theoretical Background**: See [`docs/time-space-trade-offs.md`](docs/time-space-trade-offs.md)
- **Quick Reference**: See [`docs/optimizer_quick_reference.md`](docs/optimizer_quick_reference.md)
- **Practical Examples**: Run [`examples/use_with_labs.py`](examples/use_with_labs.py)
- **Demos**: Run [`examples/optimizer_demo.py`](examples/optimizer_demo.py)

## 🎓 Academic Context

This utility implements best practices from:
- Nocedal & Wright, "Numerical Optimization" (L-BFGS theory)
- Sklearn documentation on solver selection
- Empirical benchmarks on convergence rates
- Memory-efficiency considerations for large-scale ML

**For detailed mathematical foundations**, see [`optimization_notes.pdf`](optimization_notes.pdf) which includes derivations, convergence analysis, and extensive theoretical background on second-order optimization methods.

Perfect for:
- Lab assignments requiring solver justification
- Understanding optimization trade-offs
- Comparing regularization strategies
- Scaling ML to larger datasets

## 🤝 Contributing

Feel free to:
- Add more solvers (Newton-CG, trust-constr, etc.)
- Extend for other ML algorithms (SVM, neural nets)
- Add dataset-specific heuristics
- Improve memory estimation

## License

Educational/Academic Use - Created for ML coursework

---

**Created**: October 2025  
**Purpose**: Automate solver selection for machine learning optimization  
**Context**: Master's Program - Machine Learning Labs

