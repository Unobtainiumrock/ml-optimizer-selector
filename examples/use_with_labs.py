"""
Example: Using optimizer_selector with your lab assignments

This shows how to integrate the optimizer selector into your existing
lab work on regularization and logistic regression.
"""

from optimizer_selector import select_optimizer, get_sklearn_model
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


def lab3_example():
    """
    Example for Lab 3: Regularization in Logistic Regression
    
    This demonstrates how to automatically select solvers for different
    regularization strategies in logistic regression.
    """
    print("=" * 70)
    print("LAB 3 EXAMPLE: Logistic Regression with Regularization")
    print("=" * 70)
    
    # Load a real dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nDataset: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Classes: {np.unique(y)}")
    
    # Test different regularization strategies
    print("\n" + "-" * 70)
    print("COMPARING REGULARIZATION STRATEGIES")
    print("-" * 70)
    
    strategies = [
        ("No Regularization", "none"),
        ("L2 Regularization (Ridge)", "l2"),
        ("L1 Regularization (Lasso)", "l1"),
    ]
    
    results = []
    
    for name, reg_type in strategies:
        print(f"\n{name}")
        print("  " + "─" * 65)
        
        # Get optimal solver
        rec = select_optimizer(
            n_parameters=X_train.shape[1],
            n_samples=X_train.shape[0],
            regularization=reg_type,
            dataset_in_memory=True
        )
        
        print(f"  Solver: {rec.solver}")
        print(f"  Reason: {rec.reason}")
        
        # Train model
        model = get_sklearn_model(
            rec,
            task="classification",
            max_iter=10000,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Iterations: {getattr(model, 'n_iter_', 'N/A')}")
        
        # Check for overfitting
        overfit = train_acc - test_acc
        if overfit > 0.05:
            print(f"  ⚠️  Possible overfitting detected (gap: {overfit:.4f})")
        
        results.append({
            'name': name,
            'solver': rec.solver,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'cv_mean': cv_scores.mean()
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Strategy':<35} {'Solver':<10} {'Test Acc':<10} {'CV Score':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<35} {r['solver']:<10} {r['test_acc']:<10.4f} {r['cv_mean']:<10.4f}")
    
    best = max(results, key=lambda x: x['cv_mean'])
    print(f"\n✓ Best strategy: {best['name']} (CV: {best['cv_mean']:.4f})")


def multiclass_example():
    """
    Example: Multiclass classification with automatic solver selection
    """
    print("\n\n" + "=" * 70)
    print("MULTICLASS CLASSIFICATION EXAMPLE")
    print("=" * 70)
    
    # Load iris dataset (3 classes)
    data = load_iris()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nDataset: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Classes: {len(np.unique(y))} (multiclass)")
    
    # Get recommendation
    rec = select_optimizer(
        n_parameters=X_train.shape[1],
        n_samples=X_train.shape[0],
        regularization="l2",
        dataset_in_memory=True,
        is_multiclass=True
    )
    
    print(f"\nRecommended Solver: {rec.solver}")
    print(f"Reason: {rec.reason}")
    
    # Train
    model = get_sklearn_model(
        rec,
        task="classification",
        max_iter=1000,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Detailed evaluation
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))


def parameter_tuning_example():
    """
    Example: Using optimizer selector for hyperparameter tuning
    """
    print("\n\n" + "=" * 70)
    print("HYPERPARAMETER TUNING EXAMPLE")
    print("=" * 70)
    
    from sklearn.model_selection import GridSearchCV
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nDataset: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    # Get base recommendation
    rec = select_optimizer(
        n_parameters=X_train.shape[1],
        n_samples=X_train.shape[0],
        regularization="l2"
    )
    
    print(f"\nBase Solver: {rec.solver}")
    
    # Create model with recommended solver
    model = get_sklearn_model(rec, task="classification", max_iter=10000)
    
    # Grid search over regularization strength
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100]  # Inverse of regularization strength
    }
    
    print("\nPerforming Grid Search...")
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")
    print(f"Test Accuracy: {grid_search.score(X_test_scaled, y_test):.4f}")
    
    # Show top 3 configurations
    print("\nTop 3 Configurations:")
    results = grid_search.cv_results_
    indices = np.argsort(results['mean_test_score'])[::-1][:3]
    
    for i, idx in enumerate(indices, 1):
        print(f"  {i}. C={results['param_C'][idx]}, "
              f"Score={results['mean_test_score'][idx]:.4f} "
              f"(+/- {results['std_test_score'][idx]:.4f})")


def practical_tips():
    """
    Print practical tips for using the optimizer selector
    """
    print("\n\n" + "=" * 70)
    print("PRACTICAL TIPS FOR YOUR LABS")
    print("=" * 70)
    
    tips = """
1. ALWAYS scale your features before training
   ✓ Use StandardScaler or MinMaxScaler
   ✓ Fit on training data, transform both train and test
   
2. Start with the recommended solver, then experiment
   ✓ The selector gives you a good starting point
   ✓ But always validate with cross-validation
   
3. Use cross-validation to evaluate regularization
   ✓ Don't trust a single train/test split
   ✓ 5-fold CV is a good default
   
4. For L1 regularization (feature selection):
   ✓ SAGA solver is required
   ✓ Examine coefficients to see which features are zeroed out
   ✓ model.coef_ shows the learned weights
   
5. If you get ConvergenceWarnings:
   ✓ Increase max_iter (try 10000)
   ✓ Make sure features are scaled
   ✓ Try a different solver
   
6. For small datasets (<1000 samples):
   ✓ L-BFGS will be fast
   ✓ But be careful of overfitting - use regularization!
   
7. To compare regularization strategies:
   ✓ Try none, L1, and L2
   ✓ Use cross-validation for fair comparison
   ✓ Look at train vs test accuracy gap (overfitting indicator)
   
8. For your lab reports:
   ✓ Show why you chose a particular solver
   ✓ Compare different regularization types
   ✓ Use the decision tree to justify your choices
"""
    
    print(tips)
    
    print("\nQuick Code Template:")
    print("-" * 70)
    code = '''
from optimizer_selector import select_optimizer, get_sklearn_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# 1. Get recommendation
rec = select_optimizer(
    n_parameters=X_train.shape[1],
    n_samples=X_train.shape[0],
    regularization="l2"  # or "l1", "none"
)

# 2. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Get model
model = get_sklearn_model(rec, task="classification", max_iter=10000)

# 4. Train and evaluate
model.fit(X_train_scaled, y_train)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    '''
    print(code)


if __name__ == "__main__":
    lab3_example()
    multiclass_example()
    parameter_tuning_example()
    practical_tips()

