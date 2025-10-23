"""
Demo: Using the Optimizer Selector with Real Data

This script demonstrates how to use the optimizer_selector utility
with actual sklearn models and datasets.
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from optimizer_selector import select_optimizer, get_sklearn_model
import time


def demo_classification():
    """Demo with classification task"""
    print("=" * 70)
    print("CLASSIFICATION DEMO")
    print("=" * 70)
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=5000,
        n_features=500,
        n_informative=50,
        n_redundant=10,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nDataset: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    # Test different scenarios
    scenarios = [
        {"regularization": "l2", "name": "L2 Regularization"},
        {"regularization": "l1", "name": "L1 Regularization (Lasso)"},
        {"regularization": "none", "name": "No Regularization"},
    ]
    
    for scenario in scenarios:
        print(f"\n{'-' * 70}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'-' * 70}")
        
        # Get recommendation
        rec = select_optimizer(
            n_parameters=X_train.shape[1],
            n_samples=X_train.shape[0],
            regularization=scenario["regularization"],
            dataset_in_memory=True
        )
        
        print(f"\n{rec}")
        
        # Get and train model
        model = get_sklearn_model(
            rec, 
            task="classification",
            max_iter=1000,
            random_state=42
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nResults:")
        print(f"  Training Time: {train_time:.4f}s")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Iterations: {getattr(model, 'n_iter_', 'N/A')}")


def demo_parameter_scaling():
    """Demo showing how recommendations change with problem size"""
    print("\n\n" + "=" * 70)
    print("PARAMETER SCALING DEMO")
    print("=" * 70)
    print("\nHow solver selection changes with problem size:\n")
    
    problem_sizes = [
        (100, 1_000, "Tiny problem"),
        (1_000, 10_000, "Small problem"),
        (10_000, 100_000, "Medium problem"),
        (100_000, 500_000, "Large problem"),
        (500_000, 1_000_000, "Very large problem"),
    ]
    
    print(f"{'Parameters':<15} {'Samples':<15} {'Description':<20} {'Recommended Solver':<20}")
    print("-" * 70)
    
    for n_params, n_samples, description in problem_sizes:
        rec = select_optimizer(
            n_parameters=n_params,
            n_samples=n_samples,
            regularization="l2",
            dataset_in_memory=True
        )
        print(f"{n_params:<15,} {n_samples:<15,} {description:<20} {rec.solver:<20}")


def demo_memory_constraints():
    """Demo showing memory-constrained optimization"""
    print("\n\n" + "=" * 70)
    print("MEMORY CONSTRAINT DEMO")
    print("=" * 70)
    print("\nHow solver selection changes with memory limits:\n")
    
    n_params = 50_000
    n_samples = 100_000
    
    memory_limits = [None, 10.0, 1.0, 0.1]  # GB
    
    print(f"{'Memory Limit':<20} {'Recommended Solver':<20} {'Explanation':<50}")
    print("-" * 90)
    
    for mem_limit in memory_limits:
        rec = select_optimizer(
            n_parameters=n_params,
            n_samples=n_samples,
            regularization="l2",
            dataset_in_memory=True,
            max_memory_gb=mem_limit
        )
        
        mem_str = f"{mem_limit}GB" if mem_limit else "Unlimited"
        # Truncate explanation for display
        explanation = rec.reason[:47] + "..." if len(rec.reason) > 50 else rec.reason
        print(f"{mem_str:<20} {rec.solver:<20} {explanation:<50}")


def demo_comparison():
    """Compare different solvers on the same problem"""
    print("\n\n" + "=" * 70)
    print("SOLVER COMPARISON DEMO")
    print("=" * 70)
    
    # Generate dataset
    X, y = make_classification(
        n_samples=2000,
        n_features=100,
        n_informative=20,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nComparing solvers on: {X_train.shape[0]} samples, {X_train.shape[1]} features\n")
    
    from sklearn.linear_model import LogisticRegression
    
    solvers = ['lbfgs', 'saga', 'liblinear', 'sag']
    
    print(f"{'Solver':<15} {'Train Time (s)':<20} {'Accuracy':<12} {'Iterations':<12}")
    print("-" * 70)
    
    for solver in solvers:
        try:
            model = LogisticRegression(
                solver=solver,
                max_iter=1000,
                random_state=42,
                penalty='l2'
            )
            
            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start
            
            accuracy = accuracy_score(y_test, model.predict(X_test))
            iterations = getattr(model, 'n_iter_', ['N/A'])[0]
            
            print(f"{solver:<15} {train_time:<20.6f} {accuracy:<12.4f} {iterations:<12}")
            
        except Exception as e:
            print(f"{solver:<15} ERROR: {str(e)[:40]}")


if __name__ == "__main__":
    # Run all demos
    demo_classification()
    demo_parameter_scaling()
    demo_memory_constraints()
    demo_comparison()

